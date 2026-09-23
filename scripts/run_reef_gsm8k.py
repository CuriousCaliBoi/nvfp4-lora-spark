#!/usr/bin/env python3
"""A bounded, resumable GSM8K receipt/report campaign through real REEF APIs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen


DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def token_from_env():
    token_file = os.environ.get("REEF_TOKEN_FILE")
    token = Path(token_file).read_text().strip() if token_file else os.environ.get("REEF_TOKEN", "")
    if not token:
        raise ValueError("set REEF_TOKEN_FILE or REEF_TOKEN; credentials are never command arguments")
    return token


class ReefHTTP:
    def __init__(self, url, token, timeout=600):
        parsed = urlsplit(url)
        if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("this campaign accepts only loopback HTTP endpoints")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("endpoint cannot contain credentials or query parameters")
        self.url, self.token, self.timeout = url.rstrip("/"), token, timeout

    def request(self, path, payload=None, scenario=None, *, timeout=None):
        headers = {"Authorization": "Bearer " + self.token, "Content-Type": "application/json"}
        if scenario:
            headers["x-reef-scenario"] = scenario
        body = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        with urlopen(Request(self.url + path, data=body, headers=headers), timeout=timeout or self.timeout) as response:
            return json.load(response), dict(response.headers)

    def get(self, path, *, timeout=10):
        return self.request(path, timeout=min(timeout, self.timeout))[0]


def scenario_path(scenario):
    return "/reef/scenarios/" + quote(scenario, safe="")


def make_plan(train, test):
    from nvfp4_lora.reef_data import gsm8k_reward

    train_ids = random.Random(42).sample(range(len(train)), 24)
    test_ids = random.Random(43).sample(range(len(test)), 16)
    train_rows = [{"sample_id": str(i), **train[i]} for i in train_ids]
    test_rows = [{"sample_id": str(i), **test[i]} for i in test_ids]
    if {x["question"].strip() for x in train_rows} & {x["question"].strip() for x in test_rows}:
        raise ValueError("selected training and held-out questions overlap")
    for row in train_rows + test_rows:
        if gsm8k_reward(row["answer"], row["answer"]) != 1:
            raise ValueError("dataset contains an invalid strict reference answer")
    holdout = {"schema_version": 1, "dataset": "openai/gsm8k", "dataset_revision": DATASET_REVISION,
               "split": "test", "seed": 43, "rows": test_rows}
    batches = []
    for target in (1, 2):
        for attempt in range(3):
            start = ((target - 1) * 3 + attempt) * 4
            batches.append({"target_optimizer_step": target, "attempt": attempt,
                            "cycle_id": f"update-{target}-attempt-{attempt + 1}", "rows": train_rows[start:start + 4]})
    return {"schema_version": 1, "dataset": "openai/gsm8k", "dataset_revision": DATASET_REVISION,
            "split": "train", "seed": 42, "batches": batches}, holdout


def prepare(args):
    import pyarrow.parquet as parquet

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    train = parquet.read_table(args.train_parquet, columns=["question", "answer"]).to_pylist()
    test = parquet.read_table(args.test_parquet, columns=["question", "answer"]).to_pylist()
    if len(train) != 7473 or len(test) != 1319:
        raise ValueError("expected canonical GSM8K main split lengths")
    for path in (args.train_parquet, args.test_parquet):
        if DATASET_REVISION not in str(path.absolute()):
            raise ValueError("parquet path must come from the pinned dataset snapshot")
    plan, holdout = make_plan(train, test)
    write_json(output / "holdout.json", holdout)
    plan["holdout_sha256"] = sha256(output / "holdout.json")
    plan["source_files"] = {"train": sha256(args.train_parquet), "test": sha256(args.test_parquet)}
    write_json(output / "campaign.json", plan)
    print(json.dumps({"campaign": str(output / "campaign.json"), "holdout_sha256": plan["holdout_sha256"]}))


def strict_reward(response, answer):
    from nvfp4_lora.reef_data import gsm8k_reward

    choices = response.get("choices", [])
    if len(choices) != 1 or choices[0].get("finish_reason") not in {"stop", "length"}:
        raise ValueError("expected one complete native choice")
    choice = choices[0]
    text = choice["message"]["content"]
    if not isinstance(text, str):
        raise ValueError("expected text response")
    return 0.0 if choice["finish_reason"] == "length" else float(gsm8k_reward(text, answer))


def learning_request(row, seed):
    from nvfp4_lora.reef_data import GSM8K_SYSTEM_PROMPT

    return {"model": MODEL, "messages": [{"role": "system", "content": GSM8K_SYSTEM_PROMPT},
            {"role": "user", "content": row["question"]}], "n": 1, "stream": False,
            "max_tokens": 256, "temperature": 1.2, "top_p": 1.0, "top_k": -1, "min_p": 0.0,
            "frequency_penalty": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0,
            "logit_bias": {}, "seed": seed, "chat_template_kwargs": {"enable_thinking": False}}


class SnapshotPending(RuntimeError):
    pass


def _scenario_status(status, scenario):
    if status.get("error") or status.get("preload_errors"):
        raise RuntimeError("REEF reports a runtime error; see retained status evidence")
    return status["scenarios"][scenario]


def _status_identity(current):
    return {key: current.get(key) for key in (
        "scenario_step", "artifact_head_sync", "current_runtime_load_id", "inference_admission",
    )}


def _snapshot_once(client, scenario, state_root, deadline, minimum_step):
    from nvfp4_lora.reef_checkpoint import validate_checkpoint

    def get(path):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SnapshotPending("snapshot read budget expired")
        return client.get(path, timeout=min(5, remaining))

    releases = get(scenario_path(scenario) + "/releases")["releases"]
    current = _scenario_status(get("/reef/status"), scenario)
    heads = [row for row in releases if row.get("current")]
    if len(heads) != 1 or not current.get("current_runtime_load_id"):
        raise SnapshotPending("REEF has no unique published serving head")
    if (current["scenario_step"] < minimum_step
            or current.get("artifact_head_sync", {}).get("state") != "synchronized"
            or current["artifact_head_sync"].get("release_id") != heads[0]["release_id"]
            or current.get("inference_admission", {}).get("open") is not True):
        raise SnapshotPending("REEF has not settled the durable head and reopened admission")
    serving = read_json(state_root / "serving" / "serving-state.json")
    incumbent = read_json(state_root / "incumbent.json")
    path = Path(incumbent["checkpoint_path"]).resolve()
    if not path.is_relative_to((state_root / "checkpoints").resolve()):
        raise ValueError("incumbent checkpoint escapes the declared root")
    manifest = validate_checkpoint(path)
    if manifest["checkpoint_id"] != incumbent["checkpoint_id"]:
        raise ValueError("incumbent pointer and checkpoint identity disagree")
    serving_after = read_json(state_root / "serving" / "serving-state.json")
    incumbent_after = read_json(state_root / "incumbent.json")
    current_after = _scenario_status(get("/reef/status"), scenario)
    releases_after = get(scenario_path(scenario) + "/releases")["releases"]
    if (serving_after != serving or incumbent_after != incumbent or releases_after != releases
            or _status_identity(current_after) != _status_identity(current)):
        raise SnapshotPending("publication changed while reading the snapshot")
    if serving["fenced"] or serving["active_checkpoint_id"] != manifest["checkpoint_id"]:
        raise SnapshotPending("native actor is fenced or differs from the committed learner")
    if serving["published_runtime_load_id"] != current["current_runtime_load_id"]:
        raise SnapshotPending("REEF and native actor runtime identities disagree")
    if serving["active_release"] != heads[0]["release_id"] or serving["pending"] is not None:
        raise SnapshotPending("native actor does not bind the durable published release")
    binding = serving["bindings"][manifest["checkpoint_id"]]
    if not binding["native_verified"] or binding["adapter_sha256"] != manifest["adapter_sha256"]:
        raise RuntimeError("native verification does not bind the committed adapter")
    if binding["adapter_config_sha256"] != manifest["files"]["adapter/adapter_config.json"]:
        raise RuntimeError("native adapter config differs from the checkpoint")
    if binding["reload_max_delta"] > 1e-5:
        raise RuntimeError("native reload proof exceeds the agreed tolerance")
    if manifest["optimizer_step"] and binding["adapter_effect_max_delta"] <= binding["reference_repeat_max_delta"]:
        raise RuntimeError("trained adapter lacks measurable native effect")
    return {"scenario": scenario, "status": current, "release": heads[0], "checkpoint": manifest,
            "checkpoint_path": str(path), "serving": serving, "captured_unix": time.time()}


def snapshot(client, scenario, state_root, *, timeout=30, minimum_step=0):
    deadline = time.monotonic() + timeout
    reason = "no coherent snapshot observed"
    while time.monotonic() < deadline:
        try:
            return _snapshot_once(client, scenario, state_root, deadline, minimum_step)
        except (SnapshotPending, TimeoutError) as exc:
            reason = str(exc)
        time.sleep(min(0.1, max(0, deadline - time.monotonic())))
    raise TimeoutError("REEF publication did not settle: " + reason)


def verify_continuation(before, after, *, restarted=False):
    left, right = before["checkpoint"], after["checkpoint"]
    if left["checkpoint_id"] != right["checkpoint_id"] or left["files"] != right["files"]:
        raise RuntimeError("resume/rollback did not restore the exact complete training checkpoint")
    if restarted and before["status"]["current_runtime_load_id"] == after["status"]["current_runtime_load_id"]:
        raise RuntimeError("runtime incarnation did not change after restart/rollback")


def collect_and_report(client, scenario, declaration, directory, before):
    from nvfp4_lora.reef_data import validate_capture

    samples = []
    for group, row in enumerate(declaration["rows"]):
        for rollout in range(8):
            sample_path = directory / f"sample-{group}-{rollout}.json"
            if sample_path.exists():
                sample = read_json(sample_path)
            else:
                request = learning_request(row, 42 + int(row["sample_id"]) * 8 + rollout)
                response, headers = client.request("/v1/chat/completions", request, scenario)
                receipt = next((v for k, v in headers.items() if k.lower() == "x-reef-agent-record-id"), None)
                if not receipt:
                    raise RuntimeError("inference did not return a durable REEF receipt")
                record = client.get(scenario_path(scenario) + "/records/" + quote(receipt, safe=""))
                sample = {"group_id": group, "rollout_id": rollout, "row": row, "receipt": receipt,
                          "request": request, "response": response, "record": record,
                          "reward": strict_reward(response, row["answer"])}
                write_json(sample_path, sample)
            payload = sample["record"]["payload"]
            validate_capture(payload["response"].get("training"), payload.get("runtime_load_id"))
            if payload["runtime_load_id"] != before["status"]["current_runtime_load_id"]:
                raise ValueError("captured receipt belongs to a different serving incarnation")
            samples.append(sample)
    if len({row["receipt"] for row in samples}) != 32:
        raise ValueError("rollout receipts must be distinct")
    report_ids = []
    for sample in samples:
        report_id = hashlib.sha256((scenario + ":" + declaration["cycle_id"] + ":" + sample["receipt"]).encode()).hexdigest()
        report_ids.append(report_id)
        payload = {"agent_record_id": report_id, "score": sample["reward"], "references": [sample["receipt"]],
                   "metadata": {"cycle_id": declaration["cycle_id"], "group_id": sample["group_id"],
                    "rollout_id": sample["rollout_id"], "group_size": 8, "batch_group_count": 4,
                    "dataset_id": "openai/gsm8k", "dataset_revision": DATASET_REVISION, "dataset_split": "train",
                    "dataset_row_id": int(sample["row"]["sample_id"]), "gold_answer": sample["row"]["answer"]}}
        report_path = directory / f"report-{sample['group_id']}-{sample['rollout_id']}.json"
        if not report_path.exists():
            reply, _ = client.request("/reef/report", payload, scenario)
            write_json(report_path, {"payload": payload, "reply": reply})
    return report_ids


def await_commit(client, scenario, report_ids, after_step, deadline, directory, *, state_root):
    while time.monotonic() < deadline:
        status = client.get("/reef/status")
        write_json(directory / "latest-status.json", status)
        if status.get("error") or status.get("preload_errors"):
            raise RuntimeError("REEF training failed; see latest-status.json")
        query = urlencode({"after_step": after_step, "limit": 100})
        commits = client.get(scenario_path(scenario) + "/commits?" + query)
        write_json(directory / "commits.json", commits)
        matching = [row for row in commits["commits"] if set(report_ids).issubset(row["consumed_ids"])]
        if len(matching) > 1:
            raise RuntimeError("the same complete report batch was consumed more than once")
        if matching and not matching[0]["pending"]:
            try:
                settled = snapshot(client, scenario, state_root,
                                   timeout=min(5, max(0, deadline - time.monotonic())),
                                   minimum_step=matching[0]["step"])
            except TimeoutError:
                settled = None
            if settled is not None:
                write_json(directory / "settled.json", settled)
                return matching[0]
        time.sleep(min(2, max(0, deadline - time.monotonic())))
    raise TimeoutError("no settled REEF publication consumed the complete declared report batch")


def cycle(args, client):
    plan = read_json(args.plan)
    if plan.get("dataset_revision") != DATASET_REVISION or plan.get("split") != "train":
        raise ValueError("campaign must use the pinned declared training plan")
    declarations = [b for b in plan["batches"] if b["target_optimizer_step"] == args.target_step]
    if len(declarations) != 3:
        raise ValueError("exactly three attempts must be predeclared per target update")
    args.output.mkdir(parents=True, exist_ok=True)
    binding_path = args.output / "plan-binding.json"
    binding = {"plan_sha256": sha256(args.plan), "scenario": args.scenario,
               "state_root": str(args.state_root.resolve())}
    if binding_path.exists() and read_json(binding_path) != binding:
        raise ValueError("campaign plan or state root changed after collection began")
    write_json(binding_path, binding)
    current = snapshot(client, args.scenario, args.state_root)
    if current["checkpoint"]["optimizer_step"] == args.target_step:
        if not (args.output / f"accepted-{args.target_step}.json").exists():
            raise RuntimeError("target step already exists without campaign acceptance evidence")
        return 0
    if current["checkpoint"]["optimizer_step"] != args.target_step - 1:
        raise ValueError("target optimizer step must continue the current committed checkpoint")
    for declaration in declarations:
        directory = args.output / declaration["cycle_id"]
        directory.mkdir(exist_ok=True)
        if (directory / "outcome.json").exists():
            continue
        before_path = directory / "before.json"
        if before_path.exists():
            before = read_json(before_path)
            verify_continuation(before, current)
        else:
            before = snapshot(client, args.scenario, args.state_root)
            write_json(before_path, before)
            write_json(directory / "declaration.json", declaration)
        report_ids = collect_and_report(client, args.scenario, declaration, directory, before)
        commit = await_commit(client, args.scenario, report_ids, before["status"]["scenario_step"],
                              time.monotonic() + args.timeout, directory, state_root=args.state_root)
        current = snapshot(client, args.scenario, args.state_root)
        manifest = current["checkpoint"]
        accepted = manifest["optimizer_step"] == args.target_step
        if accepted:
            if manifest["parent_checkpoint_id"] != before["checkpoint"]["checkpoint_id"]:
                raise RuntimeError("accepted update did not branch from the committed parent")
            if manifest["frozen_tensor_sha256"] != before["checkpoint"]["frozen_tensor_sha256"]:
                raise RuntimeError("accepted update changed the frozen base")
            if current["status"]["current_runtime_load_id"] == before["status"]["current_runtime_load_id"]:
                raise RuntimeError("accepted checkpoint was not published as a new native serving version")
            if manifest["files"]["optimizer.pt"] == before["checkpoint"]["files"]["optimizer.pt"]:
                raise RuntimeError("accepted update did not change optimizer state")
        else:
            verify_continuation(before, current)
        write_json(directory / "outcome.json", {"accepted": accepted, "commit": commit, "after": current})
        if accepted:
            write_json(args.output / f"accepted-{args.target_step}.json", current)
            print(json.dumps({"status": "accepted", "optimizer_step": args.target_step, "checkpoint_id": manifest["checkpoint_id"]}))
            return 0
    write_json(args.output / f"inconclusive-{args.target_step}.json",
               {"status": "inconclusive", "reason": "three predeclared batches exhausted", "target_step": args.target_step})
    return 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--train-parquet", type=Path, required=True)
    prep.add_argument("--test-parquet", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    for name in ("proxy-smoke", "create", "cycle", "snapshot", "verify-resume", "rollback"):
        item = sub.add_parser(name)
        item.add_argument("--url", default="http://127.0.0.1:8902")
        item.add_argument("--scenario", default="nvfp4-gsm8k")
        item.add_argument("--output", type=Path, required=True)
        if name not in {"proxy-smoke", "create"}:
            item.add_argument("--state-root", type=Path, required=True)
        if name == "cycle":
            item.add_argument("--plan", type=Path, required=True)
            item.add_argument("--target-step", type=int, choices=(1, 2), required=True)
            item.add_argument("--timeout", type=float, default=3600)
        if name in {"verify-resume", "rollback"}:
            item.add_argument("--expected", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
        return 0
    client = ReefHTTP(args.url, token_from_env())
    if args.command in {"create", "proxy-smoke"}:
        created, _ = client.request("/reef/scenarios", {"name": args.scenario})
        if args.command == "create":
            write_json(args.output, created)
            return 0
        request = {"model": MODEL, "messages": [{"role": "user", "content": "What is 17+25? End with #### 42."}],
                   "temperature": 0, "max_tokens": 64, "chat_template_kwargs": {"enable_thinking": False}}
        response, headers = client.request("/v1/chat/completions", request, args.scenario)
        receipt = next(v for k, v in headers.items() if k.lower() == "x-reef-agent-record-id")
        reward = strict_reward(response, "#### 42")
        report, _ = client.request("/reef/report", {"score": reward, "references": [receipt]}, args.scenario)
        record = client.get(scenario_path(args.scenario) + "/records/" + quote(receipt, safe=""))
        write_json(args.output, {"request": request, "response": response, "receipt": receipt,
                                "score": reward, "report": report, "record": record})
        return 0
    if args.command == "cycle":
        return cycle(args, client)
    if args.command == "snapshot":
        write_json(args.output, snapshot(client, args.scenario, args.state_root))
        return 0
    expected = read_json(args.expected)
    before = snapshot(client, args.scenario, args.state_root)
    response = None
    if args.command == "rollback":
        response, _ = client.request(scenario_path(args.scenario) + "/rollback",
                                     {"release_id": expected["release"]["release_id"]})
    after = snapshot(client, args.scenario, args.state_root)
    verify_continuation(expected, after, restarted=True)
    if args.command == "rollback" and before["status"]["current_runtime_load_id"] == after["status"]["current_runtime_load_id"]:
        raise RuntimeError("rollback did not publish a new runtime incarnation")
    write_json(args.output, {"before": before, "after": after, "response": response, "verified": True})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
