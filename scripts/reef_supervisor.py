#!/usr/bin/env python3
"""Own one bounded local REEF campaign and restore the original GPU server."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import time
from urllib.request import Request, urlopen

from run_reef_gsm8k import MODEL, read_json, sha256, write_json
from build_reef_marlin_image import validate_build_report, verify_actor_patch


REEF_REVISION = "b637cbe42393e91e8ee75af2179f844da586c3dd"


def command(args, **kwargs):
    kwargs.setdefault("timeout", 60)
    return subprocess.run([str(x) for x in args], check=True, text=True, capture_output=True, **kwargs).stdout.strip()


def docker_inspect(identifier):
    return json.loads(command(["docker", "inspect", identifier]))[0]


def get_json(url, token=None):
    headers = {} if token is None else {"Authorization": "Bearer " + token}
    with urlopen(Request(url, headers=headers), timeout=5) as response:
        body = response.read()
        return json.loads(body) if body else {"ok": True}


def free_port(port):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))


def require_path(path, name):
    path = Path(path)
    if not path.is_absolute() or any(c in str(path) for c in (":", "\n", "\r")):
        raise ValueError(f"{name} must be an absolute path without colons or newlines")
    return path.resolve()


def actor_argv(args, image_id, owner):
    state = args.output / "learning"
    return ["docker", "create", "--name", owner + "-actor", "--label", "nvfp4-reef.owner=" + owner,
            "--gpus", "all", "--ipc", "host", "--user", f"{os.getuid()}:{os.getgid()}",
            "-p", f"127.0.0.1:{args.actor_port}:8000", "--entrypoint", "python3",
            "-e", "HF_HOME=/hf", "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
            "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "VLLM_ALLOW_RUNTIME_LORA_UPDATING=1",
            "-e", "VLLM_CACHE_ROOT=/runtime/vllm", "-e", "TRITON_CACHE_DIR=/runtime/triton",
            "-e", "CUBLAS_WORKSPACE_CONFIG=:4096:8",
            "-e", "HOME=/runtime/home", "-e", "XDG_CACHE_HOME=/runtime/xdg",
            "-v", f"{args.hf_cache}:/hf:ro", "-v", f"{args.output / 'actor-cache'}:/runtime",
            "-v", f"{state / 'checkpoints'}:{state / 'checkpoints'}:ro", image_id,
            "-m", "vllm.entrypoints.openai.api_server", "--host", "0.0.0.0", "--port", "8000",
            "--model", args.model_dir, "--served-model-name", MODEL,
            "--revision", args.model_revision, "--dtype", "bfloat16", "--enforce-eager",
            "--gpu-memory-utilization", "0.23", "--max-model-len", "1024",
            "--max-num-batched-tokens", "1024", "--max-num-seqs", "1",
            "--moe-backend", "marlin", "--mamba-backend", "flashinfer", "--mamba-cache-mode", "none",
            "--kv-cache-dtype", "bfloat16", "--attention-backend", "TRITON_ATTN",
            "--enable-lora", "--max-lora-rank", "8",
            "--max-loras", "2", "--max-cpu-loras", "16", "--no-enable-prefix-caching", "--no-async-scheduling",
            "--logprobs-mode", "processed_logprobs", "--generation-config", "vllm", "--no-enable-log-requests"]


def worker_argv(args, image_id, owner, source_revision):
    state = args.output / "learning"
    return ["docker", "run", "--rm", "--name", owner + "-worker", "--label", "nvfp4-reef.owner=" + owner,
            "--gpus", "all", "--ipc", "host", "--network", "none", "--user", f"{os.getuid()}:{os.getgid()}",
            "--entrypoint", "python3", "-e", "HF_HOME=/hf", "-e", "HF_HUB_OFFLINE=1",
            "-e", "TRANSFORMERS_OFFLINE=1", "-e", "HF_DATASETS_OFFLINE=1",
            "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "PYTHONUNBUFFERED=1", "-e", "PYTHONPATH=/workspace",
            "-e", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", "-e", "NVFP4_EVAL_CACHE_GB=0",
            "-e", "NVFP4_TRAIN_CACHE_GB=0", "-e", "NVFP4_SOURCE_REVISION=" + source_revision,
            "-e", "HOME=/runtime/home", "-e", "XDG_CACHE_HOME=/runtime/xdg",
            "-e", "TRITON_CACHE_DIR=/runtime/triton", "-v", f"{args.repo}:/workspace:ro",
            "-v", f"{args.hf_cache}:/hf:ro", "-v", f"{args.output / 'worker-cache'}:/runtime",
            "-v", f"{state}:{state}", "-w", "/workspace", image_id, "-m", "nvfp4_lora.reef_worker"]


def build_config(args, worker, owner):
    state = args.output / "learning"
    holdout = args.plan.parent / "holdout.json"
    plan = read_json(args.plan)
    if sha256(holdout) != plan["holdout_sha256"]:
        raise ValueError("held-out file differs from the predeclared campaign")
    return {
        "schema-version": 2,
        "reef": {"host": "127.0.0.1", "port": args.reef_port, "token": "${REEF_TOKEN}",
                 "run-dir": str(args.output / "reef-stack"), "ready-timeout": args.startup_timeout},
        "inference": {"model-path": MODEL, "timeout-s": 600},
        "recipe": {"implementation": "nvfp4_lora.reef_recipe:NVFP4Recipe",
                   "config": {"group-size": 8, "batch-group-count": 4}},
        "training": {"backend": "nvfp4_lora.reef_deployment:Nvfp4Deployment", "ready-timeout": args.startup_timeout,
                     "timeout-s": args.job_timeout, "options": {
                         "state-root": str(state), "model-dir": args.model_dir,
                         "model-revision": args.model_revision, "worker-command": worker,
                         "actor-url": f"http://127.0.0.1:{args.actor_port}", "actor-instance-id": owner,
                         "probe-token-ids": read_json(args.probe_token_ids), "scenario": args.scenario,
                         "reef-revision": REEF_REVISION, "lora-rank": 8, "lora-alpha": 16, "adapter-capacity": 16}},
        "evaluation": {"module": "nvfp4_lora.reef_evaluation:Gsm8kEvaluationFactory",
                       "config": {"holdout_path": str(holdout), "holdout_sha256": sha256(holdout),
                                  "max_tokens": 256, "seed": 43}},
        "storage": {"artifact-repository": str(args.output / "artifacts.git"),
                    "artifact-work-dir": str(args.output / "artifact-work"),
                    "artifact-cache-dir": str(args.output / "artifact-cache"),
                    "agent-record-dir": str(args.output / "agent-record")},
    }


def worker_launcher(args):
    return [str(args.reef_python), str(args.repo / "scripts/reef_worker_launch.py"),
            "--command-file", str(args.output / "worker-command.json"),
            "--registry", str(args.output / "learning/jobs/worker-registry")]


def actor_attestation(inspected, owner, revision, version):
    argv = [inspected["Path"], *inspected["Args"]]
    def value(flag):
        values = [arg.split("=", 1)[1] if "=" in arg else argv[i + 1]
                  for i, arg in enumerate(argv) if arg.split("=", 1)[0] == flag]
        if len(values) > 1:
            raise ValueError("ambiguous inspected actor option: " + flag)
        return values[0] if values else None
    expected = {"--logprobs-mode": "processed_logprobs", "--generation-config": "vllm",
                "--max-lora-rank": "8", "--max-loras": "2", "--served-model-name": MODEL,
                "--revision": revision, "--dtype": "bfloat16", "--gpu-memory-utilization": "0.23",
                "--max-model-len": "1024", "--max-num-batched-tokens": "1024", "--max-num-seqs": "1",
                "--moe-backend": "marlin", "--mamba-backend": "flashinfer", "--mamba-cache-mode": "none",
                "--kv-cache-dtype": "bfloat16", "--attention-backend": "TRITON_ATTN"}
    if (any(value(k) != v for k, v in expected.items())
            or "--enable-lora" not in argv or "--enforce-eager" not in argv):
        raise ValueError("inspected actor command violates the controlled sampling contract")
    if any("speculative" in x for x in argv) or "--no-enable-prefix-caching" not in argv:
        raise ValueError("speculation and prefix caching must be disabled")
    if ("--no-async-scheduling" not in argv
            or any(arg.split("=", 1)[0] == "--async-scheduling" for arg in argv)):
        raise ValueError("the controlled actor profile requires asynchronous scheduling disabled")
    if int(value("--max-cpu-loras") or 0) < 16 or version != "0.27.1":
        raise ValueError("unsupported actor version or insufficient adapter capacity")
    if inspected["Config"].get("Labels", {}).get("nvfp4-reef.owner") != owner:
        raise ValueError("actor ownership label mismatch")
    env = inspected["Config"].get("Env", [])
    if "VLLM_ALLOW_RUNTIME_LORA_UPDATING=1" not in env:
        raise ValueError("actor does not enable runtime adapter loading")
    workspace = [entry.split("=", 1)[1] for entry in env if entry.startswith("CUBLAS_WORKSPACE_CONFIG=")]
    if workspace != [":4096:8"]:
        raise ValueError("inspected actor environment violates the controlled cuBLAS workspace setting")
    return {"schema_version": 1, "actor_instance_id": owner, "container_id": inspected["Id"],
            "image_id": inspected["Image"], "base_model": MODEL, "model_revision": revision,
            "vllm_version": version, "logprobs_mode": "processed_logprobs", "generation_config": "vllm",
            "speculative_decoding": False, "async_scheduling": False,
            "max_num_seqs": int(value("--max-num-seqs")), "kv_cache_dtype": value("--kv-cache-dtype"),
            "attention_backend": value("--attention-backend"), "mamba_cache_mode": value("--mamba-cache-mode"),
            "cublas_workspace_config": workspace[0],
            "exclusive_adapter_control": True, "command": argv}


class Supervisor:
    def __init__(self, args):
        self.args = args
        self.owner = "reef-nvfp4-" + secrets.token_hex(8)
        self.server_id = None
        self.restore_needed = False
        self.was_running = False
        self.reef_process = None
        self.child_process = None
        self.started = time.monotonic()
        self.deadline = self.started + args.max_runtime
        self.token = secrets.token_urlsafe(32)
        self.logs = []
        self.lock = None
        self.output_owned = False

    def remaining(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("campaign runtime budget expired")
        return remaining

    def wait_health(self, url, timeout, *, process=None, container_id=None):
        until = min(self.deadline, time.monotonic() + timeout)
        while time.monotonic() < until:
            if process is not None and process.poll() is not None:
                raise RuntimeError("owned service exited before becoming healthy; inspect its log")
            if container_id is not None:
                actor = docker_inspect(container_id)
                if actor["Config"].get("Labels", {}).get("nvfp4-reef.owner") != self.owner:
                    raise RuntimeError("actor ownership changed while waiting for startup")
                if not actor["State"]["Running"]:
                    write_json(self.args.output / "actor-startup-failure.json", {
                        "container_id": actor["Id"], "image_id": actor["Image"], "state": actor["State"],
                    })
                    raise RuntimeError("owned actor exited before becoming healthy; inspect its retained log")
            try:
                return get_json(url)
            except (OSError, ValueError):
                time.sleep(2)
        raise TimeoutError("owned service did not become healthy within its budget")

    def start_reef(self, suffix):
        args = self.args
        env = dict(os.environ)
        env.update(REEF_TOKEN=self.token, PYTHONPATH=str(args.repo), PYTHONDONTWRITEBYTECODE="1")
        log = (args.output / f"reef-{suffix}.log").open("w")
        self.logs.append(log)
        argv = [str(args.reef_python), "-m", "reef.service"]
        env["REEF_CONFIG"] = str(args.output / "learning.json")
        self.reef_process = subprocess.Popen(argv, cwd=args.reef_repo, env=env, stdout=log,
                                             stderr=subprocess.STDOUT, start_new_session=True)
        write_json(args.output / f"reef-{suffix}-process.json", {"pid": self.reef_process.pid, "argv": argv})
        self.wait_health(f"http://127.0.0.1:{args.reef_port}/healthz", args.startup_timeout, process=self.reef_process)

    def check_worker_permissions(self, prefix, image_id):
        probe = self.args.output / "learning/.permission-probe"
        argv = list(prefix)
        index = argv.index("--gpus")
        argv[index:index + 2] = []
        argv[argv.index("--name") + 1] = self.owner + "-permissions"
        index = argv.index(image_id)
        code = "from pathlib import Path; Path(" + repr(str(probe)) + ").write_text('host-readable')"
        argv[index + 1:] = ["-c", code]
        command(argv, timeout=60)
        if probe.read_text() != "host-readable" or probe.stat().st_uid != os.getuid():
            raise RuntimeError("worker output is not owned/readable by the CPU service user")
        probe.unlink()

    @staticmethod
    def stop_process(process):
        if process is None:
            return
        exited = process.poll() is not None
        try:
            group = os.getpgid(process.pid)
        except ProcessLookupError:
            if not exited:
                return
            group = process.pid
        else:
            if exited or group != process.pid:
                return
        if group <= 1 or group == os.getpgrp():
            raise RuntimeError("refusing to signal an unowned process group")
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        until = time.monotonic() + 5
        while time.monotonic() < until:
            try:
                os.killpg(group, 0)
            except ProcessLookupError:
                return
            time.sleep(0.1)
        try:
            current = os.getpgid(process.pid)
        except ProcessLookupError:
            current = None
        if current is not None:
            if process.poll() is not None or current != group:
                return
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)

    def campaign(self, name, extra, *, allow_inconclusive=False):
        args = self.args
        env = dict(os.environ)
        env.update(REEF_TOKEN=self.token, PYTHONPATH=str(args.repo), PYTHONDONTWRITEBYTECODE="1")
        argv = [str(args.reef_python), str(args.repo / "scripts/run_reef_gsm8k.py"), name,
                "--url", f"http://127.0.0.1:{args.reef_port}", "--scenario", args.scenario, *map(str, extra)]
        with (args.output / "campaign-driver.log").open("a") as log:
            self.child_process = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                  start_new_session=True)
            code = self.child_process.wait(timeout=self.remaining())
        self.child_process = None
        if code and not (code == 2 and allow_inconclusive):
            raise RuntimeError(f"campaign command {name} failed with status {code}; inspect campaign-driver.log")
        return code

    def run(self):
        args = self.args
        args.output.mkdir(parents=True, exist_ok=False)
        self.output_owned = True
        lock_path = Path("/tmp") / ("nvfp4-reef-" + hashlib.sha256(args.server.encode()).hexdigest()[:16] + ".lock")
        self.lock = lock_path.open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for port in (args.reef_port, args.actor_port):
            free_port(port)
        if command(["git", "-C", args.reef_repo, "rev-parse", "HEAD"]) != REEF_REVISION:
            raise ValueError("REEF checkout does not match the reviewed source revision")
        revision = command(["git", "-C", args.repo, "rev-parse", "HEAD"])
        source_status = command(["git", "-C", args.repo, "status", "--porcelain", "--untracked-files=no"])
        if source_status:
            raise ValueError("tracked experiment source must be committed before launch")
        image_id = command(["docker", "image", "inspect", "--format", "{{.Id}}", args.image])
        image_build = validate_build_report(read_json(args.image_provenance), image_id, args.repo)
        write_json(args.output / "image-build.json", image_build)
        original = docker_inspect(args.server)
        self.server_id, self.was_running = original["Id"], original["State"]["Running"]
        if self.was_running:
            get_json(args.health_url)
        for path in ("learning/checkpoints", "learning/serving", "actor-cache/home", "worker-cache/home", "campaign"):
            (args.output / path).mkdir(parents=True, exist_ok=True)
        token_path = args.output / ".token"
        fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(self.token + "\n")
        actor = actor_argv(args, image_id, self.owner)
        worker = worker_argv(args, image_id, self.owner, revision)
        write_json(args.output / "worker-command.json", {"argv": worker, "owner": self.owner,
                   "state_root": str(args.output / "learning")})
        config = build_config(args, worker_launcher(args), self.owner)
        write_json(args.output / "learning.json", config)
        write_json(args.output / "supervisor.json", {"owner": self.owner, "source_revision": revision,
                   "source_status": source_status, "reef_revision": REEF_REVISION, "image_id": image_id,
                   "original_server_id": self.server_id, "original_image_id": original["Image"],
                   "original_running": self.was_running, "max_runtime": args.max_runtime,
                   "actor_command": actor, "worker_command": worker,
                   "image_provenance_sha256": sha256(args.image_provenance),
                   "campaign_sha256": sha256(args.plan), "probe_sha256": sha256(args.probe_token_ids)})
        self.check_worker_permissions(worker, image_id)
        if self.was_running:
            self.restore_needed = True
            command(["docker", "stop", "--timeout", "60", self.server_id], timeout=90)
        time.sleep(3)
        remaining = command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"])
        if remaining.strip():
            raise RuntimeError("another process owns the GPU; leaving it untouched")
        actor_id = command(actor)
        write_json(args.output / "actor-created.json", {"container_id": actor_id, "owner": self.owner})
        command(["docker", "start", actor_id])
        self.wait_health(f"http://127.0.0.1:{args.actor_port}/health", args.startup_timeout, container_id=actor_id)
        version = get_json(f"http://127.0.0.1:{args.actor_port}/version")["version"]
        inspected_actor = docker_inspect(actor_id)
        patch_evidence = verify_actor_patch(inspected_actor, self.owner, image_build, args.repo, run=command)
        attestation = actor_attestation(inspected_actor, self.owner, args.model_revision, version)
        attestation.update(patch_evidence)
        write_json(args.output / "learning/serving/actor-contract.json", attestation)
        self.start_reef("first")
        self.campaign("create", ["--output", args.output / "scenario-created.json"])
        self.campaign("snapshot", ["--state-root", args.output / "learning", "--output", args.output / "initial.json"])
        for target in (1, 2):
            code = self.campaign("cycle", ["--state-root", args.output / "learning", "--plan", args.plan,
                                 "--target-step", target, "--timeout", args.job_timeout,
                                 "--output", args.output / "campaign"], allow_inconclusive=True)
            if code == 2:
                write_json(args.output / "result.json", {"status": "inconclusive", "accepted_updates": target - 1,
                           "reason": "predeclared batches exhausted", "learning_gain_claimed": False})
                return 2
            if target == 1:
                self.stop_process(self.reef_process)
                self.reef_process = None
                free_port(args.reef_port)
                self.start_reef("resumed")
                self.campaign("verify-resume", ["--state-root", args.output / "learning",
                              "--expected", args.output / "campaign/accepted-1.json",
                              "--output", args.output / "resume-proof.json"])
        self.campaign("rollback", ["--state-root", args.output / "learning",
                      "--expected", args.output / "campaign/accepted-1.json", "--output", args.output / "rollback-proof.json"])
        write_json(args.output / "result.json", {"status": "complete", "accepted_updates": 2,
                   "restart_verified": True, "rollback_verified": True, "learning_gain_claimed": False})
        return 0

    def cleanup(self):
        errors = []
        for process in (self.child_process, self.reef_process):
            try:
                self.stop_process(process)
            except Exception as exc:
                errors.append(type(exc).__name__ + ": owned process cleanup failed")
        try:
            ids = command(["docker", "ps", "-aq", "--filter", "label=nvfp4-reef.owner=" + self.owner]).split()
        except Exception as exc:
            ids = []
            errors.append(type(exc).__name__ + ": owned container inventory failed")
        for identifier in ids:
            try:
                inspected = docker_inspect(identifier)
                if inspected["Config"].get("Labels", {}).get("nvfp4-reef.owner") != self.owner:
                    raise RuntimeError("container ownership changed")
                if inspected["State"]["Running"]:
                    try:
                        command(["docker", "stop", "--timeout", "30", identifier], timeout=45)
                    except (subprocess.SubprocessError, OSError):
                        pass
                    try:
                        current = docker_inspect(identifier)
                    except subprocess.CalledProcessError:
                        continue
                    if current["Config"].get("Labels", {}).get("nvfp4-reef.owner") != self.owner:
                        raise RuntimeError("container ownership changed during cleanup")
                    if current["State"]["Running"]:
                        command(["docker", "kill", identifier], timeout=30)
                name = inspected.get("Name", identifier).lstrip("/")
                logs = subprocess.run(["docker", "logs", identifier], text=True, capture_output=True, timeout=30)
                (self.args.output / (name + ".log")).write_text(logs.stdout + logs.stderr)
                if docker_inspect(identifier)["State"]["Running"]:
                    raise RuntimeError("owned container remained running")
            except Exception as exc:
                errors.append(type(exc).__name__ + ": owned container cleanup failed for " + identifier)
        restoration = "not-needed-originally-stopped" if not self.was_running else "not-needed-server-untouched"
        if self.restore_needed:
            restoration = "failed"
            try:
                if not docker_inspect(self.server_id)["State"]["Running"]:
                    command(["docker", "start", self.server_id], timeout=90)
                until = time.monotonic() + self.args.health_timeout
                while time.monotonic() < until:
                    try:
                        health = get_json(self.args.health_url)
                        write_json(self.args.output / "restored-health.json", health)
                        restoration = "healthy"
                        break
                    except (OSError, ValueError):
                        time.sleep(2)
                if restoration != "healthy":
                    raise RuntimeError("original production health check timed out")
            except Exception as exc:
                errors.append(type(exc).__name__ + ": original server restoration failed")
        if self.args.proxy_url and restoration in {"healthy", "not-needed-server-untouched"}:
            try:
                token_file = os.environ["REEF_PROXY_TOKEN_FILE"]
                token = Path(token_file).read_text().strip()
                evidence = {"health": get_json(self.args.proxy_url + "/healthz"),
                            "status": get_json(self.args.proxy_url + "/reef/status", token)}
                write_json(self.args.output / "existing-proxy-postcheck.json", evidence)
            except Exception as exc:
                errors.append(type(exc).__name__ + ": existing proxy postcheck failed")
        if self.output_owned:
            write_json(self.args.output / "restoration.json", {"original_server_id": self.server_id,
                       "status": restoration, "errors": errors, "elapsed_seconds": time.monotonic() - self.started})
        for log in self.logs:
            log.close()
        if self.lock:
            self.lock.close()
        return errors


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("image", "image-provenance", "repo", "reef-repo", "reef-python", "hf-cache", "model-dir", "model-revision",
                 "output", "plan", "probe-token-ids", "server", "health-url"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--scenario", default="nvfp4-gsm8k")
    parser.add_argument("--reef-port", type=int, default=8902)
    parser.add_argument("--actor-port", type=int, default=30001)
    parser.add_argument("--max-runtime", type=int, default=10800)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--job-timeout", type=int, default=3600)
    parser.add_argument("--health-timeout", type=int, default=900)
    parser.add_argument("--proxy-url")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for name in ("repo", "reef_repo", "reef_python", "hf_cache", "output", "plan", "probe_token_ids", "image_provenance"):
        setattr(args, name, require_path(getattr(args, name), name))
    if args.reef_port in {8900, 8901, 30000} or args.actor_port in {8900, 8901, 30000} or args.reef_port == args.actor_port:
        parser.error("owned ports must not collide with preserved services")
    if not all(0 < p < 65536 for p in (args.reef_port, args.actor_port)):
        parser.error("ports must be in 1..65535")
    if min(args.max_runtime, args.startup_timeout, args.job_timeout, args.health_timeout) <= 0:
        parser.error("timeouts must be positive")
    if not args.model_dir.startswith("/hf/") or args.model_revision not in args.model_dir:
        parser.error("model-dir must be the pinned /hf snapshot containing model-revision")
    if args.proxy_url and args.proxy_url.rstrip("/") != "http://127.0.0.1:8901":
        parser.error("only the known separate loopback proxy may be postchecked")
    return args


def main():
    args = parse_args()
    if args.dry_run:
        owner = "reef-nvfp4-dry-run"
        print(json.dumps({"actor_command": actor_argv(args, args.image, owner),
                          "image_provenance": str(args.image_provenance),
                          "worker_command": worker_argv(args, args.image, owner, "HOST_REVISION"),
                          "config": build_config(args, worker_launcher(args), owner)}, indent=2))
        return 0
    supervisor = Supervisor(args)
    def interrupted(signum, frame):
        raise InterruptedError(f"supervisor received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    def expired(signum, frame):
        raise TimeoutError("global campaign deadline expired")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, args.max_runtime)
    result = 1
    try:
        result = supervisor.run()
    except Exception as exc:
        if supervisor.output_owned:
            write_json(args.output / "failure.json", {"error_type": type(exc).__name__, "message": str(exc)})
        print(f"Campaign failed: {type(exc).__name__}; inspect failure.json and logs", flush=True)
        result = 124 if isinstance(exc, TimeoutError) else 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if supervisor.cleanup():
            result = 1
    return result


if __name__ == "__main__":
    raise SystemExit(main())
