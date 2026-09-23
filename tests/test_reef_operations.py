import importlib.util
import errno
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
from types import SimpleNamespace, ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_reef_gsm8k as campaign
import reef_supervisor as ops
import reef_worker_launch as launcher


@pytest.mark.skipif(os.name != "posix", reason="REEF supervision targets POSIX socket reuse semantics")
def test_restart_port_probe_accepts_real_time_wait():
    with socket.socket() as listener, socket.socket() as client:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.settimeout(2)
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        listener.listen(1)
        client.settimeout(2)
        client.connect(("127.0.0.1", port))
        connection, _ = listener.accept()
        with connection:
            connection.settimeout(2)
            connection.shutdown(socket.SHUT_WR)
            assert client.recv(1) == b""
            client.shutdown(socket.SHUT_WR)
            assert connection.recv(1) == b""
    with socket.socket() as old_probe, pytest.raises(OSError) as failure:
        old_probe.bind(("127.0.0.1", port))
    assert failure.value.errno == errno.EADDRINUSE
    ops.free_port(port)
    with socket.socket() as restarted:
        restarted.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        restarted.bind(("127.0.0.1", port))
        restarted.listen(1)


@pytest.mark.parametrize("address", ["127.0.0.1", "0.0.0.0"])
@pytest.mark.parametrize("reuse", [False, True])
def test_port_probe_still_rejects_active_listener(address, reuse):
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, int(reuse))
        listener.bind((address, 0))
        port = listener.getsockname()[1]
        listener.listen(1)
        with pytest.raises(OSError) as failure:
            ops.free_port(port)
        assert failure.value.errno == errno.EADDRINUSE
        assert listener.getsockname()[1] == port


@pytest.fixture
def settings(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    campaign.write_json(data / "holdout.json", {"rows": []})
    campaign.write_json(data / "campaign.json", {"holdout_sha256": campaign.sha256(data / "holdout.json")})
    campaign.write_json(data / "probe.json", [1, 2, 3])
    campaign.write_json(data / "image-build.json", {"fake_build": True})
    return SimpleNamespace(output=tmp_path / "run", repo=ROOT, reef_repo=tmp_path / "reef",
        reef_python=Path(sys.executable), hf_cache=tmp_path / "hf", model_dir="/hf/snapshots/abc",
        model_revision="abc", image="local-image", server="production", health_url="http://127.0.0.1:30000/health",
        reef_port=8902, actor_port=30001, max_runtime=300, startup_timeout=60, job_timeout=120,
        health_timeout=60, scenario="nvfp4-gsm8k", plan=data / "campaign.json", probe_token_ids=data / "probe.json",
        proxy_url=None, image_provenance=data / "image-build.json")


def inspected_actor(args, owner):
    full = ops.actor_argv(args, "sha256:image", owner)
    position = full.index("sha256:image")
    return {"Id": "actor-id", "Image": "sha256:image", "Path": "python3", "Args": full[position + 1:],
            "State": {"Running": True}, "Name": "/" + owner + "-actor",
            "Config": {"Labels": {"nvfp4-reef.owner": owner},
                       "Env": [full[i + 1] for i, arg in enumerate(full[:position]) if arg == "-e"]}}


def test_actor_attestation_uses_inspected_command(settings):
    item = inspected_actor(settings, "owner")
    assert "--no-enable-log-requests" in item["Args"]
    assert "--disable-log-requests" not in item["Args"]
    assert "--no-async-scheduling" in item["Args"]
    result = ops.actor_attestation(item, "owner", "abc", "0.27.1")
    assert result["command"] == [item["Path"], *item["Args"]]
    assert result["container_id"] == "actor-id"
    assert result["speculative_decoding"] is False
    assert result["async_scheduling"] is False
    assert result["max_num_seqs"] == 1
    assert result["kv_cache_dtype"] == "bfloat16"
    assert result["attention_backend"] == "TRITON_ATTN"
    assert result["mamba_cache_mode"] == "none"
    assert result["cublas_workspace_config"] == ":4096:8"
    for flag, expected in {"--moe-backend": "marlin", "--mamba-backend": "flashinfer",
                           "--gpu-memory-utilization": "0.23", "--max-model-len": "1024",
                           "--max-lora-rank": "8"}.items():
        assert item["Args"][item["Args"].index(flag) + 1] == expected
    assert "--enforce-eager" in item["Args"]
    assert "--no-enable-prefix-caching" in item["Args"]
    item["Args"][item["Args"].index("processed_logprobs")] = "raw_logprobs"
    with pytest.raises(ValueError, match="controlled sampling"):
        ops.actor_attestation(item, "owner", "abc", "0.27.1")


@pytest.mark.parametrize("mutation", ["speculation", "version", "owner", "runtime-loading", "capacity",
                                     "async-default", "async-enabled", "async-enabled-equals"])
def test_actor_attestation_rejects_unreviewed_actor(settings, mutation):
    item = inspected_actor(settings, "owner")
    version = "0.27.1"
    if mutation == "speculation":
        item["Args"] += ["--speculative-config", "{}"]
    if mutation == "version":
        version = "0.28.0"
    if mutation == "owner":
        item["Config"]["Labels"] = {}
    if mutation == "runtime-loading":
        item["Config"]["Env"] = []
    if mutation == "capacity":
        item["Args"][item["Args"].index("--max-cpu-loras") + 1] = "2"
    if mutation == "async-default":
        item["Args"].remove("--no-async-scheduling")
    if mutation == "async-enabled":
        item["Args"].append("--async-scheduling")
    if mutation == "async-enabled-equals":
        item["Args"].append("--async-scheduling=true")
    with pytest.raises(ValueError):
        ops.actor_attestation(item, "owner", "abc", version)


@pytest.mark.parametrize("flag,alternate", [("--max-num-seqs", "32"), ("--kv-cache-dtype", "fp8"),
                                           ("--attention-backend", "FLASH_ATTN"), ("--mamba-cache-mode", "align")])
@pytest.mark.parametrize("mutation", ["missing", "changed", "conflicting"])
def test_actor_attestation_rejects_profile_drift(settings, flag, alternate, mutation):
    item = inspected_actor(settings, "owner")
    position = item["Args"].index(flag)
    if mutation == "missing":
        del item["Args"][position:position + 2]
    elif mutation == "changed":
        item["Args"][position + 1] = alternate
    else:
        item["Args"].append(flag + "=" + alternate)
    with pytest.raises(ValueError):
        ops.actor_attestation(item, "owner", "abc", "0.27.1")


@pytest.mark.parametrize("values", [[], ["CUBLAS_WORKSPACE_CONFIG=:16:8"],
                                     ["CUBLAS_WORKSPACE_CONFIG=:4096:8", "CUBLAS_WORKSPACE_CONFIG=:16:8"]])
def test_actor_attestation_requires_inspected_cublas_environment(settings, values):
    item = inspected_actor(settings, "owner")
    item["Config"]["Env"] = [entry for entry in item["Config"]["Env"]
                              if not entry.startswith("CUBLAS_WORKSPACE_CONFIG=")] + values
    with pytest.raises(ValueError, match="cuBLAS workspace"):
        ops.actor_attestation(item, "owner", "abc", "0.27.1")


def test_worker_launch_preserves_argv_paths_and_unique_ownership(settings):
    prefix = ops.worker_argv(settings, "sha256:image", "owner", "abc123")
    config = {"owner": "owner", "argv": prefix}
    state = settings.output / "learning"
    first = launcher.worker_command(config, state / "jobs/job file.json", state / "checkpoints/a", state / "registry", "one")
    second = launcher.worker_command(config, state / "jobs/job file.json", state / "checkpoints/a", state / "registry", "two")
    assert first[0] != second[0]
    assert first[2][-4:] == ["--job-file", str(state / "jobs/job file.json"), "--checkpoint-dir", str(state / "checkpoints/a")]
    assert first[2][first[2].index("--user") + 1] == f"{ops.os.getuid()}:{ops.os.getgid()}"
    assert "--network" in first[2] and "none" in first[2]
    assert f"{settings.repo}:/workspace:ro" in first[2]
    assert "NVFP4_SOURCE_REVISION=abc123" in first[2]


def test_config_separates_preserved_services_and_credentials(settings):
    value = ops.build_config(settings, ["worker", "literal space"], "owner")
    assert value["reef"]["port"] == 8902
    assert value["reef"]["token"] == "${REEF_TOKEN}"
    assert value["training"]["options"]["actor-url"] == "http://127.0.0.1:30001"
    assert value["training"]["options"]["worker-command"] == ["worker", "literal space"]
    campaign.write_json(settings.plan.parent / "holdout.json", {"changed": True})
    with pytest.raises(ValueError, match="held-out"):
        ops.build_config(settings, ["worker"], "owner")


class FakeDocker:
    def __init__(self, settings, supervisor, running=True):
        self.settings, self.supervisor = settings, supervisor
        self.calls = []
        self.items = {"production": {"Id": "original-id", "Image": "original-image", "Name": "/production",
            "State": {"Running": running}, "Config": {"Labels": {}}}}
        self.items["original-id"] = self.items["production"]
        self.other_stopped = False

    def inspect(self, identifier):
        return self.items[identifier]

    def command(self, argv, **kwargs):
        argv = list(map(str, argv))
        self.calls.append(argv)
        if argv[0] == "git":
            if "status" in argv:
                return ""
            return ops.REEF_REVISION if argv[2] == str(self.settings.reef_repo) else "a" * 40
        if argv[0] == "nvidia-smi":
            return ""
        if argv[1:3] == ["image", "inspect"]:
            return "sha256:image"
        if argv[1] == "create":
            self.items["actor-id"] = inspected_actor(self.settings, self.supervisor.owner)
            self.items["actor-id"]["State"]["Running"] = False
            return "actor-id"
        if argv[1] in {"start", "stop", "kill"}:
            identifier = argv[-1]
            if identifier not in self.items:
                self.other_stopped = True
                raise ValueError("attempted mutation of unrelated container")
            self.items[identifier]["State"]["Running"] = argv[1] == "start"
            return identifier
        if argv[1] == "ps":
            return "actor-id" if "actor-id" in self.items else ""
        raise AssertionError(argv)


@pytest.mark.parametrize("outcome", ["success", "failure", "timeout", "inconclusive", "originally-stopped"])
def test_supervisor_restores_exact_original_on_all_outcomes(settings, monkeypatch, outcome):
    supervisor = ops.Supervisor(settings)
    supervisor.owner = "reef-nvfp4-8900-test-owner"
    fake = FakeDocker(settings, supervisor, running=outcome != "originally-stopped")
    monkeypatch.setattr(ops, "command", fake.command)
    monkeypatch.setattr(ops, "docker_inspect", fake.inspect)
    monkeypatch.setattr(ops, "free_port", lambda _: None)
    monkeypatch.setattr(ops.time, "sleep", lambda _: None)
    monkeypatch.setattr(ops, "get_json", lambda url, token=None: {"version": "0.27.1"} if url.endswith("/version") else {"ok": True})
    monkeypatch.setattr(ops, "validate_build_report", lambda report, *args: report)
    monkeypatch.setattr(ops, "verify_actor_patch", lambda *args, **kwargs: {"marlin_installed_source_sha256": "verified"})
    monkeypatch.setattr(ops.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="", stderr="", returncode=0))
    monkeypatch.setattr(supervisor, "start_reef", lambda suffix: None)
    monkeypatch.setattr(supervisor, "check_worker_permissions", lambda *args: None)
    events = []
    def campaign_call(name, extra, **kwargs):
        events.append(name)
        if name == "cycle":
            if outcome == "failure":
                raise RuntimeError("fake worker failed")
            if outcome == "timeout":
                raise TimeoutError("fake deadline expired")
            if outcome == "inconclusive":
                return 2
        return 0
    monkeypatch.setattr(supervisor, "campaign", campaign_call)
    try:
        if outcome in {"failure", "timeout"}:
            with pytest.raises((RuntimeError, TimeoutError)):
                supervisor.run()
        else:
            assert supervisor.run() == (2 if outcome == "inconclusive" else 0)
    finally:
        assert supervisor.cleanup() == []
    assert fake.items["original-id"]["State"]["Running"] is (outcome != "originally-stopped")
    assert fake.items["actor-id"]["State"]["Running"] is False
    assert not fake.other_stopped
    assert (settings.output / "learning/serving/actor-contract.json").exists()
    publications = [row[row.index("-p") + 1] for row in fake.calls if "-p" in row]
    assert publications == ["127.0.0.1:30001:8000"]
    if outcome == "success":
        assert events == ["create", "snapshot", "cycle", "verify-resume", "cycle", "rollback"]


def test_existing_output_never_overwritten(settings, monkeypatch):
    settings.output.mkdir()
    marker = settings.output / "restoration.json"
    marker.write_text("existing evidence")
    supervisor = ops.Supervisor(settings)
    monkeypatch.setattr(ops, "command", lambda *a, **kw: "")
    with pytest.raises(FileExistsError):
        supervisor.run()
    supervisor.cleanup()
    assert marker.read_text() == "existing evidence"


def test_resume_checks_full_optimizer_and_rng_bytes():
    base = {"checkpoint": {"checkpoint_id": "one", "files": {"optimizer.pt": "a", "rng.pt": "b"}},
            "status": {"current_runtime_load_id": "old"}}
    after = json.loads(json.dumps(base))
    after["status"]["current_runtime_load_id"] = "new"
    campaign.verify_continuation(base, after, restarted=True)
    after["checkpoint"]["files"]["optimizer.pt"] = "changed"
    with pytest.raises(RuntimeError, match="complete training checkpoint"):
        campaign.verify_continuation(base, after, restarted=True)


def test_exited_parent_with_reused_pid_is_never_signalled(monkeypatch):
    process = SimpleNamespace(pid=12000, poll=lambda: 0)
    monkeypatch.setattr(ops.os, "getpgid", lambda _: 12000)
    monkeypatch.setattr(ops.os, "killpg", lambda *a: pytest.fail("reused group must not be signalled"))
    ops.Supervisor.stop_process(process)


def test_command_calls_have_bounded_timeout(monkeypatch):
    def run(argv, **kwargs):
        assert kwargs["timeout"] == 60
        return SimpleNamespace(stdout="ok")
    monkeypatch.setattr(ops.subprocess, "run", run)
    assert ops.command(["docker", "inspect", "owned"]) == "ok"


def test_actor_health_wait_fails_immediately_when_owned_container_exits(settings, monkeypatch):
    supervisor = ops.Supervisor(settings)
    settings.output.mkdir()
    actor = inspected_actor(settings, supervisor.owner)
    actor["State"] = {"Running": False, "ExitCode": 2, "Status": "exited"}
    monkeypatch.setattr(ops, "docker_inspect", lambda identifier: actor)
    monkeypatch.setattr(ops, "get_json", lambda *a, **kw: pytest.fail("exited actor must not reach a health request"))
    monkeypatch.setattr(ops.time, "sleep", lambda _: pytest.fail("exited actor must not wait through startup timeout"))
    with pytest.raises(RuntimeError, match="owned actor exited"):
        supervisor.wait_health("http://127.0.0.1:30001/health", 900, container_id="actor-id")
    evidence = campaign.read_json(settings.output / "actor-startup-failure.json")
    assert evidence["container_id"] == "actor-id"
    assert evidence["state"]["ExitCode"] == 2


def test_actor_health_wait_rejects_changed_ownership(settings, monkeypatch):
    supervisor = ops.Supervisor(settings)
    actor = inspected_actor(settings, "someone-else")
    monkeypatch.setattr(ops, "docker_inspect", lambda identifier: actor)
    monkeypatch.setattr(ops, "get_json", lambda *a, **kw: pytest.fail("unowned actor must not be accepted"))
    with pytest.raises(RuntimeError, match="ownership changed"):
        supervisor.wait_health("http://127.0.0.1:30001/health", 900, container_id="actor-id")


def test_worker_permissions_checked_without_gpu_before_training(settings, monkeypatch):
    supervisor = ops.Supervisor(settings)
    (settings.output / "learning").mkdir(parents=True)
    def run(argv, **kwargs):
        assert "--gpus" not in argv
        assert "--user" in argv
        assert argv[-2] == "-c"
        (settings.output / "learning/.permission-probe").write_text("host-readable")
        return ""
    monkeypatch.setattr(ops, "command", run)
    supervisor.check_worker_permissions(ops.worker_argv(settings, "image", supervisor.owner, "a" * 40), "image")
    assert not (settings.output / "learning/.permission-probe").exists()


def test_no_services_started_for_help():
    for script in ("reef_supervisor.py", "run_reef_gsm8k.py", "reef_worker_launch.py"):
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--help"], capture_output=True, text=True)
        assert result.returncode == 0
        assert "usage:" in result.stdout


def test_campaign_reports_complete_real_receipt_grid_and_reuses_saved_receipts(tmp_path, monkeypatch):
    data = ModuleType("nvfp4_lora.reef_data")
    data.GSM8K_SYSTEM_PROMPT = "strict math prompt"
    data.gsm8k_reward = lambda text, gold: float(text == gold)
    checked = []
    def validate_capture(capture, version):
        assert capture == {"runtime_load_id": "runtime"}
        assert version == "runtime"
        checked.append(version)
    data.validate_capture = validate_capture
    monkeypatch.setitem(sys.modules, "nvfp4_lora.reef_data", data)
    class Client:
        def __init__(self):
            self.requests = []
        def request(self, path, payload, scenario):
            self.requests.append((path, payload))
            if path == "/v1/chat/completions":
                n = len(self.requests)
                return {"choices": [{"message": {"content": "#### 42"}, "finish_reason": "length" if n == 1 else "stop"}]}, {"X-Reef-Agent-Record-Id": "receipt-" + str(n)}
            assert len([row for row in self.requests if row[0] == "/v1/chat/completions"]) == 32
            return {"agent_record_id": payload["agent_record_id"]}, {}
        def get(self, path):
            return {"payload": {"runtime_load_id": "runtime", "response": {"training": {"runtime_load_id": "runtime"}}}}
    client = Client()
    declaration = {"cycle_id": "cycle-1", "rows": [{"sample_id": str(i), "question": str(i), "answer": "#### 42"} for i in range(4)]}
    before = {"status": {"current_runtime_load_id": "runtime"}}
    first = campaign.collect_and_report(client, "research", declaration, tmp_path, before)
    assert len(set(first)) == 32
    reports = [body for path, body in client.requests if path == "/reef/report"]
    assert sum(row["score"] for row in reports) == 31
    assert {row["metadata"]["dataset_split"] for row in reports} == {"train"}
    assert {row["metadata"]["dataset_revision"] for row in reports} == {campaign.DATASET_REVISION}
    assert all(len(row["references"]) == 1 for row in reports)
    assert campaign.collect_and_report(client, "research", declaration, tmp_path, before) == first
    assert len(client.requests) == 64
    assert len(checked) == 64


def load_deployment():
    pytest.importorskip("reef")
    spec = importlib.util.spec_from_file_location("operations_deployment", ROOT / "nvfp4_lora/reef_deployment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def factory_config(tmp_path):
    return {"state_root": str(tmp_path), "model_dir": "/hf/model", "model_revision": "a" * 40,
            "worker_command": ["worker", "literal argument"], "actor_url": "http://127.0.0.1:30001",
            "actor_instance_id": "owner", "probe_token_ids": [1, 2], "reef_revision": ops.REEF_REVISION}


def test_factory_bootstrap_precedes_serving_and_wires_recovery_callback(tmp_path, monkeypatch):
    deployment = load_deployment()
    events = []
    class Training:
        def __init__(self, **kwargs):
            events.append("bootstrap")
            self.checkpoint_root = tmp_path / "checkpoints"
            self.base_checkpoint = self.checkpoint_root / "base"
        def restore_checkpoint(self, artifact):
            events.append("restore")
        def shutdown(self):
            events.append("shutdown")
    def inference(**kwargs):
        assert events == ["bootstrap"]
        assert kwargs["base_checkpoint"] == tmp_path / "checkpoints/base"
        kwargs["on_checkpoint_activation"]("artifact")
        events.append("serving")
        return "inference"
    training_module = ModuleType("nvfp4_lora.reef_training")
    training_module.QuantizedTrainingRuntime = Training
    inference_module = ModuleType("nvfp4_lora.reef_inference")
    inference_module.VllmInferenceRuntime = inference
    monkeypatch.setitem(sys.modules, "nvfp4_lora.reef_training", training_module)
    monkeypatch.setitem(sys.modules, "nvfp4_lora.reef_inference", inference_module)
    pair = deployment.runtime_factory(factory_config(tmp_path), campaign.MODEL, {}, {})
    assert events == ["bootstrap", "restore", "serving"]
    assert pair[1] == "inference"
    def broken(**kwargs):
        raise RuntimeError("native actor failed")
    inference_module.VllmInferenceRuntime = broken
    with pytest.raises(RuntimeError, match="native actor failed"):
        deployment.runtime_factory(factory_config(tmp_path), campaign.MODEL, {}, {})
    assert events[-1] == "shutdown"


@pytest.mark.parametrize("key,value", [("max_staleness", 1), ("lora_rank", 16),
    ("worker_command", "shell string"), ("actor_url", "https://remote.example"), ("probe_token_ids", [True, 2]),
    ("reef_revision", "unknown"), ("adapter_capacity", 2)])
def test_factory_rejects_unsupported_runtime_configuration(tmp_path, key, value):
    deployment = load_deployment()
    config = factory_config(tmp_path)
    config[key] = value
    with pytest.raises(ValueError):
        deployment.runtime_settings(config)


def snapshot_fixture(tmp_path, monkeypatch):
    state = tmp_path / "state"
    values = {"identity": "old", "step": 1}
    manifests = {}
    def publish(identity, step):
        path = state / "checkpoints" / identity
        path.mkdir(parents=True, exist_ok=True)
        manifest = {"checkpoint_id": identity, "optimizer_step": step, "adapter_sha256": "weights-" + identity,
                    "files": {"adapter/adapter_config.json": "config-" + identity, "optimizer.pt": "opt-" + identity}}
        manifests[str(path.resolve())] = manifest
        values.update(identity=identity, step=step)
        campaign.write_json(state / "incumbent.json", {"checkpoint_id": identity, "checkpoint_path": str(path)})
        campaign.write_json(state / "serving/serving-state.json", {
            "active_checkpoint_id": identity, "fenced": False, "pending": None,
            "published_runtime_load_id": "runtime-" + identity, "active_release": "release-" + identity,
            "bindings": {identity: {"native_verified": True, "adapter_sha256": "weights-" + identity,
                "adapter_config_sha256": "config-" + identity, "reload_max_delta": 0,
                "adapter_effect_max_delta": 0.1, "reference_repeat_max_delta": 0}},
        })
    def status():
        identity = values["identity"]
        return {"scenarios": {"research": {"scenario_step": values["step"],
                "artifact_head_sync": {"state": "synchronized", "release_id": "release-" + identity},
                "current_runtime_load_id": "runtime-" + identity, "inference_admission": {"open": True, "active": 0}}}}
    def releases():
        return {"releases": [{"current": True, "release_id": "release-" + values["identity"]}]}
    module = ModuleType("nvfp4_lora.reef_checkpoint")
    module.validate_checkpoint = lambda path: manifests[str(path)]
    monkeypatch.setitem(sys.modules, "nvfp4_lora.reef_checkpoint", module)
    publish("old", 1)
    return state, values, module, publish, status, releases


def test_snapshot_retries_when_publication_changes_during_checkpoint_read(tmp_path, monkeypatch):
    state, values, module, publish, status, releases = snapshot_fixture(tmp_path, monkeypatch)
    original_validate = module.validate_checkpoint
    reads = []
    def validate(path):
        manifest = original_validate(path)
        reads.append(manifest["checkpoint_id"])
        if len(reads) == 1:
            publish("new", 2)
        return manifest
    module.validate_checkpoint = validate
    class Client:
        def get(self, path, **kwargs):
            return status() if path == "/reef/status" else releases()
    result = campaign.snapshot(Client(), "research", state, timeout=1)
    assert reads == ["old", "new"]
    assert result["checkpoint"]["checkpoint_id"] == "new"
    assert result["status"]["current_runtime_load_id"] == "runtime-new"
    assert result["release"]["release_id"] == "release-new"


def test_commit_visibility_does_not_count_as_settlement(tmp_path, monkeypatch):
    state, values, module, publish, status, releases = snapshot_fixture(tmp_path, monkeypatch)
    directory = tmp_path / "evidence"
    directory.mkdir()
    events = []
    reads = 0
    class Client:
        def get(self, path, **kwargs):
            nonlocal reads
            if "/commits?" in path:
                events.append("durable-row-visible")
                return {"commits": [{"step": 2, "pending": False, "consumed_ids": ["report"]}]}
            if path.endswith("/releases"):
                return releases()
            reads += 1
            if reads == 4:
                publish("new", 2)
                events.append("publication-acknowledged")
            return status()
    result = campaign.await_commit(Client(), "research", ["report"], 1,
                                   campaign.time.monotonic() + 2, directory, state_root=state)
    assert result["step"] == 2
    assert events == ["durable-row-visible", "publication-acknowledged"]
    settled = campaign.read_json(directory / "settled.json")
    assert settled["status"]["scenario_step"] == 2
    assert settled["serving"]["published_runtime_load_id"] == "runtime-new"


def test_snapshot_never_invents_acknowledgement_when_publication_stays_pending(tmp_path, monkeypatch):
    state, values, module, publish, status, releases = snapshot_fixture(tmp_path, monkeypatch)
    now = [0.0]
    monkeypatch.setattr(campaign.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(campaign.time, "sleep", lambda delay: now.__setitem__(0, now[0] + delay))
    class Client:
        def get(self, path, **kwargs):
            if "/commits?" in path:
                return {"commits": [{"step": 2, "pending": False, "consumed_ids": ["report"]}]}
            return status() if path == "/reef/status" else releases()
    directory = tmp_path / "evidence"
    with pytest.raises(TimeoutError, match="no settled REEF publication"):
        campaign.await_commit(Client(), "research", ["report"], 1, 0.3, directory, state_root=state)
    assert not (directory / "settled.json").exists()
    assert campaign.read_json(state / "incumbent.json")["checkpoint_id"] == "old"
