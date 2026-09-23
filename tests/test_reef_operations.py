import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace, ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_reef_gsm8k as campaign
import reef_supervisor as ops
import reef_worker_launch as launcher


@pytest.fixture
def settings(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    campaign.write_json(data / "holdout.json", {"rows": []})
    campaign.write_json(data / "campaign.json", {"holdout_sha256": campaign.sha256(data / "holdout.json")})
    campaign.write_json(data / "probe.json", [1, 2, 3])
    return SimpleNamespace(output=tmp_path / "run", repo=ROOT, reef_repo=tmp_path / "reef",
        reef_python=Path(sys.executable), hf_cache=tmp_path / "hf", model_dir="/hf/snapshots/abc",
        model_revision="abc", image="local-image", server="production", health_url="http://127.0.0.1:30000/health",
        reef_port=8902, actor_port=30001, max_runtime=300, startup_timeout=60, job_timeout=120,
        health_timeout=60, scenario="nvfp4-gsm8k", plan=data / "campaign.json", probe_token_ids=data / "probe.json",
        proxy_url=None)


def inspected_actor(args, owner):
    full = ops.actor_argv(args, "sha256:image", owner)
    position = full.index("sha256:image")
    return {"Id": "actor-id", "Image": "sha256:image", "Path": "python3", "Args": full[position + 1:],
            "State": {"Running": True}, "Name": "/" + owner + "-actor",
            "Config": {"Labels": {"nvfp4-reef.owner": owner}, "Env": ["VLLM_ALLOW_RUNTIME_LORA_UPDATING=1"]}}


def test_actor_attestation_uses_inspected_command(settings):
    item = inspected_actor(settings, "owner")
    result = ops.actor_attestation(item, "owner", "abc", "0.27.1")
    assert result["command"] == [item["Path"], *item["Args"]]
    assert result["container_id"] == "actor-id"
    assert result["speculative_decoding"] is False
    item["Args"][item["Args"].index("processed_logprobs")] = "raw_logprobs"
    with pytest.raises(ValueError, match="controlled sampling"):
        ops.actor_attestation(item, "owner", "abc", "0.27.1")


@pytest.mark.parametrize("mutation", ["speculation", "version", "owner", "runtime-loading", "capacity"])
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
    with pytest.raises(ValueError):
        ops.actor_attestation(item, "owner", "abc", version)


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
    fake = FakeDocker(settings, supervisor, running=outcome != "originally-stopped")
    monkeypatch.setattr(ops, "command", fake.command)
    monkeypatch.setattr(ops, "docker_inspect", fake.inspect)
    monkeypatch.setattr(ops, "free_port", lambda _: None)
    monkeypatch.setattr(ops.time, "sleep", lambda _: None)
    monkeypatch.setattr(ops, "get_json", lambda url, token=None: {"version": "0.27.1"} if url.endswith("/version") else {"ok": True})
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
    assert not any("8900" in part for row in fake.calls for part in row)
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
