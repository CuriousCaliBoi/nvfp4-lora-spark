import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_reef_marlin_image as image
from test_reef_operations import settings


BASE = "sha256:" + "a" * 64
DERIVED = "sha256:" + "b" * 64


@pytest.fixture
def setup(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    for relative in image.INPUTS:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative + "\n")
    source = tmp_path / "marlin_moe.py"
    source.write_text("patched = True\n")
    manifest = {"patched_sha256": image.file_hash(source), "vllm_version": "0.27.1"}
    raw_manifest = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(raw_manifest)
    api = SimpleNamespace(MARLIN_PATCH_MANIFEST=manifest, MARLIN_SOURCE_PATH=source,
        MARLIN_MANIFEST_PATH=manifest_path, MARLIN_BASE_IMAGE_ID=BASE,
        MARLIN_PATCH_MANIFEST_SHA256=hashlib.sha256(raw_manifest).hexdigest())
    monkeypatch.setattr(image, "patch_api", lambda _: api)
    hashes = {path: image.file_hash(repo / path) for path in image.INPUTS}
    runtime = {"manifest": manifest, "manifest_sha256": api.MARLIN_PATCH_MANIFEST_SHA256,
               "installed_source_sha256": image.file_hash(source),
               "installer_sha256": hashes["nvfp4_lora/reef_marlin_patch.py"], "vllm_version": "0.27.1"}
    report = {"schema_version": 1, "image_id": DERIVED, "base_image_id": BASE,
              "source_revision": "c" * 40, "input_sha256": hashes, "runtime_provenance": runtime}
    return SimpleNamespace(repo=repo, api=api, report=report, runtime=runtime, source=source,
                           args=SimpleNamespace(repo=repo, output=tmp_path / "build.json",
                                                tag="reef-marlin-order:test", timeout=60))


def test_runtime_reader_hashes_actual_files_without_importing_vllm(setup, monkeypatch, tmp_path):
    installer = setup.repo / "nvfp4_lora/reef_marlin_patch.py"
    monkeypatch.setattr(image, "INSTALLER_PATH", str(installer))
    metadata = tmp_path / "vllm-0.27.1.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: vllm\nVersion: 0.27.1\n")
    (tmp_path / "vllm.py").write_text("raise AssertionError('must not import the GPU runtime')\n")
    result = image.command(image.runtime_read_command([sys.executable], setup.api),
                           env={**os.environ, "PYTHONPATH": str(tmp_path)})
    assert json.loads(result) == setup.runtime
    setup.source.write_text("tampered = True\n")
    changed = json.loads(image.command(image.runtime_read_command([sys.executable], setup.api),
                                      env={**os.environ, "PYTHONPATH": str(tmp_path)}))
    with pytest.raises(ValueError, match="installed Marlin"):
        image.validate_runtime_provenance(changed, setup.api, setup.runtime["installer_sha256"])


@pytest.mark.parametrize("failure", [subprocess.CalledProcessError(1, ["docker", "build"], "build output", "build error"),
                                    subprocess.TimeoutExpired(["docker", "build"], 1, b"partial output", b"partial error")])
def test_failed_build_commands_print_captured_diagnostics(monkeypatch, capsys, failure):
    def failed(*args, **kwargs):
        raise failure
    monkeypatch.setattr(image.subprocess, "run", failed)
    with pytest.raises(type(failure)):
        image.command(["docker", "build"])
    captured = capsys.readouterr().err
    assert "output" in captured and "error" in captured


@pytest.mark.parametrize("field", ["manifest", "manifest_sha256", "installed_source_sha256",
                                   "installer_sha256", "vllm_version"])
def test_runtime_reader_rejects_each_unreviewed_component(setup, field):
    changed = dict(setup.runtime)
    changed[field] = "changed"
    with pytest.raises(ValueError):
        image.validate_runtime_provenance(changed, setup.api, setup.runtime["installer_sha256"])


@pytest.mark.parametrize("field,value", [("image_id", BASE), ("base_image_id", DERIVED),
                                        ("source_revision", "uncommitted"), ("input_sha256", {})])
def test_build_report_rejects_wrong_image_or_sources(setup, field, value):
    changed = dict(setup.report, **{field: value})
    with pytest.raises(ValueError):
        image.validate_build_report(changed, DERIVED, setup.repo)


def test_actor_verification_binds_owned_container_and_actual_image(setup):
    calls = []
    actor = {"Id": "owned-id", "Image": DERIVED, "Config": {"Labels": {"nvfp4-reef.owner": "owner"}}}
    def run(argv, **kwargs):
        calls.append(argv)
        return json.dumps(setup.runtime)
    result = image.verify_actor_patch(actor, "owner", setup.report, setup.repo, run=run)
    assert calls[0][:4] == ["docker", "exec", "owned-id", "python3"]
    assert result["marlin_installed_source_sha256"] == setup.runtime["installed_source_sha256"]
    assert result["base_image_id"] == BASE
    for changed in [dict(actor, Image=BASE), dict(actor, Config={"Labels": {}})]:
        with pytest.raises(ValueError):
            image.verify_actor_patch(changed, "owner", setup.report, setup.repo, run=run)
    assert len(calls) == 1


class FakeBuildDocker:
    def __init__(self, setup, outcome):
        self.setup, self.outcome = setup, outcome
        self.calls, self.tags = [], {}
        self.built = False

    def __call__(self, argv, **kwargs):
        argv = list(map(str, argv))
        self.calls.append(argv)
        if argv[0] == "git":
            return "" if "status" in argv else "c" * 40
        if argv[1:3] == ["image", "ls"]:
            return DERIVED if self.outcome == "existing-tag" and argv[-1] == self.setup.args.tag else ""
        if argv[1:3] == ["image", "inspect"]:
            name = argv[-1]
            if name == BASE:
                return BASE
            if self.outcome == "base-drift" and self.built and name.startswith("reef-marlin-base:"):
                return DERIVED
            return self.tags[name]
        if argv[1] == "tag":
            self.tags[argv[-1]] = argv[-2]
            return ""
        if argv[1:3] == ["image", "rm"]:
            del self.tags[argv[-1]]
            return ""
        if argv[1] == "build":
            assert "--network=none" in argv and "--pull=false" in argv
            context = Path(argv[-1])
            assert set(path.name for path in context.iterdir()) == {"Dockerfile", "vendor", "reef_marlin_patch.py"}
            self.built = True
            if self.outcome == "build-failure":
                raise subprocess.CalledProcessError(1, argv)
            Path(argv[argv.index("--iidfile") + 1]).write_text(DERIVED + "\n")
            return ""
        if argv[1] == "run":
            assert "--gpus" not in argv and argv[argv.index("--network") + 1] == "none"
            assert argv[argv.index("--runtime") + 1] == "runc"
            assert "NVIDIA_VISIBLE_DEVICES=void" in argv and "--read-only" in argv
            evidence = dict(self.setup.runtime)
            if self.outcome == "bad-source":
                evidence["installed_source_sha256"] = "bad"
            return json.dumps(evidence)
        raise AssertionError(argv)


@pytest.mark.parametrize("outcome", ["success", "existing-tag", "base-drift", "build-failure", "bad-source"])
def test_offline_build_never_replaces_base_or_existing_tags(setup, monkeypatch, outcome):
    docker = FakeBuildDocker(setup, outcome)
    monkeypatch.setattr(image, "command", docker)
    if outcome == "success":
        result = image.build(setup.args)
        assert result["image_id"] == DERIVED
        assert json.loads(setup.args.output.read_text()) == result
        assert docker.tags == {setup.args.tag: DERIVED}
    else:
        with pytest.raises((ValueError, RuntimeError, subprocess.CalledProcessError)):
            image.build(setup.args)
        assert not setup.args.output.exists()
        assert setup.args.tag not in docker.tags
    for call in docker.calls:
        assert not any(part in {"--gpus", "start", "stop", "kill", "pull"} for part in call)
        if call[1:3] == ["image", "rm"]:
            assert call[-1].startswith("reef-marlin-base:")
        if call[1] == "tag":
            assert call[-1] == setup.args.tag or call[-1].startswith("reef-marlin-base:")


def test_patch_failure_prevents_reef_admission(settings, monkeypatch):
    import reef_supervisor as ops
    from test_reef_operations import FakeDocker
    supervisor = ops.Supervisor(settings)
    docker = FakeDocker(settings, supervisor)
    monkeypatch.setattr(ops, "command", docker.command)
    monkeypatch.setattr(ops, "docker_inspect", docker.inspect)
    monkeypatch.setattr(ops, "free_port", lambda _: None)
    monkeypatch.setattr(ops.time, "sleep", lambda _: None)
    monkeypatch.setattr(ops, "get_json", lambda *args: {"version": "0.27.1"})
    monkeypatch.setattr(ops, "validate_build_report", lambda report, *args: report)
    monkeypatch.setattr(supervisor, "check_worker_permissions", lambda *args: None)
    started = []
    monkeypatch.setattr(supervisor, "start_reef", lambda *args: started.append(args))
    def rejected(*args, **kwargs):
        raise ValueError("actual actor source mismatch")
    monkeypatch.setattr(ops, "verify_actor_patch", rejected)
    try:
        with pytest.raises(ValueError, match="actual actor source"):
            supervisor.run()
        assert not started
        assert not (settings.output / "learning/serving/actor-contract.json").exists()
    finally:
        monkeypatch.setattr(ops.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="", stderr="", returncode=0))
        assert supervisor.cleanup() == []
    assert docker.items["original-id"]["State"]["Running"]
