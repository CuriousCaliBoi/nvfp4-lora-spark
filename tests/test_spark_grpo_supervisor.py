import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_spark_grpo_smoke.sh"
FAKE_DOCKER = r'''
import json, os, pathlib, sys
root = pathlib.Path(os.environ["FAKE_STATE"])
args = sys.argv[1:]
with (root / "calls.jsonl").open("a") as f:
    f.write(json.dumps(args) + "\n")
state_path = root / "state.json"
state = json.loads(state_path.read_text())
scenario = os.environ.get("FAKE_SCENARIO", "success")
def save(): state_path.write_text(json.dumps(state))
def get(target):
    if target in ("production", "production-id"): return state["server"]
    if target in ("experiment", "training-id") and "training" in state: return state["training"]
    raise SystemExit(1)
if args[:2] == ["image", "inspect"]:
    print("sha256:cached-image")
elif args[0] == "inspect":
    item = get(args[-1])
    fmt = args[args.index("--format") + 1] if "--format" in args else ""
    if ".Config.Labels" in fmt: print(item.get("owner", ""))
    elif ".State.Running" in fmt: print(str(item["running"]).lower())
    elif ".State.ExitCode" in fmt: print(item.get("exit", 0))
    elif "json .State" in fmt: print(json.dumps({"Running": item["running"]}))
    elif ".Id" in fmt: print(item["id"])
    else: print(json.dumps(item))
elif args[0] == "create":
    owner = args[args.index("--label") + 1].split("=", 1)[1]
    state["training"] = {"id": "training-id", "running": False, "owner": owner, "exit": 0}
    save()
    pathlib.Path(args[args.index("--cidfile") + 1]).write_text("training-id\n")
    print("training-id")
    if scenario == "create_failure": raise SystemExit(125)
elif args[0] == "start":
    item = get(args[-1])
    item["running"] = True
    if item["id"] == "training-id":
        if scenario in ("success", "health_failure", "originally_stopped"):
            item["running"] = False
        if scenario == "training_failure":
            item["running"] = False
            item["exit"] = 7
    save()
    if scenario == "start_failure" and item["id"] == "training-id": raise SystemExit(125)
    if scenario == "term_signal" and item["id"] == "training-id":
        import signal
        os.kill(os.getppid(), signal.SIGTERM)
    print(item["id"])
elif args[0] in ("stop", "kill"):
    item = get(args[-1])
    item["running"] = False
    save()
    if scenario == "stop_failure" and item["id"] == "production-id": raise SystemExit(125)
    print(item["id"])
elif args[0] == "logs": print("fake training output")
else: raise SystemExit("unexpected Docker command: " + repr(args))
'''


class SparkSupervisorTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.repo = self.root / "repo"
        (self.repo / "scripts").mkdir(parents=True)
        (self.repo / "scripts" / "train_grpo.py").write_text("")
        (self.root / "hf").mkdir()
        self.output = self.root / "output"
        self.write_tool("docker", FAKE_DOCKER)
        self.write_tool("nvidia-smi", 'import os\nprint("1234" if os.environ.get("FAKE_SCENARIO") == "busy_gpu" else "")\n')
        self.write_tool("curl", 'import os, sys\nprint("healthy")\nsys.exit(1 if os.environ.get("FAKE_SCENARIO") == "health_failure" else 0)\n')
        self.write_tool("sleep", "import time\ntime.sleep(0.02)\n")

    def write_tool(self, name, source):
        path = self.bin / name
        path.write_text(f"#!{sys.executable}\n" + source)
        path.chmod(0o755)

    def run_supervisor(self, scenario="success", extra=()):
        (self.root / "state.json").write_text(json.dumps({"server": {
            "id": "production-id", "running": scenario != "originally_stopped"
        }}))
        env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
                   TMPDIR=str(self.root), FAKE_STATE=str(self.root), FAKE_SCENARIO=scenario)
        command = ["bash", str(SCRIPT), "--image", "cached:tag", "--repo", str(self.repo),
                   "--hf-cache", str(self.root / "hf"), "--vllm-cache", str(self.root / "vllm"),
                   "--output", str(self.output), "--server", "production", "--health-url", "http://localhost/health",
                   "--container-name", "experiment", "--max-runtime", "1", "--health-timeout", "1",
                   "--poll-interval", "1", "--", "--model-dir", "/hf/snapshot", *extra]
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10)
        state = json.loads((self.root / "state.json").read_text())
        calls = [json.loads(line) for line in (self.root / "calls.jsonl").read_text().splitlines()]
        return result, state, calls

    def assert_restored(self, state):
        self.assertTrue(state["server"]["running"])
        self.assertEqual((self.output / "restoration-status").read_text().strip(), "healthy")
        self.assertFalse(list(self.root.glob("nvfp4-grpo-*.lock")))

    def test_success_uses_read_only_inputs_and_pinned_image(self):
        result, state, calls = self.run_supervisor()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_restored(state)
        create = next(call for call in calls if call[0] == "create")
        self.assertIn("sha256:cached-image", create)
        self.assertIn(f"{self.repo}:/workspace:ro", create)
        self.assertIn(f"{self.root / 'hf'}:/hf:ro", create)
        self.assertIn("--network=none", create)
        self.assertIn("NVFP4_EVAL_CACHE_GB=0", create)
        self.assertEqual(create[-3:], ["--output-dir", "/experiment/results", "--offline"])
        self.assertIn("--model-dir /hf/snapshot", (self.output / "supervisor.txt").read_text())
        self.assertTrue((self.output / "original-server-state.json").exists())

    def test_training_failure_restores_server_and_preserves_exit(self):
        result, state, _ = self.run_supervisor("training_failure")
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assert_restored(state)
        self.assertEqual((self.output / "experiment-exit-code").read_text().strip(), "7")

    def test_timeout_stops_only_owned_training_container(self):
        result, state, calls = self.run_supervisor("timeout")
        self.assertEqual(result.returncode, 124, result.stderr)
        self.assert_restored(state)
        self.assertFalse(state["training"]["running"])
        stopped = [call[-1] for call in calls if call[0] in ("stop", "kill")]
        self.assertEqual(stopped, ["production-id", "training-id"])

    def test_originally_stopped_server_stays_stopped(self):
        result, state, calls = self.run_supervisor("originally_stopped")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(state["server"]["running"])
        self.assertFalse(any(call[0] in ("start", "stop") and call[-1] == "production-id" for call in calls))
        self.assertEqual((self.output / "restoration-status").read_text().strip(), "not-needed-originally-stopped")

    def test_busy_gpu_is_untouched_and_server_restored(self):
        result, state, calls = self.run_supervisor("busy_gpu")
        self.assertEqual(result.returncode, 1)
        self.assert_restored(state)
        self.assertNotIn("training", state)
        self.assertFalse(any(call[0] == "create" for call in calls))

    def test_ambiguous_create_failure_recovers_owned_id(self):
        result, state, _ = self.run_supervisor("create_failure")
        self.assertEqual(result.returncode, 125, result.stderr)
        self.assert_restored(state)
        self.assertTrue((self.output / "run.log").exists())

    def test_ambiguous_start_failure_stops_owned_container(self):
        result, state, _ = self.run_supervisor("start_failure")
        self.assertEqual(result.returncode, 125, result.stderr)
        self.assertFalse(state["training"]["running"])
        self.assert_restored(state)

    def test_ambiguous_server_stop_failure_restores_server(self):
        result, state, _ = self.run_supervisor("stop_failure")
        self.assertEqual(result.returncode, 125, result.stderr)
        self.assert_restored(state)
        self.assertNotIn("training", state)

    def test_sigterm_cleans_up_and_restores_server(self):
        result, state, _ = self.run_supervisor("term_signal")
        self.assertEqual(result.returncode, 143, result.stderr)
        self.assertFalse(state["training"]["running"])
        self.assert_restored(state)

    def test_failed_health_overrides_success(self):
        result, state, _ = self.run_supervisor("health_failure")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertTrue(state["server"]["running"])
        self.assertEqual((self.output / "experiment-exit-code").read_text().strip(), "0")
        self.assertEqual((self.output / "restoration-status").read_text().strip(), "failed-health")


if __name__ == "__main__":
    unittest.main()
