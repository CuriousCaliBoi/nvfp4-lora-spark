#!/usr/bin/env python3
"""Launch one uniquely owned GPU worker without shell argument interpolation."""

import argparse
import json
from pathlib import Path
import secrets
import signal
import subprocess
import time

from run_reef_gsm8k import read_json, write_json


def worker_command(config, job_file, checkpoint_dir, registry, nonce):
    prefix = list(config["argv"])
    owner = config["owner"]
    if prefix[:2] != ["docker", "run"] or "--name" not in prefix:
        raise ValueError("expected the supervisor-owned Docker worker command")
    if "nvfp4-reef.owner=" + owner not in prefix:
        raise ValueError("worker command lacks its ownership label")
    name = owner + "-worker-" + nonce
    prefix[prefix.index("--name") + 1] = name
    cidfile = registry / (name + ".cid")
    prefix[2:2] = ["--cidfile", str(cidfile)]
    return name, cidfile, [*prefix, "--job-file", str(job_file), "--checkpoint-dir", str(checkpoint_dir)]


def stop_owned(identifier, owner):
    inspection = subprocess.run(["docker", "inspect", identifier], text=True, capture_output=True, timeout=15)
    if inspection.returncode:
        return
    item = json.loads(inspection.stdout)[0]
    if item["Config"].get("Labels", {}).get("nvfp4-reef.owner") != owner:
        raise RuntimeError("refusing to stop a worker with a different owner")
    if item["State"]["Running"]:
        try:
            subprocess.run(["docker", "stop", "--timeout", "30", item["Id"]], capture_output=True, timeout=45)
        except subprocess.TimeoutExpired:
            pass
        latest = subprocess.run(["docker", "inspect", item["Id"]], text=True, capture_output=True, timeout=15)
        if latest.returncode == 0:
            current = json.loads(latest.stdout)[0]
            if current["Config"].get("Labels", {}).get("nvfp4-reef.owner") != owner:
                raise RuntimeError("worker ownership changed during cleanup")
            if current["State"]["Running"]:
                subprocess.run(["docker", "kill", current["Id"]], check=True, capture_output=True, timeout=30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("command-file", "registry", "job-file", "checkpoint-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.command_file)
    state_root = Path(config["state_root"]).resolve()
    for path in (args.registry, args.job_file, args.checkpoint_dir):
        if not path.is_absolute() or not path.resolve().is_relative_to(state_root):
            raise ValueError("worker job paths must remain within the declared state root")
    args.registry.mkdir(parents=True, exist_ok=True)
    name, cidfile, argv = worker_command(config, args.job_file, args.checkpoint_dir, args.registry, secrets.token_hex(6))
    record = args.registry / (name + ".json")
    evidence = {"owner": config["owner"], "container_name": name, "job_file": str(args.job_file),
                "checkpoint_dir": str(args.checkpoint_dir), "command": argv, "started_unix": time.time()}
    write_json(record, evidence)
    process = None
    result = 1
    def interrupted(signum, frame):
        raise InterruptedError(f"worker launcher received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        process = subprocess.Popen(argv)
        result = process.wait()
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        identifier = cidfile.read_text().strip() if cidfile.exists() else name
        stop_owned(identifier, config["owner"])
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=15)
        evidence.update(exit_code=result, container_id=identifier if cidfile.exists() else None, ended_unix=time.time())
        write_json(record, evidence)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
