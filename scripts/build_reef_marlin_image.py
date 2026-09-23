#!/usr/bin/env python3
"""Build and verify a separate, offline Marlin-order research image without GPUs."""

import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import tempfile


INSTALLER_PATH = "/opt/nvfp4-marlin-order/reef_marlin_patch.py"
IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
INPUTS = ("nvfp4_lora/reef_marlin_patch.py", "nvfp4_lora/vendor/marlin_order.patch",
          "nvfp4_lora/vendor/LICENSE", "nvfp4_lora/vendor/MARLIN_ORDER.md",
          "docker/reef-marlin-order.Dockerfile")


def command(argv, **kwargs):
    kwargs.setdefault("timeout", 60)
    try:
        return subprocess.run(list(map(str, argv)), text=True, capture_output=True, check=True, **kwargs).stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        for captured in (exc.stdout, exc.stderr):
            if captured:
                print(captured.decode(errors="replace") if isinstance(captured, bytes) else captured, file=sys.stderr)
        raise


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def patch_api(repo):
    path = Path(repo) / "nvfp4_lora/reef_marlin_patch.py"
    spec = importlib.util.spec_from_file_location("reef_marlin_patch_provenance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runtime_read_command(prefix, api):
    code = """import hashlib, importlib.metadata, json, pathlib, sys
source, manifest, installer = map(pathlib.Path, sys.argv[1:])
raw = manifest.read_bytes()
print(json.dumps({
    'manifest': json.loads(raw),
    'manifest_sha256': hashlib.sha256(raw).hexdigest(),
    'installed_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'installer_sha256': hashlib.sha256(installer.read_bytes()).hexdigest(),
    'vllm_version': importlib.metadata.version('vllm'),
}))
"""
    return [*prefix, "-c", code, str(api.MARLIN_SOURCE_PATH), str(api.MARLIN_MANIFEST_PATH), INSTALLER_PATH]


def validate_runtime_provenance(evidence, api, installer_sha256):
    expected = {"manifest": api.MARLIN_PATCH_MANIFEST,
                "manifest_sha256": api.MARLIN_PATCH_MANIFEST_SHA256,
                "installed_source_sha256": api.MARLIN_PATCH_MANIFEST["patched_sha256"],
                "installer_sha256": installer_sha256,
                "vllm_version": api.MARLIN_PATCH_MANIFEST["vllm_version"]}
    if evidence != expected:
        raise ValueError("installed Marlin patch, manifest, installer or vLLM version differs from reviewed source")
    return evidence


def validate_build_report(report, image_id, repo, api=None):
    api = patch_api(repo) if api is None else api
    if (not IMAGE_ID.fullmatch(image_id) or image_id == api.MARLIN_BASE_IMAGE_ID
            or report.get("schema_version") != 1 or report.get("image_id") != image_id
            or report.get("base_image_id") != api.MARLIN_BASE_IMAGE_ID
            or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", str(report.get("source_revision")))):
        raise ValueError("research image does not match its pinned build provenance")
    expected_inputs = {path: file_hash(Path(repo) / path) for path in INPUTS}
    if report.get("input_sha256") != expected_inputs:
        raise ValueError("research image build inputs differ from the current reviewed source")
    validate_runtime_provenance(report.get("runtime_provenance"), api,
                                expected_inputs["nvfp4_lora/reef_marlin_patch.py"])
    return report


def verify_actor_patch(inspected, owner, report, repo, *, run=command):
    if inspected["Config"].get("Labels", {}).get("nvfp4-reef.owner") != owner:
        raise ValueError("refusing to inspect an unowned actor")
    api = patch_api(repo)
    validate_build_report(report, inspected["Image"], repo, api)
    evidence = json.loads(run(runtime_read_command(
        ["docker", "exec", inspected["Id"], "python3"], api), timeout=30))
    validate_runtime_provenance(evidence, api, report["input_sha256"]["nvfp4_lora/reef_marlin_patch.py"])
    return {"base_image_id": api.MARLIN_BASE_IMAGE_ID, "marlin_patch": evidence["manifest"],
            "marlin_patch_manifest_sha256": evidence["manifest_sha256"],
            "marlin_installed_source_sha256": evidence["installed_source_sha256"]}


def build(args):
    repo = args.repo.resolve()
    api = patch_api(repo)
    if args.output.exists():
        raise ValueError("build evidence output must be fresh")
    if not re.fullmatch(r"reef-marlin-order:[a-zA-Z0-9_][a-zA-Z0-9_.-]{0,127}", args.tag):
        raise ValueError("use a fresh local reef-marlin-order:<tag> research image name")
    revision = command(["git", "-C", repo, "rev-parse", "HEAD"])
    if command(["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"]):
        raise ValueError("tracked build source must be committed")
    command(["git", "-C", repo, "ls-files", "--error-unmatch", *INPUTS, "scripts/build_reef_marlin_image.py"])
    if command(["docker", "image", "inspect", "--format", "{{.Id}}", api.MARLIN_BASE_IMAGE_ID]) != api.MARLIN_BASE_IMAGE_ID:
        raise ValueError("local immutable base image does not match the reviewed pin")
    if command(["docker", "image", "ls", "-q", args.tag]):
        raise ValueError("refusing to replace an existing image tag")
    input_hashes = {path: file_hash(repo / path) for path in INPUTS}
    alias = "reef-marlin-base:" + api.MARLIN_BASE_IMAGE_ID[7:19] + "-" + secrets.token_hex(8)
    if command(["docker", "image", "ls", "-q", alias]):
        raise ValueError("temporary base alias already exists")
    tagged = False
    try:
        command(["docker", "tag", api.MARLIN_BASE_IMAGE_ID, alias])
        tagged = True
        if command(["docker", "image", "inspect", "--format", "{{.Id}}", alias]) != api.MARLIN_BASE_IMAGE_ID:
            raise ValueError("temporary base alias does not resolve to the pinned image")
        with tempfile.TemporaryDirectory(prefix="reef-marlin-build-") as temporary:
            context = Path(temporary)
            (context / "vendor").mkdir()
            for source in INPUTS:
                target = context / ("Dockerfile" if source.endswith(".Dockerfile") else
                                    "vendor/" + Path(source).name if "/vendor/" in source else Path(source).name)
                shutil.copyfile(repo / source, target)
                if file_hash(target) != input_hashes[source]:
                    raise ValueError("build input changed during context creation")
            iid = context / "image-id"
            argv = ["docker", "build", "--pull=false", "--network=none", "--iidfile", iid,
                    "--build-arg", "BASE_IMAGE=" + alias,
                    "--label", "nvfp4-reef.base-image-id=" + api.MARLIN_BASE_IMAGE_ID,
                    "--label", "nvfp4-reef.marlin-manifest-sha256=" + api.MARLIN_PATCH_MANIFEST_SHA256,
                    "--label", "nvfp4-reef.source-revision=" + revision, context]
            command(argv, timeout=args.timeout)
            image_id = iid.read_text().strip()
        if not IMAGE_ID.fullmatch(image_id) or image_id == api.MARLIN_BASE_IMAGE_ID:
            raise ValueError("build did not produce a distinct content-addressed image")
        if command(["docker", "image", "inspect", "--format", "{{.Id}}", alias]) != api.MARLIN_BASE_IMAGE_ID:
            raise ValueError("temporary base alias changed during the build")
        evidence = json.loads(command(runtime_read_command(
            ["docker", "run", "--rm", "--network", "none", "--runtime", "runc", "--read-only",
             "-e", "NVIDIA_VISIBLE_DEVICES=void", "-e", "PYTHONDONTWRITEBYTECODE=1",
             "--entrypoint", "python3", image_id], api)))
        report = {"schema_version": 1, "image_id": image_id, "base_image_id": api.MARLIN_BASE_IMAGE_ID,
                  "image_tag": args.tag, "source_revision": revision, "input_sha256": input_hashes,
                  "runtime_provenance": evidence}
        validate_build_report(report, image_id, repo, api)
        if command(["docker", "image", "ls", "-q", args.tag]):
            raise ValueError("requested image tag appeared during the build; refusing to replace it")
        command(["docker", "tag", image_id, args.tag])
        if command(["docker", "image", "inspect", "--format", "{{.Id}}", args.tag]) != image_id:
            raise ValueError("derived image tag does not resolve to the verified build")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(report, stream, sort_keys=True, indent=2)
            stream.write("\n")
        return report
    finally:
        if tagged:
            if command(["docker", "image", "inspect", "--format", "{{.Id}}", alias]) != api.MARLIN_BASE_IMAGE_ID:
                raise RuntimeError("base alias changed; refusing to remove an unowned tag")
            command(["docker", "image", "rm", alias])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    with Path("/tmp/reef-marlin-image-build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = build(args)
    print(json.dumps({"image_id": report["image_id"], "image_tag": report["image_tag"],
                      "provenance": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
