"""CPU routing and source checks do not establish native kernel repeatability."""

import ast
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from nvfp4_lora import reef_marlin_patch as patch


FIXTURE = Path(__file__).parent / "fixtures/vllm_marlin_moe_0_27_1.py.txt"


@pytest.fixture(scope="module")
def patched_source():
    return patch.patched_marlin_source(FIXTURE.read_bytes())


def source_function(source, name, namespace):
    node = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name)
    code = "from __future__ import annotations\n" + ast.get_source_segment(source.decode(), node)
    exec(compile(code, "<reviewed-marlin-source>", "exec"), namespace)
    return namespace[name]


@pytest.fixture(scope="module")
def canonicalize(patched_source):
    return source_function(patched_source, "_canonicalize_marlin_moe_token_order", {"torch": torch})


def test_helper_fingerprint_covers_only_the_actual_installed_function(patched_source):
    source = patched_source.decode()
    node = next(node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == "_canonicalize_marlin_moe_token_order")
    helper = "\n".join(source.splitlines()[node.lineno - 1:node.end_lineno]) + "\n"
    assert hashlib.sha256(helper.encode()).hexdigest() == patch.MARLIN_PATCH_MANIFEST["helper_sha256"]
    assert helper.rstrip().endswith("return sorted_token_ids.index_select(0, order)")


def test_exact_patch_output_and_standalone_cli_without_torch(tmp_path, patched_source):
    source, manifest = tmp_path / "marlin_moe.py", tmp_path / "manifest.json"
    source.write_bytes(FIXTURE.read_bytes())
    source.chmod(0o644)
    command = [sys.executable, "-S", str(Path(patch.__file__)), "apply", "--source", str(source), "--manifest", str(manifest)]
    result = json.loads(subprocess.check_output(command))
    assert source.read_bytes() == patched_source
    assert source.stat().st_mode & 0o777 == 0o644
    assert hashlib.sha256(patched_source).hexdigest() == patch.MARLIN_PATCH_MANIFEST["patched_sha256"]
    assert result == patch.verify_marlin_patch(source, manifest)
    assert result["manifest"] == patch.MARLIN_PATCH_MANIFEST
    assert result["manifest_sha256"] == patch.MARLIN_PATCH_MANIFEST_SHA256
    command[3] = "verify"
    assert json.loads(subprocess.check_output(command)) == result
    with pytest.raises(ValueError, match="duplicate or unreviewed"):
        patch.apply_marlin_patch(source, manifest)


def test_source_mismatch_does_not_write(tmp_path):
    source, manifest = tmp_path / "marlin_moe.py", tmp_path / "manifest.json"
    original = FIXTURE.read_bytes() + b"\n"
    source.write_bytes(original)
    with pytest.raises(ValueError, match="original Marlin source SHA256"):
        patch.apply_marlin_patch(source, manifest)
    assert source.read_bytes() == original
    assert not manifest.exists()


@pytest.mark.parametrize("target", ["manifest", "source"])
def test_verification_rejects_changed_installed_evidence(tmp_path, target):
    source, manifest = tmp_path / "marlin_moe.py", tmp_path / "manifest.json"
    source.write_bytes(FIXTURE.read_bytes())
    patch.apply_marlin_patch(source, manifest)
    path = source if target == "source" else manifest
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="installed Marlin"):
        patch.verify_marlin_patch(source, manifest)


def test_modified_vendor_asset_is_rejected(tmp_path, monkeypatch):
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / "marlin_order.patch").write_bytes(b"unreviewed replacement")
    monkeypatch.setattr(patch, "__file__", str(tmp_path / "reef_marlin_patch.py"))
    with pytest.raises(ValueError, match="vendored Marlin patch"):
        patch.patched_marlin_source(FIXTURE.read_bytes())


def test_whole_expert_regions_and_poisoned_inactive_tail(canonicalize):
    experts = torch.tensor([2, 2, 7, 7, -123], dtype=torch.int32)
    count = torch.tensor([12], dtype=torch.int32)
    layouts = [
        [8, 3, 9, 1, 7, 4, 6, 9, 5, 9, 0, 2, -2147483648, -100, 123456, 17, -77, 42, 999999, -9],
        [9, 6, 4, 7, 1, 9, 3, 8, 2, 0, 9, 5, 314159, -2147483648, -1, 88, 7, -300, 111111, 22],
    ]
    expected = [1, 3, 4, 6, 7, 8, 9, 9, 0, 2, 5, 9]
    for layout in layouts:
        values = torch.tensor(layout, dtype=torch.int32)
        canonical = canonicalize(values, experts, count, 4, 9)
        assert canonical[:12].tolist() == expected
        assert sorted(canonical[12:].tolist()) == sorted(layout[12:])
        assert values.tolist() == layout
        assert experts.tolist() == [2, 2, 7, 7, -123]
        assert count.tolist() == [12]


@pytest.mark.parametrize("values,experts,active,valid", [
    ([], [], 0, 0),
    ([-2147483648, 2147483647, -1], [7], 0, 0),
    ([9] * 8 + [-2147483648], [2, 2, 2], 8, 9),
])
def test_empty_and_all_padding_active_regions(canonicalize, values, experts, active, valid):
    result = canonicalize(torch.tensor(values, dtype=torch.int32), torch.tensor(experts, dtype=torch.int32),
                          torch.tensor([active], dtype=torch.int32), 4, valid)
    assert result[:active].tolist() == values[:active]
    assert sorted(result.tolist()) == sorted(values)


@pytest.mark.parametrize("seed", range(20))
def test_randomized_region_semantics_and_partial_inactive_blocks(canonicalize, seed):
    rng = random.Random(seed)
    for _ in range(25):
        block = rng.choice([4, 8, 16, 32, 48, 64])
        regions, experts, next_id = [], [], 0
        for expert in [2, 7, 2, 8][:rng.randint(0, 4)]:
            blocks = rng.randint(1, 6)
            n = rng.randint(0, blocks * block)
            regions.append(list(range(next_id, next_id + n)) + [None] * (blocks * block - n))
            next_id += n
            experts.extend([expert] * blocks)
        sentinel = next_id
        expected, layouts = [], [[], []]
        for region in regions:
            region = [sentinel if value is None else value for value in region]
            expected.extend(sorted(region))
            for layout in layouts:
                shuffled = region[:]
                rng.shuffle(shuffled)
                layout.extend(shuffled)
        active = len(expected)
        tail_length = rng.randint(0, 3 * block)
        tail_experts = [experts[-1] if experts else 7] * math.ceil(tail_length / block)
        expert_tensor = torch.tensor(experts + tail_experts, dtype=torch.int32)
        count = torch.tensor([active], dtype=torch.int32)
        for layout in layouts:
            layout.extend(rng.choice([-2147483648, 2147483647, -1, 0, sentinel]) for _ in range(tail_length))
            values = torch.tensor(layout, dtype=torch.int32)
            result = canonicalize(values, expert_tensor, count, block, sentinel)
            assert result[:active].tolist() == expected
            assert sorted(result.tolist()) == sorted(layout)
            assert values.tolist() == layout
            assert expert_tensor.tolist() == experts + tail_experts
            assert count.item() == active


def test_actual_bounded_actor_geometry(canonicalize):
    rng = random.Random(43)
    tokens, topk, expert_count, block = 1024, 6, 128, 64
    routed = [[] for _ in range(expert_count)]
    for token_id in range(tokens * topk):
        routed[rng.randrange(expert_count)].append(token_id)
    layout, expected, experts = [], [], []
    for expert, ids in enumerate(routed):
        padded = math.ceil(len(ids) / block) * block
        region = ids + [tokens * topk] * (padded - len(ids))
        expected.extend(region)
        rng.shuffle(region)
        layout.extend(region)
        experts.extend([expert] * (padded // block))
    active = len(layout)
    allocated = tokens * topk + expert_count * (block - 1)
    layout.extend([-2147483648] * (allocated - active))
    experts.extend([127] * (math.ceil(allocated / block) - len(experts)))
    result = canonicalize(torch.tensor(layout, dtype=torch.int32), torch.tensor(experts, dtype=torch.int32),
                          torch.tensor([active], dtype=torch.int32), block, tokens * topk)
    assert result[:active].tolist() == expected


@pytest.mark.parametrize("tokens", [1, 3])
def test_real_patched_callsite_preserves_single_token_fast_path(patched_source, canonicalize, tokens):
    calls, captured = [], []
    sorted_ids = torch.tensor(([0] if tokens == 1 else [2, 1, 0]) + [tokens] * (8 - tokens), dtype=torch.int32)
    expert_ids, count = torch.tensor([0], dtype=torch.int32), torch.tensor([8], dtype=torch.int32)

    def ordered(*args):
        calls.append(args)
        return canonicalize(*args)

    def gemms(**kwargs):
        captured.append(kwargs)
        return torch.zeros((tokens, 16), dtype=torch.bfloat16)

    kinds = ["uint4", "uint8b128", "uint4b8", "float8_e4m3fn", "float4_e2m1f"]
    namespace = {
        "torch": torch, "math": math, "MoEActivation": SimpleNamespace(SILU="silu"),
        "apply_moe_activation": None, "scalar_types": SimpleNamespace(**{kind: kind for kind in kinds}),
        "ScalarType": SimpleNamespace(from_id=lambda _: "float4_e2m1f"),
        "moe_align_block_size": lambda *args, **kwargs: (sorted_ids, expert_ids, count),
        "_canonicalize_marlin_moe_token_order": ordered, "_fused_marlin_moe": gemms,
    }
    fused = source_function(patched_source, "fused_marlin_moe", namespace)
    fused(torch.zeros((tokens, 16), dtype=torch.bfloat16), torch.zeros((2, 1, 8)),
          torch.zeros((2, 1, 32)), None, None, None, None, torch.ones((tokens, 1)),
          torch.zeros((tokens, 1), dtype=torch.int32), 0)
    assert len(calls) == (tokens > 1)
    assert captured[0]["sorted_token_ids"].tolist() == list(range(tokens)) + [tokens] * (8 - tokens)
    assert captured[0]["expert_ids"] is expert_ids
    assert captured[0]["num_tokens_post_padded"] is count
    if tokens == 1:
        assert captured[0]["sorted_token_ids"] is sorted_ids
