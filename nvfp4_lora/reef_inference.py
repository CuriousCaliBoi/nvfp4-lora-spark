"""Versioned REEF capture for one exclusively owned vLLM 0.27.1 actor."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from threading import RLock
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from reef.artifact.artifact import Artifact, LiveWeightArtifactRef
from reef.runtime.interfaces import (
    ActivatedModel,
    InferenceHandler,
    InferenceRuntime,
    InferenceStream,
    ModelCandidate,
    StaleCandidate,
    TrainingRuntimeError,
    UpstreamStatusError,
)

from .reef_checkpoint import canonical_json, materialize_checkpoint, validate_checkpoint, write_json
from .reef_marlin_patch import validate_actor_patch_contract


_SAMPLING = {
    "temperature": 1.2,
    "top_p": 1.0,
    "top_k": -1,
    "min_p": 0.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "logit_bias": {},
}


def _token_ids(value: Any, *, minimum: int = 1) -> list[int]:
    if not isinstance(value, list) or len(value) < minimum or any(
        not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in value
    ):
        raise ValueError("native token IDs must be a nonempty list of nonnegative integers")
    return list(value)


def _finite(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("native log probabilities must be finite numbers")
    return float(value)


def _maximum_delta(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("fixed-sequence probe lengths disagree")
    return max(abs(a - b) for a, b in zip(first, second))


def _option(command: list[str], name: str) -> str | None:
    values = []
    for index, token in enumerate(command):
        if token == name:
            if index + 1 == len(command):
                raise ValueError(f"actor command is missing {name}'s value")
            values.append(command[index + 1])
        elif token.startswith(name + "="):
            values.append(token.split("=", 1)[1])
    if len(values) > 1:
        raise ValueError(f"actor command repeats {name}")
    return values[0] if values else None


@dataclass(frozen=True)
class AdapterBinding:
    checkpoint_id: str
    checkpoint_path: Path
    name: str
    reload_name: str
    actor_path: str
    adapter_sha256: str
    adapter_config_sha256: str
    evidence: Mapping[str, Any]


class VllmInferenceRuntime(InferenceRuntime):
    """Pin requests to immutable, numerically verified adapter selectors.

    The supervisor exclusively owns the actor and attests its actual launch
    command. vLLM's public API cannot itself attest processed-logprob mode or
    free native LoRA allocations when an API registry entry is unloaded.
    """

    def __init__(
        self, *, base_url: str, actor_instance_id: str, base_model: str,
        model_revision: str, checkpoint_root: str | Path, base_checkpoint: str | Path,
        state_dir: str | Path, probe_token_ids: Sequence[int],
        actor_checkpoint_root: str | Path | None = None,
        inference_timeout_s: float = 300, adapter_capacity: int = 16,
        lora_rank: int = 8, lora_alpha: float = 16,
        on_checkpoint_activation: Callable[[Artifact], None] | None = None,
    ) -> None:
        super().__init__(base_url=base_url, inference_timeout_s=inference_timeout_s)
        self._lock = RLock()
        self._model = base_model
        self._revision = model_revision
        self._actor = actor_instance_id
        self._rank, self._alpha = lora_rank, lora_alpha
        self._root = Path(checkpoint_root).resolve()
        self._base_checkpoint = Path(base_checkpoint).resolve()
        self._actor_root = Path(actor_checkpoint_root or self._root)
        self._state_dir = Path(state_dir).resolve()
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._state_file = self._state_dir / "serving-state.json"
        self._contract_path = self._state_dir / "actor-contract.json"
        self._probe_ids = _token_ids(list(probe_token_ids), minimum=2)
        if not isinstance(adapter_capacity, int) or isinstance(adapter_capacity, bool) or adapter_capacity < 2:
            raise ValueError("adapter_capacity must include primary and reload aliases")
        self._capacity = adapter_capacity
        self._incarnation = uuid.uuid4().hex
        self._loads: dict[str, AdapterBinding] = {}
        self._verified: dict[str, AdapterBinding] = {}
        self._releases: dict[str, tuple[AdapterBinding, str]] = {}
        self._active: AdapterBinding | None = None
        self._version: str | None = None
        self._active_release: str | None = None
        self._pending: dict[str, Any] | None = None
        self._restoring_checkpoint_id: str | None = None
        self._on_checkpoint_activation = on_checkpoint_activation
        self._fenced = True
        self._closed = False
        self._handler = VllmInferenceHandler(self)
        self.pause_admission()
        self._prior_state = json.loads(self._state_file.read_text()) if self._state_file.exists() else None
        if self._prior_state is not None and self._prior_state.get("schema_version") != 1:
            raise ValueError("unsupported serving-state schema")
        self._contract_hash = self._validate_actor_contract()
        self._base_manifest = validate_checkpoint(
            self._base_checkpoint, expected_base_model=self._model, expected_model_revision=self._revision,
            expected_lora_rank=self._rank, expected_lora_alpha=self._alpha,
        )
        if self._base_manifest["parent_checkpoint_id"] is not None:
            raise ValueError("base_checkpoint must identify the persisted bootstrap")
        version = self._http("GET", "/version")
        if not isinstance(version, dict) or version.get("version") != "0.27.1":
            raise ValueError("the actor must run pinned vLLM 0.27.1")
        binding = self._ensure_verified(self._base_checkpoint)
        self._active, self._version = binding, self._new_load(binding)
        # A fresh verified ID makes REEF recover old live references from their
        # durable checkpoint instead of trusting a previous process incarnation.
        self._current_runtime_load_id = self._version
        self._persist()

    @property
    def inference_handler(self) -> InferenceHandler:
        return self._handler

    @property
    def model_path(self) -> str:
        return self._model

    @property
    def pending_training_job_id(self) -> str | None:
        return None if self._pending is None else self._pending["training_job_id"]

    def serving_runtime_load_id(self) -> str | None:
        with self._lock:
            return self._version

    def serving_adapter_name(self) -> None:
        # The handler resolves each frozen artifact; a global lora_path would
        # incorrectly redirect older requests after publication.
        return None

    def _http(self, method: str, path: str, payload: Mapping[str, Any] | None = None) -> Any:
        body = None if payload is None else canonical_json(payload)
        request = Request(self.base_url + path, data=body, method=method, headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=self.inference_timeout_s) as response:
                raw = response.read()
        except HTTPError as exc:
            raise UpstreamStatusError(exc.read(4096).decode(errors="replace"), status=exc.code) from exc
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw.decode()

    def _validate_actor_contract(self) -> str:
        raw = self._contract_path.read_bytes()
        contract = json.loads(raw)
        validate_actor_patch_contract(contract)
        expected = {
            "schema_version": 1, "actor_instance_id": self._actor,
            "base_model": self._model, "model_revision": self._revision,
            "vllm_version": "0.27.1", "logprobs_mode": "processed_logprobs",
            "generation_config": "vllm", "speculative_decoding": False,
            "exclusive_adapter_control": True,
            "max_num_seqs": 1, "kv_cache_dtype": "bfloat16",
            "attention_backend": "TRITON_ATTN", "mamba_cache_mode": "none",
            "cublas_workspace_config": ":4096:8",
        }
        if (any(contract.get(key) != value for key, value in expected.items())
                or contract.get("async_scheduling") is not False
                or type(contract.get("max_num_seqs")) is not int):
            raise ValueError("actor startup attestation does not match the controlled learning actor")
        command = contract.get("command")
        if not isinstance(command, list) or not command or any(not isinstance(arg, str) for arg in command):
            raise ValueError("actor attestation must contain inspected command argv")
        if not isinstance(contract.get("container_id"), str) or not contract["container_id"]:
            raise ValueError("actor attestation lacks the inspected container ID")
        if (
            _option(command, "--logprobs-mode") != "processed_logprobs"
            or _option(command, "--generation-config") != "vllm"
            or _option(command, "--max-num-seqs") != "1"
            or _option(command, "--kv-cache-dtype") != "bfloat16"
            or _option(command, "--attention-backend") != "TRITON_ATTN"
            or _option(command, "--mamba-cache-mode") != "none"
            or "--enable-lora" not in command
            or "--no-async-scheduling" not in command
            or any(arg.split("=", 1)[0] == "--async-scheduling" for arg in command)
            or _option(command, "--speculative-config") is not None
            or int(_option(command, "--max-cpu-loras") or "0") < self._capacity
        ):
            raise ValueError("inspected actor command violates the sampling or adapter-capacity contract")
        return hashlib.sha256(raw).hexdigest()

    def _assert_actor(self) -> None:
        if self._closed or hashlib.sha256(self._contract_path.read_bytes()).hexdigest() != self._contract_hash:
            self._fence()
            raise TrainingRuntimeError("actor identity changed; supervised reconciliation is required")

    def _manifest(self, path: Path) -> dict[str, Any]:
        value = validate_checkpoint(
            path, expected_base_model=self._model, expected_model_revision=self._revision,
            expected_lora_rank=self._rank, expected_lora_alpha=self._alpha,
            expected_config_sha256=self._base_manifest["config_sha256"],
        )
        if any(value[key] != self._base_manifest[key] for key in ("model_config_sha256", "frozen_tensor_sha256")):
            raise ValueError("checkpoint frozen base/config differs from the initial checkpoint")
        return value

    def _materialize(self, artifact: Artifact) -> Path:
        path = materialize_checkpoint(artifact, self._root, base_checkpoint=self._base_checkpoint)
        self._manifest(path)
        return path

    def _models(self) -> dict[str, Mapping[str, Any]]:
        self._assert_actor()
        response = self._http("GET", "/v1/models")
        if not isinstance(response, dict) or not isinstance(response.get("data"), list):
            raise ValueError("vLLM omitted its model registry")
        entries = response["data"]
        if any(not isinstance(entry, dict) or not isinstance(entry.get("id"), str) for entry in entries):
            raise ValueError("vLLM returned a malformed model registry")
        result = {entry["id"]: entry for entry in entries}
        if len(result) != len(entries) or self._model not in result:
            raise ValueError("vLLM model registry disagrees with the served base")
        return result

    def _registry_binding(self, name: str, path: str) -> bool:
        entry = self._models().get(name)
        if entry is None:
            return False
        if entry.get("root") != path or entry.get("parent") != self._model:
            raise ValueError("immutable adapter name is already bound to different content/path")
        return True

    def _load_alias(self, name: str, path: str) -> None:
        if self._registry_binding(name, path):
            return
        if len(self._models()) - 1 >= self._capacity:
            raise RuntimeError("bounded actor adapter capacity exhausted; HTTP unload cannot prove native reclamation")
        self._http("POST", "/v1/load_lora_adapter", {"lora_name": name, "lora_path": path, "load_inplace": False})
        if not self._registry_binding(name, path):
            raise ValueError("vLLM acknowledged loading without the required adapter association")

    def _probe(self, model: str, *, evidence_path: Path | None = None) -> list[float]:
        request = {
            "model": model, "prompt": self._probe_ids, "max_tokens": 1,
            "temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0,
            "frequency_penalty": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0,
            "logit_bias": {}, "n": 1, "seed": 43, "prompt_logprobs": 0,
            "logprobs": 0, "return_token_ids": True,
        }
        response = self._http("POST", "/v1/completions", request)
        if evidence_path is not None:
            write_json(evidence_path, {"request": request, "response": response})
        choices = response.get("choices", []) if isinstance(response, dict) else []
        if not isinstance(response, dict) or response.get("model") != model or len(choices) != 1 or choices[0].get("index") != 0:
            raise ValueError("fixed-sequence response selected a different native model/output")
        choice = choices[0]
        if choice.get("prompt_token_ids") != self._probe_ids:
            raise ValueError("fixed-sequence probe changed native prompt IDs")
        rows = choice.get("prompt_logprobs")
        if not isinstance(rows, list) or len(rows) != len(self._probe_ids):
            raise ValueError("vLLM omitted aligned native prompt log probabilities")
        result = []
        for token, row in zip(self._probe_ids[1:], rows[1:]):
            if not isinstance(row, dict) or str(token) not in row or not isinstance(row[str(token)], dict):
                raise ValueError("fixed-sequence selected token log probability is missing")
            result.append(_finite(row[str(token)].get("logprob")))
        return result

    def _ensure_verified(self, path: Path) -> AdapterBinding:
        path = self._materialize(Artifact.local(path))
        manifest = self._manifest(path)
        identity = manifest["checkpoint_id"]
        existing = self._verified.get(identity)
        if existing is not None:
            if (manifest["adapter_sha256"] != existing.adapter_sha256
                    or manifest["files"]["adapter/adapter_config.json"] != existing.adapter_config_sha256):
                raise ValueError("immutable checkpoint adapter bytes changed after verification")
            if not self._registry_binding(existing.name, existing.actor_path):
                raise ValueError("verified adapter vanished from the owned actor")
            return existing
        parent = manifest["parent_checkpoint_id"]
        reference = self._model if parent is None else self._ensure_verified(self._root / parent).name
        fingerprint = hashlib.sha256(canonical_json({
            "actor": self._actor, "checkpoint": identity, "base_model": self._model,
            "revision": self._revision, "weights": manifest["adapter_sha256"],
            "config": manifest["files"]["adapter/adapter_config.json"],
        })).hexdigest()
        name, reload_name = f"reef-nvfp4-{fingerprint}-primary", f"reef-nvfp4-{fingerprint}-reload"
        actor_path = str(self._actor_root / identity / "adapter")
        attempt_id = uuid.uuid4().hex
        attempt_dir = self._state_dir / "native-probes" / identity / attempt_id
        attempt_dir.mkdir(parents=True)
        evidence_path = attempt_dir / "evidence.json"
        evidence = {
            "schema_version": 1, "attempt_id": attempt_id, "status": "collecting",
            "actor_instance_id": self._actor, "actor_contract_sha256": self._contract_hash,
            "checkpoint_id": identity, "adapter_name": name, "reload_name": reload_name,
            "adapter_sha256": manifest["adapter_sha256"],
            "adapter_config_sha256": manifest["files"]["adapter/adapter_config.json"],
            "probe_token_ids": self._probe_ids, "reference_name": reference,
            "bootstrap": parent is None, "native_verified": False,
            "evidence_path": str(evidence_path), "raw_probe_paths": {}, "completed_probes": [],
        }

        def record_probe(key: str, selector: str) -> list[float]:
            raw_path = attempt_dir / f"{len(evidence['completed_probes']):02d}-{key}.json"
            evidence["stage"] = key
            evidence["raw_probe_paths"][key] = str(raw_path)
            write_json(evidence_path, evidence)
            scores = self._probe(selector, evidence_path=raw_path)
            evidence[key] = scores
            evidence["completed_probes"].append(key)
            write_json(evidence_path, evidence)
            return scores

        try:
            first = record_probe("reference_logprobs", reference)
            repeated = record_probe("reference_repeat_logprobs", reference)
            evidence["stage"] = "load_primary"
            write_json(evidence_path, evidence)
            self._load_alias(name, actor_path)
            scores = record_probe("adapter_logprobs", name)
            adapter_repeat = record_probe("adapter_repeat_logprobs", name)
            evidence["stage"] = "load_reload"
            write_json(evidence_path, evidence)
            self._load_alias(reload_name, actor_path)
            reloaded = record_probe("reload_logprobs", reload_name)
            reload_repeat = record_probe("reload_repeat_logprobs", reload_name)
            reference_after = record_probe("reference_after_load_logprobs", reference)
            reference_after_repeat = record_probe("reference_after_load_repeat_logprobs", reference)
            adapter_after = record_probe("adapter_after_reference_logprobs", name)
            noise = _maximum_delta(first, repeated)
            effect = _maximum_delta(first, scores)
            reload_delta = _maximum_delta(scores, reloaded)
            evidence.update(
                status="measured", stage="validation", reference_repeat_max_delta=noise,
                adapter_effect_max_delta=effect, reload_max_delta=reload_delta,
                adapter_repeat_max_delta=_maximum_delta(scores, adapter_repeat),
                reload_repeat_max_delta=_maximum_delta(reloaded, reload_repeat),
                reference_after_load_max_delta=_maximum_delta(first, reference_after),
                reference_after_load_repeat_max_delta=_maximum_delta(reference_after, reference_after_repeat),
                adapter_after_reference_max_delta=_maximum_delta(scores, adapter_after),
                reference_observed_max_delta=max(_maximum_delta(a, b) for a, b in combinations(
                    (first, repeated, reference_after, reference_after_repeat), 2)),
                adapter_observed_max_delta=max(_maximum_delta(a, b) for a, b in combinations(
                    (scores, adapter_repeat, reloaded, reload_repeat, adapter_after), 2)),
            )
            stability_metrics = (
                "reference_repeat_max_delta", "adapter_repeat_max_delta", "reload_repeat_max_delta",
                "reference_after_load_max_delta", "reference_after_load_repeat_max_delta",
                "adapter_after_reference_max_delta", "reload_max_delta",
                "reference_observed_max_delta", "adapter_observed_max_delta",
            )
            observed_noise = max(evidence[key] for key in stability_metrics)
            evidence["observed_null_max_delta"] = observed_noise
            # Keep measured paths and vectors even when a parity assertion
            # rejects the adapter; failing early would hide the controls that
            # distinguish a path difference from unstable or ineffective loads.
            write_json(evidence_path, evidence)
            for key in stability_metrics:
                if evidence[key] > 1e-5:
                    raise ValueError(f"fixed-sequence native stability failed: {key} exceeds 1e-5")
            if parent is None:
                if effect > noise + 1e-5:
                    raise ValueError("bootstrap zero-delta adapter does not reproduce the base")
            elif effect <= observed_noise:
                raise ValueError("candidate has no native adapter effect above observed null/repeat/alias noise")
        except BaseException as exc:
            evidence.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            write_json(evidence_path, evidence)
            raise
        evidence.update(status="passed", native_verified=True)
        write_json(evidence_path, evidence)
        binding = AdapterBinding(identity, path, name, reload_name, actor_path,
                                 manifest["adapter_sha256"], manifest["files"]["adapter/adapter_config.json"], evidence)
        self._verified[identity] = binding
        self._persist()
        return binding

    def _new_load(self, binding: AdapterBinding) -> str:
        version = f"{self._incarnation}:{len(self._loads)}"
        self._loads[version] = binding
        return version

    def _persist(self) -> None:
        write_json(self._state_file, {
            "schema_version": 1, "actor_instance_id": self._actor,
            "actor_contract_sha256": self._contract_hash,
            "active_checkpoint_id": None if self._active is None else self._active.checkpoint_id,
            "active_release": self._active_release, "runtime_load_id": self._version,
            "published_runtime_load_id": self._current_runtime_load_id,
            "pending": self._pending, "fenced": self._fenced,
            "bindings": {key: dict(binding.evidence) for key, binding in self._verified.items()},
        })

    def _fence(self) -> None:
        with self._lock:
            self._fenced = True
            self.pause_admission()
            self._persist()

    def resume_admission(self) -> None:
        with self._lock:
            # REEF's candidate-abort scheduler also calls resume. A failed
            # receiver must not accidentally reopen because of that cleanup.
            if self._fenced or self._closed or self._pending is not None or self.current_runtime_load_id() != self._version:
                return
            super().resume_admission()

    def mark_published(self) -> None:
        with self._lock:
            if not self._fenced and self._pending is None:
                super().mark_published()
                self._persist()

    def snapshot(self, artifact: Artifact) -> tuple[AdapterBinding, str]:
        with self._lock:
            self._assert_actor()
            if self._fenced:
                raise TrainingRuntimeError("inference is fenced until durable-head reconciliation")
            if isinstance(artifact.ref, LiveWeightArtifactRef):
                version = artifact.ref.runtime_load_id
                if version not in self._loads:
                    raise ValueError("frozen artifact belongs to an unavailable serving incarnation")
                return self._loads[version], version
            if artifact.ref.release_id in self._releases:
                return self._releases[artifact.ref.release_id]
            path = self._materialize(artifact)
            if self._active is None or path.name != self._active.checkpoint_id:
                raise ValueError("unbound durable artifact cannot silently use the current adapter")
            selected = (self._active, self._version)
            self._releases[artifact.ref.release_id] = selected
            return selected

    def activate_candidate(self, candidate: ModelCandidate) -> ActivatedModel:
        self.pause_admission(wait=True, timeout=self.inference_timeout_s)
        with self._lock:
            if self._pending is not None:
                if (self._pending["training_job_id"] != candidate.training_job_id
                        or self._pending["candidate_id"] != candidate.candidate_id
                        or self._manifest(Path(candidate.checkpoint_path))["checkpoint_id"] != candidate.candidate_id):
                    raise TrainingRuntimeError("another candidate awaits durable publication")
                return ActivatedModel(candidate.candidate_id, self._version)
            if candidate.current_runtime_load_id not in (None, self._version):
                raise StaleCandidate()
            try:
                manifest = self._manifest(Path(candidate.checkpoint_path))
                if manifest["training_job_id"] != candidate.training_job_id or manifest["checkpoint_id"] != candidate.candidate_id:
                    raise ValueError("candidate identity does not match its checkpoint")
                if self._active is None or manifest["parent_checkpoint_id"] != self._active.checkpoint_id:
                    raise StaleCandidate()
                binding = self._ensure_verified(Path(candidate.checkpoint_path))
                self._active, self._version = binding, self._new_load(binding)
                self._pending = {
                    "training_job_id": candidate.training_job_id, "candidate_id": candidate.candidate_id,
                    "checkpoint_path": str(binding.checkpoint_path),
                }
                self._fenced = False
                self._persist()
                return ActivatedModel(candidate.candidate_id, self._version)
            except BaseException:
                self._fence()
                raise

    def activate_checkpoint(self, artifact: Artifact) -> str:
        self.pause_admission(wait=True, timeout=self.inference_timeout_s)
        with self._lock:
            try:
                path = self._materialize(artifact)
                binding = self._ensure_verified(path)
                initial = self._active_release is None
                pending_candidate = self._pending is not None and binding.checkpoint_id == self._pending["candidate_id"]
                published = self._loads.get(self.current_runtime_load_id())
                recovering_head = (
                    not pending_candidate and self._restoring_checkpoint_id is None
                    and (initial or published is not None and published.checkpoint_id == binding.checkpoint_id)
                )
                reconcile = not pending_candidate and (
                    recovering_head or self._fenced or self._restoring_checkpoint_id is not None or self._pending is not None
                )
                if self._restoring_checkpoint_id is not None and binding.checkpoint_id != self._restoring_checkpoint_id:
                    raise ValueError("rollback activation differs from its verified restore target")
                if reconcile and self._on_checkpoint_activation is not None:
                    # REEF can restore the learner before a rollback commit
                    # fails. Its recovered durable head must repair both sides
                    # before any request becomes admissible after restart.
                    self._on_checkpoint_activation(artifact)
                if (pending_candidate or self._active_release == artifact.ref.release_id
                        and self._active is not None and self._active.checkpoint_id == binding.checkpoint_id):
                    version = self._version
                else:
                    self._pending = None
                    version = self._new_load(binding)
                self._active, self._version = binding, version
                self._active_release = artifact.ref.release_id
                self._releases[artifact.ref.release_id] = (binding, version)
                self._fenced = False
                self._restoring_checkpoint_id = None
                self._persist()
                if recovering_head:
                    self.mark_published()
                    self.resume_admission()
                return version
            except BaseException:
                self._fence()
                raise

    def restore_checkpoint(self, artifact: Artifact) -> str:
        self.pause_admission(wait=True, timeout=self.inference_timeout_s)
        with self._lock:
            try:
                binding = self._ensure_verified(self._materialize(artifact))
                self._restoring_checkpoint_id = binding.checkpoint_id
                return self._new_load(binding)
            except BaseException:
                self._fence()
                raise

    def resume_weight_update(self, training_job_id: str) -> ActivatedModel:
        with self._lock:
            pending = self._pending or (self._prior_state or {}).get("pending")
            if pending is None or pending["training_job_id"] != training_job_id:
                raise TrainingRuntimeError("unknown durable weight-update identity")
            candidate = ModelCandidate(
                candidate_id=pending["candidate_id"], training_job_id=training_job_id,
                checkpoint_path=pending["checkpoint_path"], current_runtime_load_id=self._version,
            )
        return self.activate_candidate(candidate)

    def acknowledge_publication(self, training_job_id: str) -> None:
        with self._lock:
            if self._fenced or self._active is None:
                raise TrainingRuntimeError("cannot acknowledge an unverified serving head")
            manifest = self._manifest(self._active.checkpoint_path)
            if manifest["training_job_id"] != training_job_id:
                raise TrainingRuntimeError("publication acknowledgement names another checkpoint")
            if self._pending is not None and self._pending["training_job_id"] != training_job_id:
                raise TrainingRuntimeError("publication acknowledgement names another pending job")
            self._pending = None
            self._persist()

    def verification(self, checkpoint_path: str | Path) -> dict[str, Any]:
        with self._lock:
            try:
                return dict(self._ensure_verified(Path(checkpoint_path)).evidence)
            except BaseException:
                self._fence()
                raise

    def evaluate_adapter(self, checkpoint_path: str | Path, requests: Sequence[dict]) -> list[dict]:
        with self._lock:
            try:
                binding = self._ensure_verified(Path(checkpoint_path))
            except BaseException:
                self._fence()
                raise
        return [self._chat(binding, request, training=False)[0] for request in requests]

    def _chat(self, binding: AdapterBinding, payload: Mapping[str, Any], *, training: bool) -> tuple[dict, dict]:
        request, sampling = chat_request(payload, base_model=self._model, training=training)
        self._check_binding(binding)
        request["model"] = binding.name
        response = self._http("POST", "/v1/chat/completions", request)
        capture = native_capture(response, expected_model=binding.name)
        self._check_binding(binding)
        capture.update({
            "sampling": sampling, "adapter_sha256": binding.adapter_sha256,
            "adapter_config_sha256": binding.adapter_config_sha256,
            "base_model": self._model, "model_revision": self._revision,
        })
        return response, capture

    def _check_binding(self, binding: AdapterBinding) -> None:
        try:
            manifest = self._manifest(binding.checkpoint_path)
            if (manifest["adapter_sha256"] != binding.adapter_sha256
                    or manifest["files"]["adapter/adapter_config.json"] != binding.adapter_config_sha256):
                raise TrainingRuntimeError("request's immutable adapter bytes changed")
            if not self._registry_binding(binding.name, binding.actor_path):
                raise TrainingRuntimeError("request's pinned adapter disappeared")
        except BaseException:
            self._fence()
            raise

    def shutdown(self) -> None:
        self.pause_admission()
        with self._lock:
            self._closed = True


class VllmInferenceHandler(InferenceHandler):
    def __init__(self, runtime: VllmInferenceRuntime) -> None:
        self._runtime = runtime

    async def inference(self, artifact: Artifact, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if path != "/v1/chat/completions":
            raise UpstreamStatusError("learning capture supports /v1/chat/completions only", status=400)
        return await asyncio.to_thread(self._sample, artifact, payload)

    def _sample(self, artifact: Artifact, payload: dict[str, Any]) -> dict[str, Any]:
        binding, version = self._runtime.snapshot(artifact)
        response, capture = self._runtime._chat(binding, payload, training=True)
        response["training"] = {**capture, "runtime_load_id": version}
        return response

    async def inference_stream(self, artifact: Artifact, path: str, payload: dict[str, Any]) -> InferenceStream:
        raise UpstreamStatusError("streaming is unsupported for exact learning capture; use normal inference_proxy", status=400)


def chat_request(payload: Mapping[str, Any], *, base_model: str, training: bool = True) -> tuple[dict, dict]:
    supported = set(_SAMPLING) | {
        "model", "messages", "max_tokens", "max_completion_tokens", "seed", "n", "stream",
        "chat_template_kwargs", "return_meta_info", "logprobs", "top_logprobs",
    }
    unknown = set(payload) - supported
    if unknown or payload.get("stream", False) is not False or type(payload.get("n", 1)) is not int or payload.get("n", 1) != 1:
        raise UpstreamStatusError(f"unsupported learning chat options: {sorted(unknown)}; buffered n=1 required", status=400)
    if payload.get("model", base_model) != base_model:
        raise UpstreamStatusError("client model/adapter overrides are forbidden", status=400)
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages or any(
        not isinstance(message, dict) or set(message) != {"role", "content"}
        or message["role"] not in {"system", "user", "assistant"}
        or not isinstance(message["content"], str) for message in messages
    ):
        raise UpstreamStatusError("learning requires nonempty text system/user/assistant messages", status=400)
    template = payload.get("chat_template_kwargs", {"enable_thinking": False})
    if template != {"enable_thinking": False}:
        raise UpstreamStatusError("learning requires enable_thinking=False without parser/template overrides", status=400)
    params = dict(_SAMPLING)
    if not training:
        params["temperature"] = 0.0
    for key, default in params.items():
        value = payload.get(key, default)
        if value != default or isinstance(value, bool):
            raise UpstreamStatusError(f"unsupported {key}: this route requires {default!r}", status=400)
    seed = payload.get("seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
        raise UpstreamStatusError("seed must be a nonnegative integer or null", status=400)
    maximum = payload.get("max_completion_tokens", payload.get("max_tokens", 256))
    if (not isinstance(maximum, int) or isinstance(maximum, bool) or not 1 <= maximum <= 256
            or ("max_tokens" in payload and payload["max_tokens"] != maximum)):
        raise UpstreamStatusError("learning/evaluation max_tokens must be 1..256 and unambiguous", status=400)
    request = {**params, "messages": messages, "max_tokens": maximum, "seed": seed, "n": 1,
               "stream": False, "chat_template_kwargs": {"enable_thinking": False},
               "logprobs": True, "top_logprobs": 0, "return_token_ids": True,
               "return_tokens_as_token_ids": True, "include_reasoning": True}
    sampling = {**params, "seed": seed, "logprobs_mode": "processed_logprobs"}
    return request, sampling


def native_capture(response: Any, *, expected_model: str) -> dict[str, Any]:
    if not isinstance(response, dict) or response.get("model") != expected_model:
        raise ValueError("native response model does not match the pinned adapter")
    choices = response.get("choices")
    if (not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict)
            or type(choices[0].get("index")) is not int or choices[0]["index"] != 0):
        raise ValueError("native response must contain exactly output index0")
    choice = choices[0]
    prompt = _token_ids(response.get("prompt_token_ids"))
    completion = _token_ids(choice.get("token_ids"))
    probabilities = choice.get("logprobs")
    entries = probabilities.get("content") if isinstance(probabilities, dict) else None
    if not isinstance(entries, list) or len(entries) != len(completion):
        raise ValueError("native selected log probabilities do not align with completion IDs")
    logprobs = []
    for token, entry in zip(completion, entries):
        if not isinstance(entry, dict) or entry.get("token") != f"token_id:{token}":
            raise ValueError("native log probability token ID differs from the sampled token")
        logprobs.append(_finite(entry.get("logprob")))
    finish = choice.get("finish_reason")
    message = choice.get("message")
    if (finish not in {"stop", "length"} or not isinstance(message, dict)
            or message.get("role") != "assistant" or not isinstance(message.get("content"), str)):
        raise ValueError("unsupported native finish reason or nontext response")
    if message.get("tool_calls") or message.get("reasoning") or message.get("reasoning_content"):
        raise ValueError("parser-transformed tool/reasoning responses are unsupported for learning")
    usage = response.get("usage", {})
    if (not isinstance(usage, dict) or type(usage.get("prompt_tokens")) is not int
            or type(usage.get("completion_tokens")) is not int
            or usage["prompt_tokens"] != len(prompt) or usage["completion_tokens"] != len(completion)):
        raise ValueError("native usage counts disagree with captured token arrays")
    return {
        "tokens": prompt + completion, "prompt_token_ids": prompt, "completion_token_ids": completion,
        "loss_mask": [1] * len(completion), "rollout_log_probs": logprobs,
        "prompt_length": len(prompt), "response_length": len(completion),
        "output_index": 0, "finish_reason": finish,
    }
