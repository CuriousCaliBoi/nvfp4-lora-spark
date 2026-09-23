"""CPU REEF candidate lifecycle over durable quantized learner checkpoints."""

from __future__ import annotations

import fcntl
import json
from pathlib import Path
import re
import subprocess
from threading import RLock

from reef.runtime.interfaces import ModelCandidate, PreparedTrainingStep, StaleCandidate, TrainingRuntime
from reef.train.algos.registry import resolve_preparer

from .reef_checkpoint import (
    MANIFEST, checkpoint_identity, content_hash, materialize_checkpoint, validate_checkpoint, write_json,
)
from .reef_data import advantages, normalized_config, validate_rows
from .reef_recipe import batch_rows


class QuantizedTrainingRuntime(TrainingRuntime):
    def __init__(self, *, state_root, model_dir, base_model, model_revision, worker_command,
                 training_config, scenario="nvfp4-gsm8k", lora_rank=8, lora_alpha=16):
        self.state_root = Path(state_root).expanduser().resolve()
        self.checkpoint_root = self.state_root / "checkpoints"
        self._jobs = self.state_root / "jobs"
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        self._jobs.mkdir(exist_ok=True)
        if not isinstance(worker_command, (tuple, list)) or not worker_command or any(
            not isinstance(item, str) or not item for item in worker_command
        ):
            raise ValueError("worker_command must be a nonempty argv prefix")
        self._command = list(worker_command)
        self._model_dir, self._base_model, self._revision = str(model_dir), base_model, model_revision
        self._rank, self._alpha, self._scenario = lora_rank, lora_alpha, scenario
        self._config = normalized_config(training_config)
        self.config_sha256 = content_hash(self._config)
        self._lock = RLock()
        self._closed = False
        self._owner = (self.state_root / ".learner.lock").open("a")
        try:
            fcntl.flock(self._owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._owner.close()
            raise ValueError("quantized learner state already has an owner") from None
        try:
            job = self._job("bootstrap")
            identity = checkpoint_identity(job)
            self.base_checkpoint = self.checkpoint_root / identity
            if not self.base_checkpoint.exists():
                self._run_worker(job, self.base_checkpoint, f"bootstrap-{identity}")
            self._validated(self.base_checkpoint)
            marker = self.state_root / "incumbent.json"
            if marker.exists():
                saved = json.loads(marker.read_text())
                if saved.get("schema_version") != 1:
                    raise ValueError("invalid incumbent state")
                self.incumbent_checkpoint = self._path(saved["checkpoint_id"])
                if str(self.incumbent_checkpoint) != saved.get("checkpoint_path"):
                    raise ValueError("incumbent path disagrees with identity")
                self._validated(self.incumbent_checkpoint)
            else:
                self._set_incumbent(self.base_checkpoint)
        except BaseException:
            self._owner.close()
            raise

    def _path(self, identity):
        if not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{64}", identity):
            raise ValueError("invalid candidate identity")
        return self.checkpoint_root / identity

    def _validated(self, path):
        value = validate_checkpoint(path, expected_base_model=self._base_model,
                                    expected_model_revision=self._revision, expected_lora_rank=self._rank,
                                    expected_lora_alpha=self._alpha, expected_config_sha256=self.config_sha256)
        if value["scenario"] != self._scenario:
            raise ValueError("checkpoint belongs to another scenario")
        return value

    def _job(self, kind):
        return dict(schema_version=1, kind=kind, scenario=self._scenario, scenario_step=0,
                    checkpoint_root=str(self.checkpoint_root), model_dir=self._model_dir,
                    base_model=self._base_model, model_revision=self._revision,
                    lora_rank=self._rank, lora_alpha=self._alpha, config=dict(self._config),
                    config_sha256=self.config_sha256, parent_checkpoint=None,
                    parent_checkpoint_id=None, training_job_id=None, batch_sha256=None,
                    batch_id=None, rows=[])

    def _set_incumbent(self, path):
        manifest = self._validated(path)
        write_json(self.state_root / "incumbent.json", {
            "schema_version": 1, "checkpoint_id": manifest["checkpoint_id"], "checkpoint_path": str(path),
        })
        self.incumbent_checkpoint = Path(path)

    def _status(self, identity, status, *, parent_checkpoint_id):
        write_json(self._jobs / f"{identity}.status.json", dict(
            schema_version=1, training_job_id=identity, checkpoint_id=identity,
            parent_checkpoint_id=parent_checkpoint_id, status=status,
        ))

    def _read_status(self, identity):
        path = self._jobs / f"{identity}.status.json"
        if not path.exists():
            raise ValueError("unknown training job")
        value = json.loads(path.read_text())
        if value.get("schema_version") != 1 or value.get("training_job_id") != identity:
            raise ValueError("invalid training job journal")
        return value

    def _run_worker(self, job, checkpoint, identity):
        path = self._jobs / f"{identity}.json"
        if path.exists() and json.loads(path.read_text()) != job:
            raise ValueError("immutable training job payload changed")
        write_json(path, job)
        with (self._jobs / f"{identity}.log").open("a") as log:
            subprocess.run([*self._command, "--job-file", str(path), "--checkpoint-dir", str(checkpoint)],
                           check=True, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
        value = self._validated(checkpoint)
        if value["checkpoint_id"] != checkpoint.name:
            raise ValueError("worker returned wrong checkpoint")
        return value

    def prepare_training_step(self, batch, step_preparer, algorithm_state, scenario_step,
                              *, serving_runtime_load_id=None):
        with self._lock:
            rows = batch_rows(batch)
            incumbent = self._validated(self.incumbent_checkpoint)
            stale = False
            for row in rows:
                if serving_runtime_load_id is None or row["runtime_load_id"] != serving_runtime_load_id:
                    stale = True
                    continue
                if (row["adapter_sha256"] != incumbent["adapter_sha256"] or
                        row["adapter_config_sha256"] != incumbent["files"]["adapter/adapter_config.json"] or
                        row["base_model"] != self._base_model or row["model_revision"] != self._revision):
                    raise ValueError("receipt does not belong to committed quantized policy")
                if len(row["prompt_ids"]) + len(row["completion_ids"]) > self._config["max_model_len"]:
                    raise ValueError("native trajectory exceeds configured context")
            if stale:
                # REEF catches StaleCandidate around candidate execution, not preparation.
                return PreparedTrainingStep("train", dict(algorithm_state), {"stale_native_rows": 1}, {"stale": True})
            signal = resolve_preparer(step_preparer)(batch, algorithm_state)
            if signal.loss_family != "nvfp4_grpo" or signal.advantages != advantages(rows):
                raise ValueError("preparer does not match quantized GRPO contract")
            if signal.action == "skip":
                return PreparedTrainingStep("skip", signal.next_algorithm_state, signal.metrics)
            job = self._job("train")
            job.update(scenario_step=scenario_step, parent_checkpoint=str(self.incumbent_checkpoint),
                       parent_checkpoint_id=incumbent["checkpoint_id"], batch_id=batch.batch_id,
                       batch_sha256=content_hash({"batch_id": batch.batch_id, "rows": rows}), rows=rows)
            job["training_job_id"] = checkpoint_identity(job)
            return PreparedTrainingStep("train", signal.next_algorithm_state, signal.metrics,
                                        {**job, "source_runtime_load_id": serving_runtime_load_id})

    def train_candidate(self, payload):
        if payload.get("stale") is True:
            raise StaleCandidate({"stale_native_rows": 1})
        with self._lock:
            job = {key: value for key, value in dict(payload).items() if key != "source_runtime_load_id"}
            expected = self._job("train")
            for key in ("schema_version", "kind", "scenario", "checkpoint_root", "model_dir", "base_model",
                        "model_revision", "lora_rank", "lora_alpha", "config", "config_sha256"):
                if job.get(key) != expected[key]:
                    raise ValueError(f"training job configuration mismatch: {key}")
            rows = validate_rows(job["rows"])
            if rows != job["rows"] or job["batch_sha256"] != content_hash({"batch_id": job["batch_id"], "rows": rows}):
                raise ValueError("training batch hash mismatch")
            if payload.get("source_runtime_load_id") != rows[0]["runtime_load_id"]:
                raise ValueError("training source runtime identity mismatch")
            identity = checkpoint_identity(job)
            if job["training_job_id"] != identity or job["config"] != self._config:
                raise ValueError("training job identity/config mismatch")
            incumbent = self._validated(self.incumbent_checkpoint)
            destination = self._path(identity)
            if incumbent["checkpoint_id"] == identity:
                if self._read_status(identity)["status"] != "committed":
                    raise ValueError("incumbent candidate lacks committed job evidence")
                return self._candidate(job, payload, destination)
            if job["parent_checkpoint_id"] != incumbent["checkpoint_id"] or job["parent_checkpoint"] != str(self.incumbent_checkpoint):
                raise StaleCandidate({"stale_parent": 1})
            if not any(advantages(rows)):
                raise ValueError("zero-signal batch cannot execute an optimizer update")
            status_path = self._jobs / f"{identity}.status.json"
            if status_path.exists() and self._read_status(identity)["status"] in ("rejected", "committed"):
                raise ValueError("terminal training job cannot be retrained")
            self._status(identity, "prepared", parent_checkpoint_id=incumbent["checkpoint_id"])
            if destination.exists():
                result = self._validated(destination)
            else:
                result = self._run_worker(job, destination, identity)
            if (result["parent_checkpoint_id"] != incumbent["checkpoint_id"] or
                    result["batch_sha256"] != job["batch_sha256"] or
                    result["optimizer_step"] != incumbent["optimizer_step"] + 1):
                raise ValueError("candidate does not continue committed parent")
            self._status(identity, "complete", parent_checkpoint_id=incumbent["checkpoint_id"])
            return self._candidate(job, payload, destination)

    def _candidate(self, job, payload, destination):
        metrics = json.loads((destination / "metrics.json").read_text())
        return ModelCandidate(candidate_id=job["training_job_id"], training_job_id=job["training_job_id"],
                              checkpoint_path=str(destination),
                              current_runtime_load_id=payload.get("source_runtime_load_id"),
                              training_metrics=metrics,
                              metadata={"scenario_step": job["scenario_step"], "batch_id": job["batch_id"],
                                        "parent_checkpoint_id": job["parent_checkpoint_id"]})

    def reject_candidate(self, candidate, decision):
        with self._lock:
            identity = candidate.training_job_id
            status = self._read_status(identity)
            if status["status"] == "committed":
                raise ValueError("cannot reject a committed training job")
            self._status(identity, "rejected", parent_checkpoint_id=status["parent_checkpoint_id"])

    def commit_candidate(self, training_job_id):
        with self._lock:
            destination = self._path(training_job_id)
            manifest = self._validated(destination)
            status = self._read_status(training_job_id)
            if self.incumbent_checkpoint == destination:
                self._status(training_job_id, "committed", parent_checkpoint_id=manifest["parent_checkpoint_id"])
                return
            if status["status"] not in ("prepared", "complete"):
                raise ValueError("cannot commit an unknown or terminal candidate")
            incumbent = self._validated(self.incumbent_checkpoint)
            if manifest["parent_checkpoint_id"] != incumbent["checkpoint_id"]:
                raise ValueError("committed candidate parent is no longer incumbent")
            self._set_incumbent(destination)
            self._status(training_job_id, "committed", parent_checkpoint_id=manifest["parent_checkpoint_id"])

    def restore_checkpoint(self, artifact):
        with self._lock:
            path = materialize_checkpoint(artifact, self.checkpoint_root, base_checkpoint=self.base_checkpoint)
            self._set_incumbent(path)

    def shutdown(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                self._owner.close()
