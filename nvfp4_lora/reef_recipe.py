"""REEF recipe consuming complete groups of recorded and verified rollouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from reef.core.batches import TrainingBatch, trajectories
from reef.core.reports import ReportValidationError, ScoredRolloutReport
from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
from reef.recipe.config_fields import config_field
from reef.train.algos.base import StepPreparer
from reef.train.algos.signals import StepScheduling, StepSignal
from reef.train.processors.reported import GroupDecision, ReportedFeedbackProcessor, SampleAssembly

from .reef_data import advantages, validate_capture, validate_rows


@dataclass(frozen=True)
class NVFP4RolloutReport(ScoredRolloutReport):
    cycle_id: str
    group_id: int
    rollout_id: int
    group_size: int
    batch_group_count: int
    dataset_id: str
    dataset_revision: str
    dataset_split: str
    dataset_row_id: int
    gold_answer: str

    def validate(self):
        if not self.cycle_id or self.score not in (0.0, 1.0):
            raise ReportValidationError("cycle identity and binary score required")
        if self.group_size < 2 or self.batch_group_count < 1:
            raise ReportValidationError("invalid rollout grid")
        if not 0 <= self.group_id < self.batch_group_count or not 0 <= self.rollout_id < self.group_size:
            raise ReportValidationError("rollout lies outside declared grid")
        if self.dataset_id != "openai/gsm8k" or self.dataset_split != "train" or self.dataset_row_id < 0:
            raise ReportValidationError("GSM8K train provenance required")


class NVFP4Processor(ReportedFeedbackProcessor):
    exclusive_sources = True
    ordered_groups = True

    def __init__(self, context):
        self.group_size = int(context.config.get("group_size", 8))
        self.batch_group_count = int(context.config.get("batch_group_count", 4))
        if self.group_size < 2 or self.batch_group_count < 1:
            raise ValueError("invalid rollout grid")
        self._assembly = SampleAssembly.from_config(context)
        super().__init__(context.with_config({**context.config, "batch_size": 1}))

    def make_sample(self, context):
        report = context.parsed_report
        if not isinstance(report, NVFP4RolloutReport) or len(context.inferences) != 1:
            raise ValueError("typed feedback for exactly one inference receipt required")
        if report.group_size != self.group_size or report.batch_group_count != self.batch_group_count:
            raise ValueError("reported grid differs from configured grid")
        inference = context.inferences[0]
        payload = inference.payload
        response = payload.get("response", {})
        capture = response.get("training")
        validate_capture(capture, payload.get("runtime_load_id"))
        choices = response.get("choices", [])
        if len(choices) != 1 or choices[0].get("index") != 0:
            raise ValueError("one native response choice required")
        text = choices[0].get("message", {}).get("content")
        if text is None and capture["finish_reason"] == "length":
            text = ""
        if not isinstance(text, str):
            raise ValueError("native response text required")
        if choices[0].get("finish_reason") != capture["finish_reason"]:
            raise ValueError("native finish reasons disagree")
        row = {key: value for key, value in asdict(report).items() if key != "score"}
        row.update(source_agent_record_id=inference.agent_record_id,
                   report_agent_record_id=context.report.agent_record_id,
                   response_text=text, reward=context.require_score(),
                   prompt_ids=capture["prompt_token_ids"], completion_ids=capture["completion_token_ids"],
                   behavior_logprobs=capture["rollout_log_probs"],
                   **{key: capture[key] for key in ("runtime_load_id", "finish_reason", "adapter_sha256",
                       "adapter_config_sha256", "base_model", "model_revision", "sampling")})
        validate_rows([row], complete=False)
        sample = self._assembly.build(context, row["reward"])
        if sample.training["runtime_load_id"] != row["runtime_load_id"]:
            raise ValueError("assembled runtime version disagrees with native capture")
        return sample.with_metadata(nvfp4=row)

    def grouping(self, context):
        report = context.parsed_report
        return report.cycle_id, (report.group_id, report.rollout_id)

    def decide_group(self, key, items):
        validate_rows([dict(item.metadata["nvfp4"]) for item in items], complete=False)
        return (GroupDecision.READY if len(items) == self.group_size * self.batch_group_count
                else GroupDecision.INCOMPLETE)

    def make_batch(self, items, batch_number):
        samples = sorted(items, key=lambda item: (item.metadata["nvfp4"]["group_id"], item.metadata["nvfp4"]["rollout_id"]))
        rows = validate_rows([dict(item.metadata["nvfp4"]) for item in samples])
        return TrainingBatch(f"{self.scenario}:nvfp4:{rows[0]['cycle_id']}", tuple(
            replace(item, group_id=str(row["group_id"])) for item, row in zip(samples, rows, strict=True)
        ))


def batch_rows(batch):
    items = trajectories(batch)
    rows = [dict(item.metadata["nvfp4"]) for item in items]
    validated = validate_rows(rows)
    if rows != validated:
        raise ValueError("batch rows must preserve contiguous group/rollout order")
    for item, row in zip(items, rows, strict=True):
        if (item.group_id != str(row["group_id"]) or item.training.get("tokens") != row["prompt_ids"] + row["completion_ids"]
                or item.training.get("loss_mask") != [1] * len(row["completion_ids"])
                or item.training.get("rollout_log_probs") != row["behavior_logprobs"]
                or item.training.get("runtime_load_id") != row["runtime_load_id"]
                or row["source_agent_record_id"] not in item.source_agent_record_ids):
            raise ValueError("normalized row disagrees with actual REEF trajectory")
    return rows


class GRPOStepPreparer(StepPreparer):
    name = "nvfp4_grpo"

    def __call__(self, batch, state):
        rows = batch_rows(batch)
        values = advantages(rows)
        signal = any(value != 0 for value in values)
        next_state = {**state, "batches": int(state.get("batches", 0)) + 1}
        return StepSignal("train" if signal else "skip", self.name, next_state,
                          {"samples": len(rows), "groups": rows[0]["batch_group_count"],
                           "reward_mean": sum(row["reward"] for row in rows) / len(rows),
                           "zero_signal": not signal}, values,
                          StepScheduling(batch_size="actual", epochs=1, shuffle=False))


grpo_step_preparer = GRPOStepPreparer()


@dataclass(frozen=True, kw_only=True)
class NVFP4Recipe(WeightTrainingRecipe):
    name: str = "nvfp4_grpo"
    group_size: int = config_field(8)
    batch_group_count: int = config_field(4)

    @property
    def report_type(self):
        return NVFP4RolloutReport

    @classmethod
    def training_spec(cls):
        return WeightTrainingSpec(step_preparer="nvfp4_lora.reef_recipe:grpo_step_preparer",
                                  loss_family="nvfp4_grpo", processor=NVFP4Processor)

    def __post_init__(self):
        super().__post_init__()
        if self.group_size < 2 or self.batch_group_count < 1:
            raise ValueError("invalid rollout grid")
