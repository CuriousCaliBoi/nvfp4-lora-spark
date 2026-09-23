"""CPU-only external REEF deployment for the bounded local NVFP4 experiment."""

from pathlib import Path
from urllib.parse import urlsplit

from reef.runtime.deployment import RuntimeFactory
from reef.train.deployment import InProcessTrainingDeployment


REEF_REVISION = "b637cbe42393e91e8ee75af2179f844da586c3dd"
TRAINING_CONFIG = {
    "seed": 42, "temperature": 1.2, "learning_rate": 1e-4,
    "weight_decay": 0.0, "betas": [0.9, 0.999], "eps": 1e-8,
    "clip_epsilon": 0.2, "tis_min": 0.1, "tis_max": 10.0,
    "max_grad_norm": 1.0, "microbatch_size": 1, "updates_per_job": 1,
    "objective": "grpo_sample_mean_v1", "dropout": 0,
}


def absolute_path(value, name):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    return Path(value).resolve()


def runtime_settings(config):
    required = {
        "state_root", "model_dir", "model_revision", "worker_command",
        "actor_url", "actor_instance_id", "probe_token_ids", "reef_revision",
    }
    optional = {
        "type", "inference_timeout_s", "train_timeout_s", "max_staleness",
        "scenario", "lora_rank", "lora_alpha", "adapter_capacity",
    }
    if required - config.keys() or config.keys() - required - optional:
        raise ValueError("runtime options must match the documented NVFP4 deployment schema")
    if config["reef_revision"] != REEF_REVISION:
        raise ValueError("unsupported REEF source revision")
    if config.get("max_staleness", 0) != 0:
        raise ValueError("this experiment requires max_staleness=0")
    if config.get("lora_rank", 8) != 8 or config.get("lora_alpha", 16) != 16:
        raise ValueError("this experiment requires rank8/alpha16")
    if config.get("adapter_capacity", 16) < 16:
        raise ValueError("the bounded campaign needs at least 16 adapter aliases")
    command = config["worker_command"]
    if not isinstance(command, list) or not command or any(not isinstance(x, str) or not x for x in command):
        raise ValueError("worker_command must be a nonempty argv list")
    parsed = urlsplit(config["actor_url"])
    if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("actor_url must use local HTTP")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("actor_url cannot contain credentials or query parameters")
    probe = config["probe_token_ids"]
    if not isinstance(probe, list) or len(probe) < 2 or any(type(x) is not int or x < 0 for x in probe):
        raise ValueError("probe_token_ids must contain at least two native token IDs")
    settings = dict(config)
    settings["state_root"] = absolute_path(config["state_root"], "state_root")
    settings["model_dir"] = absolute_path(config["model_dir"], "model_dir")
    return settings


class Nvfp4Deployment(InProcessTrainingDeployment):
    runtime_type = "nvfp4_lora.reef_deployment:runtime_factory"
    requires_local_model = False


class Nvfp4RuntimeFactory(RuntimeFactory):
    kind = Nvfp4Deployment.runtime_type

    def parse_config(self, config, environ):
        runtime_settings(config)
        return dict(config)

    def __call__(self, config, model_path, recipe_config, environ):
        settings = runtime_settings(config)
        from nvfp4_lora.reef_training import QuantizedTrainingRuntime
        from nvfp4_lora.reef_inference import VllmInferenceRuntime

        training = QuantizedTrainingRuntime(
            state_root=settings["state_root"], model_dir=settings["model_dir"],
            base_model=model_path, model_revision=settings["model_revision"],
            worker_command=settings["worker_command"], training_config=dict(TRAINING_CONFIG),
            scenario=settings.get("scenario", "nvfp4-gsm8k"), lora_rank=8, lora_alpha=16,
        )
        try:
            inference = VllmInferenceRuntime(
                base_url=settings["actor_url"], actor_instance_id=settings["actor_instance_id"],
                base_model=model_path, model_revision=settings["model_revision"],
                checkpoint_root=training.checkpoint_root, base_checkpoint=training.base_checkpoint,
                state_dir=settings["state_root"] / "serving",
                probe_token_ids=tuple(settings["probe_token_ids"]),
                inference_timeout_s=settings.get("inference_timeout_s", 300),
                adapter_capacity=settings.get("adapter_capacity", 16), lora_rank=8, lora_alpha=16,
                on_checkpoint_activation=training.restore_checkpoint,
            )
        except BaseException:
            training.shutdown()
            raise
        return training, inference


runtime_factory = Nvfp4RuntimeFactory()
