"""Shared, model-agnostic training primitives.

This module deliberately does NOT import or run any model-family-specific code
(no Mamba patches, no cached-prefix-suffix, no Nemotron-H assumptions). It only
exposes the model-agnostic primitives that a NVFP4 LoRA trainer might reuse.

Packaging note: this module must import cleanly from an installed wheel, i.e.
without the repo-only top-level ``train/`` package on ``sys.path``. It therefore
imports nothing from ``train.*`` at module scope. The shipped trainers each carry
their own save/load/label-mask implementations (train/train_super_nvfp4.py and
scripts/train_nvfp4_lora.py), so the historical ``save_adapter`` /
``load_adapter_weights`` / ``mask_prompt_labels`` re-export shim that forwarded
into ``train.train_super_nvfp4`` had no callers and has been removed; keeping it
would have made a packaged ``import nvfp4_lora.training_utils`` an ImportError
land-mine the moment anything referenced those names.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Tuple


# --------------------------------------------------------------------------------------
# Optional per-step callback seam (model-agnostic, zero project-specific knowledge)
#
# The trainer can call an out-of-tree ``on_step(step, model, tokenizer, output_dir)`` hook
# on a fixed cadence (e.g. a drift monitor that projects the live model's activations onto
# concept vectors). The seam stays deliberately tiny and generic: it loads a caller-supplied
# module and invokes it defensively, so a broken or missing callback can never abort a run.
# --------------------------------------------------------------------------------------
def load_step_callback(spec: Optional[str]) -> Optional[Callable[..., Any]]:
    """Resolve an ``on_step`` callable from a dotted module path or a ``.py`` file path.

    ``spec`` is either a dotted import path (``pkg.mod``) or a filesystem path to a ``.py``
    file; a falsy ``spec`` (the default, no callback configured) returns ``None``. The
    resolved module must expose a callable ``on_step(step, model, tokenizer, output_dir)``.

    Raises on a genuine misconfiguration (unimportable module, missing/empty file, no
    ``on_step``) so a typo fails fast before the expensive model load; the trainer wraps
    this call so even that degrades to a logged warning rather than killing the run.
    """
    if not spec:
        return None
    import importlib
    import importlib.util

    path = Path(spec)
    if spec.endswith(".py") or path.is_file():
        if not path.is_file():
            raise FileNotFoundError(f"callback file not found: {spec!r}")
        mod_name = f"_nvfp4_step_callback_{path.stem}"
        mod_spec = importlib.util.spec_from_file_location(mod_name, str(path))
        if mod_spec is None or mod_spec.loader is None:
            raise ImportError(f"could not build an import spec for callback file {spec!r}")
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(spec)

    on_step = getattr(module, "on_step", None)
    if not callable(on_step):
        raise AttributeError(
            f"callback module {spec!r} has no callable on_step(step, model, tokenizer, "
            f"output_dir)"
        )
    return on_step


def resolve_callback_cadence(callback_every: int, checkpoint_every: int) -> int:
    """Cadence (in update steps) at which the step callback fires.

    ``callback_every <= 0`` (the default) means 'follow the checkpoint cadence', so the
    monitor reads exactly the weight-states that get saved. A positive value overrides it.
    Returns ``0`` when neither is positive, i.e. the callback never fires.
    """
    n = int(callback_every)
    return n if n > 0 else max(0, int(checkpoint_every))


def run_step_callback(
    callback: Optional[Callable[..., Any]],
    log_fn: Callable[..., Any],
    *,
    step: int,
    model: Any,
    tokenizer: Any,
    output_dir: Any,
) -> None:
    """Invoke ``callback`` for this step with ALL exceptions caught and logged.

    A no-op when ``callback`` is ``None``. Any exception the callback raises is reported
    via ``log_fn`` (the trainer's structured logger) and swallowed, so an unattended run is
    never killed by a monitoring hiccup. The callback is always called by keyword, matching
    ``on_step(step, model, tokenizer, output_dir)``.
    """
    if callback is None:
        return
    try:
        callback(step=step, model=model, tokenizer=tokenizer, output_dir=str(output_dir))
    except Exception as e:  # noqa: BLE001 - a callback fault must never abort training
        log_fn("step_callback_error", step=step, error=repr(e))


# --------------------------------------------------------------------------------------
# Phase-tagged watchdog labels (model-agnostic)
# --------------------------------------------------------------------------------------
_CURRENT_PHASE = "init"


def set_current_phase(label: str) -> None:
    global _CURRENT_PHASE
    _CURRENT_PHASE = label


def get_current_phase() -> str:
    return _CURRENT_PHASE


# --------------------------------------------------------------------------------------
# Optimizer dispatch with `lr` as explicit parameter (per Sonnet pass-1 note)
# --------------------------------------------------------------------------------------
def build_optimizer(
    trainable: Iterable,
    optimizer_name: str,
    lr: float,
) -> Tuple["torch.optim.Optimizer", str]:
    """Build a torch optimizer over `trainable` params with explicit LR.

    Supported: "adamw", "adamw8bit" (via torchao), "adafactor" (via transformers).
    """
    import torch
    if optimizer_name == "adamw":
        return torch.optim.AdamW(trainable, lr=lr), "AdamW"
    if optimizer_name == "adamw8bit":
        from torchao.optim import AdamW8bit
        return AdamW8bit(trainable, lr=lr), "torchao AdamW8bit"
    if optimizer_name == "adafactor":
        from transformers.optimization import Adafactor
        return (
            Adafactor(
                trainable,
                lr=lr,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False,
                weight_decay=0.0,
            ),
            "Transformers Adafactor",
        )
    raise SystemExit(f"unknown optimizer: {optimizer_name}")


__all__ = [
    "_CURRENT_PHASE",
    "set_current_phase",
    "get_current_phase",
    "build_optimizer",
    "load_step_callback",
    "resolve_callback_cadence",
    "run_step_callback",
]
