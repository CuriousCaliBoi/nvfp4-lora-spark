"""CPU tests for the generic per-step callback seam (nvfp4_lora.training_utils).

The seam lets the trainer call an out-of-tree ``on_step(step, model, tokenizer,
output_dir)`` hook on a fixed cadence. These tests cover the three load-bearing
properties in isolation from the trainer script (no GPU, no model, no torch):

  * loading a callback from a .py FILE path and from a DOTTED module path,
  * the invocation SIGNATURE (kwargs actually delivered to ``on_step``),
  * EXCEPTION ISOLATION (a raising callback is logged, never re-raised),
  * CADENCE arithmetic (``--callback-every 0`` follows ``--checkpoint-every``).
"""
from __future__ import annotations

import sys
import textwrap

import pytest

from nvfp4_lora.training_utils import (
    load_step_callback,
    resolve_callback_cadence,
    run_step_callback,
)

# A callback module that records every on_step call into a module-global list, so a test
# can read back exactly what the seam delivered (the returned callable is bound to this
# module's globals, so ``fn.__globals__["CALLS"]`` is the same list).
_DUMMY_OK = textwrap.dedent(
    """
    CALLS = []

    def on_step(step, model, tokenizer, output_dir):
        CALLS.append({"step": step, "model": model, "tokenizer": tokenizer,
                      "output_dir": output_dir})
    """
)

_DUMMY_RAISES = textwrap.dedent(
    """
    def on_step(step, model, tokenizer, output_dir):
        raise RuntimeError("boom in the monitor")
    """
)

_DUMMY_NO_HOOK = "X = 1\n"


def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body)
    return p


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------
def test_load_none_returns_none():
    assert load_step_callback(None) is None
    assert load_step_callback("") is None


def test_load_from_py_file(tmp_path):
    fn = load_step_callback(str(_write(tmp_path, "cb.py", _DUMMY_OK)))
    assert callable(fn)
    fn(step=1, model="m", tokenizer="t", output_dir="/out")
    assert fn.__globals__["CALLS"] == [
        {"step": 1, "model": "m", "tokenizer": "t", "output_dir": "/out"}
    ]


def test_load_from_dotted_module(tmp_path, monkeypatch):
    _write(tmp_path, "dotted_cb_mod.py", _DUMMY_OK)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "dotted_cb_mod", raising=False)
    fn = load_step_callback("dotted_cb_mod")
    assert callable(fn)


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_step_callback(str(tmp_path / "does_not_exist.py"))


def test_load_module_without_on_step_raises(tmp_path):
    with pytest.raises(AttributeError, match="on_step"):
        load_step_callback(str(_write(tmp_path, "no_hook.py", _DUMMY_NO_HOOK)))


# --------------------------------------------------------------------------------------
# invocation + exception isolation
# --------------------------------------------------------------------------------------
def test_run_delivers_keyword_signature(tmp_path):
    fn = load_step_callback(str(_write(tmp_path, "cb2.py", _DUMMY_OK)))
    # output_dir is stringified by the seam regardless of the caller's type.
    run_step_callback(fn, lambda *a, **k: None, step=50, model=object(),
                      tokenizer=object(), output_dir=tmp_path)
    call = fn.__globals__["CALLS"][0]
    assert call["step"] == 50
    assert call["output_dir"] == str(tmp_path)


def test_run_none_callback_is_noop():
    events = []
    run_step_callback(None, lambda ev, **kw: events.append((ev, kw)),
                      step=1, model=None, tokenizer=None, output_dir="/out")
    assert events == []


def test_run_swallows_and_logs_callback_exception(tmp_path):
    fn = load_step_callback(str(_write(tmp_path, "boom.py", _DUMMY_RAISES)))
    events = []
    # Must NOT raise, even though on_step raises.
    run_step_callback(fn, lambda ev, **kw: events.append((ev, kw)),
                      step=7, model=None, tokenizer=None, output_dir="/out")
    assert len(events) == 1
    ev, kw = events[0]
    assert ev == "step_callback_error"
    assert kw["step"] == 7
    assert "boom in the monitor" in kw["error"]


# --------------------------------------------------------------------------------------
# cadence arithmetic
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "callback_every,checkpoint_every,expected",
    [
        (0, 50, 50),    # default: follow the checkpoint cadence
        (10, 50, 10),   # explicit override wins
        (5, 0, 5),      # explicit fires even with checkpointing off
        (0, 0, 0),      # neither positive => never fires
        (-1, 50, 50),   # non-positive override treated as "follow checkpoint"
    ],
)
def test_resolve_cadence(callback_every, checkpoint_every, expected):
    assert resolve_callback_cadence(callback_every, checkpoint_every) == expected
