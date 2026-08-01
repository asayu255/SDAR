"""Gradient accumulation with one reduce per mini-batch instead of per micro-batch.

The expensive part of a step on this hardware is the FSDP collectives, so
reducing once per mini-batch rather than once per micro-batch halves them. What
has to hold: it is off unless asked for, it enters and leaves only at micro-batch
boundaries (FSDP1's no_sync asserts the module is IDLE, so it cannot wrap a
backward), the last micro-batch always runs with sync ON so the accumulated
gradient is actually reduced, and a call that died mid-context cannot leave the
flag off for the next one.
"""

import pytest

torch = pytest.importorskip("torch")

try:
    from verl.workers.actor import dp_actor as mod
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


class _FakeFSDP(torch.nn.Module):
    """Stands in for the root FSDP module, recording enter/exit."""

    def __init__(self):
        super().__init__()
        self.events = []
        self._sync_gradients = True

    def no_sync(self):
        outer = self

        class _Ctx:
            def __enter__(self):
                assert outer._sync_gradients, "entered no_sync twice without leaving"
                outer._sync_gradients = False
                outer.events.append("enter")
                return self

            def __exit__(self, *exc):
                outer._sync_gradients = True
                outer.events.append("exit")
                return False

        return _Ctx()


def _drive(n_micro, active, module):
    """The enter/exit schedule update_policy runs over a mini-batch."""
    accum_ctx = None
    seen = []
    for micro_idx in range(n_micro):
        if active:
            if micro_idx == 0 and n_micro > 1:
                accum_ctx = module.no_sync() if isinstance(module, _FakeFSDP) else None
                if accum_ctx is not None:
                    accum_ctx.__enter__()
            elif micro_idx == n_micro - 1 and accum_ctx is not None:
                accum_ctx.__exit__(None, None, None)
                accum_ctx = None
        seen.append(module._sync_gradients)
    return seen


def test_only_the_last_micro_batch_synchronizes():
    module = _FakeFSDP()
    syncing = _drive(4, active=True, module=module)
    assert syncing == [False, False, False, True], syncing
    assert module.events == ["enter", "exit"]
    # Left in the state the next mini-batch expects.
    assert module._sync_gradients is True


def test_a_single_micro_batch_never_enters():
    """accum==1 has nothing to accumulate; entering would only cost the assert."""
    module = _FakeFSDP()
    assert _drive(1, active=True, module=module) == [True]
    assert module.events == []


def test_disabled_leaves_every_micro_batch_synchronizing():
    module = _FakeFSDP()
    assert _drive(3, active=False, module=module) == [True, True, True]
    assert module.events == []


def test_context_is_none_unless_the_module_is_fsdp():
    """A plain module has no collectives to skip, so there is nothing to enter."""
    assert mod._grad_sync_context(torch.nn.Linear(2, 2), True) is None
    assert mod._grad_sync_context(_FakeFSDP(), False) is None


def test_fsdp2_no_sync_toggles_the_setter_and_restores_on_error():
    class _FSDP2:
        def __init__(self):
            self.state = []

        def set_requires_gradient_sync(self, value):
            self.state.append(value)

    module = _FSDP2()
    with mod._fsdp2_no_sync(module):
        assert module.state == [False]
    assert module.state == [False, True]

    module = _FSDP2()
    with pytest.raises(RuntimeError):
        with mod._fsdp2_no_sync(module):
            raise RuntimeError("boom")
    assert module.state == [False, True], "a raising body must still re-enable sync"


def test_the_flag_defaults_to_off():
    from omegaconf import OmegaConf

    actor = object.__new__(mod.DataParallelPPOActor)
    cfg = OmegaConf.create({})
    assert bool(cfg.get("no_sync_grad_accum", False)) is False
    assert bool(OmegaConf.create({"no_sync_grad_accum": True}).get("no_sync_grad_accum", False)) is True
    del actor
