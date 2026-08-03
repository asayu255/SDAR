"""The logger-only scalars must not be read back inside the micro-batch loop.

``.item()`` on a device tensor blocks the host until the stream drains. In a
multitask step the per-task diagnostics repeat that four-plus times per task per
micro-batch, which is several hundred forced synchronisations per step spent
waiting for numbers nothing reads until the step is over.

``update_policy`` therefore keeps them as 0-d tensors and reads them once at the
end. Two things have to hold for that to be a free change, and both are checked
here:

* nothing in the micro-batch loop reads a tensor back (the property being bought);
* the single read produces the same number the old path did. The old path
  appended per micro-batch and let the driver's ``reduce_metrics`` take the mean;
  the new one stacks and takes the mean here. Those must agree, or every logged
  loss silently shifts.
"""

import re

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

try:
    from verl.utils.metric import reduce_metrics
    from verl.utils.py_functional import append_to_dict
    from verl.workers.actor import dp_actor as mod
except Exception as e:  # pragma: no cover - environment without full deps
    pytest.skip(f"verl import unavailable: {e}", allow_module_level=True)


def _micro_batch_loop_source():
    """The body of update_policy's micro-batch loop, as text.

    Read from the file rather than via inspect.getsource: update_policy is
    wrapped by GPUMemoryLogger, which does not use functools.wraps, so getsource
    returns the decorator's inner function.
    """
    source = open(mod.__file__.replace(".pyc", ".py")).read()
    start = source.index("for micro_idx, data in enumerate(micro_batches):")
    # The loop ends where the optimizer step begins -- everything after that
    # runs once per mini-batch, not once per micro-batch.
    end = source.index('with _actor_phase("actor.optim"):', start)
    return source[start:end]


def test_no_tensor_is_read_back_inside_the_micro_batch_loop():
    body = _micro_batch_loop_source()
    offenders = [
        line.strip()
        for line in body.splitlines()
        if ".item()" in line and not line.strip().startswith("#")
    ]
    assert not offenders, (
        "these read a device tensor back once per micro-batch; defer them with "
        f"_defer(...) instead: {offenders}"
    )


def test_every_deferred_metric_is_a_tensor_expression():
    """_defer calls .detach(), so passing an already-read float would raise at
    runtime -- on a GPU host, hours into a run. Catch it here instead."""
    body = _micro_batch_loop_source()
    for call in re.findall(r"_defer\(\s*([^\n]*?),", body):
        assert ".item()" not in call, call


def test_deferral_reduces_to_the_same_number_as_the_old_append_path():
    values = [torch.tensor(v, dtype=torch.float32) for v in (1.5, -0.25, 3.0, 0.0, 7.125)]

    # New path: stack the 0-d tensors and read once.
    deferred = torch.stack([v.detach() for v in values]).mean().item()

    # Old path: read each one, append, and let the driver reduce.
    metrics = {}
    for v in values:
        append_to_dict(metrics, {"actor/pg_loss": v.detach().item()})
    reduced = reduce_metrics(metrics)["actor/pg_loss"]

    assert deferred == pytest.approx(float(reduced))


def test_weighted_loss_metrics_are_deferred_rather_than_assigned():
    """``metrics[k] = v`` inside the loop keeps only the LAST micro-batch.

    That is wrong for any per-micro-batch number, and it is *silently* wrong for
    the task-weighted losses: ``_balance_batch`` reorders rows, so the last
    micro-batch is frequently nothing but ``adjust_batch`` padding, whose task
    weight is 0. ``sdar/loss_weighted`` shipped that way and logged exactly
    0.000 for a whole run while the loss it was reporting was fine.
    """
    body = _micro_batch_loop_source()
    deferred = set(re.findall(r'_defer\(\s*"([^"]+)"', body))
    for name in set(re.findall(r'"((?:actor|sdar)/[^"]*_weighted)"', body)):
        assert name in deferred, (
            f"{name} is a per-micro-batch value; assigning it keeps only the "
            "last micro-batch, which is often all-padding (weight 0)"
        )


def test_an_all_padding_last_micro_batch_reads_zero_when_assigned():
    """Why the test above matters, in numbers."""
    # Weighted losses per micro-batch; the last one is padding-only, so every
    # row weight is 0 and the weighted aggregate is exactly 0.
    per_micro = [torch.tensor(v) for v in (0.42, 0.38, 0.45, 0.0)]

    assigned = per_micro[-1].item()  # what metrics[k] = v logs
    averaged = torch.stack([v.detach() for v in per_micro]).mean().item()

    assert assigned == 0.0
    assert averaged == pytest.approx(0.3125)


def test_a_metric_absent_from_some_micro_batches_averages_over_the_ones_it_had():
    """A task can be missing from a micro-batch entirely, so its per-task metric
    is appended fewer times than there are micro-batches. Both paths must divide
    by the number of appends, not by the micro-batch count."""
    present = [torch.tensor(2.0), torch.tensor(4.0)]  # 2 of, say, 5 micro-batches

    deferred = torch.stack([v.detach() for v in present]).mean().item()

    metrics = {}
    for v in present:
        append_to_dict(metrics, {"actor/pg_loss/search": v.detach().item()})
    reduced = reduce_metrics(metrics)["actor/pg_loss/search"]

    assert deferred == pytest.approx(3.0)
    assert deferred == pytest.approx(float(reduced))
