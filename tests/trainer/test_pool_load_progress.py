"""A pool load that prints nothing is indistinguishable from a hang.

The Stage-1 pool is 333 GiB across 90 files and takes minutes to read. Every
other progress print in ``opd_offpolicy_ray_trainer`` passes ``flush=True``; the
three inside ``_load_offpolicy_file`` did not. Under Ray a worker's stdout is a
file, not a tty, so Python block-buffers it at 8 KB -- the driver relays whatever
has crossed a buffer boundary and the tail of the load simply stops mid-stream.
A run in the middle of a normal load then looks frozen, at the one phase where
the plausible failures (swapping, an OOM kill in progress) look the same.

Two things fix that and both are pinned here:

* every print on the load path flushes, and
* there is one line per file whether or not that file had padding rows or dead
  columns to report -- a pool already filtered by ``scripts/cache_teacher_pool.py``
  has neither, so the conditional prints alone would say nothing for the whole
  load.

Source-level for the flushes because the alternative is asserting on the C-level
buffering of a subprocess; behavioural for the per-file line, which is the part
that carries the information.
"""

import inspect
import re

import pytest

from verl.trainer.ppo import opd_offpolicy_ray_trainer as mod


def _prints(func):
    """(source line, whether it flushes) for every print in ``func``."""
    src = inspect.getsource(func)
    out = []
    for match in re.finditer(r"\bprint\(", src):
        depth, i = 1, match.end()
        while depth and i < len(src):
            depth += (src[i] == "(") - (src[i] == ")")
            i += 1
        call = src[match.start():i]
        out.append((call, "flush=True" in call))
    return out


@pytest.mark.parametrize(
    "func",
    [mod.OffPolicyOPDRayTrainer._load_offpolicy_file.__func__,
     mod.OffPolicyOPDRayTrainer._load_offpolicy_data],
    ids=["_load_offpolicy_file", "_load_offpolicy_data"],
)
def test_every_print_on_the_load_path_flushes(func):
    calls = _prints(func)
    assert calls, f"no prints found in {func.__name__}; the parse matched nothing"
    unflushed = [c for c, flushed in calls if not flushed]
    assert not unflushed, (
        f"{func.__name__} has prints that do not flush, so they sit in the block "
        f"buffer of a Ray worker's stdout and the load looks hung: {unflushed}"
    )


def test_a_file_with_nothing_to_report_still_produces_a_line(tmp_path, capsys, monkeypatch):
    """The cached-pool case: no padding, no dropped columns, no narrowing."""
    import numpy as np
    import torch

    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    from tests.trainer.test_offpolicy_student_topk import _Cfg

    rows = 4
    for name in ("alfworld_0000.pt", "alfworld_0001.pt"):
        DataProto.from_dict(
            tensors={"input_ids": torch.zeros(rows, 3, dtype=torch.int32)},
            non_tensors={
                "traj_uid": np.array([f"{name}-{i}" for i in range(rows)], dtype=object),
                "task_name": np.array(["alfworld"] * rows, dtype=object),
            },
        ).save_to_disk(str(tmp_path / name))

    trainer = mod.OffPolicyOPDRayTrainer.__new__(mod.OffPolicyOPDRayTrainer)
    trainer.use_reference_policy = False
    trainer.config = _Cfg(
        algorithm=_Cfg(opd=_Cfg(teacher_data_dir=str(tmp_path), student_indexed_topk=False)),
        actor_rollout_ref=_Cfg(actor=_Cfg(teacher_kl_loss_type="topk_kl", teacher_kl_topk=20,
                                          student_indexed_topk=False)),
        data=_Cfg(task_balance=_Cfg(per_task_batch_size=15)),
        env=_Cfg(rollout=_Cfg(n=8)),
        trainer=_Cfg(val_only=False),
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(RayPPOTrainer, "__init__", lambda self, *a, **k: None)
        mod.OffPolicyOPDRayTrainer.__init__(trainer)

    out = capsys.readouterr().out
    for name in ("alfworld_0000.pt", "alfworld_0001.pt"):
        assert re.search(rf"\[\d+/2\] {re.escape(name)}: 4 rows", out), (
            f"no per-file progress line for {name}; a cached pool would load in silence:\n{out}"
        )
    assert "loading the Stage-1 pool: 2 files" in out
    assert "elapsed" in out
