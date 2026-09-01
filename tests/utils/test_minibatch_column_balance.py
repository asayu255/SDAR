# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Matching the mini-batch columns without changing what an optimizer step sees.

``deal_by_length`` matches the columns and ships OFF, for a reason written at
``_BALANCE_MINIBATCH``: it re-deals each rank's whole partition, so rows that
shared an optimizer step stop sharing one, and an arm run with it on is not
comparable with the arms already finished.

``rebalance_minibatch_columns`` is the same repair under the constraint that
removes that objection. Mini-batch k is the W*M rows sitting at positions
``[k*M, (k+1)*M)`` on every rank; it takes exactly those rows and re-partitions
THEM. The column's membership -- which is what an optimizer step is -- is held
fixed by construction, and only the rank carrying each row moves.

The tests below are about that constraint, not about the balancing: a
partitioner that balances better while quietly moving a row into the next
mini-batch would be a different experiment, and it would show up here rather
than in a loss curve six hours later.
"""

import random

from verl.utils.seqlen_balancing import (
    deal_by_length,
    get_seqlen_balanced_partitions,
    log_minibatch_unbalance,
    rebalance_minibatch_columns,
)


def _columns(partitions, minibatch_rows):
    """The row SETS an optimizer step sees, one per mini-batch."""
    n = len(partitions[0])
    return [
        {i for p in partitions for i in p[a : a + minibatch_rows]}
        for a in range(0, n, minibatch_rows)
    ]


def _random_case(rows=360, world_size=2, seed=0):
    rng = random.Random(seed)
    # Roughly this arm's shape: a 4,608-wide window, most rows well short of it.
    seqlens = [rng.randint(200, 4608) for _ in range(rows)]
    partitions = get_seqlen_balanced_partitions(seqlens, k_partitions=world_size, equal_size=True)
    return seqlens, partitions


# --------------------------------------------------------------------------- #
# The constraint. This is the whole reason the function exists rather than a
# second call to deal_by_length.
# --------------------------------------------------------------------------- #


def test_every_optimizer_step_sees_exactly_the_rows_it_saw_before():
    seqlens, partitions = _random_case()
    for minibatch_rows in (30, 60, 7):
        out = rebalance_minibatch_columns(seqlens, partitions, minibatch_rows, micro_batch_rows=5)
        assert _columns(out, minibatch_rows) == _columns(partitions, minibatch_rows), (
            f"a row changed mini-batch at minibatch_rows={minibatch_rows}; that is a "
            f"different trajectory, not a different rounding"
        )


def test_deal_by_length_does_not_preserve_them_which_is_why_it_is_off():
    """The contrast, pinned. If this ever starts passing, the two mechanisms have
    converged and one of them should go -- but until then it is what separates a
    reordering that needs a fresh A/B from one that does not."""
    seqlens, partitions = _random_case()
    dealt = [deal_by_length(seqlens, p, 30) for p in partitions]
    assert _columns(dealt, 30) != _columns(partitions, 30)


def test_it_is_a_permutation_with_the_rank_sizes_intact():
    """A row silently dropped here is a row never trained on, and a rank left one
    row short makes gradient_accumulation -- a CONFIGURED constant -- disagree
    with the micro-batches actually run."""
    seqlens, partitions = _random_case(rows=360, world_size=2)
    out = rebalance_minibatch_columns(seqlens, partitions, 30, micro_batch_rows=5)
    assert [len(p) for p in out] == [len(p) for p in partitions]
    assert sorted(i for p in out for i in p) == sorted(i for p in partitions for i in p)


def test_a_short_tail_column_is_handled_and_not_dropped():
    """adjust_batch rounds to lcm(...) and not to the mini-batch size, so the last
    column is normally short. It still has to be balanced and still has to keep
    its rows."""
    seqlens, partitions = _random_case(rows=350)      # 175 a rank: five 30s and a 25
    out = rebalance_minibatch_columns(seqlens, partitions, 30, micro_batch_rows=5)
    assert _columns(out, 30) == _columns(partitions, 30)
    assert len(_columns(out, 30)) == 6                # five full columns and the tail
    assert len(_columns(out, 30)[-1]) == 2 * 25
    assert [len(p) for p in out] == [175, 175]


def test_unequal_partitions_are_refused_rather_than_silently_realigned():
    """The columns are defined by POSITION. Partitions of different lengths do not
    have the same columns, so there is nothing to hold fixed."""
    try:
        rebalance_minibatch_columns([1, 2, 3, 4, 5], [[0, 1, 2], [3, 4]], 2)
    except ValueError as exc:
        assert "equal-size" in str(exc)
    else:
        raise AssertionError("accepted partitions whose columns do not line up")


# --------------------------------------------------------------------------- #
# That it actually balances. Weaker claims than the ones above on purpose: this
# is the part a better partitioner is allowed to improve on.
# --------------------------------------------------------------------------- #


def test_the_column_wait_all_but_disappears():
    seqlens, partitions = _random_case()
    before = log_minibatch_unbalance(seqlens, partitions, minibatch_rows=30, prefix="p")
    after_partitions = rebalance_minibatch_columns(seqlens, partitions, 30, micro_batch_rows=5)
    after = log_minibatch_unbalance(seqlens, after_partitions, minibatch_rows=30, prefix="p")
    assert before["p/minibatch_wait_frac"] > 0.02
    assert after["p/minibatch_wait_frac"] < before["p/minibatch_wait_frac"] / 10


def test_the_micro_batch_wait_is_the_larger_one_and_is_also_cut():
    """The ranks do not only meet at the mini-batch boundary on these arms:
    exchange_teacher_logprobs_multi runs a collective pair from inside the
    micro-batch loop. A column with matching totals can still be split into
    micro-batches that do not match, so this is a separate claim."""
    seqlens, partitions = _random_case()
    before = log_minibatch_unbalance(
        seqlens, partitions, minibatch_rows=30, prefix="p", micro_batch_rows=5
    )
    assert before["p/microbatch_wait_frac"] > before["p/minibatch_wait_frac"]
    after_partitions = rebalance_minibatch_columns(seqlens, partitions, 30, micro_batch_rows=5)
    after = log_minibatch_unbalance(
        seqlens, after_partitions, minibatch_rows=30, prefix="p", micro_batch_rows=5
    )
    assert after["p/microbatch_wait_frac"] < before["p/microbatch_wait_frac"] / 2


def test_balancing_the_column_alone_leaves_the_micro_batches_unmatched():
    """Why micro_batch_rows is not cosmetic. Without it the column is split by
    index order, which the partitioner's own _check_and_sort_partitions imposed
    and which carries no length information at all."""
    seqlens, partitions = _random_case()
    without = rebalance_minibatch_columns(seqlens, partitions, 30, micro_batch_rows=0)
    with_deal = rebalance_minibatch_columns(seqlens, partitions, 30, micro_batch_rows=5)
    m_without = log_minibatch_unbalance(
        seqlens, without, minibatch_rows=30, prefix="p", micro_batch_rows=5
    )["p/microbatch_wait_frac"]
    m_with = log_minibatch_unbalance(
        seqlens, with_deal, minibatch_rows=30, prefix="p", micro_batch_rows=5
    )["p/microbatch_wait_frac"]
    assert m_with < m_without / 2
    # ...and it costs nothing at the mini-batch level, since it only reorders
    # inside a rank's share of one column.
    assert _columns(with_deal, 30) == _columns(without, 30)


def test_four_ranks_too():
    """world_size is 2 on the box this was measured on; the arithmetic is not."""
    seqlens, partitions = _random_case(rows=480, world_size=4, seed=7)
    out = rebalance_minibatch_columns(seqlens, partitions, 30, micro_batch_rows=5)
    assert _columns(out, 30) == _columns(partitions, 30)
    before = log_minibatch_unbalance(seqlens, partitions, minibatch_rows=30, prefix="p")
    after = log_minibatch_unbalance(seqlens, out, minibatch_rows=30, prefix="p")
    assert after["p/minibatch_wait_frac"] < before["p/minibatch_wait_frac"] / 5


def test_one_rank_is_a_no_op_rather_than_an_error():
    seqlens = [5, 3, 9, 1]
    assert rebalance_minibatch_columns(seqlens, [[0, 1, 2, 3]], 2) == [[0, 1, 2, 3]]


# --------------------------------------------------------------------------- #
# The measurement, which runs whether or not the flag does -- the size of the
# prize has to be readable off a run that did not take the change.
# --------------------------------------------------------------------------- #


def test_the_counterfactual_is_reported_without_the_flag():
    seqlens, partitions = _random_case()
    m = log_minibatch_unbalance(
        seqlens, partitions, minibatch_rows=30, prefix="p", micro_batch_rows=5
    )
    assert m["p/minibatch_wait_frac_columns"] < m["p/minibatch_wait_frac"]
    assert m["p/microbatch_wait_frac_columns"] < m["p/microbatch_wait_frac"]


def test_the_micro_batch_columns_are_omitted_rather_than_guessed():
    """0 means the arm has no fixed micro-batch width (dynamic bsz, or the
    non-per-gpu key). Reporting a number under an assumed width would be a
    measurement of the assumption."""
    seqlens, partitions = _random_case()
    m = log_minibatch_unbalance(seqlens, partitions, minibatch_rows=30, prefix="p")
    assert "p/microbatch_wait_frac" not in m
    assert "p/minibatch_wait_frac_columns" in m       # this one needs no width


def test_the_old_keys_did_not_move():
    """Runs already in flight are charted on these names."""
    seqlens, partitions = _random_case()
    m = log_minibatch_unbalance(seqlens, partitions, minibatch_rows=30, prefix="p")
    for key in ("p/minibatch_spread_mean", "p/minibatch_spread_max",
                "p/minibatch_wait_frac", "p/minibatch_wait_frac_dealt"):
        assert key in m


# --------------------------------------------------------------------------- #
# The switch.
# --------------------------------------------------------------------------- #


def test_the_two_orderings_cannot_both_be_on():
    """They are two different orderings of the same rows and only one can be
    dispatched. Silently letting one win would make the run's trajectory depend
    on which line of _balance_batch was written first."""
    import importlib
    import os

    import verl.trainer.ppo.ray_trainer as rt

    before = (os.environ.get("BALANCE_MINIBATCH"), os.environ.get("BALANCE_MINIBATCH_COLUMNS"))
    os.environ["BALANCE_MINIBATCH"] = "1"
    os.environ["BALANCE_MINIBATCH_COLUMNS"] = "1"
    try:
        raised = False
        try:
            importlib.reload(rt)
        except ValueError as exc:
            raised = "BALANCE_MINIBATCH_COLUMNS" in str(exc)
        assert raised, "both flags on was accepted; one of the two orderings is being dropped"
    finally:
        for name, value in zip(("BALANCE_MINIBATCH", "BALANCE_MINIBATCH_COLUMNS"), before):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        importlib.reload(rt)


def test_the_reordering_the_scripts_do_not_export_is_still_off():
    """BALANCE_MINIBATCH moves rows between optimizer steps and is not exported
    by any run script. Its default is what keeps a stray import from changing an
    arm."""
    import verl.trainer.ppo.ray_trainer as rt

    assert rt._BALANCE_MINIBATCH is False


# --------------------------------------------------------------------------- #
# Where it may ship. The invariance argument is not a property of the function,
# it is a property of the function TOGETHER WITH the arm's loss aggregation, and
# both halves live in different files.
# --------------------------------------------------------------------------- #

_CROSS_TEACHER_SCRIPTS = (
    "examples/opd_grpo_trainer/run_multitask_cross_teacher_klw_qwen3.sh",
    "examples/opd_grpo_trainer/run_multitask_cross_teacher_klw_control_qwen3.sh",
    # The target arm carries the block for the same reason the pair above does:
    # its comparators are exactly those two scripts, and an arm that lacks a
    # reordering its control has differs from it in more than the mechanism.
    "examples/opd_grpo_trainer/run_multitask_cross_teacher_target_qwen3.sh",
)


def _repo_text(rel):
    import os

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    with open(os.path.join(root, rel)) as fh:
        return fh.read()


def test_both_arms_of_the_ab_export_the_same_value():
    """Summation order moves, so a control run without it and a treatment run
    with it differ in more than the mechanism under test. This is the pair the
    arm is read as, so the two scripts have to agree line for line."""
    lines = [
        [ln.strip() for ln in _repo_text(s).splitlines()
         if ln.strip().startswith("export BALANCE_MINIBATCH_COLUMNS")]
        for s in _CROSS_TEACHER_SCRIPTS
    ]
    assert all(len(x) == 1 for x in lines), f"expected exactly one export per script, got {lines}"
    assert lines[0] == lines[1], f"the two arms export different values: {lines}"


def test_it_only_ships_where_the_row_weights_do():
    """``rebalance_minibatch_columns`` leaves the gradient unchanged because the
    loss is a weighted row SUM whose weights are step-level. Under
    ``agg_loss(..., "token-mean")`` the divisor is the micro-batch's own token
    count and moving a row across ranks reweights it -- a different objective,
    not a different rounding. So a script that turns this on must also pin
    normalize_loss_by_task."""
    for script in _CROSS_TEACHER_SCRIPTS:
        text = _repo_text(script)
        if "export BALANCE_MINIBATCH_COLUMNS=${BALANCE_MINIBATCH_COLUMNS:-1}" not in text:
            continue
        assert "normalize_loss_by_task=True" in text, (
            f"{script} balances the columns without the per-task row weights that make "
            f"that value-preserving; under a token-mean it changes the objective"
        )


def test_no_other_script_turned_it_on_by_copying_the_block():
    """The two above are the pair that were changed together. Any third script
    reaching for it has to make the same argument for itself -- most of the other
    arms are already finished runs whose numbers a reordering would not match."""
    import glob
    import os

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    for path in glob.glob(os.path.join(root, "examples", "**", "*.sh"), recursive=True):
        rel = os.path.relpath(path, root)
        if rel in _CROSS_TEACHER_SCRIPTS:
            continue
        with open(path) as fh:
            assert "BALANCE_MINIBATCH_COLUMNS" not in fh.read(), (
                f"{rel} exports BALANCE_MINIBATCH_COLUMNS; it was added to the cross-teacher "
                f"pair together and every other arm has to be argued for on its own"
            )


# --------------------------------------------------------------------------- #
# The driver seam. Everything above is arithmetic on index lists; this is what
# the workers are actually handed, which is a reorder of the batch followed by a
# contiguous split -- so an off-by-one between the two would balance perfectly
# and still dispatch the wrong rows.
# --------------------------------------------------------------------------- #


def _fake_trainer(*, world_size, mini_batch_size, micro_batch_size, flag):
    import verl.trainer.ppo.ray_trainer as rt
    from omegaconf import OmegaConf

    trainer = rt.RayPPOTrainer.__new__(rt.RayPPOTrainer)
    trainer.config = OmegaConf.create({
        "actor_rollout_ref": {"actor": {
            "ppo_mini_batch_size": mini_batch_size,
            "ppo_micro_batch_size_per_gpu": micro_batch_size,
            "use_dynamic_bsz": False,
        }},
    })
    trainer.actor_rollout_wg = type("wg", (), {"world_size": world_size})()
    rt._BALANCE_MINIBATCH_COLUMNS = flag
    rt._SAID_BALANCE_MINIBATCH = True
    return trainer, rt


def _dispatched(batch, world_size, mini_rows):
    """What each rank's mini-batch k holds after reorder + the contiguous split."""
    lens = batch.batch["attention_mask"].sum(-1).tolist()
    ids = batch.batch["row_id"].tolist()
    n = len(ids) // world_size
    per_rank = [(ids[r * n : (r + 1) * n], lens[r * n : (r + 1) * n]) for r in range(world_size)]
    columns = []
    for a in range(0, n, mini_rows):
        columns.append([(rows[a : a + mini_rows], tok[a : a + mini_rows]) for rows, tok in per_rank])
    return columns


def _batch(rows, seed=3):
    import torch
    from verl.protocol import DataProto

    rng = random.Random(seed)
    width = 4608
    lens = [rng.randint(200, width) for _ in range(rows)]
    mask = torch.zeros(rows, width, dtype=torch.long)
    for i, n in enumerate(lens):
        mask[i, :n] = 1
    return DataProto.from_dict(tensors={
        "attention_mask": mask,
        "row_id": torch.arange(rows, dtype=torch.long),
    })


def test_balance_batch_dispatches_the_same_optimizer_steps_with_the_flag_on():
    world_size, mini_rows, rows = 2, 30, 360
    off_batch, on_batch = _batch(rows), _batch(rows)

    trainer, rt = _fake_trainer(world_size=world_size, mini_batch_size=mini_rows * world_size,
                                micro_batch_size=5, flag=False)
    off_metrics = {}
    trainer._balance_batch(off_batch, metrics=off_metrics)
    try:
        trainer, rt = _fake_trainer(world_size=world_size, mini_batch_size=mini_rows * world_size,
                                    micro_batch_size=5, flag=True)
        on_metrics = {}
        trainer._balance_batch(on_batch, metrics=on_metrics)
    finally:
        rt._BALANCE_MINIBATCH_COLUMNS = False

    off = _dispatched(off_batch, world_size, mini_rows)
    on = _dispatched(on_batch, world_size, mini_rows)
    assert len(off) == len(on) == rows // (mini_rows * world_size)
    for k, (a, b) in enumerate(zip(off, on)):
        assert {i for rows_, _ in a for i in rows_} == {i for rows_, _ in b for i in rows_}, (
            f"optimizer step {k} was handed a different set of rows"
        )
        # ...and that set is now split evenly, which is the point.
        spread = max(sum(t) for _, t in b) - min(sum(t) for _, t in b)
        assert spread <= 2, f"column {k} still spreads {spread} tokens across the ranks"

    # Not zero: Karmarkar-Karp leaves a remainder, and the spread assertion above
    # already pins it at a couple of tokens a column. This is that in the unit the
    # run is charted in.
    assert on_metrics["global_seqlen/minibatch_wait_frac"] < 1e-5
    assert off_metrics["global_seqlen/minibatch_wait_frac"] > 0.01
    # The whole-batch numbers are computed on the pre-column partition and must
    # not move: they are what says the rank totals were equal to begin with.
    for key in ("global_seqlen/balanced_min", "global_seqlen/balanced_max"):
        assert on_metrics[key] == off_metrics[key]


def test_balance_batch_reports_the_micro_batch_meeting_when_the_width_is_fixed():
    trainer, rt = _fake_trainer(world_size=2, mini_batch_size=60, micro_batch_size=5, flag=False)
    metrics = {}
    trainer._balance_batch(_batch(360), metrics=metrics)
    assert metrics["global_seqlen/microbatch_wait_frac"] > metrics["global_seqlen/minibatch_wait_frac"]
    assert metrics["global_seqlen/microbatch_wait_frac_columns"] < metrics["global_seqlen/microbatch_wait_frac"]


def test_dynamic_bsz_turns_the_micro_batch_reading_off_rather_than_guessing():
    """Under use_dynamic_bsz the micro-batches are cut by tokens, so there is no
    fixed row width and a number reported under one would measure the guess."""
    import verl.trainer.ppo.ray_trainer as rt
    from omegaconf import OmegaConf

    trainer, _ = _fake_trainer(world_size=2, mini_batch_size=60, micro_batch_size=5, flag=False)
    trainer.config = OmegaConf.merge(trainer.config, OmegaConf.create(
        {"actor_rollout_ref": {"actor": {"use_dynamic_bsz": True}}}))
    assert trainer._micro_batch_rows_per_rank() == 0
    metrics = {}
    trainer._balance_batch(_batch(360), metrics=metrics)
    assert "global_seqlen/microbatch_wait_frac" not in metrics
