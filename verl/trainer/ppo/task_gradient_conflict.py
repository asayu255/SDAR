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
"""Do the three tasks pull the shared parameters the same way, or against each other?

Everything the cross-teacher arms measured lives in the OUTPUT distribution --
whether the teachers agree about a token. ``docs/cross_teacher_theory.md``
concludes that a multitask student's transfer, if it has any, runs through the
SHARED TRUNK rather than through the target: the tasks meet in the parameters,
not in the loss. Nothing in this repo has looked there. This module does.

WHAT IS COMPARED. One batch, one set of parameters, no optimizer step. The rows
are split by task, and each task's rows are split again into two halves, giving
six gradients

    g[task][half],   task in {alfworld, search, webshop},  half in {A, B}

and the halves are the whole point. Two gradients of the SAME task on disjoint
rows are not identical -- minibatch noise alone drives their cosine below 1 --
so a raw cross-task cosine of, say, 0.3 means nothing until it is read against
what one task scores against itself. This is the same correction the
cross-teacher work needed and initially skipped: the ``shuffled`` placebo there,
audit P8's marginal-sign correction there, and here the within-task null.

THE CORRECTION IS THE CLASSICAL ONE FOR ATTENUATION. Write each half's gradient
as signal plus independent noise, ``g_i^A = s_i + e``. Then

    E cos(g_i^A, g_i^B) ~ ||s_i||^2 / (||s_i||^2 + sigma_i^2)      (reliability)
    E cos(g_i^A, g_j^A) ~ <s_i, s_j> / sqrt(...)(...)

so dividing the cross-task cosine by the geometric mean of the two reliabilities
recovers the cosine of the underlying task gradients:

    corrected_ij = cos(g_i, g_j) / sqrt( cos(g_i^A, g_i^B) * cos(g_j^A, g_j^B) )

This is Spearman's correction for attenuation, and it is what makes a number
from one checkpoint interpretable without a second checkpoint to compare it to.
``corrected`` near 0 says the tasks are orthogonal -- neither helping nor
fighting; near +1 that they want the same update; negative that they pull apart.

GRANULARITY IS PER FSDP UNIT, WHICH IS PER TRANSFORMER LAYER. The actor is
wrapped with ``use_orig_params=False``, so a rank holds a flat shard of each
unit's FlatParameter and the shard boundaries do not line up with individual
weight matrices. Rather than map offsets back to parameters, the statistics are
taken per unit -- and because the auto-wrap policy wraps one transformer block
per unit, that IS the layer profile the question wants. Embedding and head come
out as their own units.

The layer profile is the discriminating measurement. If sharing works, conflict
should be lowest in the trunk, where the representation is common, and rise
toward the head, where the tasks genuinely differ. Flat or inverted profiles say
the sharing story is wrong.

TWO GEOMETRIES, AND THE SECOND IS THE ONE THAT ACTS. Adam does not apply the
gradient; it applies ``g / (sqrt(v) + eps)``, which rescales every coordinate by
its own history. Conflict in the raw gradient can therefore differ from conflict
in what reaches the parameters, so both are reported. ``preconditioned`` is the
one to read when asking what actually happened to the weights.

WHAT THIS CANNOT SAY. The cosine is basis-dependent: two tasks can look opposed
coordinate-wise while being compatible in function space (they want the same
direction at different scales). Per-unit cosine is the robust reading and the
per-coordinate sign agreement is reported beside it as a secondary, noisier one.
And all of it is local -- a statement about these parameters and this batch, not
about whether the tasks are compatible in principle.
"""

import math
from collections import defaultdict
from typing import Optional

import torch

__all__ = [
    "HALVES",
    "unit_layer_name",
    "capture_unit_grads",
    "precondition_grads",
    "pairwise_moments",
    "conflict_report",
    "sum_halves",
    "probe_directions",
    "build_direction",
    "collapse_to_per_task_",
    "estimate_host_bytes",
    "save_snapshots",
    "load_snapshots",
    "direction_norm",
    "scale_direction",
    "direction_dot",
    "add_direction_",
    "snapshot_params",
    "restore_params_",
    "check_perturbable",
    "token_attribution",
    "aggregate_attribution",
    "attribution_shares",
]

HALVES = ("A", "B")

# Accumulated in float64 throughout. A dot product over a 1.7B-parameter shard in
# float32 loses the low-order bits of exactly the small-magnitude coordinates the
# late-training gradient is made of, and the cosine is a ratio of two such sums.
_ACC = torch.float64


def unit_layer_name(module_name: str) -> str:
    """A readable, sortable name for one FSDP unit.

    FSDP1 prefixes wrapped modules with ``_fsdp_wrapped_module``; the transformer
    blocks come through as ``...model.layers.<n>...``. The layer index is pulled
    out so the report sorts in depth order rather than lexically (``layer_2``
    before ``layer_10``), which is the axis the profile is read along.
    """
    clean = module_name.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "")
    parts = clean.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return f"layer_{int(parts[i + 1]):03d}"
    if not clean:
        return "root"
    for tag in ("embed_tokens", "lm_head", "norm"):
        if tag in clean:
            return tag
    return clean


def capture_unit_grads(model, *, device="cpu") -> dict:
    """``{unit_name: 1-D grad shard}`` for every FSDP unit holding a gradient.

    Reads ``flat_param.grad`` rather than per-parameter grads because
    ``use_orig_params=False`` means per-parameter grads do not exist -- the unit's
    parameters are concatenated into one flat tensor and this rank owns a slice
    of it. Two ranks' slices of the same unit are different coordinates of the
    same vector, which is why every statistic below sums over ranks before it
    divides.

    The copy is not optional -- the next group backwards into the same buffers --
    and ``device="cpu"`` is the default for a reason. Six groups at fp32 over a
    1.7B model are ~20 GB a rank held simultaneously, on top of a rollout engine
    that is still holding its own share of the card. Host memory has room for
    them and :func:`pairwise_moments` only ever needs one unit resident at a
    time, so the device copy is a per-layer transfer instead of a per-model one.
    """
    out, seen = {}, {}
    for name, module in model.named_modules():
        flat = getattr(module, "_flat_param", None)
        if flat is None or flat.grad is None:
            continue
        # DEDUPLICATED BY TENSOR IDENTITY. named_modules() yields both the root
        # ("") and its FSDP inner module ("_fsdp_wrapped_module"), and on this
        # wrapping they carry the SAME flat parameter -- bitwise identical, 155M
        # coordinates a rank. Kept twice it was 15% of every pooled coordinate
        # statistic, double-weighting the embedding, whose cross-task cosine
        # (+0.098) is well below the mid-layers' (+0.22) -- so the duplicate
        # dragged pooled numbers down and inflated the saved file by 622 MB.
        key = (flat.grad.data_ptr(), flat.grad.numel())
        if key in seen:
            continue
        seen[key] = name
        out[unit_layer_name(name)] = flat.grad.detach().reshape(-1).to(device, copy=True)
    return out


def iter_unit_grads(model):
    """``(unit_name, 1-D grad)`` per FSDP unit holding a gradient, deduplicated.

    ONE ITERATION, THREE CALLERS. The accumulator, the tail-mini-batch control
    and anything else that walks the units have to agree on which tensors exist
    and how the duplicates are collapsed; when that rule lived in each caller,
    a change in one was a silent disagreement with the others. The duplicate is
    real: named_modules() yields the root and its ``_fsdp_wrapped_module`` and on
    this wrapping they carry the SAME flat parameter.
    """
    seen = set()
    for name, module in model.named_modules():
        flat = getattr(module, "_flat_param", None)
        if flat is None or flat.grad is None:
            continue
        key = (flat.grad.data_ptr(), flat.grad.numel())
        if key in seen:
            continue
        seen.add(key)
        yield unit_layer_name(name), flat.grad.detach().reshape(-1)


def unit_grad_stats(model) -> dict:
    """``{"sq", "nonzero"}`` over the CURRENT gradients, as 0-dim tensors.

    What the pre-fix probe recorded is one mini-batch's gradient, so calling this
    once per mini-batch and keeping the last value states, per cell and for free,
    what that measurement would have said. Norm via vector_norm rather than
    ``(g*g).sum()``: this runs once per mini-batch per unit and the embedding
    shard alone would otherwise be a 1.2 GB temporary.
    """
    sq = nz = None
    for _, g in iter_unit_grads(model):
        s_ = torch.linalg.vector_norm(g, dtype=torch.float64) ** 2
        n_ = (g != 0).sum().to(torch.float64)
        sq = s_ if sq is None else sq + s_
        nz = n_ if nz is None else nz + n_
    return {} if sq is None else {"sq": sq, "nonzero": nz}


def accumulate_unit_grads_(dest: dict, model) -> dict:
    """Add the CURRENT ``flat_param.grad`` of every unit into ``dest``, in place.

    WHY THIS EXISTS AND capture_unit_grads DOES NOT SUFFICE. ``update_policy``
    runs one backward per MINI-batch and calls ``optimizer.zero_grad()`` at the
    top of each one. A probe that reads ``.grad`` once after ``update_policy``
    returns therefore captures the LAST mini-batch alone: for search in one
    measured batch that was 20 rows of 140, and the advantage-bearing rows were
    all in mini-batches 2 and 3, so the captured gradient was identically zero
    while the loss had a policy gradient the whole time.

    WHAT WOULD ALSO HAVE WORKED, AND WHY THIS IS PREFERRED. Skipping that
    ``zero_grad()`` would in fact accumulate: ``prepare_gradient_for_backward``
    moves a surviving sharded ``.grad`` back into ``_saved_grad_shard`` at the
    start of the next backward (``_flat_param.py:1609``), and the post-backward
    hook then adds into it. An earlier version of this comment asserted the
    opposite -- that FSDP always overwrites -- and that was wrong. Accumulating
    explicitly is kept because it does not depend on that internal path, does not
    depend on ``set_to_none``, and leaves the training loop's own zero_grad where
    it is; it also makes the summation visible at the call site.

    NO RESCALING. With ``use_dynamic_bsz=False`` each micro-batch contributes
    ``policy_loss / gradient_accumulation`` and ``task_agg_scale`` carries
    ``task_dp_world_size * gradient_accumulation``, so after FSDP's average over
    ranks a mini-batch's ``.grad`` is exactly the sum over ITS rows of
    ``grad(w_i * L_i)`` -- no mini-batch-dependent factor. The sum over
    mini-batches is then the gradient of the sum over all rows, which is what the
    probe is supposed to be measuring. Under ``use_dynamic_bsz=True`` each
    mini-batch is normalised by its own token count and that is no longer true;
    the caller asserts against it.

    Accumulates on the gradient's own device: this runs once per mini-batch (55
    times for one alfworld pass) and a host copy each time would move terabytes.
    """
    for unit, g in iter_unit_grads(model):
        if unit in dest:
            dest[unit] += g
        else:
            # fp32 REGARDLESS of the gradient's own dtype. This is a sum of up to
            # ~55 mini-batches for one alfworld pass; FSDP's reduce_dtype defaults
            # to fp32 so today the cast is free, but a config that reduced in bf16
            # would otherwise turn a 55-term sum into three significant digits
            # without anything saying so.
            dest[unit] = g.to(torch.float32, copy=True)
    return dest


def precondition_grads(grads: dict, optimizer, model, *, eps: float = 1e-8) -> dict:
    """Divide each unit's shard by ``sqrt(exp_avg_sq)``, Adam's own scaling.

    The update Adam applies is ``m_hat / (sqrt(v_hat) + eps)``; the direction it
    moves the parameters in is therefore not the gradient's. Conflict measured on
    the raw gradient answers "do the tasks want opposite things", and conflict
    measured here answers "do their updates cancel", which is the question about
    what the run actually did.

    The first moment is deliberately NOT applied: ``m`` is a running average over
    past steps and mixes all three tasks together, so dividing this batch's
    per-task gradient by it would compare each task against the others' history.
    Only the per-coordinate scale is borrowed.

    A unit with no optimizer state yet (step 0, or a frozen unit) is passed
    through unscaled rather than dropped. WHICH UNITS THOSE WERE IS RECORDED on
    ``precondition_grads.last_coverage`` and surfaced by the caller as
    ``precondition_coverage`` -- without it a report labelled "preconditioned"
    can be the raw geometry and nothing distinguishes the two.
    """
    scale = {}
    for name, module in model.named_modules():
        flat = getattr(module, "_flat_param", None)
        if flat is None:
            continue
        state = optimizer.state.get(flat, None) if optimizer is not None else None
        v = None if state is None else state.get("exp_avg_sq", None)
        if v is not None:
            scale[unit_layer_name(name)] = v.detach().reshape(-1)
    out, skipped = {}, {}
    for name, g in grads.items():
        v = scale.get(name, None)
        if v is None or v.numel() != g.numel():
            # SILENT PASS-THROUGH IS THE HAZARD. A unit with no Adam state, or
            # whose state has a different element count, keeps its RAW gradient
            # here -- so a "preconditioned" report can be the raw one for some or
            # all units and read identically. The docstring used to claim the
            # report said which; it did not. It does now, via
            # precondition_coverage on the caller.
            skipped[name] = "no exp_avg_sq" if v is None else f"numel {v.numel()} != {g.numel()}"
            out[name] = g
            continue
        # The snapshots live on the host and the optimizer state on the card, so
        # the scale comes to the gradient rather than the other way round: moving
        # a unit's gradient to the device here would put it there twice, once for
        # this and again for the moments.
        out[name] = g / (v.to(g.device, g.dtype).sqrt() + eps)
    precondition_grads.last_coverage = {
        "units": len(grads),
        "preconditioned": len(grads) - len(skipped),
        "skipped": skipped,
    }
    return out


def _all_reduce(t: torch.Tensor) -> torch.Tensor:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    return t


def pairwise_moments(groups: dict, *, with_signs: bool = True, device=None) -> dict:
    """Sufficient statistics for every pair of the six gradients, per unit.

    Args:
        groups: ``{(task, half): {unit: 1-D shard}}``, typically held on the host.
        device: where the arithmetic runs. One unit's six vectors are moved there
            at a time and released, so the resident cost is a layer rather than a
            model -- which is what lets the six snapshots live on the host.

    Returns:
        ``{unit: {"dot": {(k1,k2): float}, "sq": {k: float},
                  "agree": {(k1,k2): float}, "n": float}}``

    ONE all-reduce for the whole report. Every quantity here is a sum over
    coordinates, so the shards compose by addition and the ranks can be joined
    once at the end; doing it per pair would be a collective inside a double loop
    over six vectors, and a rank whose unit list differed would deadlock. The
    keys are sorted so every rank walks the same order.
    """
    keys = sorted(groups)
    units = sorted({u for g in groups.values() for u in g})

    flat_vals, index = [], []
    for unit in units:
        present = [k for k in keys if unit in groups[k]]
        if not present:
            continue
        # Resident for this unit only, then dropped at the end of the iteration.
        vec = {k: groups[k][unit].to(device=device, dtype=_ACC) for k in present}
        n = float(vec[present[0]].numel())
        for k in present:
            flat_vals.append((vec[k] * vec[k]).sum())
            index.append((unit, "sq", k, None))
        for a in range(len(present)):
            for b in range(a + 1, len(present)):
                ga, gb = vec[present[a]], vec[present[b]]
                flat_vals.append((ga * gb).sum())
                index.append((unit, "dot", present[a], present[b]))
                if with_signs:
                    same = (torch.sign(ga) == torch.sign(gb)).to(_ACC).sum()
                    flat_vals.append(same)
                    index.append((unit, "agree", present[a], present[b]))
        flat_vals.append(torch.tensor(n, dtype=_ACC))
        index.append((unit, "n", None, None))
        del vec

    if not flat_vals:
        return {}
    packed = _all_reduce(torch.stack([v.to(device=device, dtype=_ACC) for v in flat_vals]))

    out = defaultdict(lambda: {"dot": {}, "sq": {}, "agree": {}, "n": 0.0})
    for value, (unit, kind, k1, k2) in zip(packed.tolist(), index):
        if kind == "n":
            out[unit]["n"] = value
        elif kind == "sq":
            out[unit]["sq"][k1] = value
        else:
            out[unit][kind][(k1, k2)] = value
    return dict(out)


def _cos(moments: dict, k1, k2) -> float:
    if k1 == k2:
        return 1.0
    d = moments["dot"].get((k1, k2), moments["dot"].get((k2, k1), None))
    if d is None:
        return float("nan")
    n1, n2 = moments["sq"].get(k1, 0.0), moments["sq"].get(k2, 0.0)
    if n1 <= 0.0 or n2 <= 0.0:
        return float("nan")
    return d / math.sqrt(n1 * n2)


def _agree(moments: dict, k1, k2) -> float:
    a = moments["agree"].get((k1, k2), moments["agree"].get((k2, k1), None))
    n = moments.get("n", 0.0)
    return float("nan") if a is None or n <= 0 else a / n


def conflict_report(moments: dict, tasks) -> dict:
    """Per unit and task pair: reliability, raw cosine, corrected cosine, signs.

    ``within`` is the reliability of one task's gradient at this batch size --
    the cosine of its two half-gradients -- and it is the denominator everything
    else is read against. A unit where ``within`` is near zero is one where a
    single task's own gradient does not reproduce across disjoint rows; nothing
    about the tasks' relationship can be read there at all, and ``corrected`` is
    withheld rather than reported as a large number divided by noise.

    ``cross`` averages the four half-to-half cosines between two tasks instead of
    taking one, which costs nothing and halves the variance.

    ``conflicting`` is ``max(0, -corrected)``: the share of one task's update
    that the other actively opposes. It is a function of the cosine, reported
    because it is the quantity a gradient-surgery method would remove.
    """
    report = {}
    for unit, m in sorted(moments.items()):
        within = {t: _cos(m, (t, "A"), (t, "B")) for t in tasks}
        entry = {
            "n_coords": m.get("n", 0.0),
            "within": within,
            "grad_norm": {
                t: math.sqrt(max(0.0, sum(m["sq"].get((t, h), 0.0) for h in HALVES) / len(HALVES)))
                for t in tasks
            },
            "pairs": {},
        }
        for a in range(len(tasks)):
            for b in range(a + 1, len(tasks)):
                ti, tj = tasks[a], tasks[b]
                vals = [_cos(m, (ti, h1), (tj, h2)) for h1 in HALVES for h2 in HALVES]
                vals = [v for v in vals if not math.isnan(v)]
                cross = sum(vals) / len(vals) if vals else float("nan")
                wi, wj = within.get(ti, float("nan")), within.get(tj, float("nan"))
                # TWO THRESHOLDS, BOTH REPORTED. 0.05 is the arithmetic guard --
                # correcting by a reliability at or below zero turns noise into a
                # confident number of either sign. 0.3 is the threshold the
                # theory doc states as the condition for REPORTING a corrected
                # value, and the two were silently different: the doc said
                # "r > 0.3" while every corrected number in the payload came
                # from the 0.05 gate. Under 0.3 the preconditioned search pairs
                # have no eligible unit at all, which is a fact about the
                # measurement and has to be visible rather than implied.
                ok = all(not math.isnan(v) for v in (cross, wi, wj)) and wi > 0.05 and wj > 0.05
                corrected = cross / math.sqrt(wi * wj) if ok else float("nan")
                reportable = ok and wi > 0.3 and wj > 0.3
                corrected_reportable = corrected if reportable else float("nan")
                signs = [_agree(m, (ti, h1), (tj, h2)) for h1 in HALVES for h2 in HALVES]
                signs = [v for v in signs if not math.isnan(v)]
                within_signs = [_agree(m, (t, "A"), (t, "B")) for t in (ti, tj)]
                within_signs = [v for v in within_signs if not math.isnan(v)]
                entry["pairs"][f"{ti}__{tj}"] = {
                    "cross": cross,
                    # The doc's condition; NaN where either task's reliability is
                    # below 0.3. Read this one, and "corrected" only alongside
                    # the two within values.
                    "corrected_r03": corrected_reportable,
                    "within_i": wi,
                    "within_j": wj,
                    "corrected": corrected,
                    "conflicting": max(0.0, -corrected) if not math.isnan(corrected) else float("nan"),
                    "sign_agree": sum(signs) / len(signs) if signs else float("nan"),
                    "sign_agree_within": (
                        sum(within_signs) / len(within_signs) if within_signs else float("nan")
                    ),
                }
        report[unit] = entry
    return report


def summarize(report: dict, tasks) -> dict:
    """Norm-weighted means over units, for the one-line reading.

    Weighted by the units' gradient norms rather than counted per unit: a
    26-layer model reports 26 cosines and the ones carrying no gradient should
    not vote equally with the ones carrying it.
    """
    out = {}
    # BOTH GATES SUMMARISED, because "summary" was read as the headline number
    # and it aggregated only the 0.05-gated `corrected` -- so a pair whose
    # reliability never reached the 0.3 the theory doc states as the reporting
    # condition still produced a confident one-line figure. The r03 variant
    # carries how many units survived that gate, which is the number that says
    # whether the figure can be read at all (for the preconditioned search pairs
    # it is zero).
    for key, field in (("", "corrected"), ("_r03", "corrected_r03")):
        out.update(_summarize_field(report, tasks, field, key))
    return out


def _summarize_field(report: dict, tasks, field: str, suffix: str) -> dict:
    out = {}
    for a in range(len(tasks)):
        for b in range(a + 1, len(tasks)):
            pair = f"{tasks[a]}__{tasks[b]}"
            num = den = 0.0
            n_units = 0
            for entry in report.values():
                v = entry["pairs"].get(pair, {}).get(field, float("nan"))
                if math.isnan(v):
                    continue
                w = entry["grad_norm"].get(tasks[a], 0.0) * entry["grad_norm"].get(tasks[b], 0.0)
                num += w * v
                den += w
                n_units += 1
            out[pair + suffix] = num / den if den > 0 else float("nan")
            out[pair + suffix + "_n_units"] = n_units
    return out


# --------------------------------------------------------------------------- #
# attribution: which token positions produced a given update direction
# --------------------------------------------------------------------------- #
#
# The cosines above say the tasks pull together. They do not say WHAT pulls --
# and the audit's reading of this project is that what the teachers agree about
# is the harness format, not task knowledge (section 11.6), with three quarters
# of the distillation budget going to non-content tokens (section 11.3). If that
# is right, the shared direction should be made of tag and format positions, and
# that is checkable rather than arguable.
#
# It is checkable because the loss is a sum over token positions, so the gradient
# is too:
#
#     L = sum_t w_t D_t          =>      g = sum_t w_t grad D_t
#
# and the decomposition is unique. The contribution of position t to a direction
# u is the directional derivative w_t <grad D_t, u>, which needs no per-token
# gradient to be materialised -- a central difference in parameter space gives
# every position at once:
#
#     w_t [ D_t(theta + eps u) - D_t(theta - eps u) ] / (2 eps),   error O(eps^2)
#
# Two forward passes per direction. And the arithmetic checks itself: summed over
# positions the result must equal <g, u>, which pairwise_moments already computed
# from the gradients themselves. A mismatch means eps is wrong or the wiring is,
# and either way the number is not to be read.


def sum_halves(snaps: dict, tasks) -> dict:
    """``{task: {unit: g_A + g_B}}`` -- one gradient per task, halves rejoined.

    The halves exist to estimate reliability; a direction wants the task's whole
    gradient, so they are added back here rather than one of them being picked.
    """
    out = {t: {} for t in tasks}
    for key, units in snaps.items():
        # Both key shapes reach here: (task, half) straight from an accumulation,
        # bare task names when the gradients were loaded from disk already
        # collapsed. Assuming the pair shape is what broke the first
        # attribution-only run, so it is not assumed anywhere.
        task = key[0] if isinstance(key, tuple) else key
        if task not in out:
            continue
        for unit, g in units.items():
            out[task][unit] = g.clone() if unit not in out[task] else out[task][unit] + g
    return out


def collapse_to_per_task_(snaps: dict, tasks) -> dict:
    """Replace the six half-gradients with three per-task sums, freeing the halves.

    The halves exist only to estimate reliability, which the conflict report has
    already used them for. Attribution needs each task's whole gradient, so once
    the report is written the halves are 20 GB a rank of host memory held for
    nothing. Summing in place and dropping the originals cuts the attribution
    pass's resident set by half -- and it was the attribution pass that ran the
    host out of memory.

    Mutates and returns ``snaps``, now keyed by task instead of by (task, half).
    """
    per_task = {}
    for (task, _half) in sorted(snaps):
        units = snaps.pop((task, _half))
        dst = per_task.setdefault(task, {})
        for unit, g in units.items():
            if unit in dst:
                dst[unit] += g
            else:
                dst[unit] = g
        units.clear()
    snaps.clear()
    snaps.update(per_task)
    return snaps


def save_snapshots(snaps: dict, path: str, *, rank: int, meta: dict = None) -> str:
    """Write this rank's gradients so attribution can run in a FRESH process.

    Attribution kept failing for a structural reason rather than a fixable one:
    it ran inside the process that had just accumulated, so it inherited that
    process's 227 GB and had 24 GB to work in. A separate run never allocates the
    six snapshots at all -- it loads three per-task sums from disk -- and sits
    around 190 GB with room to spare.

    Persisting also makes attribution cheap to redo. As written, an eps that
    turned out too large for the linearity check cost a full re-accumulation;
    with the gradients on disk it costs one rollout.

    One file per rank, because the tensors are FSDP shards: rank 0's slice of a
    unit is not rank 1's, and a file loaded on the wrong rank would silently
    attribute one rank's coordinates to the other's parameters. The world size
    is recorded and checked on load.
    """
    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    out = f"{path}.rank{rank}.pt"
    payload = {
        "meta": dict(meta or {}),
        "rank": int(rank),
        "keys": [list(k) if isinstance(k, tuple) else k for k in snaps],
        "tensors": {str(k): {u: v for u, v in units.items()} for k, units in snaps.items()},
    }
    torch.save(payload, out)
    return out


def load_snapshots(path: str, *, rank: int, expect_world: int = None, dtype=None) -> tuple:
    """``(snaps, meta)`` for this rank, refusing a world-size mismatch.

    A shard saved at a different world size covers different coordinates, and
    loading it would attribute the wrong slice of every unit -- quietly, since
    the shapes would not necessarily disagree.

    ``dtype`` casts on the way in, and bfloat16 is what the attribution pass
    asks for. It halves a footprint that got the pass refused outright: the
    saved file now holds six half-gradients rather than three per-task sums,
    because the halves carry strictly more information (the sums are derivable,
    the halves are not) and the within-task reliability floor cannot be
    recomputed offline without them. What the direction needs is to point the
    right way; bfloat16's ~0.4% per-coordinate error, averaged over 850M
    coordinates, does not move it. The sum-check stays valid because both sides
    of it are computed from the same cast direction.
    """
    payload = torch.load(f"{path}.rank{rank}.pt", map_location="cpu", weights_only=False)
    meta = payload.get("meta", {})
    saved_world = meta.get("world_size", None)
    if expect_world is not None and saved_world is not None:
        assert int(saved_world) == int(expect_world), (
            f"gradients were saved at world_size={saved_world} and this run has "
            f"{expect_world}; the shards cover different coordinates"
        )
    assert int(payload.get("rank", rank)) == int(rank), "rank mismatch in the saved shard"
    out = {}
    for k, units in payload["tensors"].items():
        # Keys come back as strings; a "('alfworld', 'A')" repr is restored to the
        # tuple so the loaded dict has the same shape the saver was handed.
        key = k
        if isinstance(k, str) and k.startswith("("):
            try:
                import ast as _ast
                t = _ast.literal_eval(k)
                key = tuple(t) if isinstance(t, tuple) else k
            except Exception:
                key = k
        out[key] = {u: (v if dtype is None else v.to(dtype)) for u, v in units.items()}
    return out, meta


def estimate_host_bytes(snaps: dict, *, n_directions_resident: int = 2,
                        with_param_snapshot: bool = True) -> dict:
    """What the attribution pass will hold, so it can be refused before it runs.

    The first N=8 attempt was killed by the node memory monitor because nobody
    added this up: six snapshots, seven direction sets built at once, and a
    parameter copy came to about 95 GB across two ranks on top of a training
    footprint that already peaks near the box's limit. An estimate that is
    printed and checked is cheap; the run it protects is a hundred minutes.
    """
    one = 0
    for units in snaps.values():
        one = max(one, sum(v.numel() * v.element_size() for v in units.values()))
    held = len(snaps) + n_directions_resident + (1 if with_param_snapshot else 0)
    return {"per_set_bytes": one, "sets_resident": held, "total_bytes": one * held}


def build_direction(snaps: dict, tasks, name: str, per_task: dict = None) -> dict:
    """ONE direction, built on demand and discarded by the caller after use.

    Materialising the whole family at once is what killed the first N=8 run:
    ``probe_directions`` returns seven sets -- shared, three per-task, three
    residuals -- and at 1.7B parameters that is about 24 GB a rank of host
    memory ON TOP of the six gradient snapshots already there and the parameter
    snapshot the perturbation needs. Ray's node monitor killed the worker in the
    middle of attribution and, because the report had not been written yet, took
    eight batches of accumulated gradient with it.

    ``per_task`` may be passed in when the caller already has it, so that the
    halves are not rejoined once per direction.
    """
    pt = per_task if per_task is not None else sum_halves(snaps, tasks)
    if name in pt:
        return pt[name]
    units = sorted({u for d in pt.values() for u in d})
    shared = {}
    for unit in units:
        vs = [pt[t][unit] for t in tasks if unit in pt[t]]
        if vs:
            shared[unit] = sum(vs[1:], vs[0].clone()) / float(len(vs))
    if name == "shared":
        return shared
    if name.endswith("_residual"):
        task = name[: -len("_residual")]
        if task in pt:
            return {u: pt[task][u] - shared[u] for u in pt[task] if u in shared}
    return {}


def probe_directions(snaps: dict, tasks) -> dict:
    """The directions worth attributing, from the accumulated gradients.

    ``shared`` is the equal-weighted mean, which is what the multitask update
    actually moves along once ``normalize_loss_by_task`` has put each task at a
    third of the loss -- so attributing it is attributing the real update, not a
    construct. ``<task>_residual`` is what that task wants beyond the consensus,
    and it is the direction that would carry task knowledge if any is being
    carried. Both are reported because the interesting outcome is the CONTRAST:
    the same roles dominating both would mean the split says nothing.
    """
    per_task = sum_halves(snaps, tasks)
    units = sorted({u for d in per_task.values() for u in d})
    shared = {}
    for unit in units:
        vs = [per_task[t][unit] for t in tasks if unit in per_task[t]]
        if vs:
            shared[unit] = sum(vs[1:], vs[0].clone()) / float(len(vs))
    dirs = {"shared": shared}
    for t in tasks:
        dirs[t] = per_task[t]
        dirs[f"{t}_residual"] = {
            u: per_task[t][u] - shared[u] for u in per_task[t] if u in shared
        }
    return dirs


def direction_norm(direction: dict, device=None) -> float:
    """Global L2 norm over all units, summed across ranks."""
    total = torch.zeros((), dtype=_ACC, device=device)
    for v in direction.values():
        total = total + (v.to(device=device, dtype=_ACC) ** 2).sum()
    return float(_all_reduce(total).sqrt())


def scale_direction(direction: dict, factor: float) -> dict:
    return {u: v * factor for u, v in direction.items()}


def direction_dot(grads: dict, direction: dict, device=None) -> float:
    """``<g, u>``, summed across ranks. The value the attribution must reproduce."""
    total = torch.zeros((), dtype=_ACC, device=device)
    for unit, g in grads.items():
        u = direction.get(unit, None)
        if u is None:
            continue
        total = total + (g.to(device=device, dtype=_ACC) * u.to(device=device, dtype=_ACC)).sum()
    return float(_all_reduce(total))


@torch.no_grad()
def snapshot_params(model, *, device="cpu") -> dict:
    """``{unit: copy of the flat shard}``, for an exact restore afterwards.

    The probe perturbs parameters in place and has to put them back. Undoing by
    adding the opposite displacement does NOT put them back: in float32 two round
    trips at eps=1e-3 drift by 7e-5 relative, and in bfloat16 by a factor of 787,
    because adding a small displacement to a coarse mantissa and subtracting it
    again is not the identity. Every direction after the first would then be
    evaluated at a quietly different point. Copying costs one sharded model on
    the host -- next to the six gradient snapshots already there -- and removes
    the failure entirely.
    """
    out = {}
    for name, module in model.named_modules():
        flat = getattr(module, "_flat_param", None)
        if flat is None:
            continue
        out[unit_layer_name(name)] = flat.detach().to(device, copy=True)
    return out


@torch.no_grad()
def restore_params_(model, snapshot: dict) -> int:
    """Put the parameters back exactly. Returns units restored."""
    n = 0
    for name, module in model.named_modules():
        flat = getattr(module, "_flat_param", None)
        if flat is None:
            continue
        saved = snapshot.get(unit_layer_name(name), None)
        if saved is None or saved.numel() != flat.numel():
            continue
        flat.copy_(saved.to(flat.device, flat.dtype).reshape(flat.shape))
        n += 1
    return n


def check_perturbable(model, *, eps: float) -> dict:
    """Refuse to attribute through parameters too coarse for ``eps`` to register.

    A central difference needs ``theta + eps u`` to actually differ from
    ``theta``. In bfloat16 -- eight mantissa bits -- a relative displacement of
    1e-3 rounds away on most coordinates, so both passes would return the same
    per-token KL and the attribution would come back a plausible-looking zero.
    That is the failure worth refusing rather than warning about, because its
    output is not obviously wrong.

    IT IS THE FORWARD'S DTYPE THAT MATTERS, NOT THE STORAGE DTYPE, and reading
    the wrong one is how this guard let the failure through once already. Under
    FSDP mixed precision the flat parameter is HELD in float32 and the forward
    runs on a bfloat16 all-gathered copy, so inspecting ``flat.dtype`` reports
    float32 (resolution 1.2e-7), passes eps=1e-3 with five orders of margin, and
    the two passes then compute on bf16 weights that are bitwise identical. The
    attribution came back exactly 0.0 for every role of every task, which is
    precisely the output this function exists to prevent.

    So the effective dtype is taken from the handle: ``_orig_param_dtype`` when
    the handle is currently forcing full precision (eval mode with
    ``FSDP_USE_FULL_PREC_IN_EVAL=1``, or SUMMON_FULL_PARAMS), and
    ``_fwd_bwd_param_dtype`` otherwise. Because ``_force_full_precision`` is a
    live property of the module's training flag, this must be called in the SAME
    mode the forwards will run in.

    Raising eps cannot rescue a bf16 forward: a unit-norm direction over 1.7e9
    parameters puts ~2.4e-5 in each coordinate, and clearing bf16's mantissa
    step at a typical weight of 0.02 would need eps > 1.6 -- a displacement at
    which nothing is a derivative any more. The fix is the dtype, not the step.

    Returns ``{dtype: str, resolution: float, ok: bool}``; the caller aborts on
    ``ok=False`` rather than reporting an attribution nobody can read.
    """
    dtypes, storage, forced = set(), set(), set()
    for _name, module in model.named_modules():
        handle = getattr(module, "_handle", None)
        if handle is not None and getattr(handle, "flat_param", None) is not None:
            full = bool(getattr(handle, "_force_full_precision", False))
            forced.add(full)
            orig = getattr(handle, "_orig_param_dtype", None)
            fwd = getattr(handle, "_fwd_bwd_param_dtype", None)
            eff = orig if full else (fwd or orig)
            if eff is not None:
                dtypes.add(eff)
            storage.add(handle.flat_param.dtype)
            continue
        flat = getattr(module, "_flat_param", None)
        if flat is not None:
            # Not FSDP-wrapped, or a torch without handles: the parameter is
            # what the forward sees.
            dtypes.add(flat.dtype)
            storage.add(flat.dtype)
    if not dtypes:
        return {"dtype": "none", "resolution": float("nan"), "ok": False}
    worst = max(dtypes, key=lambda d: float(torch.finfo(d).eps))
    res = float(torch.finfo(worst).eps)
    out = {
        "dtype": str(worst),
        "resolution": res,
        # eps must clear the mantissa step by a healthy margin, or most
        # coordinates round back to where they started.
        "ok": bool(float(eps) > 20.0 * res),
        "dtypes": sorted(str(d) for d in dtypes),
        "storage_dtypes": sorted(str(d) for d in storage),
        "force_full_precision": sorted(forced),
    }
    if not out["ok"] and any(str(d) != str(worst) for d in storage):
        out["hint"] = (
            "the forward runs at %s while the parameters are stored at %s: set "
            "FSDP_USE_FULL_PREC_IN_EVAL=1 before FSDP is built and run the "
            "attribution forwards in eval() mode" % (worst, sorted(str(d) for d in storage))
        )
    return out


@torch.no_grad()
def add_direction_(model, direction: dict, scale: float) -> int:
    """``theta += scale * u`` in place, on the FSDP flat shards. Returns units touched.

    In place because a second copy of a 1.7B model is memory this probe has
    better uses for. It is NOT self-inverting -- see :func:`snapshot_params` for
    the measured drift -- so the caller restores from a snapshot rather than by
    adding the negation.
    """
    n, seen = 0, set()
    for name, module in model.named_modules():
        flat = getattr(module, "_flat_param", None)
        if flat is None:
            continue
        # The root and its _fsdp_wrapped_module carry the SAME flat parameter
        # under two names. Visiting both added the root's displacement twice
        # -- 2*eps on the embedding and lm_head against eps everywhere else --
        # which the sum check would report as a ratio off by the root's share.
        # Same dedup as capture_unit_grads, on the storage, not the name.
        key = (flat.data.data_ptr(), flat.numel())
        if key in seen:
            continue
        seen.add(key)
        u = direction.get(unit_layer_name(name), None)
        if u is None or u.numel() != flat.numel():
            continue
        flat.add_(u.to(flat.device, flat.dtype).reshape(flat.shape), alpha=float(scale))
        n += 1
    return n


def token_attribution(kl_plus, kl_minus, *, eps: float, row_weight=None):
    """``w_t (D_t^+ - D_t^-) / (2 eps)`` -- one number per token position.

    ``row_weight`` is the per-row loss weight the training objective carries
    (``normalize_loss_by_task``); leaving it out would attribute the direction of
    an objective the run never optimised.
    """
    attr = (kl_plus.to(torch.float64) - kl_minus.to(torch.float64)) / (2.0 * float(eps))
    if row_weight is not None:
        attr = attr * row_weight.to(attr.device, attr.dtype).reshape(-1, 1)
    return attr


def aggregate_attribution(attr, *, roles, response_mask, task_ids, task_names, role_names) -> dict:
    """Signed and absolute attribution per (task, role).

    Both signs are kept. The signed sum is what has to reconcile with ``<g, u>``;
    the absolute sum is what says where the WORK is, since a role can carry a
    great deal of push in both directions and still net out near zero -- which is
    exactly the shape the audit found for the tail.

    ``role_names`` is a ``{code: name}`` mapping, which is how ``sign_weights``
    defines it -- the codes are not a contiguous range and indexing a list by
    them would silently mislabel every role.
    """
    codes = role_names.items() if hasattr(role_names, "items") else enumerate(role_names)
    codes = list(codes)
    mask = response_mask.to(torch.float64)
    a = attr.to(torch.float64) * mask
    out = {}
    tid = task_ids.reshape(-1).tolist()
    for i, t in enumerate(tid):
        if int(t) < 0:
            continue
        name = task_names[int(t)] if int(t) < len(task_names) else str(int(t))
        bucket = out.setdefault(name, {})
        r_row = roles[i]
        for code, rname in codes:
            sel = (r_row == code) & (response_mask[i] > 0)
            if not bool(sel.any()):
                continue
            cell = bucket.setdefault(rname, {"signed": 0.0, "abs": 0.0, "tokens": 0.0})
            vals = a[i][sel]
            cell["signed"] += float(vals.sum())
            cell["abs"] += float(vals.abs().sum())
            cell["tokens"] += float(sel.sum())
    return out


def attribution_shares(agg: dict) -> dict:
    """Per task, each role's share of the absolute attribution."""
    out = {}
    for task, roles in agg.items():
        tot = sum(c["abs"] for c in roles.values())
        out[task] = {
            r: (c["abs"] / tot if tot > 0 else float("nan")) for r, c in sorted(roles.items())
        }
        out[task]["_total_abs"] = tot
        out[task]["_total_signed"] = sum(c["signed"] for c in roles.values())
    return out


# --------------------------------------------------------------------------- #
# making an FSDP forward read the shard it was just handed
# --------------------------------------------------------------------------- #
#
# THIS IS WHY EVERY ATTRIBUTION CAME BACK EXACTLY ZERO, and it is not a
# precision problem, an eps problem, or a dtype problem -- all three were
# fixed, and the zero stayed. FSDP1 all-gathers the flat parameter into a
# padded buffer before a forward and, for SHARD_GRAD_OP (which this actor runs
# under), KEEPS that buffer after the forward: _post_forward_reshard frees it
# only for FULL_SHARD / HYBRID_SHARD, and _should_free_in_backward frees it in
# the backward -- which a forward-only probe pass never runs. On the next
# forward, needs_unshard() sees the buffer still allocated and SKIPS the
# all-gather, using the stale weights.
#
# So the probe's second forward, at theta - eps*u, computed on the buffer the
# first forward built at theta + eps*u. Identical weights, identical per-token
# KL, difference exactly 0.0 at every eps in every dtype -- the observed
# result, including |.| = 0 (not cancellation; nothing moved). add_direction_
# had written the shard correctly the whole time; nothing read it.
#
# For inference this reuse is correct -- weights do not change between
# forwards -- and the probe violates the assumption by mutating the shard in
# place. The fix is to release the buffer before each probe forward so
# needs_unshard() is True and the all-gather runs again, then to ASSERT that it
# is, so that if this reasoning is wrong the run refuses loudly instead of
# filing another plausible zero.


def _fsdp_states_and_handles(model):
    """``[(state, handle)]`` for every FSDP unit that owns a flat parameter."""
    try:
        from torch.distributed.fsdp._traversal_utils import _get_fsdp_states
    except Exception:  # pragma: no cover - torch without FSDP1 internals
        return []
    out = []
    for st in _get_fsdp_states(model):
        h = getattr(st, "_handle", None)
        if h is not None and getattr(h, "flat_param", None) is not None:
            out.append((st, h))
    return out


def _fsdp_handles(model):
    return [h for _st, h in _fsdp_states_and_handles(model)]


@torch.no_grad()
def release_unsharded_(model) -> dict:
    """Free every handle's all-gathered buffer so the next forward re-gathers.

    Goes through FSDP's own ``_reshard(state, handle, free=True)`` -- the same
    call its post-backward makes -- so the flat parameter is switched back to
    its shard, the padded buffer freed, the low-precision shard released, and
    the free recorded on the all-gather rate limiter when one is configured.
    The state left behind is exactly the one FSDP leaves after a backward.
    Handles that are already released are left alone. Returns counts, so the
    caller can see what was actually done.
    """
    from torch.distributed.fsdp._runtime_utils import _reshard

    pairs = _fsdp_states_and_handles(model)
    freed = skipped = 0
    for st, h in pairs:
        if not getattr(h, "uses_sharded_strategy", False):
            skipped += 1
            continue
        if h.needs_unshard():
            skipped += 1  # nothing allocated; the next forward gathers anyway
            continue
        _reshard(st, h, True)
        freed += 1
    return {"handles": len(pairs), "freed": freed, "already_released": skipped}


def assert_will_regather(model) -> int:
    """Refuse unless every sharded handle will all-gather on its next forward.

    This is the precondition the attribution's two forwards depend on. It is
    checked rather than assumed because the previous four runs assumed it.
    """
    stale = []
    n = 0
    for h in _fsdp_handles(model):
        if not getattr(h, "uses_sharded_strategy", False):
            continue
        n += 1
        if not h.needs_unshard():
            stale.append(getattr(getattr(h, "_fully_sharded_module", None), "__class__", type(None)).__name__)
    if stale:
        raise RuntimeError(
            f"grad_probe: {len(stale)}/{n} FSDP handles still hold an all-gathered "
            f"buffer and would NOT re-read the perturbed shard on the next forward "
            f"(e.g. {stale[:3]}). Call release_unsharded_ first."
        )
    return n


# --------------------------------------------------------------------------- #
# per-batch term moments: ONE spelling of the keys, shared by writer and reader
# --------------------------------------------------------------------------- #
#
# The worker writes "sq:<task>:<term>" and "dot:<task>:<term>:<task>:<term>"
# into the payload and an offline script reads them back. Two hand-written
# copies of that format is how a forty-minute run ends in a KeyError, so both
# sides import these.


def term_key(task: str, term: str) -> str:
    return f"{task}:{term}"


def moment_key(kind: str, *names: str) -> str:
    return ":".join((kind,) + names)


def parse_moment_key(key: str):
    """``"dot:a:rl:b:opd"`` -> ``("dot", ("a","rl"), ("b","opd"))``; ``"sq:a:rl"`` -> ``("sq", ("a","rl"))``."""
    parts = key.split(":")
    if parts[0] == "sq" and len(parts) == 3:
        return "sq", (parts[1], parts[2])
    if parts[0] == "dot" and len(parts) == 5:
        return "dot", (parts[1], parts[2]), (parts[3], parts[4])
    raise ValueError(f"not a term-moment key: {key!r}")


def cosine_from_moments(whole: dict, a, b) -> float:
    """Whole-model cosine between (task, term) a and b from a moments dict."""
    ka, kb = term_key(*a), term_key(*b)
    d = whole.get(moment_key("dot", ka, kb), whole.get(moment_key("dot", kb, ka)))
    na, nb = whole.get(moment_key("sq", ka)), whole.get(moment_key("sq", kb))
    if d is None or na is None or nb is None or na <= 0 or nb <= 0:
        return float("nan")
    return d / math.sqrt(na * nb)
