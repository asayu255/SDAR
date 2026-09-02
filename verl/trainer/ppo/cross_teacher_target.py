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
"""Cross-teacher TARGET injection: move the distillation target, not its weight.

The design and every number quoted below are in ``docs/cross_teacher_target_design.md``;
the measurements that forced them are in ``docs/cross_teacher_kl_weight_offline_audit.md``.
This module exists because of one result in that audit: a positive scalar on
``KL(p_student || p_on)`` cannot inject anything, since the minimiser is
``p_student = p_on`` whatever the scalar is. The previous arm
(``cross_teacher_kl_weight.py``) was that scalar. So the two intents this arm
serves have to be written as a target instead:

    intent 1  all teachers' policy shifts agree  ->  distil harder there
    intent 2  the on-task shift is small and the off-task shifts are commonly
              large  ->  emit a learning signal from the off-task teachers

WHY THE TWO CHANNELS ARE ONE EXPRESSION. Write ``s`` for the off-task teachers'
agreed sign, ``L`` for their consensus volume and ``O = |hat_on|``. Channel A
(corroboration) is ``s * min(O, L)`` and fires only when the on-task teacher
signs along; channel B (fallback) is ``s * relu(L - O)``, the part of the
off-task voice that exceeds the on-task one. The identity
``|h| = min(|h|, O) + relu(|h| - O)`` makes that an exact partition -- no
evidence is counted twice -- and the sum telescopes:

    c = a + b = s * L                  when the on-task teacher agrees
              = s * relu(L - O)        when it is silent or opposed
              = 0                      when the off-task teachers are split

CONTINUITY, WHICH IS WHY THERE IS NO DEADZONE. The shipped predecessor spent
``report_epsilon`` on deciding when a teacher was "silent", and the audit
measured that its value (0.1 RMS) was the worst of every value tested: reach
5.1% of teacher mass at a signal purity of 0.16. Here all three boundaries close
by themselves. At ``hat_on -> 0`` the agreeing branch gives ``s*L`` and the
opposed branch gives ``s*relu(L - eps) -> s*L``, so the two meet. At ``O -> L``
the relu closes at zero. And where the off-task signs disagree, ``L`` is a
geometric mean that goes to zero with its smallest factor, so the jump as ``s``
flips is itself of size zero. Nothing needs a threshold.

GEOMETRIC MEAN, NOT MINIMUM, for ``L``. The min is dominated by the quietest
teacher and measured 1/2.19 of the geometric mean at the median (1/6.1 at p90).
The geometric mean keeps the property the min was chosen for -- it still goes to
zero if any teacher is silent, so "commonly large" is still required -- without
letting one whisper set the volume. It is also insensitive to duplicating a
teacher, which the predecessor's pairwise similarity gate was not.

THE SCALE IS A HYPERPARAMETER AND IS NOT PRETENDED OTHERWISE. ``exp(c)`` is a
probability ratio, so ``c`` has to be in nats, and ``c`` is in RMS units. One RMS
unit measured 2.148 nats, so the four combinations of {min, geomean} x {RMS,
nats} span ``exp(c)`` from 1.52 to 9.82. ``exponent_scale`` IS that conversion and
it is the one knob here. The arm runs it at 1.0 -- see revision 3 below, which
also records the 2.148 that was briefly pinned and why it was wrong.

REVISION (2026-09-02), THREE CHANGES, EACH A RETRACTION.

1. THE MASS-CONSERVING EXCHANGE IS GONE. The first shipped form moved mass only
between the acted candidates and limited the exchange by the smaller side, which
made ``Z = 1`` an identity and left the head at ``w = 1``. Measured, it cost too
much for that: the up side ran at ``T / A_U = 0.026``, so intent 1's channel
delivered 2.6% of what it asked for, and the top layer's effect came out at 0.058
nats against the predecessor weighting arm's 58.3 -- a factor of 1000. It is
replaced by a plain normalisation, ``p_tilde = p_on e^c / Z``, which removes the
throttle (38x) and gives up the two properties the exchange was built for: the
head is taxed by ``1/Z``, and sign faithfulness weakens from "``c > 0`` implies
``w > 1``" to "``c > log Z`` implies ``w > 1``".

The change it makes to the loss is

    dKL = log Z - <c>_{p_s},        |dKL| <= 2 max|c|

so it is bounded by ``c`` however the normalisation is written, where the
predecessor multiplied two unbounded quantities (``(W - 1) * KL``). That bound is
the reason normalisation alone cannot be assumed to reach the predecessor's
scale, and ``target/abs_dkl_mean`` reports the left-hand side every step so the
question is settled by measurement rather than by the estimate in revision 3.

2. THE SUPPORT IS THE STUDENT'S TOP-K AGAIN, withdrawing design constraint C7.
C7 asked for a teacher-indexed support so ``p_tilde`` would be a
student-independent fixed point. What it also did was hide the candidates where
``p_student`` is high and ``p_on`` is negligible (median ``p_on`` 2.3e-06) --
68.9% of the predecessor's effect sat there, and a teacher top-20 whose smallest
member is 1e-2 to 1e-4 cannot contain them at all. The cost is the feedback loop
C7 named: the old student-indexed arm's ``frac_agree_pos`` drifted 0.238 -> 0.193
as the student moved. The incidental benefit is that this arm's intent lock now
differs from both comparators in the mechanism keys ALONE.

3. THE SCALE, TWICE. This module first pinned ``exponent_scale = 1.0`` (the
conservative RMS reading), then 2.148 (the measured nats conversion) on an
estimate that the mechanism was still ~26x short of the predecessor, then 1.0
again. The retraction of 2.148 is worth writing down because it is a lesson about
the estimate, not about the mechanism:

* the "26x short" figure came from the TOP LAYER of the predecessor's own effect
  ranking. That layer is selected BY the predecessor's effect, so comparing a new
  mechanism against it is biased in the predecessor's favour by construction;
* correcting for that needs a candidate-to-position conversion, and the first
  attempt assumed sqrt(n) cancellation across the 20 candidates. Measured, the
  student's mass is concentrated on ONE candidate -- events with ``p_s > 0.3`` are
  5.3% of all events and carry 72.2% of ``sum |p_s c|``, which is what drawing the
  argmax uniformly from 20 would give -- so the candidates do not cancel and the
  conversion is K = 20, not sqrt(20);
* at K = 20 the mechanism's per-position gross at ``exponent_scale = 1.0`` is
  0.281 against the predecessor's, a ratio of 0.91. 2.148 gives 1.72 -- an
  OVERSHOOT of 1.7x, not a shortfall.

THE EXPONENT CLAMP IS A RATE LIMIT, not an overflow guard, since the capacity
exchange no longer bounds one candidate's tilt -- and the same measurement set
its value. See ``_EXPONENT_CLAMP``.
"""
from typing import Optional

import torch

# Numerical floors only. Neither is a threshold on the mechanism: the geometric
# mean's floor is applied inside a log whose result is overwritten wherever an
# exact zero was present, and the normaliser's floor guards a divide at positions
# that are held at w = 1 anyway.
_LOG_FLOOR = 1e-30
_NORM_FLOOR = 1e-12
# The bound on one candidate's tilt, in nats, and a DESIGN PARAMETER rather than
# the overflow guard it was at 30.0: with the capacity exchange gone nothing else
# bounds it. e^5 ~ 148 on a single candidate and |dKL| <= 10 nats at a position.
#
# 5.0 rather than the 3.0 this briefly carried, and the reason is SATURATION, not
# headroom. At (scale 2.148, clamp 3.0) 35.4% of acted candidates sat exactly at
# the cap and they carried 68.3% of the total |c| -- so two thirds of the signal's
# magnitude was the flat value ±3, and the mechanism degenerated back to a binary
# decision precisely where the teachers agreed most strongly, which is the one
# thing the continuous formulation of section 2 exists to avoid. At (1.0, 5.0) the
# cap is reached by 1.9% of acted candidates carrying 8.3% of |c|, and the gross
# per-position effect is 0.296 -- 0.96 of the predecessor's, i.e. the clamp is no
# longer paying for the scale. The cost is log Z's p99 moving 0.88 -> 1.14 and a
# single candidate being able to move 148x, which lands on p_on ~ 1e-6 and so
# reaches p_tilde ~ 1.5e-4.
#
# READ `target/clamped_mass_frac`, NOT `target/clamped_per_step`. The count ratio
# understates the saturation by 1.9-3.1x (0.090 against 0.280 at scale 1.0; 0.354
# against 0.683 at 2.148) because the clamped candidates are the large ones.
_EXPONENT_CLAMP = 5.0

BRANCHES = ("agree", "conflict", "on_silent", "split")


def off_task_consensus(hat_off: torch.Tensor) -> dict:
    """The off-task teachers' agreed sign and consensus volume.

    Args:
        hat_off: (bs, resp, k, n_off) RMS-standardised off-task shifts.

    Returns:
        ``{"s", "L"}``, both (bs, resp, k). ``s`` is the shared sign or zero when
        the teachers do not all share one; ``L`` is the geometric mean of the
        magnitudes and is exactly zero when any teacher's shift is exactly zero.

    The exact-zero branch is load-bearing rather than defensive: the audit found
    that at 65% of the head candidates -- where two thirds of the teacher's
    probability mass sits -- at least one of the three shifts is numerically
    zero. A geometric mean taken through a clamped log would report those as a
    small positive volume and the mechanism would act on the head for a rounding
    reason.
    """
    absh = hat_off.abs()
    sign = hat_off.sign()
    first = sign[..., :1]
    shared = (sign == first).all(dim=-1) & (first.squeeze(-1) != 0)
    s = torch.where(shared, first.squeeze(-1), torch.zeros_like(first.squeeze(-1)))

    # Log space so this generalises past n_off = 2 without the product
    # underflowing; the where() below is what makes an exact zero exact.
    any_zero = (absh <= 0).any(dim=-1)
    log_mean = absh.clamp(min=_LOG_FLOOR).log().mean(dim=-1)
    L = torch.where(any_zero, torch.zeros_like(log_mean), log_mean.exp())
    return {"s": s, "L": L}


def target_exponent(*, hat_on: torch.Tensor, consensus: dict,
                    exponent_scale: float = 1.0) -> dict:
    """``c``, the additive tilt on ``log p_on``, plus the branch it came from.

    Args:
        hat_on: (bs, resp, k) RMS-standardised on-task shift.
        consensus: the output of :func:`off_task_consensus`.
        exponent_scale: multiplies ``c``. 1.0 reads one RMS unit as one nat --
            the conservative end of the choice the module docstring describes.
            Pass the on-task sigma to read it in that teacher's own nats.

    Returns:
        ``{"c", "branch", "a", "b"}``. ``branch`` is an integer index into
        :data:`BRANCHES`. ``a`` and ``b`` are the two channels separately, which
        no arithmetic below needs -- they are returned so the diagnostics can
        report the split the design predicted (A carrying 0.871 of the acted
        magnitude) instead of asserting it.
    """
    s, L = consensus["s"], consensus["L"]
    O = hat_on.abs()
    agree = (hat_on.sign() == s) & (s != 0)
    split = s == 0

    a = torch.where(agree, s * torch.minimum(O, L), torch.zeros_like(L))
    b = torch.where(split, torch.zeros_like(L), s * (L - O).clamp(min=0.0))
    c = (a + b) * float(exponent_scale)

    branch = torch.full_like(L, BRANCHES.index("split"), dtype=torch.long)
    branch = torch.where(agree, torch.full_like(branch, BRANCHES.index("agree")), branch)
    opposed = (~agree) & (~split)
    branch = torch.where(opposed & (hat_on == 0),
                         torch.full_like(branch, BRANCHES.index("on_silent")),
                         torch.where(opposed,
                                     torch.full_like(branch, BRANCHES.index("conflict")),
                                     branch))
    return {"c": c, "branch": branch, "a": a * float(exponent_scale),
            "b": b * float(exponent_scale)}


def normalized_weight(*, c: torch.Tensor, p_on: torch.Tensor,
                      row_available: Optional[torch.Tensor] = None) -> dict:
    """Per-position tilt-and-renormalise: ``p_tilde(v) = p_on(v) e^{c(v)} / Z``.

    Args:
        c: (bs, resp, k) the tilt exponent, in nats.
        p_on: (bs, resp, k) the on-task teacher's probability on the support.
        row_available: (bs,) bool, false for a row with no usable sigma yet. Those
            rows get ``w = 1`` -- the cold start is a no-op rather than a guess.

    Returns:
        ``{"w", "log_w", "log_z", "inv_z", "tail", "moved", "clamped", "c_eff",
        "live"}``. ``w`` is ``p_tilde / p_on`` on the support and ``inv_z`` is that
        same ratio on the tail; ``moved`` is (bs, resp), the total variation
        between ``p_tilde`` and ``p_on`` over the WHOLE vocabulary, tail included.
        ``c_eff`` is the tilt AFTER the clamp -- what was delivered rather than
        what was asked for -- which is what the saturation metric has to weigh.

    THE TAIL IS IN Z AND IS NOT TILTED. ``c`` exists only where the four models
    were read, so the mass outside the support keeps its shape -- but it is
    rescaled with everything else, and that is the difference from the exchange
    this replaces: there the tail was untouched and the support conserved its own
    mass; here the whole distribution is divided by one ``Z``. It is also exactly
    what ``topk_kl_per_token`` reads back, since ``1 - sum_S exp(target_logprob)``
    is ``p_tail / Z``, so the loss's tail bucket needs no separate handling.

    WHAT THIS GIVES UP -- stated, because the form it replaces was built to keep it:

    * THE HEAD IS TAXED. ``c = 0`` no longer implies ``w = 1``; it implies
      ``w = 1/Z``. The old target arm took 46-55% of its amplification out of the
      token the teacher was most confident about and the capacity exchange existed
      to stop exactly that. It is back, bounded by ``|log Z| <= max|c|``;
    * SIGN FAITHFULNESS IS RELATIVE, NOT ABSOLUTE. ``w > 1`` iff ``c > log Z``, so
      a candidate the off-task teachers agreed to raise can still lose probability
      when the rest of the position was tilted up harder.

    What it buys is the reason for the change: the capacity exchange ran the up
    side at ``T / A_U = 0.026``, so intent 1's channel delivered 2.6% of what it
    asked for. Here every candidate gets its own ``e^c`` under one shared divisor,
    and the only cap left is :data:`_EXPONENT_CLAMP`.
    """
    c64 = c.to(torch.float64).clamp(min=-_EXPONENT_CLAMP, max=_EXPONENT_CLAMP)
    clamped = (c.to(torch.float64).abs() > _EXPONENT_CLAMP)
    p = p_on.to(torch.float64)
    e = c64.exp()

    # The teacher's mass outside the support, which the loss reconstructs the same
    # way. clamp(min=0) is float hygiene, not a rule: sum p can exceed 1 by an ulp.
    tail = (1.0 - p.sum(dim=-1)).clamp(min=0.0)
    z = (p * e).sum(dim=-1) + tail

    # A position with no tilt anywhere is left alone BIT-IDENTICALLY rather than
    # divided by a Z that ought to be 1: sum p + (1 - sum p) is not exactly 1 in
    # every rounding mode, and an arm that is off must not perturb the target.
    live = (c64 != 0).any(dim=-1) & (z > _NORM_FLOOR)
    if row_available is not None:
        live = live & row_available.reshape(-1, 1).to(live.device)

    # Everything downstream is built from log_z so the support and the tail cannot
    # disagree about the divisor, and so both are exactly 1 where live is false.
    log_z = torch.where(live, z.clamp(min=_NORM_FLOOR).log(), torch.zeros_like(z))
    inv_z = (-log_z).exp()
    log_w = torch.where(live.unsqueeze(-1), c64 - log_z.unsqueeze(-1),
                        torch.zeros_like(c64))
    w = log_w.exp()

    moved = 0.5 * (((w - 1.0) * p).abs().sum(dim=-1) + (tail * (inv_z - 1.0)).abs())

    return {
        "w": w,
        "log_w": log_w,
        "log_z": log_z,
        "inv_z": inv_z,
        "tail": tail,
        "moved": moved,
        "clamped": clamped,
        "c_eff": c64,
        "live": live,
    }


def build_target(*, on_logprob: torch.Tensor, off_logprob: torch.Tensor,
                 base_logprob: torch.Tensor, diag: torch.Tensor,
                 diag_valid: torch.Tensor, task_ids: torch.Tensor,
                 off_plane_tasks: torch.Tensor, exponent_scale: float = 1.0,
                 shuffle_counterfactual: bool = False,
                 channel_counterfactuals: bool = False,
                 response_mask: Optional[torch.Tensor] = None) -> dict:
    """The whole chain: standardised shifts -> ``c`` -> ``w = e^c / Z`` -> ``log p_tilde``.

    Args:
        on_logprob: (bs, resp, k) on-task teacher log-probs on the support.
        off_logprob: (bs, resp, k, n_off) the off-task teachers on the same ids.
        base_logprob: (bs, resp, k) the pre-RL policy on the same ids.
        diag, diag_valid: (n_tasks,) from ``CumulativePolicyShiftRMS.diagonal``.
        shuffle_counterfactual: build the target a second time from the
            off-task planes rolled within each row's real response
            (``decorrelated_off_shifts`` -- the audit's counterfactual, reused
            verbatim) and return its total variation as ``shuffled_moved``. This
            is the abort gate G1, not an optional extra: the predecessor's gate
            opened at 0.82 of its live rate on shuffled teachers, and a
            mechanism that scores near that is moving mass for reasons that
            survive destroying the position correspondence. Needs
            ``response_mask``.
        channel_counterfactuals: normalise twice more with ``c = a`` and ``c = b``
            alone, returning ``a_only_moved`` / ``b_only_moved``. What each intent
            would have done by itself, measured rather than argued.

    Returns:
        ``{"target_logprob", "w", "c", "branch", "moved", "log_z", ...}``. The
        support is tilted and the whole distribution -- tail included -- is
        divided by ``Z``, so it still sums to one but no longer candidate by
        candidate. ``log_z`` is the head's tax and is a first-class metric.

    THE SUPPORT IS THE STUDENT'S TOP-K (revision 2, module docstring). Every model
    is read at the ids the student just chose, which is where the reverse KL puts
    its weight and where the candidates with high ``p_student`` and negligible
    ``p_on`` live -- the ones a teacher top-20 cannot contain. It also makes the
    target a function of the student, which is the feedback loop C7 was written to
    avoid; that is the accepted cost, not an oversight.
    """
    from verl.trainer.ppo.cross_teacher_kl_weight import standardize_policy_shifts

    shifts = {
        "on": (on_logprob - base_logprob).detach(),
        "off": (off_logprob - base_logprob.unsqueeze(-1)).detach(),
    }
    hat = standardize_policy_shifts(
        shifts=shifts, diag=diag, diag_valid=diag_valid,
        task_ids=task_ids, off_plane_tasks=off_plane_tasks,
    )
    p_on = on_logprob.detach().exp()
    cons = off_task_consensus(hat["off"])
    tilt = target_exponent(hat_on=hat["on"], consensus=cons, exponent_scale=exponent_scale)
    built = normalized_weight(c=tilt["c"], p_on=p_on, row_available=hat["row_available"])

    out = {
        # log_w, not log(w): the subtraction is done in float64 and cast once, and
        # it is exactly 0.0 wherever the mechanism did not act, so an inert
        # position hands the loss the on-task teacher's own bits.
        "target_logprob": on_logprob + built["log_w"].to(on_logprob.dtype),
        "w": built["w"], "log_w": built["log_w"], "c": tilt["c"],
        "c_eff": built["c_eff"], "a": tilt["a"], "b": tilt["b"],
        "branch": tilt["branch"], "moved": built["moved"], "live": built["live"],
        "log_z": built["log_z"], "inv_z": built["inv_z"], "tail": built["tail"],
        "clamped": built["clamped"], "p_on": p_on,
        "consensus_volume": cons["L"], "consensus_sign": cons["s"],
        "row_available": hat["row_available"],
    }

    if shuffle_counterfactual:
        assert response_mask is not None, "the shuffle rolls within the real response"
        from verl.trainer.ppo.cross_teacher_kl_weight import decorrelated_off_shifts

        sh_c = target_exponent(
            hat_on=hat["on"],
            consensus=off_task_consensus(decorrelated_off_shifts(hat["off"], response_mask)),
            exponent_scale=exponent_scale,
        )["c"]
        out["shuffled_moved"] = normalized_weight(
            c=sh_c, p_on=p_on, row_available=hat["row_available"])["moved"]
    if channel_counterfactuals:
        for name in ("a", "b"):
            out[f"{name}_only_moved"] = normalized_weight(
                c=tilt[name], p_on=p_on, row_available=hat["row_available"])["moved"]
    return out


# Qwen3's tag vocabulary: the single-token think tags, the literal 3-token
# spelling of "<think" (which the audit measured at 21.7% of webshop's whole
# teacher KL), and the pieces <action> is built from. The G3 gate reads the
# share of the intervention that lands here: past 0.3 the run is measuring tag
# tokenisation, not cross-teacher structure. Tokenizer-specific by nature, so
# overridable from config as tag_token_ids.
TAG_TOKEN_IDS = (13708, 766, 27, 29, 522, 1311, 151667, 151668)

# Reused from the sign arm so the dump files, the scan script and the reader's
# vocabulary stay one thing. The mapping is exact, not approximate: the sign
# arm's states are defined from the on-task sign against the off-task consensus,
# which is precisely what the branch + consensus sign carry.
_SIGN_STATE_BY_BRANCH_AND_SIGN = {
    # (branch, s > 0): sign_weights.STATE_* index
    ("agree", True): 0,       # agree_pos
    ("agree", False): 1,      # agree_neg
    ("conflict", True): 3,    # off raises, on lowers -> conflict_on_neg
    ("conflict", False): 2,   # off lowers, on raises -> conflict_on_pos
    ("on_silent", True): 4,   # neutral_on_task_silent
    ("on_silent", False): 4,
    ("split", True): 5,       # neutral_off_task_split
    ("split", False): 5,
}


def sign_state_labels(branch: torch.Tensor, consensus_sign: torch.Tensor) -> torch.Tensor:
    """Branch indices -> the sign arm's STATE_* labels, for table/dump reuse."""
    out = torch.full_like(branch, 5)
    pos = consensus_sign > 0
    for (name, is_pos), state in _SIGN_STATE_BY_BRANCH_AND_SIGN.items():
        mask = (branch == BRANCHES.index(name)) & (pos if is_pos else ~pos)
        out = torch.where(mask, torch.full_like(out, state), out)
    return out


class TargetStepStats:
    """The step's gates and pre-registered numbers, as one reduced table.

    Everything in here answers a question the design document asked in advance,
    so the class is the run's side of that appointment. Only what no existing
    metric carries: the per-task teacher KL, the episode metrics and the RMS
    trajectory are already logged elsewhere.

    THE PRE-REGISTERED NUMBERS ARE VOID. ``target_tv = 0.019`` and the branch
    split ``0.871/0.129`` were computed on the mass-conserving exchange over a
    TEACHER-indexed support, and the revision changed both. They are not restated
    here as expectations, and the run log no longer prints them: a stale
    pre-registration is worse than none, because it reads as a violation whenever
    the mechanism is merely different. The gates G1-G3 survive unchanged --
    ``shuffled_tv_ratio``, ``acted_novelty`` and ``tag_share`` are ratios of the
    mechanism against itself and do not depend on how ``w`` was built.

    Scope 0 is the pooled batch; scope 1 + t is task t. float64 sums, one
    all_reduce at step end (SUM for the sums, MAX for the three maxima), rendered
    identically on every rank. Nothing here needs a sort or a second pass, which
    is why the correlation is carried as five running moments rather than as a
    rank statistic.
    """

    _SUMS = (
        "n_pos",              # response-masked positions
        "moved",              # sum of per-position TV(p_tilde, p_on), tail included
        "shuffled_moved",     # same, from the position-decorrelated teachers (G1)
        "a_only_moved",       # corroboration channel alone
        "b_only_moved",       # fallback channel alone
        "live_n",             # positions the mechanism actually tilted
        "log_z",              # sum over live positions of log Z -- the head's tax
        # The change this arm makes to the loss, IN THE PREDECESSOR'S UNITS:
        # dKL = KL(p_s||p_tilde) - KL(p_s||p_on) = log Z - <c>_{p_s}, nats. This
        # is the number that decides whether the scale reached the predecessor's,
        # so it is measured rather than inferred from c. Signed and absolute both:
        # the signed mean says which way the target moved on average, the absolute
        # one is what compares against the old arm's per-position magnitude.
        "dkl",
        "abs_dkl",
        # The three extra moments that turn dkl and d_on into a correlation at
        # step end. Pearson from running sums needs no sort and no second pass, so
        # it survives the all_reduce like everything else here; dkl is exactly
        # zero off the live positions, so its sums are already live-restricted.
        "dkl_sq",
        "d_on_sq",
        "dkl_d_on",
        # H(p_tilde) - H(p_on) per position, bucketed the way the loss buckets:
        # the support term by term, the tail as one. THE ONE CHANNEL THAT MOVED
        # WITH THE 150-STEP ACCURACY GAIN (Spearman +1.00, audit section 15) and
        # the mechanism had no way to say whether it reproduces it.
        "ent_delta",
        # What the support actually covers, for the two masses dkl and log_z
        # depend on. Assumed ~1.0 by a rough estimate that came out slightly
        # above 1; measured here instead.
        "student_support_mass",
        "teacher_support_mass",
        "clamped",            # candidates whose |c| hit the exponent clamp
        # Sum |c_eff| over acted / over clamped candidates. The RATIO of these two
        # is the saturation reading; the count ratio understates it 1.9-3.1x
        # because the clamped candidates are the large ones.
        "abs_c_acted",
        "abs_c_clamped",
        "n_cand",             # masked candidates
        "mass",               # sum p_on over masked candidates
        "acted_cand",         # candidates with c != 0
        "acted_mass",         # their p_on
        "a_absmass",          # sum |a| * p_on  (channel split, teacher measure)
        "b_absmass",          # sum |b| * p_on
        "intervention",       # sum |w - 1| * p_on
        "tag_intervention",   # the same, restricted to TAG_TOKEN_IDS (G3)
        "d_on",               # per-position KL(student || on-task), live positions
        "d_base",             # per-position KL(student || base), live positions (G2)
        # sum |sum_S w p_on + p_tail/Z - 1|. Under the exchange this measured the
        # Z = 1 identity; under normalisation Z = 1 is false by design and what is
        # asserted instead is that p_tilde is a distribution at all.
        "mass_error",
        # branch reach, candidate count and teacher mass per branch: the audit's
        # "read frac and mass_frac as a pair" rule, kept as one (4.3% of
        # candidates carried 64.7% of mass on the old arm).
        "branch0_cand", "branch1_cand", "branch2_cand", "branch3_cand",
        "branch0_mass", "branch1_mass", "branch2_mass", "branch3_mass",
    )

    def __init__(self, *, n_tasks: int, device):
        self.n_tasks = int(n_tasks)
        self.n_scopes = 1 + self.n_tasks
        self.sums = torch.zeros((self.n_scopes, len(self._SUMS)),
                                dtype=torch.float64, device=device)
        self.max_log_w = torch.zeros(1, dtype=torch.float64, device=device)  # G4
        self.max_log_z = torch.zeros(1, dtype=torch.float64, device=device)
        self.max_mass_error = torch.zeros(1, dtype=torch.float64, device=device)

    def _col(self, name):
        return self._SUMS.index(name)

    def update(self, *, built: dict, p_on: torch.Tensor, support_ids: torch.Tensor,
               response_mask: torch.Tensor, task_ids: Optional[torch.Tensor],
               d_on: Optional[torch.Tensor] = None,
               d_base: Optional[torch.Tensor] = None,
               student_logprob: Optional[torch.Tensor] = None,
               on_logprob: Optional[torch.Tensor] = None,
               tag_token_ids=TAG_TOKEN_IDS) -> None:
        m_pos = response_mask.to(torch.float64)                       # (bs, resp)
        m_cand = m_pos.unsqueeze(-1)                                  # (bs, resp, 1)
        p = p_on.to(torch.float64)
        w = built["w"].to(torch.float64)
        c = built["c"].to(torch.float64)
        live = built["live"].to(torch.float64) * m_pos

        tag = torch.zeros_like(support_ids, dtype=torch.bool)
        for tid in tag_token_ids:
            tag |= support_ids == tid
        acted = (c != 0).to(torch.float64) * m_cand
        inter = (w - 1.0).abs() * p * m_cand
        log_z = built["log_z"].to(torch.float64)
        # p_tilde is a distribution, measured: the support's rescaled mass plus
        # the rescaled tail must be one. Not the Z = 1 identity the exchange had
        # -- Z is deliberately not 1 here -- but still an assertion, not a metric.
        pos_err = (
            (w * p).sum(dim=-1) + built["tail"].to(torch.float64) * built["inv_z"].to(torch.float64) - 1.0
        ).abs() * m_pos

        cols = {
            "n_pos": m_pos, "moved": built["moved"].to(torch.float64) * m_pos,
            "live_n": live,
            "log_z": log_z * live,
            "clamped": built["clamped"].to(torch.float64).sum(dim=-1) * m_pos,
            "n_cand": m_cand.expand_as(p).to(torch.float64),
            "mass": p * m_cand, "acted_cand": acted, "acted_mass": p * acted,
            "abs_c_acted": built["c_eff"].to(torch.float64).abs() * acted,
            "abs_c_clamped": (built["c_eff"].to(torch.float64).abs()
                              * built["clamped"].to(torch.float64) * m_cand),
            "teacher_support_mass": (1.0 - built["tail"].to(torch.float64)) * m_pos,
            "a_absmass": built["a"].to(torch.float64).abs() * p * m_cand,
            "b_absmass": built["b"].to(torch.float64).abs() * p * m_cand,
            "intervention": inter,
            "tag_intervention": inter * tag.to(torch.float64),
            "mass_error": pos_err,
        }
        for key in ("shuffled_moved", "a_only_moved", "b_only_moved"):
            if key in built:
                cols[key] = built[key].to(torch.float64) * m_pos
        if d_on is not None:
            cols["d_on"] = d_on.detach().to(torch.float64) * live
            cols["d_base"] = d_base.detach().to(torch.float64) * live
        if student_logprob is not None:
            # dKL = -sum_S p_s log w + p_s,tail log Z, which telescopes to
            # log Z - <c>_{p_s}. Read off log_w rather than c so the clamp and the
            # cold-start no-op are already in it: this is what the loss saw, not
            # what the tilt asked for.
            p_s = student_logprob.detach().to(torch.float64).exp()
            tail_s = (1.0 - p_s.sum(dim=-1)).clamp(min=0.0)
            dkl = -(p_s * built["log_w"].to(torch.float64)).sum(dim=-1) + tail_s * log_z
            cols["dkl"] = dkl * m_pos
            cols["abs_dkl"] = dkl.abs() * m_pos
            cols["dkl_sq"] = dkl * dkl * m_pos
            cols["student_support_mass"] = p_s.sum(dim=-1) * m_pos
            if d_on is not None:
                k = d_on.detach().to(torch.float64) * live
                cols["d_on_sq"] = k * k
                cols["dkl_d_on"] = dkl * k
        if on_logprob is not None:
            # Bucketed entropy, the loss's own partition: every support candidate
            # on its own and the whole tail as one. p_tilde = w p on the support
            # and tail/Z outside it, so -sum p log p is taken twice and
            # subtracted. The tail's own term is dropped where the tail is empty,
            # which is 0 log 0 = 0 and not a floor on the mechanism.
            lp = on_logprob.detach().to(torch.float64)
            pw = w * p
            tail64 = built["tail"].to(torch.float64)
            tail_t = tail64 * built["inv_z"].to(torch.float64)
            ent_on = -(p * lp).sum(dim=-1)
            ent_tilt = -(pw * (lp + built["log_w"].to(torch.float64))).sum(dim=-1)
            has_tail = tail64 > 0
            ent_on = ent_on - torch.where(
                has_tail, tail64 * tail64.clamp(min=_LOG_FLOOR).log(), torch.zeros_like(tail64))
            ent_tilt = ent_tilt - torch.where(
                has_tail, tail_t * tail_t.clamp(min=_LOG_FLOOR).log(), torch.zeros_like(tail_t))
            cols["ent_delta"] = (ent_tilt - ent_on) * m_pos
        branch = built["branch"]
        for i in range(4):
            sel = (branch == i).to(torch.float64) * m_cand
            cols[f"branch{i}_cand"] = sel
            cols[f"branch{i}_mass"] = p * sel

        flat = {k: v.sum() for k, v in cols.items()}
        for k, v in flat.items():
            self.sums[0, self._col(k)] += v
        if task_ids is not None:
            t = task_ids.reshape(-1).to(torch.long)
            for k, v in cols.items():
                per_row = v.sum(dim=tuple(range(1, v.dim())))
                known = t >= 0
                self.sums[:, self._col(k)].index_add_(
                    0, t.clamp(min=0)[known] + 1, per_row[known]
                )
        abs_log_w = built["log_w"].to(torch.float64).abs()
        self.max_log_w = torch.maximum(
            self.max_log_w, (abs_log_w * m_cand).max().reshape(1).to(torch.float64))
        self.max_log_z = torch.maximum(
            self.max_log_z, (log_z.abs() * m_pos).max().reshape(1).to(torch.float64))
        self.max_mass_error = torch.maximum(self.max_mass_error,
                                            pos_err.max().reshape(1))

    def all_reduce(self) -> None:
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        torch.distributed.all_reduce(self.sums, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(self.max_log_w, op=torch.distributed.ReduceOp.MAX)
        torch.distributed.all_reduce(self.max_log_z, op=torch.distributed.ReduceOp.MAX)
        torch.distributed.all_reduce(self.max_mass_error, op=torch.distributed.ReduceOp.MAX)

    def metrics(self, *, task_names) -> dict:
        s = self.sums.detach().cpu()
        out = {}
        names = ["__pooled__"] + list(task_names or [])
        for scope in range(min(self.n_scopes, len(names))):
            head = "target" if scope == 0 else f"target/{names[scope]}"
            row = s[scope]
            g = lambda k: float(row[self._col(k)])
            n_pos, n_cand, mass = max(g("n_pos"), 1e-12), max(g("n_cand"), 1e-12), max(g("mass"), 1e-12)
            live = max(g("live_n"), 1e-12)
            out[f"{head}/tv"] = g("moved") / n_pos
            out[f"{head}/live_frac"] = g("live_n") / n_pos
            # The head's tax, per live position. Zero under the exchange this
            # replaced, and the first thing to read if the arm starts flattening
            # the teacher instead of tilting it.
            out[f"{head}/log_z_mean"] = g("log_z") / live
            # What the arm did to the loss, in nats. Compare abs_dkl_mean against
            # the predecessor weighting arm's per-position magnitude -- that
            # comparison is the whole point of the revision.
            out[f"{head}/dkl_mean"] = g("dkl") / n_pos
            out[f"{head}/abs_dkl_mean"] = g("abs_dkl") / n_pos
            out[f"{head}/abs_dkl_live_mean"] = g("abs_dkl") / live
            # Is the arm FOCAL -- does it push where the teacher KL already is?
            # The audit put the same question to the predecessor's weight and got
            # Spearman +0.003, which is what killed the focal reading of it. The
            # offline estimate for this arm is +0.067 (Pearson +0.023), i.e. also
            # flat; a run that comes out materially above that is doing something
            # the offline events did not predict.
            n_live = g("live_n")
            if n_live > 1:
                sd, sk = g("dkl"), g("d_on")
                vd = n_live * g("dkl_sq") - sd * sd
                vk = n_live * g("d_on_sq") - sk * sk
                if vd > 0 and vk > 0:
                    out[f"{head}/dkl_kl_corr"] = (
                        (n_live * g("dkl_d_on") - sd * sk) / ((vd * vk) ** 0.5)
                    )
            # Entropy is the only channel that tracked the 150-step accuracy gain
            # (Spearman +1.00). Negative = the target is sharper than the teacher.
            out[f"{head}/entropy_delta"] = g("ent_delta") / n_pos
            # What the top-k support covers, for each of the two models whose mass
            # the metrics above divide by.
            out[f"{head}/support_mass/student"] = g("student_support_mass") / n_pos
            out[f"{head}/support_mass/teacher"] = g("teacher_support_mass") / n_pos
            out[f"{head}/acted_cand_frac"] = g("acted_cand") / n_cand
            out[f"{head}/acted_mass_frac"] = g("acted_mass") / mass
            ab = g("a_absmass") + g("b_absmass")
            if ab > 0:
                out[f"{head}/channel/a_share"] = g("a_absmass") / ab
                out[f"{head}/channel/b_share"] = g("b_absmass") / ab
            inter = g("intervention")
            if inter > 0:
                out[f"{head}/tag_share"] = g("tag_intervention") / inter    # G3
            if g("moved") > 0 and g("shuffled_moved") >= 0 and scope == 0:
                out[f"{head}/shuffled_tv_ratio"] = g("shuffled_moved") / g("moved")  # G1
            for key, col in (("a_only_tv", "a_only_moved"), ("b_only_tv", "b_only_moved")):
                if g(col) > 0:
                    out[f"{head}/channel/{key}"] = g(col) / n_pos
            if g("d_base") > 0:
                out[f"{head}/acted_novelty"] = 1.0 - g("d_on") / g("d_base")  # G2
            # A rate, not an alarm: the clamp is the mechanism's only cap. READ
            # THE MASS RATIO. The count ratio understates saturation 1.9-3.1x
            # because the clamped candidates are the large ones, and it is the
            # mass ratio that says whether the continuous signal has collapsed
            # back to a flat +-clamp -- which is what the section 2 construction
            # exists to avoid, and what retired the (2.148, 3.0) setting.
            out[f"{head}/clamped_per_step"] = g("clamped")
            if g("acted_cand") > 0:
                out[f"{head}/clamped_frac_of_acted"] = g("clamped") / g("acted_cand")
            if g("abs_c_acted") > 0:
                out[f"{head}/clamped_mass_frac"] = g("abs_c_clamped") / g("abs_c_acted")
            for i, name in enumerate(BRANCHES):
                out[f"{head}/branch/{name}/cand_frac"] = g(f"branch{i}_cand") / n_cand
                out[f"{head}/branch/{name}/mass_frac"] = g(f"branch{i}_mass") / mass
        out["target/max_abs_log_w"] = float(self.max_log_w.item())            # G4
        out["target/max_abs_log_z"] = float(self.max_log_z.item())
        # sum p_tilde = 1, which is construction, so this is an assertion.
        out["target/mass_error_max"] = float(self.max_mass_error.item())
        return out
