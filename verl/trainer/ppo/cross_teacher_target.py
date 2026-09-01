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
nats} span ``exp(c)`` from 1.52 to 9.82. This module implements the RMS reading
(the conservative end, ``exp(c) ~ 2.9``) and ``exponent_scale`` exists to reach
the other; it is the one knob here and the design document says so.
"""
from typing import Optional

import torch

# Numerical floors only. Neither is a threshold on the mechanism: the geometric
# mean's floor is applied inside a log whose result is overwritten wherever an
# exact zero was present, and the capacity floors gate positions that are left
# at w = 1 rather than scaled by something derived from a divide-by-zero.
_LOG_FLOOR = 1e-30
_CAPACITY_FLOOR = 1e-12
# exp() guard. Chosen so float64 cannot overflow and float32 round-trips; a
# shift this large has already broken an assumption upstream, so the count is
# reported rather than the value silently truncated.
_EXPONENT_CLAMP = 30.0

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


def capacity_limited_weight(*, c: torch.Tensor, p_on: torch.Tensor,
                            row_available: Optional[torch.Tensor] = None) -> dict:
    """Per-position mass-conserving exchange. Returns ``w`` with ``sum w*p == sum p``.

    Args:
        c: (bs, resp, k) the tilt exponent.
        p_on: (bs, resp, k) the on-task teacher's probability on the support.
        row_available: (bs,) bool, false for a row with no usable sigma yet. Those
            rows get ``w = 1`` -- the cold start is a no-op rather than a guess.

    Returns:
        ``{"w", "moved", "throttle_up", "throttle_down", "clamped"}``.
        ``moved`` is (bs, resp): the mass ``T`` actually exchanged at that
        position, which is also that position's total variation.

    THE EXCHANGE IS LIMITED BY THE SMALLER SIDE, and that is the only bound in
    the mechanism. The down side can supply at most ``R_D = sum (1 - e^c) p``,
    the up side can absorb at most ``A_U = sum (e^c - 1) p``, and ``T = min``.
    Scaling each side to move exactly ``T`` gives four properties by
    construction, all of them tested rather than argued:

    * mass is conserved exactly, so ``Z = 1`` is an identity and not a metric;
    * candidates at ``c = 0`` -- the head -- keep ``w = 1``. The predecessor's
      renormalisation took 46-55% of its amplification out of the token the
      teacher was most confident about, and that is what this removes;
    * ``w <= 1`` on the down side and ``w >= 1`` on the up side ALWAYS, because
      both scale factors are in (0, 1]. "Agreed against, therefore attenuated"
      cannot invert;
    * ``w`` stays inside ``e^c``, so the teachers' own volume is the cap and no
      separate clip is needed.

    The first draft of this released the down side's mass and handed it to the up
    side in proportion to ``g * p``. It conserved mass just as exactly and was
    unusable: with the up side sometimes a single denormal candidate, the p99 of
    ``max w`` was 2,331 and the maximum 2.4e10. The failure is a capacity
    mismatch, not an allocation one, which is why the fix is ``min`` and not a
    different set of weights.
    """
    c64 = c.to(torch.float64).clamp(min=-_EXPONENT_CLAMP, max=_EXPONENT_CLAMP)
    clamped = (c.to(torch.float64).abs() > _EXPONENT_CLAMP)
    p = p_on.to(torch.float64)
    e = c64.exp()

    up, down = c64 > 0, c64 < 0
    supply = torch.where(down, (1.0 - e) * p, torch.zeros_like(p)).sum(dim=-1)
    demand = torch.where(up, (e - 1.0) * p, torch.zeros_like(p)).sum(dim=-1)
    moved = torch.minimum(supply, demand)

    live = (supply > _CAPACITY_FLOOR) & (demand > _CAPACITY_FLOOR) & (moved > 0)
    if row_available is not None:
        live = live & row_available.reshape(-1, 1).to(live.device)

    # Ratios are only read where live, but the divisor is floored anyway: an
    # unread inf still poisons the reductions the metrics take below.
    t_down = (moved / supply.clamp(min=_CAPACITY_FLOOR)).unsqueeze(-1)
    t_up = (moved / demand.clamp(min=_CAPACITY_FLOOR)).unsqueeze(-1)

    w = torch.ones_like(p)
    w = torch.where(down, 1.0 - t_down * (1.0 - e), w)
    w = torch.where(up, 1.0 + t_up * (e - 1.0), w)
    w = torch.where(live.unsqueeze(-1), w, torch.ones_like(p))

    return {
        "w": w,
        "moved": torch.where(live, moved, torch.zeros_like(moved)),
        "throttle_up": torch.where(live, t_up.squeeze(-1), torch.ones_like(moved)),
        "throttle_down": torch.where(live, t_down.squeeze(-1), torch.ones_like(moved)),
        "clamped": clamped,
        "live": live,
    }


def build_target(*, on_logprob: torch.Tensor, off_logprob: torch.Tensor,
                 base_logprob: torch.Tensor, diag: torch.Tensor,
                 diag_valid: torch.Tensor, task_ids: torch.Tensor,
                 off_plane_tasks: torch.Tensor, exponent_scale: float = 1.0,
                 shuffled_off_logprob: Optional[torch.Tensor] = None,
                 response_mask: Optional[torch.Tensor] = None) -> dict:
    """The whole chain: standardised shifts -> ``c`` -> ``w`` -> ``log p_tilde``.

    Args:
        on_logprob: (bs, resp, k) on-task teacher log-probs on the support.
        off_logprob: (bs, resp, k, n_off) the off-task teachers on the same ids.
        base_logprob: (bs, resp, k) the pre-RL policy on the same ids.
        diag, diag_valid: (n_tasks,) from ``CumulativePolicyShiftRMS.diagonal``.
        shuffled_off_logprob: the same off-task planes rolled within each row, from
            ``decorrelated_off_shifts``. When given, the counterfactual target is
            built too and its total variation is returned beside the live one --
            this is the abort gate G1, not an optional extra. The audit measured
            the predecessor's gate opening at 0.82 of its live rate on shuffled
            teachers; a mechanism that scores near that is moving mass for
            reasons that survive destroying the position correspondence.

    Returns:
        ``{"target_logprob", "w", "c", "branch", "moved", ...}``. The support's
        probabilities are rescaled; the tail is untouched and therefore still
        sums with them to one.

    The support is the ON-TASK teacher's top-k, which is not a compromise: the
    off-task teachers keep 98.4-99.3% of their own probability mass inside it
    (measured on all six ordered pairs), so widening it would buy coverage that
    is already there and pay for it in cache rows.
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
    built = capacity_limited_weight(c=tilt["c"], p_on=p_on, row_available=hat["row_available"])

    out = {
        "target_logprob": on_logprob + built["w"].to(on_logprob.dtype).clamp(min=_LOG_FLOOR).log(),
        "w": built["w"], "c": tilt["c"], "a": tilt["a"], "b": tilt["b"],
        "branch": tilt["branch"], "moved": built["moved"], "live": built["live"],
        "throttle_up": built["throttle_up"], "throttle_down": built["throttle_down"],
        "clamped": built["clamped"], "p_on": p_on,
        "consensus_volume": cons["L"], "consensus_sign": cons["s"],
        "row_available": hat["row_available"],
    }

    if shuffled_off_logprob is not None:
        sh_shifts = {"on": shifts["on"],
                     "off": (shuffled_off_logprob - base_logprob.unsqueeze(-1)).detach()}
        sh_hat = standardize_policy_shifts(
            shifts=sh_shifts, diag=diag, diag_valid=diag_valid,
            task_ids=task_ids, off_plane_tasks=off_plane_tasks,
        )
        sh_c = target_exponent(hat_on=sh_hat["on"],
                               consensus=off_task_consensus(sh_hat["off"]),
                               exponent_scale=exponent_scale)["c"]
        sh = capacity_limited_weight(c=sh_c, p_on=p_on, row_available=sh_hat["row_available"])
        out["shuffled_moved"] = sh["moved"]
        out["shuffled_c"] = sh_c
    return out
