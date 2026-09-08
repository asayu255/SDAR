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

"""What a per-task OPD coefficient did, measured while the arm runs.

The coefficients b_j are calibrated OFFLINE, on one checkpoint, from
PARAMETER-space gradients (``scripts/opd_cross_effect_qp.py``). Everything here
is LOGIT-space, on this step's data, and the two are related by the model's
Jacobian J:

    logit-space inner product      u_R . u_D
    parameter-space inner product  u_R . J J^T u_D

so no quantity below is the calibrated one measured again, and none of them is
guaranteed to reproduce a calibrated value even on the calibration data. What
they are is a reading of the term the optimizer actually took, on every step,
for the price of one all-reduce.

**1. Did b land, and where did the term go.** ``kl_ratio_eff_base`` and
``push_l2_ratio_eff_base`` are eff/base on the same tokens of the same forward,
so each equals b_task when the coefficient reached the loss -- a wiring check
that reads the loss rather than restating the config. ``kl_share_*`` say where
the term went; they are NOT b_task, because the denominator moves with every
task and a uniform amplification leaves both shares unchanged.

**2. How much of each signal there was.** ``adv_zero_frac`` is the token share
with no advantage; ``pg_live_frac`` is the share where the policy gradient's
derivative is actually non-zero, which is smaller because the PPO clip zeroes
positions with a live advantage. Splitting the KL by advantage
(``kl_mean_adv_zero`` / ``kl_mean_adv_live``) shows where the distillation term
sits relative to the reward's support. It is NOT a separation of two causal
paths: a live advantage can still be clipped to zero gradient, and the shared
parameters are updated from every other token and task in the batch.

**3. The local relation between policy and teacher.** ``pg_dot_mean`` /
``pg_cos_mean`` are the inner product of the two descent directions on the SAME
token's logits, clip included (see :func:`opd_pg_alignment_terms`). Read it as
an auxiliary, logit-space alignment. It is not the diagonal of the
parameter-space cross-effect matrix: the sign is not preserved through J J^T,
and it cannot carry the off-diagonal at all, because a token belongs to one
task.

**Control inputs, kept apart from the diagnostics above.** The metrics named so
far are for reading; ``ctl_R``, ``ctl_X``, ``ctl_D`` are for feeding a rule that
sets the coefficient. A control rule needs its three inputs to be commensurable,
and the diagnostics are not:

* **One population.** ``ctl_*`` run over response tokens whose sampled id is
  inside the KL's support, and nothing else. The support restriction is forced
  -- ``u_R . u_D`` needs ``u_D`` at the sampled id, which only the support
  carries -- but the diagnostics additionally drop clipped tokens and zero-norm
  tokens, which left R and X on a subset (``align_cover`` measured 0.23-0.53)
  while D ran over everything. A ratio built from those is a subset numerator
  over a full-set denominator.
* **One weighting.** All three carry ``basis^2``, the per-task row weight the
  loss applies, squared because both directions carry it linearly. It does NOT
  divide out of the ratios: a duplicated row gets weight 0
  (``task_loss_weights.py``), so ``basis`` is not constant within a task.
* **Beta in, b out.** ``u_D`` is the gradient of ``beta * L_OPD``. ``ctl_a =
  X/sqrt(RD)`` is invariant to beta, but ``ctl_pushback_frac = -X/R`` is linear
  in it, so any threshold on that would move by 1/beta. ``b`` is excluded
  because a rule reading its own output back is a feedback loop, not a
  measurement.
* **Aggregates, not a mean of per-token cosines.** ``ctl_a`` is
  ``X/sqrt(R D)`` over the summed quantities, so a token with a weak signal does
  not get the same vote as a strong one -- and a zero-norm token contributes 0 to
  all three sums instead of needing a special case.

What they still cannot do is reach across tasks: ``r_i . P d_j`` for i != j is
``u_R,i . J_i P J_j^T u_D,j``, and no per-token quantity carries the Jacobians.
A rule built on these balances each task's own reward against its own teacher; it
does not resolve cross-task interference, and should not be described as doing so.

Sign convention throughout is DESCENT, matching :func:`opd_logit_push`:
positive means the objective is pushing that logit UP. So ``pg_dot > 0`` is
agreement between reward and teacher, and ``pg_dot < 0`` is conflict.
"""

from __future__ import annotations

import torch

__all__ = ["opd_pg_alignment_terms", "OpdTaskDiagStats"]


def opd_pg_alignment_terms(
    *,
    student_topk_logprob: torch.Tensor,
    teacher_topk_logprob: torch.Tensor,
    teacher_kl: torch.Tensor,
    topk_ids: torch.Tensor,
    response_ids: torch.Tensor,
    log_prob: torch.Tensor,
    pg_grad_coef: torch.Tensor | None,
    opd_coef: float = 1.0,
) -> dict:
    """Per-token logit-space overlap between the OPD push and the PG push.

    With ``p`` the student's distribution, ``D`` the per-token KL and
    ``f(v) = log p(v) - log p_teacher(v)``, the descent direction of the KL on
    logit ``v`` is

        g_opd(v) = p(v) * (D - f(v))

    (derived in :func:`opd_logit_push`, checked there against autograd). The
    policy gradient's descent direction is

        g_pg(v) = -dL_pg/dlog p * (delta_{v,a} - p(v))

    so their inner product over the vocabulary collapses to two support sums:

        <g_pg, g_opd> = -dL_pg/dlog p * [ g_opd(a) - sum_v p(v) g_opd(v) ]

    **The PG coefficient is passed in, not rebuilt from A and the ratio.**
    ``-A*rho`` is the coefficient only outside the PPO clip; inside a bound clip
    branch, or the dual-clip branch, the true derivative is exactly ZERO and a
    metric using ``-A*rho`` reports a position the objective has stopped pushing
    as a full-magnitude conflict. :func:`policy_loss_gradient_coef` is that
    derivative in closed form, differentiated from the loss and checked against
    autograd, so the caller hands it over rather than this file keeping a second
    opinion about what the objective is.

    **Two approximations, both on the tail, both named in the metric keys.** The
    sums run over the KL's own top-k support only: outside it the KL keeps a
    single lumped tail bucket whose per-symbol split is not represented, and the
    terms it drops carry p(v)^2 with p(v) tiny. A token whose SAMPLED id fell
    outside the support has no ``g_opd(a)``; it is excluded, and the share that
    survives is reported as ``align_cover``. Which model's top-k the support is
    is the caller's business (student-indexed or teacher-indexed) -- this
    function reads whatever ids it is given.

    Returns per-token ``(bs, response_length)`` tensors. Everything is detached:
    this is a measurement, never a term in the loss.
    """
    # fp32 at least, and fp64 when the caller is already there. bf16 log-probs
    # differenced in bf16 would lose the small D - f that the whole sign of the
    # metric rests on.
    dt = torch.promote_types(student_topk_logprob.dtype, torch.float32)
    lp_s = student_topk_logprob.detach().to(dt)
    lp_t = teacher_topk_logprob.detach().to(dt)
    d = teacher_kl.detach().to(dt).unsqueeze(-1)

    p_s = lp_s.exp()
    # g_opd over the support. Same expression as opd_logit_push's g0 at coef=1;
    # the coefficient is deliberately left out so the norm is a b = 1 basis and
    # the row weights the loss applies can be put on it afterwards.
    g_opd = p_s * (d - (lp_s - lp_t))
    opd_sq = (g_opd * g_opd).sum(dim=-1)
    # The probability the support does NOT cover. align_cover says what share of
    # tokens the metric is defined on; this says how much of the distribution the
    # sums drop, which is the quantity a claim about the tail approximation needs.
    tail_mass = (1.0 - p_s.sum(dim=-1)).clamp(min=0.0, max=1.0)

    out = {
        "opd_sq": opd_sq,
        "dot": torch.zeros_like(opd_sq),
        "cos": torch.zeros_like(opd_sq),
        "pg_sq": torch.zeros_like(opd_sq),
        "align_mask": torch.zeros_like(opd_sq),
        "pg_live": torch.zeros_like(opd_sq),
        "tail_mass": tail_mass,
        # --- the CONTROL inputs, kept separate from the diagnostics above ----
        # One population, one weighting, beta folded in, b deliberately absent.
        # See the module docstring's "control inputs" note for why each of those
        # four is not a matter of taste.
        "ctl_mask": torch.zeros_like(opd_sq),
        "ctl_rr": torch.zeros_like(opd_sq),
        "ctl_rd": torch.zeros_like(opd_sq),
        "ctl_dd": torch.zeros_like(opd_sq),
    }
    if pg_grad_coef is None or log_prob is None or topk_ids is None:
        # No policy gradient to overlap with (pure distillation), or no support
        # ids to locate the sampled token in. Reporting a zero cosine would read
        # as "measured, and they are orthogonal"; the caller sees
        # align_tokens = 0 instead, and the budget columns are unaffected.
        return out

    hit = topk_ids == response_ids.unsqueeze(-1)
    in_support = hit.any(dim=-1)
    hit_f = hit.to(p_s.dtype)
    # Both gathered from the SAME arrays the KL was built from, so a rewritten
    # teacher (target arm) stays self-consistent instead of being mixed with a
    # separately cached sampled-token log-prob.
    g_opd_a = (hit_f * g_opd).sum(dim=-1)
    p_a = (hit_f * p_s).sum(dim=-1)
    p_dot_g = (p_s * g_opd).sum(dim=-1)
    p_sq = (p_s * p_s).sum(dim=-1)

    # Descent convention: positive means the objective pushes this logit UP.
    pg_scale = -pg_grad_coef.detach().to(dt)
    pg_live = pg_scale != 0

    dot = pg_scale * (g_opd_a - p_dot_g)
    # ||g_pg||^2 = (dL/dlogp)^2 * sum_v (delta_{v,a} - p(v))^2
    #            = (dL/dlogp)^2 (1 - 2 p_a + sum_v p^2)
    pg_sq = pg_scale * pg_scale * (1.0 - 2.0 * p_a + p_sq).clamp(min=0.0)

    # A cosine needs both norms. Where either is zero it is UNDEFINED, and
    # averaging a zero in its place would pull the mean toward "orthogonal" with
    # positions that carry no direction at all.
    live = in_support & pg_live & (pg_sq > 0) & (opd_sq > 0)
    denom = (pg_sq.clamp(min=0).sqrt() * opd_sq.clamp(min=0).sqrt()).clamp(min=1e-30)
    zero = torch.zeros_like(dot)

    out["dot"] = torch.where(live, dot, zero)
    out["cos"] = torch.where(live, dot / denom, zero)
    out["pg_sq"] = torch.where(live, pg_sq, zero)
    out["align_mask"] = live.to(opd_sq.dtype)
    out["pg_live"] = pg_live.to(opd_sq.dtype)

    # ---- control inputs -------------------------------------------------
    # POPULATION: response tokens whose sampled id is inside the support, and
    # nothing else. The support restriction is forced, not chosen -- u_R . u_D
    # needs u_D at the sampled id, which only the support carries. What is NOT
    # required is pg_live or a non-zero norm: a token the PPO clip has zeroed has
    # u_R = 0, which is a defined value that contributes 0 to R and X and its
    # real amount to D. Dropping it would make R, X and D three different
    # populations again, which is the defect this exists to fix.
    ctl = in_support
    # BETA IS IN, b IS OUT. a = X/sqrt(R D) happens to be invariant to beta, but
    # the pushback fraction -X/R is linear in it, so leaving beta out would move
    # any threshold on that by a factor of 1/beta (100x at beta = 0.01). b is out
    # because a controller that reads its own output back is a feedback loop, not
    # a measurement.
    out["ctl_mask"] = ctl.to(opd_sq.dtype)
    out["ctl_rr"] = torch.where(ctl, pg_sq, zero)
    out["ctl_rd"] = torch.where(ctl, float(opd_coef) * dot, zero)
    out["ctl_dd"] = torch.where(ctl, float(opd_coef) ** 2 * opd_sq, zero)
    return out


# Order matters: the buffer is a single tensor so the whole table costs one
# all-reduce, and the readers below index it by name through this tuple.
_COLS = (
    "n_tok",
    "n_tok_adv_zero",
    "n_pg_live",
    "kl_sum",
    "kl_sum_adv_zero",
    "kl_base_sum",
    "kl_eff_sum",
    # The OPD push's squared norm carries the ROW WEIGHT the loss applies, so
    # base and eff are the norms of the term that is actually in the objective.
    # Squared, because a norm's square takes the weight squared -- putting the
    # weight on linearly here is how a budget ratio comes out as the square root
    # of the one the loss took.
    "push_sq_base_sum",
    "push_sq_eff_sum",
    "n_align",
    "dot_sum",
    "cos_sum",
    "dot_neg",
    "pg_sq_sum",
    # The control inputs: R, X, D on ONE population with ONE weighting, plus
    # that population's size and the probability mass the support drops.
    "ctl_n",
    "ctl_rr_sum",
    "ctl_rd_sum",
    "ctl_dd_sum",
    "tail_mass_sum",
)
_IDX = {name: i for i, name in enumerate(_COLS)}


class OpdTaskDiagStats:
    """Per-task sums for the OPD coefficient arm, pooled over a whole update.

    Built on the CONFIG alone and reduced unconditionally, because
    :meth:`rows` runs a collective: a rank that skipped it because its batch
    held no rows of some task would hang the others. Accumulation is one matmul
    per micro-batch against a row/task one-hot, so nothing here reads a device
    value back to the host until the single read in :meth:`rows`.
    """

    def __init__(self, n_tasks: int, device):
        self.n_tasks = int(n_tasks)
        self.buf = torch.zeros(self.n_tasks, len(_COLS), dtype=torch.float64, device=device)

    def update(
        self,
        *,
        task_ids: torch.Tensor,
        response_mask: torch.Tensor,
        teacher_kl: torch.Tensor,
        advantages: torch.Tensor | None,
        terms: dict | None,
        row_basis: torch.Tensor | None,
        row_coef: torch.Tensor | None,
    ) -> None:
        """Fold one micro-batch in.

        ``row_basis`` is the per-row factor the loss already applies at b = 1
        (the per-task loss weight, or None for the plain token mean), and
        ``row_coef`` is b. The pair is what makes ``kl_share_eff`` and
        ``kl_share_base`` two readings of the same batch rather than two batches.
        """
        with torch.no_grad():
            mask = response_mask.to(torch.float32)
            kl = teacher_kl.detach().to(torch.float32) * mask
            n_row, _ = mask.shape

            if advantages is None:
                adv_zero = torch.zeros_like(mask)
            else:
                adv_zero = (advantages.detach() == 0).to(torch.float32) * mask

            basis = (
                torch.ones(n_row, device=mask.device, dtype=torch.float32)
                if row_basis is None
                else row_basis.detach().reshape(-1).to(torch.float32)
            )
            coef = (
                torch.ones(n_row, device=mask.device, dtype=torch.float32)
                if row_coef is None
                else row_coef.detach().reshape(-1).to(torch.float32)
            )
            row_kl = kl.sum(dim=-1)

            per_row = torch.zeros(n_row, len(_COLS), device=mask.device, dtype=torch.float32)
            per_row[:, _IDX["n_tok"]] = mask.sum(dim=-1)
            per_row[:, _IDX["n_tok_adv_zero"]] = adv_zero.sum(dim=-1)
            per_row[:, _IDX["kl_sum"]] = row_kl
            per_row[:, _IDX["kl_sum_adv_zero"]] = (kl * adv_zero).sum(dim=-1)
            per_row[:, _IDX["kl_base_sum"]] = row_kl * basis
            per_row[:, _IDX["kl_eff_sum"]] = row_kl * basis * coef
            if terms is not None:
                live = terms["align_mask"] * mask
                opd_sq = (terms["opd_sq"] * mask).sum(dim=-1)
                per_row[:, _IDX["push_sq_base_sum"]] = opd_sq * basis * basis
                per_row[:, _IDX["push_sq_eff_sum"]] = opd_sq * (basis * coef) ** 2
                per_row[:, _IDX["n_pg_live"]] = (terms["pg_live"] * mask).sum(dim=-1)
                per_row[:, _IDX["n_align"]] = live.sum(dim=-1)
                per_row[:, _IDX["dot_sum"]] = (terms["dot"] * live).sum(dim=-1)
                per_row[:, _IDX["cos_sum"]] = (terms["cos"] * live).sum(dim=-1)
                per_row[:, _IDX["dot_neg"]] = ((terms["dot"] < 0).to(mask.dtype) * live).sum(dim=-1)
                per_row[:, _IDX["pg_sq_sum"]] = (terms["pg_sq"] * live).sum(dim=-1)
                # ONE population, ONE weighting, for all three. basis^2 because
                # both u_R and u_D carry the row weight linearly, so a squared
                # norm and an inner product both take it squared. It does not
                # cancel out of the ratios: a duplicated row carries weight 0
                # (task_loss_weights.py), so basis is not constant within a task.
                ctl = terms["ctl_mask"] * mask
                b2 = (basis * basis).unsqueeze(-1)
                per_row[:, _IDX["ctl_n"]] = ctl.sum(dim=-1)
                per_row[:, _IDX["ctl_rr_sum"]] = (terms["ctl_rr"] * ctl * b2).sum(dim=-1)
                per_row[:, _IDX["ctl_rd_sum"]] = (terms["ctl_rd"] * ctl * b2).sum(dim=-1)
                per_row[:, _IDX["ctl_dd_sum"]] = (terms["ctl_dd"] * ctl * b2).sum(dim=-1)
                per_row[:, _IDX["tail_mass_sum"]] = (terms["tail_mass"] * mask).sum(dim=-1)

            flat = task_ids.reshape(-1)
            flat = flat.round().to(torch.long) if flat.is_floating_point() else flat.to(torch.long)
            # Padding rows carry a negative id and must land nowhere. clamp puts
            # them on task 0, so the one-hot is zeroed for them instead.
            onehot = torch.nn.functional.one_hot(flat.clamp(min=0), num_classes=self.n_tasks)
            onehot = onehot.to(per_row.dtype) * (flat >= 0).to(per_row.dtype).unsqueeze(-1)
            self.buf += (onehot.transpose(0, 1) @ per_row).to(torch.float64)

    def rows(self, task_names) -> dict:
        """One all-reduce, one host read, and the ratios the sums are for.

        Ratios are formed AFTER the reduction, never per micro-batch and never
        per rank: a mean of per-rank shares is not the share, and a mean of
        per-micro-batch shares weighs a short micro-batch like a full one.

        The coefficient is NOT taken from the config here. Every ratio below is
        eff/base on the same tokens of the same forward, so ``kl_ratio`` and
        ``push_l2_ratio`` are what the loss did with b, not a restatement of
        what the config asked for -- which is what makes them a wiring check.
        """
        buf = self.buf
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            buf = buf.clone()
            torch.distributed.all_reduce(buf, op=torch.distributed.ReduceOp.SUM)
        table = buf.tolist()

        def _safe(num, den):
            return float(num) / float(den) if den > 0 else 0.0

        out = {}
        l2_base, l2_eff = {}, {}
        for tid, name in enumerate(task_names[: self.n_tasks]):
            row = table[tid]
            n_tok = row[_IDX["n_tok"]]
            if n_tok <= 0:
                continue
            n_zero = row[_IDX["n_tok_adv_zero"]]
            n_live = n_tok - n_zero
            n_pg = row[_IDX["n_pg_live"]]
            kl_zero = row[_IDX["kl_sum_adv_zero"]]
            n_align = row[_IDX["n_align"]]
            base = row[_IDX["push_sq_base_sum"]] ** 0.5
            eff = row[_IDX["push_sq_eff_sum"]] ** 0.5
            l2_base[name], l2_eff[name] = base, eff

            out[f"actor/opd_diag/tokens/{name}"] = n_tok
            out[f"actor/opd_diag/kl_mean/{name}"] = _safe(row[_IDX["kl_sum"]], n_tok)
            # ---- did b actually reach the loss? Both must equal b_task. ----
            out[f"actor/opd_diag/kl_ratio_eff_base/{name}"] = _safe(
                row[_IDX["kl_eff_sum"]], row[_IDX["kl_base_sum"]]
            )
            out[f"actor/opd_diag/push_l2_ratio_eff_base/{name}"] = _safe(eff, base)
            out[f"actor/opd_diag/push_l2_logit_base/{name}"] = base
            out[f"actor/opd_diag/push_l2_logit_eff/{name}"] = eff
            # ---- how much signal each side had -----------------------------
            out[f"actor/opd_diag/adv_zero_frac/{name}"] = _safe(n_zero, n_tok)
            out[f"actor/opd_diag/pg_live_frac/{name}"] = _safe(n_pg, n_tok)
            out[f"actor/opd_diag/kl_mean_adv_zero/{name}"] = _safe(kl_zero, n_zero)
            out[f"actor/opd_diag/kl_mean_adv_live/{name}"] = _safe(row[_IDX["kl_sum"]] - kl_zero, n_live)
            # ---- the logit-space alignment, on its own population ----------
            out[f"actor/opd_diag/align_tokens/{name}"] = n_align
            out[f"actor/opd_diag/align_cover/{name}"] = _safe(n_align, n_pg)
            out[f"actor/opd_diag/pg_dot_mean/{name}"] = _safe(row[_IDX["dot_sum"]], n_align)
            out[f"actor/opd_diag/pg_cos_mean/{name}"] = _safe(row[_IDX["cos_sum"]], n_align)
            out[f"actor/opd_diag/pg_dot_neg_frac/{name}"] = _safe(row[_IDX["dot_neg"]], n_align)
            out[f"actor/opd_diag/pg_push_rms_logit/{name}"] = _safe(row[_IDX["pg_sq_sum"]], n_align) ** 0.5
            # The absolute contributions, not only the shares: two arms can hold
            # the same shares while the whole teacher-KL term moves, and the
            # share alone would report that as no change.
            out[f"actor/opd_diag/kl_sum_base/{name}"] = row[_IDX["kl_base_sum"]]
            out[f"actor/opd_diag/kl_sum_eff/{name}"] = row[_IDX["kl_eff_sum"]]

            # ---- the control inputs ------------------------------------
            # R, X and D as aggregates, NOT as a mean of per-token cosines: a
            # token with a weak signal must not get the same vote as a strong
            # one. The aggregate form also removes the per-token zero-norm
            # special case -- a zero-norm token adds 0 to all three sums, so
            # nothing divides by zero.
            R, X, D = (row[_IDX[k]] for k in ("ctl_rr_sum", "ctl_rd_sum", "ctl_dd_sum"))
            n_ctl = row[_IDX["ctl_n"]]
            out[f"actor/opd_diag/ctl_R/{name}"] = R
            out[f"actor/opd_diag/ctl_X/{name}"] = X
            out[f"actor/opd_diag/ctl_D/{name}"] = D
            out[f"actor/opd_diag/ctl_tokens/{name}"] = n_ctl
            out[f"actor/opd_diag/ctl_cover/{name}"] = _safe(n_ctl, n_tok)
            out[f"actor/opd_diag/ctl_a/{name}"] = (
                X / ((R * D) ** 0.5) if R > 0 and D > 0 else 0.0
            )
            # -X/R: the share of the reward's own descent that the teacher
            # cancels at b = 1. This is the quantity a pushback limit thresholds
            # (b <= eps / (-X/R)), which is why beta is inside X and b is not.
            out[f"actor/opd_diag/ctl_pushback_frac/{name}"] = _safe(-X, R)
            # What the support does not cover, so a claim about the tail
            # approximation rests on mass rather than on token counts.
            out[f"actor/opd_diag/tail_mass_mean/{name}"] = _safe(row[_IDX["tail_mass_sum"]], n_tok)

        base_total = sum(table[t][_IDX["kl_base_sum"]] for t in range(self.n_tasks))
        eff_total = sum(table[t][_IDX["kl_eff_sum"]] for t in range(self.n_tasks))
        for tid, name in enumerate(task_names[: self.n_tasks]):
            if table[tid][_IDX["n_tok"]] <= 0:
                continue
            # NOT b_task: the denominator moves with every task. A uniform
            # amplification leaves both shares unchanged. Read these for WHERE
            # the term went, and kl_ratio_eff_base above for whether b landed.
            out[f"actor/opd_diag/kl_share_base/{name}"] = _safe(table[tid][_IDX["kl_base_sum"]], base_total)
            out[f"actor/opd_diag/kl_share_eff/{name}"] = _safe(table[tid][_IDX["kl_eff_sum"]], eff_total)

        # sum_j b_j L_j / sum_j L_j in LOGIT space, on this step's data, with the
        # loss's own row weights. Under a uniform arm it is exactly that arm's b.
        #
        # It is NOT the quantity the offline rule set to 1.000: that budget was
        # computed on PARAMETER-space gradient norms, and the two are related by
        # the model's Jacobian, so nothing makes this read 1 even on the
        # calibration data. Read it as "how much OPD gradient, in logit space,
        # the coefficients added or removed this step".
        if l2_base:
            plain = sum(l2_base.values())
            scaled = sum(l2_eff.values())
            out["actor/opd_diag/budget_ratio_logit"] = _safe(scaled, plain)
        return out
