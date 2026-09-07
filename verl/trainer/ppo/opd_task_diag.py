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
parameter-space gradients (``scripts/opd_cross_effect_qp.py``). Three things
that calibration cannot say are exactly the three this module measures on every
step of the run itself:

1. **What the arm actually reallocated.** The claim is that the rule holds a
   budget while moving its shares. The budget it holds is
   ``sum_j b_j ||d_j||`` on the CALIBRATION data; during training nothing
   guarantees it. ``push_l2_logit`` is a per-step, per-task stand-in for
   ``||d_j||`` -- the L2 norm of the OPD term's descent direction on the
   logits -- and ``budget_ratio_logit`` is the ratio the calibration set to
   1.000. Its drift away from 1 is the honest size of "preserved only on the
   calibration data".

2. **Whether the on-task conflict the coefficients respond to is still there
   at step 0-150.** The calibration is a step-300 measurement applied from
   step 0. In logit space the on-task part of it is exact and free: at one
   token, the reward pushes the sampled logit one way and the teacher pushes
   the whole support another, and ``pg_dot`` is their inner product. This is
   the DIAGONAL of the cross-effect matrix, not the off-diagonal the arm is
   premised on -- no per-token quantity can carry the off-diagonal, because a
   token belongs to one task. Read it as "is the teacher fighting this task's
   own reward", which is 40% of what makes the webshop column negative.

3. **Which of webshop's two paths moved.** Halving b_webshop both removes
   interference and halves a distillation signal, and the design cannot tell
   them apart from a success rate. It can from where the KL sits: on tokens
   whose advantage is zero the OPD term is the ONLY gradient (nothing to
   interfere with, so a change there is the distillation path), and on tokens
   with a live advantage both act. ``kl_mean_adv_zero`` / ``kl_mean_adv_live``
   split it.

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
    old_log_prob: torch.Tensor | None,
    advantages: torch.Tensor | None,
) -> dict:
    """Per-token logit-space push of the OPD term, and its overlap with the PG term.

    With ``p`` the student's distribution, ``D`` the per-token KL and
    ``f(v) = log p(v) - log p_teacher(v)``, the descent direction of the KL on
    logit ``v`` is

        g_opd(v) = p(v) * (D - f(v))

    (derived in :func:`opd_logit_push`, checked there against autograd), and the
    unclipped PPO surrogate ``-A * rho`` with ``rho = exp(log p(a) - log p_old(a))``
    gives

        g_pg(v) = A * rho * (delta_{v,a} - p(v))

    so their inner product over the vocabulary collapses to two support sums:

        <g_pg, g_opd> = A * rho * [ g_opd(a) - sum_v p(v) g_opd(v) ]

    **Three approximations, all on the tail and all named in the metric keys.**
    The sums run over the teacher's top-k support only: outside it the KL keeps
    a single lumped tail bucket, whose per-symbol split is not represented, and
    the terms it drops carry p(v)^2 with p(v) tiny. A token whose SAMPLED id
    fell outside the support has no ``g_opd(a)``; it is excluded, and the
    fraction that survives is reported as ``align_cover``. Clipping is ignored:
    where the PPO clip binds, the true PG gradient is zero and this over-counts.

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
    # the coefficient is deliberately left out so the norm below is a b = 1
    # basis and the arm's own b can be applied to it afterwards.
    g_opd = p_s * (d - (lp_s - lp_t))
    opd_sq = (g_opd * g_opd).sum(dim=-1)

    out = {
        "opd_sq": opd_sq,
        "dot": torch.zeros_like(opd_sq),
        "cos": torch.zeros_like(opd_sq),
        "pg_sq": torch.zeros_like(opd_sq),
        "align_mask": torch.zeros_like(opd_sq),
    }
    if advantages is None or old_log_prob is None or log_prob is None or topk_ids is None:
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

    adv = advantages.detach().to(dt)
    ratio = (log_prob.detach().to(dt) - old_log_prob.detach().to(dt)).exp()
    pg_scale = adv * ratio

    dot = pg_scale * (g_opd_a - p_dot_g)
    # ||g_pg||^2 = (A rho)^2 * sum_v (delta_{v,a} - p(v))^2 = (A rho)^2 (1 - 2 p_a + sum_v p^2)
    pg_sq = pg_scale * pg_scale * (1.0 - 2.0 * p_a + p_sq).clamp(min=0.0)

    live = in_support & (adv != 0)
    denom = (pg_sq.sqrt() * opd_sq.sqrt()).clamp(min=1e-20)
    cos = torch.where(live, dot / denom, torch.zeros_like(dot))

    out["dot"] = torch.where(live, dot, torch.zeros_like(dot))
    out["cos"] = cos
    out["pg_sq"] = torch.where(live, pg_sq, torch.zeros_like(pg_sq))
    out["align_mask"] = live.to(opd_sq.dtype)
    return out


# Order matters: the buffer is a single tensor so the whole table costs one
# all-reduce, and the readers below index it by name through this tuple.
_COLS = (
    "n_tok",
    "n_tok_adv_zero",
    "kl_sum",
    "kl_sum_adv_zero",
    "kl_base_sum",
    "kl_eff_sum",
    "push_sq_sum",
    "n_align",
    "dot_sum",
    "cos_sum",
    "dot_neg",
    "pg_sq_sum",
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
                per_row[:, _IDX["push_sq_sum"]] = (terms["opd_sq"] * mask).sum(dim=-1)
                per_row[:, _IDX["n_align"]] = live.sum(dim=-1)
                per_row[:, _IDX["dot_sum"]] = (terms["dot"] * live).sum(dim=-1)
                per_row[:, _IDX["cos_sum"]] = (terms["cos"] * live).sum(dim=-1)
                per_row[:, _IDX["dot_neg"]] = ((terms["dot"] < 0).to(mask.dtype) * live).sum(dim=-1)
                per_row[:, _IDX["pg_sq_sum"]] = (terms["pg_sq"] * live).sum(dim=-1)

            flat = task_ids.reshape(-1)
            flat = flat.round().to(torch.long) if flat.is_floating_point() else flat.to(torch.long)
            # Padding rows carry a negative id and must land nowhere. clamp puts
            # them on task 0, so the one-hot is zeroed for them instead.
            onehot = torch.nn.functional.one_hot(flat.clamp(min=0), num_classes=self.n_tasks)
            onehot = onehot.to(per_row.dtype) * (flat >= 0).to(per_row.dtype).unsqueeze(-1)
            self.buf += (onehot.transpose(0, 1) @ per_row).to(torch.float64)

    def rows(self, task_names, coefs=None) -> dict:
        """One all-reduce, one host read, and the ratios the sums are for.

        Ratios are formed AFTER the reduction, never per micro-batch and never
        per rank: a mean of per-rank shares is not the share, and a mean of
        per-micro-batch shares weighs a short micro-batch like a full one.
        """
        buf = self.buf
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            buf = buf.clone()
            torch.distributed.all_reduce(buf, op=torch.distributed.ReduceOp.SUM)
        table = buf.tolist()

        def _safe(num, den):
            return float(num) / float(den) if den > 0 else 0.0

        out = {}
        push_l2 = {}
        for tid, name in enumerate(task_names[: self.n_tasks]):
            row = table[tid]
            n_tok = row[_IDX["n_tok"]]
            if n_tok <= 0:
                continue
            n_zero = row[_IDX["n_tok_adv_zero"]]
            n_live = n_tok - n_zero
            kl_zero = row[_IDX["kl_sum_adv_zero"]]
            n_align = row[_IDX["n_align"]]
            l2 = row[_IDX["push_sq_sum"]] ** 0.5
            push_l2[name] = l2

            out[f"actor/opd_diag/tokens/{name}"] = n_tok
            out[f"actor/opd_diag/adv_zero_frac/{name}"] = _safe(n_zero, n_tok)
            out[f"actor/opd_diag/kl_mean/{name}"] = _safe(row[_IDX["kl_sum"]], n_tok)
            out[f"actor/opd_diag/kl_mean_adv_zero/{name}"] = _safe(kl_zero, n_zero)
            out[f"actor/opd_diag/kl_mean_adv_live/{name}"] = _safe(row[_IDX["kl_sum"]] - kl_zero, n_live)
            out[f"actor/opd_diag/push_l2_logit/{name}"] = l2
            out[f"actor/opd_diag/push_rms_logit/{name}"] = _safe(row[_IDX["push_sq_sum"]], n_tok) ** 0.5
            out[f"actor/opd_diag/align_tokens/{name}"] = n_align
            out[f"actor/opd_diag/align_cover/{name}"] = _safe(n_align, n_live)
            out[f"actor/opd_diag/pg_dot_mean/{name}"] = _safe(row[_IDX["dot_sum"]], n_align)
            out[f"actor/opd_diag/pg_cos_mean/{name}"] = _safe(row[_IDX["cos_sum"]], n_align)
            out[f"actor/opd_diag/pg_dot_neg_frac/{name}"] = _safe(row[_IDX["dot_neg"]], n_align)
            out[f"actor/opd_diag/pg_push_rms_logit/{name}"] = _safe(row[_IDX["pg_sq_sum"]], n_align) ** 0.5
            # The absolute contributions, not only the shares: two arms can hold
            # the same shares while the whole teacher-KL term moves, and the
            # share alone would report that as no change.
            out[f"actor/opd_diag/kl_sum_base/{name}"] = row[_IDX["kl_base_sum"]]
            out[f"actor/opd_diag/kl_sum_eff/{name}"] = row[_IDX["kl_eff_sum"]]

        base_total = sum(table[t][_IDX["kl_base_sum"]] for t in range(self.n_tasks))
        eff_total = sum(table[t][_IDX["kl_eff_sum"]] for t in range(self.n_tasks))
        for tid, name in enumerate(task_names[: self.n_tasks]):
            if table[tid][_IDX["n_tok"]] <= 0:
                continue
            out[f"actor/opd_diag/kl_share_base/{name}"] = _safe(table[tid][_IDX["kl_base_sum"]], base_total)
            out[f"actor/opd_diag/kl_share_eff/{name}"] = _safe(table[tid][_IDX["kl_eff_sum"]], eff_total)

        # The invariant the rule is built on, read on THIS step's data instead of
        # the calibration set: 1.000 means the reallocation moved shares without
        # moving the total, and the distance from 1 is how far the guarantee
        # travelled. Uniform b gives exactly b, which is what makes the uniform
        # arm's number interpretable next to the redistributed one.
        if coefs and push_l2:
            plain = sum(push_l2.values())
            scaled = sum(push_l2[n] * float(coefs.get(n, 1.0)) for n in push_l2)
            out["actor/opd_diag/budget_ratio_logit"] = _safe(scaled, plain)
        return out
