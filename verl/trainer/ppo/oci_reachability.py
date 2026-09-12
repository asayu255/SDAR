"""rho: is an injected failure one the UNCONDITIONED policy could also produce?

THE NUMBER THAT DECIDES WHETHER INJECTION CAN WORK AT ALL. A row injected into a
saturated group -- seven of eight rollouts at max return, where the group's
advantage is correctly zero -- carries advantage ``-sqrt(7)``; the seven
successes each pick up ``+1/sqrt(7)``. Both figures follow from the 7:1 split
alone and do not depend on the reward scale. What actually reaches the weights
on the injected row is not that advantage but

    dJ/dlog pi = A * gamma * rho / (rho + gamma)^2,
    rho = pi_theta(a | x) / pi_theta(a | x, z),

with the behaviour probability held fixed (LUFFY's policy shaping, which
replaces the ratio rather than multiplying it, and drops the PPO clip on the
off-policy side). That coefficient is ZERO at rho = 0 and maximal at rho =
gamma. So a failure the policy would never produce without the privileged
context ``z`` carries no gradient however large its advantage, and shaping does
not rescue it -- shaping is precisely what makes the coefficient vanish there.

This is why a failure borrowed from a different set of weights (the base policy)
is suspect, and why one produced by the SAME weights under a corrupted prompt is
worth measuring: only the rho distribution says whether it is reachable.

NOT A DERIVED QUANTITY. ``f(rho) = rho/(rho+gamma)`` near 1 is NOT evidence of
reachability. At rho = gamma the shaped ratio is 0.5 while the coefficient sits
at its ceiling of 1/4. Report the coefficient, split by the sign of the
advantage; never the shaped ratio on its own.

COST. One forward pass of the policy's own weights over tokens that already
exist. No generation, no second engine.
"""

from typing import Callable, Dict, List, Optional

__all__ = ["reachability_report", "shaping_coefficient"]


def shaping_coefficient(rho, gamma: float = 0.1):
    """``gamma * rho / (rho + gamma)^2`` -- what a shaped off-policy row delivers.

    Peaks at ``rho == gamma`` with value ``1/(4*gamma) * gamma = 0.25``, and goes
    to zero in both directions. Accepts a float or a torch tensor.
    """
    return gamma * rho / (rho + gamma) ** 2


def _loss_mask(batch, response_len: int):
    """The mask update_policy aggregates over, not the response_mask column."""
    key = "loss_mask" if "loss_mask" in batch.batch.keys() else "attention_mask"
    return batch.batch[key][:, -response_len:].bool()


def reachability_report(
    policy_wg,
    batch,
    task_id_names: List[str],
    *,
    strip_fn: Callable,
    gamma: float = 0.1,
    advantage_sign: Optional[Callable] = None,
) -> Dict:
    """Per task: the distribution of rho and of the shaping coefficient.

    ``strip_fn(batch)`` must return the same rows with the privileged block
    removed from the prompt and everything else identical. The caller owns it,
    because only the caller knows how ``z`` was inserted.

    ``advantage_sign(batch) -> tensor of +1/-1 per row`` splits the report by the
    sign of the advantage, which is the split that matters: a positive injected
    row is being imitated, a negative one suppressed, and they are not
    interchangeable.
    """
    import numpy as np
    import torch

    lp_cond = batch.batch.get("rollout_log_probs", batch.batch.get("old_log_probs", None))
    if lp_cond is None:
        return {"error": "batch carries no rollout_log_probs/old_log_probs"}
    try:
        out = policy_wg.compute_log_prob(strip_fn(batch))
        lp_plain = out.batch["old_log_probs"]
    except Exception as exc:
        return {"error": f"re-scoring failed: {exc!r}"}
    if tuple(lp_plain.shape) != tuple(lp_cond.shape):
        return {"error": f"shape mismatch {tuple(lp_plain.shape)} vs {tuple(lp_cond.shape)}"}

    mask = _loss_mask(batch, lp_cond.shape[1])
    log_rho = lp_plain.to(torch.float32) - lp_cond.to(torch.float32)
    rho = log_rho.clamp(min=-30.0, max=30.0).exp()
    coef = shaping_coefficient(rho, gamma)

    sign = None
    if advantage_sign is not None:
        try:
            sign = advantage_sign(batch)
        except Exception:
            sign = None

    task_ids = batch.batch["task_ids"].reshape(-1).tolist()
    res = {"gamma": gamma, "coef_ceiling": 0.25}
    for tid in sorted({int(t) for t in task_ids}):
        name = task_id_names[tid] if tid < len(task_id_names) else str(tid)
        rows = [i for i, t in enumerate(task_ids) if int(t) == tid]
        groups = {"all": rows}
        if sign is not None:
            groups["pos"] = [i for i in rows if float(sign[i]) > 0]
            groups["neg"] = [i for i in rows if float(sign[i]) < 0]
        rec = {}
        for gname, sel in groups.items():
            if not sel:
                continue
            m = mask[sel]
            if not int(m.sum()):
                continue
            lr = log_rho[sel][m].detach().cpu().numpy()
            cf = coef[sel][m].detach().cpu().numpy()
            rec[gname] = {
                "tokens": int(lr.size),
                "log_rho": {q: float(np.percentile(lr, q)) for q in (5, 25, 50, 75, 95)},
                "rho_median": float(np.exp(np.percentile(lr, 50))),
                "coef": {q: float(np.percentile(cf, q)) for q in (5, 50, 95)},
                # share of tokens carrying at least a tenth of the ceiling
                "frac_coef_above_0.025": float((cf > 0.025).mean()),
            }
        if rec:
            res[name] = rec
    return res


# --------------------------------------------------------------------------- #
# the strip function for the wrong-plan arm
# --------------------------------------------------------------------------- #


def strip_span(input_ids, attention_mask, off, length, pad_token_id: int):
    """Remove live tokens ``[off[i], off[i]+length[i])`` from each row.

    NOT A FRONT-STRIP, WHICH IS WHY THIS EXISTS SEPARATELY FROM
    ``privileged_notice.strip_prefix``. The student-mode notice is a system
    message at position 0, so its tokens really are a prefix of the render and a
    length alone locates them. The corrupted plan goes at the head of the USER
    turn's content, behind the chat header, so stripping ``length`` leading
    tokens removes the header and leaves the plan -- the resulting prompt is not
    one the policy could ever see, and the rho computed against it is a ratio
    between two fictions. The offset is measured in the rollout loop, where the
    two renders can be compared token by token, and rides on the row.
    """
    import torch

    bs, width = input_ids.shape
    out_ids = torch.full_like(input_ids, int(pad_token_id))
    out_mask = torch.zeros_like(attention_mask)
    for i in range(bs):
        live = attention_mask[i].bool()
        toks = input_ids[i][live]
        o, n = int(off[i]), int(length[i])
        if n > 0 and 0 <= o <= toks.numel() - n:
            keep = torch.cat([toks[:o], toks[o + n:]])
        else:
            keep = toks
        out_ids[i, width - keep.numel():] = keep
        out_mask[i, width - keep.numel():] = 1
    pos = (out_mask.cumsum(-1) - 1).clamp(min=0).to(torch.long)
    return out_ids, out_mask, pos


def wrong_plan_strip_fn(tokenizer, pad_token_id: int):
    """Build a ``strip_fn`` that removes the corrupted-plan block from each row.

    Removing exactly the block's token span gives the prompt the policy would
    have seen without it, on the SAME response tokens. That is what makes rho
    well defined: numerator and denominator differ in the conditioning and in
    nothing else.

    The span rides in ``oci_plan_off`` / ``oci_plan_len``, both emitted by the
    rollout loop. A row with length 0 passes through unchanged, which covers
    three cases at once: no plan was shown, the two renders did not differ in one
    contiguous span, or the prompt was left-truncated so the offset no longer
    locates the block (``oci_plan_truncated``). All three mean "not strippable",
    and a row that is not strippable must be left out of the rho distribution
    rather than stripped approximately -- ``reachability_report`` counts them.
    """
    def _strip(batch):
        from verl.protocol import DataProto

        n_len = batch.batch.get("oci_plan_len", None)
        n_off = batch.batch.get("oci_plan_off", None)
        if n_len is None or n_off is None:
            return batch
        trunc = batch.batch.get("oci_plan_truncated", None)
        length = n_len.reshape(-1).clone()
        if trunc is not None:
            length = length * (1 - trunc.reshape(-1).clamp(0, 1)).to(length.dtype)
        ids, mask, pos = strip_span(
            batch.batch["input_ids"], batch.batch["attention_mask"],
            n_off.reshape(-1), length, pad_token_id,
        )
        tensors = {k: v for k, v in batch.batch.items()}
        tensors.update({"input_ids": ids, "attention_mask": mask, "position_ids": pos})
        out = DataProto.from_dict(tensors=tensors,
                                  non_tensors=dict(batch.non_tensor_batch))
        out.meta_info = dict(batch.meta_info)
        return out

    return _strip


def strippable_rows(batch):
    """Rows whose plan span can be removed exactly; the rho denominator's domain."""
    import numpy as np

    n_len = batch.batch.get("oci_plan_len", None)
    if n_len is None:
        return np.zeros(len(batch), dtype=bool)
    ok = n_len.reshape(-1).detach().cpu().numpy() > 0
    trunc = batch.batch.get("oci_plan_truncated", None)
    if trunc is not None:
        ok &= trunc.reshape(-1).detach().cpu().numpy() == 0
    return ok
