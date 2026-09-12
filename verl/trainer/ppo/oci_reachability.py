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
