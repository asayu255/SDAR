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

    PADDED TO THE WORKER GROUP. ``compute_log_prob`` is registered
    ``DP_COMPUTE_PROTO``, whose dispatch chunks the DataProto across the data
    parallel world and whose ``DataProto.chunk`` asserts the size divides
    exactly. The caller hands this function the CANDIDATE ROWS ONLY -- one
    trajectory per group, each as many rows as it ran turns -- so the count is
    arbitrary and on a 2-GPU host fails that assert about half the time. The
    failure lands in the except below and comes back as a recorded error, i.e.
    the one number the whole design turns on would have been missing from
    roughly every other probe batch with nothing but a string to say why.
    """
    import numpy as np
    import torch

    from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

    lp_cond = batch.batch.get("rollout_log_probs", batch.batch.get("old_log_probs", None))
    if lp_cond is None:
        return {"error": "batch carries no rollout_log_probs/old_log_probs"}
    try:
        stripped = strip_fn(batch)
        world = int(getattr(policy_wg, "world_size", 1) or 1)
        # Only when it is actually needed: a world of one needs no padding, and
        # pad_dataproto_to_divisor requires a real DataProto, which a
        # single-process caller (or a CPU test) may not have built.
        pad_size = 0
        if world > 1 and (len(stripped) % world):
            stripped, pad_size = pad_dataproto_to_divisor(stripped, world)
        out = policy_wg.compute_log_prob(stripped)
        if pad_size:
            out = unpad_dataproto(out, pad_size=pad_size)
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


def splice_span(input_ids, attention_mask, off, take, repl, repl_len, pad_token_id,
                response_length: int, position_ids=None):
    """Replace live prompt tokens ``[off, off+take)`` with ``repl[:repl_len]``.

    A REPLACEMENT, NOT A DELETION, and not a front-strip either. Two things went
    wrong before:

    1. The student-mode notice is a system message at position 0, so its tokens
       are a prefix of the render and a length alone locates them. The corrupted
       plan sits at the head of the USER turn's content, behind the chat header,
       so stripping ``take`` LEADING tokens removes the header and leaves the
       plan.
    2. Removing the plan's span does not give the no-plan render either. Without
       the plan the chat header's newline merges with the observation's leading
       newline into one "\n\n" token; with the plan it cannot, because "###"
       follows. So the two renders differ by a replacement of one token, not by a
       pure deletion, and a deletion-only rule filed every candidate row as
       unstrippable -- the first probe measured rho on 0 of 299 rows.

    The rollout loop records both halves of the edit and verifies by
    reconstruction, so what arrives here is known to reproduce the no-plan
    prompt exactly.

    THE RESPONSE REGION IS COPIED THROUGH BYTE FOR BYTE. The rollout writes
    ``input_ids = cat([prompt, response])`` with the PROMPT LEFT-PADDED and the
    RESPONSE RIGHT-PADDED (vllm_rollout.py: ``attention_mask:
    [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]``), so the live tokens are not one
    block at the right-hand end. Gathering every live token and right-aligning
    the result slid the response rightwards by however much trailing padding it
    had, and the forward reads logits at ``[-response_length-1:-1]`` -- every
    log-prob would have come from the wrong position, with the shape check
    passing. Only the prompt's own left-padded window is rebuilt.

    ``position_ids``, when given, keeps the rollout's convention: the prompt's
    live span numbered from 0 and the response continuing after it, so a net
    change of ``take - repl_len`` prompt tokens shifts the response by that much.
    """
    import torch

    bs, width = input_ids.shape
    plen = width - int(response_length)
    assert plen > 0, (
        f"response_length={response_length} leaves no prompt in a width of {width}")

    out_ids = input_ids.clone()
    out_mask = attention_mask.clone()
    # ON THE INPUT'S DEVICE. The probe ran this on driver-side tensors and every
    # test on CPU tensors; the first time the actor ran it on its own micro-batch
    # the subtraction below met a cuda position_ids and a cpu net and died at
    # step 1 of the first shaped run.
    net = torch.zeros(bs, dtype=torch.long, device=input_ids.device)
    for i in range(bs):
        p_mask = attention_mask[i, :plen].bool()
        toks = input_ids[i, :plen][p_mask]
        o, k, m = int(off[i]), int(take[i]), int(repl_len[i])
        if k > 0 and 0 <= o <= toks.numel() - k:
            keep = torch.cat([toks[:o], repl[i, :m].to(toks.dtype), toks[o + k:]])
            net[i] = k - m
        else:
            keep = toks
        out_ids[i, :plen] = int(pad_token_id)
        out_mask[i, :plen] = 0
        n_keep = keep.numel()
        if n_keep > plen:
            # The replacement cannot make the prompt longer than its own window.
            # Nothing here can be spliced without dropping real tokens, so the
            # row passes through untouched and the caller's strippable check is
            # what keeps it out of the report.
            out_ids[i, :plen] = input_ids[i, :plen]
            out_mask[i, :plen] = attention_mask[i, :plen]
            net[i] = 0
            continue
        out_ids[i, plen - n_keep:plen] = keep
        out_mask[i, plen - n_keep:plen] = 1

    if position_ids is None:
        return out_ids, out_mask, None
    out_pos = position_ids.clone()
    out_pos[:, :plen] = (out_mask[:, :plen].cumsum(-1) - 1).clamp(min=0).to(out_pos.dtype)
    out_pos[:, plen:] = (position_ids[:, plen:] - net.unsqueeze(-1)).clamp(min=0)
    return out_ids, out_mask, out_pos


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
        # Where the prompt ends. `responses` is the authority: the forward reads
        # logits at [-response_length-1:-1] and gathers against that same
        # tensor, so the prompt region is everything before it and must stay
        # exactly that wide.
        resp = batch.batch.get("responses", None)
        repl = batch.batch.get("oci_plan_repl", None)
        rlen = batch.batch.get("oci_plan_repl_len", None)
        if resp is None or repl is None or rlen is None:
            return batch
        ids, mask, pos = splice_span(
            batch.batch["input_ids"], batch.batch["attention_mask"],
            n_off.reshape(-1), length, repl, rlen.reshape(-1), pad_token_id,
            response_length=resp.shape[1],
            position_ids=batch.batch.get("position_ids", None),
        )
        tensors = {k: v for k, v in batch.batch.items()}
        tensors.update({"input_ids": ids, "attention_mask": mask})
        if pos is not None:
            tensors["position_ids"] = pos
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
