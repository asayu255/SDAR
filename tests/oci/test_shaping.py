"""CPU test for OCI-sat arm A's shaping: the GRADIENT, through autograd.

The loss is written as ``-A * f(rho)`` and the coefficient
``A * gamma * rho / (rho + gamma)^2`` is never written out in the loss path --
autograd is supposed to produce it. That claim is the whole implementation, so
it is checked against the closed form here rather than assumed, and checked at
the three points that decide the design: rho -> 0 (an unreachable injection
carries nothing), rho = gamma (the ceiling), and rho large (also nearly nothing).
"""
import math, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from verl.trainer.ppo.oci_shaping import (
    DEFAULT_GAMMA, shaped_pg_losses, shaping_coefficient, shaping_diagnostics, injected_rows)

G = DEFAULT_GAMMA
ok = True


def grad_of_loss(log_rho, adv, gamma=G):
    """d(loss)/d(log_prob_plain) at a single token, by autograd."""
    old = torch.zeros(1, 1, dtype=torch.float64)
    plain = torch.tensor([[float(log_rho)]], dtype=torch.float64, requires_grad=True)
    loss = shaped_pg_losses(plain, old, torch.tensor([[float(adv)]], dtype=torch.float64),
                            gamma=gamma)
    loss.sum().backward()
    return float(plain.grad[0, 0])


# 1. autograd == -A * gamma*rho/(rho+gamma)^2, over five decades of rho
print("  --- gradient vs the closed form, A = -sqrt(7) ---")
A = -math.sqrt(7)
for rho in (1e-4, 1e-2, G, 1.0, 10.0):
    got = grad_of_loss(math.log(rho), A)
    want = -A * shaping_coefficient(rho, G)
    good = abs(got - want) < 1e-12
    ok &= good
    print(("  OK  " if good else "  FAIL") +
          f" rho={rho:<7g} autograd {got:+.9f}  closed form {want:+.9f}")

# 2. the coefficient's shape: a band-pass, peak at rho = gamma, falling as
#    rho/gamma below it and as gamma/rho above. The thresholds are those
#    asymptotes, not round numbers -- coef(1e-6) is ~1e-5, which against a
#    ceiling of 0.25 is 4e-5 of it, so "< 1e-5 of the peak" was simply wrong
#    arithmetic on a first writing of this test.
mags = {rho: abs(grad_of_loss(math.log(rho), A)) for rho in (1e-6, 1e-3, G, 1e3, 1e6)}
good = (mags[G] == max(mags.values())
        and mags[1e-6] / mags[G] < 1e-4
        and mags[1e6] / mags[G] < 1e-4)
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" band-pass: |grad|/peak is {mags[1e-6]/mags[G]:.2g} at rho=1e-6, 1 at "
      f"rho=gamma, {mags[1e6]/mags[G]:.2g} at rho=1e6")

# the asymptote itself, which is what bounds how much advantage could compensate
good = abs(shaping_coefficient(1e-6, G) / (1e-6 / G) - 1.0) < 1e-4
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" below the band the coefficient IS rho/gamma: coef(1e-6)={shaping_coefficient(1e-6, G):.3g} "
      f"vs rho/gamma={1e-6/G:.3g}")

good = abs(mags[G] / abs(A) - 0.25) < 1e-12
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" the ceiling is exactly 1/4 of |A| ({mags[G] / abs(A):.6f})")

# 3. THE POINT OF THE WHOLE DESIGN, stated as the quantity that decides it: how
#    much advantage would be needed to make an unreachable injection carry what
#    a reachable one carries. The answer is the coefficient ratio, and it is not
#    available -- |A| in a group of n is at most sqrt(n-1), so sqrt(7) here.
need = shaping_coefficient(G, G) / shaping_coefficient(1e-5, G)
good = need > 1000 and need * abs(A) > math.sqrt(7) * 1000
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" at rho=1e-5 the coefficient is 1/{need:.0f} of the ceiling, so matching the "
      f"band needs {need:.0f}x the advantage; the most a group of 8 can give is "
      f"sqrt(7)={math.sqrt(7):.3f}")

# and a merely hundredfold advantage does not get there
huge = abs(grad_of_loss(math.log(1e-5), A * 100))
good = huge < mags[G] / 20
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" |A|x100 at rho=1e-5 gives {huge:.3g}, still {mags[G]/huge:.0f}x below the "
      f"band's {mags[G]:.4g}")

# 4. direction. A negative advantage must push the plain student's log-prob DOWN
#    (suppress the failure); a positive one up.
gneg = grad_of_loss(math.log(G), -1.0)
gpos = grad_of_loss(math.log(G), +1.0)
good = gneg > 0 and gpos < 0     # gradient DESCENT moves against the gradient
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" descent direction: A<0 -> grad {gneg:+.4f} (log-prob pushed down), "
      f"A>0 -> grad {gpos:+.4f} (up)")

# 5. no clip. A standard PPO clip would flatten the gradient outside
#    [1-eps, 1+eps]; the shaped one is alive at rho=0.1, which is where a clip
#    of 0.2 is already dead.
eps = 0.2
clipped_alive = (1 - eps) <= G <= (1 + eps)
good = (not clipped_alive) and mags[G] > 0
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" rho=gamma={G} is outside a clip of {eps} (dead there) and carries "
      f"{mags[G]:.4g} here")

# 6. the loss is finite for an absurd ratio rather than inf
big = shaped_pg_losses(torch.tensor([[500.0]]), torch.zeros(1, 1), torch.tensor([[A]]))
good = torch.isfinite(big).all()
ok &= good
print(("  OK  " if good else "  FAIL") + f" log_rho=500 is clamped, loss {float(big):+.4f} finite")

# 7. diagnostics read the same tensors and agree with the closed form
plain = torch.tensor([[math.log(G)] * 4, [math.log(1e-4)] * 4], dtype=torch.float32)
old = torch.zeros(2, 4)
mask = torch.ones(2, 4)
d = shaping_diagnostics(plain, old, mask)
good = (d["oci/shaping/tokens"] == 8
        and abs(d["oci/shaping/coef_p95"] - 0.25) < 1e-5
        and abs(d["oci/shaping/frac_in_band"] - 0.5) < 1e-6)
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" diagnostics: {d['oci/shaping/tokens']} tokens, coef p95 "
      f"{d['oci/shaping/coef_p95']:.4f}, frac_in_band {d['oci/shaping/frac_in_band']:.2f}")

# 8. the row selector: absent column and all-zero column both read as "nothing"
good = (injected_rows({}) is None
        and injected_rows({"oci_injected": torch.zeros(4, dtype=torch.long)}) is None
        and injected_rows({"oci_injected": torch.tensor([0, 1, 0, 0])}).tolist()
        == [False, True, False, False])
ok &= good
print(("  OK  " if good else "  FAIL") + " injected_rows: absent and all-zero both give None")


# --- THE dp_actor SUBSTITUTION, with a real (tiny) forward ------------------
# Everything above is the loss function. This drives the code that decides WHICH
# rows get it, strips their prompt, forwards them, and writes the result back --
# with a stub actor whose _forward_micro_batch is a real differentiable function
# of its input, so the replacement has to be wired correctly for the gradient to
# reach anything.
import numpy as np
from verl.workers.actor.dp_actor import _oci_injected_rows, _oci_shaped_rows

PL, RL, BS = 6, 3, 4
PAD = 0


def micro(inj, plan_len, trunc=None):
    ids = torch.zeros(BS, PL + RL, dtype=torch.long)
    am = torch.zeros(BS, PL + RL, dtype=torch.long)
    pos = torch.zeros(BS, PL + RL, dtype=torch.long)
    for i in range(BS):
        ids[i, 2:PL] = torch.tensor([10 + i, 20 + i, 30 + i, 40 + i])
        am[i, 2:PL] = 1
        pos[i, 2:PL] = torch.arange(4)
        ids[i, PL:] = torch.tensor([100 + i, 200 + i, 300 + i])
        am[i, PL:] = 1
        pos[i, PL:] = torch.arange(4, 4 + RL)
    return {
        "input_ids": ids, "attention_mask": am, "position_ids": pos,
        "responses": ids[:, PL:].clone(),
        "oci_injected": torch.tensor(inj, dtype=torch.long),
        "oci_plan_off": torch.zeros(BS, dtype=torch.long),
        "oci_plan_len": torch.tensor(plan_len, dtype=torch.long),
        **({"oci_plan_truncated": torch.tensor(trunc, dtype=torch.long)} if trunc else {}),
    }


class StubActor:
    """_forward_micro_batch as a differentiable function of the stripped ids,
    so a mis-wiring shows up as a missing or detached gradient."""

    def __init__(self):
        self.scale = torch.tensor(1.0, requires_grad=True)
        self.seen_rows = None
        self.seen_ids = None

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False):
        ids = micro_batch["input_ids"]
        self.seen_rows = ids.shape[0]
        self.seen_ids = ids.clone()
        live = micro_batch["attention_mask"][:, :ids.shape[1] - RL].sum(-1, keepdim=True)
        lp = -self.scale * (live.to(torch.float32) / 10.0).expand(-1, RL)
        return None, None, lp


adv = torch.full((BS, RL), -math.sqrt(7))
old = torch.full((BS, RL), math.log(0.5))
rmask = torch.ones(BS, RL)
base = torch.full((BS, RL), 0.123)

# rows 1 and 3 injected, both with a 2-token plan span
a = StubActor()
mb = micro([0, 1, 0, 1], [0, 2, 0, 2])
inj = _oci_injected_rows(mb)
out, diag = _oci_shaped_rows(a, mb, base.clone(), inj, response_mask=rmask,
                             advantages=adv, old_log_prob=old, temperature=1.0, gamma=G)

good = a.seen_rows == 2
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" only the injected rows are forwarded: {a.seen_rows} of {BS}")

good = torch.equal(out[[0, 2]], base[[0, 2]]) and not torch.equal(out[[1, 3]], base[[1, 3]])
ok &= good
print(("  OK  " if good else "  FAIL") +
      " the untouched rows keep their clipped term, the injected rows are replaced")

# the prompt really was shortened by the span, and the response region survived
good = (int(a.seen_ids.shape[1]) == PL + RL
        and torch.equal(a.seen_ids[:, PL:], mb["responses"][[1, 3]]))
ok &= good
print(("  OK  " if good else "  FAIL") +
      " the forward saw the same width and an untouched response region")

# the replacement is differentiable end to end
out[[1, 3]].sum().backward()
good = a.scale.grad is not None and float(a.scale.grad) != 0.0
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" the shaped term carries gradient back through the forward "
      f"(d/dscale = {float(a.scale.grad) if a.scale.grad is not None else None})")

# a row marked injected but with no strippable span keeps its unshaped term
a2 = StubActor()
mb2 = micro([0, 1, 0, 1], [0, 2, 0, 0])       # row 3 has no span
out2, diag2 = _oci_shaped_rows(a2, mb2, base.clone(), _oci_injected_rows(mb2),
                               response_mask=rmask, advantages=adv,
                               old_log_prob=old, temperature=1.0, gamma=G)
good = (a2.seen_rows == 1 and torch.equal(out2[3], base[3])
        and diag2["oci/shaping/rows_unstrippable"] == 1.0)
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" an unstrippable injected row is left unshaped and counted "
      f"({diag2['oci/shaping/rows_unstrippable']:.0f})")

# truncated rows are treated the same way
a3 = StubActor()
mb3 = micro([0, 1, 0, 0], [0, 2, 0, 0], trunc=[0, 1, 0, 0])
out3, diag3 = _oci_shaped_rows(a3, mb3, base.clone(), _oci_injected_rows(mb3),
                               response_mask=rmask, advantages=adv,
                               old_log_prob=old, temperature=1.0, gamma=G)
good = a3.seen_rows is None and torch.equal(out3, base) and diag3["oci/shaping/rows_unstrippable"] == 1.0
ok &= good
print(("  OK  " if good else "  FAIL") +
      " a truncated row is not stripped, not forwarded, and not shaped")

# and the diagnostics name the band
good = "oci/shaping/frac_in_band" in diag and diag["oci/shaping/rows_injected"] == 2.0
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" diagnostics report {diag['oci/shaping/rows_injected']:.0f} injected rows and "
      f"frac_in_band {diag.get('oci/shaping/frac_in_band')}")


def test_shaping():
    """Collected by pytest; the checks above ran at import and set `ok`."""
    assert ok


if __name__ == "__main__":
    print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    sys.exit(0 if ok else 1)
