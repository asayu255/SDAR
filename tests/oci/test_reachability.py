"""CPU test for oci_reachability: no GPU, no rollout, no environment."""
import math, os, sys, types
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from verl.trainer.ppo.oci_reachability import reachability_report, shaping_coefficient

TOK = 6


def build(log_rho_rows, signs=None):
    n = len(log_rho_rows)
    cond = torch.full((n, TOK), -1.0)
    plain = torch.stack([torch.full((TOK,), -1.0 + lr) for lr in log_rho_rows])
    b = {"rollout_log_probs": cond,
         "loss_mask": torch.ones(n, TOK, dtype=torch.long),
         "attention_mask": torch.ones(n, TOK, dtype=torch.long),
         "task_ids": torch.zeros(n, dtype=torch.long)}
    batch = types.SimpleNamespace(batch=b, non_tensor_batch={}, meta_info={})

    class WG:
        def compute_log_prob(self, d):
            return types.SimpleNamespace(batch={"old_log_probs": plain})

    sign_fn = (lambda _b: torch.tensor(signs, dtype=torch.float32)) if signs else None
    return WG(), batch, sign_fn


ok = True

# the coefficient's closed form, at the three points that matter
for rho, want in ((1.0, 0.1 / 1.21), (0.1, 0.25), (1e-4, 0.1 * 1e-4 / 0.1001 ** 2)):
    got = shaping_coefficient(rho)
    good = abs(got - want) <= 1e-12
    ok &= good
    print(("  OK  " if good else "  FAIL") + f" coef(rho={rho:g}) = {got:.6g} (want {want:.6g})")

# and that gamma is where it peaks
grid = [shaping_coefficient(r) for r in (1e-4, 1e-3, 0.01, 0.05, 0.1, 0.2, 1.0, 10.0)]
peak = abs(max(grid) - 0.25) <= 1e-12
ok &= peak
print(("  OK  " if peak else "  FAIL") + f" the ceiling 1/4 is attained at rho = gamma (max {max(grid):.6f})")

# end to end: rho recovered, dead rows carry no gradient
for lr, want_rho in ((0.0, 1.0), (math.log(0.1), 0.1), (math.log(1e-4), 1e-4)):
    wg, batch, _ = build([lr] * 4)
    r = reachability_report(wg, batch, ["t"], strip_fn=lambda b: b)["t"]["all"]
    good = abs(r["rho_median"] - want_rho) <= 1e-5 * max(want_rho, 1e-8)
    ok &= good
    print(("  OK  " if good else "  FAIL") +
          f" log_rho={lr:+8.3f} -> rho {r['rho_median']:.6g}, coef {r['coef'][50]:.6g}, "
          f"frac>0.025 {r['frac_coef_above_0.025']:.2f}")

wg, batch, _ = build([math.log(1e-4)] * 8)
dead = reachability_report(wg, batch, ["t"], strip_fn=lambda b: b)["t"]["all"]
good = dead["frac_coef_above_0.025"] == 0.0
ok &= good
print(("  OK  " if good else "  FAIL") + " an unreachable injection carries no gradient")

# the positive / negative split is what the report is for
wg, batch, sf = build([0.0, 0.0, math.log(1e-3), math.log(1e-3)], signs=[1, 1, -1, -1])
sp = reachability_report(wg, batch, ["t"], strip_fn=lambda b: b, advantage_sign=sf)["t"]
good = (abs(sp["pos"]["rho_median"] - 1.0) <= 1e-5
        and sp["neg"]["rho_median"] < 1e-2
        and sp["pos"]["frac_coef_above_0.025"] == 1.0
        and sp["neg"]["frac_coef_above_0.025"] == 0.0)
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" split by advantage sign: pos rho {sp['pos']['rho_median']:.3g}, neg rho {sp['neg']['rho_median']:.3g}")

# failures are reported, not raised
bad = reachability_report(None, build([0.0])[1], ["t"], strip_fn=lambda b: b)
good = "error" in bad
ok &= good
print(("  OK  " if good else "  FAIL") + f" a broken re-score is reported: {str(bad)[:56]}")


def test_reachability():
    """Collected by pytest; the checks above ran at import and set `ok`."""
    assert ok


if __name__ == "__main__":
    print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    sys.exit(0 if ok else 1)
