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


# --- THE DISPATCH SIZE. compute_log_prob is registered DP_COMPUTE_PROTO, and
# DataProto.chunk asserts the batch divides by the world size. The caller hands
# this function the CANDIDATE ROWS ONLY -- one trajectory per group, as many rows
# as it ran turns -- so the count is arbitrary and on a 2-GPU host fails that
# assert about half the time. The failure used to land in the except clause and
# come back as a recorded string, i.e. the one number the design turns on would
# have been silently missing from roughly every other probe batch.
from verl.protocol import DataProto

ODD = 7   # not divisible by 2


def real_batch(n):
    return DataProto.from_dict(tensors={
        "rollout_log_probs": torch.full((n, TOK), -1.0),
        "old_log_probs": torch.full((n, TOK), -1.0),
        "input_ids": torch.ones(n, TOK, dtype=torch.long),
        "attention_mask": torch.ones(n, TOK, dtype=torch.long),
        "loss_mask": torch.ones(n, TOK, dtype=torch.long),
        "task_ids": torch.zeros(n, dtype=torch.long),
    })


class StrictWG:
    """Rejects an indivisible batch, the way DataProto.chunk does."""

    def __init__(self, world_size):
        self.world_size = world_size
        self.seen = None

    def compute_log_prob(self, d):
        self.seen = len(d)
        assert len(d) % self.world_size == 0, (
            f"only support equal chunk. Got size of DataProto {len(d)} and "
            f"chunk {self.world_size}.")
        out = DataProto.from_dict(tensors={
            "old_log_probs": torch.full((len(d), TOK), -1.0 + math.log(0.1))})
        return out


wg = StrictWG(world_size=2)
rep = reachability_report(wg, real_batch(ODD), ["alfworld"],
                          strip_fn=lambda b: b, gamma=0.1)
good = "error" not in rep and wg.seen == ODD + 1
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" an odd batch of {ODD} is padded to {wg.seen} for a world of 2, and the "
      f"report comes back{'' if 'error' not in rep else ' as ' + str(rep)}")

# and the report is about the ODD real rows, not the padded ones
good = "error" not in rep and rep["alfworld"]["all"]["tokens"] == ODD * TOK
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" the padding is removed again: {rep.get('alfworld', {}).get('all', {}).get('tokens')} "
      f"tokens counted, {ODD * TOK} expected")

# rho survives the round trip: log_rho = log(0.1) everywhere, so the coefficient
# sits exactly on its ceiling
good = ("error" not in rep
        and abs(rep["alfworld"]["all"]["rho_median"] - 0.1) < 1e-6
        and abs(rep["alfworld"]["all"]["coef"][50] - 0.25) < 1e-6)
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" rho survives pad/unpad: median {rep.get('alfworld', {}).get('all', {}).get('rho_median')}, "
      f"coef p50 {rep.get('alfworld', {}).get('all', {}).get('coef', {}).get(50)}")

# a world of 1 must not pad at all
wg1 = StrictWG(world_size=1)
rep1 = reachability_report(wg1, real_batch(ODD), ["alfworld"],
                           strip_fn=lambda b: b, gamma=0.1)
good = "error" not in rep1 and wg1.seen == ODD
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" a world of 1 is handed {wg1.seen} rows unpadded")

# the failure this replaced, so the test is known to be able to fail
class NoPadWG(StrictWG):
    pass


try:
    NoPadWG(2).compute_log_prob(real_batch(ODD))
    raised = False
except AssertionError:
    raised = True
ok &= raised
print(("  OK  " if raised else "  FAIL") +
      " and the unpadded call really does assert, so the check above is not vacuous")


def test_reachability():
    """Collected by pytest; the checks above ran at import and set `ok`."""
    assert ok


if __name__ == "__main__":
    print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    sys.exit(0 if ok else 1)
