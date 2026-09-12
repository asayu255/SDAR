"""CPU test for the wrong-plan prefix: reads real ALFWorld games, no GPU."""
import glob, json, os, random, re, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import agent_system.environments.env_manager as em

GAMES = sorted(glob.glob(os.path.expanduser(
    "~/data/alfworld/json_2.1.1/train/*/*/game.tw-pddl")))
if not GAMES:
    print("SKIP: no alfworld games on this host"); sys.exit(0)

ok = True
os.environ.pop("PRIVILEGED_WRONG_PLAN", None)
em._WRONG_PLAN_CACHE.clear()
off = em._wrong_plan_prefix("alfworld", GAMES[0])
ok &= (off == "")
print(("  OK  " if off == "" else "  FAIL") + " switch off yields nothing")

os.environ["PRIVILEGED_WRONG_PLAN"] = "1"
em._WRONG_PLAN_CACHE.clear()
random.seed(0)
samp = random.sample(GAMES, min(300, len(GAMES)))

# every game must be corruptible, and the block must be short
lens, bad = [], 0
for g in samp:
    p = em._wrong_plan_prefix("alfworld", g)
    if not p:
        bad += 1
    else:
        lens.append(len(p))
ok &= (bad == 0)
print(("  OK  " if bad == 0 else "  FAIL") + f" every game corruptible: {len(samp)-bad}/{len(samp)}")

# exactly one step short of the expert plan, and only names the plan already uses
short, alien = 0, 0
for g in samp[:120]:
    traj = os.path.join(os.path.dirname(g), "traj_data.json")
    expert = [s["discrete_action"] for s in json.load(open(traj))["plan"]["high_pddl"]
              if s.get("discrete_action", {}).get("action")]
    names = {a for st in expert for a in (st.get("args") or []) if a}
    wrong = em._wrong_plan_prefix("alfworld", g)
    steps = [l for l in wrong.splitlines() if re.match(r"^\d+\.", l)]
    if len(expert) - len(steps) != 1:
        short += 1
    # only the numbered plan lines carry action arguments; the header contains
    # the literal "(training only)" and must not be scanned
    for line in steps:
        for m in re.findall(r"\(([^)]*)\)", line):
            for a in (x.strip() for x in m.split(",") if x.strip()):
                if a not in names:
                    alien += 1
ok &= (short == 0 and alien == 0)
print(("  OK  " if short == 0 else "  FAIL") + f" exactly one step removed: {120-short}/120")
print(("  OK  " if alien == 0 else "  FAIL") +
      f" no object or place the expert plan does not already name: {alien} violations")

lens.sort()
print(f"  info  block length: p50 {lens[len(lens)//2]} chars, max {max(lens)}")

# guards
g_ok = (em._wrong_plan_prefix("search", GAMES[0]) == ""
        and em._wrong_plan_prefix("alfworld", None) == ""
        and em._wrong_plan_prefix("alfworld", "/no/such/path") == "")
ok &= g_ok
print(("  OK  " if g_ok else "  FAIL") + " other tasks, missing and bogus paths yield nothing")


def test_wrong_plan():
    """Collected by pytest; the checks above ran at import and set `ok`."""
    assert ok


if __name__ == "__main__":
    print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    sys.exit(0 if ok else 1)
