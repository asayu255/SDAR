"""CPU test for the privileged path block: reads real ALFWorld games, no GPU.

TWO MEASUREMENTS SHAPED THIS FILE, and both refuted a design whose invariants
this suite had already checked and passed.

1. The first block removed the plan's REQUIREMENT step. The suite verified that
   exactly one step was gone and that no unnamed object appeared. Both held on
   300 games; the candidate still solved the task 14 times in 15, because the
   removed step is the one the TASK DESCRIPTION already states (92% of 853
   sampled games).
2. The second permuted the navigation but printed PDDL symbols --
   `GotoLocation(dresser)`. The median log rho between the plan-conditioned and
   the plain student came back 0.0: the block was not being read at all.

So the checks below are not about the edit's shape. They are about whether the
block is written in the environment's own language, whether the two modes are
indistinguishable apart from the path, and whether the corruption reaches every
slot that carries information the task text does not.
"""
import glob, json, os, random, re, sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import agent_system.environments.env_manager as em

GAMES = sorted(glob.glob(os.path.expanduser(
    "~/data/alfworld/json_2.1.1/train/*/*/game.tw-pddl")))
if not GAMES:
    print("SKIP: no alfworld games on this host"); sys.exit(0)

# ALFWorld's own command grammar, from the installed package.
VERBS = ("go to ", "take ", "put ", "open ", "close ", "use ",
         "heat ", "cool ", "clean ", "slice ")
TOOLS = ("microwave", "fridge", "sinkbasin", "knife")


def cfg(enable=True, mode="misdirect"):
    return SimpleNamespace(algorithm={"oci_sat": {"enable": enable,
                                                  "plan_corruption": mode}})


def traj(gamefile):
    d = gamefile
    for _ in range(4):
        p = os.path.join(d, "traj_data.json") if os.path.isdir(d) else None
        if p and os.path.exists(p):
            return json.load(open(p))
        d = os.path.dirname(d)
    return None


def path_lines(block):
    return [l.split(". ", 1)[1] for l in block.splitlines()
            if re.match(r"^\d+\. ", l)]


ok = True

# --- the switch --------------------------------------------------------------
em._WRONG_PLAN_CACHE.clear()
good = (em._wrong_plan_prefix("alfworld", GAMES[0], cfg(enable=False)) == ""
        and em._wrong_plan_prefix("alfworld", GAMES[0], None) == ""
        and em._wrong_plan_prefix("webshop", GAMES[0], cfg()) == "")
ok &= good
print(("  OK  " if good else "  FAIL") +
      " switch off, no config, and another task all yield nothing")

try:
    em._oci_plan_mode(cfg(mode="drop"))
    good = False
except ValueError as e:
    good = "plan_corruption" in str(e)
ok &= good
print(("  OK  " if good else "  FAIL") +
      " the retired 'drop' mode is refused by name, not silently accepted")

# --- coverage and size -------------------------------------------------------
em._WRONG_PLAN_CACHE.clear()
random.seed(0)
samp = random.sample(GAMES, min(250, len(GAMES)))
empty = sum(1 for g in samp if not em._wrong_plan_prefix("alfworld", g, cfg()))
good = empty <= len(samp) * 0.02
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" {len(samp)-empty}/{len(samp)} games corruptible ({empty} name <2 receptacles)")

# --- THE LANGUAGE: every line is an environment command, no PDDL -------------
bad_verb, pddl, checked = 0, 0, 0
for g in samp[:150]:
    for mode in ("intact", "misdirect"):
        em._WRONG_PLAN_CACHE.clear()
        b = em._wrong_plan_prefix("alfworld", g, cfg(mode=mode))
        if not b:
            continue
        checked += 1
        for l in path_lines(b):
            bad_verb += not l.startswith(VERBS)
            pddl += bool(re.search(r"[A-Z]\w+\(", l))
good = checked > 100 and bad_verb == 0 and pddl == 0
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" {checked} blocks: {bad_verb} lines outside the environment's grammar, "
      f"{pddl} lines still in PDDL notation")

# --- THE TWO MODES DIFFER IN THE PATH AND IN NOTHING ELSE --------------------
same_frame, path_differs, tool_same, n = 0, 0, 0, 0
for g in samp[:150]:
    em._WRONG_PLAN_CACHE.clear()
    a = em._wrong_plan_prefix("alfworld", g, cfg(mode="intact"))
    em._WRONG_PLAN_CACHE.clear()
    b = em._wrong_plan_prefix("alfworld", g, cfg(mode="misdirect"))
    if not a or not b:
        continue
    n += 1
    fa = [l for l in a.splitlines() if not re.match(r"^\d+\. ", l)]
    fb = [l for l in b.splitlines() if not re.match(r"^\d+\. ", l)]
    same_frame += (fa == fb)
    path_differs += (path_lines(a) != path_lines(b))
    ta = [l for l in path_lines(a) if any(t in l for t in TOOLS) and " with " in l]
    tb = [l for l in path_lines(b) if any(t in l for t in TOOLS) and " with " in l]
    tool_same += (ta == tb)
good = n > 100 and same_frame == n and path_differs == n
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" {same_frame}/{n} share every non-path line, {path_differs}/{n} differ in the path")

good = tool_same == n
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" {tool_same}/{n} keep the tool line identical -- a swapped tool is "
      f"inexecutable and would announce the block is unreliable")

# --- THE CORRUPTION REACHES THE put DESTINATION ------------------------------
has_put, put_moved = 0, 0
for g in samp[:150]:
    em._WRONG_PLAN_CACHE.clear()
    a = [l for l in path_lines(em._wrong_plan_prefix("alfworld", g, cfg(mode="intact")))
         if " in/on " in l]
    em._WRONG_PLAN_CACHE.clear()
    b = [l for l in path_lines(em._wrong_plan_prefix("alfworld", g, cfg(mode="misdirect")))
         if " in/on " in l]
    if not a:
        continue
    has_put += 1
    put_moved += (a != b)
good = has_put > 50 and put_moved == has_put
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" {put_moved}/{has_put} games move the `put X in/on R` destination -- "
      f"leaving it correct keeps the line that states the goal")

# --- and no receptacle is invented -------------------------------------------
invented = 0
for g in samp[:120]:
    d = traj(g)
    if d is None:
        continue
    true_recs = {a for h in d["plan"]["high_pddl"]
                 for a in h["discrete_action"].get("args", []) if a}
    em._WRONG_PLAN_CACHE.clear()
    for l in path_lines(em._wrong_plan_prefix("alfworld", g, cfg(mode="misdirect"))):
        for w in l.split():
            if w in TOOLS or w in ("go", "to", "take", "from", "put", "in/on",
                                   "use", "open", "close", "with",
                                   "heat", "cool", "clean", "slice"):
                continue
            invented += w not in true_recs
good = invented == 0
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" {invented} words name something the true plan does not")


# --- PATH LENGTHENING ---------------------------------------------------------
# The turn budget is what a wrong path can actually spend: an inadmissible line
# costs one turn and no penalty (the projection checks only for <action> and
# <think> tags), and 50 turns absorb three wasted ones. `go to {recep}` is
# admissible everywhere, so each inserted step is executed.
em._WRONG_PLAN_CACHE.clear()
lens, dups, tails, invented_d = [], 0, 0, 0
for g in samp[:120]:
    d = traj(g)
    base = em._build_wrong_plan(g, "misdirect", 0)
    long = em._build_wrong_plan(g, "misdirect", 12)
    if not base or not long:
        continue
    lb, ll = path_lines(base), path_lines(long)
    lens.append(len(ll) - len(lb))
    dups += sum(1 for a, b in zip(ll, ll[1:]) if a == b)
    # nothing after the last step that changes the world
    last_real = max((i for i, l in enumerate(ll)
                     if l.startswith(("put ", "use ")) or " with " in l), default=len(ll) - 1)
    tails += (last_real != len(ll) - 1)
    if d is not None:
        true_recs = {a for h in d["plan"]["high_pddl"]
                     for a in h["discrete_action"].get("args", []) if a}
        invented_d += sum(1 for l in ll if l.startswith("go to ")
                          and l[len("go to "):] not in true_recs)

good = lens and min(lens) > 0 and dups == 0
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" detour_steps=12 adds {min(lens)}-{max(lens)} lines, {dups} immediate repeats")

good = tails == 0
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" {tails} paths end with padding after the last world-changing step")

good = invented_d == 0
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" {invented_d} inserted steps name a receptacle the true plan does not")

em._WRONG_PLAN_CACHE.clear()
good = (em._build_wrong_plan(samp[0], "intact", 12)
        == em._build_wrong_plan(samp[0], "intact", 0))
ok &= good
print(("  OK  " if good else "  FAIL") +
      " detour_steps is ignored for intact -- the true path is never padded")

good = em._oci_detour(cfg()) == 0 and em._oci_detour(None) == 0
ok &= good
print(("  OK  " if good else "  FAIL") + " detour defaults to 0 (off)")


def test_wrong_plan():
    """Collected by pytest; the checks above ran at import and set `ok`."""
    assert ok


if __name__ == "__main__":
    print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    sys.exit(0 if ok else 1)
