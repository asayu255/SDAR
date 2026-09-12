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
# The return is BINARY (0 or 10, no partial credit) and an inadmissible action
# is not punished -- the projection validates the tags, not the admissibility --
# so the only way a wrong path turns a success into a failure is by spending the
# 50-turn budget. `go to {recep}` is admissible everywhere, so each inserted step
# is executed and costs a turn.
#
# AND THE DESTINATIONS MUST BE IRRELEVANT. A first version drew them from the
# receptacles THE PLAN NAMES, which are the object's location, the tool and the
# destination: walking round those three is the search the task requires, so the
# padding would have helped rather than hurt.
SCENE = ["go to cabinet %d" % i for i in range(1, 11)] + [
    "go to countertop 1", "go to diningtable 1", "go to drawer 1", "go to drawer 2",
    "go to fridge 1", "go to garbagecan 1", "go to microwave 1", "go to sinkbasin 1",
    "go to stoveburner 1", "go to stoveburner 2", "go to stoveburner 3",
    "go to stoveburner 4", "go to toaster 1", "take mug 1 from cabinet 1", "look"]

em._SCENE_RECEPS.clear()
recs = em._scene_receptacles("TESTGAME", SCENE)
good = len(recs) == 23 and "cabinet 3" in recs and "look" not in recs
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" {len(recs)} scene receptacles parsed from the admissible actions, numbered")

added, dups, tails, off_plan, n = [], 0, 0, 0, 0
for g in samp[:120]:
    d = traj(g)
    if d is None:
        continue
    base = em._build_wrong_plan(g, "misdirect", 0)
    long = em._build_wrong_plan(g, "misdirect", 50, recs)
    if not base or not long:
        continue
    n += 1
    lb, ll = path_lines(base), path_lines(long)
    added.append(len(ll) - len(lb))
    dups += sum(1 for a, b in zip(ll, ll[1:]) if a == b)
    last_real = max((i for i, l in enumerate(ll)
                     if l.startswith(("put ", "use ")) or " with " in l), default=len(ll) - 1)
    tails += (last_real != len(ll) - 1)
    plan_recs = {a for h in d["plan"]["high_pddl"]
                 for a in h["discrete_action"].get("args", []) if a}
    inserted = [l[len("go to "):] for l in ll
                if l.startswith("go to ") and l[len("go to "):] not in plan_recs]
    off_plan += all(x.rsplit(" ", 1)[0] not in plan_recs for x in inserted) if inserted else 0

good = n > 50 and min(added) > 20 and dups == 0
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" detour=50 adds {min(added)}-{max(added)} lines over {n} games, "
      f"{dups} immediate repeats")

good = off_plan == n
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" {off_plan}/{n} games send every inserted step to a receptacle the plan "
      f"does NOT name")

good = tails == 0
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" {tails} paths end with padding after the last world-changing step")

# without a scene list there is nowhere irrelevant to go, so the path is left alone
good = em._build_wrong_plan(samp[0], "misdirect", 50) == em._build_wrong_plan(samp[0], "misdirect", 0)
ok &= good
print(("  OK  " if good else "  FAIL") +
      " with no admissible actions to read, the detour is skipped rather than "
      "drawn from the plan's own receptacles")

em._WRONG_PLAN_CACHE.clear()
good = (em._build_wrong_plan(samp[0], "intact", 50, recs)
        == em._build_wrong_plan(samp[0], "intact", 0))
ok &= good
print(("  OK  " if good else "  FAIL") + " the true path is never padded")

good = em._oci_detour(cfg()) == 0 and em._oci_detour(None) == 0
ok &= good
print(("  OK  " if good else "  FAIL") + " detour defaults to 0 (off)")


def test_wrong_plan():
    """Collected by pytest; the checks above ran at import and set `ok`."""
    assert ok


if __name__ == "__main__":
    print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    sys.exit(0 if ok else 1)
