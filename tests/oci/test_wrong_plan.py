"""CPU test for the corrupted-plan prefix: reads real ALFWorld games, no GPU.

WHAT THE FIRST MEASUREMENT TAUGHT THIS FILE. The original mode removed the
plan's requirement step and this file checked that exactly one step was gone and
that no unnamed object appeared. Both held on 300 games -- and the mechanism
still did nothing, because the step being removed is the one the TASK
DESCRIPTION already states. The invariants were about the edit's shape and said
nothing about whether the edit destroys information the student needs. The
checks below are about the latter.
"""
import glob, json, os, random, re, sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import agent_system.environments.env_manager as em

GAMES = sorted(glob.glob(os.path.expanduser(
    "~/data/alfworld/json_2.1.1/train/*/*/game.tw-pddl")))
if not GAMES:
    print("SKIP: no alfworld games on this host"); sys.exit(0)

WORDS = {
    "CleanObject":  ("clean", "wash", "rinse"),
    "HeatObject":   ("heat", "hot", "warm", "microwav", "cook"),
    "CoolObject":   ("cool", "cold", "chill", "fridge", "refrigerat"),
    "SliceObject":  ("slice", "cut", "chop"),
    "ToggleObject": ("turn on", "lamp", "light", "switch"),
}


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


ok = True

# --- the switch ------------------------------------------------------------
em._WRONG_PLAN_CACHE.clear()
good = em._wrong_plan_prefix("alfworld", GAMES[0], cfg(enable=False)) == ""
ok &= good
print(("  OK  " if good else "  FAIL") + " switch off (config) yields nothing")

good = em._wrong_plan_prefix("alfworld", GAMES[0], None) == ""
ok &= good
print(("  OK  " if good else "  FAIL") + " no config at all yields nothing")

good = em._wrong_plan_prefix("webshop", GAMES[0], cfg()) == ""
ok &= good
print(("  OK  " if good else "  FAIL") + " another task yields nothing")

try:
    em._oci_plan_mode(cfg(mode="shuffle"))
    good = False
except ValueError as e:
    good = "plan_corruption" in str(e)
ok &= good
print(("  OK  " if good else "  FAIL") + " an unknown corruption mode is refused by name")

# --- misdirect: coverage and size ------------------------------------------
em._WRONG_PLAN_CACHE.clear()
random.seed(0)
samp = random.sample(GAMES, min(250, len(GAMES)))
lens, empty = [], 0
for g in samp:
    p = em._wrong_plan_prefix("alfworld", g, cfg())
    (lens.append(len(p)) if p else None)
    empty += (not p)
good = empty <= len(samp) * 0.02
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" {len(samp)-empty}/{len(samp)} games corruptible ({empty} have <2 nav targets)")
lens.sort()
good = lens and lens[len(lens)//2] < 1200
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" block size: p50 {lens[len(lens)//2]} chars, max {lens[-1]}")

# --- THE POINT: every navigation step is sent somewhere it should not be ----
moved, checked, invented = 0, 0, 0
for g in samp[:120]:
    d = traj(g)
    if d is None:
        continue
    p = em._wrong_plan_prefix("alfworld", g, cfg())
    if not p:
        continue
    true_nav = [h["discrete_action"]["args"][0]
                for h in d["plan"]["high_pddl"]
                if h["discrete_action"]["action"] == "GotoLocation" and h["discrete_action"]["args"]]
    shown_nav = re.findall(r"\d+\. GotoLocation\(([^)]*)\)", p)
    if len(shown_nav) != len(true_nav):
        continue
    checked += 1
    moved += all(a != b for a, b in zip(shown_nav, true_nav))
    invented += any(a not in set(true_nav) for a in shown_nav)
good = checked > 50 and moved == checked and invented == 0
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" {moved}/{checked} games have EVERY navigation step redirected, and "
      f"{invented} invent a receptacle the true plan does not name")

# --- and the steps themselves are all still there --------------------------
same_shape = 0
for g in samp[:120]:
    d = traj(g)
    if d is None:
        continue
    p = em._wrong_plan_prefix("alfworld", g, cfg())
    if not p:
        continue
    true_acts = [h["discrete_action"]["action"] for h in d["plan"]["high_pddl"]]
    shown_acts = re.findall(r"\d+\. (\w+)\(", p)
    same_shape += (shown_acts == true_acts)
good = same_shape == checked
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" {same_shape}/{checked} keep the step count and action sequence exactly "
      f"(only the destinations move)")

# --- the refuted mode, and WHY it is refuted, kept as a regression ----------
em._WRONG_PLAN_CACHE.clear()
named, total = 0, 0
for g in samp[:200]:
    d = traj(g)
    if d is None:
        continue
    acts = [h["discrete_action"]["action"] for h in d["plan"]["high_pddl"]]
    req = next((a for a in em._REQUIREMENT_ACTIONS if a in acts), None)
    if req is None:
        continue
    desc = d["turk_annotations"]["anns"][0]["task_desc"].lower()
    total += 1
    named += any(w in desc for w in WORDS[req])
good = total > 20 and named / total > 0.8
ok &= good
print(("  OK  " if good else "  FAIL") +
      f" 'drop' removes a step the task text already states in {named}/{total} "
      f"({named/max(total,1):.0%}) games -- which is why it did not induce failure")

# and misdirect does not have that property: the destinations are not in the
# task description in any useful way, because the task names the DESTINATION of
# the put, not where the object starts
em._WRONG_PLAN_CACHE.clear()
p = em._wrong_plan_prefix("alfworld", samp[0], cfg(mode="drop"))
q = em._wrong_plan_prefix("alfworld", samp[0], cfg(mode="misdirect"))
good = bool(p and q and p != q)
ok &= good
print(("  OK  " if good else "  FAIL") + " the two modes produce different text")

good = "### SOLUTION PLAN (training only) ###" in q and q.endswith("\n\n")
ok &= good
print(("  OK  " if good else "  FAIL") + " the header and trailing blank line are unchanged")


def test_wrong_plan():
    """Collected by pytest; the checks above ran at import and set `ok`."""
    assert ok


if __name__ == "__main__":
    print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    sys.exit(0 if ok else 1)
