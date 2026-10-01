"""CPU test: ALFWorld's state pointer (agent_system/environments/alf_pointer.py).

WHAT IT MUST DO. Name the walkthrough line a rollout owes from what the game actually did:
not move on an action the game refused ("Nothing happens."), follow the rollout's own
instance of the object, find an instance it put down somewhere else, count pick_two's two
objects whatever their order, take co-located receptacles from the admissible list, and say
nothing (None) when the walkthrough has no line for the state.

HOW. A small simulator of one game: locations, what is held, what is open, where each object
is. Its admissible list is built the way ALFWorld's is for these verbs, with co-location read
off the walkthrough's own "go to X" -> action-on-Y pairs. Then:
  1. every training walkthrough on this host, followed exactly: the owed line is the next
     walkthrough line at every step (an "open" may be owed later than the walkthrough puts it,
     since the pointer opens a receptacle only when the next action needs it);
  2. hand-written scenarios for each behaviour above.
"""
import collections, glob, json, os, re, sys

REPO = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(REPO, "agent_system")) and os.path.dirname(REPO) != REPO:
    REPO = os.path.dirname(REPO)
sys.path.insert(0, REPO)

from agent_system.environments.alf_pointer import (AlfStatePointer, norm, progress_line,  # noqa: E402
                                                   _GO, _OPEN, _TAKE, _PUT, _TREAT, _USE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alf_sim import Sim  # noqa: E402

ok = True


def check(good, msg):
    global ok
    ok &= bool(good)
    print(("  OK  " if good else "  FAIL") + " " + msg)


def run(sim, ptr, actions):
    owed = []
    for a in actions:
        owed.append(ptr.owed(sim.admissible()))
        ptr.step(a, sim.do(a))
    owed.append(ptr.owed(sim.admissible()))
    return owed


print("1. every training walkthrough, followed exactly")
games = sorted(glob.glob(os.path.expanduser("~/data/alfworld/json_2.1.1/train/*/*/game.tw-pddl")))
if not games:
    print("SKIP: no alfworld games on this host")
else:
    fails = collections.Counter()
    example = {}
    for g in games:
        walk = [norm(a) for a in (json.load(open(g)).get("walkthrough") or [])]
        sim, ptr = Sim(walk), AlfStatePointer(walk)
        good = True
        for k, a in enumerate(walk):
            o = ptr.owed(sim.admissible())
            fine = o is not None and o == (k, a)
            if not fine and _OPEN.match(a) and o is not None and o[1] in walk[k + 1:]:
                fine = True          # the pointer opens when the next action needs it
            if not fine:
                good = False
                t = g.split("/")[-3].split("-")[0]
                fails[t] += 1
                example.setdefault(t, (k, a, o, walk))
                break
            ptr.step(a, sim.do(a))
        if good and ptr.owed(sim.admissible()) is not None:
            fails["not finished"] += 1
    check(not fails, f"{len(games)} walkthroughs: owed = next line at every step ({dict(fails) or 'no failures'})")
    for t, ex in list(example.items())[:4]:
        print("     ", t, ex)

print("2. scenarios")
heat = ["go to sinkbasin 1", "take apple 1 from sinkbasin 1", "go to microwave 1",
        "heat apple 1 with microwave 1", "go to countertop 2", "move apple 1 to countertop 2"]

# 2a. a refused action does not move it (the string pointer's failure)
sim, ptr = Sim(heat, extra_places=["cabinet 1"]), AlfStatePointer(heat)
o = run(sim, ptr, ["go to cabinet 1", "take apple 1 from sinkbasin 1", "go to sinkbasin 1"])
check(o[1] == (0, "go to sinkbasin 1") and o[2] == (0, "go to sinkbasin 1")
      and o[3] == (1, "take apple 1 from sinkbasin 1"),
      "typing the take from the wrong place does not count; arriving does")

# 2b. another instance: the rest of the path follows the rollout's apple
sim = Sim(heat, extra_objects={"apple 2": "countertop 1"})
ptr = AlfStatePointer(heat)
o = run(sim, ptr, ["go to countertop 1", "take apple 2 from countertop 1", "go to microwave 1",
                   "heat apple 2 with microwave 1", "go to countertop 2"])
check(o[2] == (2, "go to microwave 1") and o[3] == (3, "heat apple 2 with microwave 1")
      and o[4] == (4, "go to countertop 2") and o[5] == (5, "move apple 2 to countertop 2"),
      f"apple 2 taken elsewhere: heat apple 2, move apple 2 ({o[2]}, {o[3]}, {o[4]}, {o[5]})")

# 2c. holding something that is not the target: no line
sim = Sim(heat, extra_objects={"mug 1": "countertop 1"})
ptr = AlfStatePointer(heat)
o = run(sim, ptr, ["go to countertop 1", "take mug 1 from countertop 1"])
check(o[2] is None and ptr.line(sim.admissible()) == "", "holding a mug in an apple task: mismatch, no line")

# 2d. the walkthrough's apple put down elsewhere: take it from there
sim, ptr = Sim(heat, extra_places=["diningtable 1"]), AlfStatePointer(heat)
o = run(sim, ptr, ["go to sinkbasin 1", "take apple 1 from sinkbasin 1", "go to diningtable 1",
                   "move apple 1 to diningtable 1"])
check(o[4] == (1, "take apple 1 from diningtable 1"),
      f"apple 1 left on the dining table: take it from there ({o[4]})")

# 2e. a treated apple put down elsewhere is preferred over a fresh one
sim, ptr = Sim(heat, extra_places=["diningtable 1"]), AlfStatePointer(heat)
o = run(sim, ptr, ["go to sinkbasin 1", "take apple 1 from sinkbasin 1", "go to microwave 1",
                   "heat apple 1 with microwave 1", "go to diningtable 1", "move apple 1 to diningtable 1"])
check(o[6] == (1, "take apple 1 from diningtable 1"), f"the heated apple is taken back first ({o[6]})")

# 2f. pick_two in the other order
two = ["go to shelf 1", "take alarmclock 1 from shelf 1", "go to desk 1", "move alarmclock 1 to desk 1",
       "go to dresser 1", "take alarmclock 2 from dresser 1", "go to desk 1", "move alarmclock 2 to desk 1"]
sim, ptr = Sim(two), AlfStatePointer(two)
o = run(sim, ptr, ["go to dresser 1", "take alarmclock 2 from dresser 1", "go to desk 1",
                   "move alarmclock 2 to desk 1"])
check(o[2] == (2, "go to desk 1") and o[3] == (3, "move alarmclock 2 to desk 1")
      and o[4] == (4, "go to shelf 1"),
      f"alarmclock 2 placed first: the second take is alarmclock 1 at shelf 1 ({o[2]}, {o[3]}, {o[4]})")
sim2 = Sim(two)
ptr2 = AlfStatePointer(two)
o2 = run(sim2, ptr2, ["go to dresser 1", "take alarmclock 2 from dresser 1", "go to desk 1",
                      "move alarmclock 2 to desk 1", "go to shelf 1"])
check(o2[5] == (5, "take alarmclock 1 from shelf 1"), f"...and at the shelf, take it ({o2[5]})")

# 2g. co-location: the walkthrough takes from countertop 2 standing at coffeemachine 1
cool = ["go to coffeemachine 1", "take apple 2 from countertop 2", "go to fridge 1", "cool apple 2 with fridge 1",
        "go to coffeemachine 1", "move apple 2 to countertop 2"]
sim, ptr = Sim(cool), AlfStatePointer(cool)
o = run(sim, ptr, ["go to coffeemachine 1"])
check(o[0] == (0, "go to coffeemachine 1") and o[1] == (1, "take apple 2 from countertop 2"),
      "at coffeemachine 1 the take from countertop 2 is owed (the admissible list says it is possible)")

# 2h. a closed receptacle: open it first
fridge = ["go to fridge 1", "open fridge 1", "take apple 1 from fridge 1", "go to microwave 1",
          "heat apple 1 with microwave 1", "go to countertop 2", "move apple 1 to countertop 2"]
sim, ptr = Sim(fridge), AlfStatePointer(fridge)
o = run(sim, ptr, ["go to fridge 1", "open fridge 1"])
check(o[1] == (1, "open fridge 1") and o[2] == (2, "take apple 1 from fridge 1"), "closed fridge: open, then take")

# 2i. look_at in either order
look = ["go to desk 1", "take alarmclock 1 from desk 1", "go to dresser 1", "use desklamp 1"]
sim, ptr = Sim(look), AlfStatePointer(look)
o = run(sim, ptr, ["go to desk 1", "take alarmclock 1 from desk 1"])
check(o[2] == (2, "go to dresser 1"), f"holding the clock: go to the lamp ({o[2]})")
look2 = ["go to dresser 1", "use desklamp 1", "take alarmclock 2 from dresser 1"]
sim, ptr = Sim(look2, extra_objects={"alarmclock 1": "desk 1"}), AlfStatePointer(look2)
o = run(sim, ptr, ["go to desk 1", "take alarmclock 1 from desk 1"])
check(o[2] == (0, "go to dresser 1"), f"clock 1 taken first: go to the lamp ({o[2]})")

# 2j. the line's wording is the string pointer's
check(progress_line(6, 2, "go to microwave 1")
      == "[Privileged Solution Path progress] Steps 1-2 of 6 are done. Your next action is step 3 of 6: go to microwave 1\n\n"
      and progress_line(6, 0, "go to x 1").startswith("[Privileged Solution Path progress] Your next action is step 1 of 6"),
      "same wording as env_manager._guide_line")

print("3. through the manager: the document row on the state pointer, the second on the string one")
heat_games = sorted(glob.glob(os.path.expanduser(
    "~/data/alfworld/json_2.1.1/train/pick_heat_then_place_in_recep*/*/game.tw-pddl")))
gf = next((g for g in heat_games
           if [norm(a) for a in json.load(open(g)).get("walkthrough") or []][:2]
           and _GO.match(norm(json.load(open(g))["walkthrough"][0]))
           and _TAKE.match(norm(json.load(open(g))["walkthrough"][1]))), None)
if gf is None:
    print("SKIP: no heat game on this host")
else:
    from types import SimpleNamespace
    import tempfile
    import agent_system.environments.env_manager as em
    from agent_system.environments import oci_layout as ol
    walk = [norm(a) for a in json.load(open(gf))["walkthrough"]]
    other = next(a[6:] for a in walk[2:] if _GO.match(a))      # somewhere else on the path
    N = 6

    class SimEnvs:
        group_n, is_train = N, True

        def __init__(self):
            self.sims = [Sim(walk) for _ in range(N)]

        @property
        def get_admissible_commands(self):
            return [sorted(s.admissible()) for s in self.sims]

        def reset(self):
            obs = ["You are in the middle of a room.\n\nYour task is to: heat some apple and put it somewhere."] * N
            return obs, None, [{"extra.gamefile": gf, "won": False} for _ in range(N)]

        def step(self, actions):
            obs = [s.do(a) for s, a in zip(self.sims, actions)]
            return obs, None, [0.0] * N, [False] * N, [{"extra.gamefile": gf, "won": False} for _ in range(N)]

    slots = {"enable": True, "tasks": ["alfworld"], "doc_mode": "walkthrough_stepwise", "foreign_task": "webshop",
             "gamma": 0.1, "opd_on_special": False, "doc_b": True, "foreign_slot": False,
             "doc_pointer": "state", "doc_pointer_b": "string"}
    cfg = SimpleNamespace(env=SimpleNamespace(history_length=2), algorithm={"oci_slots": slots})
    dump = tempfile.mkdtemp(prefix="alf_ptr_")
    os.environ["ALFWORLD_PROBE_DUMP"] = dump
    em._WRONG_PLAN_CACHE.clear()
    m = em.AlfWorldEnvironmentManager(SimEnvs(), lambda acts, adm: (list(acts), [1] * len(acts)), cfg)
    o0, _ = m.reset(kwargs=None)
    D, B = 5, 4
    check(o0[ol.OCI_ROLE_KEY][B] == ol.ROLE_DOC_B and o0[ol.OCI_ROLE_KEY][D] == ol.ROLE_DOC,
          "rows 4 and 5 are the two document rows")
    check(f"step 1 of {len(walk)}: {walk[0]}" in o0["text"][D] and f"step 1 of {len(walk)}: {walk[0]}" in o0["text"][B],
          "turn 1: both pointers owe the first line")
    m.step([walk[0]] * N)                       # go to the take's place: both move on
    o2, *_ = m.step([f"go to {other}"] * N)     # walk away
    check(f"step 1 of {len(walk)}: {walk[0]}" in o2["text"][D]
          and f"step 2 of {len(walk)}: {walk[1]}" in o2["text"][B],
          "after walking away: the state row owes the go back, the string row still the take")
    o3, *_ = m.step([walk[1]] * N)              # the take, typed from the wrong place: refused
    check(f"step 1 of {len(walk)}: {walk[0]}" in o3["text"][D]
          and f"step 3 of {len(walk)}: {walk[2]}" in o3["text"][B],
          "a refused take: the string row moves on (the bug), the state row does not")
    m._alf_probe_flush()
    os.environ.pop("ALFWORLD_PROBE_DUMP", None)
    recs = [json.loads(l) for f in glob.glob(os.path.join(dump, "*.jsonl")) for l in open(f)]
    by = {r["role"]: r for r in recs}
    t3d, t3b = by[ol.ROLE_DOC]["turns"][2], by[ol.ROLE_DOC_B]["turns"][2]
    check(by[ol.ROLE_DOC]["pointer"] == "state" and by[ol.ROLE_DOC_B]["pointer"] == "string"
          and by[ol.ROLE_PLAIN]["pointer"] is None,
          "the record names each row's pointer")
    check(t3d["executed"] is False and t3d["shown"] == [0, walk[0]] and t3d["state_owed"] == [0, walk[0]]
          and t3b["shown"] == [1, walk[1]] and t3b["state_owed"] == [0, walk[0]],
          "turn 3 recorded: refused; what each row was shown; the state pointer's line as a shadow on the string row")
    check(by[ol.ROLE_PLAIN]["turns"][0]["shown"] is None and by[ol.ROLE_PLAIN]["turns"][0]["executed"] is True,
          "plain rows: no line, execution still recorded")

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
