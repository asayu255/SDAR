"""Search's k in three levels (algorithm.progress_rank.search_k = evidence_answered).

WHAT IT PROTECTS.
  * The count: 0 not seen / 1 seen, never answered / 2 seen and answered, K = 2;
    an answer sent WITHOUT the evidence stays at 0 (a guess is never paid for);
    unjudgeable questions (yes/no, no answer string) have K = 0 as before.
  * The switch: search_k defaults to evidence (today's count, bit for bit in
    progress_k); only evidence_answered moves progress_k; a typo is refused.
  * The manager: both counts and the answered flag are written on EVERY row,
    whichever count is active; "answered" is the PROJECTED action (what the
    environment ends the episode on), so a turn carrying <search> and <answer>
    is a search; the flag is sticky.
  * The loop: the new columns reach the rows, NaN where a task has none.
  * The controller: the records carry "answered" per trajectory, and the metrics
    say how many of the rollouts (a) pushed up -- and of those it pushed down --
    never answered, in stuck groups and among mixed-group failures.
No model, no retriever: fake environments throughout.
"""
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(REPO, "verl")) and os.path.dirname(REPO) != REPO:
    REPO = os.path.dirname(REPO)
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from agent_system.environments import progress as P  # noqa: E402
from agent_system.environments.oci_layout import answer_strings, is_yesno  # noqa: E402

ok = True


def check(good, msg):
    global ok
    ok &= bool(good)
    print(("  OK  " if good else "  FAIL") + " " + msg)


def config(on=True, search_k=None, n=2):
    pr = {"enable": on}
    if search_k is not None:
        pr["search_k"] = search_k
    return OmegaConf.create({
        "env": {"history_length": 0, "rollout": {"n": n}},
        "algorithm": {"progress_rank": pr,
                      "oci_sat": {"enable": False}, "oci_slots": {"enable": False},
                      "oci_rank": {"enable": False}},
    })


print("1. the count")
kw = dict(answer_strings=answer_strings, is_yesno=is_yesno)
T = {"target": ["Paris"]}
check(P.search_progress_answered(False, False, T, **kw) == (0, 2), "not seen, no answer: 0 of 2")
check(P.search_progress_answered(False, True, T, **kw) == (0, 2), "a guess without the evidence stays at 0")
check(P.search_progress_answered(True, False, T, **kw) == (1, 2), "seen, never answered: 1")
check(P.search_progress_answered(True, True, T, **kw) == (2, 2), "seen, then answered (right or wrong): 2")
check(P.search_progress_answered(True, True, {"target": ["yes"]}, **kw) == (0, 0), "yes/no: nothing to count")
check(P.search_progress_answered(True, True, {"target": []}, **kw) == (0, 0), "no answer string: nothing to count")
# the order it adds is exactly one relation over today's count
cls = [(False, False), (False, True), (True, False), (True, True)]
ev = [P.search_progress(s, T, **kw)[0] for s, _ in cls]
ea = [P.search_progress_answered(s, a, T, **kw)[0] for s, a in cls]
agree = all((ev[i] > ev[j]) <= (ea[i] > ea[j]) for i in range(4) for j in range(4))
check(agree, "every pair today's count ranks, three levels rank the same way")
check(ea[3] > ea[2] and ev[3] == ev[2], "...and it adds one: seen-and-answered above seen-never-answered")

print("2. the switch")
check(P.search_k_definition(config()) == "evidence", "absent: evidence, today's count")
check(P.search_k_definition(config(search_k="evidence_answered")) == "evidence_answered", "evidence_answered")
try:
    P.search_k_definition(config(search_k="answered"))
    check(False, "a typo is refused")
except AssertionError:
    check(True, "a typo is refused")

print("3. the manager")
from agent_system.environments.env_manager import SearchEnvironmentManager  # noqa: E402
from agent_system.environments.env_package.search.projection import search_projection  # noqa: E402

QUESTION = "what is the capital of france"


class _SearchEnvs:
    """Row 0's results carry the answer, row 1's never do; answers end nothing here."""
    group_n = 2
    is_train = True

    def reset(self, kwargs=None):
        return [QUESTION] * 2, [{} for _ in range(2)]

    def step(self, actions):
        obs = ["<information> Paris is the capital of France. </information>", "<information> Lyon. </information>"]
        return obs, [0.0, 0.0], [False, False], [{"won": 0.0} for _ in actions]


def run(search_k, turns):
    mgr = SearchEnvironmentManager(_SearchEnvs(), search_projection, config(True, search_k))
    mgr.reset([{"question": QUESTION, "ground_truth": {"target": ["Paris"]}}] * 2)
    out = []
    for acts in turns:
        _, _, _, infos = mgr.step(acts)
        out.append(infos)
    return out


SEARCH = "<search> capital of france </search>"
turns = [[SEARCH, SEARCH], ["<answer> Paris </answer>", "<answer> Lyon </answer>"]]
for sk in (None, "evidence_answered"):
    t1, t2 = run(sk, turns)
    check([i["progress_k_search_evidence"] for i in t1] == [1, 0]
          and [i["progress_total_search_evidence"] for i in t1] == [1, 1],
          f"search_k={sk}: the evidence count is always written")
    check([i["progress_k_search_answered"] for i in t1] == [1, 0]
          and [i["progress_total_search_answered"] for i in t1] == [2, 2]
          and [i["search_answered"] for i in t1] == [False, False],
          f"search_k={sk}: after searching, three levels read 1 / 0, nobody answered yet")
    check([i["progress_k_search_answered"] for i in t2] == [2, 0] and [i["search_answered"] for i in t2] == [True, True],
          f"search_k={sk}: both answered; the one that saw it reaches 2, the guess stays at 0")
    if sk is None:
        check([i["progress_k"] for i in t2] == [1, 0] and t2[0]["progress_total"] == 1,
              "default: progress_k is today's count, unchanged")
    else:
        check([i["progress_k"] for i in t2] == [2, 0] and t2[0]["progress_total"] == 2,
              "evidence_answered: progress_k is the three-level count, K = 2")
t1, t2 = run("evidence_answered", [[SEARCH, SEARCH], [SEARCH + " <answer> Paris </answer>", SEARCH]])
check([i["search_answered"] for i in t2] == [False, False]
      and [i["progress_k_search_answered"] for i in t2] == [1, 0],
      "a turn carrying <search> and <answer> is a search (the projection keeps the search)")
t = run("evidence_answered", [["<answer> Paris </answer>", SEARCH], [SEARCH, SEARCH]])
check([i["search_answered"] for i in t[1]] == [True, False], "the flag is sticky once set")
check(t[1][0]["progress_k_search_answered"] == 1,
      "an answer sent BEFORE the evidence arrived is not the second stage (k stays 1 once it is seen)")
mgr = SearchEnvironmentManager(_SearchEnvs(), search_projection, config(False, "evidence_answered"))
mgr.reset([{"question": QUESTION, "ground_truth": {"target": ["Paris"]}}] * 2)
_, _, _, infos = mgr.step([SEARCH, SEARCH])
check(all("search_answered" not in i and "progress_k_search_answered" not in i for i in infos),
      "progress_rank off: nothing is written")

print("4. the rollout loop carries the columns")
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector  # noqa: E402
from verl import DataProto  # noqa: E402

c = TrajectoryCollector.__new__(TrajectoryCollector)
c._progress_rank_on = True
c._queue_row_for_prefetch = lambda *a: None
batch = DataProto.from_single_dict({"input_ids": torch.zeros(2, 3, dtype=torch.long),
                                    "traj_uid": np.array(["a", "b"], dtype=object)})
tbl, tinf = [[], []], [[], []]
c._record_turn(batch=batch, active_idx=np.array([0, 1]), active_masks=np.array([True, True]),
               infos=[dict(t2[0]), {}], traj_uid=np.array(["a", "b"], dtype=object),
               total_batch_list=tbl, total_infos=tinf, batch_size=2)
r0, r1 = tbl[0][0], tbl[1][0]
check(r0["progress_k_search_answered"] == 1.0 and r0["progress_total_search_answered"] == 2.0
      and r0["progress_k_search_evidence"] == 1.0 and r0["search_answered"] == 0.0,
      "a Search row carries both counts and the flag")
check(all(np.isnan(r1[k]) for k in ("progress_k_search_answered", "progress_total_search_answered",
                                     "progress_k_search_evidence", "search_answered")),
      "a row of another task gets NaN, so every row has the same columns")

print("5. the controller: the records and the push-side metrics")
from verl.trainer.ppo import progress_rank as pr  # noqa: E402


def build(groups, resp=4):
    """``[(uid, [(traj, turns, reward, k, K, adv, answered), ...]), ...]``, all Search."""
    cols = {k: [] for k in ("uids", "tuids", "rew", "ks", "Ks", "adv", "ans")}
    for uid, trajs in groups:
        for traj, turns, r, k, K, a, an in trajs:
            for _ in range(turns):
                for key, v in zip(cols, (uid, traj, r, k, K, a, an)):
                    cols[key].append(v)
    n = len(cols["uids"])
    mask = torch.zeros(n, resp)
    mask[:, :3] = 1.0
    obj = lambda v: np.array(v, dtype=object)  # noqa: E731
    return dict(advantages=torch.tensor(cols["adv"], dtype=torch.float32).unsqueeze(-1) * mask, mask=mask,
                uids=obj(cols["uids"]), tuids=obj(cols["tuids"]), task_names=obj(["search"] * n),
                episode_rewards=obj(cols["rew"]), k_rows=obj(cols["ks"]), total_rows=obj(cols["Ks"]),
                real_rows=np.ones(n, dtype=bool), stat_rows=np.ones(n, dtype=bool),
                answered_rows=np.array(cols["ans"], dtype=float))


# A live group (gives the scales), a stuck group ranked by today's count -- the two
# that saw the answer are pushed up, one of them never answered -- and a mixed group
# whose two failures differ in k, the higher one never answered.
LIVE = ("live", [("L1", 2, 1.0, 1, 1, 0.8, 1.0), ("L2", 3, 0.0, 0, 1, -0.8, 1.0)])
STUCK = ("stuck", [("S1", 2, 0.0, 1, 1, 0.0, 1.0), ("S2", 4, 0.0, 1, 1, 0.0, 0.0),
                   ("S3", 2, 0.0, 0, 1, 0.0, 1.0), ("S4", 2, 0.0, 0, 1, 0.0, 1.0)])
MIX = ("mix", [("M1", 2, 1.0, 1, 1, 0.9, 1.0), ("M2", 4, 0.0, 1, 1, -0.3, 0.0),
               ("M3", 2, 0.0, 0, 1, -0.3, 1.0)])
ctl = pr.ProgressRankController(rho=0.1, mixed_rho=0.1, tasks=["alfworld", "webshop", "search"],
                                min_top_k={"search": 1})
b = build([LIVE, STUCK, MIX])
new, m = ctl.apply(**b)
recs = {r["uid"]: r for r in ctl.last_group_records}
check(recs["stuck"]["answered"] == [True, False, True, True], "the stuck group's record carries answered per trajectory")
check(recs["stuck"]["verdict"] == "fired", "the stuck group fires on today's count")
check(m.get("progress_rank/search/stuck_up_n") == 2.0 and m.get("progress_rank/search/stuck_up_unanswered") == 0.5,
      f"pushed up: 2, of which never answered 1/2 ({m.get('progress_rank/search/stuck_up_unanswered')})")
check(m.get("progress_rank/search/stuck_down_n") == 2.0 and m.get("progress_rank/search/stuck_down_unanswered") == 0.0,
      "pushed down: 2, all answered")
if recs["mix"]["mixed_verdict"] == "fired":
    check(m.get("progress_rank/search/mixed_up_unanswered") == 1.0 and m.get("progress_rank/search/mixed_down_unanswered") == 0.0,
          "mixed failures: the one pushed up never answered, the one pushed down did")
else:
    check(False, f"the mixed group should fire (verdict {recs['mix']['mixed_verdict']})")
b2 = build([LIVE, STUCK, MIX])
b2.pop("answered_rows")
ctl2 = pr.ProgressRankController(rho=0.1, mixed_rho=0.1, tasks=["alfworld", "webshop", "search"],
                                 min_top_k={"search": 1})
new2, m2 = ctl2.apply(**b2)
check(torch.equal(new, new2), "the flag never touches the advantage")
check(not any("unanswered" in k for k in m2) and all(r["answered"] == [None] * len(r["trajs"]) for r in ctl2.last_group_records),
      "without the column: no push-side metric, answered is None in the records")

print("\nPASS" if ok else "\nFAIL")
sys.exit(0 if ok else 1)
