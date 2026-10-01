"""CPU test: the one-sentence probe's alfworld layout (doc_b, foreign_slot, doc_sentence).

WHAT THE PROBE NEEDS. Two document rows in one group -- the walkthrough with the
exploration or efficiency sentence, and the same walkthrough without it -- so the
sentence is read as a paired difference on the same game; no foreign slot (its rows
cost a 50-turn episode and the probe does not read them); and one JSON line per
rollout (ALFWORLD_PROBE_DUMP) with every turn's action, so turns, repeats and the
think block can be counted.

Checked: the roles; which row sees which sentence and where it sits; that both
document rows keep the pointer; that the ordinary rows see nothing; the success keys;
the per-rollout record; and that the default layout is unchanged.
"""
import glob, json, os, sys, tempfile
from types import SimpleNamespace

REPO = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(REPO, "agent_system")) and os.path.dirname(REPO) != REPO:
    REPO = os.path.dirname(REPO)
sys.path.insert(0, REPO)

import agent_system.environments.env_manager as em  # noqa: E402
from agent_system.environments import oci_layout as ol  # noqa: E402
from verl.trainer.ppo import oci_slots as osl  # noqa: E402

ok = True


def check(good, msg):
    global ok
    ok &= bool(good)
    print(("  OK  " if good else "  FAIL") + " " + msg)


def cfg(**over):
    slots = {"enable": True, "tasks": ["alfworld"], "doc_mode": "walkthrough_stepwise",
             "foreign_task": "webshop", "gamma": 0.1, "opd_on_special": False}
    slots.update(over)
    return SimpleNamespace(env=SimpleNamespace(history_length=2), algorithm={"oci_slots": slots})


print("1. roles")
check([ol.role_for_slot(i, 10, foreign=False, second_doc=True) for i in range(10)]
      == [ol.ROLE_PLAIN] * 7 + [ol.ROLE_RESERVE, ol.ROLE_DOC_B, ol.ROLE_DOC],
      "ten slots without the foreign slot: seven plain, reserve, the bare document, the document + sentence")
check(ol.has_second_doc(cfg(doc_b=True)) and not ol.has_second_doc(cfg())
      and not ol.foreign_slot_on(cfg(foreign_slot=False)) and ol.foreign_slot_on(cfg()),
      "doc_b turns the second document row on; foreign_slot=False turns the foreign slot off; defaults unchanged")

games = sorted(glob.glob(os.path.expanduser(
    "~/data/alfworld/json_2.1.1/train/pick_heat_then_place_in_recep*/*/game.tw-pddl")))
gf = next((g for g in games if len(json.load(open(g)).get("walkthrough") or []) >= 5), None)
if gf is None:
    print("SKIP: no long alfworld walkthrough on this host")
else:
    walk = json.load(open(gf))["walkthrough"]
    N = 6

    class FakeEnvs:
        group_n, is_train = N, True
        get_admissible_commands = [["look", "inventory"]] * N

        def __init__(self):
            self.t = 0

        def reset(self):
            obs = ["You are in the middle of a room.\n\nYour task is to: heat some mug and put it somewhere."] * N
            return obs, None, [{"extra.gamefile": gf, "won": False}] * N

        def step(self, actions):
            self.t += 1
            won = [self.t >= 2 and i == N - 1 for i in range(N)]
            return ([f"You did: {a}." for a in actions], None, [float(w) for w in won],
                    [w or self.t >= 3 for w in won], [{"extra.gamefile": gf, "won": w} for w in won])

    print("2. the manager")
    em._WRONG_PLAN_CACHE.clear()
    dump = tempfile.mkdtemp(prefix="alf_probe_")
    os.environ["ALFWORLD_PROBE_DUMP"] = dump
    c = cfg(doc_b=True, foreign_slot=False, doc_sentence="explore", doc_sentence_b="none")
    m = em.AlfWorldEnvironmentManager(FakeEnvs(), lambda acts, adm: (list(acts), [1] * len(acts)), c)
    obs, _ = m.reset(kwargs=None)
    text, role = obs["text"], obs[ol.OCI_ROLE_KEY]
    EXPLORE = ol.DOC_SENTENCES["alfworld"]["explore"]
    check(role == [ol.ROLE_PLAIN, ol.ROLE_PLAIN, ol.ROLE_PLAIN, ol.ROLE_RESERVE, ol.ROLE_DOC_B, ol.ROLE_DOC],
          f"the manager marks the probe layout ({role})")
    check(EXPLORE in text[5] and EXPLORE not in text[4] and all(EXPLORE not in t for t in text[:4]),
          "only the document row carries the sentence")
    check(all("[Privileged Solution Path]" in text[i] for i in (4, 5))
          and all("[Privileged Solution Path]" not in t for t in text[:4]),
          "both document rows show the walkthrough; the plain rows and the reserve see nothing")
    check(text[5].index(EXPLORE) < text[5].index(ol.PLAN_FOOTER)
          and text[5].index(f"{len(walk)}. ") < text[5].index(EXPLORE),
          "the sentence is the block's last line, after the numbered steps")
    check(all(f"step 1 of {len(walk)}: {walk[0]}" in text[i] for i in (4, 5)),
          "both rows get the pointer line at step 1")
    o1, *_ = m.step(["look", "look", "look", "look", walk[0], walk[0]])
    check(m._guide_ptr == [0, 0, 0, 0, 1, 1] and all(f"step 2 of {len(walk)}: {walk[1]}" in o1["text"][i]
                                                    for i in (4, 5)),
          f"taking the owed step moves both document rows' pointers ({m._guide_ptr})")
    m.step(["look"] * N)
    m.step(["look"] * N)
    m._alf_probe_flush()
    recs = [json.loads(l) for f in glob.glob(os.path.join(dump, "*.jsonl")) for l in open(f)]
    by_role = {r["role"]: r for r in recs}
    check(len(recs) == N and by_role[ol.ROLE_DOC]["sentence"] == "explore"
          and by_role[ol.ROLE_DOC_B]["sentence"] == "none" and by_role[ol.ROLE_PLAIN]["sentence"] is None,
          "one record per rollout, with the sentence its row wore")
    check(by_role[ol.ROLE_DOC]["won"] == 1.0 and by_role[ol.ROLE_DOC]["n_turns"] == 2
          and by_role[ol.ROLE_DOC]["turns"][0]["action"] == walk[0]
          and set(by_role[ol.ROLE_DOC]["turns"][0]) == {"action", "valid", "think", "chars"},
          "turns carry the projected action, admissibility, the think flag and the length; the win is read")
    os.environ.pop("ALFWORLD_PROBE_DUMP", None)
    batch_list = [[{"active_masks": True}] for _ in range(N)]
    infos = [[{"won": w, "extra.gamefile": gf}] for w in (False, True, False, False, True, True)]
    succ = m.success_evaluator(total_infos=infos, total_batch_list=batch_list,
                               episode_rewards=None, episode_lengths=None)
    check(succ["success_rate"].tolist() == [0.0, 1.0, 0.0, 0.0]
          and succ["oci_doc_success_rate"].tolist() == [1.0]
          and succ["oci_doc_b_success_rate"].tolist() == [1.0],
          "the ordinary slots make success_rate; each document row gets its own key")

    print("2b. the short lead (doc_lead / doc_lead_b)")
    em._WRONG_PLAN_CACHE.clear()
    c2 = cfg(doc_b=True, foreign_slot=False, doc_sentence="progress", doc_sentence_b="progress",
             doc_lead="short", doc_lead_b="strict")
    m2 = em.AlfWorldEnvironmentManager(FakeEnvs(), lambda acts, adm: (list(acts), [1] * len(acts)), c2)
    t2, _ = m2.reset(kwargs=None)
    a_txt, b_txt = t2["text"][5], t2["text"][4]
    import re as _re
    check(ol.PLAN_LEAD_SHORT in a_txt and "YOU MUST FOLLOW IT EXACTLY" not in a_txt
          and "YOU MUST FOLLOW IT EXACTLY" in b_txt and ol.PLAN_LEAD_SHORT not in b_txt,
          "the document row reads the short lead, the second document row the strict one")
    check(_re.findall(r"^\d+\. (.+)$", a_txt, flags=_re.M) == _re.findall(r"^\d+\. (.+)$", b_txt, flags=_re.M)
          and all(f"step 1 of {len(walk)}: {walk[0]}" in x for x in (a_txt, b_txt))
          and ol.DOC_SENTENCES["alfworld"]["progress"] in a_txt,
          "the numbered steps, the pointer and the sentence are the same under either lead")
    check(ol.with_doc_lead("", c2) == "" and ol.doc_lead_key(cfg()) == "strict",
          "no block passes through; the default lead is the strict one")

    print("3. the default layout is untouched")
    em._WRONG_PLAN_CACHE.clear()

    class FakeEnvs4(FakeEnvs):
        group_n = 4
        get_admissible_commands = [["look", "inventory"]] * 4

        def reset(self):
            obs = ["You are in the middle of a room.\n\nYour task is to: heat some mug and put it somewhere."] * 4
            return obs, None, [{"extra.gamefile": gf, "won": False}] * 4

    m4 = em.AlfWorldEnvironmentManager(FakeEnvs4(), lambda acts, adm: (list(acts), [1] * len(acts)), cfg())
    o4, _ = m4.reset(kwargs=None)
    check(o4[ol.OCI_ROLE_KEY] == [ol.ROLE_PLAIN, ol.ROLE_RESERVE, ol.ROLE_DOC, ol.ROLE_FOREIGN]
          and o4["text"][2].startswith(em._build_wrong_plan(gf, "walkthrough")),
          "no doc_b, foreign on, no sentence: the old layout and the old block, byte for byte")

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
