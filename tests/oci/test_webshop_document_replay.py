"""The WebShop document, replayed in a live environment: does it actually win?

THE ONLY CHECK THAT MATTERS FOR A CORRECT DOCUMENT. ALFWorld's walkthrough is
TextWorld's own solution and replays 30/30; WebShop's has to be built from the
goal record, so "it should win" is a claim about the reward's matcher and not
about the data. It is checked the same way: execute the document's lines
verbatim, one per turn, and read the reward.

SLOW AND OPTIONAL. Building the env loads the product file and an embedded
Lucene index; the whole file takes about a minute for 40 goals. It is skipped
wherever the data is not installed, and the fast document checks live in
tests/oci/test_documents.py.
"""
import os, sys

REPO = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(REPO, "agent_system")) and os.path.dirname(REPO) != REPO:
    REPO = os.path.dirname(REPO)
sys.path.insert(0, REPO)

WEBSHOP = os.path.join(REPO, "agent_system/environments/env_package/webshop/webshop")
DATA = os.path.join(WEBSHOP, "data")
N_GOALS = int(os.environ.get("WEBSHOP_DOC_GOALS", "40"))

ok = True


def check(good, msg):
    global ok
    ok &= bool(good)
    print(("  OK  " if good else "  FAIL") + " " + msg)


def _have_data():
    # The index is served by pyserini's LuceneSearcher, i.e. an embedded JVM, so
    # a host without a JDK cannot build the env at all -- jnius raises "Unable to
    # find javac" at import. That is a missing tool, not a failing document.
    import shutil

    return (os.path.exists(os.path.join(DATA, "items_shuffle_1000.json"))
            and os.path.isdir(os.path.join(WEBSHOP, "search_engine", "indexes"))
            and (shutil.which("javac") or os.environ.get("JAVA_HOME")))


def replay(query_mode="instruction", n=N_GOALS):
    """Run each goal's document verbatim; return (wins, rows)."""
    import random

    import gym

    sys.path.append(WEBSHOP)
    os.environ.setdefault("_JAVA_OPTIONS", "-Xmx512m -Xms64m -XX:+UseSerialGC")
    from web_agent_site.envs import WebAgentTextEnv  # noqa: F401

    from agent_system.environments.oci_layout import webshop_document_lines

    # The kwargs make_envs passes with this repo's training defaults.
    env = gym.make("WebAgentTextEnv-v0", observation_mode="text", num_products=None,
                   human_goals=False,
                   file_path=os.path.join(DATA, "items_shuffle_1000.json"),
                   attr_path=os.path.join(DATA, "items_ins_v2_1000.json"))
    goals = env.server.goals
    # The TRAIN range, which is what a training rollout draws (WebshopMultiProcessEnv).
    idxs = random.Random(0).sample(range(500, len(goals)), n)
    wins, rows = 0, []
    for idx in idxs:
        env.reset(session=idx)
        goal = env.server.user_sessions[env.session]["goal"]
        lines = webshop_document_lines(goal, query=query_mode)
        if not lines:
            rows.append((idx, "no document", 0.0))
            continue
        reward, missed = 0.0, None
        for line in lines:
            clickables = [c.lower() for c in env.get_available_actions()["clickables"]]
            if line.startswith("click[") and line[6:-1] not in clickables and missed is None:
                missed = line
            _, reward, _, _ = env.step(line)
        wins += int(reward >= 0.999)
        if reward < 0.999:
            rows.append((idx, f"reward={reward:.2f} first unavailable line: {missed}", reward))
    return wins, rows, len(idxs)


if not _have_data():
    print(f"  SKIP  no WebShop data under {DATA}; the document replay needs the 1000-product "
          f"file and its index")
else:
    wins, rows, n = replay("instruction")
    rate = wins / max(n, 1)
    check(rate >= 0.80,
          f"the document wins {wins}/{n} = {rate:.0%} of goals searching the INSTRUCTION "
          f"(the student's own prompt text)")
    # Where the rest go: the target not being on results page 1 is the search's
    # fault and is fixed by searching the title; a reward of 0.80-0.86 with every
    # line available is the reward matcher's own ceiling and no trajectory
    # through the target can beat it.
    unavailable = [r for r in rows if "first unavailable line: click[" in r[1]]
    capped = [r for r in rows if r[1].endswith("None") and r[2] > 0]
    print(f"        of the {len(rows)} that did not: {len(unavailable)} never saw the product "
          f"(search miss), {len(capped)} bought it and the matcher still capped them")
    for r in rows[:4]:
        print(f"        goal {r[0]}: {r[1][:110]}")
    if os.environ.get("WEBSHOP_DOC_BOTH", "1") == "1":
        wins_n, rows_n, n_n = replay("name")
        check(wins_n >= wins,
              f"searching the product TITLE instead wins {wins_n}/{n_n} = {wins_n / max(n_n, 1):.0%} "
              f"-- the upper bound the leakier query buys")


def test_webshop_document_replay():
    """Collected by pytest; the checks above ran at import and set `ok`."""
    assert ok


if __name__ == "__main__":
    print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    sys.exit(0 if ok else 1)
