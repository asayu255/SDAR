"""The Search document's FIRST line: does that query actually return the answer?

WHAT THIS CHECKS AND WHAT IT DOES NOT. The second line of the document is the
answer, so the rescue is certain whatever the retriever does -- exact match on
<answer> is the reward. What is not certain is whether the answer the student is
made to write is SUPPORTED by anything it can see. The first line exists for
that, and this asks the live retriever whether it delivers: a query the
retriever accepts, whose passages contain the answer.

A LOOSE BOUND ON PURPOSE. Measured over 200 questions, the verbatim question
returns the answer for 65/100 nq and 55/96 hotpotqa. The threshold here is 40%,
far below both, because the point is to catch OUR line being malformed -- a
wrong payload, a query with the tags still around it -- not to re-measure the
retriever's quality on a sample of twenty.

Skipped without the parquet or without a reachable retriever.
"""
import json, os, re, string, sys

REPO = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(REPO, "agent_system")) and os.path.dirname(REPO) != REPO:
    REPO = os.path.dirname(REPO)
sys.path.insert(0, REPO)

from agent_system.environments.oci_layout import search_document_lines  # noqa: E402

TRAIN = os.path.expanduser("~/data/verl-agent/sdar_multitask/train.parquet")
URL = os.environ.get("SEARCH_URL", "http://100.86.45.30:8000/retrieve")
N = int(os.environ.get("SEARCH_DOC_QUESTIONS", "20"))
TOPK = 3

ok = True


def check(good, msg):
    global ok
    ok &= bool(good)
    print(("  OK  " if good else "  FAIL") + " " + msg)


def norm(s):
    s = "".join(ch for ch in str(s).lower() if ch not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def strings(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from strings(v)


def _reachable():
    if not os.path.exists(TRAIN):
        return False
    try:
        import requests

        requests.post(URL, json={"query": "ping", "topk": 1, "return_scores": False}, timeout=10)
        return True
    except Exception:
        return False


if not _reachable():
    print(f"  SKIP  no {TRAIN} or no retriever at {URL}; the grounding check needs both")
else:
    import pandas as pd
    import requests

    df = pd.read_parquet(TRAIN)
    df = df[df["env_kwargs"].map(lambda e: isinstance(e, dict) and e.get("data_source") in ("nq", "hotpotqa"))]
    sample = df.sample(N, random_state=0)

    grounded, asked, malformed = 0, 0, 0
    for _, row in sample.iterrows():
        kw = row["env_kwargs"]
        lines = search_document_lines(kw.get("question"), kw.get("ground_truth"))
        if len(lines) != 2:
            malformed += 1
            continue
        m = re.fullmatch(r"<search> (.*) </search>", lines[0])
        if m is None:
            malformed += 1
            continue
        query = m.group(1)
        answer = re.fullmatch(r"<answer> (.*) </answer>", lines[1]).group(1)
        if norm(answer) in ("yes", "no"):
            continue  # nothing to retrieve; the answer is a judgement
        resp = requests.post(URL, json={"query": query, "topk": TOPK, "return_scores": False}, timeout=60)
        text = " " + norm(" ".join(strings(resp.json()))) + " "
        asked += 1
        grounded += int(f" {norm(answer)} " in text)

    check(malformed == 0, f"every row produced a two-line document in the task's grammar "
                          f"({malformed} malformed)")
    rate = grounded / max(asked, 1)
    check(rate >= 0.40,
          f"the document's own query returns the answer for {grounded}/{asked} = {rate:.0%} of "
          f"questions (measured at 65/100 nq, 55/96 hotpotqa over a larger sample)")


def test_search_document_grounding():
    """Collected by pytest; the checks above ran at import and set `ok`."""
    assert ok


if __name__ == "__main__":
    print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    sys.exit(0 if ok else 1)
