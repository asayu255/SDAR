"""Ask a running retrieval server the questions that matter before a run uses it.

A 200 from /retrieve is not evidence that the server is usable. The failures
this exists to catch have all happened:

  * an index and a corpus whose row numbers do not line up, which answers 200
    with documents that have nothing to do with the query;
  * a batched request whose results come back in the wrong order, which hands
    each environment another environment's documents and shows up nowhere;
  * concurrent requests that the server cannot serve at once, which is how
    eleven of them timed out together after 600 s with the server still
    accepting connections;
  * a ceiling on how many queries fit in one request, which is not a constant
    and has to be measured on the box that will serve them.

Usage:
    python examples/search/retriever/check_retriever.py
    python examples/search/retriever/check_retriever.py --url http://100.86.45.30:8000/retrieve
    python examples/search/retriever/check_retriever.py --wait 900   # right after launching it

Exit status is 0 only if the content, ordering and concurrency checks pass. The
size probe never fails the run: provoking a refusal is how it finds the ceiling.
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests


def snippet(entry, width=100):
    """One line out of whatever shape the server returned for a document."""
    doc = entry["document"] if isinstance(entry, dict) and "document" in entry else entry
    text = doc["contents"] if isinstance(doc, dict) else str(doc)
    score = entry.get("score") if isinstance(entry, dict) else None
    return score, " ".join(text.split())[:width]


def ask(url, query, topk=3, timeout=180):
    started = time.perf_counter()
    response = requests.post(
        url,
        json={"query": query, "topk": topk, "return_scores": True},
        timeout=timeout,
    )
    return response, (time.perf_counter() - started) * 1000


def wait_for(url, seconds, timeout):
    """Poll until the server answers. Loading a Flat wiki-18 index takes minutes."""
    deadline = time.monotonic() + seconds
    while True:
        try:
            ask(url, "warmup", timeout=min(timeout, 60))
            return True
        except requests.exceptions.RequestException as exc:
            if time.monotonic() >= deadline:
                print(f"    still not answering after {seconds}s: {type(exc).__name__}")
                return False
            time.sleep(5)


def check_content(url, timeout):
    """Documents come back, with text in them."""
    response, ms = ask(url, "who wrote the book zero the biography of a dangerous idea", timeout=timeout)
    print(f"[1] single      status={response.status_code}  {ms:7.0f} ms")
    if response.status_code != 200:
        print("   ", response.text[:400])
        return False
    result = response.json()["result"]
    print(f"    entries={len(result)}  docs={len(result[0])}")
    if len(result) != 1 or not result[0]:
        print("    FAIL: a one-query request must come back as one non-empty entry")
        return False
    for position, entry in enumerate(result[0]):
        score, text = snippet(entry)
        print(f"    doc{position + 1}  score={score}  {text!r}")
    return True


def check_order(url, timeout):
    """A batch comes back one entry per query, in the order it was asked."""
    queries = [
        "what is the capital of france",
        "who wrote hamlet",
        "what is the tallest mountain in the world",
    ]
    response, ms = ask(url, queries, timeout=timeout)
    print(f"\n[2] batch       status={response.status_code}  {ms:7.0f} ms  n={len(queries)}")
    if response.status_code != 200:
        print("   ", response.text[:400])
        return False
    result = response.json()["result"]
    if len(result) != len(queries):
        print(f"    FAIL: {len(result)} entries for {len(queries)} queries")
        return False
    tops = []
    for query, documents in zip(queries, result):
        if not documents:
            print(f"    FAIL: no documents for {query!r}")
            return False
        text = snippet(documents[0], width=70)[1]
        tops.append(text)
        print(f"    {query[:38]:38s} -> {text!r}")
    if len(set(tops)) != len(queries):
        # Not proof of a mix-up, but three unrelated questions answered by the
        # same passage is what a misaligned corpus looks like from out here.
        print("    FAIL: the same passage came back for different questions")
        return False
    return True


def check_concurrency(url, workers, timeout):
    """The server survives being asked several things at once."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        started = time.perf_counter()
        codes = list(
            pool.map(
                lambda i: ask(url, f"who won the world cup in {1950 + i}", timeout=timeout)[0].status_code,
                range(workers),
            )
        )
        wall = (time.perf_counter() - started) * 1000
    served = sum(1 for code in codes if code == 200)
    print(f"\n[3] {workers} concurrent  {served}/{workers} served  {wall:7.0f} ms wall")
    if served != workers:
        print(f"    FAIL: status codes {sorted(set(codes))}")
    return served == workers


def probe_sizes(url, sizes, timeout):
    """Find how many queries fit in one request. A refusal here is the answer."""
    print("\n[4] request size")
    largest_served = None
    for n in sizes:
        queries = [f"who wrote the book number {i}" for i in range(n)]
        try:
            response, ms = ask(url, queries[0] if n == 1 else queries, timeout=timeout)
            print(f"    n={n:5d}  status={response.status_code}  {ms:7.0f} ms")
            if response.status_code != 200:
                break
            largest_served = n
        except requests.exceptions.RequestException as exc:
            print(f"    n={n:5d}  {type(exc).__name__}: {exc}")
            break
    if largest_served is not None:
        print(f"    largest served: {largest_served}")
    return largest_served


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8000/retrieve", help="the /retrieve endpoint to check")
    parser.add_argument("--timeout", type=int, default=180, help="per-request timeout in seconds")
    parser.add_argument("--concurrency", type=int, default=16, help="how many requests to send at once (0 to skip)")
    parser.add_argument(
        "--sizes",
        default="1,44,128,252,384,512,1024",
        help="request sizes to probe, largest last; empty to skip",
    )
    parser.add_argument("--wait", type=int, default=0, help="seconds to wait for the server to come up first")
    args = parser.parse_args()

    print(f"checking {args.url}")
    if args.wait:
        print(f"[0] waiting up to {args.wait}s for the server")
        if not wait_for(args.url, args.wait, args.timeout):
            return 1

    try:
        passed = check_content(args.url, args.timeout)
        passed = check_order(args.url, args.timeout) and passed
        if args.concurrency:
            passed = check_concurrency(args.url, args.concurrency, args.timeout) and passed
    except requests.exceptions.RequestException as exc:
        print(f"    {type(exc).__name__}: {exc}")
        print("\nthe server did not answer. Is it running, and has it finished loading the index?")
        return 1

    if args.sizes.strip():
        probe_sizes(args.url, [int(n) for n in args.sizes.split(",")], args.timeout)

    print("\nOK" if passed else "\nFAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
