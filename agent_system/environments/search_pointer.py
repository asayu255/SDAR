"""Search's state pointer: which search of a verified route a rollout owes next, read from what came back.

WHY. The route pointer (oci_layout.advance_route) moves past a search when the row's query shares half
of the route query's words. A paraphrase that brought nothing back moves it; the same page reached by
another query does not. The design (tied-group self-distillation, 2026-10-01) places every row from its
STATE instead, as ALFWorld's alf_pointer does: a step of the route is done when what it fetches is in
the results the row has received, whichever query fetched it.

WHAT EACH STEP FETCHES.
  last step          the answer: a returned result carries it (the manager's own evidence test, passed
                     in as ``found``); the line then says where it is, never what it says.
  an earlier step    the name the next query is built from: the content words of query s+1 that are
                     neither in the question nor in queries 1..s ("From a Window" -> "Billy J. Kramer").
                     Done when at least half of them have appeared in a returned result. A next query
                     that adds no new word cannot be checked by content and falls back to the old rule:
                     done when the row ran a query sharing half of step s's words.
  a yes/no route     verified by titles, one page per step: step s is done when its page came back
                     (the manager's titles_seen); those steps are parallel, so the owed one is the first
                     page not seen yet, not the furthest.

POSITION. A chain owes the step after the furthest one done; a yes/no route the first one not done; a
found answer owes the answer. Every query can always be typed, so a routed question has no mismatched turn.
"""
import re
from typing import List, Optional, Sequence, Set

from agent_system.environments.oci_layout import fold_text, question_words, route_query_matches

_SEARCH = re.compile(r"<search>(.*?)</search>", flags=re.S)


def route_queries(lines: Sequence[str]) -> List[str]:
    out = []
    for line in lines or []:
        m = _SEARCH.search(str(line))
        out.append(m.group(1).strip() if m else str(line).strip())
    return out


def _content(text) -> List[str]:
    return [w for w in fold_text(text).split() if len(w) > 2]


class SearchStatePointer:
    """One rollout's position on its route. observe() every returned block; ptr() before every turn."""

    def __init__(self, lines: Sequence[str], question: str, titles: Optional[Sequence[str]] = None,
                 min_share: float = 0.5):
        self.lines = list(lines or [])
        self.queries = route_queries(self.lines)
        self.titles = [fold_text(t) for t in (titles or [])] or None
        self.min_share = float(min_share)
        qw = question_words(question)
        seen: Set[str] = set()
        self.targets: List[Set[str]] = []
        for s, q in enumerate(self.queries):
            seen |= set(_content(q))
            if s + 1 < len(self.queries):
                nxt = set(_content(self.queries[s + 1])) - qw - seen
                self.targets.append(nxt)
            else:
                self.targets.append(set())          # the last step fetches the answer
        self.done = [False] * len(self.queries)
        self._text = ""

    def observe(self, returned_text, query: Optional[str] = None) -> None:
        """Fold in one returned block (and the query that fetched it, for the content-free fallback)."""
        if returned_text:
            self._text += " " + fold_text(returned_text) + " "
        for s in range(len(self.queries) - 1):
            if self.done[s]:
                continue
            want = self.targets[s]
            if want:
                hit = sum(1 for w in want if f" {w} " in self._text)
                self.done[s] = hit / len(want) >= self.min_share
            elif query is not None and route_query_matches(query, self.lines[s], self.min_share):
                self.done[s] = True

    def ptr(self, found: bool, titles_seen=None) -> int:
        """The index of the owed search (len(lines) = the answer is owed)."""
        n = len(self.queries)
        if n == 0 or found:
            return n
        if self.titles:
            seen = set(titles_seen or ())
            for s, t in enumerate(self.titles[:n]):
                if t not in seen:
                    return s
            return n
        furthest = max((s for s in range(n - 1) if self.done[s]), default=-1)
        return min(furthest + 1, n - 1)
