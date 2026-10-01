"""WebShop's state pointer: which line of the goal's purchase path a rollout owes next, read from the page.

The document (oci_layout.webshop_document_lines) is
    search[<query>], click[<goal asin>], click[<option value>] ..., click[buy now]
and the row's state is the page it is on (the worker's available_actions: a search bar, the clickables)
and the session (the worker's ws_state: the product it holds and one value per option name).

    a search bar                                    owes line 1, the search
    a results page listing the goal product         owes line 2, its click
    the goal product's item page (buy now offered)  owes the first required option the session does
                                                    not hold yet, else the purchase
    anything else -- another product's page, a results page without the goal product, a
    description / features / reviews sub page       MISMATCH: the path has no line for it

The design (2026-10-01): the path's search line is the goal's instruction text, and a mismatched turn
gets no progress line and no distillation.
"""
from collections import Counter
from typing import Optional, Sequence, Tuple

from agent_system.environments.alf_pointer import progress_line

_BUY = "buy now"


class WebshopStatePointer:
    def __init__(self, lines: Sequence[str], goal_asin: str, goal_options):
        self.lines = list(lines or [])
        self.asin = str(goal_asin or "").strip().lower()
        if isinstance(goal_options, dict):
            self.pairs = {str(n).strip().lower(): str(v).strip().lower()
                          for n, v in goal_options.items() if str(v).strip()}
            self.values = list(self.pairs.values())
        else:
            self.pairs = None
            self.values = [str(v).strip().lower() for v in (goal_options or []) if str(v).strip()]
        self.ok = bool(self.lines) and bool(self.asin) and len(self.lines) == 3 + len(self.values)
        self.last: Optional[Tuple[int, str]] = None

    def _held(self, session_options) -> list:
        """The required values the session holds now, one per option name."""
        held = {str(n).strip().lower(): str(v).strip().lower() for n, v in dict(session_options or {}).items()}
        if self.pairs is not None:
            return [v for n, v in self.pairs.items() if held.get(n) == v]
        have = Counter(held.values())
        out = []
        for v in self.values:
            if have[v] > 0:
                have[v] -= 1
                out.append(v)
        return out

    def owed(self, available_actions, ws_state) -> Optional[Tuple[int, str]]:
        self.last = None
        if not self.ok:
            return None
        avail = available_actions or {}
        clickables = {str(c).strip().lower() for c in (avail.get("clickables") or [])}
        state = ws_state if isinstance(ws_state, dict) else {}
        session_asin = str(state.get("asin") or "").strip().lower()
        if avail.get("has_search_bar"):
            self.last = (0, self.lines[0])
        elif self.asin in clickables:
            self.last = (1, self.lines[1])
        elif session_asin == self.asin and _BUY in clickables:
            held = Counter(self._held(state.get("options")))
            for j, v in enumerate(self.values):
                if held[v] > 0:
                    held[v] -= 1
                    continue
                if v in clickables:
                    self.last = (2 + j, self.lines[2 + j])
                break
            else:
                self.last = (len(self.lines) - 1, self.lines[-1])
        return self.last

    def line(self, available_actions, ws_state) -> str:
        o = self.owed(available_actions, ws_state)
        return "" if o is None else progress_line(len(self.lines), o[0], o[1])
