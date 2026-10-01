"""ALFWorld's state pointer: which line of the walkthrough a rollout owes next, read from the game.

WHY. The string pointer (env_manager._advance_guide) moves past a line whenever the action
string equals it, carried out or not. On the 2026-10-01 probe (klwctl step 150, saturated
groups) half of the document rows moved it with an action the game refused ("Nothing
happens."): the line then named steps whose preconditions were never met, ended on "All N
steps of the path are done." with the task unsolved, and every failure was such a row. It
also stalls for good once the rollout takes another instance of the object than the
walkthrough's (heat apple 1 is never typed by a row holding apple 2).

WHAT IT TRACKS. Only actions the environment carried out (an observation without "Nothing
happens"): where the rollout stands, what it holds, which target instances it has heated,
cooled or cleaned, which target instances sit in a receptacle of the destination type, whether
the lamp is on, and where it put a target instance down anywhere else.

HOW IT PLACES THE ROLLOUT. The walkthrough is cut into phases: the last "go to" before an action,
an optional "open", and the action (take / heat|cool|clean / move / use). A phase is done by
STATE: a take phase when a target instance is in hand (any instance) or enough are placed; a
treatment phase when the instance in hand is treated; a put phase by the number placed; a use
phase when the lamp is on. The owed phase is the first one not done. Its action is written for
the rollout's own instance (heat apple 2, not apple 1), for where it put an instance down, and
for the receptacle pick_two's first object went into. The owed line is the first of
action / open / go that is admissible NOW -- which also covers co-located receptacles (from
coffeemachine 1 the walkthrough takes from countertop 2: the admissible list says so, a
location model would not).

MISMATCH. Holding something that is not the target type, or nothing of the owed phase
admissible: the walkthrough has no line for that state. owed() returns None; the row shows no
progress line, and the distillation design leaves that turn out.
"""
import re
from typing import Dict, List, Optional, Set, Tuple

_GO = re.compile(r"^go to (\S+ \d+)$")
_OPEN = re.compile(r"^open (\S+ \d+)$")
_TAKE = re.compile(r"^take (\S+ \d+) from (\S+ \d+)$")
_PUT = re.compile(r"^(?:move|put) (\S+ \d+) (?:to|in|on|in/on) (\S+ \d+)$")
_TREAT = re.compile(r"^(heat|cool|clean) (\S+ \d+) with (\S+ \d+)$")
_USE = re.compile(r"^use (\S+ \d+)$")
_NOTHING = "nothing happens"


def norm(action) -> str:
    return " ".join(str(action or "").strip().lower().split())


def kind(instance: str) -> str:
    """'apple 2' -> 'apple'."""
    return str(instance).rsplit(" ", 1)[0]


def executed(observation) -> bool:
    """Did the environment carry the action out? (the same test as progress.alfworld_executed)"""
    return _NOTHING not in str(observation or "").lower()


def progress_line(n: int, idx: int, text: str) -> str:
    """The progress line, worded exactly as env_manager._guide_line words it."""
    if n <= 0:
        return ""
    done = "" if idx == 0 else (f"Step 1 of {n} is done. " if idx == 1 else f"Steps 1-{idx} of {n} are done. ")
    return (f"[Privileged Solution Path progress] {done}"
            f"Your next action is step {idx + 1} of {n}: {text}\n\n")


class AlfStatePointer:
    """One rollout's position on its walkthrough. step() after every action, owed() before every turn."""

    def __init__(self, walk):
        self.walk: List[str] = [norm(a) for a in (walk or [])]
        self.phases: List[dict] = []
        self.ok = True
        go = last_go = opn = None
        for i, a in enumerate(self.walk):
            if _GO.match(a):
                go, opn = i, None
                continue
            if _OPEN.match(a):
                opn = i
                continue
            m_take, m_put, m_treat, m_use = _TAKE.match(a), _PUT.match(a), _TREAT.match(a), _USE.match(a)
            ph = {"go": go if go is not None else last_go, "open": opn, "act": i}
            if m_take:
                ph.update(kind="take", obj=m_take.group(1), src=m_take.group(2))
            elif m_put:
                ph.update(kind="put", obj=m_put.group(1), dst=m_put.group(2))
            elif m_treat:
                ph.update(kind="treat", verb=m_treat.group(1), obj=m_treat.group(2), app=m_treat.group(3))
            elif m_use:
                ph.update(kind="use", lamp=m_use.group(1))
            else:
                self.ok = False
                ph.update(kind="unknown")
            self.phases.append(ph)
            last_go = ph["go"]
            go, opn = None, None
        takes = [p for p in self.phases if p["kind"] == "take"]
        puts = [p for p in self.phases if p["kind"] == "put"]
        treats = [p for p in self.phases if p["kind"] == "treat"]
        if not takes:
            self.ok = False
        self.target = kind(takes[0]["obj"]) if takes else ""
        self.dest = puts[0]["dst"] if puts else ""
        self.dest_type = kind(self.dest) if self.dest else ""
        self.treat = treats[0]["verb"] if treats else ""
        self.n_puts = len(puts)
        # Every receptacle the walkthrough opens, wherever the open line sits: one walkthrough
        # opens the put's drawer before the clean ("go to drawer 2", "open drawer 2", "clean fork 4
        # with sinkbasin 1", "move fork 4 to drawer 2"), so the open is not always in its phase.
        self.open_idx: Dict[str, int] = {}
        for i, a in enumerate(self.walk):
            m = _OPEN.match(a)
            if m:
                self.open_idx.setdefault(m.group(1), i)
        # The walkthrough's own instances and where it takes them from, in order.
        self.take_src: Dict[str, str] = {p["obj"]: p["src"] for p in takes}
        self.take_go: Dict[str, Optional[int]] = {p["obj"]: p["go"] for p in takes}
        self.take_order: List[str] = [p["obj"] for p in takes]
        # state
        self.loc: Optional[str] = None
        self.hold: Optional[str] = None
        self.treated: Set[str] = set()
        self.inside: Dict[str, Set[str]] = {}
        self.put_down: Dict[str, str] = {}
        self.lamp = False
        self.last: Optional[Tuple[int, str]] = None

    # --- state ---------------------------------------------------------------- #
    def step(self, action, observation) -> None:
        """Update from one turn: only an action the game carried out changes anything."""
        if not executed(observation):
            return
        a = norm(action)
        m = _GO.match(a)
        if m:
            self.loc = m.group(1)
            return
        m = _TAKE.match(a)
        if m:
            obj = m.group(1)
            self.hold = obj
            self.put_down.pop(obj, None)
            for there in self.inside.values():
                there.discard(obj)
            return
        m = _PUT.match(a)
        if m:
            obj, dst = m.group(1), m.group(2)
            if self.hold == obj:
                self.hold = None
            if (kind(obj) == self.target and self.dest_type and kind(dst) == self.dest_type
                    and (not self.treat or obj in self.treated)):
                self.inside.setdefault(dst, set()).add(obj)
            else:
                self.put_down[obj] = dst
            return
        m = _TREAT.match(a)
        if m:
            if m.group(1) == self.treat:
                self.treated.add(m.group(2))
            return
        if _USE.match(a):
            self.lamp = True

    def placed(self) -> Tuple[int, str]:
        """Most target instances in one destination-type receptacle, and that receptacle."""
        best, where = 0, self.dest
        for r, objs in self.inside.items():
            if len(objs) > best:
                best, where = len(objs), r
        return best, where

    # --- position --------------------------------------------------------------- #
    def owed(self, admissible) -> Optional[Tuple[int, str]]:
        """(walkthrough line index, action to show) for the next turn, or None on a mismatch."""
        self.last = None
        if not self.ok:
            return None
        if self.hold is not None and kind(self.hold) != self.target:
            return None
        adm = {norm(x) for x in (admissible or [])}
        n_placed, dest_now = self.placed()
        holding = self.hold is not None
        j_take = j_put = 0
        for ph in self.phases:
            k = ph["kind"]
            if k == "take":
                done = holding if self.n_puts == 0 else (n_placed >= j_take + 1 or (n_placed == j_take and holding))
                j_take += 1
            elif k == "put":
                done = n_placed >= j_put + 1
                j_put += 1
            elif k == "treat":
                done = n_placed >= 1 or (holding and self.hold in self.treated)
            elif k == "use":
                done = self.lamp
            else:
                return None
            if not done:
                self.last = self._candidates(ph, adm, dest_now)
                return self.last
        return None

    def line(self, admissible) -> str:
        o = self.owed(admissible)
        return "" if o is None else progress_line(len(self.walk), o[0], o[1])

    def _go_line(self, display_idx: Optional[int], place: str, walk_place: str,
                 walk_go_idx: Optional[int]) -> Optional[Tuple[int, str]]:
        """The phase's go line, numbered as the owed phase's own go line. Its text is the
        walkthrough's (which may name a co-located receptacle) when the place is the
        walkthrough's, else a go to where the thing now is."""
        if display_idx is None:
            return None
        if place == walk_place and walk_go_idx is not None:
            return display_idx, self.walk[walk_go_idx]
        return display_idx, f"go to {place}"

    def _take_target(self) -> Tuple[str, str]:
        """Which instance to take, and from where: a treated one the rollout put down first, then
        the walkthrough's instances not yet placed, each from wherever it is now."""
        placed_now = set().union(*self.inside.values()) if self.inside else set()
        if self.treat:
            for obj, where in self.put_down.items():
                if kind(obj) == self.target and obj in self.treated:
                    return obj, where
        for obj in self.take_order:
            if obj not in placed_now:
                return obj, self.put_down.get(obj, self.take_src[obj])
        obj = self.take_order[0]
        return obj, self.put_down.get(obj, self.take_src[obj])

    def _candidates(self, ph: dict, adm: Set[str], dest_now: str) -> Optional[Tuple[int, str]]:
        k = ph["kind"]
        cands: List[Optional[Tuple[int, str]]] = []
        if k == "take":
            obj, src = self._take_target()
            cands.append((ph["act"], f"take {obj} from {src}"))
            cands.append((self.open_idx.get(src, ph["act"]), f"open {src}"))
            cands.append(self._go_line(ph["go"] if ph["go"] is not None else ph["act"], src,
                                       self.take_src.get(obj, ph["src"]), self.take_go.get(obj, ph["go"])))
        elif k == "treat":
            cands.append((ph["act"], f"{ph['verb']} {self.hold} with {ph['app']}"))
            if ph["open"] is not None:
                cands.append((ph["open"], self.walk[ph["open"]]))
            cands.append(self._go_line(ph["go"], ph["app"], ph["app"], ph["go"]))
        elif k == "put":
            cands.append((ph["act"], f"move {self.hold} to {dest_now}"))
            cands.append((self.open_idx.get(dest_now, ph["act"]), f"open {dest_now}"))
            cands.append(self._go_line(ph["go"], dest_now, ph["dst"], ph["go"]))
        elif k == "use":
            cands.append((ph["act"], self.walk[ph["act"]]))
            if ph["go"] is not None:
                cands.append((ph["go"], self.walk[ph["go"]]))
        for c in cands:
            if c is not None and c[1] in adm:
                return c
        return None
