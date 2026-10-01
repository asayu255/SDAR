"""A small ALFWorld simulator for the CPU tests: places, objects, holding, open receptacles.

Its admissible list is built the way ALFWorld's is for the walkthrough's verbs, with co-location read
off the walkthrough's own "go to X" -> action-on-Y pairs (tests/oci/test_alf_state_pointer.py checks it
reproduces all 3,553 training walkthroughs).
"""
import collections
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(REPO, "agent_system")) and os.path.dirname(REPO) != REPO:
    REPO = os.path.dirname(REPO)
sys.path.insert(0, REPO)

from agent_system.environments.alf_pointer import norm, _GO, _OPEN, _TAKE, _PUT, _TREAT, _USE  # noqa: E402


class Sim:
    """One game, enough of it for the pointer: places, objects, holding, open receptacles."""

    def __init__(self, walk, extra_objects=None, extra_places=()):
        self.walk = [norm(a) for a in walk]
        self.acc = collections.defaultdict(set)
        self.objects = {}            # instance -> receptacle
        self.places = set(extra_places)
        self.openable = set()
        self.appliances = set()
        self.lamps = set()
        loc = None
        for a in self.walk:
            m = _GO.match(a)
            if m:
                loc = m.group(1)
                self.places.add(loc)
                continue
            r = None
            if _TAKE.match(a):
                o, r = _TAKE.match(a).groups()
                self.objects.setdefault(o, r)
            elif _PUT.match(a):
                r = _PUT.match(a).group(2)
            elif _OPEN.match(a):
                r = _OPEN.match(a).group(1)
                self.openable.add(r)
            elif _TREAT.match(a):
                r = _TREAT.match(a).group(3)
                self.appliances.add((_TREAT.match(a).group(1), r))
            elif _USE.match(a):
                r = _USE.match(a).group(1)
                self.lamps.add(r)
            if r:
                self.places.add(r)
                if loc:
                    self.acc[r].add(loc)
        self.objects.update(extra_objects or {})
        self.places.update(self.objects.values())
        self.loc, self.hold, self.opened = None, None, set()

    def at(self, r):
        return self.loc == r or self.loc in self.acc.get(r, ())

    def closed(self, r):
        return r in self.openable and r not in self.opened

    def admissible(self):
        out = {f"go to {x}" for x in self.places if x != self.loc}
        for o, r in self.objects.items():
            if r is not None and self.at(r) and self.hold is None and not self.closed(r):
                out.add(f"take {o} from {r}")
        if self.hold is not None:
            for r in self.places:
                if self.at(r) and not self.closed(r):
                    out.add(f"move {self.hold} to {r}")
            for verb, app in self.appliances:
                if self.at(app):
                    out.add(f"{verb} {self.hold} with {app}")
        for lamp in self.lamps:
            if self.at(lamp):
                out.add(f"use {lamp}")
        for r in self.openable:
            if self.at(r) and r not in self.opened:
                out.add(f"open {r}")
        return out

    def do(self, action):
        """Carry the action out if it is admissible; return the observation the pointer reads."""
        a = norm(action)
        if a not in self.admissible():
            return "Nothing happens."
        m = _GO.match(a)
        if m:
            self.loc = m.group(1)
        elif _TAKE.match(a):
            o, r = _TAKE.match(a).groups()
            self.hold = o
            self.objects[o] = None
        elif _PUT.match(a):
            o, r = _PUT.match(a).groups()
            self.objects[o] = r
            self.hold = None
        elif _OPEN.match(a):
            self.opened.add(_OPEN.match(a).group(1))
        return f"You did: {a}."


