"""Run qualified production lanes on independent local clocks."""

import json
from collections import deque

from .controller import INTERACT, MOTION, STAY


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class LocalLane:
    def __init__(self, actor, lane, component, selector):
        self.actor = actor
        self.lane = lane
        self.component = set(component)
        self.selector = selector
        self.index = 0
        self.pending = deque()
        self.atom = None
        self.calls = self.nodes = self.edges = 0

    def snapshot(self):
        return {
            "lane_index": self.index,
            "pending_actions": list(self.pending),
            "pending_atom": None if self.atom is None else self.atom.data(),
        }

    def route(self, player, station):
        self.calls += 1
        start = (player.position, player.orientation)
        queue = deque([start])
        parent = {start: None}
        chosen = None
        while queue:
            p, o = queue.popleft()
            self.nodes += 1
            if (p[0] + o[0], p[1] + o[1]) == tuple(station):
                chosen = (p, o)
                break
            for a in MOTION:
                self.edges += 1
                q = (p[0] + a[0], p[1] + a[1])
                v = (q if q in self.component else p, o if a == STAY else a)
                if v not in parent:
                    parent[v] = ((p, o), a)
                    queue.append(v)
        if chosen is None:
            raise RuntimeError("NO_LOCAL_MOTION_ROUTE")
        actions = []
        while parent[chosen] is not None:
            chosen, a = parent[chosen]
            actions.append(a)
        return list(reversed(actions))

    def next(self, state):
        if self.pending:
            return self.pending.popleft()
        atom = self.lane[self.index]
        if not self.selector.ready(atom, state):
            return STAY
        path = self.route(state.players[self.actor], atom.station)
        self.atom = atom
        self.pending = deque(path + [STAY if atom.kind == "approach" else INTERACT])
        return self.pending.popleft()

    def observe(self, before, after, action):
        if self.atom is None or self.pending:
            return
        atom = self.atom
        p = after.players[self.actor]
        held = p.held_object.name if p.held_object else None
        obj = after.objects.get(atom.station)
        if atom.kind != "approach" and action != INTERACT:
            raise RuntimeError("LOCAL_PHASE_MISMATCH")
        ok = {
            "pickup": lambda: held == atom.item,
            "put_onion": lambda: held is None and obj is not None,
            "start": lambda: obj is not None and not obj.is_idle,
            "scoop": lambda: held == "soup" and obj is None,
            "approach": lambda: (
                held == "dish"
                and (p.position[0] + p.orientation[0], p.position[1] + p.orientation[1])
                == tuple(atom.station)
            ),
            "deliver": lambda: held is None,
        }[atom.kind]()
        if not ok:
            raise RuntimeError("NATIVE_LOCAL_ATOM_FAILED:" + compact(atom.data()))
        self.index = (self.index + 1) % len(self.lane)
        self.atom = None

    def work(self):
        return {
            "BFS_calls": self.calls,
            "expanded_local_poses": self.nodes,
            "tested_local_motion_edges": self.edges,
            "native_calls_during_search": 0,
        }


def local_projection(state_dict, actor, pot):
    objects = [o for o in state_dict["objects"] if tuple(o["position"]) == tuple(pot)]
    return {
        "indexed_actor": actor,
        "player": state_dict["players"][actor],
        "reserved_pot": list(pot),
        "pot_objects": objects,
    }
