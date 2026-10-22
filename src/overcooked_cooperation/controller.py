"""Deterministic lane selection and native joint-motion scheduling."""

from __future__ import annotations

import itertools
import json
from collections import deque
from dataclasses import dataclass

DIRS = ((0, -1), (0, 1), (1, 0), (-1, 0))
STAY = (0, 0)
MOTION = DIRS + (STAY,)
INTERACT = "interact"


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def add(p, d):
    return (p[0] + d[0], p[1] + d[1])


@dataclass(frozen=True)
class Atom:
    agent: int
    kind: str
    station: tuple
    item: str = ""

    def data(self):
        return [self.agent, self.kind, list(self.station), self.item]


class Motion:
    """Finite joint-motion BFS, with native orientation and collision rules."""

    def __init__(self, terrain):
        self.floor = frozenset(
            (x, y)
            for y, row in enumerate(terrain)
            for x, c in enumerate(row)
            if c == " "
        )
        self.stations = {
            c: sorted(
                (x, y)
                for y, row in enumerate(terrain)
                for x, t in enumerate(row)
                if t == c
            )
            for c in "ODPSX"
        }
        self.neighbors = {
            p: tuple(q for d in DIRS if (q := add(p, d)) in self.floor)
            for p in self.floor
        }
        self.dist = {}
        for p in sorted(self.floor):
            seen = {p: 0}
            queue = deque([p])
            while queue:
                u = queue.popleft()
                for v in self.neighbors[u]:
                    if v not in seen:
                        seen[v] = seen[u] + 1
                        queue.append(v)
            self.dist[p] = seen
        self.calls = self.nodes = self.edges = self.cache_hits = 0
        self.max_frontier = self.max_nodes_per_call = 0
        self.cache = {}
        self.single_transitions = {}
        for p in sorted(self.floor):
            for o in DIRS:
                for a in MOTION:
                    q = add(p, a)
                    self.single_transitions[(p, o, a)] = (
                        q if q in self.floor else p,
                        o if a == STAY else a,
                    )

    def poses(self, station):
        return tuple(
            (p, d)
            for d in DIRS
            if (p := (station[0] - d[0], station[1] - d[1])) in self.floor
        )

    def reachable(self, start, station):
        return any(p in self.dist[start] for p, _ in self.poses(station))

    def station_dist(self, a, b):
        ds = [
            self.dist[p][q]
            for p, _ in self.poses(a)
            for q, _ in self.poses(b)
            if q in self.dist[p]
        ]
        return min(ds) if ds else 10**6

    def nearest(self, start_station, choices, agent_start):
        feasible = [s for s in choices if self.reachable(agent_start, s)]
        return (
            min(feasible, key=lambda s: (self.station_dist(start_station, s), s))
            if feasible
            else None
        )

    def successor(self, pose, actions):
        p0, o0, p1, o1 = pose
        q0, n0 = self.single_transitions[(p0, o0, actions[0])]
        q1, n1 = self.single_transitions[(p1, o1, actions[1])]
        # Native: any same-square or swap collision cancels both translations,
        # while both new orientations are retained.
        if q0 == q1 or (q0 == p1 and q1 == p0):
            q0, q1 = p0, p1
        return (q0, n0, q1, n1)

    def search(self, start, atoms):
        goals = {a.agent: frozenset(self.poses(a.station)) for a in atoms}
        goal_key = tuple(sorted((i, tuple(sorted(v))) for i, v in goals.items()))
        cache_key = (start, goal_key)
        self.calls += 1
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]

        def done(s):
            return all((s[2 * i], s[2 * i + 1]) in ps for i, ps in goals.items())

        queue = deque([start])
        previous = {start: None}
        terminal = None
        while queue:
            self.max_frontier = max(self.max_frontier, len(queue))
            s = queue.popleft()
            self.nodes += 1
            if done(s):
                terminal = s
                break
            for actions in itertools.product(MOTION, repeat=2):
                self.edges += 1
                t = self.successor(s, actions)
                if t not in previous:
                    previous[t] = (s, actions)
                    queue.append(t)
        self.max_nodes_per_call = max(self.max_nodes_per_call, len(previous))
        path = None
        if terminal is not None:
            reverse = []
            while previous[terminal] is not None:
                terminal, actions = previous[terminal]
                reverse.append(actions)
            path = tuple(reversed(reverse))
        # Bounded cache to avoid unbounded resident-memory use.
        if len(self.cache) < 2048:
            self.cache[cache_key] = path
        return path

    def stats(self):
        return {
            "bfs_calls": self.calls,
            "cache_hits": self.cache_hits,
            "expanded_states": self.nodes,
            "tested_joint_motion_edges": self.edges,
            "max_frontier": self.max_frontier,
            "max_discovered_states_one_bfs": self.max_nodes_per_call,
            "joint_motion_state_bound": len(self.floor) * (len(self.floor) - 1) * 16,
            "native_calls_during_motion_search": 0,
        }


def supply_route(motion, starts, source_type, dest_actor, dest_station, item):
    """Choose direct supply, otherwise one shared-counter handoff, statically."""
    choices = []
    for source in motion.stations[source_type]:
        if motion.reachable(starts[dest_actor], source):
            cost = 2 * motion.station_dist(source, dest_station) + 2
            choices.append(
                (cost, compact([source]), [Atom(dest_actor, "pickup", source, item)])
            )
        other = 1 - dest_actor
        if not motion.reachable(starts[other], source):
            continue
        for counter in motion.stations["X"]:
            if all(motion.reachable(starts[a], counter) for a in (0, 1)):
                cost = (
                    2 * motion.station_dist(source, counter)
                    + 2 * motion.station_dist(counter, dest_station)
                    + 4
                )
                route = [
                    Atom(other, "pickup", source, item),
                    Atom(other, "drop", counter, item),
                    Atom(dest_actor, "take", counter, item),
                ]
                choices.append((cost, compact([source, counter]), route))
    return min(choices, key=lambda x: (x[0], x[1])) if choices else None


def delivery_route(motion, starts, actor, pot):
    choices = []
    for serving in motion.stations["S"]:
        if motion.reachable(starts[actor], serving):
            choices.append(
                (
                    motion.station_dist(pot, serving) + 1,
                    compact([serving]),
                    [Atom(actor, "deliver", serving, "soup")],
                )
            )
        other = 1 - actor
        if not motion.reachable(starts[other], serving):
            continue
        for counter in motion.stations["X"]:
            if all(motion.reachable(starts[a], counter) for a in (0, 1)):
                route = [
                    Atom(actor, "drop", counter, "soup"),
                    Atom(other, "take", counter, "soup"),
                    Atom(other, "deliver", serving, "soup"),
                ]
                choices.append(
                    (
                        motion.station_dist(pot, counter)
                        + motion.station_dist(counter, serving)
                        + 3,
                        compact([serving, counter]),
                        route,
                    )
                )
    return min(choices, key=lambda x: (x[0], x[1])) if choices else None


def solo_cycle(motion, starts, actor, pot):
    o = motion.nearest(pot, motion.stations["O"], starts[actor])
    d = motion.nearest(pot, motion.stations["D"], starts[actor])
    s = motion.nearest(pot, motion.stations["S"], starts[actor])
    if None in (o, d, s) or not motion.reachable(starts[actor], pot):
        return None
    atoms = []
    for _ in range(3):
        atoms += [
            Atom(actor, "pickup", o, "onion"),
            Atom(actor, "put_onion", pot, "onion"),
        ]
    atoms += [
        Atom(actor, "start", pot),
        Atom(actor, "pickup", d, "dish"),
        Atom(actor, "approach", pot, "dish"),
        Atom(actor, "scoop", pot, "dish"),
        Atom(actor, "deliver", s, "soup"),
    ]
    # Static route-work estimate is only deterministic resource selection,
    # never a certificate or a post-result policy-selection criterion.
    estimate = (
        6 * motion.station_dist(o, pot)
        + 7
        + max(20, motion.station_dist(pot, d) * 2 + 2)
        + 2 * motion.station_dist(pot, s)
        + 2
    )
    return estimate, atoms


class UniversalPolicy:
    def __init__(self, motion, starts):
        self.motion, self.starts = motion, starts
        self.lanes = []
        self.indices = []
        self.pending_actions = deque()
        self.pending_atoms = []
        self.pending_lane_ids = []
        self.mode = ""
        self.selection = {}
        self.choose()

    def choose(self):
        # Universal preference 1: two distinct productive pot reservations.
        candidates = []
        for p0, p1 in itertools.permutations(self.motion.stations["P"], 2):
            a = solo_cycle(self.motion, self.starts, 0, p0)
            b = solo_cycle(self.motion, self.starts, 1, p1)
            if a and b:
                candidates.append((max(a[0], b[0]), a[0] + b[0], p0, p1, a[1], b[1]))
        if candidates:
            v = min(candidates, key=lambda x: x[:4])
            self.mode = "two_reserved_pots_parallel_recipe_loops"
            self.lanes = [v[4], v[5]]
            self.selection = {"pots": [v[2], v[3]], "static_estimates": [v[0], v[1]]}
        else:
            # Universal preference 2: a shared pot with complementary roles.
            candidates = []
            for pot in self.motion.stations["P"]:
                for filler, server in ((0, 1), (1, 0)):
                    o = self.motion.nearest(
                        pot, self.motion.stations["O"], self.starts[filler]
                    )
                    d = self.motion.nearest(
                        pot, self.motion.stations["D"], self.starts[server]
                    )
                    s = self.motion.nearest(
                        pot, self.motion.stations["S"], self.starts[server]
                    )
                    if None in (o, d, s) or not all(
                        self.motion.reachable(self.starts[a], pot) for a in (0, 1)
                    ):
                        continue
                    fill = []
                    for _ in range(3):
                        fill += [
                            Atom(filler, "pickup", o, "onion"),
                            Atom(filler, "put_onion", pot, "onion"),
                        ]
                    fill += [Atom(filler, "start", pot)]
                    serve = [
                        Atom(server, "pickup", d, "dish"),
                        Atom(server, "approach", pot, "dish"),
                        Atom(server, "scoop", pot, "dish"),
                        Atom(server, "deliver", s, "soup"),
                    ]
                    cost = (
                        6 * self.motion.station_dist(o, pot)
                        + 2 * self.motion.station_dist(pot, d)
                        + 2 * self.motion.station_dist(pot, s)
                    )
                    candidates.append((cost, pot, filler, fill, serve))
            if candidates:
                v = min(candidates, key=lambda x: x[:3])
                self.mode = "one_reserved_pot_complementary_parallel_loops"
                self.lanes = [v[3], v[4]]
                self.selection = {"pot": v[1], "filler": v[2], "server": 1 - v[2]}
            else:
                # General single-counter relay fallback. A single serialized
                # recipe lane gives explicit cargo/counter reservations.
                candidates = []
                for pot in self.motion.stations["P"]:
                    for actor in (0, 1):
                        if not self.motion.reachable(self.starts[actor], pot):
                            continue
                        o = supply_route(
                            self.motion, self.starts, "O", actor, pot, "onion"
                        )
                        d = supply_route(
                            self.motion, self.starts, "D", actor, pot, "dish"
                        )
                        s = delivery_route(self.motion, self.starts, actor, pot)
                        if not all((o, d, s)):
                            continue
                        route = []
                        for _ in range(3):
                            route += o[2] + [Atom(actor, "put_onion", pot, "onion")]
                        route += (
                            [Atom(actor, "start", pot)]
                            + d[2]
                            + [
                                Atom(actor, "approach", pot, "dish"),
                                Atom(actor, "scoop", pot, "dish"),
                            ]
                            + s[2]
                        )
                        candidates.append((3 * o[0] + d[0] + s[0], pot, actor, route))
                if not candidates:
                    raise RuntimeError(
                        "NO_RESOURCE_ROUTE: no direct or one-counter onion recipe route"
                    )
                v = min(candidates, key=lambda x: x[:3])
                self.mode = "single_recipe_lane_with_generic_counter_relays"
                self.lanes = [v[3]]
                self.selection = {"pot": v[1], "pot_actor": v[2]}
        self.indices = [0] * len(self.lanes)

    def snapshot(self):
        return {
            "mode": self.mode,
            "indices": self.indices,
            "pending_actions": list(self.pending_actions),
            "pending_atoms": [a.data() for a in self.pending_atoms],
            "pending_lane_ids": self.pending_lane_ids,
        }

    def description(self):
        return {
            "mode": self.mode,
            "selection": self.selection,
            "lanes": [[a.data() for a in lane] for lane in self.lanes],
            "map_specific_action_table": False,
            "static_rule_priority": [
                "two_distinct_solo_feasible_pots",
                "one_pot_complementary_roles",
                "generic_relay",
            ],
        }

    @staticmethod
    def objname(player):
        return player.held_object.name if player.held_object else None

    def ready(self, atom, state):
        player = state.players[atom.agent]
        held = self.objname(player)
        obj = state.objects.get(atom.station)
        if atom.kind == "pickup":
            return held is None
        if atom.kind == "put_onion":
            return held == "onion" and (
                obj is None or (obj.is_idle and len(obj.ingredients) < 3)
            )
        if atom.kind == "start":
            return (
                held is None
                and obj is not None
                and obj.is_idle
                and len(obj.ingredients) == 3
            )
        if atom.kind == "scoop":
            return held == "dish" and obj is not None and obj.is_ready
        if atom.kind == "approach":
            return held == "dish" and obj is not None and not obj.is_idle
        if atom.kind == "deliver":
            return held == "soup"
        if atom.kind == "drop":
            return held == atom.item and obj is None
        if atom.kind == "take":
            return held is None and obj is not None and obj.name == atom.item
        raise RuntimeError("unknown atom")

    def next(self, state):
        if self.pending_actions:
            return self.pending_actions.popleft()
        available = []
        for lane_id, lane in enumerate(self.lanes):
            atom = lane[self.indices[lane_id]]
            if self.ready(atom, state):
                available.append((lane_id, atom))
        # An actor may be used in two relay/pipeline lanes only serially.
        selected = []
        used = set()
        for lane_id, atom in available:
            if atom.agent not in used:
                selected.append((lane_id, atom))
                used.add(atom.agent)
        if not selected:
            return (STAY, STAY)
        pose = (
            state.players[0].position,
            state.players[0].orientation,
            state.players[1].position,
            state.players[1].orientation,
        )
        atoms = [a for _, a in selected]
        path = self.motion.search(pose, atoms)
        if path is None and len(selected) > 1:
            # Simultaneous interaction poses may conflict. Fixed lane order
            # grants one task while the other actor can move out of the way.
            selected = selected[:1]
            atoms = [selected[0][1]]
            path = self.motion.search(pose, atoms)
        if path is None:
            raise RuntimeError("NO_JOINT_MOTION_ROUTE")
        self.pending_atoms = atoms
        self.pending_lane_ids = [i for i, _ in selected]
        final = [STAY, STAY]
        for atom in atoms:
            final[atom.agent] = STAY if atom.kind == "approach" else INTERACT
        self.pending_actions = deque(path + (tuple(final),))
        return self.pending_actions.popleft()

    def observe(self, before, after, actions):
        if not self.pending_atoms or self.pending_actions:
            return
        # The final queued step is exactly the selected INTERACT batch. Check
        # the native effect instead of assuming the high-level task succeeded.
        for atom in self.pending_atoms:
            if atom.kind != "approach" and actions[atom.agent] != INTERACT:
                raise RuntimeError("INTERNAL_PENDING_PHASE_MISMATCH")
            p = after.players[atom.agent]
            held = self.objname(p)
            obj = after.objects.get(atom.station)
            ok = {
                "pickup": lambda: held == atom.item,
                "put_onion": lambda: held is None and obj is not None,
                "start": lambda: obj is not None and not obj.is_idle,
                "scoop": lambda: held == "soup" and obj is None,
                "approach": lambda: (
                    held == "dish" and add(p.position, p.orientation) == atom.station
                ),
                "deliver": lambda: held is None,
                "drop": lambda: (
                    held is None and obj is not None and obj.name == atom.item
                ),
                "take": lambda: held == atom.item and obj is None,
            }[atom.kind]()
            if not ok:
                raise RuntimeError("NATIVE_ATOMIC_TASK_FAILED: " + compact(atom.data()))
        for i in self.pending_lane_ids:
            self.indices[i] = (self.indices[i] + 1) % len(self.lanes[i])
        self.pending_atoms, self.pending_lane_ids = [], []


def native_canonical(state):
    d = state.to_dict()
    d.pop("timestep", None)
    d["objects"] = sorted(d["objects"], key=lambda x: (x["position"], x["name"]))
    return d
