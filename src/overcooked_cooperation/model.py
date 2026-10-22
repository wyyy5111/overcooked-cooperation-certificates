"""Construct policy-independent event-flow bounds from native kitchen layouts."""

import ast
from collections import deque

VERSION = "native_role_material_flow_v1"
DIRECTIONS = ((0, -1), (0, 1), (1, 0), (-1, 0))
HANDS = ("E", "O", "D", "S1", "S2", "S3")
MODES = {"actor0_only": (0,), "actor1_only": (1,), "two_active": (0, 1)}


def parse_layout_text(text):
    """Match native from_layout_name's per-row strip; no eval/import/game call."""
    config = ast.literal_eval(text)
    if not isinstance(config, dict) or not isinstance(config.get("grid"), str):
        raise ValueError("layout must be a literal dictionary with a grid string")
    rows = [row.strip() for row in config["grid"].split("\n")]
    if not rows or len({len(row) for row in rows}) != 1 or not rows[0]:
        raise ValueError("native grid must be rectangular after row.strip()")
    starts = {}
    terrain = []
    for y, row in enumerate(rows):
        cells = []
        for x, ch in enumerate(row):
            if ch in "12":
                if int(ch) - 1 in starts:
                    raise ValueError("duplicate actor marker")
                starts[int(ch) - 1] = (x, y)
                ch = " "
            if ch not in " XOPSDT":
                raise ValueError("unsupported terrain character " + repr(ch))
            cells.append(ch)
        terrain.append("".join(cells))
    if set(starts) != {0, 1}:
        raise ValueError("v1 requires exactly native actor markers 1 and 2")
    if config.get("start_state") is not None:
        raise ValueError(
            "v1 requires the empty standard start, not a layout start_state"
        )
    if config.get("num_items_for_soup", 3) != 3:
        raise ValueError("v1 requires native pot capacity 3")
    if any(
        k in config
        for k in (
            "onion_time",
            "tomato_time",
            "recipe_times",
            "onion_value",
            "tomato_value",
            "recipe_values",
        )
    ):
        raise ValueError(
            "layout has nonuniform recipe timing/value configuration unsupported by v1"
        )
    if config.get("delivery_reward", 20) != 20:
        raise ValueError("layout delivery_reward conflicts with sparse20 contract")
    return config, terrain, [starts[0], starts[1]]


def component(terrain, start):
    floor = {
        (x, y) for y, row in enumerate(terrain) for x, ch in enumerate(row) if ch == " "
    }
    found = {tuple(start)}
    queue = deque(found)
    while queue:
        x, y = queue.popleft()
        for dx, dy in DIRECTIONS:
            p = x + dx, y + dy
            if p in floor and p not in found:
                found.add(p)
                queue.append(p)
    return sorted(found)


def build_model(layout_text, mode, *, layout_binding=None, source_binding=None):
    """Pure generator. The target is counts of accepted three-onion deliveries.

    Frozen executions must instantiate only that accepted order, cook_time=20,
    sparse reward 20, empty standard start, and stationary inactive actors.
    All counters/pots and actors' identities are derived from the layout.
    """
    if mode not in MODES:
        raise ValueError("unknown active-role class")
    config, terrain, starts = parse_layout_text(layout_text)
    active = MODES[mode]
    counters = [
        (x, y) for y, row in enumerate(terrain) for x, ch in enumerate(row) if ch == "X"
    ]
    pots = [
        (x, y) for y, row in enumerate(terrain) for x, ch in enumerate(row) if ch == "P"
    ]
    counter_id = {p: i for i, p in enumerate(counters)}
    pot_id = {p: i for i, p in enumerate(pots)}
    poses = []
    nodes = []
    ids = {}
    for role in range(2):
        positions = (
            component(terrain, starts[role]) if role in active else [starts[role]]
        )
        pp = (
            [(p, d) for p in positions for d in DIRECTIONS]
            if role in active
            else [(starts[role], DIRECTIONS[0])]
        )
        poses.append(pp)
        for pi, (pos, d) in enumerate(pp):
            for h in range(6 if role in active else 1):
                ids[role, pi, h] = len(nodes)
                nodes.append({"role": role, "pose": pi, "hand": h})
    arcs = []

    def add(
        role, pi, h, np, nh, event=None, site=None, n=0, delta=0, move=None, stall=False
    ):
        arcs.append(
            {
                "role": role,
                "from": ids[role, pi, h],
                "to": ids[role, np, nh],
                "event": event,
                "site": site,
                "n": n,
                "delta": delta,
                "direction": list(move) if move else None,
                "collision_stall": stall,
            }
        )

    height, width = len(terrain), len(terrain[0])

    def tile(p):
        x, y = p
        return terrain[y][x] if 0 <= x < width and 0 <= y < height else None

    for role, pp in enumerate(poses):
        pids = {q: i for i, q in enumerate(pp)}
        floor = {p for p, d in pp}
        for pi, (pos, d) in enumerate(pp):
            for h in range(6 if role in active else 1):
                add(role, pi, h, pi, h)
                if role not in active:
                    continue
                for move in DIRECTIONS:
                    dest = pos[0] + move[0], pos[1] + move[1]
                    np = dest if dest in floor else pos
                    add(role, pi, h, pids[np, move], h, move=move)
                    # Real joint collision can keep the old position even if
                    # dest is floor, but orientation still becomes move.
                    if np != pos:
                        add(role, pi, h, pids[pos, move], h, move=move, stall=True)
                front = pos[0] + d[0], pos[1] + d[1]
                ty = tile(front)
                if ty == "X":
                    k = counter_id[front]
                    if h == 0:
                        for item in range(1, 6):
                            add(role, pi, h, pi, item, "counter", k, item, -1)
                    else:
                        add(role, pi, h, pi, 0, "counter", k, h, 1)
                elif ty == "O" and h == 0:
                    add(role, pi, h, pi, 1, "source_onion")
                elif ty == "D" and h == 0:
                    add(role, pi, h, pi, 2, "source_dish")
                elif ty == "S" and h >= 3:
                    add(role, pi, h, pi, 0, "serve", n=h - 2)
                elif ty == "P":
                    k = pot_id[front]
                    if h == 0:
                        add(role, pi, h, pi, 0, "start", k)
                    elif h == 1:
                        add(role, pi, h, pi, 0, "load", k)
                    elif h == 2:
                        for n in range(1, 4):
                            add(role, pi, h, pi, n + 2, "scoop", k, n)
                # Tomato interactions are deleted by accepted-lineage
                # projection, not interpreted as native failed actions.
    labels = [f"local_flow_{i}" for i in range(len(nodes))]
    offset_counter = len(labels)
    labels += [
        f"counter_{k}_type_{h}" for k in range(len(counters)) for h in range(1, 6)
    ]
    offset_onion = len(labels)
    labels += [f"pot_{k}_onions" for k in range(len(pots))]
    offset_batch = len(labels)
    labels += [f"pot_{k}_batches" for k in range(len(pots))]
    offset_time = len(labels)
    labels += ["role_0_time", "role_1_time"]
    accepted_row = len(labels)
    labels += ["accepted_full_serve"]
    offset_cook = len(labels)
    labels += [f"pot_{k}_cook_work" for k in range(len(pots))]
    rows = [dict() for _ in labels]

    def coef(i, j, v):
        rows[i][j] = rows[i].get(j, 0) + v
        if rows[i][j] == 0:
            del rows[i][j]

    for j, a in enumerate(arcs):
        coef(a["from"], j, 1)
        coef(a["to"], j, -1)
        event, k, n = a["event"], a["site"], a["n"]
        if event == "counter":
            coef(offset_counter + 5 * k + n - 1, j, a["delta"])
        elif event == "load":
            coef(offset_onion + k, j, 1)
        elif event == "start":
            coef(offset_batch + k, j, 1)
        elif event == "scoop":
            coef(offset_onion + k, j, -n)
            coef(offset_batch + k, j, -1)
            coef(offset_cook + k, j, 20)
        elif event == "serve" and n == 3:
            coef(accepted_row, j, 1)
        coef(offset_time + a["role"], j, 1)
    T = len(arcs)
    for role in range(2):
        coef(offset_time + role, T, -1)
    for k in range(len(pots)):
        coef(offset_cook + k, T, -1)
        coef(offset_cook + k, T + 1 + k, 1)
    b = [0] * len(labels)
    b[accepted_row] = 1
    return {
        "schema_version": 1,
        "generator_version": VERSION,
        "mode": mode,
        "active_roles": list(active),
        "inactive_action": "STAY at standard start for all ticks",
        "objective_scope": "accepted three-onion soup count, not arbitrary sparse/shaped reward",
        "native_constructor_overrides": {
            "cook_time": 20,
            "start_all_orders": [{"ingredients": ["onion"] * 3}],
            "start_bonus_orders": [],
        },
        "empty_standard_start_required": True,
        "pot_capacity": 3,
        "cook_time": 20,
        "hand_types": list(HANDS),
        "layout_binding": layout_binding,
        "source_binding": source_binding,
        "terrain": terrain,
        "start_positions": [list(p) for p in starts],
        "poses": [[[list(p), list(d)] for p, d in pp] for pp in poses],
        "counter_positions": [list(p) for p in counters],
        "pot_positions": [list(p) for p in pots],
        "nodes": nodes,
        "arcs": arcs,
        "T_variable_index": T,
        "cook_slack_variable_indices": list(range(T + 1, T + 1 + len(pots))),
        "variable_count": T + 1 + len(pots),
        "row_labels": labels,
        "A_eq_sparse": [[[j, v] for j, v in sorted(row.items())] for row in rows],
        "b_eq": b,
        "objective_sparse": [[T, 1]],
        "all_variables_nonnegative": True,
        "sound_relaxations": [
            "optional collision stalls",
            "counter occupancy and sequencing removed",
            "pot sequencing removed; uniform cook-work retained",
            "zero-target-value lineage deletion",
        ],
    }
