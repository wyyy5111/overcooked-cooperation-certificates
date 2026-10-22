"""Independently reconstruct base columns and check certificate arithmetic exactly."""

import ast
import hashlib
from collections import Counter
from fractions import Fraction as F

DIRS = ((0, -1), (0, 1), (1, 0), (-1, 0))
MODES = {"actor0_only": [0], "actor1_only": [1], "two_active": [0, 1]}


def _check_layout_binding(model):
    binding = model.get("layout_binding") or {}
    raw = binding.get("text")
    require(isinstance(raw, str), "Embedded layout text is required")
    require(
        hashlib.sha256(raw.encode()).hexdigest() == binding.get("sha256"),
        "Layout digest differs",
    )
    cfg = ast.literal_eval(raw)
    grid = [list(row.strip()) for row in cfg["grid"].splitlines()]
    starts = {}
    for y, row in enumerate(grid):
        for x, tile in enumerate(row):
            if tile in "12":
                require(int(tile) - 1 not in starts, "Duplicate indexed start")
                starts[int(tile) - 1] = [x, y]
                row[x] = " "
    require(set(starts) == {0, 1}, "Two indexed starts are required")
    require([starts[0], starts[1]] == model["start_positions"], "Indexed starts differ")
    require(
        ["".join(row) for row in grid] == model["terrain"], "Embedded terrain differs"
    )
    require(
        cfg.get("start_state") is None and cfg.get("num_items_for_soup", 3) == 3,
        "Unsupported initial state or pot capacity",
    )
    require(
        not any(
            k in cfg
            for k in (
                "onion_time",
                "tomato_time",
                "recipe_times",
                "onion_value",
                "tomato_value",
                "recipe_values",
            )
        ),
        "Layout timing overrides are unsupported",
    )
    require(cfg.get("delivery_reward", 20) == 20, "Delivery reward differs")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def arc_key(a):
    return (
        a["role"],
        a["from"],
        a["to"],
        a["event"],
        a["site"],
        a["n"],
        a["delta"],
        None if a["direction"] is None else tuple(a["direction"]),
        a["collision_stall"],
    )


def validate_model(m):
    require(
        m["generator_version"] == "native_role_material_flow_v1",
        "unsupported generator",
    )
    require(m["active_roles"] == MODES[m["mode"]], "active-role class mismatch")
    require(
        m["cook_time"] == 20 and m["pot_capacity"] == 3, "wrong native timing/capacity"
    )
    require(
        m["hand_types"] == ["E", "O", "D", "S1", "S2", "S3"],
        "wrong recipe representation",
    )
    require(
        m["native_constructor_overrides"]
        == {
            "cook_time": 20,
            "start_all_orders": [{"ingredients": ["onion"] * 3}],
            "start_bonus_orders": [],
        },
        "wrong objective contract",
    )
    terrain = m["terrain"]
    starts = list(map(tuple, m["start_positions"]))
    _check_layout_binding(m)
    require(
        bool(terrain) and len({len(r) for r in terrain}) == 1, "nonrectangular terrain"
    )
    floor = {
        (x, y) for y, row in enumerate(terrain) for x, ch in enumerate(row) if ch == " "
    }
    counters = [
        (x, y) for y, row in enumerate(terrain) for x, ch in enumerate(row) if ch == "X"
    ]
    pots = [
        (x, y) for y, row in enumerate(terrain) for x, ch in enumerate(row) if ch == "P"
    ]
    require(
        counters == list(map(tuple, m["counter_positions"])),
        "counter geometry mismatch",
    )
    require(pots == list(map(tuple, m["pot_positions"])), "pot geometry mismatch")
    poses = []
    expected_nodes = []
    for role in range(2):
        if role in m["active_roles"]:
            reached = {starts[role]}
            old = None
            while old != reached:
                old = set(reached)
                reached |= {
                    (x + dx, y + dy)
                    for x, y in old
                    for dx, dy in DIRS
                    if (x + dx, y + dy) in floor
                }
            pp = [(p, d) for p in sorted(reached) for d in DIRS]
        else:
            pp = [(starts[role], DIRS[0])]
        poses.append(pp)
        require(
            pp == [(tuple(p), tuple(d)) for p, d in m["poses"][role]],
            "pose component mismatch",
        )
        for pi, pose in enumerate(pp):
            expected_nodes += [
                {"role": role, "pose": pi, "hand": h}
                for h in range(6 if role in m["active_roles"] else 1)
            ]
    require(expected_nodes == m["nodes"], "local node coding mismatch")
    ids = {(n["role"], n["pose"], n["hand"]): i for i, n in enumerate(expected_nodes)}
    expected_arcs = []

    def put(
        role,
        pi,
        h,
        pj,
        hj,
        event=None,
        site=None,
        n=0,
        delta=0,
        direction=None,
        stall=False,
    ):
        expected_arcs.append(
            (
                role,
                ids[role, pi, h],
                ids[role, pj, hj],
                event,
                site,
                n,
                delta,
                direction,
                stall,
            )
        )

    for role, pp in enumerate(poses):
        pi_of = {p: i for i, p in enumerate(pp)}
        for pi, (p, d) in enumerate(pp):
            for h in range(6 if role in m["active_roles"] else 1):
                put(role, pi, h, pi, h)
                if role not in m["active_roles"]:
                    continue
                for a in DIRS:
                    q = p[0] + a[0], p[1] + a[1]
                    to = q if q in floor else p
                    put(role, pi, h, pi_of[to, a], h, direction=a)
                    if to != p:
                        put(role, pi, h, pi_of[p, a], h, direction=a, stall=True)
                q = p[0] + d[0], p[1] + d[1]
                ch = (
                    terrain[q[1]][q[0]]
                    if 0 <= q[1] < len(terrain) and 0 <= q[0] < len(terrain[0])
                    else None
                )
                if ch == "X":
                    k = counters.index(q)
                    if h == 0:
                        for v in range(1, 6):
                            put(role, pi, h, pi, v, "counter", k, v, -1)
                    else:
                        put(role, pi, h, pi, 0, "counter", k, h, 1)
                elif ch == "O" and h == 0:
                    put(role, pi, h, pi, 1, "source_onion")
                elif ch == "D" and h == 0:
                    put(role, pi, h, pi, 2, "source_dish")
                elif ch == "S" and h >= 3:
                    put(role, pi, h, pi, 0, "serve", n=h - 2)
                elif ch == "P":
                    k = pots.index(q)
                    if h == 0:
                        put(role, pi, h, pi, 0, "start", k)
                    elif h == 1:
                        put(role, pi, h, pi, 0, "load", k)
                    elif h == 2:
                        for v in range(1, 4):
                            put(role, pi, h, pi, v + 2, "scoop", k, v)
    require(
        Counter(expected_arcs) == Counter(map(arc_key, m["arcs"])),
        "native local arc coverage differs",
    )
    nn = len(expected_nodes)
    nc = len(counters)
    np = len(pots)
    labels = [f"local_flow_{i}" for i in range(nn)]
    labels += [f"counter_{k}_type_{h}" for k in range(nc) for h in range(1, 6)]
    labels += [f"pot_{k}_onions" for k in range(np)] + [
        f"pot_{k}_batches" for k in range(np)
    ]
    labels += ["role_0_time", "role_1_time", "accepted_full_serve"] + [
        f"pot_{k}_cook_work" for k in range(np)
    ]
    require(labels == m["row_labels"], "row labels/order differ")
    on = nn + 5 * nc
    batch = on + np
    rt = batch + np
    acc = rt + 2
    cook = acc + 1
    T = len(m["arcs"])
    nv = T + 1 + np
    require(
        m["T_variable_index"] == T and m["variable_count"] == nv,
        "variable count mismatch",
    )
    require(
        m["cook_slack_variable_indices"] == list(range(T + 1, nv)),
        "cook slack coding mismatch",
    )
    expected = [Counter() for _ in labels]
    for j, a in enumerate(m["arcs"]):
        expected[a["from"]][j] += 1
        expected[a["to"]][j] -= 1
        k, n = a["site"], a["n"]
        ev = a["event"]
        if ev == "counter":
            expected[nn + 5 * k + n - 1][j] += a["delta"]
        elif ev == "load":
            expected[on + k][j] += 1
        elif ev == "start":
            expected[batch + k][j] += 1
        elif ev == "scoop":
            expected[on + k][j] -= n
            expected[batch + k][j] -= 1
            expected[cook + k][j] += 20
        elif ev == "serve" and n == 3:
            expected[acc][j] += 1
        expected[rt + a["role"]][j] += 1
    for role in range(2):
        expected[rt + role][T] -= 1
    for k in range(np):
        expected[cook + k][T] -= 1
        expected[cook + k][T + 1 + k] += 1
    desired = [sorted([j, v] for j, v in row.items() if v) for row in expected]
    require(
        desired == m["A_eq_sparse"], "matrix does not match generated semantic columns"
    )
    rhs = [0] * len(labels)
    rhs[acc] = 1
    require(
        m["b_eq"] == rhs and m["objective_sparse"] == [[T, 1]],
        "normalization/objective mismatch",
    )
    require(m["all_variables_nonnegative"] is True, "nonnegativity missing")
    return {
        "rows": len(labels),
        "arc_variables": T,
        "variables": nv,
        "geometry_and_all_columns_checked": True,
        "embedded_layout_checked": True,
    }


def check_algebra(m, cert):
    nv = m["variable_count"]
    rows = m["A_eq_sparse"]
    b = m["b_eq"]
    y = list(map(F, cert["dual_eq"]))
    x = list(map(F, cert["primal"]))
    require(len(y) == len(rows) and len(x) == nv, "certificate dimensions differ")
    c = [F(0)] * nv
    for j, v in m["objective_sparse"]:
        c[j] = F(v)
    reduced = list(c)
    for yy, row in zip(y, rows):
        for j, v in row:
            reduced[j] -= yy * v
    dual_valid = all(v >= 0 for v in reduced)
    residual = [sum(x[j] * v for j, v in row) - rhs for row, rhs in zip(rows, b)]
    primal_valid = all(v >= 0 for v in x) and all(v == 0 for v in residual)
    dv = sum(v * rhs for v, rhs in zip(y, b))
    pv = sum(a * z for a, z in zip(c, x))
    exact = dual_valid and primal_valid and dv == pv
    return {
        "status": "MATCHING_EXACT_RELAXATION_OPTIMUM"
        if exact
        else "VALID_EXACT_DUAL_BOUND"
        if dual_valid
        else "NO_VALID_DUAL_BOUND",
        "dual_valid_exact": dual_valid,
        "primal_valid_exact": primal_valid,
        "matching_exact_LP_optimum": exact,
        "dual_mean_ticks_lower": str(dv) if dual_valid else None,
        "native_rate_upper": str(F(1) / dv) if dual_valid and dv > 0 else None,
        "primal_relaxation_mean_ticks": str(pv) if primal_valid else None,
        "minimum_dual_column_slack": str(min(reduced)),
        "claim_scope": "native capacity upper only; LP primal is not a native policy",
    }
