"""Endpoint potentials and empty-start finite bounds from exact base duals."""

from fractions import Fraction as F

HORIZONS = (200, 400)


def check_dual(m, certificate):
    """Check every saved column with Fraction; no solver and no rationalization."""
    y = list(map(F, certificate["dual_eq"]))
    assert len(y) == len(m["row_labels"]) == len(m["A_eq_sparse"])
    reduced = [F(0)] * m["variable_count"]
    for j, v in m["objective_sparse"]:
        reduced[j] = F(v)
    for yi, row in zip(y, m["A_eq_sparse"]):
        for j, v in row:
            reduced[j] -= yi * v
    assert min(reduced) >= 0, "saved dual is not an exact feasible upper certificate"
    L = sum(v * b for v, b in zip(y, m["b_eq"]))
    assert L > 0
    values = dict(zip(m["row_labels"], y))
    assert L == values["accepted_full_serve"]
    return values, L, min(reduced)


def derive_potential(m, certificate):
    y, L, min_slack = check_dual(m, certificate)
    actors = []
    counters = []
    pots = []
    M = F(0)
    initial = F(0)
    node_values = {i: y[f"local_flow_{i}"] for i in range(len(m["nodes"]))}
    for role in range(2):
        indices = [i for i, n in enumerate(m["nodes"]) if n["role"] == role]
        assert indices
        initial_pose = m["poses"][role].index([m["start_positions"][role], [0, -1]])
        initial_node = next(
            i
            for i in indices
            if m["nodes"][i]["pose"] == initial_pose and m["nodes"][i]["hand"] == 0
        )
        best = max(node_values[i] for i in indices)
        at_start = node_values[initial_node]
        M += best
        initial += at_start
        actors.append(
            {
                "role": role,
                "active": role in m["active_roles"],
                "states_in_domain": len(indices),
                "initial_node": initial_node,
                "initial_value": str(at_start),
                "max_value": str(best),
                "argmax_node": next(i for i in indices if node_values[i] == best),
            }
        )
    for k, pos in enumerate(m["counter_positions"]):
        options = [("empty", F(0))] + [
            (m["hand_types"][h], -y[f"counter_{k}_type_{h}"]) for h in range(1, 6)
        ]
        best = max(value for name, value in options)
        M += best
        counters.append(
            {
                "counter": k,
                "position": pos,
                "max_value": str(best),
                "argmax_type": next(name for name, value in options if value == best),
                "domain_values": {name: str(value) for name, value in options},
            }
        )
    for k, pos in enumerate(m["pot_positions"]):
        yo = y[f"pot_{k}_onions"]
        yb = y[f"pot_{k}_batches"]
        options = [("empty", F(0))]
        for n in range(1, 4):
            options += [(f"idle{n}", -n * yo), (f"begun{n}", -n * yo - yb)]
        best = max(value for name, value in options)
        M += best
        pots.append(
            {
                "pot": k,
                "position": pos,
                "max_value": str(best),
                "argmax_state": next(name for name, value in options if value == best),
                "domain_values": {name: str(value) for name, value in options},
                "begun_includes_ready": True,
            }
        )
    B = M - initial
    assert B >= 0, "initial state must be contained in the Cartesian product domain"
    return {
        "L": str(L),
        "Phi_initial": str(initial),
        "max_Phi_separable": str(M),
        "B": str(B),
        "horizon_upper_soups": {str(H): int((F(H) + B) // L) for H in HORIZONS},
        "minimum_dual_column_slack": str(min_slack),
        "actors": actors,
        "counters": counters,
        "pots": pots,
        "coupled_feasibility_relaxed": True,
        "domain": "actor nodes x per-counter empty/O/D/S1/S2/S3 x per-pot empty/idle1..3/begun1..3",
    }


def hand_code(o):
    if o is None:
        return 0
    if o["name"] == "onion":
        return 1
    if o["name"] == "dish":
        return 2
    assert o["name"] == "soup" and o["is_ready"]
    assert all(x["name"] == "onion" for x in o["_ingredients"])
    n = len(o["_ingredients"])
    assert 1 <= n <= 3
    return n + 2


def phi(s, m, certificate):
    """Actual native-state potential; current three official maps have no T."""
    y = dict(zip(m["row_labels"], map(F, certificate["dual_eq"])))
    answer = F(0)
    for role, player in enumerate(s["players"]):
        pi = m["poses"][role].index([player["position"], player["orientation"]])
        h = hand_code(player["held_object"])
        i = next(
            i
            for i, n in enumerate(m["nodes"])
            if n == {"role": role, "pose": pi, "hand": h}
        )
        answer += y[f"local_flow_{i}"]
    objects = {tuple(o["position"]): o for o in s["objects"]}
    for k, pos in enumerate(m["counter_positions"]):
        h = hand_code(objects.get(tuple(pos)))
        if h:
            answer -= y[f"counter_{k}_type_{h}"]
    for k, pos in enumerate(m["pot_positions"]):
        o = objects.get(tuple(pos))
        if o:
            assert o["name"] == "soup" and all(
                a["name"] == "onion" for a in o["_ingredients"]
            )
            n = len(o["_ingredients"])
            assert 1 <= n <= 3
            begun = int(not o["is_idle"])
            answer -= n * y[f"pot_{k}_onions"] + begun * y[f"pot_{k}_batches"]
    return answer
