"""Strengthen base event flow with floor occupancy and local pot evolution."""

from copy import deepcopy

from .pot_states import pot_graph


def _add(row, column, value):
    row[column] = row.get(column, 0) + value
    if row[column] == 0:
        del row[column]


def strengthen_model(model, *, pot_states=False, spatial=False):
    if model.get("state_constraints"):
        raise ValueError("State constraints have already been added")
    if not pot_states and not spatial:
        return deepcopy(model)
    result = deepcopy(model)
    rows = [dict(row) for row in model["A_eq_sparse"]]
    labels = list(model["row_labels"])
    rhs = list(model["b_eq"])
    nvars = model["variable_count"]
    duration = model["T_variable_index"]

    def append(name, row):
        labels.append(name)
        rows.append(row)
        rhs.append(0)

    if spatial:
        floors = sorted({tuple(pos) for poses in model["poses"] for pos, _ in poses})
        for position in floors:
            row = {duration: -1}
            for column, arc in enumerate(model["arcs"]):
                node = model["nodes"][arc["from"]]
                pos = model["poses"][node["role"]][node["pose"]][0]
                if tuple(pos) == position:
                    _add(row, column, 1)
            row[nvars] = 1
            nvars += 1
            append("position_capacity_" + str(position), row)

    if pot_states:
        states, transitions = pot_graph(set(model["active_roles"]))
        for pot in range(len(model["pot_positions"])):
            flow = {state: {} for state in states}
            time_row = {duration: -1}
            events = {
                (role, event, size): {}
                for role in model["active_roles"]
                for event, size in [
                    ("load", 0),
                    ("start", 0),
                    ("scoop", 1),
                    ("scoop", 2),
                    ("scoop", 3),
                ]
            }
            for column, arc in enumerate(model["arcs"]):
                if arc["site"] == pot and arc["event"] in ("load", "start", "scoop"):
                    key = (
                        arc["role"],
                        arc["event"],
                        arc["n"] if arc["event"] == "scoop" else 0,
                    )
                    _add(events[key], column, 1)
            for source, destination, first_event, second_event in transitions:
                column = nvars
                nvars += 1
                _add(flow[source], column, 1)
                _add(flow[destination], column, -1)
                _add(time_row, column, 1)
                for role, event in enumerate((first_event, second_event)):
                    if event is not None:
                        _add(events[(role, event[0], event[1])], column, -1)
            for state, row in flow.items():
                append("pot_fsm_" + str(pot) + "_" + str(state), row)
            append("pot_fsm_duration_" + str(pot), time_row)
            for key, row in events.items():
                append("pot_fsm_event_" + str(pot) + "_" + str(key), row)

    result.update(
        row_labels=labels,
        b_eq=rhs,
        variable_count=nvars,
        A_eq_sparse=[sorted(row.items()) for row in rows],
        state_constraints={
            "pot_states": bool(pot_states),
            "spatial": bool(spatial),
            "base_row_count": len(model["row_labels"]),
            "base_variable_count": model["variable_count"],
        },
    )
    # JSON normalization keeps the in-memory and serialized records identical.
    import json

    from .io import canonical

    return json.loads(canonical(result))


def base_part(model):
    result = deepcopy(model)
    state = result.pop("state_constraints", None)
    if state:
        count = state["base_row_count"]
        result["row_labels"] = result["row_labels"][:count]
        result["A_eq_sparse"] = result["A_eq_sparse"][:count]
        result["b_eq"] = result["b_eq"][:count]
        result["variable_count"] = state["base_variable_count"]
    return result
