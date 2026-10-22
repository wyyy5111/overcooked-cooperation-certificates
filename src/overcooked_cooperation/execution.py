"""Reachable returning words and qualified independent-component composition."""

import json
from fractions import Fraction
from hashlib import sha256
from math import lcm

from .api import layout_text
from .components import LocalLane, local_projection
from .controller import INTERACT, Motion, UniversalPolicy, native_canonical
from .io import canonical
from .model import parse_layout_text
from .native import load_engine


def _normal(value):
    return json.loads(canonical(value))


def _action(value):
    return tuple(value) if isinstance(value, (list, tuple)) else value


def _context(text):
    config, terrain, starts = parse_layout_text(text)
    motion = Motion(terrain)
    policy = UniversalPolicy(motion, starts)
    components = [set(motion.dist[start]) for start in starts]
    pots = [tuple(p) for p in policy.selection.get("pots", [])]
    reasons = []
    if components[0] & components[1]:
        reasons.append("floor components overlap")
    if (
        policy.mode != "two_reserved_pots_parallel_recipe_loops"
        or len(policy.lanes) != 2
    ):
        reasons.append("two independent production lanes are required")
    if len(pots) != 2 or len(set(pots)) != 2:
        reasons.append("distinct pot reservations are required")
    if len(policy.lanes) == len(pots) == 2:
        for actor, lane in enumerate(policy.lanes):
            for atom in lane:
                tile = terrain[atom.station[1]][atom.station[0]]
                if atom.agent != actor or atom.kind not in (
                    "pickup",
                    "put_onion",
                    "start",
                    "approach",
                    "scoop",
                    "deliver",
                ):
                    reasons.append("lane contains a nonlocal or counter operation")
                if atom.kind == "pickup" and (
                    tile not in "OD"
                    or atom.item != ("onion" if tile == "O" else "dish")
                ):
                    reasons.append("pickup must use an immutable supply")
                if atom.kind == "deliver" and tile != "S":
                    reasons.append("delivery must use an immutable sink")
                if atom.kind in ("put_onion", "start", "approach", "scoop") and (
                    tile != "P" or atom.station != pots[actor]
                ):
                    reasons.append("pot operation must use the lane's reserved pot")
                if not motion.reachable(starts[actor], atom.station):
                    reasons.append("station is outside the lane's component")
    return config, terrain, starts, policy, components, pots, sorted(set(reasons))


def qualify_layout(name_or_path):
    text, name = layout_text(name_or_path)
    _, _, _, policy, components, pots, reasons = _context(text)
    return {
        "layout": name,
        "qualified": not reasons,
        "reasons": reasons,
        "component_sizes": list(map(len, components)),
        "reserved_pots": pots,
        "policy": policy.description(),
    }


def _environment(config, name):
    engine = load_engine()
    config = dict(config)
    grid = [row.strip() for row in config.pop("grid").splitlines()]
    config["layout_name"] = name
    mdp = engine.from_grid(
        grid,
        base_layout_params=config,
        params_to_overwrite={
            "cook_time": 20,
            "start_all_orders": [{"ingredients": ["onion"] * 3}],
            "start_bonus_orders": [],
        },
    )
    state = mdp.get_standard_start_state()
    orders = list(state.all_orders)
    if (
        state.objects
        or any(p.held_object is not None for p in state.players)
        or len(state.players) != 2
        or len(orders) != 1
        or sorted(orders[0].ingredients) != ["onion"] * 3
        or orders[0].time != 20
        or orders[0].value != 20
        or state.bonus_orders
    ):
        raise ValueError("Native recipe, clock or empty-start contract differs")
    return mdp, state


def _gains(state, info):
    rewards = info["sparse_reward_by_agent"]
    if len(rewards) != 2 or any(value not in (0, 20) for value in rewards):
        raise ValueError("Unexpected accepted-reward unit")
    gains = [int(value // 20) for value in rewards]
    for actor, gain in enumerate(gains):
        if gain:
            soup = state.players[actor].held_object
            if (
                soup is None
                or soup.name != "soup"
                or not soup.is_ready
                or sorted(soup.ingredients) != ["onion"] * 3
            ):
                raise ValueError("Delivery is not a ready accepted three-onion soup")
    return gains


def _component_state(state, components, pots):
    if any(
        player.position not in components[actor]
        for actor, player in enumerate(state.players)
    ):
        raise ValueError("Actor left its qualified component")
    if any(tuple(position) not in pots for position in state.objects):
        raise ValueError("An unowned mutable object was created")


def _component_actions(state, actions, terrain, pots):
    for actor, action in enumerate(actions):
        if action != INTERACT:
            continue
        player = state.players[actor]
        target = (
            player.position[0] + player.orientation[0],
            player.position[1] + player.orientation[1],
        )
        x, y = target
        tile = (
            terrain[y][x]
            if 0 <= y < len(terrain) and 0 <= x < len(terrain[0])
            else None
        )
        if tile == "X" or (tile == "P" and target != pots[actor]):
            raise ValueError("Action touches a counter or another lane's pot")


def run_schedule(name_or_path, *, scheduling="auto", max_steps=6000, observe_until=0):
    if scheduling not in ("auto", "synchronous", "asynchronous"):
        raise ValueError("Unknown scheduling mode")
    if (
        not isinstance(max_steps, int)
        or max_steps < 1
        or not 0 <= observe_until <= max_steps
    ):
        raise ValueError(
            "Choose positive max_steps and an observation horizon within it"
        )
    text, name = layout_text(name_or_path)
    config, terrain, starts, policy, components, pots, reasons = _context(text)
    asynchronous = scheduling == "asynchronous" or (
        scheduling == "auto" and not reasons
    )
    if asynchronous and reasons:
        raise ValueError("Asynchronous qualification failed: " + "; ".join(reasons))
    mdp, state = _environment(config, name)
    lanes = (
        [LocalLane(i, policy.lanes[i], components[i], policy) for i in (0, 1)]
        if asynchronous
        else []
    )
    cycles = [None, None] if asynchronous else [None]
    seen = [{}, {}] if asynchronous else [{}]
    actions_history = []
    counts = [[0], [0]]
    total_counts = [0]
    closed = False

    for tick in range(max_steps + 1):
        if asynchronous:
            _component_state(state, components, pots)
        snapshots = (
            [local_projection(state.to_dict(), actor, pots[actor]) for actor in (0, 1)]
            if asynchronous
            else [native_canonical(state)]
        )
        for index, snapshot in enumerate(snapshots):
            count_history = counts[index] if asynchronous else total_counts
            phase = lanes[index].snapshot() if asynchronous else policy.snapshot()
            key = canonical([snapshot, phase])
            cycle = cycles[index]
            if cycle is None and key in seen[index]:
                bootstrap, prior = seen[index][key]
                gain = count_history[tick] - prior
                if gain <= 0:
                    raise RuntimeError("A nonproductive state repeated")
                word = (
                    [pair[index] for pair in actions_history[bootstrap:tick]]
                    if asynchronous
                    else actions_history[bootstrap:tick]
                )
                cycles[index] = {
                    "actor": index if asynchronous else None,
                    "bootstrap": bootstrap,
                    "period": tick - bootstrap,
                    "soups": gain,
                    "bootstrap_soups": prior,
                    "root": snapshot,
                    "word": _normal(word),
                    "prefix_counts": [
                        x - prior for x in count_history[bootstrap : tick + 1]
                    ],
                    "origin": tick,
                    "confirmations": 0,
                    "last_count": count_history[tick],
                }
            elif cycle is None:
                seen[index][key] = (tick, count_history[tick])
            elif (
                tick > cycle["origin"]
                and (tick - cycle["origin"]) % cycle["period"] == 0
            ):
                word = (
                    [
                        pair[index]
                        for pair in actions_history[tick - cycle["period"] : tick]
                    ]
                    if asynchronous
                    else actions_history[tick - cycle["period"] : tick]
                )
                if (
                    _normal(snapshot) != _normal(cycle["root"])
                    or count_history[tick] - cycle["last_count"] != cycle["soups"]
                    or _normal(word) != cycle["word"]
                ):
                    raise ValueError("The returning word failed native confirmation")
                cycle["confirmations"] += 1
                cycle["last_count"] = count_history[tick]
        closed = all(c is not None and c["confirmations"] >= 2 for c in cycles)
        if closed and tick >= observe_until:
            break
        if tick == max_steps:
            raise RuntimeError("No confirmed productive return within max_steps")
        if asynchronous:
            actions = tuple(
                lanes[i].next(state)
                if cycles[i] is None
                else _action(
                    cycles[i]["word"][
                        (tick - cycles[i]["origin"]) % cycles[i]["period"]
                    ]
                )
                for i in (0, 1)
            )
            _component_actions(state, actions, terrain, pots)
        else:
            cycle = cycles[0]
            actions = (
                policy.next(state)
                if cycle is None
                else tuple(
                    _action(a)
                    for a in cycle["word"][(tick - cycle["origin"]) % cycle["period"]]
                )
            )
        after, info = mdp.get_state_transition(state, actions)
        gains = _gains(state, info)
        for actor in (0, 1):
            counts[actor].append(counts[actor][-1] + gains[actor])
            if asynchronous and cycles[actor] is None:
                lanes[actor].observe(state, after, actions[actor])
        if not asynchronous and cycles[0] is None:
            policy.observe(state, after, actions)
        total_counts.append(total_counts[-1] + sum(gains))
        actions_history.append(_normal(actions))
        state = after
    if not closed:
        raise RuntimeError("A returning word was not established")
    rate = sum((Fraction(c["soups"], c["period"]) for c in cycles), Fraction(0))
    result = {
        "schema_version": 1,
        "layout": name,
        "layout_text": text,
        "layout_sha256": sha256(text.encode()).hexdigest(),
        "scheduling": "asynchronous" if asynchronous else "synchronous",
        "observed_steps": tick,
        "counts_by_actor": counts,
        "actions": actions_history,
        "cycles": cycles,
        "rate": str(rate),
        "joint_period": lcm(*(c["period"] for c in cycles)),
        "joint_period_type": "constructed_from_local_words"
        if asynchronous
        else "observed_joint_return",
        "qualification": {"qualified": not reasons, "reasons": reasons},
        "policy": _normal(policy.description()),
    }
    return result


def finite_output(schedule, horizon):
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 0:
        raise ValueError("Horizon must be a nonnegative integer")
    if horizon <= schedule["observed_steps"]:
        count = sum(counts[horizon] for counts in schedule["counts_by_actor"])
        return {"horizon": horizon, "soups": count, "type": "observed"}
    total = 0
    for cycle in schedule["cycles"]:
        quotient, remainder = divmod(horizon - cycle["bootstrap"], cycle["period"])
        total += (
            cycle["bootstrap_soups"]
            + quotient * cycle["soups"]
            + cycle["prefix_counts"][remainder]
        )
    return {"horizon": horizon, "soups": total, "type": "extrapolated"}


def verify_schedule(schedule):
    """Replay the supplied actions and check three copies of each returning word."""
    text = schedule["layout_text"]
    if sha256(text.encode()).hexdigest() != schedule["layout_sha256"]:
        raise ValueError("Schedule layout digest differs")
    config, terrain, starts, policy, components, pots, reasons = _context(text)
    asynchronous = schedule["scheduling"] == "asynchronous"
    if schedule["scheduling"] not in ("synchronous", "asynchronous") or (
        asynchronous and reasons
    ):
        raise ValueError("Schedule mode or component qualification differs")
    cycles = schedule["cycles"]
    if len(cycles) != (2 if asynchronous else 1):
        raise ValueError("Incorrect number of returning words")
    actions = schedule["actions"]
    if len(actions) != schedule["observed_steps"]:
        raise ValueError("Observed action count differs")
    required_ticks = set()
    for index, cycle in enumerate(cycles):
        if (
            cycle["bootstrap"] < 0
            or cycle["period"] < 1
            or cycle["soups"] < 1
            or cycle["actor"] != (index if asynchronous else None)
            or cycle["bootstrap"] + 3 * cycle["period"] > len(actions)
        ):
            raise ValueError("Incomplete returning-word evidence")
        required_ticks.update(
            cycle["bootstrap"] + k * cycle["period"] for k in range(4)
        )
    mdp, state = _environment(config, schedule["layout"])
    roots = {}
    counts = [[0], [0]]
    for tick in range(len(actions) + 1):
        if asynchronous:
            _component_state(state, components, pots)
        if tick in required_ticks:
            roots[tick] = (
                [local_projection(state.to_dict(), i, pots[i]) for i in (0, 1)]
                if asynchronous
                else [native_canonical(state)]
            )
        if tick == len(actions):
            break
        joint = tuple(_action(a) for a in actions[tick])
        if len(joint) != 2:
            raise ValueError("A joint action must contain two indexed actions")
        if asynchronous:
            _component_actions(state, joint, terrain, pots)
        after, info = mdp.get_state_transition(state, joint)
        for actor, gain in enumerate(_gains(state, info)):
            counts[actor].append(counts[actor][-1] + gain)
        state = after
    if counts != schedule["counts_by_actor"]:
        raise ValueError("Native observed counts differ")
    for index, cycle in enumerate(cycles):
        bootstrap, period, soups = cycle["bootstrap"], cycle["period"], cycle["soups"]
        history = counts[index] if asynchronous else [sum(x) for x in zip(*counts)]
        if canonical(cycle["root"]) != canonical(roots[bootstrap][index]):
            raise ValueError("Recorded returning root differs from native replay")
        if history[bootstrap] != cycle["bootstrap_soups"]:
            raise ValueError("Bootstrap count differs")
        prefix = [
            x - history[bootstrap] for x in history[bootstrap : bootstrap + period + 1]
        ]
        if prefix != cycle["prefix_counts"]:
            raise ValueError("Within-word delivery phase differs")
        for k in range(1, 4):
            end = bootstrap + k * period
            word = (
                [pair[index] for pair in actions[end - period : end]]
                if asynchronous
                else actions[end - period : end]
            )
            if (
                canonical(roots[end][index]) != canonical(roots[bootstrap][index])
                or history[end] - history[end - period] != soups
                or word != cycle["word"]
            ):
                raise ValueError("Returning state, word or output differs")
    rate = sum((Fraction(c["soups"], c["period"]) for c in cycles), Fraction(0))
    if Fraction(schedule["rate"]) != rate or schedule["joint_period"] != lcm(
        *(c["period"] for c in cycles)
    ):
        raise ValueError("Derived rate or joint period differs")
    return {
        "native_replay_verified": True,
        "observed_steps": len(actions),
        "returning_words_verified": len(cycles),
        "rate": str(rate),
        "finite_count_types": "observed within replay; extrapolated beyond returning words",
    }
