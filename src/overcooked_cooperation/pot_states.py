"""Local pot-state flow under native actor order and clock updates."""


def pot_step_options(state, enabled):
    n, t = state
    out = [(state, None)]
    if enabled:
        if t == -1:
            if n < 3:
                out.append(((n + 1, -1), ("load", 0)))
            if n > 0:
                out.append(((n, 0), ("start", 0)))
        elif t == 20:
            out.append(((0, -1), ("scoop", n)))
    return out


def pot_graph(active):
    states = [(n, -1) for n in range(4)] + [
        (n, t) for n in (1, 2, 3) for t in range(1, 21)
    ]
    trans = []
    for s in states:
        for a, e0 in pot_step_options(s, 0 in active):
            for b, e1 in pot_step_options(a, 1 in active):
                n, t = b
                dest = (n, min(20, t + 1)) if 0 <= t < 20 else b
                trans.append((s, dest, e0, e1))
    return states, trans
