# Overcooked Cooperation Certificates

Policy-independent output bounds and reachable production schedules for Overcooked.

The core method turns movement, manipulation, ingredients and cooking time into
event-flow constraints. Exact rational checks validate the resulting output
bounds. Returning native action words establish repeatable team output, and
qualified production lanes advance on independent local clocks.

## Install

Use Python 3.10 or later.

```bash
git clone https://github.com/wyyy5111/overcooked-cooperation-certificates.git
cd overcooked-cooperation-certificates
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[solver,native]'
```

On Windows, activate the environment with `.venv\Scripts\activate`.
The solver extra installs SciPy and NumPy; native execution needs NumPy.
Model construction and saved-certificate checks use the Python standard library.

## Quick start

```bash
occ demo --layout ring --out results/ring
```

The command builds and solves the two solo models, executes a returning team
schedule, replays its native actions, and writes four JSON records:

```text
results/ring/
  solo0.json
  solo1.json
  schedule.json
  certificate.json
```

Ring gives a solo rate upper bound of `1/42`, a reachable team rate of `3/97`,
and a relative resource gain guarantee of `29/97`. At 200 and 400 joint steps,
the observed team counts are 5 and 11; the solo upper bounds are 4 and 9.

## Core commands

Construct and solve one active-role model:

```bash
occ build --layout ring --mode actor0_only --out results/model.json
occ solve --layout ring --mode actor0_only --out results/solo0.json
occ check results/solo0.json
occ finite results/solo0.json --horizon 200
```

The active-role classes are `actor0_only`, `actor1_only` and `two_active`.
The bundled layouts are `ring`, `asymmetric` and `circuit`. A `.layout` path can
also be supplied. Each layout uses indexed empty starts, a three-onion accepted
recipe, cook time 20, reward 20 and no bonus orders.

Add the method's state constraints for sustained output:

```bash
occ solve --layout asymmetric --mode two_active --pot-states --out results/pot-bound.json
occ check results/pot-bound.json
```

`--pot-states` adds local pot-state flow. `--spatial` adds floor occupancy rows.
Finite endpoint bounds use the base event-flow certificates.

Execute and verify a schedule:

```bash
occ qualify --layout asymmetric
occ schedule --layout asymmetric --scheduling auto --out results/asymmetric.json
occ verify-schedule results/asymmetric.json
occ finite results/asymmetric.json --horizon 400
```

Automatic scheduling uses independent clocks when floor components are disjoint,
pots have distinct owners, transfers use immutable supplies and serving sites,
and no mutable counter is used. Connected layouts retain synchronous execution.
`--scheduling synchronous` and `--scheduling asynchronous` select an explicit mode.

On Asymmetric, local periods of 45 and 50 yield rate `19/450`. The native prefix
contains 152 steps; their constructed product period is 450. Counts at 200 and
400 steps are 8 and 16 by returning-word extrapolation. Output records distinguish
observed counts, extrapolated counts and a constructed product period.

Compare two saved solo certificates with a native-verified team schedule:

```bash
occ certify --solo0 results/ring/solo0.json --solo1 results/ring/solo1.json \
  --schedule results/ring/schedule.json --out results/resource-certificate.json
```

All commands also work through `python -m overcooked_cooperation`. Output paths
are created as new records; use a new path for a fresh run.

## Python API

```python
from overcooked_cooperation import make_model, solve_model, certify_cooperation
from overcooked_cooperation.execution import run_schedule

solo0 = solve_model(make_model("ring", "actor0_only"))
solo1 = solve_model(make_model("ring", "actor1_only"))
schedule = run_schedule("ring")
certificate = certify_cooperation(solo0, solo1, schedule)
```

See [examples/core_example.py](examples/core_example.py).

## Source layout

| Module | Role |
| --- | --- |
| `model.py` | Native event graph, material balances and workload model |
| `exact.py`, `api.py` | Column reconstruction and exact primal/dual checks |
| `finite.py` | Endpoint potentials and finite output bounds |
| `controller.py` | Deterministic production lanes and joint motion |
| `components.py` | Independent lane motion and local task clocks |
| `pot_states.py`, `state_constraints.py` | Pot evolution and spatial constraints |
| `execution.py` | Native returning words, replay verification and composition |
| `cli.py` | Command-line entry points |

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover exact bounds, finite output, rejected certificate edits, native word
closure, component eligibility, independent-clock composition and a complete Ring
certificate. The optional extras enable solver and native-execution tests.

## Native source and license

This project is released under the [MIT license](LICENSE).

The transition engine and three layout assets come from
[Overcooked-AI](https://github.com/HumanCompatibleAI/overcooked_ai), commit
`7e774a1aa29c28b7b69dc0a8903822ac2c6b4f23`. Their MIT notices remain with the
bundled files. A minimal initializer supports direct engine use. Source and asset
fingerprints are recorded in
[`provenance.json`](src/overcooked_cooperation/provenance.json).

Citation metadata for this software is provided in [CITATION.cff](CITATION.cff).
