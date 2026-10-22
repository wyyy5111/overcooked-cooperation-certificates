"""A complete Ring certificate using the public API."""

from overcooked_cooperation import certify_cooperation, make_model, solve_model
from overcooked_cooperation.execution import run_schedule

solo0 = solve_model(make_model("ring", "actor0_only"))
solo1 = solve_model(make_model("ring", "actor1_only"))
schedule = run_schedule("ring")
print(certify_cooperation(solo0, solo1, schedule))
