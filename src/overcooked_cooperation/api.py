"""Build, solve and certify policy-independent output bounds."""

import warnings
from fractions import Fraction
from hashlib import sha256
from pathlib import Path

from .exact import check_algebra, validate_model
from .io import canonical, fingerprint, read_json
from .model import build_model
from .state_constraints import base_part, strengthen_model

PACKAGE_ROOT = Path(__file__).resolve().parent
LAYOUTS = {
    "ring": "coordination_ring",
    "asymmetric": "asymmetric_advantages",
    "circuit": "counter_circuit_o_1order",
}


def layout_text(name_or_path):
    path = Path(name_or_path)
    if path.is_file():
        return path.read_text(encoding="utf-8"), path.stem
    name = LAYOUTS.get(str(name_or_path), str(name_or_path))
    path = PACKAGE_ROOT / "layouts" / (name + ".layout")
    if not path.is_file():
        raise ValueError("Choose ring, asymmetric, circuit, or a .layout file")
    return path.read_text(encoding="utf-8"), name


def make_model(name_or_path, mode="actor0_only", *, pot_states=False, spatial=False):
    text, name = layout_text(name_or_path)
    sources = read_json(PACKAGE_ROOT / "provenance.json")["native_engine"]
    model = build_model(
        text,
        mode,
        layout_binding={
            "id": name,
            "text": text,
            "sha256": sha256(text.encode()).hexdigest(),
        },
        source_binding=sources,
    )
    return strengthen_model(model, pot_states=pot_states, spatial=spatial)


def check_model(model):
    if (
        model.get("source_binding")
        != read_json(PACKAGE_ROOT / "provenance.json")["native_engine"]
    ):
        raise ValueError("Model native-source identity differs from this release")
    state = model.get("state_constraints")
    result = validate_model(base_part(model))
    if state:
        expected = strengthen_model(
            base_part(model), pot_states=state["pot_states"], spatial=state["spatial"]
        )
        if canonical(expected) != canonical(model):
            raise ValueError(
                "State-constraint rows differ from the native construction"
            )
        result.update(
            rows=len(model["row_labels"]),
            variables=model["variable_count"],
            state_constraint_rows_checked=True,
        )
    return result


def check_bundle(bundle):
    model, certificate = bundle["model"], bundle["certificate"]
    if certificate.get("model_sha256") != fingerprint(model):
        raise ValueError("Certificate belongs to a different model")
    result = check_model(model)
    result.update(check_algebra(model, certificate))
    return result


def solve_model(model, *, time_limit=50, max_denominator=1_000_000):
    """One numerical proposal, followed by exact checks of every column."""
    if time_limit <= 0 or max_denominator < 1:
        raise ValueError("Positive solver limits are required")
    check_model(model)
    import numpy as np
    from scipy.optimize import OptimizeWarning, linprog
    from scipy.sparse import coo_matrix

    rr, cc, vv = [], [], []
    for row_id, row in enumerate(model["A_eq_sparse"]):
        for column, value in row:
            rr.append(row_id)
            cc.append(column)
            vv.append(value)
    matrix = coo_matrix(
        (vv, (rr, cc)), shape=(len(model["b_eq"]), model["variable_count"])
    ).tocsr()
    objective = np.zeros(model["variable_count"])
    for column, value in model["objective_sparse"]:
        objective[column] = value
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=OptimizeWarning,
            message="Unrecognized options detected.*threads.*",
        )
        solution = linprog(
            objective,
            A_eq=matrix,
            b_eq=np.asarray(model["b_eq"]),
            bounds=(0, None),
            method="highs",
            options={"time_limit": float(time_limit), "threads": 1},
        )
    if not solution.success:
        raise RuntimeError(
            "The solver returned no certificate proposal: " + solution.message
        )

    def rational(value):
        return str(Fraction(float(value)).limit_denominator(max_denominator))

    certificate = {
        "model_sha256": fingerprint(model),
        "dual_eq": [rational(x) for x in solution.eqlin.marginals],
        "primal": [rational(x) for x in solution.x],
        "rationalization_denominator_limit": max_denominator,
    }
    bundle = {"schema_version": 1, "model": model, "certificate": certificate}
    result = check_bundle(bundle)
    if not result["dual_valid_exact"] or Fraction(result["dual_mean_ticks_lower"]) <= 0:
        raise RuntimeError(
            "The numerical proposal did not yield a positive exact dual bound"
        )
    bundle["result"] = result
    return bundle


def finite_bound(bundle, horizon):
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 0:
        raise ValueError("Horizon must be a nonnegative integer")
    result = check_bundle(bundle)
    if not result["dual_valid_exact"]:
        raise ValueError("An exactly feasible dual is required")
    if bundle["model"].get("state_constraints"):
        raise ValueError(
            "Endpoint bounds use base event flow; solve without state flags"
        )
    from .finite import derive_potential

    potential = derive_potential(bundle["model"], bundle["certificate"])
    workload, credit = Fraction(potential["L"]), Fraction(potential["B"])
    return {
        "horizon": horizon,
        "upper_soups": (horizon + credit) // workload,
        "L": str(workload),
        "B": str(credit),
    }


def certify_cooperation(solo0, solo1, schedule, horizons=(200, 400)):
    from .execution import finite_output, verify_schedule

    verify_schedule(schedule)
    bundles = [solo0, solo1]
    expected_modes = ["actor0_only", "actor1_only"]
    upper = []
    for bundle, mode in zip(bundles, expected_modes):
        if bundle["model"]["mode"] != mode:
            raise ValueError(
                "Provide actor0_only and actor1_only certificates in order"
            )
        if bundle["model"]["layout_binding"]["sha256"] != schedule["layout_sha256"]:
            raise ValueError("Bounds and schedule use different layouts")
        result = check_bundle(bundle)
        if not result["dual_valid_exact"] or not result["native_rate_upper"]:
            raise ValueError("An exactly verified positive solo bound is required")
        upper.append(Fraction(result["native_rate_upper"]))
    solo_upper = max(upper)
    team_rate = Fraction(schedule["rate"])
    relative = team_rate / solo_upper - 1
    finite = []
    for horizon in horizons:
        count = finite_output(schedule, horizon)
        limit = max(finite_bound(b, horizon)["upper_soups"] for b in bundles)
        finite.append(
            {
                **count,
                "solo_upper_soups": limit,
                "additional_soups": max(0, count["soups"] - limit),
            }
        )
    return {
        "solo_rate_upper": str(solo_upper),
        "team_rate": str(team_rate),
        "strict_resource_benefit": relative > 0,
        "resource_gain_guarantee": str(relative) if relative > 0 else None,
        "at_least_ten_percent": relative >= Fraction(1, 10),
        "finite": finite,
    }
