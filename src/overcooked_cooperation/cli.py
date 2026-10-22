"""Command-line access to the core method."""

import argparse
import sys
from pathlib import Path

from .api import (
    certify_cooperation,
    check_bundle,
    finite_bound,
    make_model,
    solve_model,
)
from .execution import finite_output, qualify_layout, run_schedule, verify_schedule
from .io import canonical, read_json, write_json
from .model import MODES


def parser():
    root = argparse.ArgumentParser(
        description="Exact Overcooked output bounds and returning schedules"
    )
    commands = root.add_subparsers(dest="command", required=True)
    for name, help_text in [
        ("build", "Construct an event-flow model"),
        ("solve", "Solve one model and verify the rational certificate"),
    ]:
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--layout", default="ring")
        command.add_argument("--mode", choices=MODES, default="actor0_only")
        command.add_argument(
            "--pot-states",
            action="store_true",
            help="Add local pot-state flow for sustained bounds",
        )
        command.add_argument(
            "--spatial",
            action="store_true",
            help="Add initial-position occupancy constraints",
        )
        command.add_argument("--out", required=True)
        if name == "solve":
            command.add_argument("--time-limit", type=float, default=50)
    command = commands.add_parser(
        "check", help="Check a saved exact certificate without a solver"
    )
    command.add_argument("record")
    command = commands.add_parser(
        "qualify", help="Check eligibility for independent local clocks"
    )
    command.add_argument("--layout", default="asymmetric")
    command = commands.add_parser(
        "schedule", help="Execute and confirm returning production words"
    )
    command.add_argument("--layout", default="ring")
    command.add_argument(
        "--scheduling", choices=("auto", "synchronous", "asynchronous"), default="auto"
    )
    command.add_argument("--max-steps", type=int, default=6000)
    command.add_argument("--observe-until", type=int, default=0)
    command.add_argument("--out", required=True)
    command = commands.add_parser(
        "verify-schedule", help="Replay and verify a saved schedule"
    )
    command.add_argument("record")
    command = commands.add_parser(
        "finite", help="Compute an endpoint upper bound or verified schedule count"
    )
    command.add_argument("record")
    command.add_argument("--horizon", type=int, required=True)
    command = commands.add_parser(
        "certify", help="Compare two exact solo bounds with a replayed team schedule"
    )
    command.add_argument("--solo0", required=True)
    command.add_argument("--solo1", required=True)
    command.add_argument("--schedule", required=True)
    command.add_argument("--out", required=True)
    command = commands.add_parser("demo", help="Run one end-to-end core example")
    command.add_argument("--layout", default="ring")
    command.add_argument(
        "--scheduling", choices=("auto", "synchronous", "asynchronous"), default="auto"
    )
    command.add_argument("--out", required=True)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command in ("build", "solve"):
            model = make_model(
                args.layout, args.mode, pot_states=args.pot_states, spatial=args.spatial
            )
            value = (
                model
                if args.command == "build"
                else solve_model(model, time_limit=args.time_limit)
            )
            write_json(args.out, value)
            result = value.get(
                "result",
                {
                    "rows": len(model["row_labels"]),
                    "variables": model["variable_count"],
                },
            )
        elif args.command == "check":
            result = check_bundle(read_json(args.record))
            if not result["dual_valid_exact"]:
                raise ValueError("The record has no valid exact dual bound")
        elif args.command == "qualify":
            result = qualify_layout(args.layout)
        elif args.command == "schedule":
            result = run_schedule(
                args.layout,
                scheduling=args.scheduling,
                max_steps=args.max_steps,
                observe_until=args.observe_until,
            )
            write_json(args.out, result)
            result = {
                key: result[key]
                for key in (
                    "layout",
                    "scheduling",
                    "observed_steps",
                    "rate",
                    "joint_period",
                    "joint_period_type",
                )
            }
        elif args.command == "verify-schedule":
            result = verify_schedule(read_json(args.record))
        elif args.command == "finite":
            record = read_json(args.record)
            if "certificate" in record:
                result = finite_bound(record, args.horizon)
            else:
                verify_schedule(record)
                result = finite_output(record, args.horizon)
        elif args.command == "certify":
            result = certify_cooperation(
                read_json(args.solo0), read_json(args.solo1), read_json(args.schedule)
            )
            write_json(args.out, result)
        else:
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=False)
            solo0 = solve_model(make_model(args.layout, "actor0_only"))
            solo1 = solve_model(make_model(args.layout, "actor1_only"))
            schedule = run_schedule(args.layout, scheduling=args.scheduling)
            result = certify_cooperation(solo0, solo1, schedule)
            for name, value in [
                ("solo0", solo0),
                ("solo1", solo1),
                ("schedule", schedule),
                ("certificate", result),
            ]:
                write_json(out / (name + ".json"), value)
        print(canonical(result))
        return 0
    except (
        ValueError,
        RuntimeError,
        FileExistsError,
        FileNotFoundError,
        ImportError,
        KeyError,
    ) as exc:
        print("error: " + str(exc), file=sys.stderr)
        return 1
