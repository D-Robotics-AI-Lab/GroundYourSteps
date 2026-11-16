"""Command line entry point for calibration pairs and schedule optimization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ConfigError, load_config


def _config(args: argparse.Namespace):
    return load_config(
        args.config,
        task=args.task,
        budget=getattr(args, "budget", 4),
        run_root=args.run_root,
        device=args.device,
        micro_batch=getattr(args, "micro_batch", None),
    )


def cmd_config(args: argparse.Namespace) -> None:
    import yaml

    config = _config(args)
    print(yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True))


def cmd_pairs(args: argparse.Namespace) -> None:
    from .runtime import evaluator_session

    config = _config(args)
    with evaluator_session(
        config, args.policy_python, pairs_only=True, overwrite=args.overwrite
    ) as session:
        print(f"Wrote {session.num_pairs} calibration pairs to {config.pairs_dir}")


def cmd_solve(args: argparse.Namespace) -> None:
    from .deploy import knot_axis_from_metadata, schedule_document
    from .run.artifacts import write_json
    from .runtime import evaluator_session
    from .search.pso import search

    config = _config(args)
    with evaluator_session(config, args.policy_python) as session:
        solution = search(
            session,
            budget=config.budget,
            method=config.method,
            solver=config.solver,
            logger=sys.stderr,
        )
        knot_axis = knot_axis_from_metadata(session.metadata)
    write_json(
        config.schedule_path,
        schedule_document(
            solution.nodes,
            knot_axis=knot_axis,
            fingerprint=config.fingerprint,
            provenance={
                "task": config.task,
                "budget": config.budget,
                "order": solution.order,
                "observation_weight": solution.observation_weight,
                "loss": solution.loss,
            },
        ),
    )
    print(
        json.dumps(
            {
                "schedule": str(config.schedule_path),
                "nodes": solution.nodes,
                "loss": solution.loss,
                "order": solution.order,
                "evaluations": solution.evaluations,
            }
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gys", description="Ground Your Steps")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("config", cmd_config), ("pairs", cmd_pairs), ("solve", cmd_solve)):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--task", help="task name used to resolve {task} in the data path")
        command.add_argument("--run-root", type=Path)
        command.add_argument("--device")
        if name in ("pairs", "solve"):
            command.add_argument(
                "--policy-python",
                type=Path,
                help="Python interpreter in the policy environment",
            )
        if name != "pairs":
            command.add_argument("--budget", type=int, required=True)
        if name == "pairs":
            command.add_argument("--overwrite", action="store_true")
        if name == "solve":
            command.add_argument("--micro-batch", type=int)
        command.set_defaults(handler=handler)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.handler(args)
    except (ConfigError, OSError, ValueError, FloatingPointError, RuntimeError) as exc:
        build_parser().exit(2, f"gys: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
