#!/usr/bin/env python3
"""Run fixed-architecture E151 loss-curriculum recipes from random initialization."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

SIMPLEXFOLD_ROOT = Path(__file__).resolve().parents[1]
NANOFOLD_ROOT = SIMPLEXFOLD_ROOT.parents[1]
RUNNER = SIMPLEXFOLD_ROOT / "scripts" / "run_nanofold_public_benchmarks.py"
DEFAULT_OUTPUT_DIR = SIMPLEXFOLD_ROOT / "artifacts" / "nanofold_public_benchmarks"
RECIPE_ORDER = ("no_staged_losses", "staged_losses", "two_stage_losses")
RECIPES = {
    "no_staged_losses": {
        "schedule": SIMPLEXFOLD_ROOT / "configs" / "e151_fixed_arch_no_staged_losses_schedule.toml",
        "run_name": "e151_fixed_arch_no_staged_losses_s30000_c256_m64",
        "description": "final E151 loss recipe active from step 0",
    },
    "staged_losses": {
        "schedule": SIMPLEXFOLD_ROOT / "configs" / "e151_fixed_arch_staged_losses_schedule.toml",
        "run_name": "e151_fixed_arch_staged_losses_s30000_c256_m64",
        "description": "historical E151 loss stages with architecture changes removed",
    },
    "two_stage_losses": {
        "schedule": SIMPLEXFOLD_ROOT / "configs" / "e151_fixed_arch_two_stage_losses_schedule.toml",
        "run_name": "e151_fixed_arch_two_stage_losses_s30000_c256_m64",
        "description": "base losses through step 9000, then final E151 losses",
    },
}
FIXED_E151_ARCH_ARGS = [
    "--simplex-edge-frame-message-runtime-scale",
    "0.0125",
    "--simplex-boundary-readout-directionality-runtime-scale",
    "0.25",
    "--simplex-face-top-k",
    "24",
    "--simplex-tetra-top-k",
    "48",
    "--simplex-vertex-star-context-runtime-scale",
    "1.0",
    "--simplex-edge-star-context-runtime-scale",
    "0.5",
    "--simplex-geometry-distance-weight",
    "0.025",
]


def _selected_recipes(args: argparse.Namespace) -> tuple[str, ...]:
    return RECIPE_ORDER if args.all_recipes else (args.recipe,)


def _run_name_for_recipe(args: argparse.Namespace, recipe: str) -> str:
    if args.run_name:
        return f"{args.run_name}_{recipe}" if args.all_recipes else args.run_name
    return str(RECIPES[recipe]["run_name"])


def build_command(args: argparse.Namespace, recipe: str | None = None) -> list[str]:
    recipe = recipe or args.recipe
    if recipe not in RECIPES:
        raise ValueError(f"Unknown E151 recipe: {recipe}")
    extra_args = list(args.extra_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    command = [
        sys.executable,
        str(RUNNER),
        "--nanofold-root",
        str(args.nanofold_root),
        "--model-config",
        str(args.model_config),
        "--variants",
        "full_msa_to_face",
        "--output-dir",
        str(args.output_dir),
        "--run-name",
        _run_name_for_recipe(args, recipe),
        "--steps",
        str(args.steps),
        "--crop-size",
        str(args.crop_size),
        "--msa-depth",
        str(args.msa_depth),
        "--extra-msa-depth",
        "0",
        "--max-templates",
        "0",
        "--batch-size",
        "1",
        "--grad-accum-steps",
        "8",
        "--learning-rate",
        "0.001",
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--eval-every",
        str(args.eval_every),
        "--n-cycles",
        "4",
        "--mixed-precision",
        "off",
        "--num-workers",
        "0",
        "--max-parameters",
        str(args.max_parameters),
        *FIXED_E151_ARCH_ARGS,
        "--training-stage-schedule",
        str(RECIPES[recipe]["schedule"]),
    ]
    if args.device:
        command.extend(["--device", args.device])
    command.extend(extra_args)
    return command


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    return [build_command(args, recipe) for recipe in _selected_recipes(args)]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nanofold-root", type=Path, default=NANOFOLD_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--recipe",
        choices=RECIPE_ORDER,
        default="staged_losses",
        help="Fixed-architecture E151 loss-curriculum recipe to run.",
    )
    parser.add_argument(
        "--all-recipes",
        action="store_true",
        help="Run or dry-run all fixed-architecture E151 loss-curriculum recipes.",
    )
    parser.add_argument("--run-name", default=None, help="Override the run name for a single recipe.")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--model-config", default="simplexfold_medium_param_matched")
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--msa-depth", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--max-parameters", type=int, default=3261974)
    parser.add_argument("--device", default="")
    parser.add_argument("--dry-run", action="store_true", help="Print the underlying command and exit.")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Arguments after -- are passed to the benchmark runner.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    commands = build_commands(args)
    if args.dry_run:
        for command in commands:
            print(shlex.join(command))
        return 0
    for command in commands:
        status = subprocess.call(command)
        if status != 0:
            return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
