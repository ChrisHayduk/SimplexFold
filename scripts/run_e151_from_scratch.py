#!/usr/bin/env python3
"""Prepare or explicitly execute one fresh E151-inspired research-v2 recipe.

Historical E151 configuration is not reconstructed. A positive constructor
context scale and a seed are mandatory. No job runs without --execute.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from make_e151_research_recipe import make_recipe


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("features", "labels", "train-manifest", "val-manifest", "scorer-root", "output-dir"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--export-dir", type=Path, help="Required from the main checkout; omit inside a pinned E07 export")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--global-context-scale", required=True, type=float)
    parser.add_argument("--curriculum", choices=["no_staged_losses", "staged_losses", "two_stage_losses"], default="no_staged_losses")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    provenance_path = Path(__file__).resolve().parents[1] / "source_provenance.json"
    if provenance_path.exists():
        if json.loads(provenance_path.read_text()).get("revision") != "e07" or args.export_dir is not None:
            raise ValueError("Use an E07 export directly, without --export-dir")
    elif args.export_dir is None:
        raise ValueError("Main checkout execution requires a fresh --export-dir")
    if args.output_dir.exists() or (args.export_dir is not None and args.export_dir.exists()):
        raise FileExistsError("Fresh initialization requires new output and export directories")
    recipe = make_recipe(features=args.features.resolve(), labels=args.labels.resolve(),
                         train_manifest=args.train_manifest.resolve(), val_manifest=args.val_manifest.resolve(),
                         scorer_root=args.scorer_root.resolve(), seed=args.seed,
                         global_context_scale=args.global_context_scale, curriculum=args.curriculum)
    recipe["training"]["device"] = args.device
    recipe_path = args.output_dir.with_name(args.output_dir.name + ".recipe.json").resolve()
    command = [sys.executable, str(Path(__file__).with_name("run_research_experiment.py").resolve()),
               "--recipe", str(recipe_path), "--output-dir", str(args.output_dir.resolve())]
    if args.export_dir is not None:
        command += ["--revision", "e07", "--export-dir", str(args.export_dir.resolve())]
    if args.execute:
        recipe_path.parent.mkdir(parents=True, exist_ok=True)
        with recipe_path.open("x") as handle:
            json.dump(recipe, handle, indent=2, allow_nan=False)
        return subprocess.call(command)
    print(json.dumps({"recipe": recipe, "command": command, "shell_preview": shlex.join(command)}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
