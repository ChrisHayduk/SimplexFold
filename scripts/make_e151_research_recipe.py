#!/usr/bin/env python3
"""Write an explicit fresh-initialization E151-inspired research-v2 recipe.

This fixes inactive runtime knobs in the historical launcher. It is not proof
that the missing historical E151 configuration/checkpoint used these settings.
The global-context constructor scale must be supplied explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E07 = "e960359b0168fc3e1c402f61dd698499102af239"


def make_recipe(
    *,
    features,
    labels,
    train_manifest,
    val_manifest,
    scorer_root,
    seed,
    global_context_scale,
    curriculum="no_staged_losses",
):
    if not math.isfinite(global_context_scale) or global_context_scale <= 0:
        raise ValueError(
            "A positive global-context constructor scale is needed for star context"
        )
    schedule = f"configs/e151_fixed_arch_{curriculum}_schedule.toml"
    provenance = ROOT / "source_provenance.json"
    if provenance.exists():
        receipt = json.loads(provenance.read_text())
        if receipt.get("commit") != E07 or receipt.get("revision") != "e07":
            raise ValueError("E151-inspired recipes require the pinned E07 source")
        payload = (ROOT / schedule).read_bytes()
        if hashlib.sha256(payload).hexdigest() != receipt["original_files"].get(schedule):
            raise ValueError("Exported curriculum differs from the pinned original source")
    else:
        payload = subprocess.check_output(["git", "show", f"{E07}:{schedule}"], cwd=ROOT)
    stages = tomllib.loads(payload.decode())["stage"]
    for stage in stages:
        stage.pop("source_run", None)
        stage.setdefault("training_config", {})
    return {
        "model_config": "simplexfold_medium_param_matched",
        "variant": "full_msa_to_face",
        "model_overrides": {
            "simplex_edge_frame_message_scale": 0.0125,
            "simplex_global_context_scale": global_context_scale,
            "simplex_vertex_star_context_scale": 1.0,
            "simplex_edge_star_context_scale": 0.5,
            "simplex_boundary_readout_directionality": 0.25,
            "simplex_face_top_k": 24,
            "simplex_tetra_top_k": 48,
            "simplex_geometry_distance_weight": 0.025,
        },
        "data": {
            "processed_features_dir": str(features),
            "processed_labels_dir": str(labels),
            "train_manifest": str(train_manifest),
            "val_manifest": str(val_manifest),
            "crop_size": 256,
            "msa_depth": 64,
            "extra_msa_depth": 0,
            "max_templates": 0,
            "block_delete_training_msa": False,
        },
        "training": {
            "epochs": 30000,
            "batch_size": 1,
            "grad_accum_steps": 8,
            "seed": seed,
            "device": "cuda",
            "learning_rate": 0.001,
            "n_cycles": 4,
            "n_ensemble": 1,
            "simplex_edge_frame_message_runtime_scale": 0.0125,
            "simplex_vertex_star_context_runtime_scale": 1.0,
            "simplex_edge_star_context_runtime_scale": 0.5,
            "simplex_boundary_readout_directionality_runtime_scale": 0.25,
        },
        "stages": stages,
        "eval_every": 500,
        "scorer_root": str(scorer_root),
        "max_parameters": 3261974,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in (
        "features",
        "labels",
        "train-manifest",
        "val-manifest",
        "scorer-root",
        "output",
    ):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--global-context-scale", required=True, type=float)
    parser.add_argument(
        "--curriculum",
        choices=["no_staged_losses", "staged_losses", "two_stage_losses"],
        default="no_staged_losses",
    )
    args = vars(parser.parse_args())
    output = args.pop("output")
    recipe = make_recipe(**args)
    with output.open("x") as handle:
        json.dump(recipe, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
