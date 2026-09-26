#!/usr/bin/env python3
"""SimplexFold public-data ablations using the version 2 research protocol.

The compatibility CLI resolves old profile/recipe flags into the same
content-bound runner as run_research_experiment.py. Each variant owns a separate
output directory. Whole-target validation and cyclic shuffled training replace
the historical cropped/finite-only evaluation and unreplayable DataLoader.
Historical branch executables remain available through their immutable refs.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from dataclasses import asdict, replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from typing import Any

from minalphafold.data import read_chain_id_manifest
from minalphafold.research import atomic_save, run
from minalphafold.trainer import (
    DataConfig,
    TrainingConfig,
    load_model_config,
    zero_dropout_model_config,
)


def _parse_chain_id_args(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    chain_ids: list[str] = []
    for value in values:
        chain_ids.extend(item.strip() for item in value.split(",") if item.strip())
    return chain_ids or None


def _write_manifest_subset(
    source: Path,
    limit: int | None,
    destination: Path,
    *,
    selected_chain_ids: list[str] | None = None,
    validate_selected: bool = True,
) -> Path:
    """Return a manifest path, optionally writing selected or prefix IDs."""
    if selected_chain_ids is None and (limit is None or limit <= 0):
        return source
    source_ids = read_chain_id_manifest(source)
    if selected_chain_ids is None:
        chain_ids = source_ids[:limit]
    else:
        if validate_selected:
            available = set(source_ids)
            missing = [
                chain_id for chain_id in selected_chain_ids if chain_id not in available
            ]
            if missing:
                sample = ", ".join(missing[:8])
                raise ValueError(f"Selected chains are not in {source}: {sample}")
        chain_ids = selected_chain_ids
    destination.parent.mkdir(parents=True, exist_ok=True)
    contents = "\n".join(chain_ids) + "\n"
    if destination.exists() and destination.read_text() != contents:
        raise ValueError("Existing selected manifest differs; use a new run directory")
    destination.write_text(contents, encoding="utf-8")
    return destination


def _variant_config(base_config: Any, variant: str) -> Any:
    if variant == "no_simplex":
        return replace(base_config, use_simplicial_evoformer=False)
    if variant == "faces":
        return replace(
            base_config,
            use_simplicial_evoformer=True,
            simplex_use_faces=True,
            simplex_use_tetra=False,
            simplex_use_msa_to_face=False,
        )
    if variant == "full":
        return replace(
            base_config,
            use_simplicial_evoformer=True,
            simplex_use_faces=True,
            simplex_use_tetra=True,
            simplex_use_msa_to_face=False,
        )
    if variant == "msa_to_face":
        return replace(
            base_config,
            use_simplicial_evoformer=True,
            simplex_use_faces=True,
            simplex_use_tetra=False,
            simplex_use_msa_to_face=True,
        )
    raise ValueError(f"Unknown variant: {variant}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SimplexFold ablations on NanoFold public train/val data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--nanofold-root", type=Path, default=ROOT.parent / "nanoFold-Competition"
    )
    parser.add_argument("--model-config", default="tiny")
    parser.add_argument(
        "--zero-dropout",
        action="store_true",
        help="Clone the selected model profile with all dropout rates set to 0. Useful for memorization/debug runs.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["no_simplex", "faces", "full"],
        choices=["no_simplex", "faces", "full", "msa_to_face"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "nanofold_public_benchmarks",
    )
    parser.add_argument(
        "--run-name",
        help="Use a stable output subdirectory instead of a timestamp. Useful with --auto-resume.",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=0,
        help="0 means use the full official train manifest.",
    )
    parser.add_argument(
        "--val-limit",
        type=int,
        default=0,
        help="0 means use the full official validation manifest.",
    )
    parser.add_argument(
        "--train-chain-ids",
        nargs="+",
        help="Explicit train chain IDs, comma or space separated.",
    )
    parser.add_argument(
        "--val-chain-ids",
        nargs="+",
        help="Explicit validation chain IDs, comma or space separated.",
    )
    parser.add_argument(
        "--overfit-chain-id",
        help="Use the same chain ID for train and validation manifests, bypassing official split membership checks.",
    )
    parser.add_argument(
        "--steps", type=int, default=50, help="Optimizer steps per variant."
    )
    parser.add_argument(
        "--eval-every", type=int, default=0, help="0 evaluates only at the final step."
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="0 logs only the first step and eval steps.",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=0,
        help="0 evaluates the whole selected val manifest.",
    )
    parser.add_argument(
        "--eval-max-val-batches",
        type=int,
        default=None,
        help="Validation batches for intermediate --eval-every evaluations. Defaults to --max-val-batches.",
    )
    parser.add_argument(
        "--final-max-val-batches",
        type=int,
        default=None,
        help="Validation batches for final evaluation. Defaults to --max-val-batches.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save a resumable latest checkpoint every N optimizer steps. 0 saves only at stop/final.",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Resume from <checkpoint-dir>/<variant>_latest.pt when present.",
    )
    parser.add_argument(
        "--stop-after-seconds",
        type=int,
        default=0,
        help="Gracefully stop and checkpoint this process after N seconds. 0 disables the guard.",
    )
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--msa-depth", type=int, default=32)
    parser.add_argument("--extra-msa-depth", type=int, default=0)
    parser.add_argument("--max-templates", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Optimizer-step accumulation factor; effective batch = batch-size * grad-accum-steps.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-samples", type=int, default=0)
    parser.add_argument("--lr-decay-samples", type=int, default=None)
    parser.add_argument("--lr-decay-factor", type=float, default=1.0)
    parser.add_argument("--grad-clip-norm", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=None)
    parser.add_argument(
        "--use-clamped-fape",
        type=float,
        default=0.9,
        help="Backbone FAPE clamp mixture. 0.9 matches AF2's expected 90/10 clamped/unclamped batches; -1 restores fully clamped legacy behavior.",
    )
    parser.add_argument("--msa-loss-weight", type=float, default=2.0)
    parser.add_argument("--distogram-loss-weight", type=float, default=0.3)
    parser.add_argument("--confidence-loss-weight", type=float, default=0.01)
    parser.add_argument("--simplex-aux-weight", type=float, default=1.0)
    parser.add_argument("--backbone-loss-weight", type=float, default=1.0)
    parser.add_argument("--sidechain-fape-loss-weight", type=float, default=1.0)
    parser.add_argument("--torsion-loss-weight", type=float, default=1.0)
    parser.add_argument("--loss-weight-ramp-start-step", type=int, default=None)
    parser.add_argument("--loss-weight-ramp-steps", type=int, default=1)
    parser.add_argument("--msa-loss-weight-final", type=float, default=None)
    parser.add_argument("--distogram-loss-weight-final", type=float, default=None)
    parser.add_argument("--confidence-loss-weight-final", type=float, default=None)
    parser.add_argument("--simplex-aux-weight-final", type=float, default=None)
    parser.add_argument("--backbone-loss-weight-final", type=float, default=None)
    parser.add_argument("--sidechain-fape-loss-weight-final", type=float, default=None)
    parser.add_argument("--torsion-loss-weight-final", type=float, default=None)
    parser.add_argument("--finetune-start-step", type=int, default=None)
    parser.add_argument("--finetune-lr-scale", type=float, default=0.5)
    parser.add_argument("--violation-ramp-steps", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n-cycles",
        type=int,
        default=None,
        help="Recycling cycles. Defaults to the model profile's recommended_n_cycles.",
    )
    parser.add_argument("--n-ensemble", type=int, default=1)
    parser.add_argument(
        "--mixed-precision", choices=["off", "bf16", "fp16"], default="off"
    )
    for option in ("features-dir", "labels-dir", "train-manifest", "val-manifest"):
        parser.add_argument("--" + option, type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.overfit_chain_id:
        raise ValueError(
            "Public-data experiments require disjoint cohorts; use the dedicated overfit script for debugging"
        )
    if args.mixed_precision != "off":
        raise ValueError(
            "Research v2 currently validates FP32 execution; AMP requires its own validated recipe"
        )
    if any(
        value and value > 0
        for value in (
            args.max_val_batches,
            args.eval_max_val_batches,
            args.final_max_val_batches,
        )
    ):
        raise ValueError(
            "Research v2 evaluates the complete selected manifest; use --val-limit for an explicitly smaller smoke cohort"
        )
    if args.checkpoint_dir is not None:
        raise ValueError(
            "Research v2 binds each checkpoint to its variant output directory"
        )
    if len(args.variants) != len(set(args.variants)):
        raise ValueError("Variants must be distinct")
    nanofold_root = args.nanofold_root.resolve()
    features_dir = args.features_dir or nanofold_root / "data/processed_features"
    labels_dir = args.labels_dir or nanofold_root / "data/processed_labels"
    train_manifest = args.train_manifest or nanofold_root / "data/manifests/train.txt"
    val_manifest = args.val_manifest or nanofold_root / "data/manifests/val.txt"
    output_name = args.run_name or str(uuid.uuid4())
    if Path(output_name).name != output_name:
        raise ValueError("Run name must be a single path component")
    output_dir = (args.output_dir / output_name).resolve()
    if (
        output_dir.exists()
        and any(output_dir.iterdir())
        and not (args.auto_resume or args.resume_from_checkpoint)
    ):
        raise FileExistsError(
            "Run directory is not empty; choose a new run name or explicit resume"
        )
    if args.resume_from_checkpoint and len(args.variants) != 1:
        raise ValueError("Checkpoint resume requires a single variant")
    subset_dir = output_dir / "manifests"
    train_manifest_used = _write_manifest_subset(
        train_manifest,
        args.train_limit or None,
        subset_dir / "train.txt",
        selected_chain_ids=_parse_chain_id_args(args.train_chain_ids),
    )
    val_manifest_used = _write_manifest_subset(
        val_manifest,
        args.val_limit or None,
        subset_dir / "val.txt",
        selected_chain_ids=_parse_chain_id_args(args.val_chain_ids),
    )
    base_config = load_model_config(args.model_config)
    if args.zero_dropout:
        base_config = zero_dropout_model_config(base_config)
    n_cycles = (
        args.n_cycles if args.n_cycles is not None else base_config.recommended_n_cycles
    )
    data_config = DataConfig(
        processed_features_dir=features_dir,
        processed_labels_dir=labels_dir,
        train_manifest=train_manifest_used,
        val_manifest=val_manifest_used,
        val_fraction=0.0,
        crop_size=args.crop_size,
        msa_depth=args.msa_depth,
        extra_msa_depth=args.extra_msa_depth,
        max_templates=args.max_templates,
        block_delete_training_msa=False,
    )
    training_config = TrainingConfig(
        epochs=args.steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_samples=args.warmup_samples,
        lr_decay_samples=args.lr_decay_samples,
        lr_decay_factor=args.lr_decay_factor,
        grad_clip_norm=None if args.grad_clip_norm <= 0 else args.grad_clip_norm,
        ema_decay=args.ema_decay,
        use_clamped_fape=None if args.use_clamped_fape < 0 else args.use_clamped_fape,
        msa_loss_weight=args.msa_loss_weight,
        distogram_loss_weight=args.distogram_loss_weight,
        confidence_loss_weight=args.confidence_loss_weight,
        simplex_aux_weight=args.simplex_aux_weight,
        backbone_loss_weight=args.backbone_loss_weight,
        sidechain_fape_loss_weight=args.sidechain_fape_loss_weight,
        torsion_loss_weight=args.torsion_loss_weight,
        loss_weight_ramp_start_step=args.loss_weight_ramp_start_step,
        loss_weight_ramp_steps=args.loss_weight_ramp_steps,
        msa_loss_weight_final=args.msa_loss_weight_final,
        distogram_loss_weight_final=args.distogram_loss_weight_final,
        confidence_loss_weight_final=args.confidence_loss_weight_final,
        simplex_aux_weight_final=args.simplex_aux_weight_final,
        backbone_loss_weight_final=args.backbone_loss_weight_final,
        sidechain_fape_loss_weight_final=args.sidechain_fape_loss_weight_final,
        torsion_loss_weight_final=args.torsion_loss_weight_final,
        finetune_start_step=args.finetune_start_step,
        finetune_lr_scale=args.finetune_lr_scale,
        violation_ramp_steps=args.violation_ramp_steps,
        device=args.device,
        seed=args.seed,
        num_workers=args.num_workers,
        n_cycles=n_cycles,
        n_ensemble=args.n_ensemble,
    )
    rows = []
    for variant in args.variants:
        target = output_dir / variant
        if (
            args.resume_from_checkpoint
            and args.resume_from_checkpoint.resolve() != target / "checkpoint.pt"
        ):
            raise ValueError(
                "Resume checkpoint must be this variant's research-v2 checkpoint.pt"
            )
        overrides = asdict(base_config)
        # Variant selectors remain authoritative for the corresponding booleans.
        for key in (
            "use_simplicial_evoformer",
            "simplex_use_faces",
            "simplex_use_tetra",
            "simplex_use_msa_to_face",
        ):
            overrides.pop(key, None)
        recipe = {
            "model_config": args.model_config,
            "model_overrides": overrides,
            "variant": variant,
            "data": asdict(data_config),
            "training": asdict(training_config),
            "eval_every": args.eval_every,
            "scorer_root": str(nanofold_root),
        }
        row = run(
            recipe,
            target,
            resume=bool(
                args.resume_from_checkpoint
                or (args.auto_resume and (target / "checkpoint.pt").exists())
            ),
            max_runtime_seconds=args.stop_after_seconds
            if args.stop_after_seconds > 0
            else None,
        )
        rows.append(dict(variant=variant, **row))
    atomic_save(output_dir / "results.json", rows)
    return rows


if __name__ == "__main__":
    rows = main()
    raise SystemExit(75 if any(row["paused"] for row in rows) else 0)
