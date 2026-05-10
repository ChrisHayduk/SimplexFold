#!/usr/bin/env python3
"""Train and evaluate SimplexFold ablations on NanoFold public data.

The NanoFold public data contract is a pair of official manifests
(``data/manifests/train.txt`` and ``data/manifests/val.txt``) plus encoded
per-chain NPZ caches.  This runner keeps that split intact, optionally writes
small prefix manifests for smoke tests, and reports metrics that are useful for
research figures:

* train / validation AlphaFold loss,
* validation C-alpha RMSD after Kabsch alignment,
* validation lDDT-Cα,
* simplex auxiliary loss components when the variant exposes them,
* parameter count and throughput.

Example local smoke run:

    python scripts/run_nanofold_public_benchmarks.py \\
      --nanofold-root /Users/christopherhayduk/Projects/nanoFold-Competition \\
      --model-config tiny \\
      --variants no_simplex faces full \\
      --train-limit 8 --val-limit 4 --steps 2 --crop-size 32 \\
      --msa-depth 8 --extra-msa-depth 0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any, ContextManager

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minalphafold.data import read_chain_id_manifest  # noqa: E402
from minalphafold.losses import AlphaFoldLoss  # noqa: E402
from minalphafold.model import AlphaFold2  # noqa: E402
from minalphafold.trainer import (  # noqa: E402
    DataConfig,
    TrainingConfig,
    apply_loss_weight_schedule,
    build_dataloader,
    build_ema_model,
    build_optimizer,
    learning_rate_at_step,
    load_model_config,
    loss_inputs_from_batch,
    model_inputs_from_batch,
    move_to_device,
    resolve_device,
    set_optimizer_learning_rate,
    set_seed,
    simplex_local_neighbor_k_at_step,
    simplex_topology_teacher_forcing_weight_at_step,
    simplex_update_scale_at_step,
    zero_dropout_model_config,
)


def _mean(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return statistics.fmean(clean) if clean else float("nan")


def _prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def _autocast_context(device: torch.device, mixed_precision: str) -> ContextManager[Any]:
    if mixed_precision == "off" or device.type not in {"cuda", "cpu"}:
        return nullcontext()
    dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


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
            missing = [chain_id for chain_id in selected_chain_ids if chain_id not in available]
            if missing:
                sample = ", ".join(missing[:8])
                raise ValueError(f"Selected chains are not in {source}: {sample}")
        chain_ids = selected_chain_ids
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(chain_ids) + "\n", encoding="utf-8")
    return destination


def _load_foldscore_components(nanofold_root: Path):
    """Return NanoFold's official FoldScore component function when present."""
    if str(nanofold_root) not in sys.path:
        sys.path.insert(0, str(nanofold_root))
    try:
        from nanofold.metrics import foldscore_components
    except Exception as exc:  # pragma: no cover - defensive fallback for standalone installs.
        print(f"[nanofold-public] FoldScore components unavailable: {exc}")
        return None
    return foldscore_components


def _kabsch_aligned_rmsd(pred: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor) -> float:
    valid = mask.bool()
    if int(valid.sum().item()) < 3:
        return float("nan")
    pred_sel = pred[valid]
    truth_sel = truth[valid]
    pred_center = pred_sel.mean(dim=0, keepdim=True)
    truth_center = truth_sel.mean(dim=0, keepdim=True)
    pred_centered = pred_sel - pred_center
    truth_centered = truth_sel - truth_center
    covariance = pred_centered.transpose(0, 1) @ truth_centered
    u, _, vh = torch.linalg.svd(covariance)
    rotation = u @ vh
    if torch.det(rotation) < 0:
        u = u.clone()
        u[:, -1] *= -1
        rotation = u @ vh
    aligned = (pred_sel - pred_center) @ rotation + truth_center
    return float(torch.sqrt(torch.mean(torch.sum((aligned - truth_sel) ** 2, dim=-1))).item())


def _lddt_ca(pred: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor, cutoff: float = 15.0) -> float:
    valid = mask.bool()
    if int(valid.sum().item()) < 2:
        return float("nan")
    d_true = torch.cdist(truth, truth)
    d_pred = torch.cdist(pred, pred)
    eye = torch.eye(truth.shape[0], dtype=torch.bool, device=truth.device)
    pair_mask = (d_true < cutoff) & valid[:, None] & valid[None, :] & (~eye)
    per_res_counts = pair_mask.sum(dim=-1)
    valid_res = per_res_counts > 0
    if int(valid_res.sum().item()) == 0:
        return float("nan")
    diff = torch.abs(d_true - d_pred)
    thresholds = torch.as_tensor([0.5, 1.0, 2.0, 4.0], device=truth.device, dtype=truth.dtype)
    within = (diff[None] < thresholds[:, None, None]) & pair_mask[None]
    fractions = within.sum(dim=-1).to(truth.dtype) / per_res_counts[None].clamp_min(1).to(truth.dtype)
    per_res_lddt = fractions.mean(dim=0)
    return float(per_res_lddt[valid_res].mean().item())


def _radius_of_gyration(coords: torch.Tensor, mask: torch.Tensor) -> float:
    valid = mask.bool()
    if int(valid.sum().item()) == 0:
        return float("nan")
    coords_valid = coords[valid]
    center = coords_valid.mean(dim=0, keepdim=True)
    return float(torch.sqrt(torch.mean(torch.sum((coords_valid - center) ** 2, dim=-1))).item())


def _ca_drmsd(pred: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor) -> float:
    valid = mask.bool()
    if int(valid.sum().item()) < 2:
        return float("nan")
    pred_valid = pred[valid]
    truth_valid = truth[valid]
    pred_dist = torch.cdist(pred_valid, pred_valid)
    truth_dist = torch.cdist(truth_valid, truth_valid)
    pair_mask = ~torch.eye(pred_valid.shape[0], dtype=torch.bool, device=pred_valid.device)
    return float(torch.sqrt(torch.mean((pred_dist[pair_mask] - truth_dist[pair_mask]) ** 2)).item())


def _structure_metrics(
    outputs: dict[str, Any],
    batch: dict[str, Any],
    *,
    foldscore_components_fn: Any | None,
) -> dict[str, list[float]]:
    pred_atom14 = outputs["atom14_coords"].detach().float().cpu()
    true_atom14 = batch["true_atom_positions"].detach().float().cpu()
    true_atom14_mask = batch["true_atom_mask"].detach().float().cpu()
    pred_ca = outputs["atom14_coords"][:, :, 1, :].detach().float().cpu()
    true_ca = batch["true_atom_positions"][:, :, 1, :].detach().float().cpu()
    ca_mask = batch["true_atom_mask"][:, :, 1].detach().cpu()
    aatype = batch["aatype"].detach().cpu()
    if "seq_mask" in batch:
        ca_mask = ca_mask * batch["seq_mask"].detach().cpu()

    rmsd_values: list[float] = []
    lddt_values: list[float] = []
    drmsd_values: list[float] = []
    pred_rg_values: list[float] = []
    true_rg_values: list[float] = []
    foldscore_values: dict[str, list[float]] = {}
    for index in range(pred_ca.shape[0]):
        rmsd_values.append(_kabsch_aligned_rmsd(pred_ca[index], true_ca[index], ca_mask[index]))
        lddt_values.append(_lddt_ca(pred_ca[index], true_ca[index], ca_mask[index]))
        drmsd_values.append(_ca_drmsd(pred_ca[index], true_ca[index], ca_mask[index]))
        pred_rg_values.append(_radius_of_gyration(pred_ca[index], ca_mask[index]))
        true_rg_values.append(_radius_of_gyration(true_ca[index], ca_mask[index]))
        if foldscore_components_fn is not None:
            components = foldscore_components_fn(
                pred_atom14[index],
                true_atom14[index],
                true_atom14_mask[index].bool(),
                aatype[index],
            )
            for name, value in components.items():
                foldscore_values.setdefault(name, []).append(float(value.detach().cpu().item()))
    return {
        "ca_rmsd": rmsd_values,
        "lddt_ca": lddt_values,
        "ca_drmsd": drmsd_values,
        "pred_ca_rg": pred_rg_values,
        "true_ca_rg": true_rg_values,
        **foldscore_values,
    }


def _tensor_mean(value: Any) -> float | None:
    if not torch.is_tensor(value):
        return None
    if value.numel() == 0:
        return None
    return float(value.detach().float().mean().cpu().item())


def _loss_ready_outputs(outputs: dict[str, Any]) -> dict[str, Any]:
    """Return a lightweight output dict with floating loss tensors in fp32.

    BF16/FP16 autocast is useful for the forward pass, but FAPE, geometry
    losses, and distance-based targets are numerically safer in fp32.  We also
    avoid carrying the huge representation tensors into the loss path because
    the loss only consumes heads, structure outputs, and simplex auxiliaries.
    """
    skip_keys = {"pair_representation", "msa_representation", "single_representation"}
    result: dict[str, Any] = {}
    for key, value in outputs.items():
        if key in skip_keys:
            continue
        if torch.is_tensor(value) and value.is_floating_point():
            result[key] = value.float()
        else:
            result[key] = value
    return result


def _global_grad_norm(parameters: Any) -> float:
    norms = [
        parameter.grad.detach().float().norm(2)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not norms:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(norms), 2).cpu().item())


def _loss_with_terms(
    loss_fn: AlphaFoldLoss,
    batch: dict[str, Any],
    outputs: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_kwargs = loss_inputs_from_batch(batch, _loss_ready_outputs(outputs))
    loss, terms = loss_fn(**loss_kwargs, return_breakdown=True)
    return loss, terms


def _checkpoint_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any] | None, device: torch.device) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch_state = state["torch"]
        if torch.is_tensor(torch_state):
            torch_state = torch_state.cpu()
        torch.set_rng_state(torch_state)
    if device.type == "cuda" and "cuda" in state and torch.cuda.is_available():
        cuda_states = [
            item.cpu() if torch.is_tensor(item) else item
            for item in state["cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def _save_training_checkpoint(
    path: Path,
    *,
    model: AlphaFold2,
    optimizer: torch.optim.Optimizer,
    ema_model: torch.optim.swa_utils.AveragedModel | None,
    variant: str,
    model_profile: str,
    step: int,
    total_examples: int,
    train_losses: list[float],
    history: list[dict[str, Any]],
    last_eval: dict[str, float] | None,
    elapsed_seconds_total: float,
    stopped_early: bool,
) -> None:
    payload: dict[str, Any] = {
        "format_version": 1,
        "variant": variant,
        "model_profile": model_profile,
        "step": int(step),
        "total_examples": int(total_examples),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "ema_state_dict": ema_model.state_dict() if ema_model is not None else None,
        "train_losses": list(train_losses),
        "history": list(history),
        "last_eval": last_eval,
        "elapsed_seconds_total": float(elapsed_seconds_total),
        "stopped_early": bool(stopped_early),
        "rng_state": _checkpoint_rng_state(),
    }
    _atomic_torch_save(payload, path)


def _load_training_checkpoint(
    path: Path,
    *,
    model: AlphaFold2,
    optimizer: torch.optim.Optimizer,
    ema_model: torch.optim.swa_utils.AveragedModel | None,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if ema_model is not None and checkpoint.get("ema_state_dict") is not None:
        ema_model.load_state_dict(checkpoint["ema_state_dict"])
    _restore_rng_state(checkpoint.get("rng_state"), device)
    return checkpoint


def _evaluate(
    model: AlphaFold2,
    loss_fn: AlphaFoldLoss,
    dataloader: DataLoader,
    training_config: TrainingConfig,
    device: torch.device,
    *,
    max_batches: int | None,
    foldscore_components_fn: Any | None,
    mixed_precision: str,
    detail_path: Path | None = None,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    metric_values: dict[str, list[float]] = {"ca_rmsd": [], "lddt_ca": []}
    term_values: dict[str, list[float]] = {}
    detail_rows: list[dict[str, Any]] = []
    total_examples = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = move_to_device(batch, device)
            with _autocast_context(device, mixed_precision):
                outputs = model(**model_inputs_from_batch(batch, training_config))
            per_example_loss, terms = _loss_with_terms(loss_fn, batch, outputs)
            losses.extend(float(v) for v in per_example_loss.detach().cpu())
            metrics = _structure_metrics(
                outputs,
                batch,
                foldscore_components_fn=foldscore_components_fn,
            )
            for key, values in metrics.items():
                metric_values.setdefault(key, []).extend(values)
            if detail_path is not None:
                batch_size = int(batch["aatype"].shape[0])
                chain_ids = batch.get("chain_id", [f"batch{batch_index}_example{i}" for i in range(batch_size)])
                seq_mask = batch.get("seq_mask")
                true_ca_mask = batch["true_atom_mask"][:, :, 1]
                if torch.is_tensor(seq_mask):
                    true_ca_mask = true_ca_mask * seq_mask
                for example_index in range(batch_size):
                    row: dict[str, Any] = {
                        "batch_index": batch_index,
                        "example_index": example_index,
                        "chain_id": chain_ids[example_index],
                        "length": int(seq_mask[example_index].detach().cpu().sum().item())
                        if torch.is_tensor(seq_mask)
                        else int(true_ca_mask.shape[1]),
                        "resolved_ca": int(true_ca_mask[example_index].detach().cpu().sum().item()),
                    }
                    for key, values in metrics.items():
                        if example_index < len(values):
                            row[key] = values[example_index]
                    detail_rows.append(row)
            total_examples += int(batch["aatype"].shape[0])
            for key, value in terms.items():
                mean_value = _tensor_mean(value)
                if mean_value is not None:
                    term_values.setdefault(key, []).append(mean_value)

    result = {
        "val_examples": float(total_examples),
        "val_loss": _mean(losses),
    }
    for key, values in metric_values.items():
        result[f"val_{key}"] = _mean(values)
    for key, values in term_values.items():
        result[f"val_{key}"] = _mean(values)
    if detail_path is not None:
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in detail_rows for key in row})
        with detail_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detail_rows)
    return result


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
    if variant == "full_msa_to_face":
        return replace(
            base_config,
            use_simplicial_evoformer=True,
            simplex_use_faces=True,
            simplex_use_tetra=True,
            simplex_use_msa_to_face=True,
        )
    if variant == "full_msa_to_face_aux_closure":
        return replace(
            base_config,
            use_simplicial_evoformer=True,
            simplex_use_faces=True,
            simplex_use_tetra=True,
            simplex_use_msa_to_face=True,
        )
    if variant == "full_msa_to_face_topology_curriculum":
        return replace(
            base_config,
            use_simplicial_evoformer=True,
            simplex_use_faces=True,
            simplex_use_tetra=True,
            simplex_use_msa_to_face=True,
        )
    if variant == "full_msa_to_face_expanded_complex":
        return replace(
            base_config,
            use_simplicial_evoformer=True,
            simplex_use_faces=True,
            simplex_use_tetra=True,
            simplex_use_msa_to_face=True,
            simplex_neighbor_k=14,
        )
    if variant == "full_msa_to_face_long":
        return replace(
            base_config,
            use_simplicial_evoformer=True,
            simplex_use_faces=True,
            simplex_use_tetra=True,
            simplex_use_msa_to_face=True,
            simplex_long_min_sep=16,
            simplex_long_bias=2.0,
        )
    if variant == "full_msa_to_face_mixed":
        return replace(
            base_config,
            use_simplicial_evoformer=True,
            simplex_use_faces=True,
            simplex_use_tetra=True,
            simplex_use_msa_to_face=True,
            simplex_local_neighbor_k=4,
            simplex_local_bias=0.0,
        )
    if variant == "full_msa_to_face_mixed_soft":
        return replace(
            base_config,
            use_simplicial_evoformer=True,
            simplex_use_faces=True,
            simplex_use_tetra=True,
            simplex_use_msa_to_face=True,
            simplex_local_neighbor_k=4,
            simplex_local_bias=2.0,
        )
    if variant == "full_msa_to_face_strong_messages":
        return replace(
            base_config,
            use_simplicial_evoformer=True,
            simplex_use_faces=True,
            simplex_use_tetra=True,
            simplex_use_msa_to_face=True,
            simplex_pair_update_scale=1.5,
            simplex_single_update_scale=1.5,
        )
    if variant == "full_msa_to_face_damped_messages":
        return replace(
            base_config,
            use_simplicial_evoformer=True,
            simplex_use_faces=True,
            simplex_use_tetra=True,
            simplex_use_msa_to_face=True,
            simplex_pair_update_scale=0.5,
            simplex_single_update_scale=0.5,
        )
    if variant == "full_msa_to_face_edge_messages":
        return replace(
            base_config,
            use_simplicial_evoformer=True,
            simplex_use_faces=True,
            simplex_use_tetra=True,
            simplex_use_msa_to_face=True,
            simplex_pair_update_scale=1.5,
            simplex_single_update_scale=0.5,
        )
    if variant == "full_msa_to_face_no_recycled_topology":
        return replace(
            base_config,
            use_simplicial_evoformer=True,
            simplex_use_faces=True,
            simplex_use_tetra=True,
            simplex_use_msa_to_face=True,
            simplex_use_recycled_geometry=False,
        )
    if variant == "full_msa_to_face_structure_readout":
        return replace(
            base_config,
            use_simplicial_evoformer=True,
            simplex_use_faces=True,
            simplex_use_tetra=True,
            simplex_use_msa_to_face=True,
            simplex_structure_readout_scale=0.25,
        )
    if variant == "full_msa_to_face_structure_readout_only":
        return replace(
            base_config,
            use_simplicial_evoformer=True,
            simplex_use_faces=True,
            simplex_use_tetra=True,
            simplex_use_msa_to_face=True,
            simplex_pair_update_scale=0.0,
            simplex_single_update_scale=0.0,
            simplex_structure_readout_scale=0.5,
        )
    if variant == "full_msa_to_face_outer_edge":
        return replace(
            base_config,
            use_simplicial_evoformer=True,
            simplex_use_faces=True,
            simplex_use_tetra=True,
            simplex_use_msa_to_face=True,
            simplex_outer_edge_update_scale=0.25,
        )
    if variant == "full_msa_to_face_outer_edge_context":
        return replace(
            base_config,
            use_simplicial_evoformer=True,
            simplex_use_faces=True,
            simplex_use_tetra=True,
            simplex_use_msa_to_face=True,
            simplex_outer_edge_context_scale=0.25,
        )
    if variant == "full_msa_to_face_hodge_residual":
        return replace(
            base_config,
            use_simplicial_evoformer=True,
            simplex_use_faces=True,
            simplex_use_tetra=True,
            simplex_use_msa_to_face=True,
            simplex_hodge_face_update_scale=0.25,
        )
    if variant == "full_msa_to_face_flag_closure":
        return replace(
            base_config,
            use_simplicial_evoformer=True,
            simplex_use_faces=True,
            simplex_use_tetra=True,
            simplex_use_msa_to_face=True,
            simplex_boundary_closure_weight=0.5,
            simplex_boundary_closure_temperature=1.0,
        )
    if variant == "full_msa_to_face_flag_closure_soft":
        return replace(
            base_config,
            use_simplicial_evoformer=True,
            simplex_use_faces=True,
            simplex_use_tetra=True,
            simplex_use_msa_to_face=True,
            simplex_boundary_closure_weight=0.1,
            simplex_boundary_closure_temperature=1.0,
        )
    if variant == "full_msa_to_face_edge_frame_messages":
        return replace(
            base_config,
            use_simplicial_evoformer=True,
            simplex_use_faces=True,
            simplex_use_tetra=True,
            simplex_use_msa_to_face=True,
            simplex_edge_frame_message_scale=0.25,
        )
    if variant == "full_msa_to_face_segment_cells":
        return replace(
            base_config,
            use_simplicial_evoformer=True,
            simplex_use_faces=True,
            simplex_use_tetra=True,
            simplex_use_msa_to_face=True,
            simplex_segment_cell_scale=0.25,
            simplex_segment_radius=4,
            simplex_c_segment=12,
        )
    if variant == "face_structure_readout_only":
        return replace(
            base_config,
            use_simplicial_evoformer=True,
            simplex_use_faces=True,
            simplex_use_tetra=False,
            simplex_use_msa_to_face=True,
            simplex_pair_update_scale=0.0,
            simplex_single_update_scale=0.0,
            simplex_structure_readout_scale=0.5,
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


def _build_loss_fn(training_config: TrainingConfig) -> AlphaFoldLoss:
    return AlphaFoldLoss(
        finetune=False,
        use_clamped_fape=training_config.use_clamped_fape,
        simplex_aux_weight=training_config.simplex_aux_weight,
        simplex_face_coordinate_weight=training_config.simplex_face_coordinate_weight,
        simplex_face_coordinate_distance_weight=training_config.simplex_face_coordinate_distance_weight,
        simplex_face_shape_weight=training_config.simplex_face_shape_weight,
        simplex_face_normal_weight=training_config.simplex_face_normal_weight,
        simplex_face_boundary_lddt_weight=training_config.simplex_face_boundary_lddt_weight,
        simplex_tetra_coordinate_weight=training_config.simplex_tetra_coordinate_weight,
        simplex_tetra_coordinate_distance_weight=training_config.simplex_tetra_coordinate_distance_weight,
        simplex_tetra_shape_weight=training_config.simplex_tetra_shape_weight,
        simplex_tetra_boundary_lddt_weight=training_config.simplex_tetra_boundary_lddt_weight,
        simplex_topology_margin_weight=training_config.simplex_topology_margin_weight,
        simplex_topology_margin=training_config.simplex_topology_margin,
        simplex_topology_margin_hard_negatives=training_config.simplex_topology_margin_hard_negatives,
        simplex_boundary_degree_normalize=training_config.simplex_boundary_degree_normalize,
        simplex_cell_closure_weight=training_config.simplex_cell_closure_weight,
        simplex_cell_closure_cutoff=training_config.simplex_cell_closure_cutoff,
        simplex_cell_closure_temperature=training_config.simplex_cell_closure_temperature,
        backbone_loss_weight=training_config.backbone_loss_weight,
        sidechain_fape_loss_weight=training_config.sidechain_fape_loss_weight,
        torsion_loss_weight=training_config.torsion_loss_weight,
    )


def _train_variant(
    *,
    variant: str,
    model_config: Any,
    data_config: DataConfig,
    training_config: TrainingConfig,
    output_dir: Path,
    eval_every: int,
    log_every: int,
    eval_max_val_batches: int | None,
    final_max_val_batches: int | None,
    checkpoint_every: int,
    checkpoint_dir: Path,
    resume_from_checkpoint: Path | None,
    auto_resume: bool,
    stop_after_seconds: int | None,
    foldscore_components_fn: Any | None,
    mixed_precision: str,
) -> dict[str, Any]:
    set_seed(training_config.seed)
    device = resolve_device(training_config.device)
    model = AlphaFold2(model_config).to(device)
    loss_fn = _build_loss_fn(training_config).to(device)
    loss_fn.msa_weight = training_config.msa_loss_weight
    loss_fn.distogram_weight = training_config.distogram_loss_weight
    loss_fn.confidence_weight = training_config.confidence_loss_weight
    target_violation_weight = loss_fn.structural_violation_weight
    finetune_started = False
    finetune_start_step = training_config.finetune_start_step
    optimizer = build_optimizer(model, training_config)
    ema_model = (
        build_ema_model(model, training_config.ema_decay).to(device)
        if training_config.ema_decay is not None
        else None
    )
    train_loader = build_dataloader(
        "train",
        data_config,
        training=True,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
        device=str(device),
        seed=training_config.seed,
        n_cycles=training_config.n_cycles,
        n_ensemble=training_config.n_ensemble,
    )
    val_loader = build_dataloader(
        "val",
        data_config,
        training=False,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
        device=str(device),
        seed=training_config.seed,
        n_cycles=training_config.n_cycles,
        n_ensemble=training_config.n_ensemble,
    )

    latest_checkpoint_path = checkpoint_dir / f"{variant}_latest.pt"
    resume_checkpoint_path = resume_from_checkpoint
    if resume_checkpoint_path is None and auto_resume and latest_checkpoint_path.exists():
        resume_checkpoint_path = latest_checkpoint_path

    history: list[dict[str, Any]] = []
    history_path = output_dir / f"history_{variant}.json"
    train_iter = iter(train_loader)
    train_losses: list[float] = []
    prior_elapsed_seconds = 0.0
    start_step = 1
    total_examples = 0
    last_eval: dict[str, float] | None = None
    grad_accum_steps = max(int(training_config.grad_accum_steps), 1)
    model_profile = str(getattr(model_config, "model_profile", "custom"))

    if resume_checkpoint_path is not None:
        if not resume_checkpoint_path.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_checkpoint_path}")
        checkpoint = _load_training_checkpoint(
            resume_checkpoint_path,
            model=model,
            optimizer=optimizer,
            ema_model=ema_model,
            device=device,
        )
        if checkpoint.get("variant") not in {None, variant}:
            raise ValueError(
                f"Checkpoint variant {checkpoint.get('variant')!r} does not match requested variant {variant!r}"
            )
        start_step = int(checkpoint.get("step", 0)) + 1
        total_examples = int(checkpoint.get("total_examples", checkpoint.get("global_samples", 0)))
        train_losses = [float(value) for value in checkpoint.get("train_losses", [])]
        history = list(checkpoint.get("history", []))
        last_eval = checkpoint.get("last_eval")
        prior_elapsed_seconds = float(checkpoint.get("elapsed_seconds_total", 0.0))
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"[{variant}] resumed from {resume_checkpoint_path} "
            f"at step={start_step - 1} examples={total_examples}"
        )

    start_time = time.perf_counter()
    stopped_early = False
    completed_step = start_step - 1

    def elapsed_total() -> float:
        return prior_elapsed_seconds + (time.perf_counter() - start_time)

    def save_latest_checkpoint(*, stopped: bool) -> None:
        _save_training_checkpoint(
            latest_checkpoint_path,
            model=model,
            optimizer=optimizer,
            ema_model=ema_model,
            variant=variant,
            model_profile=model_profile,
            step=completed_step,
            total_examples=total_examples,
            train_losses=train_losses,
            history=history,
            last_eval=last_eval,
            elapsed_seconds_total=elapsed_total(),
            stopped_early=stopped,
        )
        print(f"[{variant}] checkpoint -> {latest_checkpoint_path}", flush=True)

    for step in range(start_step, training_config.epochs + 1):
        model.train()
        if finetune_start_step is not None and (not finetune_started) and step > finetune_start_step:
            loss_fn.finetune = True
            loss_fn.structural_violation_weight = 0.0
            finetune_started = True
            print(
                f"[{variant}] step={step}: enabling fine-tune losses "
                f"and LR scale {training_config.finetune_lr_scale}"
            )
        if finetune_started and loss_fn.structural_violation_weight < target_violation_weight:
            ramp_steps = max(int(getattr(training_config, "violation_ramp_steps", 0)), 1)
            ramp_progress = (step - int(finetune_start_step or step)) / ramp_steps
            loss_fn.structural_violation_weight = min(target_violation_weight, ramp_progress * target_violation_weight)
        apply_loss_weight_schedule(loss_fn, training_config, step)
        teacher_forcing_weight = simplex_topology_teacher_forcing_weight_at_step(training_config, step)
        simplex_update_scale = simplex_update_scale_at_step(training_config, step)
        simplex_local_neighbor_k = simplex_local_neighbor_k_at_step(training_config, step)
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        term_accum: dict[str, list[float]] = {}
        micro_examples = 0
        terms: dict[str, torch.Tensor] = {}
        for _ in range(grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            batch = move_to_device(batch, device)
            with _autocast_context(device, mixed_precision):
                outputs = model(
                    **model_inputs_from_batch(
                        batch,
                        training_config,
                        use_simplex_teacher_forcing=True,
                        use_simplex_update_scale=True,
                        use_simplex_local_neighbor_k=True,
                        step=step,
                    )
                )
            per_example_loss, terms = _loss_with_terms(loss_fn, batch, outputs)
            micro_loss = per_example_loss.float().mean()
            (micro_loss / grad_accum_steps).backward()
            batch_examples = int(batch["aatype"].shape[0])
            micro_examples += batch_examples
            loss_accum += float(micro_loss.detach().cpu().item()) * batch_examples
            for key, value in terms.items():
                mean_value = _tensor_mean(value)
                if mean_value is not None:
                    term_accum.setdefault(key, []).append(mean_value)

        if training_config.grad_clip_norm is not None:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.grad_clip_norm)
                .detach()
                .float()
                .cpu()
                .item()
            )
        else:
            grad_norm = _global_grad_norm(model.parameters())
        current_lr = learning_rate_at_step(
            training_config,
            step - 1,
            training_config.epochs,
            is_finetune=loss_fn.finetune,
            samples_seen=total_examples + micro_examples,
        )
        set_optimizer_learning_rate(optimizer, current_lr)
        optimizer.step()
        if ema_model is not None:
            ema_model.update_parameters(model)

        loss_value = loss_accum / max(micro_examples, 1)
        train_losses.append(loss_value)
        total_examples += micro_examples
        completed_step = step

        is_final_step = step == training_config.epochs
        should_eval = is_final_step or (eval_every > 0 and step % eval_every == 0)
        should_log = step == 1 or should_eval or (log_every > 0 and step % log_every == 0)
        if should_log:
            row: dict[str, float | int | str] = {
                "variant": variant,
                "step": step,
                "train_loss": loss_value,
                "learning_rate": current_lr,
                "train_examples": total_examples,
                "sampled_n_cycles": int(getattr(model, "last_n_cycles", training_config.n_cycles)),
                "grad_norm": grad_norm,
                "finetune": int(loss_fn.finetune),
                "structural_violation_weight": float(loss_fn.structural_violation_weight),
                "msa_loss_weight": float(loss_fn.msa_weight),
                "distogram_loss_weight": float(loss_fn.distogram_weight),
                "simplex_aux_weight": float(loss_fn.simplex_aux_weight),
                "simplex_face_coordinate_weight": float(loss_fn.simplex_geometry_loss.face_coordinate_weight),
                "simplex_face_coordinate_distance_weight": float(
                    loss_fn.simplex_geometry_loss.face_coordinate_distance_weight
                ),
                "simplex_face_shape_weight": float(loss_fn.simplex_geometry_loss.face_shape_weight),
                "simplex_face_normal_weight": float(loss_fn.simplex_geometry_loss.face_normal_weight),
                "simplex_face_boundary_lddt_weight": float(
                    loss_fn.simplex_geometry_loss.face_boundary_lddt_weight
                ),
                "simplex_tetra_coordinate_weight": float(loss_fn.simplex_geometry_loss.tetra_coordinate_weight),
                "simplex_tetra_coordinate_distance_weight": float(
                    loss_fn.simplex_geometry_loss.tetra_coordinate_distance_weight
                ),
                "simplex_tetra_shape_weight": float(loss_fn.simplex_geometry_loss.tetra_shape_weight),
                "simplex_tetra_boundary_lddt_weight": float(
                    loss_fn.simplex_geometry_loss.tetra_boundary_lddt_weight
                ),
                "simplex_topology_margin_weight": float(loss_fn.simplex_geometry_loss.topology_margin_weight),
                "simplex_topology_margin": float(loss_fn.simplex_geometry_loss.topology_margin),
                "simplex_topology_margin_hard_negatives": int(
                    loss_fn.simplex_geometry_loss.topology_margin_hard_negatives
                ),
                "simplex_boundary_degree_normalize": int(
                    loss_fn.simplex_geometry_loss.boundary_degree_normalize
                ),
                "simplex_cell_closure_weight": float(loss_fn.simplex_geometry_loss.cell_closure_weight),
                "simplex_cell_closure_cutoff": float(loss_fn.simplex_geometry_loss.cell_closure_cutoff),
                "simplex_cell_closure_temperature": float(
                    loss_fn.simplex_geometry_loss.cell_closure_temperature
                ),
                "simplex_topology_teacher_forcing_weight": teacher_forcing_weight,
                "simplex_update_scale": float("nan") if simplex_update_scale is None else simplex_update_scale,
                "simplex_local_neighbor_k": (
                    float("nan") if simplex_local_neighbor_k is None else simplex_local_neighbor_k
                ),
                "backbone_loss_weight": float(loss_fn.backbone_loss_weight),
                "sidechain_fape_loss_weight": float(loss_fn.sidechain_fape_loss_weight),
                "torsion_loss_weight": float(loss_fn.torsion_loss_weight),
            }
            for key, values in term_accum.items():
                row[f"train_{key}"] = _mean(values)
            if should_eval:
                val_batch_limit = final_max_val_batches if is_final_step else eval_max_val_batches
                last_eval = _evaluate(
                    model,
                    loss_fn,
                    val_loader,
                    training_config,
                    device,
                    max_batches=val_batch_limit,
                    foldscore_components_fn=foldscore_components_fn,
                    mixed_precision=mixed_precision,
                    detail_path=output_dir / f"eval_details_{variant}.csv"
                    if is_final_step
                    else None,
                )
                row.update(last_eval)
                if ema_model is not None and is_final_step:
                    ema_eval = _evaluate(
                        ema_model,
                        loss_fn,
                        val_loader,
                        training_config,
                        device,
                        max_batches=final_max_val_batches,
                        foldscore_components_fn=foldscore_components_fn,
                        mixed_precision=mixed_precision,
                        detail_path=output_dir / f"eval_details_ema_{variant}.csv",
                    )
                    prefixed_ema_eval = _prefix_metrics(ema_eval, "ema_")
                    row.update(prefixed_ema_eval)
                    last_eval.update(prefixed_ema_eval)
            history.append(row)
            history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
            val_parts = []
            for key in (
                "grad_norm",
                "train_backbone_loss",
                "train_weighted_backbone_loss",
                "val_loss",
                "val_ca_rmsd",
                "val_lddt_ca",
                "val_ca_drmsd",
                "val_pred_ca_rg",
                "val_true_ca_rg",
                "val_foldscore",
                "val_FoldScore",
            ):
                if key in row:
                    value = row[key]
                    if isinstance(value, (float, int)) and math.isfinite(float(value)):
                        val_parts.append(f"{key}={float(value):.4f}")
            val_fragment = (" " + " ".join(val_parts)) if val_parts else ""
            print(
                f"[{variant}] step={step}/{training_config.epochs} "
                f"train_loss={loss_value:.4f}{val_fragment}"
            )

        if checkpoint_every > 0 and step % checkpoint_every == 0:
            save_latest_checkpoint(stopped=False)
        if stop_after_seconds is not None and stop_after_seconds > 0:
            if time.perf_counter() - start_time >= stop_after_seconds:
                stopped_early = step < training_config.epochs
                save_latest_checkpoint(stopped=stopped_early)
                print(
                    f"[{variant}] stopping after {time.perf_counter() - start_time:.1f}s "
                    f"at step={step}/{training_config.epochs}; resume from {latest_checkpoint_path}",
                    flush=True,
                )
                break

    elapsed = elapsed_total()
    if completed_step >= training_config.epochs and last_eval is None:
        final_eval = _evaluate(
            model,
            loss_fn,
            val_loader,
            training_config,
            device,
            max_batches=final_max_val_batches,
            foldscore_components_fn=foldscore_components_fn,
            mixed_precision=mixed_precision,
            detail_path=output_dir / f"eval_details_{variant}.csv",
        )
        if ema_model is not None:
            ema_eval = _evaluate(
                ema_model,
                loss_fn,
                val_loader,
                training_config,
                device,
                max_batches=final_max_val_batches,
                foldscore_components_fn=foldscore_components_fn,
                mixed_precision=mixed_precision,
                detail_path=output_dir / f"eval_details_ema_{variant}.csv",
            )
            final_eval.update(_prefix_metrics(ema_eval, "ema_"))
    elif last_eval is None:
        final_eval = {}
    else:
        final_eval = last_eval
    save_latest_checkpoint(stopped=stopped_early)
    parameter_count = sum(p.numel() for p in model.parameters())
    use_simplicial = bool(getattr(model_config, "use_simplicial_evoformer", False))
    result: dict[str, Any] = {
        "variant": variant,
        "model_profile": getattr(model_config, "model_profile", "custom"),
        "parameters": parameter_count,
        "steps": training_config.epochs,
        "completed_steps": completed_step,
        "stopped_early": bool(stopped_early),
        "train_examples": total_examples,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch_size": training_config.batch_size * grad_accum_steps,
        "learning_rate": training_config.learning_rate,
        "warmup_samples": training_config.warmup_samples,
        "lr_decay_samples": training_config.lr_decay_samples,
        "lr_decay_factor": training_config.lr_decay_factor,
        "grad_clip_norm": training_config.grad_clip_norm,
        "ema_decay": training_config.ema_decay,
        "use_clamped_fape": training_config.use_clamped_fape,
        "msa_loss_weight": training_config.msa_loss_weight,
        "distogram_loss_weight": training_config.distogram_loss_weight,
        "confidence_loss_weight": training_config.confidence_loss_weight,
        "simplex_aux_weight": training_config.simplex_aux_weight,
        "simplex_face_coordinate_weight": training_config.simplex_face_coordinate_weight,
        "simplex_face_coordinate_distance_weight": training_config.simplex_face_coordinate_distance_weight,
        "simplex_face_shape_weight": training_config.simplex_face_shape_weight,
        "simplex_face_boundary_lddt_weight": training_config.simplex_face_boundary_lddt_weight,
        "simplex_tetra_coordinate_weight": training_config.simplex_tetra_coordinate_weight,
        "simplex_tetra_coordinate_distance_weight": training_config.simplex_tetra_coordinate_distance_weight,
        "simplex_tetra_shape_weight": training_config.simplex_tetra_shape_weight,
        "simplex_tetra_boundary_lddt_weight": training_config.simplex_tetra_boundary_lddt_weight,
        "simplex_boundary_degree_normalize": training_config.simplex_boundary_degree_normalize,
        "simplex_cell_closure_weight": training_config.simplex_cell_closure_weight,
        "simplex_cell_closure_weight_final": training_config.simplex_cell_closure_weight_final,
        "simplex_cell_closure_ramp_start_step": training_config.simplex_cell_closure_ramp_start_step,
        "simplex_cell_closure_ramp_steps": training_config.simplex_cell_closure_ramp_steps,
        "simplex_cell_closure_cutoff": training_config.simplex_cell_closure_cutoff,
        "simplex_cell_closure_temperature": training_config.simplex_cell_closure_temperature,
        "simplex_topology_teacher_forcing_weight": training_config.simplex_topology_teacher_forcing_weight,
        "simplex_topology_teacher_forcing_weight_final": (
            training_config.simplex_topology_teacher_forcing_weight_final
        ),
        "simplex_topology_teacher_forcing_ramp_start_step": (
            training_config.simplex_topology_teacher_forcing_ramp_start_step
        ),
        "simplex_topology_teacher_forcing_ramp_steps": (
            training_config.simplex_topology_teacher_forcing_ramp_steps
        ),
        "simplex_update_scale": training_config.simplex_update_scale,
        "simplex_update_scale_final": training_config.simplex_update_scale_final,
        "simplex_update_scale_ramp_start_step": training_config.simplex_update_scale_ramp_start_step,
        "simplex_update_scale_ramp_steps": training_config.simplex_update_scale_ramp_steps,
        "simplex_local_neighbor_k": training_config.simplex_local_neighbor_k,
        "simplex_local_neighbor_k_final": training_config.simplex_local_neighbor_k_final,
        "simplex_local_neighbor_k_ramp_start_step": training_config.simplex_local_neighbor_k_ramp_start_step,
        "simplex_local_neighbor_k_ramp_steps": training_config.simplex_local_neighbor_k_ramp_steps,
        "backbone_loss_weight": training_config.backbone_loss_weight,
        "sidechain_fape_loss_weight": training_config.sidechain_fape_loss_weight,
        "torsion_loss_weight": training_config.torsion_loss_weight,
        "loss_weight_ramp_start_step": training_config.loss_weight_ramp_start_step,
        "loss_weight_ramp_steps": training_config.loss_weight_ramp_steps,
        "msa_loss_weight_final": training_config.msa_loss_weight_final,
        "distogram_loss_weight_final": training_config.distogram_loss_weight_final,
        "confidence_loss_weight_final": training_config.confidence_loss_weight_final,
        "simplex_aux_weight_final": training_config.simplex_aux_weight_final,
        "backbone_loss_weight_final": training_config.backbone_loss_weight_final,
        "sidechain_fape_loss_weight_final": training_config.sidechain_fape_loss_weight_final,
        "torsion_loss_weight_final": training_config.torsion_loss_weight_final,
        "finetune_start_step": training_config.finetune_start_step,
        "finetune_lr_scale": training_config.finetune_lr_scale,
        "structural_violation_weight": float(loss_fn.structural_violation_weight),
        "elapsed_seconds": elapsed,
        "examples_per_second": total_examples / elapsed if elapsed > 0 else float("nan"),
        "train_loss_final": train_losses[-1] if train_losses else float("nan"),
        "train_loss_mean": _mean(train_losses),
        "use_simplicial_evoformer": use_simplicial,
        "simplex_use_faces": use_simplicial and bool(getattr(model_config, "simplex_use_faces", False)),
        "simplex_use_tetra": use_simplicial and bool(getattr(model_config, "simplex_use_tetra", False)),
        "simplex_use_msa_to_face": use_simplicial and bool(getattr(model_config, "simplex_use_msa_to_face", False)),
        "simplex_neighbor_k": int(getattr(model_config, "simplex_neighbor_k", 0)) if use_simplicial else 0,
        "simplex_local_neighbor_k": int(getattr(model_config, "simplex_local_neighbor_k", 0)) if use_simplicial else 0,
        "simplex_local_radius": int(getattr(model_config, "simplex_local_radius", 0)) if use_simplicial else 0,
        "simplex_local_bias": float(getattr(model_config, "simplex_local_bias", 0.0)) if use_simplicial else 0.0,
        "simplex_long_min_sep": int(getattr(model_config, "simplex_long_min_sep", 0)) if use_simplicial else 0,
        "simplex_long_bias": float(getattr(model_config, "simplex_long_bias", 0.0)) if use_simplicial else 0.0,
        "simplex_boundary_closure_weight": (
            float(getattr(model_config, "simplex_boundary_closure_weight", 0.0)) if use_simplicial else 0.0
        ),
        "simplex_boundary_closure_temperature": (
            float(getattr(model_config, "simplex_boundary_closure_temperature", 1.0)) if use_simplicial else 0.0
        ),
        "simplex_pair_update_scale": (
            float(getattr(model_config, "simplex_pair_update_scale", 1.0)) if use_simplicial else 0.0
        ),
        "simplex_single_update_scale": (
            float(getattr(model_config, "simplex_single_update_scale", 1.0)) if use_simplicial else 0.0
        ),
        "simplex_structure_readout_scale": (
            float(getattr(model_config, "simplex_structure_readout_scale", 0.0)) if use_simplicial else 0.0
        ),
        "simplex_outer_edge_update_scale": (
            float(getattr(model_config, "simplex_outer_edge_update_scale", 0.0)) if use_simplicial else 0.0
        ),
        "simplex_outer_edge_context_scale": (
            float(getattr(model_config, "simplex_outer_edge_context_scale", 0.0)) if use_simplicial else 0.0
        ),
        "simplex_hodge_face_update_scale": (
            float(getattr(model_config, "simplex_hodge_face_update_scale", 0.0)) if use_simplicial else 0.0
        ),
        "simplex_edge_frame_message_scale": (
            float(getattr(model_config, "simplex_edge_frame_message_scale", 0.0)) if use_simplicial else 0.0
        ),
        "simplex_segment_cell_scale": (
            float(getattr(model_config, "simplex_segment_cell_scale", 0.0)) if use_simplicial else 0.0
        ),
        "simplex_segment_radius": int(getattr(model_config, "simplex_segment_radius", 0)) if use_simplicial else 0,
        "simplex_c_segment": int(getattr(model_config, "simplex_c_segment", 0)) if use_simplicial else 0,
        "latest_checkpoint": str(latest_checkpoint_path),
        "resume_from_checkpoint": str(resume_checkpoint_path) if resume_checkpoint_path is not None else "",
        "eval_every": eval_every,
        "eval_max_val_batches": eval_max_val_batches or 0,
        "final_max_val_batches": final_max_val_batches or 0,
        "checkpoint_every": checkpoint_every,
        **final_eval,
    }
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    preferred = [
        "variant",
        "parameters",
        "steps",
        "completed_steps",
        "stopped_early",
        "train_examples",
        "grad_accum_steps",
        "effective_batch_size",
        "learning_rate",
        "warmup_samples",
        "grad_clip_norm",
        "ema_decay",
        "use_clamped_fape",
        "simplex_aux_weight",
        "simplex_face_coordinate_weight",
        "simplex_face_coordinate_distance_weight",
        "simplex_face_shape_weight",
        "simplex_face_normal_weight",
        "simplex_face_boundary_lddt_weight",
        "simplex_tetra_coordinate_weight",
        "simplex_tetra_coordinate_distance_weight",
        "simplex_tetra_shape_weight",
        "simplex_tetra_boundary_lddt_weight",
        "simplex_topology_margin_weight",
        "simplex_topology_margin",
        "simplex_topology_margin_hard_negatives",
        "simplex_boundary_degree_normalize",
        "simplex_cell_closure_weight",
        "simplex_cell_closure_weight_final",
        "simplex_cell_closure_ramp_start_step",
        "simplex_cell_closure_ramp_steps",
        "simplex_cell_closure_cutoff",
        "simplex_cell_closure_temperature",
        "simplex_topology_teacher_forcing_weight",
        "simplex_topology_teacher_forcing_weight_final",
        "simplex_topology_teacher_forcing_ramp_start_step",
        "simplex_topology_teacher_forcing_ramp_steps",
        "simplex_update_scale",
        "simplex_update_scale_final",
        "simplex_update_scale_ramp_start_step",
        "simplex_update_scale_ramp_steps",
        "simplex_local_neighbor_k",
        "simplex_local_neighbor_k_final",
        "simplex_local_neighbor_k_ramp_start_step",
        "simplex_local_neighbor_k_ramp_steps",
        "elapsed_seconds",
        "examples_per_second",
        "train_loss_final",
        "train_loss_mean",
        "eval_every",
        "eval_max_val_batches",
        "final_max_val_batches",
        "checkpoint_every",
        "val_examples",
        "val_loss",
        "val_foldscore",
        "val_FoldScore",
        "val_ca_rmsd",
        "val_lddt_ca",
        "val_ca_drmsd",
        "val_pred_ca_rg",
        "val_true_ca_rg",
        "val_gdt_ha_ca",
        "val_gdt_ts_ca",
        "val_lddt_atom14",
        "val_bb_atom14",
        "val_sc_atom14",
        "val_molprobity_clash_atom14",
        "val_simplex_aux_loss",
        "val_simplex_face_coordinate_area_loss",
        "val_simplex_face_coordinate_distance_loss",
        "val_simplex_face_shape_loss",
        "val_simplex_face_boundary_lddt_loss",
        "val_simplex_face_distance_loss",
        "val_simplex_tetra_coordinate_geometry_loss",
        "val_simplex_tetra_coordinate_distance_loss",
        "val_simplex_tetra_shape_loss",
        "val_simplex_tetra_boundary_lddt_loss",
        "val_simplex_tetra_distance_loss",
        "val_simplex_pair_face_consistency_loss",
        "val_simplex_face_tetra_consistency_loss",
        "use_simplicial_evoformer",
        "simplex_use_faces",
        "simplex_use_tetra",
        "simplex_use_msa_to_face",
        "simplex_neighbor_k",
        "simplex_local_neighbor_k",
        "simplex_local_radius",
        "simplex_local_bias",
        "simplex_long_min_sep",
        "simplex_long_bias",
        "simplex_boundary_closure_weight",
        "simplex_boundary_closure_temperature",
        "simplex_pair_update_scale",
        "simplex_single_update_scale",
        "simplex_structure_readout_scale",
        "simplex_outer_edge_update_scale",
        "simplex_outer_edge_context_scale",
        "simplex_hodge_face_update_scale",
        "simplex_edge_frame_message_scale",
        "simplex_segment_cell_scale",
        "simplex_segment_radius",
        "simplex_c_segment",
    ]
    extra = sorted({key for row in rows for key in row} - set(preferred))
    fieldnames = [key for key in preferred if any(key in row for row in rows)] + extra
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SimplexFold ablations on NanoFold public train/val data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nanofold-root", type=Path, default=Path("/Users/christopherhayduk/Projects/nanoFold-Competition"))
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
        choices=[
            "no_simplex",
            "faces",
            "full",
            "full_msa_to_face",
            "full_msa_to_face_aux_closure",
            "full_msa_to_face_topology_curriculum",
            "full_msa_to_face_expanded_complex",
            "full_msa_to_face_long",
            "full_msa_to_face_mixed",
            "full_msa_to_face_mixed_soft",
            "full_msa_to_face_strong_messages",
            "full_msa_to_face_damped_messages",
            "full_msa_to_face_edge_messages",
            "full_msa_to_face_no_recycled_topology",
            "full_msa_to_face_structure_readout",
            "full_msa_to_face_structure_readout_only",
            "full_msa_to_face_outer_edge",
            "full_msa_to_face_outer_edge_context",
            "full_msa_to_face_hodge_residual",
            "full_msa_to_face_flag_closure",
            "full_msa_to_face_flag_closure_soft",
            "full_msa_to_face_edge_frame_messages",
            "full_msa_to_face_segment_cells",
            "face_structure_readout_only",
            "msa_to_face",
        ],
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "nanofold_public_benchmarks")
    parser.add_argument(
        "--run-name",
        help="Use a stable output subdirectory instead of a timestamp. Useful with --auto-resume.",
    )
    parser.add_argument("--train-limit", type=int, default=0, help="0 means use the full official train manifest.")
    parser.add_argument("--val-limit", type=int, default=0, help="0 means use the full official validation manifest.")
    parser.add_argument("--train-chain-ids", nargs="+", help="Explicit train chain IDs, comma or space separated.")
    parser.add_argument("--val-chain-ids", nargs="+", help="Explicit validation chain IDs, comma or space separated.")
    parser.add_argument(
        "--overfit-chain-id",
        help="Use the same chain ID for train and validation manifests, bypassing official split membership checks.",
    )
    parser.add_argument("--steps", type=int, default=50, help="Optimizer steps per variant.")
    parser.add_argument("--eval-every", type=int, default=0, help="0 evaluates only at the final step.")
    parser.add_argument("--log-every", type=int, default=0, help="0 logs only the first step and eval steps.")
    parser.add_argument("--max-val-batches", type=int, default=0, help="0 evaluates the whole selected val manifest.")
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
    parser.add_argument("--auto-resume", action="store_true", help="Resume from <checkpoint-dir>/<variant>_latest.pt when present.")
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
    parser.add_argument(
        "--simplex-face-coordinate-weight",
        type=float,
        default=None,
        help="Override the selected-face coordinate-area realization loss weight.",
    )
    parser.add_argument(
        "--simplex-face-coordinate-distance-weight",
        type=float,
        default=None,
        help="Override the selected-face boundary-edge coordinate-distance realization loss weight.",
    )
    parser.add_argument(
        "--simplex-face-shape-weight",
        type=float,
        default=None,
        help="Override the selected-face rigid local shape realization loss weight.",
    )
    parser.add_argument(
        "--simplex-face-normal-weight",
        type=float,
        default=None,
        help="Override the selected-face backbone-frame normal orientation realization loss weight.",
    )
    parser.add_argument(
        "--simplex-face-boundary-lddt-weight",
        type=float,
        default=None,
        help="Override the selected-face boundary-edge lDDT-style realization loss weight.",
    )
    parser.add_argument(
        "--simplex-tetra-coordinate-weight",
        type=float,
        default=None,
        help="Override the selected-tetra coordinate-geometry realization loss weight.",
    )
    parser.add_argument(
        "--simplex-tetra-coordinate-distance-weight",
        type=float,
        default=None,
        help="Override the selected-tetra boundary-edge coordinate-distance realization loss weight.",
    )
    parser.add_argument(
        "--simplex-tetra-shape-weight",
        type=float,
        default=None,
        help="Override the selected-tetra rigid local shape realization loss weight.",
    )
    parser.add_argument(
        "--simplex-tetra-boundary-lddt-weight",
        type=float,
        default=None,
        help="Override the selected-tetra boundary-edge lDDT-style realization loss weight.",
    )
    parser.add_argument(
        "--simplex-topology-margin-weight",
        type=float,
        default=None,
        help="Override the hard-negative margin loss weight for simplex topology logits.",
    )
    parser.add_argument(
        "--simplex-topology-margin",
        type=float,
        default=None,
        help="Override the logit margin between true topology neighbors and hard non-neighbors.",
    )
    parser.add_argument(
        "--simplex-topology-margin-hard-negatives",
        type=int,
        default=None,
        help="Override the number of hard non-contact neighbors used by the topology margin.",
    )
    parser.add_argument(
        "--simplex-boundary-degree-normalize",
        action="store_true",
        help="Normalize selected simplex boundary-edge losses by undirected edge incidence degree.",
    )
    parser.add_argument(
        "--simplex-cell-closure-weight",
        type=float,
        default=0.0,
        help="Weight selected-cell coordinate realization by true boundary flag-complex closure.",
    )
    parser.add_argument("--simplex-cell-closure-weight-final", type=float, default=None)
    parser.add_argument("--simplex-cell-closure-ramp-start-step", type=int, default=None)
    parser.add_argument("--simplex-cell-closure-ramp-steps", type=int, default=1)
    parser.add_argument("--simplex-cell-closure-cutoff", type=float, default=15.0)
    parser.add_argument("--simplex-cell-closure-temperature", type=float, default=2.0)
    parser.add_argument(
        "--simplex-topology-teacher-forcing-weight",
        type=float,
        default=0.0,
        help="Training-only weight for selecting simplex cells from true C-alpha distances.",
    )
    parser.add_argument("--simplex-topology-teacher-forcing-weight-final", type=float, default=None)
    parser.add_argument("--simplex-topology-teacher-forcing-ramp-start-step", type=int, default=None)
    parser.add_argument("--simplex-topology-teacher-forcing-ramp-steps", type=int, default=1)
    parser.add_argument(
        "--simplex-update-scale",
        type=float,
        default=None,
        help="Training-only scale for simplex residual messages into pair/single states.",
    )
    parser.add_argument("--simplex-update-scale-final", type=float, default=None)
    parser.add_argument("--simplex-update-scale-ramp-start-step", type=int, default=None)
    parser.add_argument("--simplex-update-scale-ramp-steps", type=int, default=1)
    parser.add_argument(
        "--simplex-local-neighbor-k",
        type=float,
        default=None,
        help="Training-only override for reserved local simplex neighbor slots.",
    )
    parser.add_argument("--simplex-local-neighbor-k-final", type=float, default=None)
    parser.add_argument("--simplex-local-neighbor-k-ramp-start-step", type=int, default=None)
    parser.add_argument("--simplex-local-neighbor-k-ramp-steps", type=int, default=1)
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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n-cycles",
        type=int,
        default=None,
        help="Recycling cycles. Defaults to the model profile's recommended_n_cycles.",
    )
    parser.add_argument("--n-ensemble", type=int, default=1)
    parser.add_argument("--mixed-precision", choices=["off", "bf16", "fp16"], default="off")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> list[dict[str, Any]]:
    args = parse_args(argv)
    nanofold_root = args.nanofold_root.resolve()
    features_dir = nanofold_root / "data" / "processed_features"
    labels_dir = nanofold_root / "data" / "processed_labels"
    train_manifest = nanofold_root / "data" / "manifests" / "train.txt"
    val_manifest = nanofold_root / "data" / "manifests" / "val.txt"
    for path in (features_dir, labels_dir, train_manifest, val_manifest):
        if not path.exists():
            raise FileNotFoundError(f"Required NanoFold public data path is missing: {path}")

    output_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_dir / output_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    subset_dir = output_dir / "manifests"
    overfit_ids = [args.overfit_chain_id] if args.overfit_chain_id else None
    train_manifest_used = _write_manifest_subset(
        train_manifest,
        args.train_limit if args.train_limit > 0 else None,
        subset_dir / "train.txt",
        selected_chain_ids=overfit_ids or _parse_chain_id_args(args.train_chain_ids),
        validate_selected=overfit_ids is None,
    )
    val_manifest_used = _write_manifest_subset(
        val_manifest,
        args.val_limit if args.val_limit > 0 else None,
        subset_dir / "val.txt",
        selected_chain_ids=overfit_ids or _parse_chain_id_args(args.val_chain_ids),
        validate_selected=overfit_ids is None,
    )

    base_config = load_model_config(args.model_config)
    if args.zero_dropout:
        base_config = zero_dropout_model_config(base_config)
    n_cycles = args.n_cycles
    if n_cycles is None:
        n_cycles = int(getattr(base_config, "recommended_n_cycles", 1))
    foldscore_components_fn = _load_foldscore_components(nanofold_root)
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
        fixed_feature_seed=args.seed,
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
        simplex_face_coordinate_weight=args.simplex_face_coordinate_weight,
        simplex_face_coordinate_distance_weight=args.simplex_face_coordinate_distance_weight,
        simplex_face_shape_weight=args.simplex_face_shape_weight,
        simplex_face_normal_weight=args.simplex_face_normal_weight,
        simplex_face_boundary_lddt_weight=args.simplex_face_boundary_lddt_weight,
        simplex_tetra_coordinate_weight=args.simplex_tetra_coordinate_weight,
        simplex_tetra_coordinate_distance_weight=args.simplex_tetra_coordinate_distance_weight,
        simplex_tetra_shape_weight=args.simplex_tetra_shape_weight,
        simplex_tetra_boundary_lddt_weight=args.simplex_tetra_boundary_lddt_weight,
        simplex_topology_margin_weight=args.simplex_topology_margin_weight,
        simplex_topology_margin=args.simplex_topology_margin,
        simplex_topology_margin_hard_negatives=args.simplex_topology_margin_hard_negatives,
        simplex_boundary_degree_normalize=args.simplex_boundary_degree_normalize,
        simplex_cell_closure_weight=args.simplex_cell_closure_weight,
        simplex_cell_closure_weight_final=args.simplex_cell_closure_weight_final,
        simplex_cell_closure_ramp_start_step=args.simplex_cell_closure_ramp_start_step,
        simplex_cell_closure_ramp_steps=args.simplex_cell_closure_ramp_steps,
        simplex_cell_closure_cutoff=args.simplex_cell_closure_cutoff,
        simplex_cell_closure_temperature=args.simplex_cell_closure_temperature,
        simplex_topology_teacher_forcing_weight=args.simplex_topology_teacher_forcing_weight,
        simplex_topology_teacher_forcing_weight_final=args.simplex_topology_teacher_forcing_weight_final,
        simplex_topology_teacher_forcing_ramp_start_step=args.simplex_topology_teacher_forcing_ramp_start_step,
        simplex_topology_teacher_forcing_ramp_steps=args.simplex_topology_teacher_forcing_ramp_steps,
        simplex_update_scale=args.simplex_update_scale,
        simplex_update_scale_final=args.simplex_update_scale_final,
        simplex_update_scale_ramp_start_step=args.simplex_update_scale_ramp_start_step,
        simplex_update_scale_ramp_steps=args.simplex_update_scale_ramp_steps,
        simplex_local_neighbor_k=args.simplex_local_neighbor_k,
        simplex_local_neighbor_k_final=args.simplex_local_neighbor_k_final,
        simplex_local_neighbor_k_ramp_start_step=args.simplex_local_neighbor_k_ramp_start_step,
        simplex_local_neighbor_k_ramp_steps=args.simplex_local_neighbor_k_ramp_steps,
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
    max_val_batches = args.max_val_batches if args.max_val_batches > 0 else None
    eval_max_val_batches = (
        args.eval_max_val_batches
        if args.eval_max_val_batches is not None and args.eval_max_val_batches > 0
        else max_val_batches
    )
    final_max_val_batches = (
        args.final_max_val_batches
        if args.final_max_val_batches is not None and args.final_max_val_batches > 0
        else max_val_batches
    )
    checkpoint_dir = (args.checkpoint_dir or (output_dir / "checkpoints")).resolve()

    metadata = {
        "nanofold_root": str(nanofold_root),
        "features_dir": str(features_dir),
        "labels_dir": str(labels_dir),
        "train_manifest": str(train_manifest_used),
        "val_manifest": str(val_manifest_used),
        "train_manifest_size": len(read_chain_id_manifest(train_manifest_used)),
        "val_manifest_size": len(read_chain_id_manifest(val_manifest_used)),
        "overfit_chain_id": args.overfit_chain_id,
        "model_config": args.model_config,
        "zero_dropout": bool(args.zero_dropout),
        "variants": args.variants,
        "crop_size": args.crop_size,
        "msa_depth": args.msa_depth,
        "extra_msa_depth": args.extra_msa_depth,
        "max_templates": args.max_templates,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_size": args.batch_size * max(args.grad_accum_steps, 1),
        "learning_rate": args.learning_rate,
        "warmup_samples": args.warmup_samples,
        "lr_decay_samples": args.lr_decay_samples,
        "lr_decay_factor": args.lr_decay_factor,
        "grad_clip_norm": args.grad_clip_norm,
        "ema_decay": args.ema_decay,
        "use_clamped_fape": None if args.use_clamped_fape < 0 else args.use_clamped_fape,
        "msa_loss_weight": args.msa_loss_weight,
        "distogram_loss_weight": args.distogram_loss_weight,
        "confidence_loss_weight": args.confidence_loss_weight,
        "simplex_aux_weight": args.simplex_aux_weight,
        "simplex_face_coordinate_weight": args.simplex_face_coordinate_weight,
        "simplex_face_coordinate_distance_weight": args.simplex_face_coordinate_distance_weight,
        "simplex_face_shape_weight": args.simplex_face_shape_weight,
        "simplex_face_normal_weight": args.simplex_face_normal_weight,
        "simplex_face_boundary_lddt_weight": args.simplex_face_boundary_lddt_weight,
        "simplex_tetra_coordinate_weight": args.simplex_tetra_coordinate_weight,
        "simplex_tetra_coordinate_distance_weight": args.simplex_tetra_coordinate_distance_weight,
        "simplex_tetra_shape_weight": args.simplex_tetra_shape_weight,
        "simplex_tetra_boundary_lddt_weight": args.simplex_tetra_boundary_lddt_weight,
        "simplex_topology_margin_weight": args.simplex_topology_margin_weight,
        "simplex_topology_margin": args.simplex_topology_margin,
        "simplex_topology_margin_hard_negatives": args.simplex_topology_margin_hard_negatives,
        "simplex_boundary_degree_normalize": args.simplex_boundary_degree_normalize,
        "simplex_cell_closure_weight": args.simplex_cell_closure_weight,
        "simplex_cell_closure_weight_final": args.simplex_cell_closure_weight_final,
        "simplex_cell_closure_ramp_start_step": args.simplex_cell_closure_ramp_start_step,
        "simplex_cell_closure_ramp_steps": args.simplex_cell_closure_ramp_steps,
        "simplex_cell_closure_cutoff": args.simplex_cell_closure_cutoff,
        "simplex_cell_closure_temperature": args.simplex_cell_closure_temperature,
        "simplex_topology_teacher_forcing_weight": args.simplex_topology_teacher_forcing_weight,
        "simplex_topology_teacher_forcing_weight_final": args.simplex_topology_teacher_forcing_weight_final,
        "simplex_topology_teacher_forcing_ramp_start_step": args.simplex_topology_teacher_forcing_ramp_start_step,
        "simplex_topology_teacher_forcing_ramp_steps": args.simplex_topology_teacher_forcing_ramp_steps,
        "simplex_update_scale": args.simplex_update_scale,
        "simplex_update_scale_final": args.simplex_update_scale_final,
        "simplex_update_scale_ramp_start_step": args.simplex_update_scale_ramp_start_step,
        "simplex_update_scale_ramp_steps": args.simplex_update_scale_ramp_steps,
        "simplex_local_neighbor_k": args.simplex_local_neighbor_k,
        "simplex_local_neighbor_k_final": args.simplex_local_neighbor_k_final,
        "simplex_local_neighbor_k_ramp_start_step": args.simplex_local_neighbor_k_ramp_start_step,
        "simplex_local_neighbor_k_ramp_steps": args.simplex_local_neighbor_k_ramp_steps,
        "backbone_loss_weight": args.backbone_loss_weight,
        "sidechain_fape_loss_weight": args.sidechain_fape_loss_weight,
        "torsion_loss_weight": args.torsion_loss_weight,
        "loss_weight_ramp_start_step": args.loss_weight_ramp_start_step,
        "loss_weight_ramp_steps": args.loss_weight_ramp_steps,
        "msa_loss_weight_final": args.msa_loss_weight_final,
        "distogram_loss_weight_final": args.distogram_loss_weight_final,
        "confidence_loss_weight_final": args.confidence_loss_weight_final,
        "simplex_aux_weight_final": args.simplex_aux_weight_final,
        "backbone_loss_weight_final": args.backbone_loss_weight_final,
        "sidechain_fape_loss_weight_final": args.sidechain_fape_loss_weight_final,
        "torsion_loss_weight_final": args.torsion_loss_weight_final,
        "finetune_start_step": args.finetune_start_step,
        "finetune_lr_scale": args.finetune_lr_scale,
        "violation_ramp_steps": args.violation_ramp_steps,
        "device": args.device,
        "seed": args.seed,
        "n_cycles": n_cycles,
        "n_ensemble": args.n_ensemble,
        "mixed_precision": args.mixed_precision,
        "log_every": args.log_every,
        "eval_every": args.eval_every,
        "max_val_batches": args.max_val_batches,
        "eval_max_val_batches": eval_max_val_batches or 0,
        "final_max_val_batches": final_max_val_batches or 0,
        "checkpoint_every": args.checkpoint_every,
        "checkpoint_dir": str(checkpoint_dir),
        "resume_from_checkpoint": str(args.resume_from_checkpoint) if args.resume_from_checkpoint else "",
        "auto_resume": bool(args.auto_resume),
        "stop_after_seconds": args.stop_after_seconds,
        "run_name": args.run_name or "",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[nanofold-public] artifacts -> {output_dir}")
    print(
        f"[nanofold-public] train={metadata['train_manifest_size']} "
        f"val={metadata['val_manifest_size']} crop={args.crop_size} msa={args.msa_depth}"
    )
    if args.resume_from_checkpoint is not None and len(args.variants) != 1:
        raise ValueError("--resume-from-checkpoint is only supported when running one variant.")

    rows: list[dict[str, Any]] = []
    for variant in args.variants:
        config = _variant_config(base_config, variant)
        rows.append(
            _train_variant(
                variant=variant,
                model_config=config,
                data_config=data_config,
                training_config=training_config,
                output_dir=output_dir,
                eval_every=args.eval_every,
                log_every=args.log_every,
                eval_max_val_batches=eval_max_val_batches,
                final_max_val_batches=final_max_val_batches,
                checkpoint_every=max(args.checkpoint_every, 0),
                checkpoint_dir=checkpoint_dir,
                resume_from_checkpoint=args.resume_from_checkpoint.resolve()
                if args.resume_from_checkpoint is not None
                else None,
                auto_resume=bool(args.auto_resume),
                stop_after_seconds=args.stop_after_seconds if args.stop_after_seconds > 0 else None,
                foldscore_components_fn=foldscore_components_fn,
                mixed_precision=args.mixed_precision,
            )
        )
        (output_dir / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        _write_csv(output_dir / "results.csv", rows)

    print(f"[nanofold-public] results -> {output_dir / 'results.csv'}")
    return rows


if __name__ == "__main__":
    main()
