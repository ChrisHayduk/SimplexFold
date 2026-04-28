#!/usr/bin/env python3
"""Benchmark SimplexFold model variants on synthetic crops.

This script is intentionally data-free: it measures architecture overhead,
parameter count, peak CUDA memory, and forward latency for the same crop/MSA
shapes used by the training and NanoFold-style evaluation scripts.  Use it as
the first reproducible number in a benchmark suite before running expensive
accuracy experiments on held-out structures.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minalphafold.model import AlphaFold2
from minalphafold.trainer import load_model_config, resolve_device, set_seed


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_memory_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)


def _random_inputs(
    *,
    batch_size: int,
    length: int,
    msa_depth: int,
    extra_msa_depth: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    residue_index = torch.arange(length, device=device).unsqueeze(0).expand(batch_size, -1)
    return {
        "target_feat": torch.randn(batch_size, length, 22, device=device),
        "residue_index": residue_index,
        "msa_feat": torch.randn(batch_size, msa_depth, length, 49, device=device),
        "extra_msa_feat": torch.randn(batch_size, extra_msa_depth, length, 25, device=device),
        "template_pair_feat": torch.empty(batch_size, 0, length, length, 88, device=device),
        "aatype": torch.randint(0, 20, (batch_size, length), device=device),
        "template_angle_feat": torch.empty(batch_size, 0, length, 57, device=device),
        "template_mask": torch.empty(batch_size, 0, device=device),
        "seq_mask": torch.ones(batch_size, length, device=device),
        "msa_mask": torch.ones(batch_size, msa_depth, length, device=device),
        "extra_msa_mask": torch.ones(batch_size, extra_msa_depth, length, device=device),
    }


def _simplex_counts(length: int, neighbor_k: int) -> dict[str, int]:
    k = min(neighbor_k, max(length - 1, 0))
    faces = length * k * max(k - 1, 0) // 2
    tetras = length * k * max(k - 1, 0) * max(k - 2, 0) // 6
    return {"neighbor_k_effective": k, "faces_per_example": faces, "tetras_per_example": tetras}


def benchmark_variant(
    *,
    name: str,
    model_config: Any,
    inputs: dict[str, torch.Tensor],
    device: torch.device,
    n_cycles: int,
    warmup_steps: int,
    timed_steps: int,
) -> dict[str, Any]:
    model = AlphaFold2(model_config).to(device)
    model.eval()
    param_count = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    with torch.no_grad():
        for _ in range(warmup_steps):
            model(**inputs, n_cycles=n_cycles, n_ensemble=1)
        _sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        times_ms: list[float] = []
        for _ in range(timed_steps):
            start = time.perf_counter()
            model(**inputs, n_cycles=n_cycles, n_ensemble=1)
            _sync(device)
            times_ms.append((time.perf_counter() - start) * 1000.0)

    use_simplicial = bool(getattr(model_config, "use_simplicial_evoformer", False))
    use_tetra = use_simplicial and bool(getattr(model_config, "simplex_use_tetra", False))
    use_msa_to_face = use_simplicial and bool(getattr(model_config, "simplex_use_msa_to_face", False))
    if use_simplicial:
        counts = _simplex_counts(
            inputs["target_feat"].shape[1],
            int(getattr(model_config, "simplex_neighbor_k", 0)),
        )
        if not use_tetra:
            counts["tetras_per_example"] = 0
    else:
        counts = {"neighbor_k_effective": 0, "faces_per_example": 0, "tetras_per_example": 0}
    return {
        "variant": name,
        "profile": getattr(model_config, "model_profile", "custom"),
        "use_simplicial_evoformer": use_simplicial,
        "simplex_use_tetra": use_tetra,
        "simplex_use_msa_to_face": use_msa_to_face,
        "parameters": param_count,
        "trainable_parameters": trainable_param_count,
        "mean_ms": statistics.fmean(times_ms),
        "median_ms": statistics.median(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "peak_memory_mb": _peak_memory_mb(device),
        **counts,
    }


def main(argv: list[str] | None = None) -> list[dict[str, Any]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default="tiny", help="Profile name or path to TOML config.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--msa-depth", type=int, default=32)
    parser.add_argument("--extra-msa-depth", type=int, default=0)
    parser.add_argument("--n-cycles", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--timed-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["simplex", "faces_only", "no_simplex"],
        choices=["simplex", "faces_only", "msa_to_face", "no_simplex"],
        help="Ablation variants to benchmark.",
    )
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = resolve_device(args.device)
    base_config = load_model_config(args.model_config)
    inputs = _random_inputs(
        batch_size=args.batch_size,
        length=args.length,
        msa_depth=args.msa_depth,
        extra_msa_depth=args.extra_msa_depth,
        device=device,
    )

    variant_configs: dict[str, Any] = {
        "simplex": base_config,
        "faces_only": replace(base_config, simplex_use_tetra=False),
        "msa_to_face": replace(base_config, simplex_use_msa_to_face=True),
        "no_simplex": replace(base_config, use_simplicial_evoformer=False),
    }
    results = [
        benchmark_variant(
            name=name,
            model_config=variant_configs[name],
            inputs=inputs,
            device=device,
            n_cycles=args.n_cycles,
            warmup_steps=args.warmup_steps,
            timed_steps=args.timed_steps,
        )
        for name in args.variants
    ]

    header = (
        "variant,params,faces,tetras,mean_ms,median_ms,peak_memory_mb,"
        "use_simplicial,use_tetra,use_msa_to_face"
    )
    print(header)
    for row in results:
        peak = "" if row["peak_memory_mb"] is None else f"{row['peak_memory_mb']:.1f}"
        print(
            f"{row['variant']},{row['parameters']},{row['faces_per_example']},"
            f"{row['tetras_per_example']},{row['mean_ms']:.3f},"
            f"{row['median_ms']:.3f},{peak},{row['use_simplicial_evoformer']},"
            f"{row['simplex_use_tetra']},{row['simplex_use_msa_to_face']}"
        )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    main()
