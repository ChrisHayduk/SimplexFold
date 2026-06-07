#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Generate NanoFold/SimplexFold experiment tracking figures.

The plots are intentionally lightweight and scriptable: they use only the
experiment ledger plus returned benchmark history JSON files, so they can be
refreshed by a Runpod monitor without touching training code.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


TABLE_HEADER_PREFIX = "| Run | Status | Best step | Best `val_lddt_ca` |"
RUN_RE = re.compile(r"\bE(\d+)\b")


@dataclass(frozen=True)
class ExperimentRow:
    table_index: int
    run_num: int | None
    run_label: str
    status: str
    best_step: int | None
    best_val_lddt_ca: float | None
    decision: str
    running_best_val_lddt_ca: float | None
    running_best_label: str | None


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _optional_float(value: str) -> float | None:
    text = value.strip().strip("`")
    if not text or text == "-":
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def _optional_int(value: str) -> int | None:
    number = _optional_float(value)
    if number is None:
        return None
    return int(number)


def parse_experiment_results(path: Path) -> list[ExperimentRow]:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith(TABLE_HEADER_PREFIX))
    except StopIteration as exc:
        raise ValueError(f"Could not find experiment table in {path}") from exc

    rows: list[ExperimentRow] = []
    running_best: float | None = None
    running_best_label: str | None = None
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        cells = _cells(line)
        if len(cells) < 9:
            continue
        run_label = cells[0]
        match = RUN_RE.search(run_label)
        run_num = int(match.group(1)) if match else None
        best_val = _optional_float(cells[3])
        if best_val is not None and (running_best is None or best_val > running_best):
            running_best = best_val
            running_best_label = run_label
        rows.append(
            ExperimentRow(
                table_index=len(rows) + 1,
                run_num=run_num,
                run_label=run_label,
                status=cells[1],
                best_step=_optional_int(cells[2]),
                best_val_lddt_ca=best_val,
                decision=cells[8],
                running_best_val_lddt_ca=running_best,
                running_best_label=running_best_label,
            )
        )
    return rows


def write_experiment_csv(rows: Iterable[ExperimentRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "table_index",
        "run_num",
        "run_label",
        "status",
        "best_step",
        "best_val_lddt_ca",
        "decision",
        "running_best_val_lddt_ca",
        "running_best_label",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "table_index": row.table_index,
                    "run_num": row.run_num if row.run_num is not None else "",
                    "run_label": row.run_label,
                    "status": row.status,
                    "best_step": row.best_step if row.best_step is not None else "",
                    "best_val_lddt_ca": _format_float(row.best_val_lddt_ca),
                    "decision": row.decision,
                    "running_best_val_lddt_ca": _format_float(row.running_best_val_lddt_ca),
                    "running_best_label": row.running_best_label or "",
                }
            )


def _format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def plot_experiment_progress(rows: list[ExperimentRow], path: Path, *, square: bool = False) -> None:
    numeric = [row for row in rows if row.best_val_lddt_ca is not None and row.running_best_val_lddt_ca is not None]
    if not numeric:
        raise ValueError("No numeric experiment rows to plot")
    x_values = [row.table_index for row in numeric]
    best_values = [row.best_val_lddt_ca for row in numeric if row.best_val_lddt_ca is not None]
    running_values = [
        row.running_best_val_lddt_ca for row in numeric if row.running_best_val_lddt_ca is not None
    ]
    best_row = max(numeric, key=lambda row: row.best_val_lddt_ca or float("-inf"))

    fig_size = (9.5, 9.3) if square else (12, 6.6)
    fig, ax = plt.subplots(figsize=fig_size, dpi=200)
    ax.plot(x_values, best_values, color="#8A8F98", linewidth=1.1, alpha=0.55, label="Returned run")
    ax.plot(x_values, running_values, color="#087F8C", linewidth=2.6, label="Running best")
    ax.scatter(
        [best_row.table_index],
        [best_row.best_val_lddt_ca],
        color="#B83B5E",
        s=50,
        zorder=5,
        label=f"Best: {best_row.run_label}",
    )
    ax.axhline(0.45, color="#D99000", linestyle="--", linewidth=1.4, alpha=0.8, label="Short gate 0.45")
    ax.axhline(0.70, color="#335C67", linestyle=":", linewidth=1.4, alpha=0.8, label="Target 0.70")
    ax.set_title("SimplexFold Public Benchmark Progress", fontsize=16, pad=14)
    ax.set_xlabel("Experiment ledger row")
    ax.set_ylabel("Best validation C-alpha lDDT")
    ax.set_ylim(0.0, max(0.75, max(running_values) + 0.08))
    ax.grid(True, color="#D9DEE3", linewidth=0.8, alpha=0.8)
    ax.legend(loc="lower right", frameon=False)
    ax.text(
        0.02,
        0.96,
        f"{best_row.run_label}: {best_row.best_val_lddt_ca:.4f} at step {best_row.best_step}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        color="#243B53",
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


HISTORY_METRICS = [
    ("val_lddt_ca", "Val C-alpha lDDT", "higher"),
    ("val_foldscore", "FoldScore", "higher"),
    ("val_ca_drmsd", "C-alpha dRMSD", "lower"),
    ("rg_ratio", "C-alpha Rg ratio", "target"),
    ("boundary_lddt_mean", "Selected boundary lDDT", "higher"),
    ("boundary_contraction_mean", "Selected contraction fraction", "lower"),
    ("val_loss", "Val loss", "lower"),
    ("train_loss", "Train loss", "lower"),
]


def _history_value(row: dict[str, object], metric: str) -> float | None:
    if metric == "rg_ratio":
        pred = _json_float(row.get("val_pred_ca_rg"))
        true = _json_float(row.get("val_true_ca_rg"))
        if pred is None or true is None or true == 0.0:
            return None
        return pred / true
    if metric == "boundary_lddt_mean":
        return _mean_present(
            [
                _json_float(row.get("val_simplex_face_boundary_lddt")),
                _json_float(row.get("val_simplex_tetra_boundary_lddt")),
            ]
        )
    if metric == "boundary_contraction_mean":
        return _mean_present(
            [
                _json_float(row.get("val_simplex_face_boundary_contraction_fraction")),
                _json_float(row.get("val_simplex_tetra_boundary_contraction_fraction")),
            ]
        )
    return _json_float(row.get(metric))


def _json_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean_present(values: Iterable[float | None]) -> float | None:
    numeric = [value for value in values if value is not None]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def read_history(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"History JSON must contain a list: {path}")
    return [row for row in rows if isinstance(row, dict)]


def write_history_metric_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["step", *(name for name, _label, _direction in HISTORY_METRICS)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            step = _json_float(row.get("step"))
            if step is None:
                continue
            writer.writerow(
                {
                    "step": int(step),
                    **{
                        metric: "" if (value := _history_value(row, metric)) is None else f"{value:.6g}"
                        for metric, _label, _direction in HISTORY_METRICS
                    },
                }
            )


def plot_history_metrics(rows: list[dict[str, object]], path: Path, *, title: str) -> None:
    usable = [row for row in rows if _json_float(row.get("step")) is not None]
    if not usable:
        raise ValueError("No step rows in history")

    fig, axes = plt.subplots(4, 2, figsize=(12, 12.5), dpi=200, sharex=True)
    axes_flat = list(axes.flatten())
    for ax, (metric, label, direction) in zip(axes_flat, HISTORY_METRICS, strict=True):
        xs: list[float] = []
        ys: list[float] = []
        for row in usable:
            step = _json_float(row.get("step"))
            value = _history_value(row, metric)
            if step is not None and value is not None:
                xs.append(step)
                ys.append(value)
        if not xs:
            ax.set_axis_off()
            continue
        color = "#087F8C" if direction == "higher" else "#B83B5E" if direction == "lower" else "#D99000"
        ax.plot(xs, ys, color=color, linewidth=2.0)
        ax.scatter(xs[-1:], ys[-1:], color=color, s=22)
        if metric == "val_lddt_ca":
            ax.axhline(0.45, color="#D99000", linestyle="--", linewidth=1.0, alpha=0.7)
            ax.axhline(0.70, color="#335C67", linestyle=":", linewidth=1.0, alpha=0.7)
        if metric == "rg_ratio":
            ax.axhline(1.0, color="#335C67", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.set_title(label, fontsize=11)
        ax.grid(True, color="#D9DEE3", linewidth=0.7, alpha=0.75)
        ax.tick_params(axis="both", labelsize=9)
    for ax in axes[-1, :]:
        ax.set_xlabel("Optimizer step")
    fig.suptitle(title, fontsize=16, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _run_label_from_path(path: Path) -> str:
    name = path.name
    match = RUN_RE.search(name)
    prefix = match.group(0) if match else name
    return f"{prefix} metric trace"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-md", type=Path, default=Path("EXPERIMENT_RESULTS.md"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/nanofold_public_benchmarks"))
    parser.add_argument("--plot-dir", type=Path, default=Path("artifacts/nanofold_public_benchmarks/plots"))
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run artifact directory name or path whose history JSON should be plotted.",
    )
    args = parser.parse_args()

    rows = parse_experiment_results(args.results_md)
    write_experiment_csv(rows, args.plot_dir / "best_val_lddt_by_experiment_run.csv")
    plot_experiment_progress(rows, args.plot_dir / "best_val_lddt_by_experiment_run.png")
    plot_experiment_progress(rows, args.plot_dir / "best_val_lddt_by_experiment_run_social.png")
    plot_experiment_progress(rows, args.plot_dir / "best_val_lddt_by_experiment_run_social_square.png", square=True)

    for run in args.run:
        run_path = Path(run)
        if not run_path.is_absolute():
            run_path = args.artifact_root / run_path
        history_path = run_path / "history_full_msa_to_face.json"
        if not history_path.exists():
            print(f"[plot] missing history, skipping: {history_path}")
            continue
        history = read_history(history_path)
        stem = run_path.name
        write_history_metric_csv(history, args.plot_dir / f"{stem}_metric_trace.csv")
        plot_history_metrics(
            history,
            args.plot_dir / f"{stem}_metric_trace.png",
            title=f"{_run_label_from_path(run_path)}: validation and topology diagnostics",
        )


if __name__ == "__main__":
    main()
