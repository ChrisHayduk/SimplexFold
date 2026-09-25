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
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nanofold-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


TABLE_HEADER_PREFIX = "| Run | Status | Best step | Best `val_lddt_ca` |"
RUN_RE = re.compile(r"\bE(\d+)\b")
AF2_MEDIUM_FINAL_FOLDSCORE = 0.4570501643


class Theme:
    ink = "#1f2937"
    muted = "#5f6f7a"
    grid = "#d8dee4"
    returned = "#8b949e"
    running_best = "#008c8c"
    highlight = "#b83b5e"
    reference = "#cc9933"


E151_LINEAGE_CONTRIBUTIONS = {
    "main/full control": "baseline AF2-style control",
    "E03 warm boundary": "warm boundary signal",
    "E04 coordinate cells diagnostic": "coordinate-cell realization",
    "E07 boundary coordinate d=0.5 scaled": "selected boundary-coordinate loss",
    "E09 full MSA-to-face d=0.5 scaled": "full MSA-to-face simplex path",
    "E12 E09 continuation to 6000": "continued full MSA-to-face path",
    "E15 simplex aux anneal to 0.5": "simplex auxiliary anneal",
    "E53 longer effective-batch-8 scaffold": "batch-8 scaffold",
    "E54 effective-batch-8 aux anneal": "batch-8 auxiliary anneal",
    "E55 effective-batch-8 aux 0.5 continuation": "batch-8 branch checkpoint",
    "E63 selected-boundary lDDT 0.05": "selected-boundary lDDT objective",
    "E64 E63 confirmation to 4000": "confirmed selected-boundary lDDT",
    "E70 edge-frame messages 0.025": "edge-frame boundary messages",
    "E71 continue edge-frame 0.025": "continued edge-frame path",
    "E72 continue edge-frame 0.025 to 5500": "retained edge-frame checkpoint",
    "E116 global selected-complex context from E72": "global selected-complex context",
    "E117 continue global selected-complex context": "continued global context",
    "E118 vertex-star selected-complex context": "vertex-star context",
    "E120 mixed vertex/edge-star selected-complex context": "mixed vertex/edge-star context",
    "E124 face boundary-edge-frame gate": "oriented face-edge gate",
    "E128 damped triangle-attention bias from E124": "damped triangle-attention bias",
    "E147 selected-boundary expansion retry": "selected-boundary expansion",
    "E151 E147 best full 30k continuation": "30k continuation",
}

LINEAGE_PLOT_LABELS = {
    "main/full control": "control",
    "E03 warm boundary": "E03: warm boundary",
    "E07 boundary coordinate d=0.5 scaled": "E04/E07: coordinate-cell realization",
    "E15 simplex aux anneal to 0.5": "E09/E15: MSA-to-face + aux anneal",
    "E55 effective-batch-8 aux 0.5 continuation": "E55: batch-8 aux scaffold",
    "E64 E63 confirmation to 4000": "E64: selected-boundary lDDT",
    "E72 continue edge-frame 0.025 to 5500": "E72: edge-frame boundary msgs",
    "E120 mixed vertex/edge-star selected-complex context": "E116-E120: global/star context",
    "E128 damped triangle-attention bias from E124": "E124/E128: face-edge gate + triangle bias",
    "E147 selected-boundary expansion retry": "E147: selected-boundary expansion",
    "E151 E147 best full 30k continuation": "E151: 30k continuation",
}

LINEAGE_LABEL_OFFSETS = {
    "main/full control": (3, -0.050),
    "E03 warm boundary": (5, 0.105),
    "E07 boundary coordinate d=0.5 scaled": (17, -0.080),
    "E15 simplex aux anneal to 0.5": (12, 0.060),
    "E55 effective-batch-8 aux 0.5 continuation": (-13, -0.060),
    "E64 E63 confirmation to 4000": (7, 0.067),
    "E72 continue edge-frame 0.025 to 5500": (15, -0.052),
    "E120 mixed vertex/edge-star selected-complex context": (-32, 0.078),
    "E128 damped triangle-attention bias from E124": (-4, -0.056),
    "E147 selected-boundary expansion retry": (-8, 0.083),
    "E151 E147 best full 30k continuation": (-23, -0.010),
}


def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "axes.edgecolor": Theme.grid,
            "axes.labelcolor": Theme.ink,
            "axes.titlecolor": Theme.ink,
            "xtick.color": Theme.muted,
            "ytick.color": Theme.muted,
            "text.color": Theme.ink,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


@dataclass(frozen=True)
class ExperimentRow:
    table_index: int
    run_num: int | None
    run_label: str
    status: str
    best_step: int | None
    best_val_lddt_ca: float | None
    final_stop_foldscore: float | None
    decision: str
    running_best_val_lddt_ca: float | None
    running_best_label: str | None
    running_best_final_stop_foldscore: float | None
    running_best_foldscore_label: str | None


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
    running_best_lddt: float | None = None
    running_best_lddt_label: str | None = None
    running_best_foldscore: float | None = None
    running_best_foldscore_label: str | None = None
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
        final_stop_foldscore = _optional_float(cells[5])
        if best_val is not None and (running_best_lddt is None or best_val > running_best_lddt):
            running_best_lddt = best_val
            running_best_lddt_label = run_label
        if final_stop_foldscore is not None and (
            running_best_foldscore is None or final_stop_foldscore > running_best_foldscore
        ):
            running_best_foldscore = final_stop_foldscore
            running_best_foldscore_label = run_label
        rows.append(
            ExperimentRow(
                table_index=len(rows) + 1,
                run_num=run_num,
                run_label=run_label,
                status=cells[1],
                best_step=_optional_int(cells[2]),
                best_val_lddt_ca=best_val,
                final_stop_foldscore=final_stop_foldscore,
                decision=cells[8],
                running_best_val_lddt_ca=running_best_lddt,
                running_best_label=running_best_lddt_label,
                running_best_final_stop_foldscore=running_best_foldscore,
                running_best_foldscore_label=running_best_foldscore_label,
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


def write_foldscore_experiment_csv(rows: Iterable[ExperimentRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "table_index",
        "run_num",
        "run_label",
        "status",
        "best_step",
        "final_stop_foldscore",
        "decision",
        "running_best_final_stop_foldscore",
        "running_best_foldscore_label",
        "e151_lineage_contribution",
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
                    "final_stop_foldscore": _format_float(row.final_stop_foldscore),
                    "decision": row.decision,
                    "running_best_final_stop_foldscore": _format_float(row.running_best_final_stop_foldscore),
                    "running_best_foldscore_label": row.running_best_foldscore_label or "",
                    "e151_lineage_contribution": E151_LINEAGE_CONTRIBUTIONS.get(row.run_label, ""),
                }
            )


def _format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def _style_progress_axis(ax) -> None:
    ax.grid(True, color=Theme.grid, linewidth=0.65, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(Theme.grid)
    ax.spines["bottom"].set_color(Theme.grid)
    ax.tick_params(axis="both", colors=Theme.muted, length=3)


def _save_plot(fig, path: Path, formats: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out = path.with_suffix(f".{fmt}")
        kwargs: dict[str, object] = {"format": fmt, "bbox_inches": "tight", "pad_inches": 0.035}
        if fmt == "png":
            kwargs["dpi"] = 300
        fig.savefig(out, **kwargs)


def plot_experiment_progress(rows: list[ExperimentRow], path: Path, *, square: bool = False) -> None:
    apply_paper_style()
    numeric = [row for row in rows if row.best_val_lddt_ca is not None and row.running_best_val_lddt_ca is not None]
    if not numeric:
        raise ValueError("No numeric experiment rows to plot")
    x_values = [row.table_index for row in numeric]
    best_values = [row.best_val_lddt_ca for row in numeric if row.best_val_lddt_ca is not None]
    running_values = [
        row.running_best_val_lddt_ca for row in numeric if row.running_best_val_lddt_ca is not None
    ]
    best_row = max(numeric, key=lambda row: row.best_val_lddt_ca or float("-inf"))
    assert best_row.best_val_lddt_ca is not None

    fig_size = (5.4, 5.1) if square else (6.6, 3.6)
    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(x_values, best_values, color=Theme.returned, linewidth=0.9, alpha=0.42, label="Returned run")
    ax.plot(x_values, running_values, color=Theme.running_best, linewidth=2.0, label="Running best")
    ax.scatter(
        [best_row.table_index],
        [best_row.best_val_lddt_ca],
        color=Theme.highlight,
        s=34,
        zorder=5,
        label=f"Best: {best_row.run_label}",
    )
    ax.axhline(0.45, color=Theme.reference, linestyle="--", linewidth=1.0, alpha=0.8, label="Short gate 0.45")
    ax.axhline(0.70, color=Theme.muted, linestyle=":", linewidth=1.0, alpha=0.8, label="Target 0.70")
    ax.set_title("SimplexFold public benchmark progress", pad=6, fontweight="bold")
    ax.set_xlabel("Experiment # (ledger order)")
    ax.set_ylabel("Best validation C-alpha lDDT")
    ax.set_ylim(0.0, max(0.75, max(running_values) + 0.08))
    _style_progress_axis(ax)
    ax.legend(loc="lower right", frameon=False, handlelength=2.8)
    ax.text(
        0.02,
        0.96,
        f"{best_row.run_label}: {best_row.best_val_lddt_ca:.4f} at step {best_row.best_step}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color=Theme.ink,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_foldscore_experiment_progress(rows: list[ExperimentRow], path: Path) -> None:
    apply_paper_style()
    numeric = [
        row
        for row in rows
        if row.final_stop_foldscore is not None and row.running_best_final_stop_foldscore is not None
    ]
    if not numeric:
        raise ValueError("No numeric FoldScore rows to plot")
    x_values = [row.table_index for row in numeric]
    foldscore_values = [row.final_stop_foldscore for row in numeric if row.final_stop_foldscore is not None]
    lineage = [
        row
        for row in numeric
        if row.run_label in LINEAGE_PLOT_LABELS
    ]
    if not lineage:
        raise ValueError("No E151 lineage rows to plot")
    lineage_x = [row.table_index for row in lineage]
    lineage_y = [row.final_stop_foldscore for row in lineage if row.final_stop_foldscore is not None]
    best_row = max(lineage, key=lambda row: row.final_stop_foldscore or float("-inf"))
    assert best_row.final_stop_foldscore is not None

    fig, ax = plt.subplots(figsize=(7.15, 4.15))
    ax.plot(
        x_values,
        foldscore_values,
        color=Theme.returned,
        linewidth=0.75,
        alpha=0.30,
        zorder=1,
    )
    ax.scatter(
        x_values,
        foldscore_values,
        color=Theme.returned,
        s=7,
        alpha=0.32,
        linewidths=0,
        label="Scored experiments",
        zorder=2,
    )
    ax.plot(
        lineage_x,
        lineage_y,
        color=Theme.running_best,
        linewidth=1.8,
        marker="o",
        markersize=4.2,
        label="E151 lineage milestones",
        zorder=4,
    )
    ax.scatter(
        [best_row.table_index],
        [best_row.final_stop_foldscore],
        color=Theme.highlight,
        s=36,
        zorder=5,
        label="Final E151",
    )
    ax.axhline(
        AF2_MEDIUM_FINAL_FOLDSCORE,
        color=Theme.reference,
        linestyle=(0, (4, 2)),
        linewidth=1.1,
        alpha=0.9,
        label=f"AF2 medium final {AF2_MEDIUM_FINAL_FOLDSCORE:.3f}",
    )
    label_positions: list[tuple[float, float]] = []
    for row in lineage:
        if row.final_stop_foldscore is None or row.run_label not in LINEAGE_PLOT_LABELS:
            continue
        label_dx, label_dy = LINEAGE_LABEL_OFFSETS[row.run_label]
        label_x = row.table_index + label_dx
        label_y = row.final_stop_foldscore + label_dy
        label_positions.append((label_x, label_y))
        ax.annotate(
            LINEAGE_PLOT_LABELS[row.run_label],
            xy=(row.table_index, row.final_stop_foldscore),
            xytext=(label_x, label_y),
            textcoords="data",
            ha="center",
            va="center",
            annotation_clip=False,
            fontsize=6.4,
            color=Theme.ink,
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": Theme.grid,
                "linewidth": 0.55,
                "alpha": 0.92,
            },
            arrowprops={"arrowstyle": "-", "color": Theme.grid, "linewidth": 0.65},
            zorder=6,
        )
    ax.set_title("Goal Mode autoresearch progress", pad=6, fontweight="bold")
    ax.set_xlabel("Experiment # (ledger order)")
    ax.set_ylabel("Final/stop public-val FoldScore")
    label_x_values = [label_x for label_x, _label_y in label_positions]
    label_y_values = [label_y for _label_x, label_y in label_positions]
    y_min = max(0.14, min([*foldscore_values, *label_y_values]) - 0.020)
    y_max = max([*lineage_y, AF2_MEDIUM_FINAL_FOLDSCORE, *label_y_values]) + 0.035
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(min([*x_values, *label_x_values]) - 3, max([*x_values, *label_x_values]) + 5)
    _style_progress_axis(ax)
    ax.legend(loc="lower right", frameon=False, handlelength=2.8)
    fig.tight_layout()
    _save_plot(fig, path, ("png", "svg", "pdf"))
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
    if not isinstance(value, (str, int, float)):
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
    write_foldscore_experiment_csv(rows, args.plot_dir / "foldscore_by_experiment_run.csv")
    plot_experiment_progress(rows, args.plot_dir / "best_val_lddt_by_experiment_run.png")
    plot_experiment_progress(rows, args.plot_dir / "best_val_lddt_by_experiment_run_social.png")
    plot_experiment_progress(rows, args.plot_dir / "best_val_lddt_by_experiment_run_social_square.png", square=True)
    plot_foldscore_experiment_progress(rows, args.plot_dir / "foldscore_by_experiment_run.png")

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
