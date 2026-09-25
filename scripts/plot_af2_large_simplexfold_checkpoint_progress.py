#!/usr/bin/env python3
"""Plot AF2-large vs one SimplexFold large checkpoint curve.

Each run gets its own stable prefix, for example ``e152-vs-af2-large`` or
``e154-vs-af2-large``. Optional aliases must identify the same experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_RUN_KEY_RE = re.compile(r"(?<![a-z0-9])(e\d+)(?![a-z0-9])")


@dataclass(frozen=True)
class CurvePoint:
    step: int
    foldscore: float
    lddt_ca: float
    val_loss: float | None = None
    ca_drmsd: float | None = None
    sample_budget_fraction: float | None = None
    cumulative_samples_seen: int | None = None


def _finite_float(value: object, *, field: str) -> float:
    if not isinstance(value, (str, int, float)):
        raise ValueError(f"{field} is not numeric: {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} is not finite: {value!r}")
    return number


def _optional_float(value: object) -> float | None:
    if not isinstance(value, (str, int, float)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _optional_int(value: object) -> int | None:
    number = _optional_float(value)
    return None if number is None else int(number)


def _load_af2_points(path: Path) -> list[CurvePoint]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    checkpoints = payload.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise ValueError(f"{path} does not contain a checkpoints list")
    points = [
        CurvePoint(
            step=int(row["step"]),
            foldscore=_finite_float(row.get("mean_foldscore"), field="mean_foldscore"),
            lddt_ca=_finite_float(row.get("mean_lddt_ca"), field="mean_lddt_ca"),
            sample_budget_fraction=_optional_float(row.get("sample_budget_fraction")),
            cumulative_samples_seen=_optional_int(row.get("cumulative_samples_seen")),
        )
        for row in checkpoints
    ]
    return sorted(points, key=lambda point: point.step)


def _load_simplexfold_points(path: Path) -> list[CurvePoint]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not contain a history list")
    points: list[CurvePoint] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if "val_foldscore" not in row or "val_lddt_ca" not in row:
            continue
        points.append(
            CurvePoint(
                step=int(row["step"]),
                foldscore=_finite_float(row.get("val_foldscore"), field="val_foldscore"),
                lddt_ca=_finite_float(row.get("val_lddt_ca"), field="val_lddt_ca"),
                val_loss=_optional_float(row.get("val_loss")),
                ca_drmsd=_optional_float(row.get("val_ca_drmsd")),
            )
        )
    if not points:
        raise ValueError(f"{path} has no validation points with FoldScore/lDDT-Ca")
    return sorted(points, key=lambda point: point.step)


def _load_json_if_present(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.10f}"


def _af2_by_step(points: Iterable[CurvePoint]) -> dict[int, CurvePoint]:
    return {point.step: point for point in points}


def _write_checkpoint_curve_csv(
    path: Path,
    *,
    run_key: str,
    simplexfold_points: Sequence[CurvePoint],
    af2_points: Sequence[CurvePoint],
) -> None:
    af2 = _af2_by_step(af2_points)
    fieldnames = [
        "step",
        f"{run_key}_foldscore",
        "af2_large_foldscore",
        f"foldscore_delta_{run_key}_minus_af2",
        f"{run_key}_lddt_ca",
        "af2_large_lddt_ca",
        f"lddt_ca_delta_{run_key}_minus_af2",
        f"{run_key}_val_loss",
        f"{run_key}_ca_drmsd",
        "af2_large_sample_budget_fraction",
        "af2_large_cumulative_samples_seen",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for point in simplexfold_points:
            baseline = af2.get(point.step)
            if baseline is None:
                continue
            writer.writerow(
                {
                    "step": point.step,
                    f"{run_key}_foldscore": _format_float(point.foldscore),
                    "af2_large_foldscore": _format_float(baseline.foldscore),
                    f"foldscore_delta_{run_key}_minus_af2": _format_float(
                        point.foldscore - baseline.foldscore
                    ),
                    f"{run_key}_lddt_ca": _format_float(point.lddt_ca),
                    "af2_large_lddt_ca": _format_float(baseline.lddt_ca),
                    f"lddt_ca_delta_{run_key}_minus_af2": _format_float(point.lddt_ca - baseline.lddt_ca),
                    f"{run_key}_val_loss": _format_float(point.val_loss),
                    f"{run_key}_ca_drmsd": _format_float(point.ca_drmsd),
                    "af2_large_sample_budget_fraction": _format_float(baseline.sample_budget_fraction),
                    "af2_large_cumulative_samples_seen": baseline.cumulative_samples_seen
                    if baseline.cumulative_samples_seen is not None
                    else "",
                }
            )


def _write_plot_csv(
    path: Path,
    *,
    simplexfold_label: str,
    af2_points: Sequence[CurvePoint],
    simplexfold_points: Sequence[CurvePoint],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "step", "foldscore", "lddt_ca"])
        writer.writeheader()
        for point in af2_points:
            writer.writerow(
                {
                    "model": "AF2 large baseline",
                    "step": point.step,
                    "foldscore": _format_float(point.foldscore),
                    "lddt_ca": _format_float(point.lddt_ca),
                }
            )
        for point in simplexfold_points:
            writer.writerow(
                {
                    "model": simplexfold_label,
                    "step": point.step,
                    "foldscore": _format_float(point.foldscore),
                    "lddt_ca": _format_float(point.lddt_ca),
                }
            )


def _plot(
    path_base: Path,
    *,
    run_key: str,
    title: str,
    simplexfold_label: str,
    af2_points: Sequence[CurvePoint],
    simplexfold_points: Sequence[CurvePoint],
    formats: Sequence[str],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=200)
    for ax, metric, ylabel in [
        (axes[0], "foldscore", "FoldScore"),
        (axes[1], "lddt_ca", "C-alpha lDDT"),
    ]:
        ax.plot(
            [point.step for point in af2_points],
            [getattr(point, metric) for point in af2_points],
            color="#C03A64",
            marker="o",
            linewidth=2.0,
            label="AF2-large checkpoint eval",
        )
        ax.plot(
            [point.step for point in simplexfold_points],
            [getattr(point, metric) for point in simplexfold_points],
            color="#078A94",
            marker="o",
            linewidth=2.0,
            label=simplexfold_label,
        )
        ax.set_xlabel("Training step")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#D6DEE6", linewidth=0.8, alpha=0.8)

    latest = simplexfold_points[-1]
    baseline = _af2_by_step(af2_points).get(latest.step)
    if baseline is not None:
        axes[0].annotate(
            f"latest gap {latest.foldscore - baseline.foldscore:+.3f}",
            xy=(latest.step, latest.foldscore),
            xytext=(-70, -36),
            textcoords="offset points",
            arrowprops={"arrowstyle": "-", "color": "#078A94"},
            fontsize=9,
            color="#243B53",
        )
        axes[1].annotate(
            f"latest gap {latest.lddt_ca - baseline.lddt_ca:+.3f}",
            xy=(latest.step, latest.lddt_ca),
            xytext=(-70, -36),
            textcoords="offset points",
            arrowprops={"arrowstyle": "-", "color": "#078A94"},
            fontsize=9,
            color="#243B53",
        )

    fig.suptitle(title, fontsize=16, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.94))
    fig.text(
        0.5,
        0.025,
        f"{run_key.upper()} and AF2-large curves use actual checkpoint/history evaluations.",
        ha="center",
        fontsize=9,
        color="#5F6B7A",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.88))
    path_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(path_base.with_suffix(f".{fmt}"))
    plt.close(fig)


def _metadata_run_key(payload: dict[str, object]) -> str | None:
    value = payload.get("simplexfold_run_key")
    if isinstance(value, str):
        return value.lower()
    for key in ("simplexfold_source", "simplexfold_local_source", "note"):
        text = payload.get(key)
        if isinstance(text, str):
            match = _RUN_KEY_RE.search(text.lower())
            if match is not None:
                return match.group(1)
    return None


def _parse_run_key(value: str) -> str:
    lowered = value.lower()
    if re.fullmatch(r"e\d+", lowered) is None:
        raise argparse.ArgumentTypeError("run key must look like e154")
    return lowered


def _assert_prefix_matches_run(prefix: str, run_key: str) -> None:
    lowered = prefix.lower()
    if f"_{run_key}_" in lowered or f"-{run_key}-" in lowered:
        return
    raise ValueError(f"Refusing to write {run_key.upper()} into mismatched prefix {prefix!r}")


def _write_metadata(path: Path, metadata: dict[str, object], *, run_key: str) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        existing_key = _metadata_run_key(existing)
        if existing_key is not None and existing_key != run_key:
            _assert_prefix_matches_run(path.stem, run_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_outputs(
    prefix: Path,
    *,
    run_key: str,
    simplexfold_label: str,
    title: str,
    metadata: dict[str, object],
    af2_points: Sequence[CurvePoint],
    simplexfold_points: Sequence[CurvePoint],
    formats: Sequence[str],
) -> None:
    _write_checkpoint_curve_csv(
        prefix.with_name(f"{prefix.name}_checkpoint_curve.csv"),
        run_key=run_key,
        simplexfold_points=simplexfold_points,
        af2_points=af2_points,
    )
    plot_base = prefix.with_name(f"{prefix.name}_foldscore_lddt")
    _write_plot_csv(
        plot_base.with_suffix(".csv"),
        simplexfold_label=simplexfold_label,
        af2_points=af2_points,
        simplexfold_points=simplexfold_points,
    )
    _plot(
        plot_base,
        run_key=run_key,
        title=title,
        simplexfold_label=simplexfold_label,
        af2_points=af2_points,
        simplexfold_points=simplexfold_points,
        formats=formats,
    )
    _write_metadata(prefix.with_name(f"{prefix.name}_metadata.json"), metadata, run_key=run_key)


def _write_legacy_alias(
    prefix: Path,
    *,
    run_key: str,
    simplexfold_label: str,
    title: str,
    metadata: dict[str, object],
    af2_points: Sequence[CurvePoint],
    simplexfold_points: Sequence[CurvePoint],
    formats: Sequence[str],
) -> None:
    _assert_prefix_matches_run(prefix.name, run_key)
    _write_plot_csv(
        prefix.with_suffix(".csv"),
        simplexfold_label=simplexfold_label,
        af2_points=af2_points,
        simplexfold_points=simplexfold_points,
    )
    _plot(
        prefix,
        run_key=run_key,
        title=title,
        simplexfold_label=simplexfold_label,
        af2_points=af2_points,
        simplexfold_points=simplexfold_points,
        formats=formats,
    )
    _write_metadata(prefix.with_name(f"{prefix.name}_metadata.json"), metadata, run_key=run_key)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-key", required=True, type=_parse_run_key)
    parser.add_argument("--simplexfold-label", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--af2-checkpoints", required=True, type=Path)
    parser.add_argument("--simplexfold-history", required=True, type=Path)
    parser.add_argument("--simplexfold-status", type=Path)
    parser.add_argument("--run-metadata", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--formats", nargs="+", default=["png", "svg"], choices=["png", "svg", "pdf"])
    parser.add_argument("--write-legacy-alias", action="store_true")
    parser.add_argument(
        "--legacy-prefix",
        help="Optional legacy alias stem. Must contain the selected run key.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, str]:
    args = build_parser().parse_args(argv)
    af2_points = _load_af2_points(args.af2_checkpoints)
    simplexfold_points = _load_simplexfold_points(args.simplexfold_history)
    status = _load_json_if_present(args.simplexfold_status)
    run_metadata = _load_json_if_present(args.run_metadata)
    run_key = args.run_key.lower()
    prefix = args.output_dir / f"{run_key}-vs-af2-large"
    latest = simplexfold_points[-1]
    baseline = _af2_by_step(af2_points).get(latest.step)
    metadata: dict[str, object] = {
        "af2_source": str(args.af2_checkpoints),
        "af2_steps": [point.step for point in af2_points],
        "generated_at": datetime.now(UTC).isoformat(),
        "latest_simplexfold_foldscore": latest.foldscore,
        "latest_simplexfold_lddt_ca": latest.lddt_ca,
        "simplexfold_history_source": str(args.simplexfold_history),
        "simplexfold_run_key": run_key,
        "simplexfold_steps": [point.step for point in simplexfold_points],
        "status_completed_step": status.get("completed_step"),
        "status_target_steps": status.get("target_steps"),
        "run_metadata": run_metadata,
    }
    if baseline is not None:
        metadata.update(
            {
                "same_step_af2_foldscore": baseline.foldscore,
                "same_step_af2_lddt_ca": baseline.lddt_ca,
                "same_step_foldscore_delta": latest.foldscore - baseline.foldscore,
                "same_step_lddt_ca_delta": latest.lddt_ca - baseline.lddt_ca,
            }
        )
    _write_outputs(
        prefix,
        run_key=run_key,
        simplexfold_label=args.simplexfold_label,
        title=args.title,
        metadata=metadata,
        af2_points=af2_points,
        simplexfold_points=simplexfold_points,
        formats=args.formats,
    )
    written = {
        "prefix": str(prefix),
        "metadata": str(prefix.with_name(f"{prefix.name}_metadata.json")),
        "plot_csv": str(prefix.with_name(f"{prefix.name}_foldscore_lddt.csv")),
        "checkpoint_curve_csv": str(prefix.with_name(f"{prefix.name}_checkpoint_curve.csv")),
    }
    if args.write_legacy_alias:
        legacy_stem = args.legacy_prefix or f"af2_large_vs_{run_key}_simplexfold_large_foldscore_trace"
        legacy_prefix = args.output_dir / legacy_stem
        _write_legacy_alias(
            legacy_prefix,
            run_key=run_key,
            simplexfold_label=args.simplexfold_label,
            title=args.title,
            metadata=metadata,
            af2_points=af2_points,
            simplexfold_points=simplexfold_points,
            formats=args.formats,
        )
        written["legacy_prefix"] = str(legacy_prefix)
    print(json.dumps(written, indent=2, sort_keys=True))
    return written


if __name__ == "__main__":
    main()
