#!/usr/bin/env python3
"""Summarize NanoFold eval-detail CSVs for local-to-global failure analysis."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

DEFAULT_LENGTH_BINS: tuple[tuple[float, float], ...] = (
    (0.0, 80.0),
    (80.0, 120.0),
    (120.0, 160.0),
    (160.0, 220.0),
    (220.0, math.inf),
)

DEFAULT_METRICS: tuple[str, ...] = (
    "lddt_ca",
    "foldscore",
    "ca_drmsd",
    "pred_ca_rg",
    "true_ca_rg",
    "rg_ratio",
    "rg_abs_err",
    "boundary_lddt_mean",
    "boundary_contraction_mean",
    "boundary_edge_degree_mean",
    "outer_edge_mean",
)


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else None


def _quantile(values: list[float], q: float) -> float | None:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return None
    index = min(len(finite) - 1, max(0, int(q * len(finite))))
    return finite[index]


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys, strict=True) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return None
    mean_x = sum(x for x, _ in pairs) / len(pairs)
    mean_y = sum(y for _, y in pairs) / len(pairs)
    var_x = sum((x - mean_x) ** 2 for x, _ in pairs)
    var_y = sum((y - mean_y) ** 2 for _, y in pairs)
    if var_x <= 0.0 or var_y <= 0.0:
        return None
    cov = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    return cov / math.sqrt(var_x * var_y)


def _metric_stats(rows: list[dict[str, float]], metric: str) -> dict[str, float | int | None]:
    values = [row.get(metric, float("nan")) for row in rows]
    lddt = [row.get("lddt_ca", float("nan")) for row in rows]
    return {
        "n": sum(1 for value in values if math.isfinite(value)),
        "mean": _mean(values),
        "p10": _quantile(values, 0.10),
        "p50": _quantile(values, 0.50),
        "p90": _quantile(values, 0.90),
        "corr_lddt_ca": None if metric == "lddt_ca" else _pearson(lddt, values),
    }


def _safe_divide(numerator: float | None, denominator: float | None) -> float:
    if numerator is None or denominator is None or denominator == 0.0:
        return float("nan")
    return numerator / denominator


def _average_present(*values: float | None) -> float:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return sum(finite) / len(finite) if finite else float("nan")


def load_eval_rows(path: Path) -> list[dict[str, float]]:
    """Load an eval-detail CSV and derive aggregate-friendly helper columns."""
    with path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows: list[dict[str, float]] = []
    for raw in raw_rows:
        row = {}
        for key, value in raw.items():
            number = _finite_float(value)
            row[key] = number if number is not None else float("nan")
        row["lddt_ca"] = row.get("lddt_ca", row.get("val_lddt_ca", float("nan")))
        row["rg_ratio"] = _safe_divide(row.get("pred_ca_rg"), row.get("true_ca_rg"))
        row["rg_abs_err"] = abs(row.get("pred_ca_rg", float("nan")) - row.get("true_ca_rg", float("nan")))
        row["boundary_lddt_mean"] = _average_present(
            row.get("simplex_face_boundary_lddt"),
            row.get("simplex_tetra_boundary_lddt"),
        )
        row["boundary_contraction_mean"] = _average_present(
            row.get("simplex_face_boundary_contraction_fraction"),
            row.get("simplex_tetra_boundary_contraction_fraction"),
        )
        row["boundary_edge_degree_mean"] = _average_present(
            row.get("simplex_face_boundary_edge_mean_degree"),
            row.get("simplex_tetra_boundary_edge_mean_degree"),
        )
        row["outer_edge_mean"] = _average_present(
            row.get("simplex_face_outer_edge_mean_degree"),
            row.get("simplex_tetra_outer_edge_mean_degree"),
        )
        rows.append(row)
    return rows


def _bin_label(lower: float, upper: float) -> str:
    if math.isinf(upper):
        return f">={int(lower)}"
    return f"{int(lower)}-{int(upper - 1)}"


def _summarize_subset(rows: list[dict[str, float]]) -> dict[str, float | int | None]:
    return {
        "n": len(rows),
        "mean_length": _mean([row.get("length", float("nan")) for row in rows]),
        "mean_lddt_ca": _mean([row.get("lddt_ca", float("nan")) for row in rows]),
        "mean_foldscore": _mean([row.get("foldscore", float("nan")) for row in rows]),
        "mean_ca_drmsd": _mean([row.get("ca_drmsd", float("nan")) for row in rows]),
        "mean_rg_ratio": _mean([row.get("rg_ratio", float("nan")) for row in rows]),
        "mean_boundary_lddt": _mean([row.get("boundary_lddt_mean", float("nan")) for row in rows]),
        "mean_boundary_contraction": _mean(
            [row.get("boundary_contraction_mean", float("nan")) for row in rows]
        ),
        "mean_boundary_edge_degree": _mean(
            [row.get("boundary_edge_degree_mean", float("nan")) for row in rows]
        ),
        "mean_outer_edge_degree": _mean([row.get("outer_edge_mean", float("nan")) for row in rows]),
    }


def summarize_eval_details(
    rows: list[dict[str, float]],
    *,
    stratum_size: int = 100,
    high_boundary_threshold: float = 0.75,
    low_lddt_threshold: float = 0.40,
    length_bins: tuple[tuple[float, float], ...] = DEFAULT_LENGTH_BINS,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("No eval-detail rows found")
    sorted_by_lddt = sorted(rows, key=lambda row: row.get("lddt_ca", float("nan")))
    stratum_size = max(1, min(stratum_size, len(rows)))
    middle_start = max(0, len(rows) // 2 - stratum_size // 2)
    high_boundary = [
        row for row in rows if row.get("boundary_lddt_mean", float("nan")) >= high_boundary_threshold
    ]
    high_boundary_low_global = [
        row for row in high_boundary if row.get("lddt_ca", float("nan")) < low_lddt_threshold
    ]
    length_summary = []
    for lower, upper in length_bins:
        subset = [
            row
            for row in rows
            if lower <= row.get("length", float("nan")) < upper
        ]
        length_summary.append(
            {
                "label": _bin_label(lower, upper),
                "lower": lower,
                "upper": None if math.isinf(upper) else upper,
                **_summarize_subset(subset),
            }
        )
    return {
        "n": len(rows),
        "metrics": {metric: _metric_stats(rows, metric) for metric in DEFAULT_METRICS},
        "length_bins": length_summary,
        "lddt_strata": {
            "bottom": _summarize_subset(sorted_by_lddt[:stratum_size]),
            "middle": _summarize_subset(sorted_by_lddt[middle_start : middle_start + stratum_size]),
            "top": _summarize_subset(sorted_by_lddt[-stratum_size:]),
        },
        "high_boundary_low_global": {
            "high_boundary_threshold": high_boundary_threshold,
            "low_lddt_threshold": low_lddt_threshold,
            "high_boundary_count": len(high_boundary),
            "high_boundary_low_global_count": len(high_boundary_low_global),
            **_summarize_subset(high_boundary_low_global),
        },
    }


def _format_number(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    number = _finite_float(value)
    return "-" if number is None else f"{number:.4f}"


def format_summary(summary: dict[str, Any]) -> str:
    lines = [f"Rows: {summary['n']}", "", "Metric summary:"]
    for metric, stats in summary["metrics"].items():
        lines.append(
            f"- {metric}: mean={_format_number(stats['mean'])}, "
            f"p10={_format_number(stats['p10'])}, "
            f"p50={_format_number(stats['p50'])}, "
            f"p90={_format_number(stats['p90'])}, "
            f"corr_lddt_ca={_format_number(stats['corr_lddt_ca'])}"
        )
    lines.extend(["", "Length bins:"])
    for row in summary["length_bins"]:
        lines.append(
            f"- {row['label']}: n={row['n']}, "
            f"lddt={_format_number(row['mean_lddt_ca'])}, "
            f"boundary={_format_number(row['mean_boundary_lddt'])}, "
            f"rg_ratio={_format_number(row['mean_rg_ratio'])}, "
            f"drmsd={_format_number(row['mean_ca_drmsd'])}"
        )
    lines.extend(["", "lDDT strata:"])
    for name, row in summary["lddt_strata"].items():
        lines.append(
            f"- {name}: n={row['n']}, "
            f"lddt={_format_number(row['mean_lddt_ca'])}, "
            f"length={_format_number(row['mean_length'])}, "
            f"boundary={_format_number(row['mean_boundary_lddt'])}, "
            f"rg_ratio={_format_number(row['mean_rg_ratio'])}"
        )
    high = summary["high_boundary_low_global"]
    lines.extend(
        [
            "",
            "High-boundary / low-global subset:",
            (
                f"- count={high['high_boundary_low_global_count']} of "
                f"{high['high_boundary_count']} high-boundary rows; "
                f"lddt={_format_number(high['mean_lddt_ca'])}, "
                f"length={_format_number(high['mean_length'])}, "
                f"boundary={_format_number(high['mean_boundary_lddt'])}, "
                f"rg_ratio={_format_number(high['mean_rg_ratio'])}"
            ),
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_details_csv", type=Path)
    parser.add_argument("--stratum-size", type=int, default=100)
    parser.add_argument("--high-boundary-threshold", type=float, default=0.75)
    parser.add_argument("--low-lddt-threshold", type=float, default=0.40)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    summary = summarize_eval_details(
        load_eval_rows(args.eval_details_csv),
        stratum_size=args.stratum_size,
        high_boundary_threshold=args.high_boundary_threshold,
        low_lddt_threshold=args.low_lddt_threshold,
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(format_summary(summary))
    return summary


if __name__ == "__main__":
    main()
