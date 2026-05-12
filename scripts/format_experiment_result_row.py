#!/usr/bin/env python3
"""Format a returned NanoFold benchmark result as an EXPERIMENT_RESULTS row."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load_result(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        if not data:
            raise ValueError(f"No result rows in {path}")
        row = data[0]
    elif isinstance(data, dict):
        row = data
    else:
        raise TypeError(f"Unsupported result JSON shape in {path}: {type(data).__name__}")
    if not isinstance(row, dict):
        raise TypeError(f"Result row must be an object in {path}")
    return row


def _load_history(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"History must be a list in {path}")
    return [row for row in data if isinstance(row, dict)]


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _format_float(value: Any) -> str:
    result = _finite_float(value)
    return "-" if result is None else f"{result:.4f}"


def _metric(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return None


def _best_from_history(
    result: dict[str, Any],
    history: list[dict[str, Any]],
    *,
    start_after_step: int | None,
) -> tuple[int | None, float | None]:
    candidates = []
    for row in history:
        step = _finite_float(row.get("step"))
        lddt = _finite_float(row.get("val_lddt_ca"))
        if step is None or lddt is None:
            continue
        if start_after_step is not None and step <= start_after_step:
            continue
        candidates.append((int(step), lddt))
    if candidates:
        return max(candidates, key=lambda item: item[1])
    step = _finite_float(result.get("completed_steps") or result.get("steps"))
    lddt = _finite_float(result.get("val_lddt_ca"))
    return (int(step) if step is not None else None, lddt)


def format_result_row(
    result: dict[str, Any],
    *,
    run_label: str,
    status: str,
    decision: str,
    history: list[dict[str, Any]] | None = None,
    start_after_step: int | None = None,
) -> str:
    best_step, best_lddt = _best_from_history(result, history or [], start_after_step=start_after_step)
    rg_pred = _metric(result, "val_pred_ca_rg")
    rg_true = _metric(result, "val_true_ca_rg")
    rg = (
        f"{_format_float(rg_pred)} / {_format_float(rg_true)}"
        if _finite_float(rg_pred) is not None or _finite_float(rg_true) is not None
        else "-"
    )
    foldscore = _metric(result, "val_foldscore", "val_FoldScore")
    return (
        f"| {run_label} | {status} | {best_step if best_step is not None else '-'} | "
        f"{_format_float(best_lddt)} | {_format_float(result.get('val_lddt_ca'))} | "
        f"{_format_float(foldscore)} | {_format_float(result.get('val_ca_drmsd'))} | "
        f"{rg} | {decision} |"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_json", type=Path)
    parser.add_argument("--history-json", type=Path)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--status", default="completed")
    parser.add_argument("--decision", default="pending")
    parser.add_argument(
        "--start-after-step",
        type=int,
        help="Ignore inherited resume history rows at or before this step when finding the best validation lDDT.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> str:
    args = parse_args(argv)
    row = format_result_row(
        _load_result(args.results_json),
        run_label=args.run_label,
        status=args.status,
        decision=args.decision,
        history=_load_history(args.history_json),
        start_after_step=args.start_after_step,
    )
    print(row)
    return row


if __name__ == "__main__":
    main()
