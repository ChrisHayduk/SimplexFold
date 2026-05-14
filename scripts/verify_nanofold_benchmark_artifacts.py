#!/usr/bin/env python3
"""Verify returned NanoFold benchmark artifacts before recording a result."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _single_result(path: Path, variant: str) -> dict[str, Any]:
    data = _load_json(path)
    rows = data if isinstance(data, list) else [data]
    if not rows:
        raise ValueError(f"No result rows in {path}")
    matches = [row for row in rows if isinstance(row, dict) and row.get("variant", variant) == variant]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one result row for variant {variant!r} in {path}, found {len(matches)}")
    return matches[0]


def _history_rows(path: Path) -> list[dict[str, Any]]:
    data = _load_json(path)
    if not isinstance(data, list):
        raise TypeError(f"History must be a list in {path}")
    return [row for row in data if isinstance(row, dict)]


def _csv_row_count(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _finite_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _parse_expected_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _parse_metadata_expectation(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(f"Metadata expectation must be KEY=VALUE, got {raw!r}")
    key, value = raw.split("=", 1)
    if not key:
        raise ValueError(f"Metadata expectation has empty key: {raw!r}")
    return key, _parse_expected_value(value)


def _values_match(actual: Any, expected: Any) -> bool:
    actual_number = _finite_number(actual)
    expected_number = _finite_number(expected)
    if actual_number is not None and expected_number is not None:
        return math.isclose(actual_number, expected_number, rel_tol=0.0, abs_tol=1e-9)
    return actual == expected


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_artifacts(
    run_dir: Path,
    *,
    variant: str = "full_msa_to_face",
    expected_completed_steps: int | None = None,
    expected_effective_batch_size: int | None = None,
    max_parameters: int | None = None,
    expect_stopped_early: bool | None = None,
    expected_eval_rows: int | None = None,
    expected_history_last_step: int | None = None,
    require_checkpoint: bool = True,
    metadata_expectations: list[tuple[str, Any]] | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    result = _single_result(run_dir / "results.json", variant)
    metadata = _load_json(run_dir / "run_metadata.json")
    if not isinstance(metadata, dict):
        raise TypeError(f"run_metadata.json must contain an object in {run_dir}")

    csv_results = run_dir / "results.csv"
    _require(csv_results.exists(), f"Missing required file: {csv_results}")

    history = _history_rows(run_dir / f"history_{variant}.json")
    _require(bool(history), f"History has no rows for variant {variant!r}")

    eval_rows = _csv_row_count(run_dir / f"eval_details_{variant}.csv")

    checkpoint = run_dir / "checkpoints" / f"{variant}_latest.pt"
    if require_checkpoint:
        _require(checkpoint.exists(), f"Missing required checkpoint: {checkpoint}")

    completed_steps = int(result.get("completed_steps", -1))
    effective_batch_size = int(result.get("effective_batch_size", -1))
    parameters = int(result.get("parameters", -1))
    stopped_early = bool(result.get("stopped_early"))
    last_history_step = int(history[-1].get("step", -1))

    if expected_completed_steps is not None:
        _require(
            completed_steps == expected_completed_steps,
            f"completed_steps={completed_steps}, expected {expected_completed_steps}",
        )
    if expected_effective_batch_size is not None:
        _require(
            effective_batch_size == expected_effective_batch_size,
            f"effective_batch_size={effective_batch_size}, expected {expected_effective_batch_size}",
        )
    if max_parameters is not None:
        _require(parameters <= max_parameters, f"parameters={parameters} exceeds max {max_parameters}")
    if expect_stopped_early is not None:
        _require(stopped_early is expect_stopped_early, f"stopped_early={stopped_early}, expected {expect_stopped_early}")
    if expected_eval_rows is not None:
        _require(eval_rows == expected_eval_rows, f"eval rows={eval_rows}, expected {expected_eval_rows}")
    if expected_history_last_step is not None:
        _require(
            last_history_step == expected_history_last_step,
            f"history last step={last_history_step}, expected {expected_history_last_step}",
        )

    for key, expected in metadata_expectations or []:
        _require(key in metadata, f"Missing metadata key: {key}")
        actual = metadata[key]
        _require(_values_match(actual, expected), f"metadata {key}={actual!r}, expected {expected!r}")

    for metric in ("val_lddt_ca", "val_foldscore", "val_ca_drmsd"):
        _require(_finite_number(result.get(metric)) is not None, f"Missing or non-finite result metric: {metric}")

    return {
        "run_dir": str(run_dir),
        "variant": variant,
        "completed_steps": completed_steps,
        "effective_batch_size": effective_batch_size,
        "parameters": parameters,
        "stopped_early": stopped_early,
        "eval_rows": eval_rows,
        "history_rows": len(history),
        "history_last_step": last_history_step,
        "val_lddt_ca": float(result["val_lddt_ca"]),
        "val_foldscore": float(result["val_foldscore"]),
        "val_ca_drmsd": float(result["val_ca_drmsd"]),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--variant", default="full_msa_to_face")
    parser.add_argument("--expected-completed-steps", type=int)
    parser.add_argument("--expected-effective-batch-size", type=int)
    parser.add_argument("--max-parameters", type=int)
    parser.add_argument("--expected-eval-rows", type=int)
    parser.add_argument("--expected-history-last-step", type=int)
    parser.add_argument("--expect-stopped-early", choices=("true", "false"))
    parser.add_argument("--metadata", action="append", default=[], help="Require a run_metadata.json KEY=VALUE pair.")
    parser.add_argument("--allow-missing-checkpoint", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    stopped_early = None
    if args.expect_stopped_early is not None:
        stopped_early = args.expect_stopped_early == "true"
    summary = verify_artifacts(
        args.run_dir,
        variant=args.variant,
        expected_completed_steps=args.expected_completed_steps,
        expected_effective_batch_size=args.expected_effective_batch_size,
        max_parameters=args.max_parameters,
        expect_stopped_early=stopped_early,
        expected_eval_rows=args.expected_eval_rows,
        expected_history_last_step=args.expected_history_last_step,
        require_checkpoint=not args.allow_missing_checkpoint,
        metadata_expectations=[_parse_metadata_expectation(raw) for raw in args.metadata],
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
