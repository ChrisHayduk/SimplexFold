#!/usr/bin/env python3
"""Summarize live or returned NanoFold benchmark run artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


STATUS_KEYS = (
    "active_step",
    "active_microbatch",
    "active_microbatches",
    "active_eval_batch",
    "active_eval_batches",
    "active_eval_examples",
    "completed_step",
    "target_steps",
    "start_step",
    "elapsed_seconds_total",
    "elapsed_seconds_run",
    "last_history_step",
    "history_rows",
    "last_lddt_ca",
    "last_train_loss",
    "total_examples",
    "stopped_early",
    "effective_batch_size",
    "num_workers",
)

RESULT_KEYS = (
    "completed_steps",
    "val_lddt_ca",
    "val_foldscore",
    "val_ca_drmsd",
    "val_ca_pred_rg",
    "val_ca_true_rg",
    "parameters",
    "effective_batch_size",
    "num_workers",
    "stopped_early",
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "size": 0}
    stat = path.stat()
    return {"exists": True, "size": stat.st_size, "mtime": stat.st_mtime}


def _matching_result(path: Path, variant: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = _load_json(path)
    rows = data if isinstance(data, list) else [data]
    for row in rows:
        if isinstance(row, dict) and row.get("variant", variant) == variant:
            return {key: row.get(key) for key in RESULT_KEYS if key in row}
    return None


def _history_last_row(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = _load_json(path)
    if not isinstance(data, list) or not data:
        return None
    last = data[-1]
    if not isinstance(last, dict):
        return None
    return {
        key: last[key]
        for key in (
            "step",
            "val_lddt_ca",
            "val_foldscore",
            "val_ca_drmsd",
            "val_ca_pred_rg",
            "val_ca_true_rg",
        )
        if key in last
    }


def _csv_row_count(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _progress_summary(
    status: dict[str, Any] | None,
    files: dict[str, dict[str, Any]],
) -> dict[str, float | int | str] | None:
    if not status:
        return None
    completed_step = status.get("completed_step")
    start_step = status.get("start_step")
    target_steps = status.get("target_steps")
    elapsed_source = "status_elapsed_seconds_run"
    run_elapsed_seconds = status.get("elapsed_seconds_run")
    if run_elapsed_seconds is None:
        status_mtime = files["status"].get("mtime")
        metadata_mtime = files["metadata"].get("mtime")
        if status_mtime is None or metadata_mtime is None:
            return None
        elapsed_source = "status_metadata_mtime_delta"
        run_elapsed_seconds = max(0.0, float(status_mtime) - float(metadata_mtime))
    if completed_step is None or start_step is None:
        return None
    completed_delta = max(0, int(completed_step) - int(start_step))
    run_elapsed_seconds = float(run_elapsed_seconds)
    if completed_delta <= 0 or run_elapsed_seconds <= 0.0:
        return None
    seconds_per_step = run_elapsed_seconds / completed_delta
    progress: dict[str, float | int | str] = {
        "completed_delta_steps": completed_delta,
        "run_elapsed_seconds": run_elapsed_seconds,
        "elapsed_source": elapsed_source,
        "seconds_per_step": seconds_per_step,
        "steps_per_hour": 3600.0 / seconds_per_step,
    }
    if target_steps is not None:
        remaining_steps = max(0, int(target_steps) - int(completed_step))
        progress["remaining_steps"] = remaining_steps
        progress["estimated_seconds_remaining"] = remaining_steps * seconds_per_step
    return progress


def summarize_run(run_dir: Path, *, variant: str = "full_msa_to_face") -> dict[str, Any]:
    run_dir = run_dir.resolve()
    paths = {
        "status": run_dir / f"status_{variant}.json",
        "results_json": run_dir / "results.json",
        "results_csv": run_dir / "results.csv",
        "history": run_dir / f"history_{variant}.json",
        "eval_details": run_dir / f"eval_details_{variant}.csv",
        "metadata": run_dir / "run_metadata.json",
        "checkpoint": run_dir / "checkpoints" / f"{variant}_latest.pt",
    }
    files = {name: _file_summary(path) for name, path in paths.items()}
    status = None
    if paths["status"].exists():
        loaded = _load_json(paths["status"])
        if isinstance(loaded, dict):
            status = {key: loaded.get(key) for key in STATUS_KEYS if key in loaded}

    result = _matching_result(paths["results_json"], variant)
    eval_rows = _csv_row_count(paths["eval_details"])
    history_last = _history_last_row(paths["history"])
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "variant": variant,
        "state": _state(files),
        "files": files,
        "status": status,
        "progress": _progress_summary(status, files),
        "result": result,
        "history_last": history_last,
        "history_last_step": None if history_last is None else history_last.get("step"),
        "eval_rows": eval_rows,
    }
    return summary


def _state(files: dict[str, dict[str, Any]]) -> str:
    has_results = files["results_json"]["exists"] and files["results_csv"]["exists"]
    has_return_bundle = has_results and files["eval_details"]["exists"] and files["checkpoint"]["exists"]
    if has_return_bundle:
        return "returned"
    if files["status"]["exists"] and not has_results:
        return "active"
    if has_results:
        return "partial-return"
    return "startup-or-missing"


def _format_line(summary: dict[str, Any]) -> str:
    status = summary.get("status") or {}
    result = summary.get("result") or {}
    history_last = summary.get("history_last") or {}
    progress = summary.get("progress") or {}
    eta_seconds = progress.get("estimated_seconds_remaining")
    eta_hours = f"{eta_seconds / 3600.0:.1f}h" if isinstance(eta_seconds, (float, int)) else "-"
    steps_per_hour = progress.get("steps_per_hour")
    rate = f"{steps_per_hour:.1f}/h" if isinstance(steps_per_hour, (float, int)) else "-"
    lddt_ca = result.get("val_lddt_ca")
    if lddt_ca is None:
        lddt_ca = status.get("last_lddt_ca")
    if lddt_ca is None:
        lddt_ca = history_last.get("val_lddt_ca", "-")
    parts = [
        f"{Path(summary['run_dir']).name}: {summary['state']}",
        f"completed={status.get('completed_step', result.get('completed_steps', '-'))}",
        f"target={status.get('target_steps', '-')}",
        f"rate={rate}",
        f"eta={eta_hours}",
        f"last_history={status.get('last_history_step', summary.get('history_last_step', '-'))}",
        f"lddt_ca={lddt_ca}",
        f"eval_rows={summary.get('eval_rows') if summary.get('eval_rows') is not None else '-'}",
        f"checkpoint={summary['files']['checkpoint']['exists']}",
    ]
    return " | ".join(parts)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="+", type=Path)
    parser.add_argument("--variant", default="full_msa_to_face")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON summaries.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> list[dict[str, Any]]:
    args = parse_args(argv)
    summaries = [summarize_run(path, variant=args.variant) for path in args.run_dir]
    if args.json:
        print(json.dumps(summaries, indent=2, sort_keys=True))
    else:
        for summary in summaries:
            print(_format_line(summary))
    return summaries


if __name__ == "__main__":
    main()
