#!/usr/bin/env python3
"""Record one verified NanoFold benchmark result in EXPERIMENT_RESULTS.md."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.format_experiment_result_row import _load_history, _load_result, format_result_row  # noqa: E402
from scripts.refresh_experiment_results_summary import refresh_summary  # noqa: E402
from scripts.upsert_experiment_result_row import upsert_result_row  # noqa: E402


def record_experiment_result(
    *,
    results_md: Path,
    run_dir: Path,
    run_label: str,
    status: str,
    decision: str,
    variant: str = "full_msa_to_face",
    start_after_step: int | None = None,
    updated_date: str,
    target: float = 0.7,
    confirmation_steps: int = 30_000,
) -> str:
    row = format_result_row(
        _load_result(run_dir / "results.json", variant=variant),
        run_label=run_label,
        status=status,
        decision=decision,
        history=_load_history(run_dir / f"history_{variant}.json"),
        start_after_step=start_after_step,
    )
    markdown = results_md.read_text(encoding="utf-8")
    markdown = upsert_result_row(markdown, row)
    markdown = refresh_summary(
        markdown,
        updated_date=updated_date,
        target=target,
        confirmation_steps=confirmation_steps,
    )
    results_md.write_text(markdown, encoding="utf-8")
    return row


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--results-md", type=Path, default=Path("EXPERIMENT_RESULTS.md"))
    parser.add_argument("--variant", default="full_msa_to_face")
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--status", default="returned")
    parser.add_argument("--decision", required=True)
    parser.add_argument("--start-after-step", type=int)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--target", type=float, default=0.7)
    parser.add_argument("--confirmation-steps", type=int, default=30_000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> str:
    args = parse_args(argv)
    row = record_experiment_result(
        results_md=args.results_md,
        run_dir=args.run_dir,
        run_label=args.run_label,
        status=args.status,
        decision=args.decision,
        variant=args.variant,
        start_after_step=args.start_after_step,
        updated_date=args.date,
        target=args.target,
        confirmation_steps=args.confirmation_steps,
    )
    print(row)
    return row


if __name__ == "__main__":
    main()
