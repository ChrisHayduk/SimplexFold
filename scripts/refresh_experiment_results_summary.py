#!/usr/bin/env python3
"""Refresh the EXPERIMENT_RESULTS.md date and best-score summary."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import re


TABLE_HEADER = "| Run | Status | Best step | Best `val_lddt_ca` |"


@dataclass(frozen=True)
class BestResult:
    label: str
    display_label: str
    step: int
    lddt_ca: float


def _cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def _display_label(label: str) -> str:
    match = re.search(r"\bE\d+[A-Za-z]?\b", label)
    return match.group(0) if match else label


def _parse_best_row(row: str) -> BestResult | None:
    cells = _cells(row)
    if len(cells) < 4:
        return None
    try:
        step = int(cells[2])
        lddt_ca = float(cells[3])
    except ValueError:
        return None
    return BestResult(label=cells[0], display_label=_display_label(cells[0]), step=step, lddt_ca=lddt_ca)


def best_result_from_table(markdown: str) -> BestResult:
    lines = markdown.splitlines()
    header_index = next((index for index, line in enumerate(lines) if line.startswith(TABLE_HEADER)), None)
    if header_index is None:
        raise ValueError("Could not find EXPERIMENT_RESULTS.md table header")
    table_index = header_index + 2
    candidates = []
    while table_index < len(lines) and lines[table_index].startswith("|"):
        parsed = _parse_best_row(lines[table_index])
        if parsed is not None:
            candidates.append(parsed)
        table_index += 1
    if not candidates:
        raise ValueError("Could not find any numeric best validation lDDT rows")
    return max(candidates, key=lambda item: item.lddt_ca)


def _target_sentence(best: BestResult, *, target: float, confirmation_steps: int) -> str:
    if best.lddt_ca > target and best.step >= confirmation_steps:
        return f"The `{confirmation_steps}`-step confirmation target has been met."
    if best.lddt_ca > target:
        return (
            f"The validation threshold exceeds `{target:.1f}` in a short gate, but the "
            f"{confirmation_steps:,}-step confirmation remains pending."
        )
    return f"The target remains `val_lddt_ca > {target:.1f}`, so the goal is not yet met."


def refresh_summary(
    markdown: str,
    *,
    updated_date: str,
    target: float = 0.7,
    confirmation_steps: int = 30_000,
) -> str:
    best = best_result_from_table(markdown)
    lines = markdown.splitlines()

    date_index = next((index for index, line in enumerate(lines) if line.startswith("Last updated: ")), None)
    if date_index is None:
        raise ValueError("Could not find Last updated line")
    lines[date_index] = f"Last updated: {updated_date}."

    best_index = next(
        (index for index, line in enumerate(lines) if line.startswith("Best validation C-alpha lDDT so far: ")),
        None,
    )
    if best_index is None:
        raise ValueError("Could not find best validation summary line")
    summary_end = best_index + 1
    while summary_end < len(lines) and "This file records returned Runpod results" not in lines[summary_end]:
        summary_end += 1
    if summary_end >= len(lines):
        raise ValueError("Could not find end of best validation summary block")
    replacement = [
        (
            f"Best validation C-alpha lDDT so far: **{best.display_label}**, "
            f"`val_lddt_ca={best.lddt_ca:.4f}` at step"
        ),
        f"{best.step}. {_target_sentence(best, target=target, confirmation_steps=confirmation_steps)}",
        "",
    ]
    lines[best_index:summary_end] = replacement

    trailing_newline = "\n" if markdown.endswith("\n") else ""
    return "\n".join(lines) + trailing_newline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_md", type=Path)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--target", type=float, default=0.7)
    parser.add_argument("--confirmation-steps", type=int, default=30_000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> str:
    args = parse_args(argv)
    updated = refresh_summary(
        args.results_md.read_text(encoding="utf-8"),
        updated_date=args.date,
        target=args.target,
        confirmation_steps=args.confirmation_steps,
    )
    args.results_md.write_text(updated, encoding="utf-8")
    return updated


if __name__ == "__main__":
    main()
