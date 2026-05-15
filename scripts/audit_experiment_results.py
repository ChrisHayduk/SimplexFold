#!/usr/bin/env python3
"""Audit EXPERIMENT_RESULTS.md against the SimplexFold NanoFold target."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re


TABLE_HEADER = "| Run | Status | Best step | Best `val_lddt_ca` |"


@dataclass(frozen=True)
class ExperimentResult:
    label: str
    status: str
    best_step: int | None
    best_lddt_ca: float | None
    final_lddt_ca: float | None
    foldscore: float | None
    ca_drmsd: float | None
    ca_rg: str
    decision: str

    @property
    def display_label(self) -> str:
        match = re.search(r"\bE\d+[A-Za-z]?\b", self.label)
        return match.group(0) if match else self.label


def _cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def _parse_optional_float(value: str) -> float | None:
    if value == "-" or not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_optional_int(value: str) -> int | None:
    if value == "-" or not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_experiment_results(markdown: str) -> list[ExperimentResult]:
    lines = markdown.splitlines()
    header_index = next((index for index, line in enumerate(lines) if line.startswith(TABLE_HEADER)), None)
    if header_index is None:
        raise ValueError("Could not find EXPERIMENT_RESULTS.md table header")

    rows = []
    table_index = header_index + 2
    while table_index < len(lines) and lines[table_index].startswith("|"):
        cells = _cells(lines[table_index])
        if len(cells) >= 9:
            rows.append(
                ExperimentResult(
                    label=cells[0],
                    status=cells[1],
                    best_step=_parse_optional_int(cells[2]),
                    best_lddt_ca=_parse_optional_float(cells[3]),
                    final_lddt_ca=_parse_optional_float(cells[4]),
                    foldscore=_parse_optional_float(cells[5]),
                    ca_drmsd=_parse_optional_float(cells[6]),
                    ca_rg=cells[7],
                    decision=cells[8],
                )
            )
        table_index += 1
    return rows


def _numeric_best_rows(rows: list[ExperimentResult]) -> list[ExperimentResult]:
    return [row for row in rows if row.best_lddt_ca is not None]


def best_result(rows: list[ExperimentResult]) -> ExperimentResult:
    candidates = _numeric_best_rows(rows)
    if not candidates:
        raise ValueError("No rows with numeric best validation C-alpha lDDT")
    return max(candidates, key=lambda row: row.best_lddt_ca or float("-inf"))


def confirmed_results(
    rows: list[ExperimentResult],
    *,
    target: float,
    confirmation_steps: int,
) -> list[ExperimentResult]:
    return [
        row
        for row in rows
        if row.best_step is not None
        and row.best_step >= confirmation_steps
        and row.final_lddt_ca is not None
        and row.final_lddt_ca > target
    ]


def short_gate_results(
    rows: list[ExperimentResult],
    *,
    short_gate_threshold: float,
    confirmation_steps: int,
) -> list[ExperimentResult]:
    return [
        row
        for row in rows
        if row.best_step is not None
        and row.best_step < confirmation_steps
        and row.best_lddt_ca is not None
        and row.best_lddt_ca >= short_gate_threshold
    ]


def audit_experiment_results(
    markdown: str,
    *,
    target: float = 0.7,
    confirmation_steps: int = 30_000,
    short_gate_threshold: float = 0.45,
    top_n: int = 5,
) -> str:
    rows = parse_experiment_results(markdown)
    best = best_result(rows)
    confirmed = confirmed_results(rows, target=target, confirmation_steps=confirmation_steps)
    short_gate = short_gate_results(rows, short_gate_threshold=short_gate_threshold, confirmation_steps=confirmation_steps)
    top = sorted(_numeric_best_rows(rows), key=lambda row: row.best_lddt_ca or float("-inf"), reverse=True)[:top_n]

    lines = [
        (
            f"Best returned score: {best.display_label} "
            f"`val_lddt_ca={best.best_lddt_ca:.4f}` at step {best.best_step}."
        ),
        (
            f"30k confirmation: {len(confirmed)} run(s) above `{target:.1f}` "
            f"at >= {confirmation_steps:,} steps."
        ),
        (
            f"Short gates >= `{short_gate_threshold:.2f}` before {confirmation_steps:,} steps: "
            f"{len(short_gate)}."
        ),
        f"Top {len(top)} numeric rows:",
    ]
    for row in top:
        final = "-" if row.final_lddt_ca is None else f"{row.final_lddt_ca:.4f}"
        lines.append(
            f"- {row.display_label}: best `{row.best_lddt_ca:.4f}` at step {row.best_step}; "
            f"final/stop `{final}`; status `{row.status}`"
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_md", type=Path, nargs="?", default=Path("EXPERIMENT_RESULTS.md"))
    parser.add_argument("--target", type=float, default=0.7)
    parser.add_argument("--confirmation-steps", type=int, default=30_000)
    parser.add_argument("--short-gate-threshold", type=float, default=0.45)
    parser.add_argument("--top-n", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> str:
    args = parse_args(argv)
    summary = audit_experiment_results(
        args.results_md.read_text(encoding="utf-8"),
        target=args.target,
        confirmation_steps=args.confirmation_steps,
        short_gate_threshold=args.short_gate_threshold,
        top_n=args.top_n,
    )
    print(summary)
    return summary


if __name__ == "__main__":
    main()
