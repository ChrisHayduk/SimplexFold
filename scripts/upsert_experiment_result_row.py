#!/usr/bin/env python3
"""Insert or replace one EXPERIMENT_RESULTS.md table row."""

from __future__ import annotations

import argparse
from pathlib import Path


TABLE_HEADER = "| Run | Status | Best step | Best `val_lddt_ca` |"


def _row_label(row: str) -> str:
    cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
    if len(cells) < 2 or not cells[0]:
        raise ValueError(f"Expected a Markdown table row with a non-empty run label, got: {row!r}")
    return cells[0]


def upsert_result_row(markdown: str, row: str) -> str:
    normalized_row = row.strip()
    label = _row_label(normalized_row)
    lines = markdown.splitlines()
    header_index = next((index for index, line in enumerate(lines) if line.startswith(TABLE_HEADER)), None)
    if header_index is None:
        raise ValueError("Could not find EXPERIMENT_RESULTS.md table header")

    table_start = header_index + 2
    table_end = table_start
    while table_end < len(lines) and lines[table_end].startswith("|"):
        table_end += 1

    for index in range(table_start, table_end):
        if _row_label(lines[index]) == label:
            lines[index] = normalized_row
            break
    else:
        lines.insert(table_end, normalized_row)

    trailing_newline = "\n" if markdown.endswith("\n") else ""
    return "\n".join(lines) + trailing_newline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_md", type=Path)
    parser.add_argument("--row", required=True, help="Complete Markdown table row to insert or replace.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> str:
    args = parse_args(argv)
    updated = upsert_result_row(args.results_md.read_text(encoding="utf-8"), args.row)
    args.results_md.write_text(updated, encoding="utf-8")
    return updated


if __name__ == "__main__":
    main()
