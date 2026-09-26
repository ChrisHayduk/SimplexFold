#!/usr/bin/env python3
"""Research-v2 goal successor; use verified run directories and a separate JSON ledger."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minalphafold.reporting import main as reporting_main


def main(argv=None):
    return reporting_main("goal", argv)


if __name__ == "__main__":
    report = main()
    raise SystemExit(0 if report["single_run_candidate"] else 1)
