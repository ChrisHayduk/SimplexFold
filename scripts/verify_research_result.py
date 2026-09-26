#!/usr/bin/env python3
"""Verify research-v2 results before plotting or recording them in a ledger."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minalphafold.research import verify_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_result(args.output_dir), indent=2))
