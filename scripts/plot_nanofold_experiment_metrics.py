#!/usr/bin/env python3
"""Plot verified research-v2 runs on one complete cohort; each seed stays distinct."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minalphafold.reporting import compact, comparable_runs, write_new


def main(argv=None, *, checkpoint_curve=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--metric", choices=["val_lddt_ca", "val_foldscore"], default="val_foldscore")
    args = parser.parse_args(argv)
    records = comparable_runs(args.run_dirs, allow_paused=checkpoint_curve)
    labels = args.labels or [f"{r['variant']} seed={r['seed']} step={r['result']['completed_steps']}" for r in records]
    if len(labels) != len(records) or any(not label.strip() for label in labels):
        raise ValueError("Supply one nonempty label per run")
    if checkpoint_curve:
        for label in set(labels):
            group = [r for r, name in zip(records, labels) if name == label]
            if len({r["contract_sha256"] for r in group}) != 1:
                raise ValueError("A checkpoint curve must use one exact recipe/source/seed contract")
            if len({r["result"]["completed_steps"] for r in group}) != len(group):
                raise ValueError("A checkpoint curve cannot repeat an update")
    elif len(set(labels)) != len(labels):
        raise ValueError("Independent results need distinct plot labels")
    if args.output_dir.exists():
        raise FileExistsError("Choose a new plot output directory")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.output_dir.mkdir(parents=True)
    rows = [{"label": label, **compact(record)} for label, record in zip(labels, records)]
    write_new(args.output_dir / "provenance.json", {
        "protocol": "research-v2-verified-plot", "checkpoint_curve": checkpoint_curve,
        "metric": args.metric, "runs": rows,
        "claim": "Individual seeds and saved checkpoint snapshots; no across-seed mean or confirmation",
    })
    with (args.output_dir / "values.csv").open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "seed", "updates", "train_examples", args.metric, "checkpoint_sha256"])
        for row in rows:
            result = row["result"]
            writer.writerow([row["label"], row["seed"], result["completed_steps"], result["train_examples"],
                             result[args.metric], row["checkpoint_sha256"]])
    figure, axis = plt.subplots(figsize=(max(6, len(records) * 0.75), 4))
    try:
        if checkpoint_curve:
            for label in dict.fromkeys(labels):
                points = sorted((r for r, name in zip(records, labels) if name == label),
                                key=lambda r: r["result"]["train_examples"])
                axis.plot([r["result"]["train_examples"] for r in points],
                          [r["result"][args.metric] for r in points], marker="o",
                          label=f"{label} (seed {points[0]['seed']})")
            axis.set_xlabel("Training examples seen")
            axis.legend()
        else:
            axis.scatter(range(len(records)), [r["result"][args.metric] for r in records])
            axis.set_xticks(range(len(records)), labels, rotation=25, ha="right")
            axis.set_xlabel("Verified completed runs (individual seeds)")
        axis.set_ylabel(args.metric.removeprefix("val_"))
        axis.set_title("Research v2 · identical complete validation cohort")
        axis.set_ylim(0, 1)
        figure.tight_layout()
        figure.savefig(args.output_dir / "metrics.svg")
        figure.savefig(args.output_dir / "metrics.png", dpi=180)
    finally:
        plt.close(figure)
    return rows


if __name__ == "__main__":
    main()
