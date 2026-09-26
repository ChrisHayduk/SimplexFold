"""Verified research-v2 reporting, separate from historical experiment ledgers."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
from pathlib import Path

import torch

from .research import PROTOCOL_VERSION, fingerprint, verify_result

LEDGER_PROTOCOL = "simplexfold-research-v2-ledger"


def load_verified(run_dir, *, allow_paused=False):
    """Bind reportable metrics to a complete cohort and an exact saved checkpoint."""
    directory = Path(run_dir).resolve()
    result = verify_result(directory)
    checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=False)
    contract = checkpoint["contract"]
    if contract.get("protocol") != PROTOCOL_VERSION:
        raise ValueError("Reporting requires the current research protocol")
    if json.loads((directory / "contract.json").read_text()) != {
        "sha256": checkpoint["contract_sha256"], "contract": contract,
    }:
        raise ValueError("Saved contract differs from the checkpoint")
    if json.loads((directory / "history.json").read_text()) != checkpoint["history"]:
        raise ValueError("Saved history differs from the checkpoint")
    training = contract["training"]
    step, budget, seed = result["completed_steps"], training["epochs"], training["seed"]
    if any(type(value) is not int for value in (step, budget, seed)) or not 0 < step <= budget or seed < 0:
        raise ValueError("Invalid completed step, budget, or seed")
    if type(result["paused"]) is not bool or result["paused"] != (step < budget):
        raise ValueError("Completion state disagrees with the fixed update budget")
    if result["paused"] and not allow_paused:
        raise ValueError("A paused run is not a completed experiment result")
    if result["train_examples"] != step * training["batch_size"] * training["grad_accum_steps"]:
        raise ValueError("Training exposure does not match the update protocol")
    if result.get("foldscore_version") != contract["scorer"]["version"]:
        raise ValueError("Score version disagrees with the contract")
    if type(result.get("parameters")) is not int or result["parameters"] < 1:
        raise ValueError("Invalid parameter count")
    if checkpoint.get("parameters") != result["parameters"]:
        raise ValueError("Parameter count must be bound in the checkpoint")
    rows = json.loads((directory / "eval_details.json").read_text())
    return {
        "run_dir": str(directory), "protocol": contract["protocol"],
        "contract_sha256": checkpoint["contract_sha256"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "validation_sha256": fingerprint(contract["corpus"]["val"]),
        "scorer_sha256": fingerprint(contract["scorer"]),
        "source_sha256": fingerprint(contract["source"]),
        "seed": seed, "variant": contract["variant"],
        "model_profile": contract["model"]["model_profile"],
        "status": "paused" if result["paused"] else "complete",
        "result": result, "targets": rows,
    }


def comparable_runs(run_dirs, *, allow_paused=False):
    records = [load_verified(path, allow_paused=allow_paused) for path in run_dirs]
    if not records:
        raise ValueError("At least one verified run is required")
    if len({r["checkpoint_sha256"] for r in records}) != len(records):
        raise ValueError("The same checkpoint cannot count as multiple experiments")
    if len({(r["validation_sha256"], r["scorer_sha256"]) for r in records}) != 1:
        raise ValueError("Comparison requires identical complete validation data and scorer")
    return records


def write_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def compact(record):
    return {key: value for key, value in record.items() if key != "targets"}


def load_ledger(path):
    payload = json.loads(Path(path).read_text())
    if payload.get("protocol") != LEDGER_PROTOCOL or not isinstance(payload.get("runs"), list):
        raise ValueError("Use a separate research-v2 JSON ledger; historical Markdown is unchanged")
    entries = payload["runs"]
    labels = [row["label"] for row in entries]
    if any(not isinstance(label, str) or not label.strip() for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("Ledger labels must be nonempty and unique")
    records = comparable_runs([row["run_dir"] for row in entries]) if entries else []
    for saved, current in zip(entries, records):
        if {k: v for k, v in saved.items() if k != "label"} != compact(current):
            raise ValueError("Ledger entry no longer matches its verified checkpoint; record an explicit new result")
    return payload


def record_result(ledger, run_dir, label, *, replace_existing=False):
    """Update only a v2 ledger, under an exclusive lock and atomic replacement."""
    import fcntl

    if not isinstance(label, str) or not label.strip() or "\n" in label or "|" in label:
        raise ValueError("Run label must be nonempty single-line text without a table separator")
    path = Path(ledger)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        payload = load_ledger(path) if path.exists() else {"protocol": LEDGER_PROTOCOL, "runs": []}
        entry = {"label": label, **compact(load_verified(run_dir))}
        matching = [index for index, row in enumerate(payload["runs"]) if row["label"] == label]
        if matching and not replace_existing:
            raise ValueError("Label already exists; explicit upsert is required")
        if matching:
            payload["runs"][matching[0]] = entry
        else:
            payload["runs"].append(entry)
        comparable_runs([row["run_dir"] for row in payload["runs"]])
        fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return entry


def format_row(entry):
    result = entry["result"]
    return (f"| {entry.get('label', entry['variant'])} | {entry['protocol']} | {entry['seed']} | "
            f"{result['completed_steps']} | {result['train_examples']} | {result['val_examples']} | "
            f"{result['val_lddt_ca']:.6f} | {result['val_foldscore']:.6f} | "
            f"{entry['checkpoint_sha256']} |")


def ledger_summary(payload):
    lines = ["# Verified research-v2 runs", "", "Each row is one completed seed; no across-seed confirmation is inferred.", "",
             "| Run | Protocol | Seed | Updates | Train examples | Validation targets | CA lDDT | FoldScore | Checkpoint SHA256 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    return "\n".join(lines + [format_row(row) for row in payload["runs"]]) + "\n"


def audit_goal(record, *, target, confirmation_steps, max_parameters):
    if not math.isfinite(target) or not 0 <= target <= 1 or confirmation_steps < 1 or max_parameters < 1:
        raise ValueError("Invalid goal bounds")
    result = record["result"]
    checks = {"target": result["val_lddt_ca"] > target,
              "updates": result["completed_steps"] >= confirmation_steps,
              "parameters": result["parameters"] <= max_parameters,
              "complete": record["status"] == "complete"}
    return {"record": compact(record), "checks": checks, "single_run_candidate": all(checks.values()),
            "claim": "One verified seed meeting numeric bounds; this is not replicated confirmation"}


def analyze_targets(record):
    """Summarize all verified targets without dropping nonfinite or poor scores."""
    rows = record["targets"]
    summaries = {}
    for key in sorted(set(rows[0]) - {"chain_id"}):
        values = [float(row[key]) for row in rows]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Nonfinite target statistic: {key}")
        summaries[key] = {"count": len(values), "mean": statistics.fmean(values),
                          "min": min(values), "median": statistics.median(values), "max": max(values)}
    return {**compact(record), "target_summary": summaries,
            "targets_by_ca_lddt": sorted(rows, key=lambda row: (row["lddt_ca"], row["chain_id"]))}


def main(action, argv=None):
    parser = argparse.ArgumentParser(description="Verified research-v2 reporting. Historical artifacts remain separate.")
    if action in {"record", "upsert"}:
        parser.add_argument("run_dir", type=Path)
        parser.add_argument("--ledger", type=Path, required=True)
        parser.add_argument("--run-label", required=True)
    elif action in {"refresh", "audit_ledger"}:
        parser.add_argument("ledger", type=Path)
    else:
        parser.add_argument("run_dir", type=Path)
    if action == "goal":
        parser.add_argument("--target", type=float, default=0.7)
        parser.add_argument("--confirmation-steps", type=int, default=30000)
        parser.add_argument("--max-parameters", type=int, required=True)
    if action == "status":
        parser.add_argument("--allow-paused", action="store_true")
    if action == "refresh":
        parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if action in {"record", "upsert"}:
        result = record_result(args.ledger, args.run_dir, args.run_label, replace_existing=action == "upsert")
    elif action in {"refresh", "audit_ledger"}:
        result = load_ledger(args.ledger)
        if action == "refresh":
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(ledger_summary(result))
    else:
        record = load_verified(args.run_dir, allow_paused=getattr(args, "allow_paused", False))
        if action == "goal":
            result = audit_goal(record, target=args.target, confirmation_steps=args.confirmation_steps,
                                max_parameters=args.max_parameters)
        elif action == "format":
            print(format_row(record))
            return record
        elif action == "analyze":
            result = analyze_targets(record)
        else:
            result = compact(record)
    print(json.dumps(result, indent=2, allow_nan=False))
    return result
