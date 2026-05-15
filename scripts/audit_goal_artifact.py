#!/usr/bin/env python3
"""Audit one verified NanoFold artifact directory against the SimplexFold goal."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verify_nanofold_benchmark_artifacts import verify_artifacts


@dataclass(frozen=True)
class GoalArtifactAudit:
    verified: dict[str, Any]
    target: float
    confirmation_steps: int
    max_parameters: int
    target_pass: bool
    confirmation_step_pass: bool
    stopped_early_pass: bool
    goal_ready: bool


def audit_goal_artifact(
    run_dir: Path,
    *,
    variant: str = "full_msa_to_face",
    target: float = 0.7,
    confirmation_steps: int = 30_000,
    expected_completed_steps: int | None = None,
    expected_effective_batch_size: int | None = None,
    expected_num_workers: int | None = None,
    max_parameters: int,
    expected_results_rows: int | None = None,
    expected_eval_rows: int | None = None,
    expected_history_last_step: int | None = None,
    expect_stopped_early: bool | None = None,
    require_checkpoint: bool = True,
) -> GoalArtifactAudit:
    verified = verify_artifacts(
        run_dir,
        variant=variant,
        expected_completed_steps=expected_completed_steps,
        expected_effective_batch_size=expected_effective_batch_size,
        expected_num_workers=expected_num_workers,
        max_parameters=max_parameters,
        expect_stopped_early=expect_stopped_early,
        expected_results_rows=expected_results_rows,
        expected_eval_rows=expected_eval_rows,
        expected_history_last_step=expected_history_last_step,
        require_checkpoint=require_checkpoint,
    )
    target_pass = float(verified["val_lddt_ca"]) > float(target)
    confirmation_step_pass = int(verified["completed_steps"]) >= int(confirmation_steps)
    stopped_early_pass = not bool(verified["stopped_early"])
    return GoalArtifactAudit(
        verified=verified,
        target=target,
        confirmation_steps=confirmation_steps,
        max_parameters=max_parameters,
        target_pass=target_pass,
        confirmation_step_pass=confirmation_step_pass,
        stopped_early_pass=stopped_early_pass,
        goal_ready=target_pass and confirmation_step_pass and stopped_early_pass,
    )


def _format_bool(value: bool) -> str:
    return "PASS" if value else "FAIL"


def format_goal_audit(audit: GoalArtifactAudit) -> str:
    verified = audit.verified
    lines = [
        f"Artifact: {verified['run_dir']}",
        (
            f"Verifier passed: completed_steps={verified['completed_steps']}, "
            f"effective_batch_size={verified['effective_batch_size']}, "
            f"parameters={verified['parameters']} <= {audit.max_parameters}, "
            f"eval_rows={verified['eval_rows']}, checkpoint required."
        ),
        f"Validation C-alpha lDDT: {verified['val_lddt_ca']:.4f} > {audit.target:.1f}: {_format_bool(audit.target_pass)}",
        (
            f"Confirmation steps: {verified['completed_steps']} >= "
            f"{audit.confirmation_steps:,}: {_format_bool(audit.confirmation_step_pass)}"
        ),
        f"Stopped early: {verified['stopped_early']} -> {_format_bool(audit.stopped_early_pass)}",
        f"Goal-ready candidate: {_format_bool(audit.goal_ready)}",
    ]
    return "\n".join(lines)


def _audit_as_dict(audit: GoalArtifactAudit) -> dict[str, Any]:
    return {
        "verified": audit.verified,
        "target": audit.target,
        "confirmation_steps": audit.confirmation_steps,
        "max_parameters": audit.max_parameters,
        "target_pass": audit.target_pass,
        "confirmation_step_pass": audit.confirmation_step_pass,
        "stopped_early_pass": audit.stopped_early_pass,
        "goal_ready": audit.goal_ready,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--variant", default="full_msa_to_face")
    parser.add_argument("--target", type=float, default=0.7)
    parser.add_argument("--confirmation-steps", type=int, default=30_000)
    parser.add_argument("--expected-completed-steps", type=int)
    parser.add_argument("--expected-effective-batch-size", type=int)
    parser.add_argument("--expected-num-workers", type=int)
    parser.add_argument("--max-parameters", type=int, required=True)
    parser.add_argument("--expected-results-rows", type=int)
    parser.add_argument("--expected-eval-rows", type=int)
    parser.add_argument("--expected-history-last-step", type=int)
    parser.add_argument("--expect-stopped-early", choices=("true", "false"))
    parser.add_argument("--allow-missing-checkpoint", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> GoalArtifactAudit:
    args = parse_args(argv)
    stopped_early = None
    if args.expect_stopped_early is not None:
        stopped_early = args.expect_stopped_early == "true"
    audit = audit_goal_artifact(
        args.run_dir,
        variant=args.variant,
        target=args.target,
        confirmation_steps=args.confirmation_steps,
        expected_completed_steps=args.expected_completed_steps,
        expected_effective_batch_size=args.expected_effective_batch_size,
        expected_num_workers=args.expected_num_workers,
        max_parameters=args.max_parameters,
        expected_results_rows=args.expected_results_rows,
        expected_eval_rows=args.expected_eval_rows,
        expected_history_last_step=args.expected_history_last_step,
        expect_stopped_early=stopped_early,
        require_checkpoint=not args.allow_missing_checkpoint,
    )
    if args.json:
        print(json.dumps(_audit_as_dict(audit), indent=2, sort_keys=True))
    else:
        print(format_goal_audit(audit))
    return audit


if __name__ == "__main__":
    main()
