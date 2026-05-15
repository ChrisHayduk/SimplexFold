import csv
import json

import pytest

from scripts.audit_goal_artifact import audit_goal_artifact, format_goal_audit, main


def _write_run(run_dir, *, completed_steps=30_000, val_lddt_ca=0.705, parameters=3_240_738):
    variant = "full_msa_to_face"
    run_dir.mkdir()
    (run_dir / "checkpoints").mkdir()
    (run_dir / "checkpoints" / f"{variant}_latest.pt").write_bytes(b"checkpoint")
    row = {
        "variant": variant,
        "completed_steps": completed_steps,
        "effective_batch_size": 8,
        "num_workers": 4,
        "parameters": parameters,
        "stopped_early": False,
        "val_lddt_ca": val_lddt_ca,
        "val_foldscore": 0.62,
        "val_ca_drmsd": 7.1,
    }
    (run_dir / "results.json").write_text(json.dumps([row]), encoding="utf-8")
    (run_dir / "results.csv").write_text(
        f"variant,completed_steps\n{variant},{completed_steps}\n",
        encoding="utf-8",
    )
    history = [
        {"step": completed_steps - 500, "val_lddt_ca": max(0.0, val_lddt_ca - 0.01)},
        {"step": completed_steps, "val_lddt_ca": val_lddt_ca},
    ]
    (run_dir / f"history_{variant}.json").write_text(json.dumps(history), encoding="utf-8")
    with (run_dir / f"eval_details_{variant}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["chain_id", "val_lddt_ca"])
        writer.writeheader()
        writer.writerow({"chain_id": "chain_0", "val_lddt_ca": val_lddt_ca})
    metadata = {
        "effective_batch_size": 8,
        "num_workers": 4,
        "run_name": "test_goal_artifact",
        "simplex_outer_edge_residual_context_scale": 0.25,
        "disabled_gate": None,
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    status = {
        "variant": variant,
        "completed_step": completed_steps,
        "target_steps": completed_steps,
        "effective_batch_size": 8,
        "num_workers": 4,
        "stopped_early": False,
    }
    (run_dir / f"status_{variant}.json").write_text(json.dumps(status), encoding="utf-8")


def test_goal_artifact_audit_accepts_confirmed_candidate(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir)

    audit = audit_goal_artifact(
        run_dir,
        expected_completed_steps=30_000,
        expected_effective_batch_size=8,
        expected_num_workers=4,
        max_parameters=3_261_974,
        expected_results_rows=1,
        expected_eval_rows=1,
        expected_history_last_step=30_000,
        expect_stopped_early=False,
    )

    assert audit.target_pass is True
    assert audit.confirmation_step_pass is True
    assert audit.stopped_early_pass is True
    assert audit.goal_ready is True


def test_goal_artifact_audit_rejects_short_gate_even_above_target(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir, completed_steps=9000, val_lddt_ca=0.71)

    audit = audit_goal_artifact(
        run_dir,
        max_parameters=3_261_974,
        expected_results_rows=1,
        expected_eval_rows=1,
    )

    assert audit.target_pass is True
    assert audit.confirmation_step_pass is False
    assert audit.goal_ready is False


def test_goal_artifact_audit_rejects_below_target_confirmation(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir, completed_steps=30_000, val_lddt_ca=0.699)

    audit = audit_goal_artifact(run_dir, max_parameters=3_261_974, expected_eval_rows=1)

    assert audit.target_pass is False
    assert audit.confirmation_step_pass is True
    assert audit.goal_ready is False


def test_goal_artifact_audit_uses_verifier_parameter_cap(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir, parameters=3_500_000)

    with pytest.raises(ValueError, match="exceeds max"):
        audit_goal_artifact(run_dir, max_parameters=3_261_974)


def test_goal_artifact_audit_accepts_metadata_expectations(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir)

    audit = audit_goal_artifact(
        run_dir,
        max_parameters=3_261_974,
        metadata_expectations=[
            ("run_name", "test_goal_artifact"),
            ("simplex_outer_edge_residual_context_scale", 0.25),
            ("disabled_gate", None),
        ],
    )

    assert audit.goal_ready is True


def test_goal_artifact_audit_rejects_metadata_mismatch(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir)

    with pytest.raises(ValueError, match="metadata run_name"):
        audit_goal_artifact(
            run_dir,
            max_parameters=3_261_974,
            metadata_expectations=[("run_name", "wrong")],
        )


def test_format_goal_audit_mentions_all_completion_gates(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir, completed_steps=9000, val_lddt_ca=0.71)

    audit = audit_goal_artifact(run_dir, max_parameters=3_261_974, expected_eval_rows=1)
    text = format_goal_audit(audit)

    assert "Validation C-alpha lDDT: 0.7100 > 0.7: PASS" in text
    assert "Confirmation steps: 9000 >= 30,000: FAIL" in text
    assert "parameters=3240738 <= 3261974" in text
    assert "Goal-ready candidate: FAIL" in text


def test_main_accepts_metadata_and_can_emit_json(tmp_path, capsys):
    run_dir = tmp_path / "run"
    _write_run(run_dir)

    audit = main(
        [
            str(run_dir),
            "--max-parameters",
            "3261974",
            "--expected-eval-rows",
            "1",
            "--metadata",
            "run_name=test_goal_artifact",
            "--metadata",
            "disabled_gate=null",
            "--json",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert audit.goal_ready is True
    assert output["goal_ready"] is True
    assert output["verified"]["parameters"] == 3_240_738
