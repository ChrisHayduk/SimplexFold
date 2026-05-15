import csv
import json

import pytest

from scripts.verify_nanofold_benchmark_artifacts import main, verify_artifacts


def _write_run(run_dir, *, eval_rows=3, metadata=None, result=None):
    variant = "full_msa_to_face"
    run_dir.mkdir()
    (run_dir / "checkpoints").mkdir()
    (run_dir / "checkpoints" / f"{variant}_latest.pt").write_bytes(b"checkpoint")
    row = {
        "variant": variant,
        "completed_steps": 8000,
        "effective_batch_size": 8,
        "num_workers": 4,
        "parameters": 3_201_970,
        "stopped_early": False,
        "val_lddt_ca": 0.51,
        "val_foldscore": 0.44,
        "val_ca_drmsd": 9.25,
    }
    if result:
        row.update(result)
    (run_dir / "results.json").write_text(json.dumps([row]), encoding="utf-8")
    (run_dir / "results.csv").write_text("variant,completed_steps\nfull_msa_to_face,8000\n", encoding="utf-8")
    history = [
        {"step": 7500, "val_lddt_ca": 0.4248},
        {"step": 8000, "val_lddt_ca": row["val_lddt_ca"]},
    ]
    (run_dir / f"history_{variant}.json").write_text(json.dumps(history), encoding="utf-8")
    with (run_dir / f"eval_details_{variant}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["chain_id", "val_lddt_ca"])
        writer.writeheader()
        for index in range(eval_rows):
            writer.writerow({"chain_id": f"chain_{index}", "val_lddt_ca": 0.5})
    meta = {
        "effective_batch_size": 8,
        "num_workers": 4,
        "simplex_global_context_scale": 0.1,
        "simplex_face_top_k": 24,
        "simplex_tetra_top_k": 48,
        "simplex_disabled_gate": None,
    }
    if metadata:
        meta.update(metadata)
    (run_dir / "run_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    status = {
        "variant": variant,
        "completed_step": row["completed_steps"],
        "target_steps": row["completed_steps"],
        "effective_batch_size": meta["effective_batch_size"],
        "num_workers": meta["num_workers"],
        "stopped_early": row["stopped_early"],
    }
    (run_dir / f"status_{variant}.json").write_text(json.dumps(status), encoding="utf-8")


def test_verify_artifacts_accepts_coherent_run(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir)

    summary = verify_artifacts(
        run_dir,
        expected_completed_steps=8000,
        expected_effective_batch_size=8,
        expected_num_workers=4,
        max_parameters=3_261_974,
        expect_stopped_early=False,
        expected_results_rows=1,
        expected_eval_rows=3,
        expected_history_last_step=8000,
        metadata_expectations=[
            ("simplex_global_context_scale", 0.1),
            ("simplex_face_top_k", 24),
        ],
    )

    assert summary["val_lddt_ca"] == 0.51
    assert summary["parameters"] == 3_201_970
    assert summary["num_workers"] == 4
    assert summary["results_rows"] == 1


def test_verify_artifacts_rejects_wrong_eval_row_count(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir, eval_rows=2)

    with pytest.raises(ValueError, match="eval rows=2, expected 3"):
        verify_artifacts(run_dir, expected_eval_rows=3)


def test_verify_artifacts_rejects_wrong_results_csv_row_count(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir)
    (run_dir / "results.csv").write_text(
        "variant,completed_steps\nfull_msa_to_face,8000\nother_variant,8000\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="results.csv rows=2, expected 1"):
        verify_artifacts(run_dir, expected_results_rows=1)


def test_verify_artifacts_rejects_missing_results_csv_variant(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir)
    (run_dir / "results.csv").write_text("variant,completed_steps\nother_variant,8000\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected exactly one results.csv row"):
        verify_artifacts(run_dir)


def test_main_rejects_metadata_mismatch(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir, metadata={"simplex_face_top_k": 12})

    with pytest.raises(ValueError, match="metadata simplex_face_top_k=12"):
        main([str(run_dir), "--metadata", "simplex_face_top_k=24"])


def test_verify_artifacts_rejects_wrong_num_workers(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir, metadata={"num_workers": 0}, result={"num_workers": 0})

    with pytest.raises(ValueError, match="num_workers=0, expected 4"):
        verify_artifacts(run_dir, expected_num_workers=4)


def test_verify_artifacts_rejects_status_effective_batch_mismatch(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir)
    variant = "full_msa_to_face"
    status_path = run_dir / f"status_{variant}.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["effective_batch_size"] = 4
    status_path.write_text(json.dumps(status), encoding="utf-8")

    with pytest.raises(ValueError, match="status effective_batch_size=4"):
        verify_artifacts(run_dir, expected_effective_batch_size=8)


def test_main_accepts_expected_num_workers_flag(tmp_path, capsys):
    run_dir = tmp_path / "run"
    _write_run(run_dir)

    summary = main([str(run_dir), "--expected-num-workers", "4"])
    output = capsys.readouterr().out

    assert summary["num_workers"] == 4
    assert json.loads(output)["num_workers"] == 4


def test_main_accepts_null_metadata_expectation(tmp_path, capsys):
    run_dir = tmp_path / "run"
    _write_run(run_dir)

    summary = main([str(run_dir), "--metadata", "simplex_disabled_gate=null"])
    output = capsys.readouterr().out

    assert summary["run_dir"] == str(run_dir.resolve())
    assert json.loads(output)["run_dir"] == str(run_dir.resolve())
