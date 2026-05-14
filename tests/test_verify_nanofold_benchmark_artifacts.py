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
        "simplex_global_context_scale": 0.1,
        "simplex_face_top_k": 24,
        "simplex_tetra_top_k": 48,
    }
    if metadata:
        meta.update(metadata)
    (run_dir / "run_metadata.json").write_text(json.dumps(meta), encoding="utf-8")


def test_verify_artifacts_accepts_coherent_run(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir)

    summary = verify_artifacts(
        run_dir,
        expected_completed_steps=8000,
        expected_effective_batch_size=8,
        max_parameters=3_261_974,
        expect_stopped_early=False,
        expected_eval_rows=3,
        expected_history_last_step=8000,
        metadata_expectations=[
            ("simplex_global_context_scale", 0.1),
            ("simplex_face_top_k", 24),
        ],
    )

    assert summary["val_lddt_ca"] == 0.51
    assert summary["parameters"] == 3_201_970


def test_verify_artifacts_rejects_wrong_eval_row_count(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir, eval_rows=2)

    with pytest.raises(ValueError, match="eval rows=2, expected 3"):
        verify_artifacts(run_dir, expected_eval_rows=3)


def test_main_rejects_metadata_mismatch(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir, metadata={"simplex_face_top_k": 12})

    with pytest.raises(ValueError, match="metadata simplex_face_top_k=12"):
        main([str(run_dir), "--metadata", "simplex_face_top_k=24"])
