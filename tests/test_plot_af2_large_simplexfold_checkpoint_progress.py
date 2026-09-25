import json

import pytest
from scripts.plot_af2_large_simplexfold_checkpoint_progress import main


def _write_inputs(tmp_path):
    af2_path = tmp_path / "af2.json"
    af2_path.write_text(
        json.dumps(
            {
                "checkpoints": [
                    {
                        "step": 0,
                        "mean_foldscore": 0.1,
                        "mean_lddt_ca": 0.02,
                        "sample_budget_fraction": 0.0,
                        "cumulative_samples_seen": 0,
                    },
                    {
                        "step": 3000,
                        "mean_foldscore": 0.3,
                        "mean_lddt_ca": 0.2,
                        "sample_budget_fraction": 0.1,
                        "cumulative_samples_seen": 24000,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    history_path = tmp_path / "history.json"
    history_path.write_text(
        json.dumps(
            [
                {
                    "step": 0,
                    "val_foldscore": 0.09,
                    "val_lddt_ca": 0.02,
                    "val_loss": 12.0,
                    "val_ca_drmsd": 20.0,
                },
                {
                    "step": 3000,
                    "val_foldscore": 0.31,
                    "val_lddt_ca": 0.25,
                    "val_loss": 4.0,
                    "val_ca_drmsd": 10.0,
                },
            ]
        ),
        encoding="utf-8",
    )
    status_path = tmp_path / "status.json"
    status_path.write_text(json.dumps({"completed_step": 3000, "target_steps": 30000}), encoding="utf-8")
    metadata_path = tmp_path / "run_metadata.json"
    metadata_path.write_text(json.dumps({"msa_depth": 64}), encoding="utf-8")
    return af2_path, history_path, status_path, metadata_path


def test_writes_run_specific_e152_outputs(tmp_path):
    af2_path, history_path, status_path, metadata_path = _write_inputs(tmp_path)
    out_dir = tmp_path / "plots"

    written = main(
        [
            "--run-key",
            "e152",
            "--simplexfold-label",
            "E152 SimplexFold large",
            "--title",
            "E152 SimplexFold Large vs AF2-Large Checkpoint Eval",
            "--af2-checkpoints",
            str(af2_path),
            "--simplexfold-history",
            str(history_path),
            "--simplexfold-status",
            str(status_path),
            "--run-metadata",
            str(metadata_path),
            "--output-dir",
            str(out_dir),
            "--formats",
            "svg",
        ]
    )

    assert (out_dir / "e152-vs-af2-large_foldscore_lddt.svg").exists()
    assert (out_dir / "e152-vs-af2-large_foldscore_lddt.csv").exists()
    assert (out_dir / "e152-vs-af2-large_checkpoint_curve.csv").exists()
    metadata = json.loads((out_dir / "e152-vs-af2-large_metadata.json").read_text(encoding="utf-8"))
    assert metadata["simplexfold_run_key"] == "e152"
    assert metadata["run_metadata"]["msa_depth"] == 64
    assert written["prefix"].endswith("e152-vs-af2-large")


def test_writes_run_specific_e154_outputs(tmp_path):
    af2_path, history_path, status_path, metadata_path = _write_inputs(tmp_path)
    out_dir = tmp_path / "plots"

    written = main(
        [
            "--run-key",
            "e154",
            "--simplexfold-label",
            "E154 SimplexFold large",
            "--title",
            "E154 SimplexFold Large vs AF2-Large Checkpoint Eval",
            "--af2-checkpoints",
            str(af2_path),
            "--simplexfold-history",
            str(history_path),
            "--simplexfold-status",
            str(status_path),
            "--run-metadata",
            str(metadata_path),
            "--output-dir",
            str(out_dir),
            "--formats",
            "svg",
        ]
    )

    assert (out_dir / "e154-vs-af2-large_foldscore_lddt.svg").exists()
    assert (out_dir / "e154-vs-af2-large_foldscore_lddt.csv").exists()
    assert (out_dir / "e154-vs-af2-large_checkpoint_curve.csv").exists()
    metadata = json.loads((out_dir / "e154-vs-af2-large_metadata.json").read_text(encoding="utf-8"))
    assert metadata["simplexfold_run_key"] == "e154"
    assert metadata["run_metadata"]["msa_depth"] == 64
    assert written["prefix"].endswith("e154-vs-af2-large")


def test_legacy_alias_must_match_run_key(tmp_path):
    af2_path, history_path, status_path, _metadata_path = _write_inputs(tmp_path)

    with pytest.raises(ValueError, match="Refusing to write E153"):
        main(
            [
                "--run-key",
                "e153",
                "--simplexfold-label",
                "E153 SimplexFold large",
                "--title",
                "E153 SimplexFold Large vs AF2-Large Checkpoint Eval",
                "--af2-checkpoints",
                str(af2_path),
                "--simplexfold-history",
                str(history_path),
                "--simplexfold-status",
                str(status_path),
                "--output-dir",
                str(tmp_path / "plots"),
                "--formats",
                "svg",
                "--write-legacy-alias",
                "--legacy-prefix",
                "af2_large_vs_e152_simplexfold_large_foldscore_trace",
            ]
        )
