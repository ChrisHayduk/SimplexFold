import csv
import json

from scripts.record_experiment_result import main, record_experiment_result


RESULTS_MD = """# SimplexFold Experiment Results

Last updated: 2026-05-14.

Best validation C-alpha lDDT so far: **E128**, `val_lddt_ca=0.4311` at step
8500. The target remains `val_lddt_ca > 0.7`, so the goal is not yet met.

This file records returned Runpod results and terminal stopped-run outcomes.

| Run | Status | Best step | Best `val_lddt_ca` | Final/stop `val_lddt_ca` | Final/stop FoldScore | Final/stop `val_ca_drmsd` | Final/stop C-alpha Rg | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| E128 damped triangle-attention bias from E124 | returned | 8500 | 0.4311 | 0.4311 | 0.4025 | 11.0046 | 11.7198 / 16.3091 | current best |
"""


def _write_run(run_dir):
    variant = "full_msa_to_face"
    run_dir.mkdir()
    result = {
        "variant": variant,
        "completed_steps": 9000,
        "val_lddt_ca": 0.4401,
        "val_foldscore": 0.4102,
        "val_ca_drmsd": 10.9,
        "val_ca_pred_rg": 12.1,
        "val_ca_true_rg": 16.3091,
    }
    (run_dir / "results.json").write_text(json.dumps([result]), encoding="utf-8")
    (run_dir / f"history_{variant}.json").write_text(
        json.dumps(
            [
                {"step": 8500, "val_lddt_ca": 0.4311},
                {"step": 9000, "val_lddt_ca": 0.4401},
            ]
        ),
        encoding="utf-8",
    )
    with (run_dir / f"eval_details_{variant}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["chain_id", "val_lddt_ca"])
        writer.writeheader()
        writer.writerow({"chain_id": "not_read_by_this_helper", "val_lddt_ca": 0.44})


def test_record_experiment_result_formats_upserts_and_refreshes_summary(tmp_path):
    results_md = tmp_path / "EXPERIMENT_RESULTS.md"
    results_md.write_text(RESULTS_MD, encoding="utf-8")
    run_dir = tmp_path / "run"
    _write_run(run_dir)

    row = record_experiment_result(
        results_md=results_md,
        run_dir=run_dir,
        run_label="E140 selected-boundary realization anti-collapse",
        status="returned",
        decision="pending decision",
        start_after_step=8500,
        updated_date="2026-05-15",
    )
    updated = results_md.read_text(encoding="utf-8")

    assert row in updated
    assert "| E140 selected-boundary realization anti-collapse | returned | 9000 | 0.4401 |" in updated
    assert "Last updated: 2026-05-15." in updated
    assert "Best validation C-alpha lDDT so far: **E140**, `val_lddt_ca=0.4401`" in updated


def test_main_prints_recorded_row(tmp_path, capsys):
    results_md = tmp_path / "EXPERIMENT_RESULTS.md"
    results_md.write_text(RESULTS_MD, encoding="utf-8")
    run_dir = tmp_path / "run"
    _write_run(run_dir)

    main(
        [
            str(run_dir),
            "--results-md",
            str(results_md),
            "--run-label",
            "E140 selected-boundary realization anti-collapse",
            "--decision",
            "pending decision",
            "--start-after-step",
            "8500",
            "--date",
            "2026-05-15",
        ]
    )

    assert "E140 selected-boundary realization anti-collapse" in capsys.readouterr().out
