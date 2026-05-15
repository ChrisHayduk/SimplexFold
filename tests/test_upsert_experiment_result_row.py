from scripts.upsert_experiment_result_row import main, upsert_result_row


RESULTS_MD = """# SimplexFold Experiment Results

Last updated: 2026-05-15.

| Run | Status | Best step | Best `val_lddt_ca` | Final/stop `val_lddt_ca` | Final/stop FoldScore | Final/stop `val_ca_drmsd` | Final/stop C-alpha Rg | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| E128 damped triangle-attention bias from E124 | returned | 8500 | 0.4311 | 0.4311 | 0.4025 | 11.0046 | 11.7198 / 16.3091 | current best |
"""


def test_upsert_result_row_appends_missing_run():
    row = (
        "| E140 selected-boundary realization anti-collapse | returned | 9000 | "
        "0.4400 | 0.4400 | 0.4100 | 10.9000 | 12.1000 / 16.3091 | pending |"
    )

    updated = upsert_result_row(RESULTS_MD, row)

    assert updated.endswith(row + "\n")
    assert updated.count("| E140 selected-boundary realization anti-collapse |") == 1


def test_upsert_result_row_replaces_existing_run():
    old_row = (
        "| E140 selected-boundary realization anti-collapse | returned | 9000 | "
        "0.4400 | 0.4400 | 0.4100 | 10.9000 | 12.1000 / 16.3091 | pending |"
    )
    new_row = (
        "| E140 selected-boundary realization anti-collapse | returned | 9000 | "
        "0.4450 | 0.4450 | 0.4110 | 10.8000 | 12.3000 / 16.3091 | rejected |"
    )

    updated = upsert_result_row(RESULTS_MD + old_row + "\n", new_row)

    assert old_row not in updated
    assert new_row in updated
    assert updated.count("| E140 selected-boundary realization anti-collapse |") == 1


def test_main_updates_results_file(tmp_path):
    results_md = tmp_path / "EXPERIMENT_RESULTS.md"
    results_md.write_text(RESULTS_MD, encoding="utf-8")
    row = (
        "| E141 signed face-cyclic boundary readout | returned | 9000 | "
        "0.4390 | 0.4390 | 0.4090 | 10.9500 | 12.0000 / 16.3091 | pending |"
    )

    main([str(results_md), "--row", row])

    assert row in results_md.read_text(encoding="utf-8")
