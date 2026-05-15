from scripts.refresh_experiment_results_summary import best_result_from_table, main, refresh_summary


RESULTS_MD = """# SimplexFold Experiment Results

Last updated: 2026-05-14.

Best validation C-alpha lDDT so far: **E128**, `val_lddt_ca=0.4311` at step
8500. The target remains `val_lddt_ca > 0.7`, so the goal is not yet met.

This file records returned Runpod results and terminal stopped-run outcomes.

| Run | Status | Best step | Best `val_lddt_ca` | Final/stop `val_lddt_ca` | Final/stop FoldScore | Final/stop `val_ca_drmsd` | Final/stop C-alpha Rg | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| E128 damped triangle-attention bias from E124 | returned | 8500 | 0.4311 | 0.4311 | 0.4025 | 11.0046 | 11.7198 / 16.3091 | current best |
| E140 selected-boundary realization anti-collapse | returned | 9000 | 0.4400 | 0.4400 | 0.4100 | 10.9000 | 12.1000 / 16.3091 | pending |
| E139 no-Hodge oriented boundary cochain | stopped pre-eval | - | - | - | - | - | - | no scored result |
"""


def test_best_result_from_table_ignores_non_numeric_rows():
    best = best_result_from_table(RESULTS_MD)

    assert best.display_label == "E140"
    assert best.step == 9000
    assert best.lddt_ca == 0.44


def test_refresh_summary_updates_date_and_best_block():
    updated = refresh_summary(RESULTS_MD, updated_date="2026-05-15")

    assert "Last updated: 2026-05-15." in updated
    assert "Best validation C-alpha lDDT so far: **E140**, `val_lddt_ca=0.4400` at step\n9000." in updated
    assert "The target remains `val_lddt_ca > 0.7`, so the goal is not yet met." in updated


def test_refresh_summary_notes_short_gate_above_threshold():
    markdown = RESULTS_MD.replace(
        "| E140 selected-boundary realization anti-collapse | returned | 9000 | 0.4400 |",
        "| E140 selected-boundary realization anti-collapse | returned | 9000 | 0.7100 |",
    )

    updated = refresh_summary(markdown, updated_date="2026-05-15")

    assert "Best validation C-alpha lDDT so far: **E140**, `val_lddt_ca=0.7100` at step\n9000." in updated
    assert "30,000-step confirmation remains pending" in updated


def test_main_updates_results_file(tmp_path):
    results_md = tmp_path / "EXPERIMENT_RESULTS.md"
    results_md.write_text(RESULTS_MD, encoding="utf-8")

    main([str(results_md), "--date", "2026-05-15"])

    assert "Last updated: 2026-05-15." in results_md.read_text(encoding="utf-8")
