from scripts.audit_experiment_results import (
    audit_experiment_results,
    best_result,
    confirmed_results,
    parse_experiment_results,
    short_gate_results,
)


RESULTS_MD = """# SimplexFold Experiment Results

| Run | Status | Best step | Best `val_lddt_ca` | Final/stop `val_lddt_ca` | Final/stop FoldScore | Final/stop `val_ca_drmsd` | Final/stop C-alpha Rg | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| E128 damped triangle-attention bias from E124 | returned | 8500 | 0.4311 | 0.4311 | 0.4025 | 11.0046 | 11.7198 / 16.3091 | current best |
| E140 selected-boundary realization anti-collapse | returned | 9000 | 0.4520 | 0.4520 | 0.4100 | 10.9000 | 12.1000 / 16.3091 | short-gate keep |
| E141 signed face-cyclic boundary readout | stopped pre-eval | - | - | - | - | - | - | no scored result |
| E200 full confirmation | returned | 30000 | 0.7100 | 0.7050 | 0.6200 | 7.1000 | 15.9000 / 16.3091 | confirmed |
"""


def test_parse_experiment_results_keeps_numeric_and_stopped_rows():
    rows = parse_experiment_results(RESULTS_MD)

    assert len(rows) == 4
    assert rows[0].display_label == "E128"
    assert rows[2].best_lddt_ca is None


def test_best_result_uses_numeric_best_lddt_ca():
    best = best_result(parse_experiment_results(RESULTS_MD))

    assert best.display_label == "E200"
    assert best.best_step == 30000
    assert best.best_lddt_ca == 0.71


def test_confirmation_requires_final_score_above_target_at_confirmation_steps():
    rows = parse_experiment_results(RESULTS_MD)

    confirmed = confirmed_results(rows, target=0.7, confirmation_steps=30_000)

    assert [row.display_label for row in confirmed] == ["E200"]


def test_short_gate_results_exclude_confirmation_runs():
    rows = parse_experiment_results(RESULTS_MD)

    short_gate = short_gate_results(rows, short_gate_threshold=0.45, confirmation_steps=30_000)

    assert [row.display_label for row in short_gate] == ["E140"]


def test_audit_experiment_results_summarizes_target_state():
    summary = audit_experiment_results(RESULTS_MD, top_n=2)

    assert "Best returned score: E200 `val_lddt_ca=0.7100` at step 30000." in summary
    assert "30k confirmation: 1 run(s) above `0.7` at >= 30,000 steps." in summary
    assert "Short gates >= `0.45` before 30,000 steps: 1." in summary
    assert "Parameter-budget evidence: not present in EXPERIMENT_RESULTS.md" in summary
    assert "- E200: best `0.7100` at step 30000; final/stop `0.7050`; status `returned`" in summary
