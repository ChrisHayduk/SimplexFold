import csv

from scripts.plot_nanofold_experiment_metrics import (
    LINEAGE_LABEL_OFFSETS,
    LINEAGE_PLOT_LABELS,
    parse_experiment_results,
    write_foldscore_experiment_csv,
)

RESULTS_MD = """# SimplexFold Experiment Results

| Run | Status | Best step | Best `val_lddt_ca` | Final/stop `val_lddt_ca` | Final/stop FoldScore | Final/stop `val_ca_drmsd` | Final/stop C-alpha Rg | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| main/full control | completed | 400 | 0.0992 | 0.0401 | 0.2024 | - | - | baseline |
| E55 effective-batch-8 aux 0.5 continuation | completed | 3000 | 0.3604 | 0.3604 | 0.3451 | 11.3280 | 10.0507 / 15.4034 | current best; branch checkpoint |
| E64 E63 confirmation to 4000 | completed | 4000 | 0.3739 | 0.3739 | 0.3634 | 10.5481 | 11.3344 / 15.4034 | current best |
| E123 ramped pair-only pre-triangle simplex injection | returned | 8000 | 0.4270 | 0.4270 | 0.3992 | 11.1927 | 11.4700 / 16.3091 | rejected |
| E128 damped triangle-attention bias from E124 | returned | 8500 | 0.4311 | 0.4311 | 0.4025 | 11.0046 | 11.7198 / 16.3091 | New primary-lDDT/FoldScore leader |
| E151 E147 best full 30k continuation | returned | 30000 | 0.5678 | 0.5678 | 0.5312 | 6.9642 | 14.3693 / 16.3091 | Returned coherently |
"""


def test_parse_experiment_results_tracks_foldscore_running_best():
    rows = parse_experiment_results_from_text()

    assert rows[-1].run_num == 151
    assert rows[-1].final_stop_foldscore == 0.5312
    assert rows[-1].running_best_final_stop_foldscore == 0.5312
    assert rows[-1].running_best_foldscore_label == "E151 E147 best full 30k continuation"


def test_foldscore_csv_marks_e151_lineage_only(tmp_path):
    rows = parse_experiment_results_from_text()
    path = tmp_path / "foldscore_by_experiment_run.csv"

    write_foldscore_experiment_csv(rows, path)

    with path.open(encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))

    by_run = {row["run_num"]: row for row in csv_rows}
    by_label = {row["run_label"]: row for row in csv_rows}
    assert by_label["main/full control"]["e151_lineage_contribution"] == "baseline AF2-style control"
    assert by_run["55"]["e151_lineage_contribution"] == "batch-8 branch checkpoint"
    assert by_run["123"]["e151_lineage_contribution"] == ""
    assert by_run["128"]["e151_lineage_contribution"] == "damped triangle-attention bias"
    assert by_run["151"]["e151_lineage_contribution"] == "30k continuation"


def test_e151_lineage_plot_labels_alternate_vertical_offsets():
    offsets = [LINEAGE_LABEL_OFFSETS[label][1] for label in LINEAGE_PLOT_LABELS]

    assert all(offset != 0 for offset in offsets)
    assert all((previous < 0) != (current < 0) for previous, current in zip(offsets, offsets[1:], strict=False))


def parse_experiment_results_from_text():
    path = type("_FakePath", (), {"read_text": lambda self, encoding: RESULTS_MD})()
    return parse_experiment_results(path)  # type: ignore[arg-type]
