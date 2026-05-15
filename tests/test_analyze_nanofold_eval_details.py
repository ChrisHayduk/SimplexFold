import csv
import json

from scripts.analyze_nanofold_eval_details import format_summary, load_eval_rows, main, summarize_eval_details


def _write_eval_details(path, rows):
    fieldnames = [
        "length",
        "lddt_ca",
        "foldscore",
        "ca_drmsd",
        "pred_ca_rg",
        "true_ca_rg",
        "simplex_face_boundary_lddt",
        "simplex_tetra_boundary_lddt",
        "simplex_face_boundary_contraction_fraction",
        "simplex_tetra_boundary_contraction_fraction",
        "simplex_face_boundary_edge_mean_degree",
        "simplex_tetra_boundary_edge_mean_degree",
        "simplex_face_outer_edge_mean_degree",
        "simplex_tetra_outer_edge_mean_degree",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_summarize_eval_details_reports_length_and_boundary_gap(tmp_path):
    path = tmp_path / "eval_details_full_msa_to_face.csv"
    _write_eval_details(
        path,
        [
            {
                "length": 60,
                "lddt_ca": 0.60,
                "foldscore": 0.50,
                "ca_drmsd": 5.0,
                "pred_ca_rg": 9.0,
                "true_ca_rg": 10.0,
                "simplex_face_boundary_lddt": 0.82,
                "simplex_tetra_boundary_lddt": 0.80,
                "simplex_face_boundary_contraction_fraction": 0.50,
                "simplex_tetra_boundary_contraction_fraction": 0.52,
                "simplex_face_boundary_edge_mean_degree": 20.0,
                "simplex_tetra_boundary_edge_mean_degree": 30.0,
                "simplex_face_outer_edge_mean_degree": 31.0,
                "simplex_tetra_outer_edge_mean_degree": 37.0,
            },
            {
                "length": 240,
                "lddt_ca": 0.35,
                "foldscore": 0.32,
                "ca_drmsd": 12.0,
                "pred_ca_rg": 12.0,
                "true_ca_rg": 20.0,
                "simplex_face_boundary_lddt": 0.80,
                "simplex_tetra_boundary_lddt": 0.78,
                "simplex_face_boundary_contraction_fraction": 0.60,
                "simplex_tetra_boundary_contraction_fraction": 0.62,
                "simplex_face_boundary_edge_mean_degree": 18.0,
                "simplex_tetra_boundary_edge_mean_degree": 28.0,
                "simplex_face_outer_edge_mean_degree": 32.0,
                "simplex_tetra_outer_edge_mean_degree": 38.0,
            },
            {
                "length": 140,
                "lddt_ca": 0.45,
                "foldscore": 0.40,
                "ca_drmsd": 9.0,
                "pred_ca_rg": 13.0,
                "true_ca_rg": 16.0,
                "simplex_face_boundary_lddt": 0.70,
                "simplex_tetra_boundary_lddt": 0.68,
                "simplex_face_boundary_contraction_fraction": 0.55,
                "simplex_tetra_boundary_contraction_fraction": 0.57,
                "simplex_face_boundary_edge_mean_degree": 19.0,
                "simplex_tetra_boundary_edge_mean_degree": 29.0,
                "simplex_face_outer_edge_mean_degree": 33.0,
                "simplex_tetra_outer_edge_mean_degree": 39.0,
            },
        ],
    )

    summary = summarize_eval_details(load_eval_rows(path), stratum_size=1)

    assert summary["n"] == 3
    assert summary["length_bins"][0]["label"] == "0-79"
    assert summary["length_bins"][0]["mean_lddt_ca"] == 0.60
    assert summary["length_bins"][-1]["label"] == ">=220"
    assert summary["length_bins"][-1]["mean_rg_ratio"] == 0.60
    assert summary["high_boundary_low_global"]["high_boundary_count"] == 2
    assert summary["high_boundary_low_global"]["high_boundary_low_global_count"] == 1
    assert summary["lddt_strata"]["bottom"]["mean_length"] == 240.0


def test_format_summary_is_aggregate_only(tmp_path):
    path = tmp_path / "eval_details_full_msa_to_face.csv"
    _write_eval_details(
        path,
        [
            {
                "length": 70,
                "lddt_ca": 0.50,
                "foldscore": 0.40,
                "ca_drmsd": 6.0,
                "pred_ca_rg": 8.0,
                "true_ca_rg": 10.0,
                "simplex_face_boundary_lddt": 0.75,
                "simplex_tetra_boundary_lddt": 0.75,
                "simplex_face_boundary_contraction_fraction": 0.50,
                "simplex_tetra_boundary_contraction_fraction": 0.50,
                "simplex_face_boundary_edge_mean_degree": 20.0,
                "simplex_tetra_boundary_edge_mean_degree": 30.0,
                "simplex_face_outer_edge_mean_degree": 32.0,
                "simplex_tetra_outer_edge_mean_degree": 38.0,
            }
        ],
    )

    rendered = format_summary(summarize_eval_details(load_eval_rows(path), stratum_size=1))

    assert "Rows: 1" in rendered
    assert "Length bins:" in rendered
    assert "chain_id" not in rendered


def test_main_emits_json(tmp_path, capsys):
    path = tmp_path / "eval_details_full_msa_to_face.csv"
    _write_eval_details(
        path,
        [
            {
                "length": 70,
                "lddt_ca": 0.50,
                "foldscore": 0.40,
                "ca_drmsd": 6.0,
                "pred_ca_rg": 8.0,
                "true_ca_rg": 10.0,
                "simplex_face_boundary_lddt": 0.75,
                "simplex_tetra_boundary_lddt": 0.75,
                "simplex_face_boundary_contraction_fraction": 0.50,
                "simplex_tetra_boundary_contraction_fraction": 0.50,
                "simplex_face_boundary_edge_mean_degree": 20.0,
                "simplex_tetra_boundary_edge_mean_degree": 30.0,
                "simplex_face_outer_edge_mean_degree": 32.0,
                "simplex_tetra_outer_edge_mean_degree": 38.0,
            }
        ],
    )

    summary = main([str(path), "--json", "--stratum-size", "1"])
    captured = json.loads(capsys.readouterr().out)

    assert summary["n"] == 1
    assert captured["n"] == 1
    assert captured["metrics"]["lddt_ca"]["mean"] == 0.5
