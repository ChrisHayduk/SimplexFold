import json

from scripts.format_experiment_result_row import format_result_row, load_history, load_result, main


def test_format_result_row_ignores_inherited_resume_history():
    result = {
        "completed_steps": 5500,
        "val_lddt_ca": 0.37177472934126854,
        "val_foldscore": 0.3721824511885643,
        "val_ca_drmsd": 10.102694064378738,
        "val_pred_ca_rg": 12.087152183055878,
        "val_true_ca_rg": 15.40340667963028,
    }
    history = [
        {"step": 5000, "val_lddt_ca": 0.37505385652184486},
        {"step": 5500, "val_lddt_ca": 0.37177472934126854},
    ]

    row = format_result_row(
        result,
        run_label="E72 continue edge-frame 0.025 to 5500",
        status="completed",
        decision="rejected",
        history=history,
        start_after_step=5000,
    )

    assert row == (
        "| E72 continue edge-frame 0.025 to 5500 | completed | 5500 | "
        "0.3718 | 0.3718 | 0.3722 | 10.1027 | 12.0872 / 15.4034 | rejected |"
    )


def test_main_formats_row_from_json_files(tmp_path, capsys):
    result_path = tmp_path / "results.json"
    history_path = tmp_path / "history.json"
    result_path.write_text(
        json.dumps(
            [
                {
                    "completed_steps": 600,
                    "stopped_early": True,
                    "val_lddt_ca": 0.25,
                    "val_FoldScore": 0.3,
                }
            ]
        ),
        encoding="utf-8",
    )
    history_path.write_text(
        json.dumps(
            [
                {"step": 400, "val_lddt_ca": 0.5},
                {"step": 600, "val_lddt_ca": 0.25},
            ]
        ),
        encoding="utf-8",
    )

    main(
        [
            str(result_path),
            "--history-json",
            str(history_path),
            "--run-label",
            "E-test",
            "--status",
            "stopped early",
            "--decision",
            "rejected",
            "--start-after-step",
            "500",
        ]
    )

    assert (
        capsys.readouterr().out.strip()
        == "| E-test | stopped early | 600 | 0.2500 | 0.2500 | 0.3000 | - | - | rejected |"
    )


def test_main_selects_requested_variant_from_multirow_results(tmp_path, capsys):
    result_path = tmp_path / "results.json"
    result_path.write_text(
        json.dumps(
            [
                {
                    "variant": "other_variant",
                    "completed_steps": 9000,
                    "val_lddt_ca": 0.1,
                    "val_foldscore": 0.2,
                },
                {
                    "variant": "full_msa_to_face",
                    "completed_steps": 9000,
                    "val_lddt_ca": 0.46,
                    "val_foldscore": 0.41,
                    "val_ca_drmsd": 10.5,
                },
            ]
        ),
        encoding="utf-8",
    )

    main(
        [
            str(result_path),
            "--variant",
            "full_msa_to_face",
            "--run-label",
            "E-test",
            "--decision",
            "pending",
        ]
    )

    assert (
        capsys.readouterr().out.strip()
        == "| E-test | completed | 9000 | 0.4600 | 0.4600 | 0.4100 | 10.5000 | - | pending |"
    )


def test_public_load_helpers_select_variant_and_history(tmp_path):
    result_path = tmp_path / "results.json"
    history_path = tmp_path / "history.json"
    result_path.write_text(
        json.dumps(
            [
                {"variant": "other_variant", "val_lddt_ca": 0.1},
                {"variant": "full_msa_to_face", "val_lddt_ca": 0.46},
            ]
        ),
        encoding="utf-8",
    )
    history_path.write_text(json.dumps([{"step": 9000, "val_lddt_ca": 0.46}]), encoding="utf-8")

    assert load_result(result_path, variant="full_msa_to_face")["val_lddt_ca"] == 0.46
    assert load_history(history_path) == [{"step": 9000, "val_lddt_ca": 0.46}]


def test_format_result_row_accepts_canonical_ca_rg_keys():
    result = {
        "completed_steps": 9000,
        "val_lddt_ca": 0.4311057258844376,
        "val_foldscore": 0.4025340421795845,
        "val_ca_drmsd": 11.004606088757514,
        "val_ca_pred_rg": 11.719762571811676,
        "val_ca_true_rg": 16.30911695623398,
    }

    row = format_result_row(
        result,
        run_label="E128 damped triangle-attention bias from E124",
        status="returned",
        decision="leader",
    )

    assert row == (
        "| E128 damped triangle-attention bias from E124 | returned | 9000 | "
        "0.4311 | 0.4311 | 0.4025 | 11.0046 | 11.7198 / 16.3091 | leader |"
    )
