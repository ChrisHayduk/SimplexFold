import csv
import json
import os

from scripts.summarize_nanofold_run_status import main, summarize_run


def _write_active_run(run_dir):
    variant = "full_msa_to_face"
    run_dir.mkdir()
    (run_dir / f"status_{variant}.json").write_text(
        json.dumps(
            {
                "active_step": 8568,
                "active_microbatch": 1,
                "active_microbatches": 8,
                "active_eval_batch": 17,
                "active_eval_batches": 1000,
                "active_eval_examples": 16,
                "completed_step": 8567,
                "target_steps": 9000,
                "start_step": 8500,
                "elapsed_seconds_total": 3600.0,
                "elapsed_seconds_run": 3600.0,
                "last_history_step": 8500,
                "history_rows": 18,
                "effective_batch_size": 8,
                "stopped_early": False,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / f"history_{variant}.json").write_text(
        json.dumps([{"step": 8500, "val_lddt_ca": 0.4311}]),
        encoding="utf-8",
    )
    (run_dir / "run_metadata.json").write_text(json.dumps({"effective_batch_size": 8}), encoding="utf-8")
    launch_time = 1_800_000_000.0
    os.utime(run_dir / "run_metadata.json", (launch_time, launch_time))
    os.utime(run_dir / f"status_{variant}.json", (launch_time + 3600.0, launch_time + 3600.0))


def _write_returned_run(run_dir):
    variant = "full_msa_to_face"
    _write_active_run(run_dir)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "checkpoints" / f"{variant}_latest.pt").write_bytes(b"checkpoint")
    result = {
        "variant": variant,
        "completed_steps": 9000,
        "val_lddt_ca": 0.46,
        "val_foldscore": 0.41,
        "val_ca_drmsd": 10.5,
        "parameters": 3_240_738,
        "effective_batch_size": 8,
    }
    (run_dir / "results.json").write_text(json.dumps([result]), encoding="utf-8")
    (run_dir / "results.csv").write_text("variant,completed_steps\nfull_msa_to_face,9000\n", encoding="utf-8")
    with (run_dir / f"eval_details_{variant}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["chain_id", "val_lddt_ca"])
        writer.writeheader()
        writer.writerow({"chain_id": "hidden_from_summary", "val_lddt_ca": 0.46})


def test_summarize_run_reports_active_status_without_result_details(tmp_path):
    run_dir = tmp_path / "active"
    _write_active_run(run_dir)

    summary = summarize_run(run_dir)

    assert summary["state"] == "active"
    assert summary["status"]["completed_step"] == 8567
    assert summary["status"]["active_eval_batch"] == 17
    assert summary["status"]["active_eval_batches"] == 1000
    assert summary["status"]["active_eval_examples"] == 16
    assert summary["history_last_step"] == 8500
    assert summary["history_last"]["val_lddt_ca"] == 0.4311
    assert summary["progress"]["completed_delta_steps"] == 67
    assert summary["progress"]["remaining_steps"] == 433
    assert summary["progress"]["elapsed_source"] == "status_elapsed_seconds_run"
    assert summary["result"] is None
    assert summary["eval_rows"] is None


def test_summarize_run_reports_returned_bundle_without_reading_chain_ids(tmp_path):
    run_dir = tmp_path / "returned"
    _write_returned_run(run_dir)

    summary = summarize_run(run_dir)

    assert summary["state"] == "returned"
    assert summary["result"]["completed_steps"] == 9000
    assert summary["result"]["val_lddt_ca"] == 0.46
    assert summary["eval_rows"] == 1
    assert "hidden_from_summary" not in json.dumps(summary)


def test_main_emits_json_summaries(tmp_path, capsys):
    run_dir = tmp_path / "active"
    _write_active_run(run_dir)

    summaries = main([str(run_dir), "--json"])
    output = capsys.readouterr().out

    assert summaries[0]["state"] == "active"
    assert json.loads(output)[0]["state"] == "active"
    assert json.loads(output)[0]["progress"]["remaining_steps"] == 433


def test_main_formats_active_rate_and_eta(tmp_path, capsys):
    run_dir = tmp_path / "active"
    _write_active_run(run_dir)

    main([str(run_dir)])
    output = capsys.readouterr().out

    assert "rate=67.0/h" in output
    assert "eta=6.5h" in output
    assert "lddt_ca=0.4311" in output


def test_summarize_run_falls_back_to_file_timing_for_old_status(tmp_path):
    run_dir = tmp_path / "active"
    _write_active_run(run_dir)
    variant = "full_msa_to_face"
    status_path = run_dir / f"status_{variant}.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.pop("elapsed_seconds_run")
    status["elapsed_seconds_total"] = 100_000.0
    status_path.write_text(json.dumps(status), encoding="utf-8")
    launch_time = 1_800_000_000.0
    os.utime(status_path, (launch_time + 3600.0, launch_time + 3600.0))

    summary = summarize_run(run_dir)

    assert summary["progress"]["elapsed_source"] == "status_metadata_mtime_delta"
    assert summary["progress"]["steps_per_hour"] == 67.0
