import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from minalphafold.reporting import (
    analyze_targets,
    audit_goal,
    comparable_runs,
    load_ledger,
    load_verified,
    main,
    record_result,
)
from minalphafold.research import run
from tests.test_research_integrity import fixture_recipe


@pytest.fixture
def verified_run(tmp_path):
    recipe = fixture_recipe(tmp_path)
    recipe["training"]["epochs"] = 1
    directory = tmp_path / "run"
    run(recipe, directory)
    return directory, recipe


def test_real_tiny_runner_to_verified_ledger_analysis_and_plot(verified_run, tmp_path, monkeypatch):
    from plot_nanofold_experiment_metrics import main as plot

    directory, _ = verified_run
    ledger = tmp_path / "research-v2.json"
    entry = record_result(ledger, directory, "tiny seed 19")
    assert load_ledger(ledger)["runs"] == [entry]
    report = analyze_targets(load_verified(directory))
    assert report["target_summary"]["lddt_ca"]["count"] == 2
    assert len(report["targets_by_ca_lddt"]) == 2
    assert audit_goal(load_verified(directory), target=0.7, confirmation_steps=1, max_parameters=1)["single_run_candidate"] is False
    original = ledger.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        record_result(ledger, directory, "tiny seed 19")
    with pytest.raises(ValueError, match="multiple experiments"):
        record_result(ledger, directory, "same seed disguised")
    assert ledger.read_bytes() == original
    summary = tmp_path / "summary.md"
    main("refresh", [str(ledger), "--output", str(summary)])
    assert "research-v2" in summary.read_text()
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl-cache"))
    output = tmp_path / "plot"
    plot([str(directory), "--output-dir", str(output)])
    assert (output / "metrics.svg").read_text().startswith("<?xml")
    assert json.loads((output / "provenance.json").read_text())["runs"][0]["seed"] == 19
    with pytest.raises(FileExistsError):
        plot([str(directory), "--output-dir", str(output)])


@pytest.mark.parametrize("mutation", ["missing_target", "unbound_parameters", "history"])
def test_reporting_rejects_incomplete_or_misbound_artifacts(verified_run, tmp_path, mutation):
    directory, _ = verified_run
    if mutation == "missing_target":
        path = directory / "eval_details.json"
        payload = json.loads(path.read_text())[:-1]
    elif mutation == "unbound_parameters":
        path = directory / "results.json"
        payload = json.loads(path.read_text())
        payload["parameters"] = 1
    else:
        path = directory / "history.json"
        payload = json.loads(path.read_text())
        payload[-1]["train_loss"] = -1000
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        record_result(tmp_path / "ledger.json", directory, "invalid")
    assert not (tmp_path / "ledger.json").exists()


def test_reporting_does_not_compare_different_complete_cohorts(verified_run, tmp_path):
    first, recipe = verified_run
    other = copy.deepcopy(recipe)
    train, val = tmp_path / "other_train.txt", tmp_path / "other_val.txt"
    train.write_text("target0\ntarget1\n")
    val.write_text("target2\ntarget4\n")
    other["data"].update(train_manifest=str(train), val_manifest=str(val))
    second = tmp_path / "other"
    run(other, second)
    with pytest.raises(ValueError, match="identical complete"):
        comparable_runs([first, second])


def test_paused_run_is_explicit_status_not_completed_result(tmp_path):
    recipe = fixture_recipe(tmp_path)
    directory = tmp_path / "paused"
    result = run(recipe, directory, max_runtime_seconds=1e-9)
    assert result["paused"]
    assert load_verified(directory, allow_paused=True)["status"] == "paused"
    with pytest.raises(ValueError, match="paused"):
        record_result(tmp_path / "ledger.json", directory, "partial")


def test_e151_dry_run_is_fresh_explicit_and_does_not_launch(tmp_path, capsys):
    from run_e151_from_scratch import main as launch

    argv = []
    for name in ("features", "labels", "train-manifest", "val-manifest", "scorer-root", "output-dir", "export-dir"):
        argv.extend(["--" + name, str(tmp_path / name)])
    launch(argv + ["--seed", "23", "--global-context-scale", "0.25"])
    preview = json.loads(capsys.readouterr().out)
    assert preview["recipe"]["training"]["seed"] == 23
    assert preview["recipe"]["model_overrides"]["simplex_global_context_scale"] == 0.25
    assert "--revision" in preview["command"] and "e07" in preview["command"]
    assert not any(tmp_path.iterdir())
    Path(tmp_path / "output-dir").mkdir()
    with pytest.raises(FileExistsError):
        launch(argv + ["--seed", "23", "--global-context-scale", "0.25"])


def test_gitless_e151_recipe_requires_original_curriculum_bytes(tmp_path, monkeypatch):
    import make_e151_research_recipe as recipe_module

    name = "configs/e151_fixed_arch_no_staged_losses_schedule.toml"
    source = subprocess.check_output(["git", "show", f"{recipe_module.E07}:{name}"], cwd=recipe_module.ROOT)
    (tmp_path / "configs").mkdir()
    (tmp_path / name).write_bytes(source)
    (tmp_path / "source_provenance.json").write_text(json.dumps({
        "commit": recipe_module.E07, "revision": "e07",
        "original_files": {name: hashlib.sha256(source).hexdigest()},
    }))
    monkeypatch.setattr(recipe_module, "ROOT", tmp_path)
    kwargs = {key: tmp_path / key for key in ("features", "labels", "train_manifest", "val_manifest", "scorer_root")}
    recipe = recipe_module.make_recipe(**kwargs, seed=3, global_context_scale=0.25)
    assert recipe["stages"]
    (tmp_path / name).write_bytes(source + b"\n# changed curriculum\n")
    with pytest.raises(ValueError, match="pinned original"):
        recipe_module.make_recipe(**kwargs, seed=3, global_context_scale=0.25)


def test_gitless_e01_shell_uses_existing_runtime_without_export(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    source = Path(__file__).resolve().parents[1] / "scripts/run_runpod_e01_pilot.sh"
    launcher = scripts / source.name
    launcher.write_bytes(source.read_bytes())
    (scripts / "run_research_experiment.py").write_text("import json,sys; print(json.dumps(sys.argv[1:]))\n")
    (tmp_path / "source_provenance.json").write_text(json.dumps({
        "revision": "e01", "commit": "a596b6b5c234e44aed79d66766f1515b9bc86b1b",
    }))
    command = ["bash", str(launcher), "--recipe", "explicit.json", "--output-dir", "fresh"]
    environment = {**os.environ, "PYTHON_BIN": sys.executable}
    result = subprocess.run(command, env=environment, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == ["--recipe", "explicit.json", "--output-dir", "fresh"]
    failed = subprocess.run(command + ["--export-dir", "second"], env=environment, capture_output=True, text=True)
    assert failed.returncode != 0 and not failed.stdout
