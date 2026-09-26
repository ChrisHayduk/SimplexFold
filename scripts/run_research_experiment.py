#!/usr/bin/env python3
"""Run a content-bound research recipe using a pinned SimplexFold architecture.

--revision exports immutable Git source into a new directory, overlays the
reviewed execution/loss/data fixes, and starts the same CLI in that runtime.
No checkout, ref, historical artifact, network or implicit compute job changes.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISIONS = {
    "main": "a2994386581a05f1569b785c473347cc4f4a9558",
    "e01": "a596b6b5c234e44aed79d66766f1515b9bc86b1b",
    "e02": "e09b88f770a9fc297a54bb60dca27b003c62a697",
    "e03": "0259d0ce59da4bab059e733cbcb2c895e772fd86",
    "e04": "02a8bad3f6f40500e7a7e7a3c96117a2c7e0e6f1",
    "e05": "88b9f86bd2e1f47157ccba27616394c620bf872d",
    "e07": "e960359b0168fc3e1c402f61dd698499102af239",
}


def _definitions(text):
    return {
        node.name: node
        for node in ast.parse(text).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }


def _overlay_definitions(target, reviewed, names):
    """Replace shared owning functions, leaving branch architecture hooks intact."""
    original, source = target.read_text(), reviewed.read_text()
    nodes, replacements = _definitions(original), _definitions(source)
    lines = original.splitlines(keepends=True)
    additions = []
    for name in names:
        node = replacements[name]
        snippet = ast.get_source_segment(source, node) + "\n"
        if name in nodes:
            old = nodes[name]
            start = min([old.lineno] + [d.lineno for d in old.decorator_list]) - 1
            additions.append((start, old.end_lineno, snippet))
        else:
            # Insert missing helpers before any top-level executable entrypoint.
            insertion = next(
                (
                    n.lineno - 1
                    for n in ast.parse(original).body
                    if isinstance(n, ast.If)
                    and "__name__" in ast.get_source_segment(original, n)
                ),
                len(lines),
            )
            additions.append((insertion, insertion, "\n" + snippet + "\n"))
    for start, end, snippet in sorted(
        additions, key=lambda row: (row[0], row[1]), reverse=True
    ):
        lines[start:end] = [snippet]
    target.write_text("".join(lines))


def _overlay_runtime(destination):
    trainer = destination / "minalphafold/trainer.py"
    names = [
        "set_seed",
        "load_training_protocol",
        "load_checkpoint_for_resume",
        "require_finite_state",
        "_validate_resume_metadata",
        "checked_loss",
        "clip_gradients",
        "_limit_batch",
        "validate_training_config",
        "_file_digest",
        "_resume_contract",
        "_rng_state",
        "_restore_rng",
        "train_step",
        "evaluate",
        "save_checkpoint",
        "fit",
        "_collate_for_loader",
        "build_dataloader",
        "_apply_training_loss_schedule",
    ]
    _overlay_definitions(trainer, ROOT / "minalphafold/trainer.py", names)
    source = (ROOT / "minalphafold/trainer.py").read_text()
    text = trainer.read_text()
    for clsname in ("DataConfig", "TrainingConfig"):
        current_cls = _definitions(text)[clsname]
        fields = {n.target.id for n in current_cls.body if isinstance(n, ast.AnnAssign)}
        new_cls = _definitions(source)[clsname]
        new_fields = [
            ast.get_source_segment(source, n)
            for n in new_cls.body
            if isinstance(n, ast.AnnAssign) and n.target.id not in fields
        ]
        lines = text.splitlines(keepends=True)
        lines[current_cls.end_lineno : current_cls.end_lineno] = [
            "\n    " + value + "\n" for value in new_fields
        ]
        text = "".join(lines)
    text = text.replace(
        "import argparse\n",
        "import argparse\nimport hashlib\nimport inspect\nimport os\nimport tempfile\nimport uuid\n",
        1,
    )
    text = text.replace(
        "from .losses import AlphaFoldLoss",
        "from .data import MSA_SAMPLE_FEATURE_KEYS, validate_split_manifests, _npz_path_for_chain_id\nfrom .losses import AlphaFoldLoss",
        1,
    )
    trainer.write_text(text)
    _overlay_definitions(
        destination / "minalphafold/data.py",
        ROOT / "minalphafold/data.py",
        [
            "read_chain_id_manifest",
            "_npz_path_for_chain_id",
            "discover_chain_ids",
            "_load_processed_features",
            "_load_processed_labels",
            "ProcessedOpenProteinSetDataset",
            "_manifest_records",
            "validate_split_manifests",
        ],
    )
    shared_scripts = [
        "download_openproteinset.py",
        "filter_openproteinset.py",
        "modal_overfit.py",
        "modal_overfit_single_pdb.py",
        "modal_train_af2.py",
        "overfit_processed_chain.py",
        "overfit_single_pdb.py",
        "preprocess_openproteinset.py",
        "relax_pdb.py",
        "train_af2.py",
        "benchmark_simplexfold.py",
        "modal_nanofold_public_benchmark.py",
    ]
    shared_scripts += [
        "analyze_nanofold_eval_details.py",
        "verify_nanofold_benchmark_artifacts.py",
        "record_experiment_result.py",
        "upsert_experiment_result_row.py",
        "refresh_experiment_results_summary.py",
        "format_experiment_result_row.py",
        "audit_experiment_results.py",
        "audit_goal_artifact.py",
        "summarize_nanofold_run_status.py",
        "plot_nanofold_experiment_metrics.py",
        "plot_af2_large_simplexfold_checkpoint_progress.py",
        "run_e151_from_scratch.py",
        "run_runpod_e01_pilot.sh",
    ]
    shutil.copyfile(
        ROOT / "minalphafold/reporting.py", destination / "minalphafold/reporting.py"
    )
    for name in shared_scripts:
        shutil.copyfile(ROOT / "scripts" / name, destination / "scripts" / name)
    # Preserve the branch's architecture variants, replace the unsafe execution
    # loop. Rich E07 controls remain available through explicit JSON recipes.
    runner = destination / "scripts/run_nanofold_public_benchmarks.py"
    branch_text = runner.read_text()
    branch_variant = ast.get_source_segment(
        branch_text, _definitions(branch_text)["_variant_config"]
    )
    runner.write_text((ROOT / "scripts/run_nanofold_public_benchmarks.py").read_text())
    text = runner.read_text()
    node = _definitions(text)["_variant_config"]
    lines = text.splitlines(keepends=True)
    lines[node.lineno - 1 : node.end_lineno] = [branch_variant + "\n"]
    runner.write_text("".join(lines))


def export_revision(revision, destination):
    commit = REVISIONS[revision]
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    blob = subprocess.check_output(
        [
            "git",
            "archive",
            commit,
            "minalphafold",
            "configs",
            "scripts",
        ],
        cwd=ROOT,
    )
    originals = {}
    with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe exported path")
            contents = archive.extractfile(member).read()
            target = destination / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(contents)
            originals[str(path)] = hashlib.sha256(contents).hexdigest()
    fixes = json.loads((ROOT / "scripts/research_source_fixes.json").read_text())
    for fix in fixes:
        path = destination / fix["path"]
        text = path.read_text()
        if text.count(fix["before"]) != 1:
            raise ValueError(
                f"Frozen source differs from the reviewed fix: {fix['path']}"
            )
        path.write_text(text.replace(fix["before"], fix["after"]))
    _overlay_runtime(destination)
    # Runtime sanitation is shared by the replacement runner before any geometry.
    for name in (
        "minalphafold/research.py",
        "minalphafold/mmcif.py",
        "minalphafold/pdbio.py",
        "scripts/run_research_experiment.py",
        "scripts/verify_research_result.py",
        "scripts/make_e151_research_recipe.py",
    ):
        shutil.copyfile(ROOT / name, destination / name)
    receipt = {
        "protocol": 2,
        "revision": revision,
        "commit": commit,
        "original_files": originals,
        "effective_files": {
            str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(destination.rglob("*"))
            if p.is_file()
        },
    }
    (destination / "source_provenance.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n"
    )
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-runtime-seconds", type=float)
    parser.add_argument("--revision", choices=REVISIONS)
    parser.add_argument("--export-dir", type=Path)
    args = parser.parse_args(argv)
    if args.revision:
        if args.export_dir is None:
            parser.error("--revision requires a new --export-dir")
        exported = export_revision(args.revision, args.export_dir)
        command = [
            sys.executable,
            str(exported / "scripts/run_research_experiment.py"),
            "--recipe",
            str(args.recipe.resolve()),
            "--output-dir",
            str(args.output_dir.resolve()),
        ]
        if args.resume:
            command.append("--resume")
        if args.max_runtime_seconds is not None:
            command += ["--max-runtime-seconds", str(args.max_runtime_seconds)]
        return subprocess.call(command, cwd=exported)
    sys.path.insert(0, str(ROOT))
    from minalphafold.research import run

    result = run(
        json.loads(args.recipe.read_text()),
        args.output_dir,
        resume=args.resume,
        max_runtime_seconds=args.max_runtime_seconds,
    )
    print(json.dumps(result, indent=2))
    return 75 if result["paused"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
