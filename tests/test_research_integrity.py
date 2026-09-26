from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from minalphafold.a3m import sequence_to_ids
from minalphafold.losses import TorsionAngleLoss
from minalphafold.research import (
    SampleStream,
    aggregate,
    checked_example,
    collate_indices,
    data_inventory,
    run,
)
from minalphafold.simplex import _masked_cross_entropy_from_bins, _masked_symmetric_kl
from minalphafold.trainer import DataConfig, TrainingConfig

ROOT = Path(__file__).resolve().parents[1]
SCORER = ROOT.parent / "nanoFold-Competition"


def fixture_recipe(tmp_path):
    features, labels = tmp_path / "features", tmp_path / "labels"
    features.mkdir()
    labels.mkdir()
    sequences = ["AGASVA", "AGASVC", "AGASVD", "AGASVE", "AGASVF"]
    for i, sequence in enumerate(sequences):
        aatype = np.array(sequence_to_ids(sequence), dtype=np.int64)
        n = len(sequence)
        pos = np.zeros((n, 14, 3), np.float32)
        mask = np.zeros((n, 14), np.float32)
        for j in range(n):
            pos[j, :5] = np.array(
                [[0, 1, 0], [1.3, 0, 0], [2.6, 0.2, 0], [3, 0.7, 0], [1.2, -0.8, 1.1]]
            ) + [3.8 * j, 0, 0]
            mask[j, : 4 if sequence[j] == "G" else 5] = 1
        np.savez(
            features / f"target{i}.npz",
            aatype=aatype,
            msa=aatype[None],
            deletions=np.zeros((1, n), np.int64),
            residue_index=np.arange(n),
        )
        np.savez(
            labels / f"target{i}.npz",
            atom14_positions=pos,
            atom14_mask=mask,
            residue_index=np.arange(n),
            resolution=np.float32(2),
        )
    train, val = tmp_path / "train.txt", tmp_path / "val.txt"
    train.write_text("target0\ntarget1\ntarget2\n")
    val.write_text("target3\ntarget4\n")
    return {
        "model_config": "tiny",
        "variant": "full",
        "model_overrides": {},
        "data": {
            "processed_features_dir": str(features),
            "processed_labels_dir": str(labels),
            "train_manifest": str(train),
            "val_manifest": str(val),
            "crop_size": 4,
            "msa_depth": 1,
            "extra_msa_depth": 0,
            "max_templates": 0,
            "block_delete_training_msa": False,
        },
        "training": {
            "epochs": 3,
            "batch_size": 2,
            "grad_accum_steps": 2,
            "seed": 19,
            "device": "cpu",
            "num_workers": 0,
            "n_cycles": 1,
            "n_ensemble": 1,
            "learning_rate": 1e-4,
        },
        "eval_every": 1,
        "scorer_root": str(SCORER),
    }


def test_torsion_layer_duplication_preserves_loss_and_summed_gradient():
    pred = torch.randn(1, 2, 7, 2, requires_grad=True)
    true = torch.randn_like(pred)
    mask = torch.ones(1, 2, 7)
    loss_fn = TorsionAngleLoss()
    one = loss_fn(pred, pred, true, true, mask).sum()
    grad = torch.autograd.grad(one, pred)[0]
    repeated = pred.detach().unsqueeze(0).repeat(4, 1, 1, 1, 1).requires_grad_()
    four = loss_fn(repeated, repeated, true, true, mask).sum()
    grad4 = torch.autograd.grad(four, repeated)[0]
    torch.testing.assert_close(one, four)
    torch.testing.assert_close(grad, grad4.sum(0))


@pytest.mark.parametrize("kind", ["ce", "kl"])
def test_masked_aux_logits_ignore_nonfinite_padding_with_finite_gradients(kind):
    pred = torch.tensor(
        [[[1.0, 2.0], [float("nan"), float("inf")]]], requires_grad=True
    )
    mask = torch.tensor([[1.0, 0.0]])
    if kind == "ce":
        loss = _masked_cross_entropy_from_bins(
            pred, torch.zeros((1, 2), dtype=torch.long), mask
        )
    else:
        loss = _masked_symmetric_kl(pred, torch.zeros_like(pred), mask)
    loss.sum().backward()
    assert torch.isfinite(loss).all() and torch.isfinite(pred.grad).all()
    assert torch.equal(pred.grad[0, 1], torch.zeros(2))


def test_stream_resume_covers_odd_tail_without_rng_interference():
    stream = SampleStream(7, 13)
    prefix = stream.take(11)
    torch.manual_seed(567)
    resumed = SampleStream(7, 13, stream.cursor)
    assert stream.take(31) == resumed.take(31)
    assert set(prefix[:7]) == set(range(7))


def test_aggregate_rejects_omission_duplicate_and_nan():
    assert (
        aggregate(
            [
                {"chain_id": "a", "length": 2, "FoldScore": 0.2},
                {"chain_id": "b", "length": 8, "FoldScore": 0.8},
            ]
        )["val_FoldScore"]
        == 0.5
    )
    for rows in [
        [],
        [{"chain_id": "a", "length": 2, "FoldScore": float("nan")}],
        [{"chain_id": "a", "length": 2, "FoldScore": 0.2}] * 2,
        [
            {"chain_id": "a", "length": 2, "FoldScore": 0.2},
            {"chain_id": "b", "length": 2},
        ],
    ]:
        with pytest.raises(ValueError):
            aggregate(rows)


def test_dataset_identity_masks_and_full_length_eval(tmp_path):
    recipe = fixture_recipe(tmp_path)
    data = DataConfig(**recipe["data"])
    datasets, _ = data_inventory(data)
    example = datasets["val"][0]
    absent = example["atom14_mask"] == 0
    example["atom14_positions"][absent] = float("nan")
    checked = checked_example(example)
    assert torch.isfinite(checked["atom14_positions"]).all()
    example["atom14_positions"][0, 1, 0] = float("nan")
    with pytest.raises(ValueError, match="observed"):
        checked_example(example)
    batch = collate_indices(
        datasets["val"], [0], data, TrainingConfig(), training=False, seed=0
    )
    assert batch["seq_mask"].sum() == 6
    Path(recipe["data"]["val_manifest"]).write_text("target0\n")
    with pytest.raises(ValueError, match="overlap"):
        data_inventory(data)


def assert_state_equal(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for k in left:
            assert_state_equal(left[k], right[k])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_state_equal(a, b)
    else:
        assert left == right


def test_actual_simplex_training_pause_resume_and_eval_isolation(tmp_path):
    torch.set_num_threads(1)
    recipe = fixture_recipe(tmp_path)
    full = run(recipe, tmp_path / "full")
    paused = run(recipe, tmp_path / "resumed", max_runtime_seconds=1e-9)
    assert paused["paused"] and paused["completed_steps"] == paused["metric_step"] == 1
    resumed = run(recipe, tmp_path / "resumed", resume=True)
    assert full["val_foldscore"] == resumed["val_foldscore"]
    assert full["train_examples"] == resumed["train_examples"] == 12
    ckpts = [
        torch.load(tmp_path / name / "checkpoint.pt", weights_only=False)
        for name in ["full", "resumed"]
    ]
    for key in ["model", "optimizer", "history", "cursor", "feature_batches"]:
        assert_state_equal(ckpts[0][key], ckpts[1][key])
    every_final = copy.deepcopy(recipe)
    every_final["eval_every"] = 0
    run(every_final, tmp_path / "finalonly")
    ckpt = torch.load(tmp_path / "finalonly/checkpoint.pt", weights_only=False)
    assert_state_equal(ckpts[0]["model"], ckpt["model"])
    with pytest.raises(FileExistsError):
        run(recipe, tmp_path / "full")
    changed = copy.deepcopy(recipe)
    changed["training"]["learning_rate"] = 2e-4
    with pytest.raises(ValueError, match="differs"):
        run(changed, tmp_path / "resumed", resume=True)
    Path(recipe["data"]["processed_labels_dir"], "target1.npz").touch()
    # A byte change, not mtime, is the scientific identity boundary.
    with open(
        Path(recipe["data"]["processed_labels_dir"], "target1.npz"), "ab"
    ) as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="differs"):
        run(recipe, tmp_path / "resumed", resume=True)


def test_all_pinned_architectures_train_evaluate_and_resume(tmp_path):
    import subprocess
    import sys

    spec = importlib.util.spec_from_file_location(
        "simplex_export", ROOT / "scripts/run_research_experiment.py"
    )
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    recipe = fixture_recipe(tmp_path)
    recipe["training"].update(epochs=2, n_cycles=2, grad_accum_steps=1, batch_size=1)
    recipe["data"]["crop_size"] = 6
    for revision in exporter.REVISIONS:
        runtime = exporter.export_revision(revision, tmp_path / ("source_" + revision))
        current = copy.deepcopy(recipe)
        if revision == "e07":
            current["variant"] = "full_msa_to_face"
            current["model_overrides"].update(
                simplex_edge_frame_message_scale=0.0125,
                simplex_global_context_scale=1.0,
                simplex_vertex_star_context_scale=1.0,
                simplex_edge_star_context_scale=0.5,
                simplex_boundary_readout_directionality=0.25,
            )
            current["training"].update(
                simplex_edge_frame_message_runtime_scale=0.0125,
                simplex_vertex_star_context_runtime_scale=1.0,
                simplex_edge_star_context_runtime_scale=0.5,
                simplex_boundary_readout_directionality_runtime_scale=0.25,
            )
            current["stages"] = [
                {
                    "start_step": 2,
                    "optimizer_reset": True,
                    "reset_train_iterator": True,
                    "training_config": {"simplex_face_coordinate_weight": 0.5},
                }
            ]
        (runtime / "recipe.json").write_text(json.dumps(current))
        code = """
import json,torch,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from minalphafold.research import run,verify_result
torch.set_num_threads(1)
r=json.loads(Path('recipe.json').read_text())
a=run(r,'full')
b=run(r,'resumed',max_runtime_seconds=1e-9)
c=run(r,'resumed',resume=True)
assert a['val_foldscore']==c['val_foldscore']
assert verify_result('full')==a
x=torch.load('full/checkpoint.pt',weights_only=False)
y=torch.load('resumed/checkpoint.pt',weights_only=False)
assert x['cursor']==y['cursor'] and x['total_examples']==y['total_examples']
for k in x['model']:assert torch.equal(x['model'][k],y['model'][k]),k
# The generic epoch API shares hardened owning layers and preserves branch hooks.
from dataclasses import replace
from minalphafold import trainer
config=replace(trainer.load_model_config('tiny'), **r.get('model_overrides', {}))
data=trainer.DataConfig(**r['data']); data.val_fraction=0
training=trainer.TrainingConfig(epochs=2, seed=19, n_cycles=2, batch_size=1, device='cpu')
one,h1=trainer.fit(config,data,training)
trainer.fit(config,data,replace(training,epochs=1,latest_checkpoint_path='generic.pt'))
two,h2=trainer.fit(config,data,replace(training,resume_from_checkpoint='generic.pt'))
assert h1==h2
for key in one.state_dict(): assert torch.equal(one.state_dict()[key],two.state_dict()[key]),key
print(json.dumps(dict(parameters=a['parameters'],foldscore=a['val_foldscore'])))
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=runtime,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, (
            revision + "\n" + completed.stdout + "\n" + completed.stderr
        )
        print(revision, completed.stdout.strip())


def test_verifier_rejects_metric_and_cohort_tampering(tmp_path):
    from minalphafold.research import verify_result

    torch.set_num_threads(1)
    recipe = fixture_recipe(tmp_path)
    recipe["training"]["epochs"] = 1
    run(recipe, tmp_path / "run")
    verify_result(tmp_path / "run")
    path = tmp_path / "run/results.json"
    original = path.read_text()
    result = json.loads(original)
    result["val_foldscore"] += 0.01
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="Aggregate"):
        verify_result(tmp_path / "run")
    path.write_text(original)
    details = tmp_path / "run/eval_details.json"
    rows = json.loads(details.read_text())
    rows[1]["chain_id"] = rows[0]["chain_id"]
    details.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="cohort"):
        verify_result(tmp_path / "run")


def test_fixed_recycling_api_and_compatibility_cli(tmp_path, monkeypatch):
    from minalphafold.model import AlphaFold2
    from minalphafold.research import model_inputs
    from minalphafold.trainer import load_model_config

    torch.set_num_threads(1)
    recipe = fixture_recipe(tmp_path)
    data = DataConfig(**recipe["data"])
    datasets, _ = data_inventory(data)
    cfg = TrainingConfig(n_cycles=2, sample_recycles=False)
    batch = collate_indices(datasets["train"], [0], data, cfg, training=True, seed=3)
    model = AlphaFold2(load_model_config("tiny")).train()
    kwargs = model_inputs(batch, cfg, 1, training=True)
    original_randint = torch.randint

    def forbid_recycle_sampling(*args, **kwargs):
        raise AssertionError("Fixed-cycle recipe sampled recycle count")

    monkeypatch.setattr(torch, "randint", forbid_recycle_sampling)
    outputs = model(**kwargs)
    assert outputs["sampled_n_cycles"] == 2
    monkeypatch.setattr(torch, "randint", original_randint)
    with pytest.raises(ValueError, match="boolean"):
        model(**{**kwargs, "sample_recycles": 1})
    with pytest.raises(ValueError, match="positive integer"):
        model(**{**kwargs, "n_cycles": True})
    module_spec = importlib.util.spec_from_file_location(
        "public_cli", ROOT / "scripts/run_nanofold_public_benchmarks.py"
    )
    public = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(public)
    args = [
        "--nanofold-root",
        str(SCORER),
        "--features-dir",
        recipe["data"]["processed_features_dir"],
        "--labels-dir",
        recipe["data"]["processed_labels_dir"],
        "--train-manifest",
        recipe["data"]["train_manifest"],
        "--val-manifest",
        recipe["data"]["val_manifest"],
        "--steps",
        "1",
        "--variants",
        "full",
        "--crop-size",
        "4",
        "--msa-depth",
        "1",
        "--extra-msa-depth",
        "0",
        "--output-dir",
        str(tmp_path),
        "--run-name",
        "cli",
        "--device",
        "cpu",
    ]
    rows = public.main(args)
    assert rows[0]["val_examples"] == 2 and rows[0]["metric_step"] == 1
    assert (
        public.main(args + ["--auto-resume"])[0]["val_foldscore"]
        == rows[0]["val_foldscore"]
    )
