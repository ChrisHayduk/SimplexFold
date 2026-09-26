"""Version 2 research execution shared by pinned SimplexFold architectures.

This is a new, deterministic experiment protocol, not a historical checkpoint
migration. Samples use a cyclic shuffled stream (no dropped tail), feature RNG
is keyed by consumed batch position, and evaluation uses every whole target.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import inspect
import json
import math
import os
import random
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

PROTOCOL_VERSION = 2


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str
    )


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def atomic_save(path, value, *, tensor=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            if tensor:
                torch.save(value, handle)
            else:
                handle.write((canonical(value) + "\n").encode())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"].cpu())


@contextlib.contextmanager
def isolated_rng(seed):
    state = rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.manual_seed(seed)
        yield
    finally:
        restore_rng(state)


def positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


class SampleStream:
    """Index stream independent of model, feature and worker RNG state."""

    def __init__(self, size, seed, cursor=0):
        self.size = positive_int(size, "dataset size")
        self.seed = seed
        self.cursor = cursor
        self._epoch = -1
        self._order = []

    def take(self, count):
        indices = []
        for _ in range(positive_int(count, "batch size")):
            epoch, position = divmod(self.cursor, self.size)
            if epoch != self._epoch:
                generator = torch.Generator().manual_seed(self.seed + epoch)
                self._order = torch.randperm(self.size, generator=generator).tolist()
                self._epoch = epoch
            indices.append(self._order[position])
            self.cursor += 1
        return indices


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_inventory(root):
    paths = sorted(
        p for p in (root / "minalphafold").rglob("*") if p.suffix in {".py", ".txt"}
    )
    paths += sorted((root / "configs").glob("*.toml"))
    paths += [root / "scripts/run_nanofold_public_benchmarks.py"]
    return {str(p.relative_to(root)): digest(p) for p in paths}


def checked_example(example):
    """Ignore nonexistent atoms before geometry; reject corrupt observed atoms."""
    example = dict(example)
    length = len(example["aatype"])
    for positions, mask in [
        ("atom14_positions", "atom14_mask"),
        ("template_atom14_positions", "template_atom14_mask"),
    ]:
        xyz, valid = example[positions], example[mask]
        if xyz.shape != (*valid.shape, 3) or valid.shape[-2:] != (length, 14):
            raise ValueError(f"Invalid {positions}/{mask} dimensions")
        if not torch.all(torch.isfinite(valid) & ((valid == 0) | (valid == 1))):
            raise ValueError(f"Invalid binary {mask}")
        if not torch.isfinite(xyz[valid.bool()]).all():
            raise ValueError(f"Nonfinite observed {positions}")
        example[positions] = torch.where(valid[..., None].bool(), xyz, 0.0)
    if "label_residue_index" in example and not torch.equal(
        example["residue_index"], example["label_residue_index"]
    ):
        raise ValueError("Feature and label residue indices disagree")
    return example


def data_inventory(data_config):
    from .data import (
        ProcessedOpenProteinSetDataset,
        _npz_path_for_chain_id,
        validate_split_manifests,
    )

    if data_config.train_manifest is not None and data_config.val_manifest is not None:
        validate_split_manifests(data_config.train_manifest, data_config.val_manifest)
    datasets, inventory, sequence_sets = {}, {}, {}
    for split in ("train", "val"):
        manifest = getattr(data_config, split + "_manifest")
        if manifest is None:
            raise ValueError("Explicit train and validation manifests are required")
        ds = ProcessedOpenProteinSetDataset(
            data_config.processed_features_dir,
            data_config.processed_labels_dir,
            split=split,
            manifest_path=manifest,
        )
        if not ds.chain_ids or len(ds.chain_ids) != len(set(ds.chain_ids)):
            raise ValueError(f"Empty or duplicate {split} cohort")
        rows, seqs = [], set()
        for index, chain_id in enumerate(ds.chain_ids):
            example = checked_example(ds[index])
            seq = fingerprint(example["aatype"].tolist())
            seqs.add(seq)
            rows.append(
                {
                    "chain_id": chain_id,
                    "sequence_sha256": seq,
                    "files": {
                        key: digest(_npz_path_for_chain_id(directory, chain_id))
                        for key, directory in [
                            ("features", data_config.processed_features_dir),
                            ("labels", data_config.processed_labels_dir),
                        ]
                    },
                }
            )
        datasets[split], inventory[split], sequence_sets[split] = ds, rows, seqs
    if set(datasets["train"].chain_ids) & set(datasets["val"].chain_ids):
        raise ValueError("Training and validation target cohorts overlap")
    if sequence_sets["train"] & sequence_sets["val"]:
        raise ValueError("Training and validation exact sequences overlap")
    return datasets, inventory


def collate_indices(dataset, indices, data_config, training_config, *, training, seed):
    from .data import collate_batch

    examples = [checked_example(dataset[i]) for i in indices]
    # Full-length evaluation: crop size never selects a validation subsequence.
    crop = (
        data_config.crop_size if training else max(len(e["aatype"]) for e in examples)
    )
    if training and data_config.fixed_feature_seed is not None:
        seed = data_config.fixed_feature_seed
    extra = {}
    if "stochastic_msa_sampling" in inspect.signature(collate_batch).parameters:
        extra["stochastic_msa_sampling"] = getattr(
            data_config, "stochastic_msa_sampling", False
        )
    with isolated_rng(seed):
        return collate_batch(
            examples,
            crop_size=crop,
            msa_depth=data_config.msa_depth,
            extra_msa_depth=data_config.extra_msa_depth,
            max_templates=data_config.max_templates,
            training=training,
            block_delete_training_msa=data_config.block_delete_training_msa,
            block_delete_msa_fraction=data_config.block_delete_msa_fraction,
            block_delete_msa_randomize_num_blocks=data_config.block_delete_msa_randomize_num_blocks,
            block_delete_msa_num_blocks=data_config.block_delete_msa_num_blocks,
            masked_msa_probability=data_config.masked_msa_probability,
            random_seed=seed,
            num_recycling_samples=training_config.n_cycles,
            num_ensemble_samples=training_config.n_ensemble,
            **extra,
        )


def model_inputs(batch, training_config, step, *, training):
    from .trainer import model_inputs_from_batch

    parameters = inspect.signature(model_inputs_from_batch).parameters
    kwargs = {"step": step} if "step" in parameters else {}
    # E07's recipe helpers require explicit enabling of runtime overrides.
    kwargs.update({key: True for key in parameters if key.startswith("use_simplex_")})
    if "use_simplex_teacher_forcing" in parameters:
        kwargs["use_simplex_teacher_forcing"] = training
    inputs = model_inputs_from_batch(batch, training_config, **kwargs)
    inputs["sample_recycles"] = (
        training and training_config.sample_recycles is not False
    )
    return inputs


def fp32_tree(value):
    if torch.is_tensor(value):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, dict):
        return {k: fp32_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(fp32_tree(v) for v in value)
    return value


def compute_loss(loss_fn, batch, outputs):
    from .trainer import loss_inputs_from_batch

    inputs = loss_inputs_from_batch(batch, fp32_tree(outputs))
    inputs["residue_index"] = batch.get("loss_residue_index", batch["residue_index"])
    loss, terms = loss_fn(**inputs, return_breakdown=True)
    if loss.shape != (len(batch["chain_id"]),) or not torch.isfinite(loss).all():
        raise ValueError("Every target must have one finite loss")
    return loss, terms


def aggregate(rows):
    if not rows or len({r["chain_id"] for r in rows}) != len(rows):
        raise ValueError("Empty or duplicate evaluation targets")
    keys = set(rows[0]) - {"chain_id", "length"}
    if any(set(r) - {"chain_id", "length"} != keys for r in rows):
        raise ValueError("Metric support differs between targets")
    result = {
        "val_examples": len(rows),
        "target_cohort_sha256": fingerprint([r["chain_id"] for r in rows]),
    }
    for key in sorted(keys):
        values = [float(row[key]) for row in rows]
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"Nonfinite {key}; refusing to omit a target")
        result["val_" + key] = sum(values) / len(values)
        result["val_" + key + "_count"] = len(values)
    return result


def evaluate(
    model, loss_fn, dataset, data_config, training_config, device, scorer, step
):
    from .trainer import move_to_device

    modes = [(module, module.training) for module in model.modules()]
    rows = []
    try:
        with isolated_rng(training_config.seed + 90817), torch.no_grad():
            model.eval()
            # One whole target per call makes predictions independent of padding
            # and validation batch composition, including adaptive simplex topology.
            for index, chain_id in enumerate(dataset.chain_ids):
                batch = collate_indices(
                    dataset,
                    [index],
                    data_config,
                    training_config,
                    training=False,
                    seed=training_config.seed + 90817,
                )
                batch = move_to_device(batch, device)
                outputs = model(
                    **model_inputs(batch, training_config, step, training=False)
                )
                losses, _ = compute_loss(loss_fn, batch, outputs)
                length = int(batch["seq_mask"][0].sum())
                predicted = outputs["atom14_coords"][0, :length].detach().float()
                true = batch["true_atom_positions"][0, :length].float()
                mask = batch["true_atom_mask"][0, :length].bool()
                if not torch.isfinite(predicted).all():
                    raise ValueError(
                        "Nonfinite prediction on complete validation target"
                    )
                feature = checked_example(dataset[index])
                components = scorer(
                    predicted,
                    true,
                    mask,
                    batch["aatype"][0, :length],
                    residue_index=batch["residue_index"][0, :length],
                    between_segment_residues=feature["between_segment_residues"].to(
                        device
                    ),
                )
                row = {"chain_id": chain_id, "length": length, "loss": float(losses[0])}
                row.update({k: float(v) for k, v in components.items()})
                if "foldscore" not in row:
                    raise ValueError("The required scorer did not return FoldScore")
                rows.append(row)
    finally:
        for module, mode in modes:
            module.training = mode
    if [r["chain_id"] for r in rows] != list(dataset.chain_ids):
        raise ValueError(
            "Evaluation did not preserve the complete ordered target cohort"
        )
    result = aggregate(rows)
    result["metric_step"] = step
    return result, rows


def build_loss(training_config, loss_class=None):
    from .losses import AlphaFoldLoss
    from .trainer import apply_loss_weight_schedule

    loss_class = AlphaFoldLoss if loss_class is None else loss_class
    accepted = inspect.signature(loss_class).parameters
    kwargs = {
        key: value for key, value in asdict(training_config).items() if key in accepted
    }
    kwargs["finetune"] = training_config.finetune
    loss_fn = loss_class(**kwargs)
    loss_fn.structural_violation_weight = training_config.structural_violation_weight
    apply_loss_weight_schedule(loss_fn, training_config, 1)
    return loss_fn


def validate_runtime_modules(model_config, training_config):
    """A runtime multiplier cannot create absent parameterized model modules."""
    cfg = asdict(model_config)
    for key, value in asdict(training_config).items():
        if (
            not key.endswith(("_runtime_scale", "_runtime_scale_final"))
            or value is None
            or float(value) == 0
        ):
            continue
        stem = key.split("_runtime_scale")[0]
        construction = stem + "_scale"
        if stem == "simplex_boundary_readout_directionality":
            construction = stem
        if stem in {"simplex_vertex_star_context", "simplex_edge_star_context"}:
            construction = "simplex_global_context_scale"
        if construction in cfg and float(cfg[construction]) == 0:
            raise ValueError(
                f"{key} activates absent constructor setting {construction}"
            )


def stage_config(base, stages, step):
    overrides, active = {}, None
    for stage in stages:
        if stage["start_step"] <= step:
            overrides.update(stage["training_config"])
            active = stage
    return replace(base, **overrides), active


def _run(recipe, output_dir, *, resume=False, max_runtime_seconds=None):
    """Run one seeded architecture. A recipe is persisted in full before training."""
    from . import trainer
    from .model import AlphaFold2
    from .trainer import DataConfig, TrainingConfig, load_model_config, move_to_device

    root = Path(__file__).resolve().parents[1]
    allowed = {
        "model_config",
        "model_overrides",
        "variant",
        "data",
        "training",
        "stages",
        "eval_every",
        "scorer_root",
        "max_parameters",
    }
    if set(recipe) - allowed:
        raise ValueError(f"Unknown recipe fields: {set(recipe) - allowed}")
    training = TrainingConfig(**recipe["training"])
    for name in ("epochs", "batch_size", "grad_accum_steps", "n_cycles", "n_ensemble"):
        positive_int(getattr(training, name), name)
    if max_runtime_seconds is not None and (
        not math.isfinite(max_runtime_seconds) or max_runtime_seconds <= 0
    ):
        raise ValueError("Runtime limit must be positive and finite")
    # Stateful workers and generic fit checkpoint flags are not part of this runner.
    for field in (
        "num_workers",
        "resume_from_checkpoint",
        "init_weights_from_checkpoint",
        "latest_checkpoint_path",
        "best_checkpoint_path",
        "max_samples",
    ):
        if getattr(training, field, None):
            raise ValueError(
                f"{field} is not used by research v2; use its explicit resume flag"
            )
    data_config = DataConfig(**recipe["data"])
    data_config = replace(data_config, val_fraction=0.0)
    for name in ("crop_size", "msa_depth"):
        positive_int(getattr(data_config, name), name)
    for name in ("extra_msa_depth", "max_templates"):
        value = getattr(data_config, name)
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if data_config.chains_manifest is not None:
        raise ValueError(
            "Use explicit already-filtered train/val manifests; no silent cohort filtering"
        )
    interval = recipe.get("eval_every", 500)
    if type(interval) is not int or interval < 0:
        raise ValueError("eval_every must be a nonnegative integer")
    if (
        training.sample_recycles is not None
        and type(training.sample_recycles) is not bool
    ):
        raise ValueError("sample_recycles must be a boolean or None")
    if type(training.seed) is not int or not 0 <= training.seed < 2**62:
        raise ValueError("seed must be an integer in [0, 2**62)")
    datasets, data_pins = data_inventory(data_config)
    old_runner = _module(
        root / "scripts/run_nanofold_public_benchmarks.py",
        "_simplex_architecture_recipes",
    )
    cfg = old_runner._variant_config(
        load_model_config(recipe["model_config"]), recipe["variant"]
    )
    cfg = replace(cfg, **recipe.get("model_overrides", {}))
    stages = recipe.get("stages", [])
    previous = 0
    forbidden_stage = {
        "epochs",
        "batch_size",
        "grad_accum_steps",
        "seed",
        "device",
        "num_workers",
        "n_cycles",
        "n_ensemble",
        "ema_decay",
        "resume_from_checkpoint",
        "init_weights_from_checkpoint",
    }
    for stage in stages:
        if set(stage) - {
            "start_step",
            "training_config",
            "optimizer_reset",
            "reset_train_iterator",
            "label",
        }:
            raise ValueError("Unknown stage fields")
        for flag in ("optimizer_reset", "reset_train_iterator"):
            if type(stage.get(flag, False)) is not bool:
                raise ValueError(f"{flag} must be boolean")
        if set(stage["training_config"]) & {
            "adam_beta1",
            "adam_beta2",
            "adam_eps",
            "weight_decay",
        } and not stage.get("optimizer_reset", False):
            raise ValueError(
                "Optimizer parameter changes require an explicit optimizer_reset stage"
            )
        start = positive_int(stage["start_step"], "stage start")
        if start <= previous or start > training.epochs:
            raise ValueError(
                "Stages must have distinct increasing starts within the run"
            )
        previous = start
        if set(stage["training_config"]) & forbidden_stage:
            raise ValueError(
                "Stage changes fixed sampling, budget, initialization or EMA fields"
            )
        candidate, _ = stage_config(training, stages, start)
        validate_runtime_modules(cfg, candidate)
    validate_runtime_modules(cfg, training)
    import sys

    sys.path.insert(0, str(Path(recipe["scorer_root"]).resolve()))
    import nanofold.metrics as scorer_module
    from nanofold.metrics import FOLDSCORE_VERSION, foldscore_components

    expected_scorer = Path(recipe["scorer_root"]).resolve() / "nanofold/metrics.py"
    if Path(scorer_module.__file__).resolve() != expected_scorer:
        raise ValueError("Scorer import does not match the declared source root")
    runtime = {
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": training.device,
        "deterministic_algorithms": True,
    }
    data_semantics = asdict(data_config)
    for key in (
        "processed_features_dir",
        "processed_labels_dir",
        "train_manifest",
        "val_manifest",
        "chains_manifest",
    ):
        data_semantics.pop(key, None)
    contract = {
        "protocol": PROTOCOL_VERSION,
        "model": asdict(cfg),
        "training": asdict(training),
        "stages": stages,
        "data": data_semantics,
        "corpus": data_pins,
        "split_manifest_sha256": {
            role: digest(Path(getattr(data_config, role + "_manifest")))
            for role in ("train", "val")
        },
        "source": source_inventory(root),
        "scorer": {
            "version": FOLDSCORE_VERSION,
            "sha256": digest(expected_scorer),
            "constants_sha256": digest(
                expected_scorer.with_name("residue_constants.py")
            ),
        },
        "runtime": runtime,
        "eval_every": recipe.get("eval_every", 500),
        "variant": recipe["variant"],
    }
    provenance_path = root / "source_provenance.json"
    if provenance_path.exists():
        contract["source_provenance"] = json.loads(provenance_path.read_text())
    contract_id = fingerprint(contract)
    output_dir = Path(output_dir)
    checkpoint_path = output_dir / "checkpoint.pt"
    if (
        output_dir.exists()
        and any(p.name != ".run.lock" for p in output_dir.iterdir())
        and not resume
    ):
        raise FileExistsError(
            "Run directory is not empty; choose a new directory or explicit resume"
        )
    checkpoint = None
    if resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            checkpoint["contract_sha256"] != contract_id
            or checkpoint["contract"] != contract
        ):
            raise ValueError(
                "Checkpoint source, data, scorer or scientific recipe differs"
            )
        stored_result = output_dir / "results.json"
        if (
            stored_result.exists()
            and json.loads(stored_result.read_text())["completed_steps"]
            > checkpoint["step"]
        ):
            raise ValueError(
                "Output contains future results relative to requested checkpoint"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.set_seed(training.seed)
    torch.use_deterministic_algorithms(True)
    device = trainer.resolve_device(training.device)
    model = AlphaFold2(cfg).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    if (
        recipe.get("max_parameters") is not None
        and parameters > recipe["max_parameters"]
    ):
        raise ValueError("Architecture exceeds the declared parameter ceiling")
    optimizer = trainer.build_optimizer(model, training)
    ema = (
        trainer.build_ema_model(model, training.ema_decay).to(device)
        if training.ema_decay is not None
        else None
    )
    step, total_examples, feature_batches, stream_epoch_base, cursor = 0, 0, 0, 0, 0
    history = []
    if checkpoint:
        step, total_examples = checkpoint["step"], checkpoint["total_examples"]
        feature_batches, stream_epoch_base, cursor = (
            checkpoint["feature_batches"],
            checkpoint["stream_epoch_base"],
            checkpoint["cursor"],
        )
        if (
            step < 0
            or step > training.epochs
            or total_examples != step * training.batch_size * training.grad_accum_steps
        ):
            raise ValueError("Checkpoint budget counters disagree")
        reset_step = max(
            [
                stage["start_step"]
                for stage in stages
                if stage["start_step"] <= step
                and stage.get("reset_train_iterator", False)
            ]
            or [0]
        )
        expected_cursor = (
            (step - reset_step + (1 if reset_step else 0))
            * training.batch_size
            * training.grad_accum_steps
        )
        if (
            feature_batches != step * training.grad_accum_steps
            or cursor != expected_cursor
            or stream_epoch_base != reset_step * 1000003
        ):
            raise ValueError(
                "Checkpoint sampling counters disagree with the bound stage schedule"
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if ema is not None:
            ema.load_state_dict(checkpoint["ema"])
        history = checkpoint["history"]
        restore_rng(checkpoint["rng"])
    else:
        atomic_save(
            output_dir / "contract.json", {"sha256": contract_id, "contract": contract}
        )
        atomic_save(
            output_dir / "parameter_inventory.json",
            {k: list(v.shape) for k, v in model.state_dict().items()},
        )
    stream = SampleStream(
        len(datasets["train"]), training.seed + stream_epoch_base, cursor
    )
    started = time.monotonic()
    paused = False

    def save():
        trainer.require_finite_state(
            (
                model.state_dict(),
                optimizer.state_dict(),
                ema.state_dict() if ema is not None else None,
            )
        )
        atomic_save(
            checkpoint_path,
            {
                "contract_sha256": contract_id,
                "contract": contract,
                "step": step,
                "parameters": parameters,
                "parameter_shapes": {
                    name: list(value.shape) for name, value in model.named_parameters()
                },
                "total_examples": total_examples,
                "feature_batches": feature_batches,
                "cursor": stream.cursor,
                "stream_epoch_base": stream_epoch_base,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "ema": ema.state_dict() if ema is not None else None,
                "rng": rng_state(),
                "history": history,
            },
            tensor=True,
        )

    while step < training.epochs:
        next_step = step + 1
        current, stage = stage_config(training, stages, next_step)
        if stage and stage["start_step"] == next_step:
            if stage.get("optimizer_reset", False):
                optimizer = trainer.build_optimizer(model, current)
            if stage.get("reset_train_iterator", False):
                stream_epoch_base = next_step * 1000003
                stream = SampleStream(
                    len(datasets["train"]), training.seed + stream_epoch_base
                )
        loss_fn = build_loss(current).to(device)
        if (
            current.finetune_start_step is not None
            and next_step > current.finetune_start_step
        ):
            loss_fn.finetune = True
            loss_fn.structural_violation_weight *= min(
                1.0,
                (next_step - current.finetune_start_step)
                / max(current.violation_ramp_steps, 1),
            )
        trainer.apply_loss_weight_schedule(loss_fn, current, next_step)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(current.grad_accum_steps):
            indices = stream.take(current.batch_size)
            batch = collate_indices(
                datasets["train"],
                indices,
                data_config,
                current,
                training=True,
                seed=training.seed + 2000003 + feature_batches,
            )
            feature_batches += 1
            batch = move_to_device(batch, device)
            outputs = model(**model_inputs(batch, current, next_step, training=True))
            losses, _ = compute_loss(loss_fn, batch, outputs)
            loss = losses.mean() / current.grad_accum_steps
            loss.backward()
            total_loss += float(loss.detach())
        if any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in model.parameters()
        ):
            raise ValueError("Nonfinite gradient; refusing optimizer update")
        if current.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), current.grad_clip_norm, error_if_nonfinite=True
            )
        total_examples += current.batch_size * current.grad_accum_steps
        learning_rate = trainer.learning_rate_at_step(
            current,
            step,
            training.epochs,
            is_finetune=loss_fn.finetune,
            samples_seen=total_examples,
        )
        trainer.set_optimizer_learning_rate(optimizer, learning_rate)
        optimizer.step()
        if ema is not None:
            ema.update_parameters(model)
        step = next_step
        row = {"step": step, "train_examples": total_examples, "train_loss": total_loss}
        interval = recipe.get("eval_every", 500)
        if interval and step % interval == 0 and step < training.epochs:
            result, _ = evaluate(
                model,
                loss_fn,
                datasets["val"],
                data_config,
                current,
                device,
                foldscore_components,
                step,
            )
            row.update(result)
        history.append(row)
        save()
        if (
            max_runtime_seconds is not None
            and time.monotonic() - started >= max_runtime_seconds
            and step < training.epochs
        ):
            paused = True
            break
    # Always score the current model; never attach an old metric to a newer step.
    current, _ = stage_config(training, stages, step)
    loss_fn = build_loss(current).to(device)
    if current.finetune_start_step is not None and step > current.finetune_start_step:
        loss_fn.finetune = True
        loss_fn.structural_violation_weight *= min(
            1.0,
            (step - current.finetune_start_step) / max(current.violation_ramp_steps, 1),
        )
    trainer.apply_loss_weight_schedule(loss_fn, current, step)
    result, rows = evaluate(
        model,
        loss_fn,
        datasets["val"],
        data_config,
        current,
        device,
        foldscore_components,
        step,
    )
    result.update(
        completed_steps=step,
        train_examples=total_examples,
        paused=paused,
        parameters=parameters,
        contract_sha256=contract_id,
        foldscore_version=FOLDSCORE_VERSION,
    )
    if ema is not None:
        ema_result, ema_rows = evaluate(
            ema,
            loss_fn,
            datasets["val"],
            data_config,
            current,
            device,
            foldscore_components,
            step,
        )
        result.update({"ema_" + key: value for key, value in ema_result.items()})
        atomic_save(output_dir / "eval_details_ema.json", ema_rows)
    save()
    result["checkpoint_sha256"] = digest(checkpoint_path)
    atomic_save(output_dir / "eval_details.json", rows)
    atomic_save(output_dir / "history.json", history)
    atomic_save(output_dir / "results.json", result)
    return result


def verify_result(output_dir):
    """Reconcile exact checkpoint, complete cohort and every aggregate before use."""
    directory = Path(output_dir)
    result = json.loads((directory / "results.json").read_text())
    checkpoint = torch.load(
        directory / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    contract = checkpoint["contract"]
    if (
        fingerprint(contract) != checkpoint["contract_sha256"]
        or result["contract_sha256"] != checkpoint["contract_sha256"]
    ):
        raise ValueError("Contract identity mismatch")
    if result["checkpoint_sha256"] != digest(directory / "checkpoint.pt"):
        raise ValueError("Checkpoint bytes mismatch")
    if (
        result["completed_steps"] != checkpoint["step"]
        or result["metric_step"] != checkpoint["step"]
    ):
        raise ValueError("Metric/checkpoint step mismatch")
    if result["train_examples"] != checkpoint["total_examples"]:
        raise ValueError("Exposure/checkpoint count mismatch")
    parameter_shapes = checkpoint.get("parameter_shapes")
    if not isinstance(parameter_shapes, dict) or not parameter_shapes:
        raise ValueError("Checkpoint lacks parameter inventory")
    if any(
        name not in checkpoint["model"]
        or list(checkpoint["model"][name].shape) != shape
        for name, shape in parameter_shapes.items()
    ):
        raise ValueError("Parameter inventory disagrees with model state")
    parameter_count = sum(math.prod(shape) for shape in parameter_shapes.values())
    if (
        checkpoint.get("parameters") != parameter_count
        or result.get("parameters") != parameter_count
    ):
        raise ValueError("Parameter count disagrees with checkpoint inventory")
    expected = [r["chain_id"] for r in contract["corpus"]["val"]]
    for filename, prefix in [
        ("eval_details.json", ""),
        ("eval_details_ema.json", "ema_"),
    ]:
        if prefix and checkpoint["ema"] is None:
            continue
        rows = json.loads((directory / filename).read_text())
        if [r["chain_id"] for r in rows] != expected:
            raise ValueError("Incomplete or reordered validation cohort")
        for key, value in aggregate(rows).items():
            if result[prefix + key] != value:
                raise ValueError(
                    f"Aggregate differs from detailed targets: {prefix + key}"
                )
    return result


def run(recipe, output_dir, *, resume=False, max_runtime_seconds=None):
    """Reserve the run for one process, releasing ownership even after failure."""
    import fcntl

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    deterministic = torch.are_deterministic_algorithms_enabled()
    with (output_dir / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process owns this output directory") from exc
        try:
            return _run(
                recipe,
                output_dir,
                resume=resume,
                max_runtime_seconds=max_runtime_seconds,
            )
        finally:
            torch.use_deterministic_algorithms(deterministic)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
