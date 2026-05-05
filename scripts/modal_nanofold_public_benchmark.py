"""Run SimplexFold NanoFold-public benchmarks on Modal GPUs.

This wraps ``scripts/run_nanofold_public_benchmarks.py`` for cloud execution.
It mounts the already-prepared NanoFold public feature/label volumes read-only,
adds the public manifests and NanoFold metric package to the image, and writes
JSON/CSV benchmark artifacts to a persistent Modal volume.

The default local entrypoint launches the full SimplexFold variant:

    modal run scripts/modal_nanofold_public_benchmark.py

Useful quick smoke:

    modal run scripts/modal_nanofold_public_benchmark.py --train-limit 1 \
      --val-limit 1 --steps 1 --crop-size 256 --msa-depth 128 \
      --extra-msa-depth 256 --n-cycles 4 --max-val-batches 1

Set ``SIMPLEXFOLD_MODAL_GPU`` to override the default H200 GPU class.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_NANOFOLD_ROOT = Path(
    os.environ.get(
        "SIMPLEXFOLD_NANOFOLD_ROOT",
        "/Users/christopherhayduk/Projects/nanoFold-Competition",
    )
)
REMOTE_ROOT = Path("/root")
REMOTE_NANOFOLD_ROOT = REMOTE_ROOT / "nanofold_public"
REMOTE_OUTPUT_ROOT = REMOTE_ROOT / "artifacts"

GPU_SPEC = os.environ.get("SIMPLEXFOLD_MODAL_GPU") or "H200"
FEATURES_VOLUME_NAME = os.environ.get("SIMPLEXFOLD_FEATURES_VOLUME", "nanofold-public-features")
LABELS_VOLUME_NAME = os.environ.get("SIMPLEXFOLD_LABELS_VOLUME", "nanofold-public-labels")
OUTPUT_VOLUME_NAME = os.environ.get("SIMPLEXFOLD_BENCHMARK_VOLUME", "simplexfold-nanofold-benchmarks")

FEATURES_MOUNT = Path("/mnt/nanofold-public-features")
LABELS_MOUNT = Path("/mnt/nanofold-public-labels")


app = modal.App("simplexfold-nanofold-public-benchmark")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.3", "numpy")
    .workdir(str(REMOTE_ROOT))
    .add_local_dir(str(REPO_ROOT / "minalphafold"), remote_path=str(REMOTE_ROOT / "minalphafold"))
    .add_local_dir(str(REPO_ROOT / "configs"), remote_path=str(REMOTE_ROOT / "configs"))
    .add_local_file(
        str(REPO_ROOT / "scripts" / "run_nanofold_public_benchmarks.py"),
        remote_path=str(REMOTE_ROOT / "scripts" / "run_nanofold_public_benchmarks.py"),
    )
    .add_local_dir(
        str(LOCAL_NANOFOLD_ROOT / "nanofold"),
        remote_path=str(REMOTE_NANOFOLD_ROOT / "nanofold"),
    )
    .add_local_file(
        str(LOCAL_NANOFOLD_ROOT / "data" / "manifests" / "train.txt"),
        remote_path=str(REMOTE_NANOFOLD_ROOT / "data" / "manifests" / "train.txt"),
    )
    .add_local_file(
        str(LOCAL_NANOFOLD_ROOT / "data" / "manifests" / "val.txt"),
        remote_path=str(REMOTE_NANOFOLD_ROOT / "data" / "manifests" / "val.txt"),
    )
)

features_volume = modal.Volume.from_name(FEATURES_VOLUME_NAME, create_if_missing=False)
labels_volume = modal.Volume.from_name(LABELS_VOLUME_NAME, create_if_missing=False)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)


def _link_mount(source: Path, destination: Path) -> None:
    """Create/replace ``destination`` as a symlink to the mounted volume path."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        if destination.is_dir() and not destination.is_symlink():
            raise RuntimeError(f"Refusing to replace real directory: {destination}")
        destination.unlink()
    destination.symlink_to(source, target_is_directory=True)


@app.function(
    image=image,
    gpu=GPU_SPEC,
    volumes={
        str(FEATURES_MOUNT): features_volume.read_only(),
        str(LABELS_MOUNT): labels_volume.read_only(),
        str(REMOTE_OUTPUT_ROOT): output_volume,
    },
    timeout=60 * 60 * 24,
)
def run_benchmark(argv: list[str]) -> str:
    import os
    import sys
    import threading
    import time

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.chdir(REMOTE_ROOT)
    sys.path.insert(0, str(REMOTE_ROOT))
    sys.path.insert(0, str(REMOTE_ROOT / "scripts"))

    _link_mount(FEATURES_MOUNT, REMOTE_NANOFOLD_ROOT / "data" / "processed_features")
    _link_mount(LABELS_MOUNT, REMOTE_NANOFOLD_ROOT / "data" / "processed_labels")

    from run_nanofold_public_benchmarks import main as benchmark_main

    full_argv = [
        "--nanofold-root",
        str(REMOTE_NANOFOLD_ROOT),
        "--output-dir",
        str(REMOTE_OUTPUT_ROOT / "nanofold_public_benchmarks"),
        *argv,
    ]
    print("[modal] argv:", " ".join(full_argv), flush=True)
    stop_commits = threading.Event()

    def commit_loop() -> None:
        interval_seconds = int(os.environ.get("SIMPLEXFOLD_MODAL_COMMIT_INTERVAL_SECONDS", "600"))
        while not stop_commits.wait(interval_seconds):
            try:
                output_volume.commit()
                print("[modal] committed benchmark volume checkpoint", flush=True)
            except Exception as exc:
                print(f"[modal] volume checkpoint commit failed: {exc}", flush=True)

    commit_thread = threading.Thread(target=commit_loop, name="volume-commit-loop", daemon=True)
    commit_thread.start()
    try:
        benchmark_main(full_argv)
    finally:
        stop_commits.set()
        commit_thread.join(timeout=5)
        output_volume.commit()
        print("[modal] committed final benchmark volume", flush=True)
    return str(REMOTE_OUTPUT_ROOT / "nanofold_public_benchmarks")


@app.local_entrypoint()
def main(
    model_config: str = "simplexfold_param_matched",
    zero_dropout: bool = False,
    run_name: str = "",
    variant: str = "full",
    steps: int = 10_000,
    train_limit: int = 0,
    val_limit: int = 0,
    crop_size: int = 256,
    msa_depth: int = 128,
    extra_msa_depth: int = 256,
    max_templates: int = 0,
    batch_size: int = 1,
    grad_accum_steps: int = 1,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    warmup_samples: int = 0,
    lr_decay_samples: int = 0,
    lr_decay_factor: float = 1.0,
    grad_clip_norm: float = 0.1,
    ema_decay: float = 0.0,
    use_clamped_fape: float = 0.9,
    msa_loss_weight: float = 2.0,
    distogram_loss_weight: float = 0.3,
    confidence_loss_weight: float = 0.01,
    simplex_aux_weight: float = 1.0,
    backbone_loss_weight: float = 1.0,
    sidechain_fape_loss_weight: float = 1.0,
    torsion_loss_weight: float = 1.0,
    loss_weight_ramp_start_step: int = 0,
    loss_weight_ramp_steps: int = 1,
    msa_loss_weight_final: float = -1.0,
    distogram_loss_weight_final: float = -1.0,
    confidence_loss_weight_final: float = -1.0,
    simplex_aux_weight_final: float = -1.0,
    backbone_loss_weight_final: float = -1.0,
    sidechain_fape_loss_weight_final: float = -1.0,
    torsion_loss_weight_final: float = -1.0,
    finetune_start_step: int = 0,
    finetune_lr_scale: float = 0.5,
    violation_ramp_steps: int = 0,
    num_workers: int = 4,
    seed: int = 0,
    n_cycles: int = 4,
    n_ensemble: int = 1,
    max_val_batches: int = 0,
    eval_max_val_batches: int = -1,
    final_max_val_batches: int = -1,
    eval_every: int = 0,
    log_every: int = 100,
    checkpoint_every: int = 0,
    resume_from_checkpoint: str = "",
    auto_resume: bool = False,
    stop_after_seconds: int = 0,
    mixed_precision: str = "bf16",
    train_chain_ids: str = "",
    val_chain_ids: str = "",
    overfit_chain_id: str = "",
) -> None:
    """Launch a SimplexFold public benchmark run on Modal.

    ``train_limit=0`` and ``val_limit=0`` mean the full official public
    manifests: 10,000 train chains and 1,000 validation chains.
    """
    argv = [
        "--model-config",
        model_config,
        "--variants",
        variant,
        "--steps",
        str(steps),
        "--train-limit",
        str(train_limit),
        "--val-limit",
        str(val_limit),
        "--crop-size",
        str(crop_size),
        "--msa-depth",
        str(msa_depth),
        "--extra-msa-depth",
        str(extra_msa_depth),
        "--max-templates",
        str(max_templates),
        "--batch-size",
        str(batch_size),
        "--grad-accum-steps",
        str(grad_accum_steps),
        "--learning-rate",
        str(learning_rate),
        "--weight-decay",
        str(weight_decay),
        "--warmup-samples",
        str(warmup_samples),
        "--lr-decay-factor",
        str(lr_decay_factor),
        "--grad-clip-norm",
        str(grad_clip_norm),
        "--use-clamped-fape",
        str(use_clamped_fape),
        "--msa-loss-weight",
        str(msa_loss_weight),
        "--distogram-loss-weight",
        str(distogram_loss_weight),
        "--confidence-loss-weight",
        str(confidence_loss_weight),
        "--simplex-aux-weight",
        str(simplex_aux_weight),
        "--backbone-loss-weight",
        str(backbone_loss_weight),
        "--sidechain-fape-loss-weight",
        str(sidechain_fape_loss_weight),
        "--torsion-loss-weight",
        str(torsion_loss_weight),
        "--finetune-lr-scale",
        str(finetune_lr_scale),
        "--violation-ramp-steps",
        str(violation_ramp_steps),
        "--device",
        "cuda",
        "--num-workers",
        str(num_workers),
        "--seed",
        str(seed),
        "--n-cycles",
        str(n_cycles),
        "--n-ensemble",
        str(n_ensemble),
        "--max-val-batches",
        str(max_val_batches),
        "--eval-every",
        str(eval_every),
        "--log-every",
        str(log_every),
        "--mixed-precision",
        mixed_precision,
    ]
    if run_name:
        argv.extend(["--run-name", run_name])
    if eval_max_val_batches >= 0:
        argv.extend(["--eval-max-val-batches", str(eval_max_val_batches)])
    if final_max_val_batches >= 0:
        argv.extend(["--final-max-val-batches", str(final_max_val_batches)])
    if checkpoint_every > 0:
        argv.extend(["--checkpoint-every", str(checkpoint_every)])
    if resume_from_checkpoint:
        argv.extend(["--resume-from-checkpoint", resume_from_checkpoint])
    if auto_resume:
        argv.append("--auto-resume")
    if stop_after_seconds > 0:
        argv.extend(["--stop-after-seconds", str(stop_after_seconds)])
    if lr_decay_samples > 0:
        argv.extend(["--lr-decay-samples", str(lr_decay_samples)])
    if zero_dropout:
        argv.append("--zero-dropout")
    if loss_weight_ramp_start_step > 0:
        argv.extend(["--loss-weight-ramp-start-step", str(loss_weight_ramp_start_step)])
        argv.extend(["--loss-weight-ramp-steps", str(loss_weight_ramp_steps)])
    final_weight_args = {
        "--msa-loss-weight-final": msa_loss_weight_final,
        "--distogram-loss-weight-final": distogram_loss_weight_final,
        "--confidence-loss-weight-final": confidence_loss_weight_final,
        "--simplex-aux-weight-final": simplex_aux_weight_final,
        "--backbone-loss-weight-final": backbone_loss_weight_final,
        "--sidechain-fape-loss-weight-final": sidechain_fape_loss_weight_final,
        "--torsion-loss-weight-final": torsion_loss_weight_final,
    }
    for flag, value in final_weight_args.items():
        if value >= 0:
            argv.extend([flag, str(value)])
    if ema_decay > 0:
        argv.extend(["--ema-decay", str(ema_decay)])
    if finetune_start_step > 0:
        argv.extend(["--finetune-start-step", str(finetune_start_step)])
    if train_chain_ids:
        argv.extend(["--train-chain-ids", *[item.strip() for item in train_chain_ids.split(",") if item.strip()]])
    if val_chain_ids:
        argv.extend(["--val-chain-ids", *[item.strip() for item in val_chain_ids.split(",") if item.strip()]])
    if overfit_chain_id:
        argv.extend(["--overfit-chain-id", overfit_chain_id])
    print(f"[modal] app={app.name} gpu={GPU_SPEC}")
    print(f"[modal] feature volume={FEATURES_VOLUME_NAME} label volume={LABELS_VOLUME_NAME}")
    print(f"[modal] output volume={OUTPUT_VOLUME_NAME}")
    print("[modal] benchmark argv:", " ".join(argv))
    function_call = run_benchmark.spawn(argv)
    print(f"[modal] spawned function_call={function_call.object_id}")
