#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/ChrisHayduk/SimplexFold.git}"
MAIN_COMMIT="${MAIN_COMMIT:-a2994386581a05f1569b785c473347cc4f4a9558}"
E01_COMMIT="${E01_COMMIT:-6c20faa94f5581c9f40b6919dddacd9c08b56813}"
WORKDIR="${RUNPOD_WORKDIR:-/workspace/codex-simplexfold-e01-runpod-20260509}"
NANOFOLD_ROOT="${NANOFOLD_ROOT:-/workspace/nanoFold-Competition}"
PYTHON_BIN="${PYTHON_BIN:-python}"

STEPS="${STEPS:-1000}"
EVAL_EVERY="${EVAL_EVERY:-100}"
TRAIN_LIMIT="${TRAIN_LIMIT:-256}"
VAL_LIMIT="${VAL_LIMIT:-64}"
CROP_SIZE="${CROP_SIZE:-128}"
MSA_DEPTH="${MSA_DEPTH:-32}"
EVAL_MAX_VAL_BATCHES="${EVAL_MAX_VAL_BATCHES:-8}"
FINAL_MAX_VAL_BATCHES="${FINAL_MAX_VAL_BATCHES:-16}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-100}"
LOG_EVERY="${LOG_EVERY:-10}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

mkdir -p "${WORKDIR}/repos" "${WORKDIR}/results" "${WORKDIR}/logs"

require_path() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "missing required path: ${path}" >&2
    exit 1
  fi
}

checkout_repo() {
  local destination="$1"
  local commit="$2"
  if [[ -d "${destination}/.git" ]]; then
    git -C "${destination}" fetch --quiet origin
  else
    git clone --quiet "${REPO_URL}" "${destination}"
  fi
  git -C "${destination}" checkout --quiet --detach "${commit}"
}

run_benchmark() {
  local repo_dir="$1"
  local run_name="$2"
  (
    cd "${repo_dir}"
    "${PYTHON_BIN}" scripts/run_nanofold_public_benchmarks.py \
      --nanofold-root "${NANOFOLD_ROOT}" \
      --model-config simplexfold_medium_param_matched \
      --variants full \
      --train-limit "${TRAIN_LIMIT}" \
      --val-limit "${VAL_LIMIT}" \
      --steps "${STEPS}" \
      --eval-every "${EVAL_EVERY}" \
      --log-every "${LOG_EVERY}" \
      --eval-max-val-batches "${EVAL_MAX_VAL_BATCHES}" \
      --final-max-val-batches "${FINAL_MAX_VAL_BATCHES}" \
      --crop-size "${CROP_SIZE}" \
      --msa-depth "${MSA_DEPTH}" \
      --extra-msa-depth 0 \
      --max-templates 0 \
      --batch-size 1 \
      --grad-accum-steps 1 \
      --checkpoint-every "${CHECKPOINT_EVERY}" \
      --checkpoint-dir "${WORKDIR}/checkpoints/${run_name}" \
      --auto-resume \
      --device cuda \
      --num-workers "${NUM_WORKERS}" \
      --mixed-precision "${MIXED_PRECISION}" \
      --output-dir "${WORKDIR}/results" \
      --run-name "${run_name}"
  )
}

summarize_results() {
  "${PYTHON_BIN}" - <<'PY'
import csv
import json
import os
from pathlib import Path

workdir = Path(os.environ.get("RUNPOD_WORKDIR", "/workspace/codex-simplexfold-e01-runpod-20260509"))
runs = [
    ("e00_main_full_runpod_pilot", os.environ.get("MAIN_COMMIT", "a2994386581a05f1569b785c473347cc4f4a9558")),
    ("e01_balanced_contact_full_runpod_pilot", os.environ.get("E01_COMMIT", "6c20faa94f5581c9f40b6919dddacd9c08b56813")),
]
rows = []
for run_name, commit in runs:
    result_path = workdir / "results" / run_name / "results.json"
    if not result_path.exists():
        continue
    for row in json.loads(result_path.read_text()):
        rows.append(
            {
                "run_name": run_name,
                "commit": commit,
                "variant": row.get("variant", ""),
                "parameters": row.get("parameters", ""),
                "completed_steps": row.get("completed_steps", ""),
                "val_lddt_ca": row.get("val_lddt_ca", ""),
                "val_ca_rmsd": row.get("val_ca_rmsd", ""),
                "val_loss": row.get("val_loss", ""),
                "elapsed_seconds": row.get("elapsed_seconds", ""),
            }
        )
summary_path = workdir / "results" / "e01_pilot_summary.csv"
summary_path.parent.mkdir(parents=True, exist_ok=True)
fieldnames = [
    "run_name",
    "commit",
    "variant",
    "parameters",
    "completed_steps",
    "val_lddt_ca",
    "val_ca_rmsd",
    "val_loss",
    "elapsed_seconds",
]
with summary_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(summary_path)
PY
}

require_path "${NANOFOLD_ROOT}/data/manifests/train.txt"
require_path "${NANOFOLD_ROOT}/data/manifests/val.txt"
require_path "${NANOFOLD_ROOT}/data/processed_features"
require_path "${NANOFOLD_ROOT}/data/processed_labels"

checkout_repo "${WORKDIR}/repos/SimplexFold-main" "${MAIN_COMMIT}"
checkout_repo "${WORKDIR}/repos/SimplexFold-e01" "${E01_COMMIT}"

run_benchmark "${WORKDIR}/repos/SimplexFold-main" "e00_main_full_runpod_pilot"
run_benchmark "${WORKDIR}/repos/SimplexFold-e01" "e01_balanced_contact_full_runpod_pilot"
summarize_results
