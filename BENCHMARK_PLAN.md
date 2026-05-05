# SimplexFold Benchmark Plan

This benchmark plan is meant to produce publication-grade evidence for the
sparse 2-/3-simplex Evoformer adapter, not just smoke-test numbers.

## Claims To Test

1. Sparse face/tetra states improve structure accuracy over a matched
   pair-only trunk at fixed training data, crop size, recycling count, and
   parameter budget.
2. Recycled geometry topology is more useful than latent pair-only topology
   after the first structure pass.
3. Auxiliary face/tetra metric and geometry losses make the higher-order
   states carry interpretable shape information instead of becoming decorative.
4. The compute overhead is controlled by `neighbor_k` and stays practical for
   `L=256` crops.

## Model Variants

Run every accuracy benchmark with at least three seeds.

| Variant | Purpose |
| --- | --- |
| `pair_only` | `use_simplicial_evoformer=false`; same trunk channels/blocks. |
| `faces_latent` | Faces only, topology from pair logits, no recycled geometry. |
| `faces_recycled` | Faces only, pair logits plus recycled Cα distance topology. |
| `faces_msa_moment` | Faces plus low-rank MSA-to-face third-order moment. |
| `faces_tetra_recycled` | Full SimplexFold adapter. |
| `no_simplex_aux` | Full adapter with simplex auxiliary loss weights set to zero. |
| `k_sweep` | Full adapter with `K={8,12,16}` for cost/accuracy scaling. |

Use `configs/simplexfold_param_matched.toml` for the main SimplexFold result.
It keeps `num_evoformer=48`, but scales MSA/pair/single/simplex widths so the
full model is approximately the same size as `configs/alphafold2.toml` with
`use_simplicial_evoformer=false`.

## Data Splits

Use the same processed OpenProteinSet/NanoFold-compatible feature schema for
all variants. Keep templates disabled (`T=0`) and do not retrieve external
MSAs or structures during training/evaluation.

Recommended splits:

- Train: filtered training chains passing resolution, length, and composition
  filters already implemented in the OpenProteinSet scripts.
- Validation: chain-level held-out split, no sequence identity leakage above
  the chosen threshold.
- Test-easy: validation-like chains with `L <= 256`.
- Test-long: chains requiring cropping/stitching or `L > 256`.
- Test-low-MSA: examples with shallow MSA depth after official preprocessing.

## Metrics

Report accuracy and efficiency together.

- Structure: Cα RMSD after Kabsch alignment, GDT_TS, lDDT-Cα, TM-score/pTM
  calibration, FAPE, side-chain FAPE.
- Pair geometry: distogram CE, top-L/2 and top-L contact precision at 8 Å.
- Simplex auxiliaries: face edge-distance CE, tetra edge-distance CE,
  face-area MAE, tetra signed-volume MAE, tetra radius of gyration MAE,
  topology contact AUROC/AUPRC, pair-face consistency, and face-tetra
  consistency.
- Efficiency: parameters, wall-clock train tokens/sec, forward latency, peak
  memory, simplex faces/tetrahedra per crop.
- Stability: NaN rate, gradient norm distribution, time to first valid fold on
  single-chain overfit tasks.

## Figures And Tables

1. Accuracy table: mean plus 95% confidence interval over seeds for every
   variant on each split.
2. Pareto plot: lDDT-Cα or GDT_TS versus peak memory/latency.
3. K sweep: `neighbor_k` versus accuracy, faces/tetra count, and memory.
4. Recycling ablation: pass-0 latent topology versus recycled geometry across
   `n_cycles={1,2,3}`.
5. Calibration: predicted pLDDT/pTM versus actual lDDT/TM-score.
6. Interpretability: predicted versus true face edge distances, face areas,
   tetra edge distances, and tetra volumes, with residuals binned by residue
   separation.
7. Qualitative structures: overlay predictions for representative wins,
   losses, long-range-contact improvements, and low-MSA cases.

## Immediate Microbenchmark

Use the included harness for architecture-level cost numbers:

```bash
python scripts/benchmark_simplexfold.py \
  --model-config tiny \
  --device cpu \
  --length 128 \
  --msa-depth 32 \
  --extra-msa-depth 0 \
  --n-cycles 2 \
  --variants simplex faces_only no_simplex \
  --json-out artifacts/benchmarks/tiny_cpu.json
```

For GPU:

```bash
python scripts/benchmark_simplexfold.py \
  --model-config medium \
  --device cuda \
  --length 256 \
  --msa-depth 128 \
  --extra-msa-depth 256 \
  --n-cycles 2 \
  --timed-steps 20 \
  --json-out artifacts/benchmarks/medium_cuda_l256.json
```

## NanoFold Public Accuracy Benchmarks

Use NanoFold's official public train/validation manifests rather than a random
local split:

```bash
python scripts/run_nanofold_public_benchmarks.py \
  --nanofold-root /Users/christopherhayduk/Projects/nanoFold-Competition \
  --model-config tiny \
  --variants no_simplex faces full msa_to_face \
  --train-limit 256 \
  --val-limit 64 \
  --steps 1000 \
  --crop-size 128 \
  --msa-depth 32 \
  --extra-msa-depth 0 \
  --max-templates 0 \
  --output-dir artifacts/nanofold_public_benchmarks
```

The runner writes `run_metadata.json`, `history_<variant>.json`,
`results.json`, and `results.csv`. The CSV is the first figure-ready artifact:
it includes loss, Cα RMSD, lDDT-Cα, throughput, parameter count, and any active
simplex auxiliary/consistency losses. When the NanoFold repo is present, it
also imports the official FoldScore component implementation and reports
`foldscore`, GDT-HA, atom14 lDDT, CAD, side-chain, clash, backbone, and
dip-difference components on the evaluated crops.

Scale this ladder in three passes:

1. Smoke: `train-limit=8`, `val-limit=4`, `steps=2`, `crop-size=32`.
2. Development: `train-limit=256`, `val-limit=64`, `steps=1000`,
   `crop-size=128`, three seeds.
3. Paper run: full `10,000` public train chains, full `1,000` public validation
   chains, `crop-size=256`, `msa-depth=128`, `extra-msa-depth=1024` when
   memory permits, `n-cycles=4`, BF16, three seeds. Use the fixed phased
   coordinate recipe below as the default, then ablate the AF2-style
   `use-clamped-fape=0.9` setting separately. Evaluate EMA weights when the
   run is long enough for EMA to be meaningful.

Keep `max_templates=0` for all official NanoFold runs. The public feature
schema includes template placeholders, but the competition track is no-template.

### Modal Full-Model Run

The Modal runner mounts the existing NanoFold public feature/label volumes and
writes results to `simplexfold-nanofold-benchmarks`:

```bash
modal run --detach --timestamps scripts/modal_nanofold_public_benchmark.py \
  --model-config simplexfold_param_matched \
  --variant full \
  --steps 10000 \
  --train-limit 0 \
  --val-limit 0 \
  --crop-size 256 \
  --msa-depth 128 \
  --extra-msa-depth 256 \
  --max-templates 0 \
  --batch-size 1 \
  --grad-accum-steps 1 \
  --learning-rate 0.0003 \
  --warmup-samples 2048 \
  --ema-decay 0.999 \
  --use-clamped-fape 0.0 \
  --msa-loss-weight 0.2 \
  --distogram-loss-weight 0.3 \
  --confidence-loss-weight 0.01 \
  --simplex-aux-weight 1.0 \
  --backbone-loss-weight 1.0 \
  --sidechain-fape-loss-weight 1.0 \
  --torsion-loss-weight 1.0 \
  --loss-weight-ramp-start-step 6000 \
  --loss-weight-ramp-steps 1000 \
  --msa-loss-weight-final 0.05 \
  --distogram-loss-weight-final 0.1 \
  --simplex-aux-weight-final 0.5 \
  --backbone-loss-weight-final 6.0 \
  --sidechain-fape-loss-weight-final 2.0 \
  --torsion-loss-weight-final 0.5 \
  --grad-clip-norm 1.0 \
  --num-workers 4 \
  --n-cycles 4 \
  --mixed-precision bf16 \
  --max-val-batches 0 \
  --log-every 100
```

The `2048` warmup-sample value is intentionally scaled to the small effective
batch used here (`batch_size=1`, `grad_accum_steps=2`). A literal AF2
`128000`-sample warmup would last 64,000 optimizer steps and would keep this
10,000-step diagnostic run in a permanently low-LR regime.

The April 29-30 debugging runs showed that the earlier full run should be
treated as a training-dynamics stress test, not a final benchmark: it used
constant LR `1e-3`, batch/effective batch `1`, no EMA, no fine-tuning/violation
stage, and an overly weak unclamped coordinate signal. Its validation FAPE was
near the clamp ceiling, and the mean Cα RMSD matched the validation-set mean
Cα radius of gyration, consistent with mostly collapsed/unformed global
structures.

The fixed diagnostic recipe supports a delayed coordinate phase transition
rather than a simple data/metric bug. On the one-chain public NanoFold overfit
target `4d40_B`, early predictions are collapsed (`pred_rg ~3-6 Å` versus
`true_rg ~13 Å`). Once auxiliary losses fall and the backbone/side-chain FAPE
weights ramp upward, the model expands to the correct radius and RMSD drops
sharply:

| Run | Params | Recipe | Final FoldScore | Final Cα RMSD | Best Cα RMSD | Final lDDT-Cα |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| SimplexFold full | 93.8M | phased FAPE ramp | 0.729 | 1.37 Å | 1.21 Å | 0.820 |
| Reduced no-simplex | 46.1M | phased FAPE ramp | 0.655 | 1.92 Å | 1.92 Å | 0.744 |
| Full-width AF2 no-simplex | 94.2M | phased FAPE ramp | 0.574 | 4.85 Å | 4.45 Å | 0.668 |
| SimplexFold full | 93.8M | static mixed loss | 0.413 | 4.41 Å | 4.41 Å | 0.458 |

These are overfit diagnostics, not generalization claims. They do establish
that the model/loss/data path can learn real coordinates, that the collapse
mode is recipe-sensitive, and that the phased coordinate schedule is the right
default for the next public-train/public-val comparison.

On April 28, 2026, `crop-size=256`, `msa-depth=128`, `extra-msa-depth=256`,
and `n-cycles=4` OOMed before checkpointing the simplicial trunk. After
enabling BF16 autocast and activation checkpointing for `SimplicialEvoformer`,
the full-length H200 smoke completed on explicit 256-residue train/val chains
(`4mf5_A`, `1zww_B`) with the command above's crop/MSA/recycling settings.

Monitor:

```bash
modal app logs <app-id>
```

Fetch completed artifacts:

```bash
modal volume get simplexfold-nanofold-benchmarks \
  /nanofold_public_benchmarks/<run_id> \
  artifacts/modal_nanofold_public_benchmarks/<run_id>
```

## Acceptance Bar

A NeurIPS-worthy result should show at least one statistically reliable
accuracy gain at matched or clearly justified compute, plus an ablation that
identifies which part of the simplex construction matters. A strong target is
`faces_recycled` improving low-MSA or long-range-contact cases while
`faces_tetra_recycled` improves packing/side-chain geometry without a large
memory penalty.
