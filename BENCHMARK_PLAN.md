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

## Acceptance Bar

A NeurIPS-worthy result should show at least one statistically reliable
accuracy gain at matched or clearly justified compute, plus an ablation that
identifies which part of the simplex construction matters. A strong target is
`faces_recycled` improving low-MSA or long-range-contact cases while
`faces_tetra_recycled` improves packing/side-chain geometry without a large
memory penalty.
