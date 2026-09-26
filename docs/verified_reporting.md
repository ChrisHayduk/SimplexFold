# Research v2 reporting and legacy script migration

The current reporting scripts take a **single variant run directory** containing
`results.json`, `checkpoint.pt`, `eval_details.json`, `history.json`, and
`contract.json`. They check the complete ordered validation cohort, exact
checkpoint and contract, update/exposure counts, parameter count, seed, and score
version before reporting. Historical CSVs, freeform Markdown rows, and historical
best-in-history values cannot be promoted into v2 results.

The same-named E07 scripts are deliberate successors with new `--help` interfaces.
Immutable historical branch source and ledgers remain available separately.

```sh
python scripts/verify_nanofold_benchmark_artifacts.py /path/to/run/full
python scripts/analyze_nanofold_eval_details.py /path/to/run/full
python scripts/record_experiment_result.py /path/to/run/full \
  --ledger artifacts/research-v2-ledger.json --run-label 'full seed 19'
python scripts/audit_experiment_results.py artifacts/research-v2-ledger.json
python scripts/refresh_experiment_results_summary.py artifacts/research-v2-ledger.json \
  --output artifacts/research-v2-summary-attempt-01.md
python scripts/plot_nanofold_experiment_metrics.py /path/to/run/full \
  --output-dir artifacts/plots-attempt-01
```

Use `upsert_experiment_result_row.py` with the same `run_dir`, `--ledger`, and
`--run-label` arguments for an explicit replacement of a label. It still requires
a verified result. The ledger uses an exclusive lock and atomic replacement;
summary and plot outputs require new paths. Nothing rewrites the old
`EXPERIMENT_RESULTS.md`. Every ledger entry is reverified whenever it is used.

Comparisons require identical complete validation source/tensor receipts and the
same scorer. Each point remains an individual seed: the scripts do not treat a
best seed as replicated confirmation or combine different recipes into one
average. `audit_goal_artifact.py RUN --max-parameters N` reports a **single-run
candidate** against the requested CA-lDDT/update/parameter bounds; its CLI returns
a nonzero status when any bound fails. It does not certify a publication claim.

`plot_af2_large_simplexfold_checkpoint_progress.py` accepts the same run-directory
arguments and `--labels`. Give snapshots of actual saved checkpoints, with repeated
labels for one curve. Every point in a curve must have the same recipe, source,
seed, and budget contract and a distinct update number. It can show a verified
paused checkpoint, but cannot read intermediate scores from an unbound history
file. Both plots emit source/cohort/seed/checkpoint metadata and a values CSV.
Plotting requires Matplotlib. A stopped run can be inspected with
`summarize_nanofold_run_status.py RUN --allow-paused`; it cannot enter a completed
experiment ledger.

For a fresh E151-inspired run, `run_e151_from_scratch.py` requires all data paths,
a seed, and an explicit positive `--global-context-scale`. Its default only prints
the recipe and command. `--execute` is required to launch; from the main checkout,
also provide a new `--export-dir`. Inside a pinned E07 export, omit that option to
use the current runtime. Exported curriculum bytes must match the original E07
receipt. This is a new explicit recipe, not a reconstruction of missing historical
E151 settings. The old `run_runpod_e01_pilot.sh` filename now runs one explicit local
pinned E01 recipe through `run_research_experiment.py`; it does not provision a
host, fetch source, launch an implicit control, or silently summarize a partial
validation cohort.

`benchmark_simplexfold.py` measures synthetic architecture latency. Its named
variants explicitly select full simplices, faces only, MSA-to-face, or no simplex
adapter while retaining other profile dimensions and scales. Results record the
resolved model configuration, exact synthetic-input and source hashes, seed,
timing samples, and device. Face/tetra counts mean allocated candidate slots,
not unique geometric simplices; disabling a mechanism does not necessarily remove
its allocated parameters. Use a fresh `--json-out` for each attempt.
