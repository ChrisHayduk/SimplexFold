# Reproducible paper experiments: protocol 2

Use `scripts/run_research_experiment.py` for new paper/rebuttal experiments.
The public benchmark CLI now resolves its arguments into this same runner.
Historical Git refs and result artifacts are preserved; their old executables,
checkpoints, cropped metrics, and adaptive checkpoint ladders do not become
protocol-2 evidence by being present in the repository.

## One recipe, one architecture, one run directory

A recipe JSON contains `model_config`, `variant`, optional `model_overrides`,
`data`, `training`, `stages`, `eval_every`, `scorer_root`, and optionally
`max_parameters`. `data` and `training` use the selected architecture's
`DataConfig` and `TrainingConfig` fields. Unknown fields fail. In this runner
`training.epochs` means **optimizer updates**, matching the historical public
benchmark's `--steps`; total training examples are exactly
`epochs × batch_size × grad_accum_steps`.

Supply the regenerated corpus's explicit feature/label directories and train
and public-validation manifests. The runner hashes every selected NPZ, exact
sequence, complete ordered cohort, resolved model/training/data configuration,
all model sources/config profiles, the scorer and residue constants, and the
PyTorch/NumPy/device runtime. It rejects duplicate IDs, train/validation ID or
exact-sequence overlap, invalid observed coordinates, and changed resume
contracts. Independent group ownership and production quality qualification
remain the dataset release/audit's responsibility; this runner does not infer
homology groups from sequence hashes.

Training uses a cyclic shuffled index stream with no dropped or underweighted
last batch. It records the consumed cursor and deterministic feature-batch seed.
By default each consumed batch has a new reproducible feature seed; set
`fixed_feature_seed` explicitly to study fixed feature sampling. E07's
`stochastic_msa_sampling` option is also honored. `sample_recycles: false`
fixes the training recycle count; `null`/`true` sample uniformly from 1 through
`n_cycles`. Evaluation always uses the full declared `n_cycles`.

Stage changes apply cumulatively. Optimizer and sample-stream resets are
explicit, independently recorded choices. Optimizer hyperparameter changes
require an explicit reset; stage boundaries are replayed exactly once across a
pause. Model construction, sample budget, EMA definition and batch dimensions
remain fixed for a run. Initialization uses the declared seed with no implicit
weights-only checkpoint loading.

Validation scores every **whole target**, one at a time, with deterministic
features and isolated RNG. It passes features to the model, retains labels only
for loss/scoring, respects residue gaps/segment flags, and requires a finite
result from every target. No mean silently drops failures. Current-model and
EMA results are separate. Every final or paused result is freshly evaluated
and bound to its exact checkpoint bytes, optimizer step, cohort, scorer version
and denominators.

Checkpoints are replaced atomically after every optimizer update. A process
lock prevents concurrent writers; fresh runs reject nonempty output directories.
`--resume` accepts only this protocol's bound checkpoint. Runtime-only
`--max-runtime-seconds` can change on resume; incomplete runs stop at optimizer
boundaries, finish the current endpoint evaluation, save, and exit 75. It is
not a hard process timeout: reserve enough time for complete validation and
checkpoint persistence. Completed runs exit 0. No jobs or relaunches are implied.

## Pinned historical architectures

`--revision` accepts `main`, `e01`, `e02`, `e03`, `e04`, `e05`, `e07` and exports
the corresponding immutable commit into a **new** `--export-dir`. It applies
the exact, cardinality-checked replacements in
`scripts/research_source_fixes.json`, then overlays reviewed shared training,
loading, preprocessing and reporting functions and installs the v2 executor.
Branch model/configuration classes, simplex numerics, geometry hooks and loss
schedules remain in their pinned architecture. E07-specific runtime controls
are available through the JSON recipe, rather than an undocumented CLI default.
`source_provenance.json` records every original and effective source hash. It
never checks out, resets, commits, or changes a historical ref/result.

```sh
python scripts/run_research_experiment.py \
  --revision e07 --export-dir /path/to/new/e07-runtime \
  --recipe /path/to/recipe.json --output-dir /path/to/new/run

python /path/to/new/e07-runtime/scripts/run_research_experiment.py \
  --recipe /path/to/recipe.json --output-dir /path/to/new/run --resume

python scripts/verify_research_result.py /path/to/new/run
```

The verifier reconciles checkpoint bytes and scientific contract, target IDs
and order, every detailed metric and denominator, and endpoint/exposure counts.
Only verified v2 outputs should enter a new comparison or ledger. Compare final
endpoints at equal examples, with matched feature/crop/MSA/recycling/precision,
scorer, validation cohort, and initialization/allocation seed design. Do not mix
historical development maxima with a fixed-budget final FoldScore.

## E151

`scripts/make_e151_research_recipe.py` writes fresh-initialization, fixed-architecture
recipes using the E07 tracked final/staged/two-stage loss schedules. It explicitly
constructs edge-frame and global/star-context modules before applying runtime
scales. The global-context constructor scale must be specified, rather than
silently treating a missing module as an active ablation. Parameters and module
shapes are saved before training; runtime multipliers cannot activate absent
parameterized modules.

These are **E151-inspired confirmation experiments**, not verified reproductions
of the adaptive historical E147→E151 ladder. The original serialized final
configuration/checkpoint bundle is still missing. For example, the corrected
medium profile with global-context scale 1 has 3,201,970 parameters, versus the
ledger's 3,240,738. Resolving that difference requires the original artifact;
neither silently changing widths nor loading a partial state dict establishes
identity. The tracked 0.5311576097 FoldScore and paper's rounded 0.533 also remain
unreconciled historical claims.

## Verification boundary

CPU tests cover all seven exported architectures with actual model forward,
loss, backward, checkpoint/pause/resume and full evaluation. They include E07
stage resets and live E151-related modules. The broad E07 topology/shape/loss
suite was also run against the corrected exported sources. This proves neither
GPU equivalence nor historical result reproduction. V2 currently executes FP32
with synchronous deterministic feature loading on one device; GPU memory,
throughput, CUDA deterministic kernels, and full-corpus training remain runtime
gates. Distributed execution and AMP are not silently substituted.

The generic `trainer.fit` and `train_af2.py` API uses epochs plus an optional
exact sample cap. It binds model/data/source/manifest content, global and loader
RNG, optimizer, EMA and completed exposure/history counters for epoch-boundary
resume; it rejects stale or unrelated output checkpoints. Validation covers
every full target with fixed recycling and isolated RNG. The loader uses batch
one during validation to avoid cross-target padding effects. Explicit train
and validation manifests require `val_fraction=0`; supplied homology groups
must be disjoint. Plain ID manifests alone do not establish homology isolation.

Standalone overfit utilities remain memorization diagnostics, with finite
losses/updates, explicit selected targets, provenance and fresh output ownership.
Their checkpoint/metric conventions are distinct from v2 confirmation receipts.
The public Modal wrapper defaults to FP32, synchronous loading and a 23-hour
safe-boundary pause, reports paused status and commits the output volume even
on errors. It does not automatically relaunch paid compute. Native cloud GPU
execution and complete production data loading remain unexecuted runtime gates.
