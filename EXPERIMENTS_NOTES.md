# SimplexFold Experiment Notes

## 2026-05-16 E142 Final-Validation Watch; Eval Progress Instrumentation

- Rechecked only the owned Runpod pod `723hbew2jrvxjx` through
  `2026-05-17T00:14:54Z`. E142 is still in final validation watch with
  `phase=evaluating`, `completed_step=9000`, active step `9000`, effective
  batch size `8`, `num_workers=0`, finite last train loss
  `4.721518278121948`, and status mtime `2026-05-16T23:09:53Z`.
- No scored return exists yet. The artifact directory still contains only
  `status_full_msa_to_face.json`, `history_full_msa_to_face.json`, and
  `run_metadata.json`; there is no `results.json`, result CSV,
  `eval_details_full_msa_to_face.csv`, or E142 checkpoint. History still ends
  at inherited E128 step `8500`.
- Stall classification: not terminal yet. Trainer PID `13262` remains alive
  and CPU-active with process time still advancing after `1-21:20:16`, state
  `Rl`, and `194`
  threads; GPU utilization sampled at `0%` with `43101 / 81920` MiB allocated.
  The log still contains only startup/resume lines, and `/proc/13262/wchan`
  returned `0`. Continue monitoring rather than stopping the run while CPU
  progress continues in the final validation path.
- A one-minute interval from `2026-05-16T23:58:39Z` to `23:59:39Z` showed OS
  progress despite unchanged artifacts: trainer CPU time increased from
  `1-15:34:19` to `1-15:57:28`, `rchar` increased from `651303601` to
  `651728284`, and `read_bytes` increased from `3182592` to `3219456`.
  This supports live final-eval compute, not a terminal no-score stall.
- Additional sample through `2026-05-17T00:14:54Z`: artifacts were still
  unchanged, but the process remained runnable and `rchar` increased from
  `660859786` to `661384631` over one minute while RSS increased from
  `1712184` to `1762512` KiB. The current final-eval path computes C-alpha and
  FoldScore-related metrics on CPU, so this remains plausible slow validation.
- Added local runner observability for future short gates: validation now
  reports `active_eval_batch`, `active_eval_batches`, and
  `active_eval_examples` through the live status file, and
  `scripts/summarize_nanofold_run_status.py` includes those counters. This is
  instrumentation only; it does not change architecture, losses, training
  data, validation scoring, or the already-running E142 checkout.
- Successor readiness: `/workspace/SimplexFold_e143` and
  `/workspace/SimplexFold_e144` were missing on the current owned pod despite
  older staging notes, so both were re-cloned from GitHub at branch commit
  `0e4ebd8` without touching the active E142 process. Remote
  `python3 -m py_compile` passed for model/trainer/runner files in both
  checkouts. `/workspace/nanoFold-Competition` exists, and the E128 checkpoint
  exists under `/workspace/SimplexFold_e145`, so E143 is ready as the next
  topology-native fallback if E142 returns weak or becomes terminal no-score.

## 2026-05-16 E141 Final-Step Stall; Pod Stopped

- Rechecked only the owned E141 Runpod pod `5ox436mhzej7j4`. E141 reached the
  final training step but did not return a scored bundle: status stayed at
  `phase=microbatch_done`, `completed_step=8999`, active step `9000`, active
  microbatch `7/8`, effective batch size `8`, and `stopped_early=false`.
  `results.json`, `results.csv`, `eval_details_full_msa_to_face.csv`, and
  `checkpoints/full_msa_to_face_latest.pt` were absent; history still ended at
  inherited E128 step `8500`.
- Stall evidence: `status_full_msa_to_face.json` mtime was
  `2026-05-16T09:00:40Z`, while checks around `2026-05-16T15:06Z` still saw
  the same status and no artifacts. GPU utilization was `0%` with about
  `43055 MiB` allocated. A one-minute interval from `15:07:34Z` to
  `15:08:34Z` showed process CPU time advancing from `utime=131967578,
  stime=961396` to `utime=132047167, stime=961856`, but the status mtime,
  artifact set, and history stayed unchanged. The log still only contained
  startup/resume lines.
- Pulled the available trace locally under ignored
  `artifacts/runpod_traces/e141_stalled_20260516T1508Z/`: the run directory
  contains `run_metadata.json`, `history_full_msa_to_face.json`, and
  `status_full_msa_to_face.json`; the log was also preserved. Added
  `/artifacts/runpod_traces` to `.gitignore` so these local traces cannot be
  staged accidentally.
- Stopped only the owned E141 pod `5ox436mhzej7j4` after trace preservation.
  Record E141 as a failed final-step stall/no-score outcome, not as evidence
  that signed face-cyclic boundary readout helps or hurts. E145 is now the
  next eligible short gate.

## 2026-05-15 Parked Signed-Boundary Recipe Guards

- Added local launch-recipe guards for the documented parked E142, E143, and
  E144 fallbacks. These tests lock run names, step target `9000`, effective
  batch size `8`, crop `256`, MSA depth `64`, no extra MSA/templates, the
  `3,261,974` parameter cap, default `num_workers=0`, each topology-native
  static scale, and each `8500`-to-`9000` runtime ramp.
- This is non-invasive prep only. E141 remains the active owned Runpod run,
  and E145 remains the next launch candidate if E141 returns below threshold
  or fails coherently. Do not launch E142/E143/E144 ahead of E141/E145 without
  an explicit branch-choice reason.
- Focused validation passed:
  `python -m pytest tests/test_nanofold_public_benchmarks.py::test_e141_signed_face_cyclic_recipe_matches_running_gate
  tests/test_nanofold_public_benchmarks.py::test_e142_signed_tetra_coboundary_recipe_matches_documented_gate
  tests/test_nanofold_public_benchmarks.py::test_e143_signed_tetra_to_face_recipe_matches_documented_gate
  tests/test_nanofold_public_benchmarks.py::test_e144_edge_star_residual_recipe_matches_documented_gate
  tests/test_nanofold_public_benchmarks.py::test_e145_outer_residual_context_recipe_matches_documented_gate`
  (`5 passed`), `python -m py_compile
  tests/test_nanofold_public_benchmarks.py`, and
  `../../.venv/bin/ruff check --select F821,F822,F823
  tests/test_nanofold_public_benchmarks.py`. Pytest cache writes were blocked
  by the sandbox, but the selected tests passed.

## 2026-05-15 E141 Owned-Pod Heartbeat 15:37Z

- Rechecked only the owned E141 Runpod pod `5ox436mhzej7j4` at
  `2026-05-15T15:37:26Z`. The process is still alive as PID `576`, elapsed
  `05:04:06`, with `completed_step=8612`, `active_step=8613`, active
  microbatch `6/8`, effective batch size `8`, target step `9000`, and
  `stopped_early=false`.
- No returned bundle exists yet: `results.json`, `results.csv`,
  `eval_details_full_msa_to_face.csv`, and
  `checkpoints/full_msa_to_face_latest.pt` are still absent. The inherited
  history remains at `18` rows with last history step `8500`, so
  `EXPERIMENT_RESULTS.md` remains unchanged.
- E141 remains slow but coherent. Continue monitoring this owned pod only; do
  not launch E145 while E141 is advancing.

## 2026-05-15 E141 Owned-Pod Heartbeat 15:02Z

- Rechecked only the owned E141 Runpod pod `5ox436mhzej7j4` at
  `2026-05-15T15:02:29Z`. The process is still alive as PID `576`, elapsed
  `04:29:09`, with `completed_step=8600`, `active_step=8601`, active
  microbatch `1/8`, effective batch size `8`, target step `9000`, and
  `stopped_early=false`.
- No returned bundle exists yet: `results.json`, `results.csv`,
  `eval_details_full_msa_to_face.csv`, and
  `checkpoints/full_msa_to_face_latest.pt` are still absent. History still has
  `18` rows and last history step `8500`, so `EXPERIMENT_RESULTS.md` remains
  unchanged.
- E141 is slow but coherent. Do not launch E145 or stop the E141 pod while
  status heartbeats continue advancing.

## 2026-05-15 Goal Completion Audit

- Objective decomposed into concrete gates: keep SimplexFold
  AF2-medium-matched within `3,261,974` parameters, train/evaluate on NanoFold
  with effective batch size `8`, confirm validation C-alpha lDDT `>0.7` at
  `>=30000` completed steps, keep all candidate changes topology/simplicial in
  spirit, run experiments on owned Runpod pods, and maintain `PLAN.md`,
  `EXPERIMENTS.md`, `EXPERIMENTS_NOTES.md`, and final outcomes in
  `EXPERIMENT_RESULTS.md`.
- Current evidence says the goal is not achieved. `python
  scripts/audit_experiment_results.py EXPERIMENT_RESULTS.md` reports best
  returned score E128 at `val_lddt_ca=0.4311` / step `8500`, zero short gates
  at or above `0.45`, and zero `>=30000`-step confirmations above `0.7`.
- Artifact-level evidence for the current best E128 passes the verifier side
  of the contract but fails the actual goal gates: `completed_steps=8500`,
  effective batch size `8`, parameters `3,240,738 <= 3,261,974`, `1000` eval
  rows, checkpoint present, and `stopped_early=false`; goal audit exits `1`
  because `0.4311 < 0.7` and `8500 < 30000`.
- Active next evidence source remains E141 on owned Runpod pod
  `5ox436mhzej7j4`. It is still pre-eval, so no new result row or goal audit
  exists yet. Do not mark the goal complete, launch a 30k confirmation, or
  launch E145 until E141 returns or fails coherently.

## 2026-05-15 E141 Heartbeat After Goal-Audit Fix

- Rechecked only the owned active E141 Runpod pod `5ox436mhzej7j4` after
  pushing the goal-audit exit-code fix. The process is still alive as PID
  `576`, elapsed `03:59:46`, with `completed_step=8589`, `active_step=8590`,
  active microbatch `3/8`, effective batch size `8`, target step `9000`, and
  `stopped_early=false`.
- No returned bundle exists yet: `results.json`, `results.csv`,
  `eval_details_full_msa_to_face.csv`, and
  `checkpoints/full_msa_to_face_latest.pt` are still absent. The inherited
  history remains at `18` rows with last history step `8500`, so
  `EXPERIMENT_RESULTS.md` correctly remains unchanged.

## 2026-05-15 Artifact Goal Audit Exit-Code Fix

- Fixed `scripts/audit_goal_artifact.py` so the script entry point exits
  nonzero when the artifact verifies but fails any explicit goal gate. The
  importable `main()` still returns the `GoalArtifactAudit` object for tests
  and helper composition, while the new `cli()` wrapper returns `0` only for a
  true goal-ready candidate.
- Added regression coverage for the CLI wrapper and the exact
  `python scripts/audit_goal_artifact.py ...` entry point. The real E128
  metadata dry run now prints `Goal-ready candidate: FAIL` and exits with
  code `1`, as expected for a below-target 8500-step short gate.
- Updated the experiment contract in `EXPERIMENTS.md`: goal-audit nonzero
  exits are expected for ordinary non-goal short gates, so returned results
  should still be recorded after artifact verification passes.
- Updated heartbeat automation `check-simplexfold-e57-runpod` with the same
  interpretation: capture goal-audit output/exit code, but keep eval-detail
  analysis and result recording moving after verifier success when a short
  gate simply fails the full 30k/0.7 readiness criteria.
- Focused validation passed:
  `python -m pytest tests/test_audit_goal_artifact.py
  tests/test_verify_nanofold_benchmark_artifacts.py` (`21 passed`),
  `python -m py_compile scripts/audit_goal_artifact.py
  tests/test_audit_goal_artifact.py scripts/verify_nanofold_benchmark_artifacts.py
  tests/test_verify_nanofold_benchmark_artifacts.py`,
  `../../.venv/bin/ruff check --select F821,F822,F823
  scripts/audit_goal_artifact.py tests/test_audit_goal_artifact.py
  scripts/verify_nanofold_benchmark_artifacts.py
  tests/test_verify_nanofold_benchmark_artifacts.py`, and the E128 artifact
  goal-audit dry run with asserted exit code `1`.

## 2026-05-15 Current Handoff Wording Cleanup

- Rechecked the active handoff text while E141 continues. Updated the current
  `PLAN.md` return handoff to point at the E141 verifier template rather than
  the older E140/E141 pair, and updated E145's current decision rule so the
  only active threshold-clearing branch it can defer to is E141. Historical
  notes that describe past E140/E141 monitoring windows were left intact.

## 2026-05-15 Runpod Status Heartbeat Hardening

- Hardened `scripts/run_nanofold_public_benchmarks.py` against the E140
  failure mode. Live `status_full_msa_to_face.json` updates now write through
  a temporary file and atomically replace the prior status. If status I/O
  raises `OSError`, the runner logs a warning, preserves the last good status
  file, and keeps training instead of killing a multi-hour Runpod job.
- Focused validation passed:
  `python -m pytest tests/test_nanofold_public_benchmarks.py::test_run_status_payload_tracks_live_progress
  tests/test_nanofold_public_benchmarks.py::test_write_run_status_file_is_best_effort_and_preserves_prior_status`,
  `python -m py_compile scripts/run_nanofold_public_benchmarks.py
  tests/test_nanofold_public_benchmarks.py`, and
  `../../.venv/bin/ruff check --select F821,F822,F823
  scripts/run_nanofold_public_benchmarks.py tests/test_nanofold_public_benchmarks.py`.

## 2026-05-15 E140 Failed Pre-Eval; E141 Still Active

- Rechecked only the two owned Runpod pods. E140 on pod
  `c67fbk189vnvfp` has no live Python PID `55949`, no `results.json`, no
  `results.csv`, no `eval_details_full_msa_to_face.csv`, and no checkpoint.
  Its pulled history still ends at inherited E128 step `8500` with
  `val_lddt_ca=0.4311057258844376`; the pulled status file is empty after a
  failed write.
- The preserved E140 log ends with `OSError: [Errno 5] Input/output error`
  inside `write_run_status` while writing `status_full_msa_to_face.json`.
  Record E140 as a failed pre-eval/no-score outcome, not as evidence that the
  selected-boundary expansion objective helped or hurt. After pulling the
  trace, stopped only the owned E140 pod `c67fbk189vnvfp`; E141 was left
  running.
- E141 on pod `5ox436mhzej7j4` remains alive with PID `576`, completed step
  `8577`, active step `8578`, active microbatch `1/8`, effective batch size
  `8`, and no result bundle, eval details, result CSV, or checkpoint yet.
  Continue monitoring E141 only; do not launch E145 while E141 is slow but
  coherent.
- Updated heartbeat automation `check-simplexfold-e57-runpod` to monitor only
  E141 as the active run. It now treats E140 as documented/stopped, preserves
  the E141 verifier/audit handoff, and keeps E145 gated behind a returned
  below-threshold or coherent failed E141 outcome.

## 2026-05-15 E145 Branch-Tip Audit

- Re-audited the parked E145 path while E140/E141 were being monitored. The
  implementation remains topology-native: `cell_outer_edge_context` pools
  directed external edges for each selected face/tetra cell,
  `parameter_free_outer_edge_context_delta` splits symmetric versus oriented
  pair context and folds it into the active cochain width without learned
  weights, and the adapter RMS-matches/gates that update into face/tetra
  states before their boundary information returns to the pair trunk.
- The branch-tip checks still support the AF2-medium budget claim:
  `simplex_outer_edge_residual_context_scale=0.25` changes the cochain path
  but adds no parameters to `simplexfold_medium_param_matched`; the documented
  E145 recipe remains guarded at effective batch size `8`, crop `256`, MSA
  `64`, no templates, `--num-workers 4`, max-parameter cap `3261974`, and the
  `8500`-to-`9000` runtime ramp.
- Validation passed:
  `python -m pytest tests/test_simplex.py::test_parameter_free_outer_edge_context_delta_uses_external_directed_edges
  tests/test_simplex.py::test_parameter_free_outer_edge_context_adapter_scale_changes_outputs_without_new_parameters
  tests/test_simplex.py::test_outer_edge_residual_context_runtime_scale_gates_parameter_free_path
  tests/test_trainer.py::test_simplicial_parameter_free_outer_edge_context_adds_no_parameters
  tests/test_trainer.py::test_simplicial_runtime_overrides_reach_model_path
  tests/test_trainer.py::test_model_inputs_add_training_only_simplex_curricula
  tests/test_trainer.py::test_trainer_cli_accepts_simplex_star_context_overrides
  tests/test_nanofold_public_benchmarks.py::test_model_config_override_flags_are_accepted_by_cli_parser
  tests/test_nanofold_public_benchmarks.py::test_runtime_simplex_message_scales_ramp_and_enter_model_inputs
  tests/test_nanofold_public_benchmarks.py::test_evaluate_uses_runtime_simplex_overrides_for_validation
  tests/test_nanofold_public_benchmarks.py::test_model_config_overrides_preserve_resume_compatible_variant_name
  tests/test_nanofold_public_benchmarks.py::test_e145_outer_residual_context_recipe_matches_documented_gate`
  (`12 passed`), `python -m py_compile minalphafold/simplex.py
  minalphafold/trainer.py scripts/run_nanofold_public_benchmarks.py
  tests/test_simplex.py tests/test_trainer.py
  tests/test_nanofold_public_benchmarks.py`,
  `../../.venv/bin/ruff check --select F821,F822,F823` on the same
  model/runner/test files, and `git diff --check`.

## 2026-05-15 Result-Table Audit Helper

- Added `scripts/audit_experiment_results.py`, a read-only audit helper for
  `EXPERIMENT_RESULTS.md`. It parses the Markdown table, reports the best
  numeric validation C-alpha lDDT row, counts any short gates above the
  `0.45` threshold, and separately counts true 30,000-step confirmations above
  the `0.7` target using the final/stop lDDT.
- Current live-table audit output confirms the experiment state: best returned
  score is E128 at `val_lddt_ca=0.4311` and step `8500`; there are `0` short
  gates at or above `0.45` before `30,000` steps; and there are `0` confirmed
  `30,000`-step runs above `0.7`.
- Focused validation passed:
  `python -m pytest tests/test_audit_experiment_results.py`,
  `python -m py_compile scripts/audit_experiment_results.py
  tests/test_audit_experiment_results.py`,
  `../../.venv/bin/ruff check --select F821,F822,F823
  scripts/audit_experiment_results.py tests/test_audit_experiment_results.py`,
  `python scripts/audit_experiment_results.py EXPERIMENT_RESULTS.md`, and
  `git diff --check`.
- Updated heartbeat automation `check-simplexfold-e57-runpod` in place so a
  coherent returned E140/E141 handoff now runs
  `python scripts/audit_experiment_results.py EXPERIMENT_RESULTS.md` after
  verifier/analyzer/recording, before committing. The owned-pod scope,
  Runpod-management limits, E145 gating rule, and 30-minute schedule are
  unchanged.
- Tightened the helper to state that parameter-budget evidence is not present
  in `EXPERIMENT_RESULTS.md` itself. A future apparent `>0.7` row still needs
  artifact metadata/verifier evidence for the `<=3261974` parameter cap before
  it can count toward the goal. Focused validation passed:
  `python -m pytest tests/test_audit_experiment_results.py`,
  `python -m py_compile scripts/audit_experiment_results.py
  tests/test_audit_experiment_results.py`,
  `../../.venv/bin/ruff check --select F821,F822,F823
  scripts/audit_experiment_results.py tests/test_audit_experiment_results.py`,
  `python scripts/audit_experiment_results.py EXPERIMENT_RESULTS.md`, and
  `git diff --check`.

## 2026-05-15 Artifact Goal Audit Helper

- Added `scripts/audit_goal_artifact.py`, a verifier-backed audit for one
  returned NanoFold artifact directory. It composes the existing returned-run
  verifier with the explicit goal criteria: validation C-alpha lDDT must exceed
  `0.7`, completed steps must be at least `30,000`, the run must not be
  stopped early, and the artifact must verify under the supplied parameter cap
  such as `3261974`.
- Dry-ran it on the existing E128 returned artifact. The verifier-backed
  evidence passed for completed step `8500`, effective batch size `8`,
  parameters `3,240,738 <= 3,261,974`, `1000` eval rows, checkpoint present,
  and `stopped_early=false`; the goal audit correctly failed E128 because
  `val_lddt_ca=0.4311` is below `0.7` and the run is not a 30,000-step
  confirmation.
- Updated heartbeat automation `check-simplexfold-e57-runpod` in place so a
  coherent returned E140/E141 handoff now runs this artifact goal audit after
  verifier success and before eval-detail analysis / result recording. The
  owned-pod scope, E145 gating rule, and 30-minute schedule are unchanged.
- Focused validation passed:
  `python -m pytest tests/test_audit_goal_artifact.py
  tests/test_verify_nanofold_benchmark_artifacts.py`,
  `python -m py_compile scripts/audit_goal_artifact.py
  tests/test_audit_goal_artifact.py scripts/verify_nanofold_benchmark_artifacts.py
  tests/test_verify_nanofold_benchmark_artifacts.py`,
  `../../.venv/bin/ruff check --select F821,F822,F823
  scripts/audit_goal_artifact.py tests/test_audit_goal_artifact.py
  scripts/verify_nanofold_benchmark_artifacts.py
  tests/test_verify_nanofold_benchmark_artifacts.py`,
  the E128 artifact-audit dry run, and `git diff --check`.
- Extended the artifact goal audit with the same `--metadata KEY=VALUE`
  expectations used by `scripts/verify_nanofold_benchmark_artifacts.py`, via
  the public `parse_metadata_expectation` helper. This lets a future returned
  candidate be checked for score/steps/parameter cap and for exact
  topology-recipe metadata in one artifact-level audit command.
- Metadata dry-run behavior is intentionally strict. Asking E128 to prove the
  absent key `simplex_outer_edge_residual_context_scale=null` failed with
  `Missing metadata key`, while checking present E128 metadata
  (`run_name`, `model_config`, `simplex_face_top_k`, `simplex_tetra_top_k`,
  and `simplex_triangle_attention_bias_scale`) passed artifact verification
  and still failed the goal gates for the right reasons:
  `val_lddt_ca=0.4311 < 0.7` and `8500 < 30000` steps.
- Focused validation passed after the metadata-threading change:
  `python -m pytest tests/test_audit_goal_artifact.py
  tests/test_verify_nanofold_benchmark_artifacts.py` (`18 passed`),
  `python -m py_compile scripts/audit_goal_artifact.py
  tests/test_audit_goal_artifact.py scripts/verify_nanofold_benchmark_artifacts.py
  tests/test_verify_nanofold_benchmark_artifacts.py`,
  `../../.venv/bin/ruff check --select F821,F822,F823
  scripts/audit_goal_artifact.py tests/test_audit_goal_artifact.py
  scripts/verify_nanofold_benchmark_artifacts.py
  tests/test_verify_nanofold_benchmark_artifacts.py`, the E128 metadata audit
  dry run, and `git diff --check`.
- Added explicit goal-level artifact audit command templates beside the
  returned-artifact verifier templates for E140, E141, and E145 in
  `EXPERIMENTS.md`. The E140/E141 templates intentionally omit
  `--expected-num-workers`; the E145 template includes `--expected-num-workers
  4` and the outer-edge residual context metadata.
- Updated heartbeat automation `check-simplexfold-e57-runpod` in place so
  returned E140/E141 handoffs now run the exact goal-audit templates in
  `EXPERIMENTS.md` after verifier success, before eval-detail analysis and
  result recording. Owned-pod scope, Runpod-management limits, E145 gating,
  and the 30-minute schedule are unchanged.

## 2026-05-15 Owned Runpod Heartbeat

- 2026-05-15T13:26Z Rechecked only the two owned active Runpod pods. E140 on
  pod `c67fbk189vnvfp` is alive with PID `55949`, active step `8611`,
  completed step `8610`, active microbatch `1/8`, effective batch size `8`,
  and no `results.json`, eval-details CSV, result CSV, or checkpoint yet.
- E141 on pod `5ox436mhzej7j4` is alive with PID `576`, active step `8565`,
  completed step `8564`, active microbatch `3/8`, effective batch size `8`,
  and no `results.json`, eval-details CSV, result CSV, or checkpoint yet.
- Both histories still end at inherited E128 step `8500` with
  `val_lddt_ca=0.4311057258844376`, so `EXPERIMENT_RESULTS.md` correctly
  remains unchanged. Do not launch E145 or stop any owned pod while E140/E141
  are slow but still coherently progressing.

## 2026-05-15 Result Recording Helper Cleanup

- Promoted the result/history JSON loaders in
  `scripts/format_experiment_result_row.py` from private helpers to public
  `load_result` / `load_history` functions so
  `scripts/record_experiment_result.py` can compose the formatter without
  importing underscored internals.
- Added a focused regression test that the public helpers select the requested
  benchmark variant and preserve history rows. This keeps the E140/E141 return
  workflow on the supported result-recording path: verify artifacts, format the
  row, upsert `EXPERIMENT_RESULTS.md`, and refresh the summary.

## 2026-05-12 Sparse-Cell Branch

- E82 completed on owned Runpod pod `o1dy17ouv8w5mz` as
  `e82_sparse_topk_from_e79_s7500_c256_m64`.
- Result: `completed_steps=7500`, `train_examples=60000`,
  `parameters=3,154,242`, `val_lddt_ca=0.39241083338856697`,
  FoldScore `0.3788187075406313`, `val_ca_drmsd=10.252252995967865`,
  and predicted/true C-alpha radius `11.336281925439835 /
  15.403406739234924`.
- Selected-complex diagnostics improved over E79: face/tetra boundary lDDT
  `0.7135092802345753` / `0.6986573338508606`, boundary length MAE
  `1.1559935696423054` / `1.257934581488371`, boundary-edge mean degree
  `12.294965147972107` / `34.21513247489929`, and unique-edge fraction
  `0.08146965243978029` / `0.029440750303401327`.
- Interpretation: keep. Fixed sparse face/tetra caps improved primary lDDT,
  FoldScore, dRMSD, radius, and selected-boundary diagnostics. This supports
  the paper-aligned topology-construction path rather than an immediate pivot
  to another loss.
- Pulled E82 artifacts locally under
  `artifacts/nanofold_public_benchmarks/e82_sparse_topk_from_e79_s7500_c256_m64/`
  and `logs/e82_sparse_topk_from_e79.log`.
- Launched E83 on the same owned pod as
  `e83_sparse_topk_from_e82_s8000_c256_m64`, PID `4020`, log
  `/workspace/SimplexFold/logs/e83_sparse_topk_from_e82.log`.
- E83 resumes the E82 checkpoint from step 7500 to 8000 with the same fixed
  sparse caps (`--simplex-face-top-k 24`, `--simplex-tetra-top-k 48`) and the
  same selected-boundary / edge-frame / light-geometry recipe.
- E83 completed and was pulled locally under
  `artifacts/nanofold_public_benchmarks/e83_sparse_topk_from_e82_s8000_c256_m64/`
  plus `logs/e83_sparse_topk_from_e82.log`.
- E83 result: `completed_steps=8000`, `train_examples=64000`,
  `parameters=3,154,242`, `val_lddt_ca=0.3875777218490839`,
  FoldScore `0.374724255874753`, `val_ca_drmsd=10.3538738489151`,
  and predicted/true C-alpha radius `11.175740718841553 /
  15.403406739234924`.
- E83 selected-complex diagnostics softened from E82: face/tetra boundary
  lDDT `0.7033939026296139` / `0.6880575008690357`, boundary length MAE
  `1.229560237377882` / `1.334547944366932`, contraction fraction
  `0.6593854054808617` / `0.6625464409589767`, boundary-edge mean degree
  `12.405892848968506` / `34.80449962615967`, and unique-edge fraction
  `0.08078695792292354` / `0.02897821614822659`.
- Interpretation: reject E83 as a primary branch. A plain fixed-cap
  continuation fell below E82 and slightly below E79, so the sparse selector
  needs a construction change rather than more blind continuation.
- Launched E81 on the same owned pod as
  `e81_degree_penalty_from_e82_s8000_c256_m64`, PID `4434`, log
  `/workspace/SimplexFold/logs/e81_degree_penalty_from_e82.log`.
- E81 resumes the E82 checkpoint from step 7500 to 8000 with the same fixed
  sparse caps and adds `--simplex-cell-score-degree-penalty 0.75` to
  down-rank candidate cells that reuse overrepresented boundary edges.
- E81 completed and was pulled locally under
  `artifacts/nanofold_public_benchmarks/e81_degree_penalty_from_e82_s8000_c256_m64/`
  plus `logs/e81_degree_penalty_from_e82.log`.
- E81 result: `completed_steps=8000`, `train_examples=64000`,
  `parameters=3,154,242`, `val_lddt_ca=0.39799308963119984`, FoldScore
  `0.3825651463121176`, `val_ca_drmsd=10.095411986112595`, and
  predicted/true C-alpha radius `11.497344255447388 / 15.403406739234924`.
- E81 selected-complex diagnostics improved over E82/E83: face/tetra boundary
  lDDT `0.7335464172065258` / `0.717821329832077`, boundary length MAE
  `1.0733469985425472` / `1.1726838611066341`, contraction fraction
  `0.5781249664723873` / `0.5791049208492041`, boundary-edge mean degree
  `11.727135837078094` / `33.19685733318329`, and unique-edge fraction
  `0.08556290429844117` / `0.030409362497511753`.
- Interpretation: keep E81 as the new primary-lDDT branch. Penalizing
  candidate cells that reuse overrepresented boundary edges improved lDDT,
  FoldScore, dRMSD, selected-boundary lDDT, boundary length error,
  contraction, and boundary-edge reuse. This is a clean topology-construction
  win rather than an output-side metric hack.
- Launched E84 on the same owned pod as
  `e84_degree_penalty_from_e81_s8500_c256_m64`, PID `4828`, log
  `/workspace/SimplexFold/logs/e84_degree_penalty_from_e81.log`.
- E84 resumes the E81 checkpoint from step 8000 to 8500 with the same fixed
  sparse caps and degree penalty (`--simplex-cell-score-degree-penalty 0.75`).
  A status check at `2026-05-12T10:43:02Z` confirmed the process was alive and
  no `results.json` had returned yet.
- E84 completed and was pulled locally under
  `artifacts/nanofold_public_benchmarks/e84_degree_penalty_from_e81_s8500_c256_m64/`
  plus `logs/e84_degree_penalty_from_e81.log`, excluding the checkpoint.
- E84 result: `completed_steps=8500`, `train_examples=68000`,
  `parameters=3,154,242`, `val_lddt_ca=0.39635433070361614`, FoldScore
  `0.37667176872491837`, `val_ca_drmsd=10.404743939638138`, and
  predicted/true C-alpha radius `11.024506539106369 / 15.403406739234924`.
- E84 selected-complex diagnostics regressed from E81: face/tetra boundary
  lDDT `0.7215762287378311` / `0.7045434713363647`, boundary length MAE
  `1.143507219851017` / `1.2522367648780346`, contraction fraction
  `0.5862371735274792` / `0.5870118048042059`, boundary-edge mean degree
  `11.508779883384705` / `32.077237367630005`, and unique-edge fraction
  `0.08712434698956144` / `0.031449490492374596`.
- Interpretation: reject E84 as a continuation. It slightly improved
  boundary-edge reuse, but primary lDDT, FoldScore, dRMSD, radius, selected
  boundary lDDT, boundary length error, and contraction all moved the wrong
  way relative to E81.

## 2026-05-12 PDF Reference Pass

- Verified the two user-provided PDFs are saved locally under
  `references/papers/`:
  `hands_on_geometric_deep_learning_nodes_to_complexes.pdf` and
  `2509.03885v1.pdf`.
- The saved copies hash-match the files in `/Users/christopherhayduk/Downloads/`.
  They remain ignored by git pending redistribution-rights confirmation, but
  are available in the repo working tree for future experiment reference.
- Re-extracted both PDFs with `pdftotext -layout` and reread the full text:
  the TDL guide is 28 pages and the Topotein paper is 22 pages, totaling
  about 14.5k extracted words.
- The useful experiment pressure is not a new generic lDDT loss. The papers
  both argue that the topological domain construction and the cochain message
  routes are the core modeling choices.
- The guide reinforces the rule that SimplexFold changes should affect the
  selected neighbor graph, sparse face/tetra cells, incidence relations,
  intra-rank aggregation, inter-rank aggregation, or topology-aware
  diagnostics.
- Topotein adds protein-specific guidance: preserve directed incidence,
  consider outer-edge neighborhoods for cell-to-cell communication, use
  edge-centric scalarization for orientation-aware updates, and avoid shallow
  higher-rank features that do not have dedicated update routes.
- This supports the E81/E84 degree-penalized sparse-cell branch. E81 alters
  the learned combinatorial complex by changing which face/tetra cochains
  exist, and the improved boundary-edge reuse metrics are exactly the kind of
  topology-aware evidence the PDFs suggest tracking.
- If E84 does not produce a stable gain, the next paper-aligned idea should be
  incidence-normalized boundary or outer-edge transport. The current sparse
  branch has strong selected-boundary lDDT but still nontrivial boundary-edge
  reuse, so edge-cell incidence degree remains the natural pressure point.
- E85 local implementation prepared while E84 runs: added zero-parameter
  `simplex_boundary_incidence_normalization`. This is intentionally different
  from rejected E77's final pair-readout attenuation. E85 normalizes the
  selected cochain transport itself: face edge-to-cell updates use the mean
  inverse boundary incidence degree, face/tetra cell-to-edge messages are
  scaled per selected boundary edge before scattering into pair state, and the
  tetra face-to-tetra message is scaled by tetra boundary incidence degree.
- E85 local validation passed:
  `python -m py_compile minalphafold/simplex.py minalphafold/model_config.py scripts/run_nanofold_public_benchmarks.py`;
  `python -m pytest tests/test_simplex.py::test_boundary_incidence_weights_normalize_selected_cell_edges tests/test_simplex.py::test_boundary_incidence_normalization_changes_cochain_transport tests/test_nanofold_public_benchmarks.py::test_model_config_override_flags_are_accepted_by_cli_parser tests/test_trainer.py::test_simplicial_boundary_incidence_normalization_adds_no_parameters`;
  `python -m pytest tests/test_simplex.py tests/test_nanofold_public_benchmarks.py tests/test_trainer.py`
  reported `154 passed`.
- Do not sync or launch E85 while E84 is active. If E84 regresses from E81 or
  shows no stable lDDT gain with continued boundary-edge reuse, launch E85 from
  the strongest E81/E84 checkpoint as a short gate with the E84 recipe plus
  `--simplex-boundary-incidence-normalization 1.0`.
- E84 regressed, so E85 was synced to the same owned pod and launched as
  `e85_incidence_norm_from_e81_s8500_c256_m64`, PID `5476`, log
  `/workspace/SimplexFold/logs/e85_incidence_norm_from_e81.log`. It resumes
  the stronger E81 checkpoint from step 8000 to 8500 with the E84 recipe plus
  `--simplex-boundary-incidence-normalization 1.0`. Startup verification at
  `2026-05-12T11:19:55Z` confirmed the process was active, the run metadata
  path existed, and the checkpoint loaded cleanly.
- E86 inspection after E85 launch: the code already has the directed
  outgoing/incoming outer-edge context path through
  `simplex_outer_edge_context_scale`. Do not add a duplicate module. If E85
  still leaves a strong selected-boundary complex but weak global/FoldScore
  geometry, revisit outer-edge context by combining the existing runtime-gated
  outer-edge path with sparse cells and E85 incidence normalization.
- E86 code sanity check: `simplex_outer_edge_context_runtime_scale_at_step`
  already ramps the training-time override, `model_inputs_from_batch` emits
  `simplex_outer_edge_context_scale_override`, and the adapter consumes that
  override for both face and tetra context updates. The existing tests cover
  parser/runtime gating and the parameter budget. A cautious post-E85 gate
  should allocate the context modules with
  `--simplex-outer-edge-context-scale 0.05`, but ramp runtime contribution
  only from `0.0` to `0.025` over the 8000-8500 gate, retaining incidence
  normalization and sparse degree-penalized cells. Do not launch this
  automatically while E85 is active.
- E87 local implementation prepared while E85 runs: added zero-parameter
  `simplex_boundary_readout_directionality` plus training-time runtime ramp
  flags. This keeps the previous symmetric face/tetra boundary scatter at
  `0.0`, but can blend toward a directed boundary-edge readout that writes
  selected cochain messages only into the selected source/target pair
  orientation. The change is topological rather than metric-driven: it
  preserves directed incidence information in the simplex-to-pair cochain
  route.
- E87 local validation passed:
  `python -m py_compile minalphafold/simplex.py minalphafold/evoformer.py minalphafold/model.py minalphafold/trainer.py scripts/run_nanofold_public_benchmarks.py`;
  `python -m pytest tests/test_simplex.py::test_boundary_readout_directionality_preserves_pair_orientation tests/test_simplex.py::test_boundary_readout_directionality_override_gates_pair_readout tests/test_nanofold_public_benchmarks.py::test_model_config_override_flags_are_accepted_by_cli_parser tests/test_nanofold_public_benchmarks.py::test_runtime_simplex_message_scales_ramp_and_enter_model_inputs tests/test_nanofold_public_benchmarks.py::test_evaluate_uses_runtime_simplex_overrides_for_validation tests/test_trainer.py::test_model_inputs_add_training_only_simplex_curricula tests/test_trainer.py::test_simplicial_boundary_readout_directionality_adds_no_parameters`;
  `python -m pytest tests/test_simplex.py tests/test_nanofold_public_benchmarks.py tests/test_trainer.py`
  reported `157 passed`.
- E85 live health check at `2026-05-12T11:56:18Z`: no `results.json` yet.
  The process is still PID `5476` on the owned pod, inherited history is
  present through E81 step 8000, and a five-sample `nvidia-smi` check showed
  H100 utilization ranging from `0%` to `67%` with `13417 MiB` allocated.
  Interpretation: quiet logs are expected before the 8500-step validation;
  leave the run undisturbed.
- E85 completed and was pulled locally under
  `artifacts/nanofold_public_benchmarks/e85_incidence_norm_from_e81_s8500_c256_m64/`
  plus `logs/e85_incidence_norm_from_e81.log`, excluding the checkpoint.
- E85 result: `completed_steps=8500`, `train_examples=68000`,
  `parameters=3,154,242`, `val_lddt_ca=0.38579474948346615`, FoldScore
  `0.37667502649128437`, `val_ca_drmsd=10.111229300498962`, and
  predicted/true C-alpha radius `11.705282628536224 / 15.403406739234924`.
- E85 selected-complex diagnostics regressed from E81: face/tetra boundary
  lDDT `0.7265209071338177` / `0.7089754231274128`, boundary length MAE
  `1.1374360658228397` / `1.2519447579979897`, contraction fraction
  `0.6081784218549728` / `0.6088353879749775`, boundary-edge mean degree
  `11.795296669006348` / `33.32111215591431`, and unique-edge fraction
  `0.08507029338800498` / `0.030259001247997887`.
- Interpretation: reject E85. Normalizing selected edge-cell incidences inside
  the cochain transport did not reduce boundary-edge reuse and erased the E81
  primary-lDDT gain. The next paper-aligned gate should be weak directed
  outer-edge transport on the sparse degree-penalized complex, using incidence
  normalization only as a controlled communication ingredient rather than as a
  standalone continuation.
- Launched E86 on the same owned pod as
  `e86_weak_outer_edge_from_e81_s8500_c256_m64`, Python PID `6369`, log
  `/workspace/SimplexFold/logs/e86_weak_outer_edge_from_e81.log`.
- E86 resumes the E81 checkpoint from step 8000 to 8500, keeps the
  degree-penalized fixed sparse-cell complex and selected-boundary realization
  recipe, keeps `--simplex-boundary-incidence-normalization 1.0` as a
  controlled cochain-transport ingredient, and adds weak directed outer-edge
  context with `--simplex-outer-edge-context-scale 0.05` plus runtime ramp
  `0.0 -> 0.025` over steps 8000-8500.
- E86 startup verification at `2026-05-12T12:24:15Z` confirmed the run was
  active, the run metadata existed, and the E81 checkpoint loaded cleanly:
  `1244` tensors loaded and `48` new/missing tensors initialized for the
  outer-edge context path.
- E88 local implementation prepared while E86 runs: added a training-time
  runtime override for `simplex_segment_cell_scale`. This lets a resumed model
  allocate the existing latent contiguous segment-cell path but ramp its
  segment-to-face contribution gently, instead of switching a fresh rank-2
  cochain route on at full strength. The motivation is Topotein's
  secondary-structure-cell hierarchy, adapted without DSSP/SSE labels by using
  official sequence/MSA/pair features and recycled geometry only.
- E88 is a fallback after E86/E87, not an active run. A cautious gate should
  resume the strongest sparse-complex checkpoint, allocate
  `--simplex-segment-cell-scale 0.05 --simplex-segment-radius 4
  --simplex-c-segment 12`, and ramp runtime scale from `0.0` to `0.05` across
  the 500-step gate while keeping the degree-penalized sparse selector and
  selected-boundary recipe fixed.
- E88 local validation passed:
  `python -m py_compile minalphafold/simplex.py minalphafold/evoformer.py
  minalphafold/model.py minalphafold/trainer.py
  scripts/run_nanofold_public_benchmarks.py`;
  `python -m pytest
  tests/test_simplex.py::test_segment_cells_change_face_mediated_outputs_within_adapter
  tests/test_nanofold_public_benchmarks.py::test_model_config_override_flags_are_accepted_by_cli_parser
  tests/test_nanofold_public_benchmarks.py::test_runtime_simplex_message_scales_ramp_and_enter_model_inputs
  tests/test_nanofold_public_benchmarks.py::test_evaluate_uses_runtime_simplex_overrides_for_validation
  tests/test_trainer.py::test_model_inputs_add_training_only_simplex_curricula
  tests/test_trainer.py::test_simplicial_segment_cells_stay_within_medium_budget`
  reported `6 passed`.
- Broader E88 validation also passed:
  `python -m pytest tests/test_simplex.py
  tests/test_nanofold_public_benchmarks.py tests/test_trainer.py` reported
  `157 passed`.
- Added outer-edge topology diagnostics for future returned runs. The adapter
  now exposes `simplex_neighbor_indices`, and the NanoFold runner reports
  selected face/tetra outer-edge mean degree, max degree, and active fraction
  whenever neighbor indices are present. These are diagnostics only; they add
  no training objective and do not change model parameters.
- Outer-edge diagnostic validation passed:
  `python -m py_compile minalphafold/simplex.py
  scripts/run_nanofold_public_benchmarks.py`;
  `python -m pytest
  tests/test_nanofold_public_benchmarks.py::test_simplex_topology_metrics_report_boundary_reuse`;
  `python -m pytest tests/test_simplex.py
  tests/test_nanofold_public_benchmarks.py tests/test_trainer.py` reported
  `157 passed`.
- E89 local implementation prepared while E86 runs: added separate training
  runtime gates for simplex readout into pair/edge states and residue/single
  states. This reuses the existing adapter overrides and adds no parameters.
  The topological motivation is to test whether selected face/tetra cochains
  should write primarily back into the AF2-style pair tensor `Z_ij`, while
  damping direct single-stream perturbations that may disrupt the structure
  module.
- E89 is a fallback, not an active run. A cautious gate should keep the E81
  sparse-complex recipe and test `--simplex-pair-update-runtime-scale 1.0`
  with `--simplex-single-update-runtime-scale 1.0` ramped to `0.5` over the
  500-step gate. This isolates the cochain readout route without changing the
  selected complex or adding a new loss.
- E89 local validation passed:
  `python -m py_compile minalphafold/trainer.py
  scripts/run_nanofold_public_benchmarks.py`;
  `python -m pytest
  tests/test_nanofold_public_benchmarks.py::test_model_config_override_flags_are_accepted_by_cli_parser
  tests/test_nanofold_public_benchmarks.py::test_runtime_simplex_message_scales_ramp_and_enter_model_inputs
  tests/test_nanofold_public_benchmarks.py::test_evaluate_uses_runtime_simplex_overrides_for_validation
  tests/test_trainer.py::test_model_inputs_add_training_only_simplex_curricula`
  reported `4 passed`; `python -m pytest tests/test_simplex.py
  tests/test_nanofold_public_benchmarks.py tests/test_trainer.py` reported
  `157 passed`.
- E86 status at 2026-05-12T13:05:48Z on owned Runpod pod
  `o1dy17ouv8w5mz`: process `6369` was still alive after about 42 minutes,
  the run directory existed, and `results.json` was not present yet. Because
  no run has returned, `EXPERIMENT_RESULTS.md` remains unchanged.
- E86 returned on owned Runpod pod `o1dy17ouv8w5mz`: step 8500
  `val_lddt_ca=0.3990174550563097`, FoldScore `0.38581269793212414`,
  `val_ca_drmsd=10.02808192372322`, predicted/true C-alpha radius
  `11.538068354129791 / 15.403406739234924`, selected face/tetra boundary
  lDDT `0.7385465689003468` / `0.7216474078595638`, boundary length MAE
  `1.0705599561333656` / `1.1790090538561344`, contraction fractions
  `0.5956619083881378` / `0.5951620377600193`, and parameters `3,230,834`
  (+4.00% vs AF2-medium). E86 is kept as the new tiny primary-lDDT best and
  should be continued one short gate before launching E87/E88/E89/E90.
- Copied E86 returned artifacts locally under ignored
  `artifacts/nanofold_public_benchmarks/e86_weak_outer_edge_from_e81_s8500_c256_m64/`
  and copied the launch log to ignored `logs/e86_weak_outer_edge_from_e81.log`.
  The local artifact pull excluded the checkpoint directory; the remote E86
  checkpoint remains available for the continuation.
- Used `scripts/format_experiment_result_row.py` with `--start-after-step
  8000` to add the E86 row to `EXPERIMENT_RESULTS.md`, so inherited E81
  history does not count as E86's best validation lDDT.
- E91 launched on the same owned Runpod H100 pod `o1dy17ouv8w5mz` with run
  name `e91_weak_outer_edge_from_e86_s9000_c256_m64`, log path
  `/workspace/SimplexFold/logs/e91_weak_outer_edge_from_e86.log`, and artifact
  path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e91_weak_outer_edge_from_e86_s9000_c256_m64/`.
  Before launch, local source/docs/tests were synced, remote py_compile
  passed for the simplex/model-config/trainer/runner files, parser smoke
  confirmed the new E90 outer-edge score flag and the E91 outer-edge runtime
  flag, no previous benchmark process was active, and the E86 checkpoint was
  present. Main Python PID is `6904`. The runner resumed E86 at step
  8500/examples 68000, loaded 1292 matching model tensors, initialized 0
  new/missing tensors, and started a fresh optimizer. Startup poll at
  2026-05-12T13:30:01Z showed the process alive, GPU active, and no
  `results.json` yet.
- Heartbeat `check-simplexfold-e57-runpod` has been retargeted to E91 on
  owned pod `o1dy17ouv8w5mz` and must not touch any other Runpod instance.
- E90 local implementation prepared during the E86 run: added a zero-parameter
  outer-edge-support bonus for capped face/tetra cell scoring. The term counts
  selected neighbor edges leaving each candidate cell's vertices and can
  reward cells that are better embedded in the outer-edge neighborhood before
  selected-cell top-k masking. This is a topological domain-construction
  change, not a generic output-coordinate loss.
- E90 was extended with runtime schedule support so the outer-edge-support
  score can be ramped in on a resumed checkpoint, e.g. `0.0 -> 0.25`, instead
  of abruptly changing the selected cell complex.
- E90 focused validation passed:
  `python -m py_compile minalphafold/simplex.py minalphafold/model_config.py
  minalphafold/trainer.py scripts/run_nanofold_public_benchmarks.py`;
  `python -m pytest
  tests/test_simplex.py::test_cell_score_outer_edge_weight_prefers_context_supported_cells
  tests/test_nanofold_public_benchmarks.py::test_model_config_override_flags_are_accepted_by_cli_parser
  tests/test_nanofold_public_benchmarks.py::test_runtime_simplex_message_scales_ramp_and_enter_model_inputs
  tests/test_nanofold_public_benchmarks.py::test_evaluate_uses_runtime_simplex_overrides_for_validation
  tests/test_trainer.py::test_model_inputs_add_training_only_simplex_curricula
  tests/test_trainer.py::test_simplicial_cell_outer_edge_score_adds_no_parameters`
  reported `6 passed`.
- Broader E90 validation also passed: `python -m pytest tests/test_simplex.py
  tests/test_nanofold_public_benchmarks.py tests/test_trainer.py` reported
  `159 passed`; `git diff --check` was clean.

## 2026-05-09

- Target is `val_lddt_ca > 0.7` on NanoFold validation with parameters within
  5% of AF2 medium.
- All experiment runs should be on Runpod CUDA pods. Local execution is only
  for tests, linting, and non-evidential smoke debugging.
- Current public data size: 10,000 train chains and 1,000 validation chains.
- `simplexfold_medium_param_matched` is the primary profile because it is
  essentially exactly parameter-matched to AF2 medium while preserving medium
  layer counts.
- Important correction: generic metric-aligned losses are not enough. Changes
  must be mediated by simplex topology, selected face/tetra cells, boundary
  geometry, topology quality, or simplex-to-pair/single message passing.
- Rejected a generic all-pairs C-alpha distance loss because it is independent
  of the simplicial view.
- E00 smoke baseline completed locally:
  `artifacts/nanofold_public_benchmarks/e00_smoke_baseline/results.csv`.
  This was only `train-limit=8`, `val-limit=4`, `steps=2`, `crop=32`,
  `msa-depth=8`, so the absolute scores are not meaningful. It verified that
  the runner works on MPS and that `no_simplex`, `faces`, and `full` all run.
- E00 smoke final `val_lddt_ca`: `no_simplex=0.0429`, `faces=0.0935`,
  `full=0.0934`.
- The local smoke numbers above are not part of the experiment evidence after
  the Runpod-only decision. Use `e00_runpod_baseline` as the control run.
- E01 implemented a balanced contact loss inside `SimplexGeometryLoss` for
  `simplex_contact_logits`. Positive and negative contact terms are normalized
  separately, then averaged per example. This is a topology-quality change:
  the supervised logits are the same logits used to build the first-pass
  sparse neighbor graph.
- E01 local targeted tests passed:
  `tests/test_simplex.py::test_simplex_contact_loss_balances_contacts_and_non_contacts`,
  `tests/test_simplex.py::test_simplex_geometry_loss_adds_distance_and_consistency_terms`,
  `tests/test_simplex.py::test_tiny_alphafold2_profile_emits_simplex_training_tensors`,
  `tests/test_trainer.py::test_simplexfold_medium_param_matched_matches_af2_medium_budget`,
  and `tests/test_trainer.py::test_load_model_config_selects_requested_profile`.
- Launched dedicated Runpod pod `sytp4e4kjs7e61`
  (`codex-simplexfold-e01-runpod-20260509`) for SimplexFold experiments.
  Do not manage or modify pre-existing Runpod instances for this study.
- Runpod BF16 smoke failed on main/full before training completed due a
  PyTorch activation-checkpoint recomputation metadata mismatch. Conservative
  pilot runs should use `--mixed-precision off` until checkpointing and BF16
  are made compatible.
- Runpod fp32 smoke passed for main/full and E01/full with
  `train-limit=8`, `val-limit=4`, `steps=2`, `crop=128`, `msa-depth=32`.
  These numbers are smoke-only and not evidence for the study.
- E01 Runpod pilot completed on pod `sytp4e4kjs7e61`:
  main/full control commit `a299438` vs E01/full commit `6c20faa`,
  `train-limit=256`, `val-limit=64`, `steps=1000`, `crop=128`,
  `msa-depth=32`, `mixed-precision=off`.
- E01 final `val_lddt_ca` did not beat control:
  control `0.0401`, E01 `0.0316`.
- E01 best interim `val_lddt_ca` was slightly higher:
  control best `0.0992` at step 400, E01 best `0.1096` at step 400.
  Interpretation: balanced contact supervision can improve some early topology
  selection points but is not sufficient as a standalone change.
- Next candidate: a row-wise topology neighborhood loss that directly trains
  each anchor residue's contact logits as a distribution over `N(i)`, the
  sparse neighbor star used to instantiate faces and tetras.
- E02 implemented the row-wise topology neighborhood loss inside
  `SimplexGeometryLoss` with no parameter-count change.
- E02 local targeted tests passed:
  `tests/test_simplex.py::test_simplex_topology_neighborhood_loss_targets_anchor_neighbors`,
  E01 simplex loss/runner tests, and the medium budget/profile tests.
- E02 Runpod pilot completed on pod `sytp4e4kjs7e61`, treatment commit
  `e09b88f`, same 1000-step full-variant protocol. Final `val_lddt_ca`
  was `0.0281`; best interim was `0.1127` at step 800. This is the best
  interim score so far but does not solve final collapse.
- E03 implemented warm-started simplex boundary message projections so
  face/tetra states influence pair and single streams immediately rather than
  starting as identity residuals. No parameter-count change.
- E03 Runpod pilot completed on pod `sytp4e4kjs7e61`, treatment commit
  `8320da1`, same 1000-step full-variant protocol. Final `val_lddt_ca`
  improved to `0.0742`; best interim improved to `0.1311` at step 800.
  This is the first tested change that improves both best interim and final
  validation against the main/full control.

## Current Runpod Pilot Summary

| Run | Best step | Best `val_lddt_ca` | Final `val_lddt_ca` | Final FoldScore |
| --- | ---: | ---: | ---: | ---: |
| main/full control | 400 | 0.0992 | 0.0401 | 0.2024 |
| E01 balanced contact | 400 | 0.1096 | 0.0316 | 0.1996 |
| E02 topology neighborhood | 800 | 0.1127 | 0.0281 | 0.1944 |
| E03 warm boundary | 800 | 0.1311 | 0.0742 | 0.2132 |
| E04 coordinate cells diagnostic | 1000 | 0.2200 | 0.2200 | 0.2422 |
| E04 coordinate cells scaled | 1500 | 0.2394 | 0.1985 | 0.2399 |
| E05 coordinate weights 0.5 scaled | 3000 | 0.2948 | 0.2948 | 0.2647 |
| E06 coordinate weights 1.0 scaled | 3000 | 0.3127 | 0.3127 | 0.2511 |
| E07 boundary coordinate d=0.5 scaled | 2000 | 0.3247 | 0.3187 | 0.2617 |
| E09 full MSA-to-face d=0.5 scaled | 3000 | 0.3429 | 0.3429 | 0.2689 |
| E12 E09 continuation to 6000 | 5000 | 0.3472 | 0.3449 | 0.2856 |
| E14 mixed soft selector | 2000 | 0.3264 | 0.3015 | 0.2589 |
| E15 simplex aux anneal to 0.5 | 9000 | 0.3556 | 0.3556 | 0.3025 |

## Scaled E03 Pilot

- Ran E03 on the same dedicated Runpod pod with `train-limit=1024`,
  `val-limit=128`, `steps=3000`, `crop=256`, `msa-depth=64`,
  `mixed-precision=off`.
- Result path on the Runpod network volume:
  `/workspace/codex-simplexfold-e03-scaled-20260509/results/e03_warm_boundary_scaled_c256_m64_s3000`.
- Best scaled `val_lddt_ca`: `0.1219` at step 1000.
- Final scaled `val_lddt_ca`: `0.1009` at step 3000.
- Final scaled FoldScore: `0.2346`.
- Interpretation: E03 improves over the initial control but remains far from
  the `0.7` target. The persistent issue is structural collapse or
  under-expansion: predicted C-alpha radius of gyration remains much smaller
  than the true structures in these pilots.
- Stopped dedicated Runpod pod `sytp4e4kjs7e61` after runs completed.
- E04 implemented selected-simplex coordinate realization losses: predicted
  C-alpha face areas and tetra signed/absolute volume plus radius of gyration
  are compared only on the model's selected sparse face/tetra cells. This is
  intended to address collapse while staying mediated by the explicit
  simplicial complex.
- E04 1000-step Runpod diagnostic completed on pod `sytp4e4kjs7e61` at commit
  `02a8bad`: final `val_lddt_ca=0.2200`, final FoldScore `0.2422`,
  `val_ca_rmsd=14.1964`, `val_ca_drmsd=14.9499`,
  `val_pred_ca_rg=5.3194`, `val_true_ca_rg=14.6867`. This is the best
  validation score so far and supports the selected-simplex coordinate
  realization direction.
- E04 scaled Runpod follow-up completed with `train-limit=1024`,
  `val-limit=128`, `steps=3000`, `crop=256`, `msa-depth=64`,
  `mixed-precision=off`.
- E04 scaled best checkpoint was step 1500: `val_lddt_ca=0.2394`,
  `val_ca_rmsd=15.1042`, `val_ca_drmsd=15.2812`, FoldScore `0.2341`,
  `val_pred_ca_rg=6.2108`, `val_true_ca_rg=15.4034`.
- E04 scaled final checkpoint was step 3000: `val_lddt_ca=0.1985`,
  `val_ca_rmsd=15.3330`, `val_ca_drmsd=16.7843`, FoldScore `0.2399`,
  `val_pred_ca_rg=4.8384`, `val_true_ca_rg=15.7622`.
- Interpretation: E04 is the strongest tested direction and confirms selected
  simplex coordinate realization can lift scaled validation lDDT, but the
  realized structure still collapses after the peak and remains far below the
  `0.7` target.
- E05 exposes separate selected face/tetra coordinate-realization weights so
  the next Runpod experiment can strengthen only the topology-mediated
  coordinate terms. Defaults are unchanged unless the runner passes
  `--simplex-face-coordinate-weight` or `--simplex-tetra-coordinate-weight`.
- E05 launched on pod `sytp4e4kjs7e61` at commit `5672991` with the same
  scaled protocol as E04 and both selected coordinate weights set to `0.5`.
- E05 completed on pod `sytp4e4kjs7e61`: final and best
  `val_lddt_ca=0.2948`, final FoldScore `0.2647`, `val_ca_rmsd=15.1656`,
  `val_ca_drmsd=14.9898`, `val_pred_ca_rg=6.5441`, and
  `val_true_ca_rg=15.7622`.
- E05 interpretation: increasing selected face/tetra coordinate weights from
  `0.1` to `0.5` improved every scaled comparison point that matters for the
  collapse diagnosis: final lDDT, final FoldScore, final dRMSD, and realized
  radius of gyration. The run still remains far below the `0.7` target.
- E06 launched on pod `sytp4e4kjs7e61` at commit `1b5fae3` with the same
  scaled protocol and both selected coordinate weights set to `1.0`.
- E06 completed on pod `sytp4e4kjs7e61`: final and best
  `val_lddt_ca=0.3127`, final FoldScore `0.2511`, `val_ca_rmsd=15.5934`,
  `val_ca_drmsd=14.5496`, `val_pred_ca_rg=7.1388`, and
  `val_true_ca_rg=15.7622`.
- E06 interpretation: stronger selected face/tetra coordinate realization
  improved final lDDT and dRMSD over E05 and opened the structure further,
  but the radius-of-gyration gap remains large and the run is still far from
  the `0.7` target.
- E07 implemented selected simplex boundary-coordinate realization losses.
  The new losses supervise predicted C-alpha edge lengths only on boundary
  edges of the model-selected face/tetra cells, so the signal is mediated by
  the sparse simplicial complex instead of a dense all-pairs distance matrix.
  Defaults are off unless the runner passes
  `--simplex-face-coordinate-distance-weight` or
  `--simplex-tetra-coordinate-distance-weight`.
- E07 launched on pod `sytp4e4kjs7e61` at commit `b880e43` with the same
  scaled protocol as E06, selected face/tetra coordinate weights set to
  `1.0`, and selected boundary coordinate-distance weights set to `0.5`.
- E07 completed on pod `sytp4e4kjs7e61`: best `val_lddt_ca=0.3247` at step
  2000, final `val_lddt_ca=0.3187`, final FoldScore `0.2617`,
  final `val_ca_rmsd=16.0872`, final `val_ca_drmsd=12.9483`,
  `val_pred_ca_rg=9.1383`, and `val_true_ca_rg=15.7622`.
- E07 interpretation: selected boundary coordinate distances improve the best
  lDDT and materially improve realized local geometry over E06. This is the
  strongest topology-mediated direction so far, but still far from the `0.7`
  target.
- E08 launched on pod `sytp4e4kjs7e61` at commit `e77f141` with the same
  scaled protocol as E07 and selected boundary coordinate-distance weights
  raised from `0.5` to `1.0`.
- E08 was stopped early after the step-500 validation point regressed to
  `val_lddt_ca=0.2636`, FoldScore `0.2386`, `val_ca_drmsd=15.1396`, and
  `val_pred_ca_rg=6.0681`. Doubling selected-boundary distance pressure is
  worse than E07's `0.5` setting.
- E09 adds a `full_msa_to_face` benchmark variant so selected face states can
  receive low-rank third-order MSA moments while keeping tetra message passing
  active. This is a zero-parameter activation of the existing MSA-to-face
  path and directly tests the README's `MSA <-> sparse face tensor` claim.
- E09 launched on pod `sytp4e4kjs7e61` at commit `680f68c` with the same
  scaled protocol and E07 loss weights, using `--variants full_msa_to_face`.
- E09 completed on pod `sytp4e4kjs7e61`: final and best
  `val_lddt_ca=0.3429`, final FoldScore `0.2689`, `val_ca_rmsd=15.0614`,
  `val_ca_drmsd=12.9189`, `val_pred_ca_rg=8.6544`, and
  `val_true_ca_rg=15.7622`.
- E09 interpretation: activating the existing MSA-to-face topological pathway
  improves over E07 and is now the strongest result, but the validation score
  remains far from the `0.7` target.
- E10 warms the MSA-to-face projection by switching its final initializer from
  zero to the default SimplexMLP initializer, matching the E03 principle that
  useful topology-mediated residual paths should not start as exact identity
  updates.
- E10 launched on pod `sytp4e4kjs7e61` at commit `f10866b`, using
  `--variants full_msa_to_face` with the E07/E09 selected-coordinate and
  selected-boundary loss weights.
- E10 was stopped early after the step-500 validation point regressed to
  `val_lddt_ca=0.2232`, FoldScore `0.2190`, `val_ca_drmsd=12.3197`, and
  `val_pred_ca_rg=10.5325`. Warm-starting the MSA-to-face projection expands
  structures but damages lDDT/FoldScore, so the code was restored to the E09
  zero-final MSA-to-face initializer.
- E11 adds a `full_msa_to_face_long` benchmark variant that keeps the E09
  face/tetra/MSA-to-face pathway and biases sparse neighbor selection toward
  nonlocal pairs with sequence separation at least 16. This tests whether the
  explicit simplex complex needs more long-range cells to address remaining
  under-expansion.
- E11 launched on pod `sytp4e4kjs7e61` at commit `91445a8`, using
  `--variants full_msa_to_face_long` with the E07/E09 selected-coordinate
  and selected-boundary loss weights.
- E11 was stopped early after the step-500 validation point regressed to
  `val_lddt_ca=0.2288`, FoldScore `0.2244`, `val_ca_drmsd=15.2809`, and
  `val_pred_ca_rg=6.0858`. A direct long-range topology bias is harmful under
  the current selector and loss settings.
- E12 launched on pod `sytp4e4kjs7e61` at commit `14e14e2`, resuming E09's
  `full_msa_to_face` checkpoint from step 3000 and continuing the same
  scaled protocol to step 6000.
- E12 completed on pod `sytp4e4kjs7e61`: best `val_lddt_ca=0.3472` at step
  5000, final `val_lddt_ca=0.3449`, final FoldScore `0.2856`,
  final `val_ca_drmsd=11.7918`, `val_pred_ca_rg=9.8828`, and
  `val_true_ca_rg=15.7622`. Continuing E09 modestly improved the reference
  and global geometry, but it remains far from the `0.7` target.
- E13 implements a mixed local/global simplex selector. The new
  `simplex_local_neighbor_k` knob reserves nearest-neighbor sequence slots
  before filling the remaining simplex vertices from the learned topology
  score. The `full_msa_to_face_mixed` variant uses 4 reserved local slots,
  disables the broad local bias, and keeps the E09 MSA-to-face face/tetra
  pathway. This is intended to keep local manifold continuity without the
  harmful static long-range bias observed in E11.
- E13 focused local checks passed:
  `tests/test_simplex.py::test_build_simplex_topology_excludes_self_and_respects_masks`,
  `tests/test_simplex.py::test_build_simplex_topology_reserves_local_neighbor_slots`,
  `tests/test_simplex.py::test_optional_low_rank_msa_to_face_path_runs`,
  `tests/test_nanofold_public_benchmarks.py`,
  `tests/test_trainer.py::test_simplexfold_medium_param_matched_matches_af2_medium_budget`,
  `tests/test_trainer.py::test_load_model_config_selects_requested_profile`, and
  `ruff check --select F821,F822,F823` on the touched files.
- E13 launched on pod `sytp4e4kjs7e61` at commit `a9f63ac`, using
  `--variants full_msa_to_face_mixed` with the E07/E09 selected-coordinate
  and selected-boundary loss weights. It uses separate checkpoints/results
  under `/workspace/codex-simplexfold-e13-runpod-20260510`.
- E13 was stopped early after the step-500 validation point:
  `val_lddt_ca=0.2371`, FoldScore `0.2238`, `val_ca_drmsd=15.3413`,
  and `val_pred_ca_rg=6.2290`. Reserving 4 local slots while removing the
  broad local bias appears too noisy for early learned/global cell selection.
- E14 adds a soft mixed selector variant,
  `full_msa_to_face_mixed_soft`, that keeps `simplex_local_neighbor_k=4` but
  uses `simplex_local_bias=2.0` for the remaining learned slots. The intent is
  to keep the E13 manifold scaffold while restoring enough local pressure to
  prevent arbitrary early nonlocal cells.
- E14 launched on pod `sytp4e4kjs7e61` at commit `b90a9b8`.
- E14 completed on pod `sytp4e4kjs7e61`: best `val_lddt_ca=0.3264` at step
  2000, final `val_lddt_ca=0.3015`, final FoldScore `0.2589`,
  final `val_ca_drmsd=12.1838`, `val_pred_ca_rg=10.1755`, and
  `val_true_ca_rg=15.7622`. The soft selector improved over E13 but remained
  below E09/E12 on lDDT and FoldScore, so it is rejected.
- E15 plan: resume the E12/E09 `full_msa_to_face` checkpoint at step 6000,
  continue to step 9000, and ramp only the overall `simplex_aux_weight` from
  `1.0` to `0.5` over steps 6000-7000. This tests whether selected
  face/tetra realization should act as an early scaffold and then relax while
  the structure module consolidates.
- E15 launched on pod `sytp4e4kjs7e61`, resuming the E12 checkpoint from
  step 6000 and writing separate outputs under
  `/workspace/codex-simplexfold-e15-runpod-20260510`.
- E15 completed on pod `sytp4e4kjs7e61`: best and final
  `val_lddt_ca=0.3556` at step 9000, final FoldScore `0.3025`,
  final `val_ca_drmsd=12.3527`, `val_pred_ca_rg=9.0217`, and
  `val_true_ca_rg=15.7622`. This is the strongest result so far. The
  `simplex_aux_weight` anneal from `1.0` to `0.5` improved both lDDT and
  FoldScore over E12, supporting a curriculum where selected simplex
  realization acts as a scaffold before relaxing.
- E16 plan: resume E15 at step 9000 and continue to step 12000 while ramping
  `simplex_aux_weight` from `0.5` to `0.25` over steps 9000-10000. Keep the
  selected face/tetra coordinate and boundary-distance weights unchanged.
- E16 launched on pod `sytp4e4kjs7e61` and was stopped after step 10500.
  Best during the deeper anneal was `val_lddt_ca=0.3506` at step 9500; step
  10000 reached FoldScore `0.3062` but lDDT fell to `0.3400`, and step 10500
  was `val_lddt_ca=0.3438`. Lowering `simplex_aux_weight` to `0.25` improves
  aggregate FoldScore briefly but hurts the C-alpha lDDT objective.
- E17 plan: resume E15 again at step 9000 and continue to step 12000 with
  `simplex_aux_weight=0.5` held constant, isolating more training at the E15
  scaffold strength from E16's deeper anneal.
- E17 launched on pod `sytp4e4kjs7e61` and was stopped after step 11000.
  It nearly tied E15 at step 9500 with `val_lddt_ca=0.3554` and improved
  FoldScore to `0.3041`; step 10000 reached FoldScore `0.3094` but lDDT was
  `0.3541`. The later checkpoints fell to `0.3454` and `0.3441`, so more
  training at `simplex_aux_weight=0.5` does not break the lDDT plateau.
- E18 shifts from scheduler-only continuation to a simplex-only capacity test.
  The new `simplexfold_medium_topology_plus` profile keeps the medium AF2
  trunk fixed and spends the available 5% allowance inside persistent
  face/tetra state and the MSA-to-face adapter: face channels 24 -> 28, tetra
  channels 12 -> 14, simplex hidden 80 -> 87, and MSA-to-face rank 12 -> 16.
  The parameter audit is 3,256,126 parameters versus the AF2-medium baseline
  of 3,106,642, or +4.81%, under the 3,261,974 cap.
- E18 is topologically motivated rather than a generic lDDT hack: it tests
  whether the selected sparse 2-/3-simplex complex needs more representational
  capacity to store patch/packing geometry while preserving the same
  pair/MSA/structure trunk.
- E18 launched on owned Runpod pod `sytp4e4kjs7e61` at commit `6b35675`,
  under `/workspace/codex-simplexfold-e18-runpod-20260510`, using
  `simplexfold_medium_topology_plus`, `full_msa_to_face`, and the E09/E15
  selected-coordinate plus selected-boundary loss weights.
- E18 completed and the owned pod was stopped. The validation curve was:
  step 500 `val_lddt_ca=0.3032`, step 1000 `0.3029`, step 1500 `0.3196`,
  step 2000 `0.3324`, step 2500 `0.3313`, final step 3000 `0.3350`.
  Step 2000 briefly beat E09 at the same point (`0.3283`) with better
  FoldScore/dRMSD, but final step 3000 did not beat E09 final
  (`0.3429`). Result: reject as a replacement; added simplex capacity alone
  improves mid-run geometry but does not solve the C-alpha lDDT plateau.
- E19 implements selected-boundary lDDT realization. This deliberately avoids
  a dense all-pairs C-alpha lDDT loss: the new terms only see boundary edges
  of selected face/tetra cells, so they are still mediated by the learned
  sparse simplex topology. The expectation is that tolerance-style local
  distance pressure may improve lDDT where the existing smooth log-distance
  losses improved global geometry but plateaued locally.
- The original owned pod `sytp4e4kjs7e61` could not be restarted because
  Runpod reported no free GPUs on that host. A new H100 pod,
  `0hesaxxfhq8soj`, was launched for E19/E20. Only public processed features,
  public processed labels, and public train/val manifests were used; the
  accidentally transferred `hidden_val.txt` manifest was removed before any
  experiment was launched.
- E19 launched on pod `0hesaxxfhq8soj` at commit `add2cae` with selected
  boundary-lDDT weights `0.25`. It was stopped after the step-500 validation:
  `val_lddt_ca=0.2832`, FoldScore `0.2448`, `val_ca_drmsd=14.4789`,
  `val_pred_ca_rg=7.1624`, and `val_true_ca_rg=15.4034`.
- E20 reran the same setup with lower selected boundary-lDDT weights `0.05`.
  It was stopped after the step-500 validation collapsed to
  `val_lddt_ca=0.2364`, FoldScore `0.2447`, `val_ca_drmsd=15.5881`,
  `val_pred_ca_rg=5.6076`, and `val_true_ca_rg=15.4034`. The new pod was then
  stopped.
- E21 returns to an architecture-mediated change rather than another loss:
  `simplex_pair_update_scale` and `simplex_single_update_scale` amplify the
  selected face/tetra boundary messages before they are added back into pair
  and single streams. This tests whether the explicit higher-order states are
  under-coupled to the structure module after averaging over incident cells.
- E21 was stopped after its first validation on owned pod `0hesaxxfhq8soj`.
  Step 500 reached only `val_lddt_ca=0.2315`, FoldScore `0.2328`,
  `val_ca_drmsd=15.1343`, `val_pred_ca_rg=6.3715`, and
  `val_true_ca_rg=15.4034`. Scaling simplex-to-trunk messages up by `1.5`
  worsens collapse, so message strength is not a simple under-coupling issue.
- E22 tests the complementary coupling direction with
  `full_msa_to_face_damped_messages`: keep the same selected simplex complex
  and realization losses, but scale pair/single simplex residuals by `0.5`.
  This asks whether higher-order cells should be a softer topological scaffold
  rather than a strong residual driver of early coordinate formation.
- E22 was stopped after step 500 on owned pod `0hesaxxfhq8soj`:
  `val_lddt_ca=0.2917`, FoldScore `0.2458`, `val_ca_drmsd=14.4541`,
  `val_pred_ca_rg=6.6487`, and `val_true_ca_rg=15.4034`. Damping recovers the
  E09 early band but does not improve it.
- E23 tests an edge-biased coupling split: scale the simplex-to-pair residual
  up to `1.5` while damping the simplex-to-single residual to `0.5`. This
  keeps the explicit higher-order cells active in the pair/edge 1-skeleton but
  reduces direct residue-state pressure that may be collapsing coordinates.
- E23 was stopped after step 500 on owned pod `0hesaxxfhq8soj`:
  `val_lddt_ca=0.2509`, FoldScore `0.2355`, `val_ca_drmsd=15.0561`,
  `val_pred_ca_rg=6.2181`, and `val_true_ca_rg=15.4034`. Edge-biased simplex
  coupling is worse than damping both streams, so the message-scale direction
  is not a promising path without a more structural change to how cells are
  selected or normalized.
- After E21-E23, pod `0hesaxxfhq8soj` was stopped. Both owned Runpod pods
  (`sytp4e4kjs7e61` and `0hesaxxfhq8soj`) are stopped.
- `EXPERIMENTS_NOTES.md` is the live lab notebook for plans, launch commands,
  audits, and in-flight observations. `EXPERIMENT_RESULTS.md` is the durable
  returned-results tracker; add a row there only after a Runpod experiment
  returns a final or early-stop validation point.
- E24 live plan: add degree-normalized selected simplex boundary realization.
  The current selected boundary losses count an edge once per incident
  selected face/tetra; E24 weights boundary losses by inverse undirected edge
  incidence so the selected cell complex supervises its 1-skeleton evenly
  instead of over-weighting high-degree local edges. This is a simplicial
  boundary-normalization change, not a dense all-pairs metric loss.
- E24 ran on owned pod `0hesaxxfhq8soj` at commit `8fe6c75`, using
  `full_msa_to_face`, the E09/E15 selected coordinate and boundary-distance
  weights, and `--simplex-boundary-degree-normalize`. It was stopped after the
  step-500 validation: `val_lddt_ca=0.2724`, FoldScore `0.2383`,
  `val_ca_drmsd=14.1528`, `val_pred_ca_rg=7.2673`, and
  `val_true_ca_rg=15.4034`. The run opened the structure more than E21-E23
  but regressed lDDT/FoldScore, so degree normalization alone is rejected.
- After E24, pod `0hesaxxfhq8soj` was stopped.
- E25 live plan: run the best E09/E15 topology-mediated stack with effective
  batch size 8 (`batch_size=1`, `grad_accum_steps=8`) for a short 500-step
  Runpod gate. This aligns with the final objective's required optimization
  regime before spending on a 30,000-step confirmation run.
- E25 completed on owned pod `0hesaxxfhq8soj` at commit `a54d709`, using
  `full_msa_to_face`, the E09/E15 selected coordinate and boundary-distance
  weights, `batch_size=1`, and `grad_accum_steps=8`. Step 250 reached
  `val_lddt_ca=0.2637`, FoldScore `0.2294`, `val_ca_drmsd=14.5203`,
  `val_pred_ca_rg=6.7816`, and `val_true_ca_rg=15.4034`. Final step 500
  recovered to `val_lddt_ca=0.2946`, FoldScore `0.2466`,
  `val_ca_drmsd=14.3073`, `val_pred_ca_rg=7.6818`, and
  `val_true_ca_rg=15.7622`. Effective batch 8 is not harmful enough to fail
  operationally, but it does not improve over the E09/E15 early band, so E25
  is rejected.
- After E25, pod `0hesaxxfhq8soj` was stopped. Both owned Runpod pods
  (`sytp4e4kjs7e61` and `0hesaxxfhq8soj`) are stopped.
- E26 live plan: run the existing `msa_to_face` variant as a 2-skeleton
  stabilization gate. It keeps learned triangular face states and the
  MSA-to-face moment but disables tetra states, testing whether the 3-simplex
  packing layer is hurting before face geometry has stabilized. Use the same
  500-step Runpod gate and E09 selected face coordinate/boundary-distance
  weights.
- E26 first attempt was killed before validation because the fresh pod had
  public data but not the NanoFold Python package, so official FoldScore
  components were unavailable. The local `nanofold/` package was synced to the
  remote NanoFold root, the import was verified, and the run was restarted
  under `e26_msa_face_2skeleton_s500_c256_m64_v2`.
- E26 completed on owned pod `0hesaxxfhq8soj`: step 250 reached
  `val_lddt_ca=0.2517`, FoldScore `0.2106`, `val_ca_drmsd=14.9939`,
  `val_pred_ca_rg=6.4038`, and `val_true_ca_rg=15.4034`. Final step 500 was
  `val_lddt_ca=0.2489`, FoldScore `0.2214`, `val_ca_drmsd=15.8143`,
  `val_pred_ca_rg=5.9651`, and `val_true_ca_rg=15.7622`. The 2-skeleton is
  worse than the full E09/E15 complex and is rejected.
- After E26, pod `0hesaxxfhq8soj` was stopped. Both owned Runpod pods
  (`sytp4e4kjs7e61` and `0hesaxxfhq8soj`) are stopped.
- E27 live plan: add and run `full_msa_to_face_no_recycled_topology`, keeping
  the full selected face/tetra complex and MSA-to-face path but setting
  `simplex_use_recycled_geometry=false`. This tests whether collapsed recycled
  coordinates are contaminating the learned sparse simplex selector.
- E27 completed on owned pod `0hesaxxfhq8soj` at commit `a6a896a`. Step 250
  reached `val_lddt_ca=0.2317`, FoldScore `0.2169`,
  `val_ca_drmsd=15.5788`, `val_pred_ca_rg=5.7226`, and
  `val_true_ca_rg=15.4034`. Final step 500 reached
  `val_lddt_ca=0.2369`, FoldScore `0.2354`, `val_ca_drmsd=16.3061`,
  `val_pred_ca_rg=5.7967`, and `val_true_ca_rg=15.7622`. Disabling recycled
  topology feedback makes the early geometry worse, so E27 is rejected.
- After E27, pod `0hesaxxfhq8soj` was stopped. Both owned Runpod pods
  (`sytp4e4kjs7e61` and `0hesaxxfhq8soj`) are stopped.
- E28 direction: a training-only topology teacher-forcing curriculum. Build
  the initial selected face/tetra complex from public training-label C-alpha
  distances for early steps, then anneal to learned MSA/pair topology. Keep
  validation/inference feature-only and do not add hidden labels, external
  data, templates, or dense all-pairs metric objectives.
- E28 implemented locally. The model accepts optional
  `simplex_teacher_ca_coords`, `simplex_teacher_ca_mask`, and
  `simplex_teacher_forcing_weight`; training code passes these only when the
  opt-in `TrainingConfig.simplex_topology_teacher_forcing_*` schedule is
  positive. The adapter uses true C-alpha distances only for sparse neighbor
  selection, not as simplex geometry features. Validation and inference calls
  keep the teacher fields absent.
- E28 Runpod gate plan: use `full_msa_to_face` with E09 selected coordinate
  and boundary-distance weights, teacher weight `1.0`, final teacher weight
  `0.0`, ramp start step `250`, and ramp length `250`, for a 500-step gate.
- E28 ran on owned pod `sytp4e4kjs7e61` at commit `d2b1cd1`. The stale
  `/workspace/nanoFold-Competition/data/manifests/hidden_val.txt` manifest on
  that pod was removed before launch; the remote data audit then found no
  hidden or sidecar paths, train/val/all counts were `10000/1000/11000`, and
  official FoldScore import was verified.
- E28 completed and the owned pod was stopped. Step 250 reached
  `val_lddt_ca=0.1560`, FoldScore `0.2252`, `val_ca_drmsd=17.2788`,
  `val_pred_ca_rg=4.2006`, and `val_true_ca_rg=15.4034`. Final step 500 was
  `val_lddt_ca=0.2398`, FoldScore `0.2222`, `val_ca_drmsd=15.5485`,
  `val_pred_ca_rg=6.1752`, and `val_true_ca_rg=15.7622`. Full
  teacher-forced topology is rejected.
- E29 live plan: rerun the same topology-curriculum gate with a softer
  teacher blend, `simplex_topology_teacher_forcing_weight=0.25` annealed to
  `0.0` from step 250 to step 500.
- E29 launched a fresh owned Runpod pod `p2roc93zgk4ho9`
  (`codex-simplexfold-e29-runpod-20260510`) after both prior owned pods could
  not restart due host GPU capacity. Only public processed features, public
  processed labels, public train/val/all manifests, and the NanoFold metrics
  package were transferred. Remote audit found no hidden or sidecar paths,
  train/val/all counts were `10000/1000/11000`, official FoldScore import
  worked, and the SimplexFold commit was `1531511`.
- E29 completed and pod `p2roc93zgk4ho9` was stopped. Step 250 reached
  `val_lddt_ca=0.2161`, FoldScore `0.2196`, `val_ca_drmsd=16.0005`,
  `val_pred_ca_rg=5.5601`, and `val_true_ca_rg=15.4034`. Final step 500 was
  `val_lddt_ca=0.2451`, FoldScore `0.2169`, `val_ca_drmsd=15.4451`,
  `val_pred_ca_rg=6.7226`, and `val_true_ca_rg=15.7622`. Soft teacher forcing
  is rejected.
- E30 direction: add a simplex coupling warmup so selected face/tetra states
  are trained by their auxiliary realization losses before their residual
  messages are ramped back into pair/single streams at full strength.
- E30 implemented locally as a training-only simplex update-scale curriculum.
  `TrainingConfig.simplex_update_scale*` schedules optional residual coupling
  overrides, `model_inputs_from_batch` passes them only for training calls, and
  the simplex adapter applies the override to both face/tetra-to-pair and
  face/tetra-to-single boundary messages. Evaluation/inference keep the model
  config's static coupling.
- E30 local checks: `pytest
  tests/test_trainer.py::test_train_step_updates_model_parameters
  tests/test_trainer.py::test_model_inputs_add_training_only_simplex_curricula
  tests/test_simplex.py::test_simplicial_adapter_update_scale_override_gates_boundary_residuals
  tests/test_nanofold_public_benchmarks.py` passed (`11 passed`);
  `python -m py_compile minalphafold/simplex.py minalphafold/evoformer.py
  minalphafold/model.py minalphafold/trainer.py
  scripts/run_nanofold_public_benchmarks.py` passed; `git diff --check`
  passed; benchmark help exposes `--simplex-update-scale*`; parameter count
  for `simplexfold_medium_param_matched` remains `3,106,690`. Ruff could not
  be run locally because neither `ruff` nor `.venv/bin/ruff` is installed.
- E30 Runpod gate plan: `full_msa_to_face`, E09 selected coordinate and
  boundary-distance weights, `--simplex-update-scale 0.0`,
  `--simplex-update-scale-final 1.0`,
  `--simplex-update-scale-ramp-start-step 250`, and
  `--simplex-update-scale-ramp-steps 250`, 500 steps. Do not add E30 to
  `EXPERIMENT_RESULTS.md` until the run returns a final or early-stop
  validation point.
- E30 launched on owned Runpod pod `p2roc93zgk4ho9`, reusing the fresh pod
  created for E29 after restart. The pod was re-provisioned from an empty
  workspace with only public processed features, public processed labels,
  public train/val/all manifests, and the NanoFold metrics package. Remote
  audit before launch: SimplexFold commit `6d66960`, train/val/all counts
  `10000/1000/11000`, feature/label file counts `11001/11000`, no hidden or
  sidecar data paths, H100 CUDA available, FoldScore import works, and
  parameter count `3,106,690`.
- E30 completed on owned pod `p2roc93zgk4ho9`, and the pod was stopped. Step
  250 reached `val_lddt_ca=0.2411`, FoldScore `0.2168`,
  `val_ca_drmsd=15.4721`, `val_pred_ca_rg=5.8110`, and
  `val_true_ca_rg=15.4034` while `simplex_update_scale=0.0`. Final step 500,
  after the ramp reached `1.0`, reached `val_lddt_ca=0.2854`, FoldScore
  `0.2405`, `val_ca_drmsd=13.9247`, `val_pred_ca_rg=8.9047`, and
  `val_true_ca_rg=16.3091`. The warmup improved global expansion and dRMSD
  during the ramp but remained below the E09/E15 lDDT band, so E30 is
  rejected. Runtime also increased sharply once simplex coupling entered the
  main structure-loss path.
- E31 live direction: reuse the E30 training-only coupling schedule but ramp
  to a damped target, `simplex_update_scale_final=0.5`, rather than full
  `1.0`. This follows E22's observation that damped simplex messages were
  less damaging than strong coupling, while keeping E30's staged
  higher-order-cell warmup.
- E31 launched on owned Runpod pod `p2roc93zgk4ho9`, after re-provisioning
  the empty restarted workspace with only public processed features, public
  processed labels, public train/val/all manifests, and the NanoFold metrics
  package. Mac tar emitted AppleDouble `._*` metadata files; these were
  deleted before launch. Remote audit before launch: SimplexFold commit
  `63f2640`, train/val/all counts `10000/1000/11000`, feature/label file
  counts `11001/11000`, no hidden or sidecar data paths, H100 CUDA available,
  FoldScore import works, and parameter count `3,106,690`.
- E31 completed on owned pod `p2roc93zgk4ho9`, and the pod was stopped. Step
  250 reached `val_lddt_ca=0.2422`, FoldScore `0.2133`,
  `val_ca_drmsd=14.8018`, `val_pred_ca_rg=6.7519`, and
  `val_true_ca_rg=15.4034` while `simplex_update_scale=0.0`. Final step 500,
  after the ramp reached `0.5`, reached `val_lddt_ca=0.2578`, FoldScore
  `0.2332`, `val_ca_drmsd=14.7889`, `val_pred_ca_rg=8.9024`, and
  `val_true_ca_rg=16.3091`. The damped warmup kept the global expansion seen
  in E30 but worsened lDDT/FoldScore, so E31 is rejected.
- E32 live direction: combine the best architecture-facing capacity result
  (E18 `simplexfold_medium_topology_plus`, within the 5% AF2-medium budget)
  with the best curriculum result (E15 auxiliary scaffold annealed to `0.5`).
  This stays within the simplex view: spend the allowed headroom on persistent
  face/tetra state capacity, then relax selected-cell auxiliary pressure.
- E32 launched on owned Runpod pod `p2roc93zgk4ho9` at 2026-05-10 07:19 EDT.
  The restarted workspace was re-provisioned with only public processed
  features, public processed labels, public train/val/all manifests, and the
  NanoFold metrics package. Remote audit before launch: SimplexFold commit
  `3f21b94`, train/val/all counts `10000/1000/11000`, feature/label file
  counts `11001/11000`, no hidden or sidecar data paths, H100 CUDA available,
  FoldScore import works, AF2-medium baseline parameter count `3,106,642`,
  topology-plus parameter count `3,256,126`, and the topology-plus profile is
  within the 5% AF2-medium budget.
- E32 was stopped early on owned Runpod pod `p2roc93zgk4ho9`, and the pod was
  stopped. Step 250 reached only `val_lddt_ca=0.2545`, FoldScore `0.2059`,
  `val_ca_drmsd=14.2821`, `val_pred_ca_rg=7.2877`, and
  `val_true_ca_rg=15.4034`. The topology-plus capacity profile combined with
  E15-style auxiliary annealing underperformed the E18/E25 early band and did
  not improve FoldScore enough to justify waiting for the full 500-step
  final validation. E32 is rejected.
- E33 direction: stop trying to improve the selected simplex scaffold through
  stronger auxiliary/coupling schedules alone. The next topological change
  should change the readout path: let persistent face/tetra states contribute
  a small gated boundary summary directly to the structure input, so the
  learned 2-/3-cells influence coordinate generation as realized cells rather
  than only as repeated residual perturbations to the AF2 pair/single trunk.
- E33 implemented locally as a zero-parameter simplicial structure readout.
  `SimplicialAdapter` now exposes the normalized selected face/tetra boundary
  summaries it already scatters to pair/single streams when
  `simplex_structure_readout_scale > 0`. `AlphaFold2` accumulates those
  summaries across ensembles and injects a scaled copy into the structure
  module's single/pair inputs, while keeping the dense readout tensors out of
  the final prediction dictionary. The benchmark variant
  `full_msa_to_face_structure_readout` enables `full_msa_to_face` with
  `simplex_structure_readout_scale=0.25`.
- E33 local checks: `python -m py_compile minalphafold/model_config.py
  minalphafold/simplex.py minalphafold/model.py
  scripts/run_nanofold_public_benchmarks.py` passed; focused `pytest`
  covering the adapter readout, benchmark variant, and no-parameter budget
  passed (`5 passed`); broader affected `pytest tests/test_simplex.py
  tests/test_nanofold_public_benchmarks.py
  tests/test_trainer.py::test_train_step_updates_model_parameters
  tests/test_trainer.py::test_model_inputs_add_training_only_simplex_curricula
  tests/test_trainer.py::test_simplicial_structure_readout_adds_no_parameters
  tests/test_trainer.py::test_simplicial_structure_readout_forward_keeps_internal_tensors_private`
  passed (`31 passed` after adding the CLI parser regression); `git diff
  --check` passed. Parameter audit:
  AF2-medium `3,106,642`, SimplexFold medium `3,106,690`, E33 readout
  `3,106,690`, within 5% budget.
- E33 Runpod gate plan: run `simplexfold_medium_param_matched` with variant
  `full_msa_to_face_structure_readout`, E09 selected coordinate and
  boundary-distance weights, 500 steps, validation every 250 steps, crop 256,
  MSA 64, no templates, fp32, and only public NanoFold train/val data. Add a
  row to `EXPERIMENT_RESULTS.md` only after the run returns a final or
  early-stop validation point.
- E33 launched on owned Runpod pod `p2roc93zgk4ho9` at 2026-05-10 08:19 EDT.
  The restarted workspace was empty before provisioning, then cloned
  SimplexFold commit `a71cc25` and received only the public NanoFold
  package/data needed for scoring. Remote audit before launch: train/val/all
  counts `10000/1000/11000`, feature/label file counts `11000/11000`, no
  hidden or sidecar data paths, no AppleDouble files, H100 CUDA available,
  FoldScore import works, AF2-medium baseline `3,106,642`, E33 readout
  `3,106,690`, `simplex_structure_readout_scale=0.25`, within the 5%
  AF2-medium budget.
- E33 first launch attempt exited before training because the new variant was
  implemented in `_variant_config` but omitted from the CLI `--variants`
  `choices` list. Fixed locally, added
  `test_structure_readout_variant_is_accepted_by_cli_parser`, reran the
  affected tests (`31 passed`), and will relaunch from the corrected commit.
- E33 relaunched on owned Runpod pod `p2roc93zgk4ho9` at 2026-05-10 08:22 EDT
  from corrected SimplexFold commit `855bce5`; public data audit from the
  same provisioned workspace remains clean. Run name:
  `e33_structure_readout_s500_c256_m64_r2`.
- E33 was stopped early on owned Runpod pod `p2roc93zgk4ho9`, and the pod was
  stopped. Step 250 reached `val_lddt_ca=0.2405`, FoldScore `0.2108`,
  `val_ca_drmsd=14.8467`, `val_pred_ca_rg=6.9826`, and
  `val_true_ca_rg=15.4034`. This misses the E22/E25 early band and does not
  justify a longer run. E33 is rejected.
- E34 direction: isolate the new structure readout from the repeated
  simplex-to-trunk residual path. Add a readout-only sidecar variant with
  `simplex_pair_update_scale=0.0`, `simplex_single_update_scale=0.0`, and a
  `simplex_structure_readout_scale=0.5`, so selected 2-/3-simplex states
  learn under auxiliary supervision and feed only the structure input.
- E34 implemented locally as the zero-parameter benchmark variant
  `full_msa_to_face_structure_readout_only`. Local checks:
  `python -m py_compile scripts/run_nanofold_public_benchmarks.py` passed;
  `pytest tests/test_nanofold_public_benchmarks.py
  tests/test_trainer.py::test_simplicial_structure_readout_adds_no_parameters`
  passed (`13 passed`); parameter audit remains AF2-medium `3,106,642`,
  E34 readout-only `3,106,690`, within 5% budget.
- E34 launched on owned Runpod pod `p2roc93zgk4ho9` at 2026-05-10 08:54 EDT.
  The restarted workspace was empty before provisioning, then cloned
  SimplexFold commit `b171666` and received only public NanoFold package/data.
  Remote audit before launch: train/val/all counts `10000/1000/11000`,
  feature/label file counts `11000/11000`, no hidden or sidecar data paths,
  no AppleDouble files, H100 CUDA available, FoldScore import works,
  AF2-medium baseline `3,106,642`, E34 readout-only `3,106,690`,
  `simplex_pair_update_scale=0.0`, `simplex_single_update_scale=0.0`,
  `simplex_structure_readout_scale=0.5`, within the 5% AF2-medium budget.
- E34 was stopped early on owned Runpod pod `p2roc93zgk4ho9`, and the pod was
  stopped. Step 250 reached `val_lddt_ca=0.2426`, FoldScore `0.2103`,
  `val_ca_drmsd=14.9311`, `val_pred_ca_rg=6.5743`, and
  `val_true_ca_rg=15.4034`. Readout-only sidecar mode did not recover the
  E22/E25 early band, so E34 is rejected.
- E35 direction: try a face-only structure sidecar. E33/E34 still fed both
  face and tetra summaries to the structure input; disable tetra construction
  and test whether the learned 2-skeleton alone is a less noisy local patch
  signal.
- E35 implemented locally as the zero-parameter benchmark variant
  `face_structure_readout_only`: MSA-to-face enabled, tetra disabled,
  `simplex_pair_update_scale=0.0`, `simplex_single_update_scale=0.0`, and
  `simplex_structure_readout_scale=0.5`.
- E35 local checks: `python -m py_compile
  scripts/run_nanofold_public_benchmarks.py` passed; `pytest
  tests/test_nanofold_public_benchmarks.py
  tests/test_trainer.py::test_simplicial_structure_readout_adds_no_parameters`
  passed (`15 passed`); parameter audit remains AF2-medium `3,106,642`,
  E35 face-only readout `3,106,690`, within 5% budget.
- E35 launched on owned Runpod pod `p2roc93zgk4ho9` at 2026-05-10 09:13 EDT
  from SimplexFold commit `b6fd28a`. Remote audit before launch: branch
  `codex/simplexfold-topology-e07-boundary-coordinate`, train/val/all counts
  `10000/1000/11000`, feature/label file counts `11000/11000`, no hidden or
  sidecar data paths, no AppleDouble files, H100 CUDA available, FoldScore
  import works, AF2-medium baseline `3,106,642`, E35 face-only readout
  `3,106,690`, `simplex_use_tetra=False`,
  `simplex_pair_update_scale=0.0`, `simplex_single_update_scale=0.0`,
  `simplex_structure_readout_scale=0.5`, within the 5% AF2-medium budget.
  Run name: `e35_face_readout_only_s500_c256_m64`.
- E35 was stopped early on owned Runpod pod `p2roc93zgk4ho9`, and the pod was
  stopped. Step 250 reached `val_lddt_ca=0.2406`, FoldScore `0.2062`,
  `val_ca_drmsd=13.0352`, `val_pred_ca_rg=9.1316`, and
  `val_true_ca_rg=15.4034`. Face-only structure sidecar readout did not
  improve over E33/E34 and misses the E22/E25 early band, so E35 is rejected.
- E36 direction: move upstream of the failed structure readout path and
  improve the selector that builds the sparse 1-skeleton. Add an optional
  hard-negative margin on `simplex_contact_logits` so true contact-neighborhood
  energy outranks the highest non-contact logits in each anchor row before
  the top-k face/tetra complex is constructed.
- E36 local checks: `python -m py_compile minalphafold/simplex.py
  minalphafold/losses.py minalphafold/trainer.py
  scripts/run_nanofold_public_benchmarks.py` passed; focused tests for the
  margin loss, existing topology-neighborhood loss, `AlphaFoldLoss` overrides,
  and benchmark CLI args passed (`4 passed`); broader affected
  `pytest tests/test_simplex.py
  tests/test_trainer.py::test_alphafold_loss_overrides_simplex_coordinate_weights
  tests/test_nanofold_public_benchmarks.py` passed (`34 passed`);
  `git diff --check` passed; parameter audit remains AF2-medium
  `3,106,642`, E36 `3,106,690`, within 5% budget.
- E36 launched on owned Runpod pod `p2roc93zgk4ho9` at 2026-05-10 09:51 EDT
  from SimplexFold commit `13607aa`. Remote audit before launch: branch
  `codex/simplexfold-topology-e07-boundary-coordinate`, train/val/all counts
  `10000/1000/11000`, feature/label file counts `11000/11000`, no hidden or
  sidecar data paths, no AppleDouble files, H100 CUDA available, FoldScore
  import works, AF2-medium baseline `3,106,642`, E36 topology-margin model
  `3,106,690`, `simplex_use_faces=True`, `simplex_use_tetra=True`,
  `simplex_use_msa_to_face=True`, `simplex_topology_margin_weight=0.05`,
  `simplex_topology_margin=1.0`, `simplex_topology_margin_hard_negatives=8`,
  within the 5% AF2-medium budget. Run name:
  `e36_topology_margin_s500_c256_m64`.
- E36 first launch was stopped as invalid before validation. The step-1
  history row showed `simplex_topology_margin_loss` was computed but
  `simplex_topology_margin_weight=0.0`, meaning the benchmark runner's local
  `AlphaFoldLoss` construction path had not received the new override. Fixed
  by centralizing the runner loss construction in `_build_loss_fn` and adding
  a regression test for the margin override.
- E36 relaunched on owned Runpod pod `p2roc93zgk4ho9` at 2026-05-10 09:55 EDT
  from corrected SimplexFold commit `c44bf2c`. Remote re-audit: public
  train/val/all counts `10000/1000/11000`, feature/label counts
  `11000/11000`, no hidden or sidecar data paths, H100 CUDA available,
  FoldScore import works, AF2-medium `3,106,642`, E36 `3,106,690`, runner
  `_build_loss_fn` reports `simplex_topology_margin_weight=0.05`, within the
  5% AF2-medium budget. Run name:
  `e36_topology_margin_s500_c256_m64_r2`.
- E36 was stopped early on owned Runpod pod `p2roc93zgk4ho9`, and the pod was
  stopped. The corrected run reached step 250 with
  `val_lddt_ca=0.1286`, FoldScore `0.1857`, `val_ca_drmsd=13.5268`,
  `val_pred_ca_rg=13.1096`, and `val_true_ca_rg=15.4034`. The topology
  margin term was active (`simplex_topology_margin_weight=0.05`,
  `val_weighted_simplex_topology_margin_loss=0.0160`), but validation local
  accuracy collapsed below the readout experiments. E36 is rejected.
- E37 direction: add a selected-face normal orientation realization term. The
  current selected face losses supervise edge lengths and area, but the
  SimplexFold hypothesis specifically asks explicit 2-simplices to represent
  oriented local patches. Compare predicted and true face normals after
  expressing each normal in the local N-CA-C frame of the face's boundary
  residues; this keeps the signal globally rigid-motion invariant and tied to
  selected learned faces.
- E37 local checks: `python -m py_compile minalphafold/simplex.py
  minalphafold/losses.py minalphafold/trainer.py
  scripts/run_nanofold_public_benchmarks.py` passed; focused tests for
  face-normal invariance, `AlphaFoldLoss` overrides, and benchmark CLI/loss
  builder passed (`4 passed`); broader affected `pytest tests/test_simplex.py
  tests/test_trainer.py::test_alphafold_loss_overrides_simplex_coordinate_weights
  tests/test_nanofold_public_benchmarks.py` passed (`36 passed`); parameter
  audit remains AF2-medium `3,106,642`, E37 `3,106,690`, within 5% budget.
- E37 launched on owned Runpod pod `p2roc93zgk4ho9` at 2026-05-10 10:33 EDT
  from SimplexFold commit `4059b48`. Remote audit before launch: branch
  `codex/simplexfold-topology-e07-boundary-coordinate`, train/val/all counts
  `10000/1000/11000`, feature/label file counts `11000/11000`, no hidden or
  sidecar data paths, no AppleDouble files, H100 CUDA available, FoldScore
  import works, AF2-medium baseline `3,106,642`, E37 face-normal model
  `3,106,690`, `simplex_use_faces=True`, `simplex_use_tetra=True`,
  `simplex_use_msa_to_face=True`, `simplex_face_normal_weight=0.1`, within
  the 5% AF2-medium budget. Run name: `e37_face_normal_s500_c256_m64`.
- E37 was stopped early on owned Runpod pod `p2roc93zgk4ho9`, and the pod was
  stopped. Step 250 reached `val_lddt_ca=0.2464`, FoldScore `0.2109`,
  `val_ca_drmsd=14.9943`, `val_pred_ca_rg=6.4679`, and
  `val_true_ca_rg=15.4034`. The face-normal term was active
  (`simplex_face_normal_weight=0.1`,
  `val_weighted_simplex_face_normal_loss=0.0501`) but did not improve beyond
  the weak E33-E35 band, so E37 is rejected.
- E38 direction: add selected simplex shape realization. E37's face-normal
  signal was topologically motivated but still scalar; E38 supervises each
  selected face/tetra as a local cell by centering its predicted and true
  C-alpha vertices, aligning with a proper Kabsch rotation, and scoring
  normalized vertex RMSD. This keeps the loss globally rigid-motion invariant,
  preserves tetra chirality by disallowing reflection, and stays tied to the
  selected sparse 2-/3-simplex complex rather than adding a dense all-pairs
  lDDT-shaped objective.
- E38 focused local checks so far: `python -m py_compile minalphafold/simplex.py
  minalphafold/losses.py minalphafold/trainer.py
  scripts/run_nanofold_public_benchmarks.py` passed. Focused tests for local
  shape invariance, `AlphaFoldLoss` overrides, and benchmark CLI/loss builder
  passed (`4 passed`).
- E38 broader local checks: `pytest tests/test_simplex.py
  tests/test_trainer.py::test_alphafold_loss_overrides_simplex_coordinate_weights
  tests/test_nanofold_public_benchmarks.py` passed (`38 passed`), full
  `python -m pytest` passed (`215 passed`), and `git diff --check` passed.
  Parameter audit using `configs/medium.toml` with
  `use_simplicial_evoformer=false` gives AF2-medium `3,106,642` and E38
  `full_msa_to_face` `3,106,690`, within the 5% budget. `python -m ruff` is
  unavailable locally. `python -m mypy minalphafold
  scripts/run_nanofold_public_benchmarks.py` still fails on pre-existing
  structure-module typing, missing `nanofold.metrics`, EMA model typing, and
  runner row typing. `python -m pyright --warnings` still fails broadly because
  this local interpreter does not resolve Torch/NumPy/Modal/OpenMM plus
  existing optional/type issues.
- Reference PDFs saved for later use in `references/papers/`:
  `hands_on_geometric_deep_learning_nodes_to_complexes.pdf` and
  `2509.03885v1.pdf`. I read both extracted texts in full and visually skimmed
  rendered page contact sheets. The hands-on TDL guide reinforces the basic
  modeling discipline: topological models should explicitly construct domains,
  maintain signals on multiple ranks, and use incidence/adjacency operators for
  intra- and inter-neighborhood message passing. This supports keeping E38 tied
  to selected cells rather than using a dense all-pairs lDDT-shaped objective.
- The Topotein paper is directly relevant. Its Protein Combinatorial Complex
  uses ranks for residues, directed interaction edges, secondary-structure
  cells, and a protein-level cell; its TCPNet uses edge-centric local frames and
  a bottom-up/down message order across ranks. Two lessons matter for
  SimplexFold: first, topological enhancements are most useful when deeply
  integrated into the architecture; second, superficial SSE/topology features
  can hurt if higher-rank cells are not updated by dedicated neighborhoods.
  Candidate follow-ups after E38 should therefore prioritize cell communication
  and local-frame scalarization over additional standalone scalar losses.
- E38 launched on owned Runpod pod `p2roc93zgk4ho9` at 2026-05-10 11:26 EDT
  from SimplexFold commit `64a9598`. Remote audit before launch: branch
  `codex/simplexfold-topology-e07-boundary-coordinate`, public train/val/all
  counts `10000/1000/11000`, feature/label file counts `11001/11000`,
  no hidden or sidecar data paths, no AppleDouble files, H100 CUDA available,
  FoldScore import works, AF2-medium baseline `3,106,642`, E38
  `full_msa_to_face` model `3,106,690`, `simplex_use_faces=True`,
  `simplex_use_tetra=True`, `simplex_use_msa_to_face=True`,
  `simplex_face_shape_weight=0.1`, `simplex_tetra_shape_weight=0.1`, within
  the 5% AF2-medium budget. Run name: `e38_simplex_shape_s500_c256_m64`.
- E38 first launch was stopped as invalid before validation. Step 1 was finite
  and the shape losses were active, but by step 50 all reported losses were
  `NaN`. The likely cause is backpropagation through SVD in the per-cell Kabsch
  alignment when selected cells become near-degenerate early in training. Patch
  direction: compute the proper Kabsch rotation under `no_grad` and backprop
  only through the aligned predicted cell vertices, then relaunch as E38r2.
- E38r2 local patch checks: `python -m py_compile minalphafold/simplex.py
  minalphafold/losses.py minalphafold/trainer.py
  scripts/run_nanofold_public_benchmarks.py` passed; focused/affected tests
  passed (`38 passed`), including a finite-gradient regression for collapsed
  selected cells; full `python -m pytest` passed (`215 passed`); `git diff
  --check` passed.
- E38r2 relaunched on owned Runpod pod `p2roc93zgk4ho9` at 2026-05-10 11:36
  EDT from SimplexFold commit `ec0d640`. Remote re-audit: public
  train/val/all counts `10000/1000/11000`, feature/label counts
  `11001/11000`, no hidden or sidecar data paths, no AppleDouble files, H100
  CUDA available, FoldScore import works, AF2-medium `3,106,642`, E38r2
  `3,106,690`, `simplex_face_shape_weight=0.1`,
  `simplex_tetra_shape_weight=0.1`, within the 5% AF2-medium budget. Run
  name: `e38r2_simplex_shape_detached_s500_c256_m64`.
- E38r2 passed the E38 failure point: step 50 was finite with
  `train_loss=8.2621`, `grad_norm=21.3335`, `train_simplex_face_shape_loss=0.1362`,
  and `train_simplex_tetra_shape_loss=0.1710`. Step 100 remained finite with
  `train_loss=7.9037`, `grad_norm=3.6255`, `train_simplex_face_shape_loss=0.1181`,
  and `train_simplex_tetra_shape_loss=0.1557`.
- E38r2 was stopped early on owned Runpod pod `p2roc93zgk4ho9`, and the pod
  was stopped. Step 250 reached `val_lddt_ca=0.2402`, FoldScore `0.2113`,
  `val_ca_drmsd=14.9614`, `val_pred_ca_rg=6.6367`, and
  `val_true_ca_rg=15.4034`. The shape terms were active
  (`val_weighted_simplex_face_shape_loss=0.0153`,
  `val_weighted_simplex_tetra_shape_loss=0.0190`), but validation remained in
  the weak E33-E38 band. E38 is rejected; next priority should be a
  Topotein-inspired architecture change such as outer-edge cell communication
  or edge-frame scalarized simplex messages.
- E39 local implementation: add `simplex_outer_edge_update_scale` and
  `full_msa_to_face_outer_edge`. The adapter now has an optional zero-parameter
  face-to-face update through shared undirected boundary edges: selected face
  states scatter to their boundary edges, gather other incident face states,
  and apply a gated residual before pair/single readout. This directly follows
  the reference topological-deep-learning lesson that useful higher-rank cells
  need incidence/adjacency neighborhoods, not just scalar cell losses.
- E39 local checks so far: `python -m py_compile minalphafold/simplex.py
  minalphafold/model_config.py scripts/run_nanofold_public_benchmarks.py`
  passed. Focused tests for outer-edge incidence behavior, output changes,
  parser acceptance, and zero-parameter behavior passed (`41 passed` across
  the affected selections). Parameter audit gives AF2-medium `3,106,642`,
  SimplexFold `3,106,690`, and E39 outer-edge `3,106,690`, within the 5%
  budget.
- E39 broader local checks: full `python -m pytest -q` passed and
  `git diff --check` passed. `python -m ruff check .` is unavailable locally
  (`No module named ruff`). `python -m mypy minalphafold
  scripts/run_nanofold_public_benchmarks.py` still fails on the pre-existing
  structure-module typing, missing `nanofold.metrics`, EMA model typing, and
  runner row typing issues. `python -m pyright --warnings` still fails broadly
  because this local interpreter does not resolve Torch/NumPy/Modal/OpenMM and
  reports existing optional/type issues.
- E39 launched on owned Runpod pod `p2roc93zgk4ho9` at 2026-05-10 11:53 EDT
  from SimplexFold commit `087308c`. The stopped pod's `/workspace` was empty,
  so I recloned the pushed SimplexFold branch and copied only public NanoFold
  assets: `data/processed_features`, `data/processed_labels`,
  `data/manifests/train.txt`, `data/manifests/val.txt`,
  `data/manifests/all.txt`, and `nanofold/`. Remote audit after AppleDouble
  cleanup: public train/val/all counts `10000/1000/11000`, feature/label
  counts `11000/11000`, no hidden or sidecar data paths, H100 CUDA available,
  FoldScore components import works, AF2-medium `3,106,642`, E39 outer-edge
  `3,106,690`, `simplex_outer_edge_update_scale=0.25`, within the 5%
  AF2-medium budget. Run name: `e39_outer_edge_face_s500_c256_m64`.
- E39 passed the previous NaN failure point: step 50 reached
  `train_loss=8.1281`, `grad_norm=18.2726`; step 100 reached
  `train_loss=7.8448`, `grad_norm=3.7852`; step 150 reached
  `train_loss=6.1836`, `grad_norm=5.8400`; step 200 reached
  `train_loss=6.4083`, `grad_norm=3.1336`.
- E39 was stopped early on owned Runpod pod `p2roc93zgk4ho9`, and the pod was
  stopped. Step 250 reached `val_lddt_ca=0.2460`, FoldScore `0.2163`,
  `val_ca_drmsd=14.7805`, `val_pred_ca_rg=6.7531`, and
  `val_true_ca_rg=15.4034`. The coordinate cell terms were active
  (`val_weighted_simplex_face_coordinate_area_loss=0.0322`,
  `val_weighted_simplex_face_coordinate_distance_loss=0.0147`,
  `val_weighted_simplex_tetra_coordinate_geometry_loss=0.0738`,
  `val_weighted_simplex_tetra_coordinate_distance_loss=0.0144`), but the
  validation result stayed in the weak E33-E38 band. E39 is rejected; E40
  should try edge-frame scalarized simplex messages if we continue the
  Topotein-inspired branch.
- E40 local implementation: add `simplex_edge_frame_message_scale` and
  `full_msa_to_face_edge_frame_messages`. The adapter now optionally builds
  directed local frames on selected boundary edges from recycled C-alpha
  coordinates and residue frames. Face-to-pair messages receive scalarized
  opposite-vertex and face-normal features in the boundary-edge frame; tetra
  messages receive scalarized opposite-vertex, plane-normal, angle, and signed
  volume features in each tetra boundary-edge frame. This is a topological
  architecture change because it changes how cochains on selected 2-/3-cells
  write to the pair-stream 1-skeleton.
- E40 local checks so far: `python -m py_compile minalphafold/simplex.py
  minalphafold/model_config.py scripts/run_nanofold_public_benchmarks.py`
  passed. Focused tests for edge-frame rigid-transform invariance, adapter
  pair-readout changes, parser acceptance, and budget passed (`5 passed`).
  Parameter audit gives AF2-medium `3,106,642`, SimplexFold `3,106,690`, and
  E40 edge-frame `3,154,242`, within the 5% budget.
- E40 broader local checks: affected tests passed (`45 passed`), full
  `python -m pytest -q` passed, and `git diff --check` passed.
  `python -m ruff check .` is unavailable locally (`No module named ruff`).
  `python -m mypy minalphafold scripts/run_nanofold_public_benchmarks.py`
  still fails on the pre-existing structure-module typing, missing
  `nanofold.metrics`, EMA model typing, and runner row typing issues.
  `python -m pyright --warnings` still fails broadly because this local
  interpreter does not resolve Torch/NumPy/Modal/OpenMM and reports existing
  optional/type issues.
- E40 launched on owned Runpod pod `p2roc93zgk4ho9` at 2026-05-10 12:19 EDT
  from SimplexFold commit `3839f43`. The stopped pod's `/workspace` was empty,
  so I recloned the pushed SimplexFold branch and copied only public NanoFold
  assets: `data/processed_features`, `data/processed_labels`,
  `data/manifests/train.txt`, `data/manifests/val.txt`,
  `data/manifests/all.txt`, and `nanofold/`. Remote audit after AppleDouble
  cleanup: public train/val/all counts `10000/1000/11000`, feature/label
  counts `11000/11000`, no hidden or sidecar data paths, H100 CUDA available,
  FoldScore components import works, AF2-medium `3,106,642`, E40 edge-frame
  `3,154,242`, `simplex_edge_frame_message_scale=0.25`, within the 5%
  AF2-medium budget. Run name: `e40_edge_frame_s500_c256_m64`.
- E40 passed the previous NaN failure point: step 50 reached
  `train_loss=8.4029`, `grad_norm=28.3162`; step 100 reached
  `train_loss=7.6904`, `grad_norm=4.6000`; step 150 reached
  `train_loss=6.3642`, `grad_norm=5.4251`; step 200 reached
  `train_loss=6.2218`, `grad_norm=3.1321`.
- E40 was stopped early on owned Runpod pod `p2roc93zgk4ho9`, and the pod was
  stopped. Step 250 reached `val_lddt_ca=0.2350`, FoldScore `0.2139`,
  `val_ca_drmsd=15.2338`, `val_pred_ca_rg=6.3502`, and
  `val_true_ca_rg=15.4034`. The coordinate cell terms were active
  (`val_weighted_simplex_face_coordinate_area_loss=0.0293`,
  `val_weighted_simplex_face_coordinate_distance_loss=0.0137`,
  `val_weighted_simplex_tetra_coordinate_geometry_loss=0.0618`,
  `val_weighted_simplex_tetra_coordinate_distance_loss=0.0134`), but the
  validation result was worse than E39 and stayed in the weak E33-E40 band.
  E40 is rejected; E41 latent rank-2 segment cells are the remaining
  Topotein-inspired branch to test.
- E41 local implementation at 2026-05-10 12:36 EDT: add latent rank-2 segment
  cells through `simplex_segment_cell_scale`, `simplex_segment_radius`,
  `simplex_c_segment`, and the `full_msa_to_face_segment_cells` benchmark
  variant. Each residue owns one contiguous local segment cell; the segment
  state pools official single/MSA-derived states, anchor-to-window pair
  features, and invariant recycled C-alpha segment geometry, then writes into
  selected face states by vertex incidence. This is a topological architecture
  change rather than an lDDT hack because it adds a learned cell rank and
  explicit incidence communication into the selected two-skeleton.
- E41 local checks so far: `python -m py_compile minalphafold/simplex.py
  minalphafold/model_config.py scripts/run_nanofold_public_benchmarks.py`
  passed; focused segment-cell tests passed (`6 passed`); affected tests
  passed (`50 passed`). Parameter audit gives AF2-medium `3,106,642`,
  SimplexFold `3,106,690`, and E41 segment cells `3,234,450`, within the 5%
  AF2-medium cap of `3,261,974`.
- E41 broader local checks: full `python -m pytest -q` passed and
  `git diff --check` passed. `python -m ruff check .` is unavailable locally
  (`No module named ruff`). `python -m mypy minalphafold
  scripts/run_nanofold_public_benchmarks.py` still fails on the pre-existing
  structure-module typing, missing `nanofold.metrics`, EMA model typing, and
  runner row typing issues. `python -m pyright --warnings` still fails broadly
  because this local interpreter does not resolve Torch/NumPy/Modal/OpenMM and
  reports existing optional/type issues.
- E41 launched on owned Runpod pod `p2roc93zgk4ho9` at 2026-05-10 12:47 EDT
  from SimplexFold commit `355d4b7`. The stopped pod's `/workspace` was
  empty, so I recloned the pushed SimplexFold branch and copied only public
  NanoFold assets: `data/processed_features`, `data/processed_labels`,
  `data/manifests/train.txt`, `data/manifests/val.txt`,
  `data/manifests/all.txt`, and `nanofold/`. Remote audit: public
  train/val/all counts `10000/1000/11000`, feature/label counts
  `11000/11000`, no hidden or sidecar data paths, H100 CUDA available,
  FoldScore import works, AF2-medium `3,106,642`, E41 segment cells
  `3,234,450`, `simplex_segment_cell_scale=0.25`,
  `simplex_segment_radius=4`, `simplex_c_segment=12`, within the 5%
  AF2-medium budget. Run name: `e41_segment_cells_s500_c256_m64`.
- E41 passed the previous NaN failure point: step 50 reached
  `train_loss=8.0738`, `grad_norm=13.5769`; step 100 reached
  `train_loss=7.8239`, `grad_norm=4.3516`; step 150 reached
  `train_loss=6.4231`, `grad_norm=6.8139`; step 200 reached
  `train_loss=6.2640`, `grad_norm=3.1682`.
- E41 was stopped early on owned Runpod pod `p2roc93zgk4ho9`, and the pod was
  stopped. Step 250 reached `val_lddt_ca=0.2393`, FoldScore `0.2125`,
  `val_ca_drmsd=15.2012`, `val_pred_ca_rg=6.2747`, and
  `val_true_ca_rg=15.4034`. The coordinate cell terms were active
  (`val_weighted_simplex_face_coordinate_area_loss=0.0312`,
  `val_weighted_simplex_face_coordinate_distance_loss=0.0147`,
  `val_weighted_simplex_tetra_coordinate_geometry_loss=0.0662`,
  `val_weighted_simplex_tetra_coordinate_distance_loss=0.0143`), but the
  validation result stayed in the weak E33-E41 band. E41 is rejected.
- E42 local implementation at 2026-05-10 12:58 EDT: add
  `simplex_hodge_face_update_scale`, `face_tetra_coboundary_delta`, and the
  `full_msa_to_face_hodge_residual` benchmark variant. The residual averages
  selected face states through both lower shared-boundary-edge adjacency and
  upper selected-tetra coface adjacency, then applies a damped gated update
  before face/tetra states write back to pair/single streams. This is a
  topological architecture change, not a loss hack: no new scalar objective is
  added, and the update operates on cochains and incidence maps in the
  selected complex.
- E42 local checks so far: `python -m py_compile minalphafold/simplex.py
  minalphafold/model_config.py scripts/run_nanofold_public_benchmarks.py`
  passed; focused Hodge tests passed (`5 passed`); affected tests passed
  (`55 passed`). Parameter audit gives AF2-medium `3,106,642`, SimplexFold
  `3,106,690`, and E42 Hodge residual `3,106,690`, adding no parameters and
  staying within the 5% AF2-medium budget.
- E42 broader local checks: full `python -m pytest -q` passed and
  `git diff --check` passed. `python -m ruff check .` is unavailable locally
  (`No module named ruff`). `python -m mypy minalphafold
  scripts/run_nanofold_public_benchmarks.py` still fails on the pre-existing
  structure-module typing, missing `nanofold.metrics`, EMA model typing, and
  runner row typing issues. `python -m pyright --warnings` still fails broadly
  because this local interpreter does not resolve Torch/NumPy/Modal/OpenMM and
  reports existing optional/type issues.
- E42 launched on owned Runpod pod `p2roc93zgk4ho9` at 2026-05-10 13:20 EDT
  from SimplexFold commit `9faa0fd`. The stopped pod's `/workspace` was
  empty, so I recloned the pushed SimplexFold branch and copied only public
  NanoFold assets: `data/processed_features`, `data/processed_labels`,
  `data/manifests/train.txt`, `data/manifests/val.txt`,
  `data/manifests/all.txt`, and `nanofold/`. Remote audit: public
  train/val/all counts `10000/1000/11000`, feature/label `.npz` counts
  `11000/11000`, no hidden or sidecar data paths, H100 CUDA available,
  FoldScore import works, AF2-medium `3,106,642`, E42 Hodge residual
  `3,106,690`, `simplex_hodge_face_update_scale=0.25`, within the 5%
  AF2-medium budget. Run name: `e42_hodge_residual_s500_c256_m64`.
- E42 was stable through the first validation point: step 50 reached
  `train_loss=8.2494`, `grad_norm=20.9149`; step 100 reached
  `train_loss=7.8037`, `grad_norm=3.1864`; step 150 reached
  `train_loss=6.4399`, `grad_norm=5.5352`; step 200 reached
  `train_loss=6.3583`, `grad_norm=3.0872`. Step 250 reached
  `val_lddt_ca=0.2545`, FoldScore `0.2112`, `val_ca_drmsd=14.7096`,
  `val_pred_ca_rg=6.7897`, and `val_true_ca_rg=15.4034`. The coordinate cell
  terms were active (`val_weighted_simplex_face_coordinate_area_loss=0.0381`,
  `val_weighted_simplex_face_coordinate_distance_loss=0.0174`,
  `val_weighted_simplex_tetra_coordinate_geometry_loss=0.0763`,
  `val_weighted_simplex_tetra_coordinate_distance_loss=0.0166`).
- E42 continued through step 450 (`train_loss=6.1144`, `grad_norm=3.4824`)
  and entered the final validation path, but no final validation row was
  written after an extended wait. The process was stopped manually and the
  pod was stopped at 2026-05-10 13:52 EDT. E42 is rejected: the Hodge residual
  is a mild positive over E33-E41 but not strong enough to continue without a
  better training/curriculum context.
- E43 plan at 2026-05-10 13:55 EDT: run the zero-parameter
  `full_msa_to_face_hodge_residual` architecture inside the E15-style
  selected-simplex scaffold relaxation. Keep selected face/tetra coordinate
  weights at `1.0` and boundary-distance weights at `0.5`, but ramp only the
  overall `simplex_aux_weight` from `1.0` to `0.5` over steps 250-500. Use
  16-batch validation for both intermediate and final checkpoints so the
  pilot returns promptly; do not write to `EXPERIMENT_RESULTS.md` until a
  validation point returns.
- E43 launched on owned Runpod pod `p2roc93zgk4ho9` at 2026-05-10 14:28 EDT
  from SimplexFold commit `bb17c86`. The first staging attempt produced
  macOS AppleDouble sidecars, so it was interrupted and restarted with
  `COPYFILE_DISABLE=1`. Remote audit for the clean staging pass: public
  train/val/all counts `10000/1000/11000`, feature/label `.npz` counts
  `11000/11000`, no hidden, sidecar, or `.DS_Store` paths, H100 CUDA
  available, FoldScore import works, AF2-medium `3,106,642`, SimplexFold and
  E43 Hodge residual `3,106,690`, `simplex_hodge_face_update_scale=0.25`,
  within the 5% AF2-medium budget. Run name:
  `e43_hodge_aux_anneal_s500_c256_m64`.
- E43 completed and the owned Runpod pod was stopped at 2026-05-10
  14:38 UTC. Step 250 reached `val_lddt_ca=0.2388`, FoldScore `0.2120`,
  `val_ca_drmsd=15.0913`, `val_pred_ca_rg=6.5181`, and
  `val_true_ca_rg=15.4034` with `simplex_aux_weight=1.0`. The anneal was
  active afterward: steps 300/350/400/450/500 recorded
  `simplex_aux_weight=0.9/0.8/0.7/0.6/0.5`. Step 500 reached
  `val_lddt_ca=0.2492`, FoldScore `0.2232`, `val_ca_drmsd=15.1139`,
  `val_pred_ca_rg=6.1772`, and `val_true_ca_rg=15.4034`. The run is rejected:
  the anneal improved the E43 checkpoint internally but still trailed E42 and
  the stronger E22/E25/E30 early range.
- E44 local implementation at 2026-05-10 14:58 EDT: add a soft
  flag-complex closure gate through `simplex_boundary_closure_weight`,
  `simplex_boundary_closure_temperature`, and the
  `full_msa_to_face_flag_closure` benchmark variant. During topology
  construction, selected face/tetra masks are blended with the geometric mean
  of the learned probabilities on their boundary edges. This is a topological
  construction change, not an lDDT hack: it asks whether filled 2-/3-cells
  should be trusted only when their learned 1-skeleton is also plausible.
  Focused checks pass for closure-mask behavior, CLI variant parsing, and
  zero parameter growth.
- E44 broader local checks: `python -m py_compile minalphafold/simplex.py
  minalphafold/model_config.py scripts/run_nanofold_public_benchmarks.py`
  passed; affected tests passed (`110 passed`); full `python -m pytest -q`
  passed; `git diff --check` passed. Parameter audit gives AF2-medium
  `3,106,642`, SimplexFold `3,106,690`, and E44 flag closure `3,106,690`.
  `python -m ruff check .` remains unavailable locally (`No module named
  ruff`). `python -m mypy minalphafold scripts/run_nanofold_public_benchmarks.py`
  still fails on the pre-existing structure-module typing, missing
  `nanofold.metrics`, EMA model typing, and runner row typing issues.
  `python -m pyright --warnings` still fails broadly because this local
  interpreter does not resolve Torch/NumPy/Modal/OpenMM and reports existing
  optional/type issues.
- E44 launched on owned Runpod pod `p2roc93zgk4ho9` at 2026-05-10
  15:07 EDT from SimplexFold commit `808a3f0`. The stopped pod's
  `/workspace` was empty, so I recloned the pushed SimplexFold branch and
  copied only public NanoFold assets with `COPYFILE_DISABLE=1`: processed
  features, processed labels, public train/val/all manifests, and `nanofold/`.
  Remote audit: public train/val/all counts `10000/1000/11000`,
  feature/label `.npz` counts `11000/11000`, no hidden or sidecar data paths,
  H100 CUDA available, FoldScore import works, AF2-medium `3,106,642`, E44
  flag closure `3,106,690`, `simplex_boundary_closure_weight=0.5`,
  `simplex_boundary_closure_temperature=1.0`, within the 5% AF2-medium
  budget. Run name: `e44_flag_closure_s500_c256_m64`.
- E44 completed and the owned Runpod pod was stopped at 2026-05-10
  19:20 UTC. Step 250 reached `val_lddt_ca=0.2449`, FoldScore `0.2105`,
  `val_ca_drmsd=14.8883`, `val_pred_ca_rg=6.6400`, and
  `val_true_ca_rg=15.4034` with `simplex_aux_weight=1.0`. Step 500 reached
  `val_lddt_ca=0.2111`, FoldScore `0.2241`, `val_ca_drmsd=16.1468`,
  `val_pred_ca_rg=5.0536`, and `val_true_ca_rg=15.4034` with
  `simplex_aux_weight=0.5`. The run is rejected: fixed-strength flag closure
  was not enough to beat E42 and appears to over-suppress the sparse complex
  late in the gate.
- E45 local implementation at 2026-05-10 15:23 EDT: add
  `full_msa_to_face_flag_closure_soft` with the same soft flag-complex
  construction as E44 but `simplex_boundary_closure_weight=0.1`. This is a
  zero-parameter ablation to distinguish whether E44 failed because closure
  is harmful or because a `0.5` mask blend suppresses selected cells too
  strongly from step 1.
- E45 local checks: `python -m py_compile scripts/run_nanofold_public_benchmarks.py`
  passed; focused runner tests passed (`2 passed`); affected runner/trainer
  tests passed (`82 passed`); full `python -m pytest -q` passed; `git diff
  --check` passed. Parameter audit gives AF2-medium `3,106,642`,
  SimplexFold `3,106,690`, and E45 soft flag closure `3,106,690`.
- E45 launched on owned Runpod pod `p2roc93zgk4ho9` at 2026-05-10
  15:33 EDT from SimplexFold commit `daa5e59`. The stopped pod's
  `/workspace` was empty, so I recloned the pushed SimplexFold branch and
  copied only public NanoFold assets with `COPYFILE_DISABLE=1`. Remote audit:
  public train/val/all counts `10000/1000/11000`, feature/label `.npz`
  counts `11000/11000`, no hidden or sidecar data paths, H100 CUDA
  available, FoldScore import works, AF2-medium `3,106,642`, E45 soft flag
  closure `3,106,690`, `simplex_boundary_closure_weight=0.1`,
  `simplex_boundary_closure_temperature=1.0`, within the 5% AF2-medium
  budget. Run name: `e45_flag_closure_soft_s500_c256_m64`.
- E45 completed and the owned Runpod pod was stopped at 2026-05-10
  19:43 UTC. Step 250 reached `val_lddt_ca=0.2477`, FoldScore `0.2112`,
  `val_ca_drmsd=14.8438`, `val_pred_ca_rg=6.4528`, and
  `val_true_ca_rg=15.4034` with `simplex_aux_weight=1.0`. Step 500 reached
  `val_lddt_ca=0.2273`, FoldScore `0.1992`, `val_ca_drmsd=14.9228`,
  `val_pred_ca_rg=7.3539`, and `val_true_ca_rg=15.4034` with
  `simplex_aux_weight=0.5`. The run is rejected: lower closure strength was
  slightly better than E44 early and less collapsed late, but still below E42
  and still degraded by the final checkpoint.
- E46 local implementation at 2026-05-10 15:51 EDT: add
  `full_msa_to_face_expanded_complex`, a zero-parameter variant that enables
  MSA-to-face messages and raises `simplex_neighbor_k` from 12 to 14. This
  expands selected candidate faces from 66 to 91 and selected candidate tetras
  from 220 to 364 per anchor before masking. The intent is topological:
  increase sparse 2-/3-cell coverage without widening the AF2 trunk or adding
  a generic dense coordinate loss.
- E46 local checks: `python -m py_compile scripts/run_nanofold_public_benchmarks.py`
  passed; focused runner/budget tests passed (`3 passed`); affected
  runner/trainer tests passed (`85 passed`); full `python -m pytest -q`
  passed; `git diff --check` passed. Parameter audit gives AF2-medium
  `3,106,642`, SimplexFold `3,106,690`, and E46 expanded complex
  `3,106,690` with `simplex_neighbor_k=14`.
- E46 launched on owned Runpod pod `p2roc93zgk4ho9` at 2026-05-10
  15:59 EDT from commit `a53fe3f`, with the full public NanoFold manifests
  staged cleanly (`10000/1000/11000` train/val/all; `11000/11000`
  feature/label files; `bad_paths=0`). H100 CUDA was available, FoldScore
  import worked, AF2-medium had `3,106,642` parameters, and E46 had
  `3,106,690` parameters.
- E46 completed and the owned Runpod pod was stopped at 2026-05-10
  16:08 EDT. Step 250 was best:
  `val_lddt_ca=0.25172581151127815`, FoldScore `0.20692050736397505`,
  `val_ca_drmsd=14.45149451494217`, `val_pred_ca_rg=7.104867935180664`,
  and `val_true_ca_rg=15.403406739234924` with
  `simplex_aux_weight=1.0`. Step 500 ended at
  `val_lddt_ca=0.23270462825894356`, FoldScore `0.22145121730864048`,
  `val_ca_drmsd=15.505868315696716`, `val_pred_ca_rg=5.784029453992844`,
  and `val_true_ca_rg=15.403406739234924` with
  `simplex_aux_weight=0.5`.
- E46 interpretation: increasing selected complex coverage is not enough by
  itself. The early checkpoint is slightly better than E43-E45 but remains
  below the more promising E22/E25/E30 gates, and the final radius-of-gyration
  collapse argues against carrying a larger fixed `K` into the main branch.
- E47 local implementation at 2026-05-10 16:31 EDT: add an auxiliary
  flag-closure curriculum. The new `simplex_cell_closure_weight` reweights
  selected face/tetra coordinate-realization losses by a soft true-boundary
  closure score, but leaves the selected cell masks and message passing
  unchanged. The `full_msa_to_face_aux_closure` variant is architecturally
  identical to `full_msa_to_face`, so this is a zero-parameter topological
  loss/curriculum rather than a new module.
- E47 focused local checks: `python -m py_compile minalphafold/simplex.py
  minalphafold/losses.py minalphafold/trainer.py
  scripts/run_nanofold_public_benchmarks.py` passed; focused closure,
  scheduler, loss-builder, and CLI tests passed (`7 passed`).
- E47 broader local checks: affected suites
  `tests/test_simplex.py tests/test_nanofold_public_benchmarks.py
  tests/test_trainer.py -q` passed; full `python -m pytest -q` passed;
  `git diff --check` passed. `python -m ruff check .` could not run because
  `ruff` is not installed in the local interpreter. `python -m mypy
  minalphafold scripts` still fails on pre-existing typing/import issues,
  including `nanofold.metrics`, OpenMM/PDBFixer stubs, and existing
  structure-module/script annotations. `python -m pyright --warnings` still
  fails because the local pyright environment cannot resolve `torch`, `numpy`,
  `nanofold.metrics`, and other existing optional dependencies.
- E47 parameter audit: AF2-medium pair-only baseline `3,106,642`,
  SimplexFold medium `3,106,690`, and `full_msa_to_face_aux_closure`
  `3,106,690`. The variant leaves `simplex_boundary_closure_weight=0.0`, so
  it does not apply the E44/E45 message-mask closure.
- E47 launched on newly owned Runpod pod `lx0mqtjofo36wd`
  (`codex-simplexfold-e47-runpod-20260510`) at 2026-05-10 16:28 EDT from
  commit `ce35f94`. The first remote audit caught `hidden_val.txt` in the
  copied manifests and missing `nanofold.metrics`; I removed the hidden
  manifest from `/workspace` and synced only the public NanoFold package code.
  Clean launch audit after correction: public train/val/all manifest counts
  `10000/1000/11000`, feature/label `.npz` counts `11000/11000`,
  `bad_paths=0`, H100 CUDA available, FoldScore import works, AF2-medium
  pair-only `3,106,642`, E47 `3,106,690`, and
  `simplex_boundary_closure_weight=0.0`.
- E47 completed and the owned Runpod pod was stopped at 2026-05-10
  16:44 EDT. Step 250 was best:
  `val_lddt_ca=0.2466127322986722`, FoldScore `0.2070165267214179`,
  `val_ca_drmsd=14.511326014995575`, `val_pred_ca_rg=7.16933998465538`,
  `val_true_ca_rg=15.403406739234924`, `simplex_aux_weight=1.0`, and
  `simplex_cell_closure_weight=0.0`. Step 500 ended at
  `val_lddt_ca=0.22621477395296097`, FoldScore `0.2182228360325098`,
  `val_ca_drmsd=15.733196377754211`, `val_pred_ca_rg=5.558059215545654`,
  `val_true_ca_rg=15.403406739234924`, `simplex_aux_weight=0.5`, and
  `simplex_cell_closure_weight=0.5`.
- E47 interpretation: the auxiliary-only closure curriculum avoids the
  message-mask suppression of E44/E45 but still does not recover the stronger
  early-validation band and still collapses by the final checkpoint. Treat
  closure heuristics as a rejected family for now.
- Reference PDFs saved locally under `references/papers/` and read in full:
  `hands_on_geometric_deep_learning_nodes_to_complexes.pdf` and
  `2509.03885v1.pdf`. The directory ignores PDFs by default, so the local
  copies are available in the repo checkout but not tracked for public
  redistribution. Tracked distilled notes were added to
  `references/papers/READING_NOTES.md`.
- Reading-note synthesis: the TDL guide reinforces that the topology
  construction step, incidence/adjacency operators, and intra-/inter-rank
  aggregation are part of the model. Topotein adds the protein-specific lesson
  that combinatorial complexes are useful partly because they avoid strict
  boundary constraints. This agrees with E44-E47: closure heuristics are a
  principled idea in a flag/simplicial-complex view, but they appear mismatched
  to the flexible protein complex we need here.
- Updated next direction: E48 should test an adaptive local-to-global
  topology curriculum rather than another closure variant. Follow-on E49
  should test Topotein-style outer-edge selected-cell communication through
  boundary edges if E48 does not recover the stronger early-validation band.
- E48 local implementation: added a training-only
  `simplex_local_neighbor_k` schedule that passes
  `simplex_local_neighbor_k_override` through trainer/model/evoformer into
  the simplex adapter. The override changes only selected-neighborhood
  construction during training; eval/inference keep the static model config.
  The new benchmark variant `full_msa_to_face_topology_curriculum` is
  architecturally identical to `full_msa_to_face`.
- E48 local checks: py_compile passed for `minalphafold/simplex.py`,
  `minalphafold/evoformer.py`, `minalphafold/model.py`,
  `minalphafold/trainer.py`, and
  `scripts/run_nanofold_public_benchmarks.py`; focused tests passed for the
  adapter override, schedule/model-input propagation, variant config, and CLI
  parsing (`4 passed`); affected suites
  `tests/test_simplex.py tests/test_nanofold_public_benchmarks.py
  tests/test_trainer.py -q` passed; full `python -m pytest -q` passed;
  `git diff --check` passed. `python -m ruff check .` could not run because
  `ruff` is not installed. `python -m mypy minalphafold scripts` still fails
  on pre-existing typing/import issues including NumPy savez typing,
  structure-module list/tensor annotations, OpenMM/PDBFixer stubs,
  `nanofold.metrics`, EMA model typing, and runner row typing.
  `python -m pyright --warnings` still fails broadly because the local
  pyright environment cannot resolve Torch/NumPy/Modal/OpenMM and reports
  existing optional/type issues.
- E48 parameter audit: AF2-medium pair-only baseline `3,106,642`,
  SimplexFold medium `3,106,690`, and
  `full_msa_to_face_topology_curriculum` `3,106,690`. The schedule is
  zero-parameter and leaves the static `simplex_local_neighbor_k` at `0`.
- E48 Runpod gate plan: 500 steps at crop 256 / MSA depth 64, E15 selected
  coordinate and boundary-distance losses, `simplex_aux_weight` annealed
  `1.0 -> 0.5` over steps 250-500, and `simplex_local_neighbor_k` annealed
  `4 -> 0` over steps 250-500. Do not write to `EXPERIMENT_RESULTS.md` until
  the Runpod validation returns.
- E48 launched on newly owned Runpod pod `mttz64sa9mhut2`
  (`codex-simplexfold-e48-runpod-20260510`) at 2026-05-10 17:50 EDT from
  commit `30f51a3`. Clean launch audit: public train/val/all manifest counts
  `10000/1000/11000`, feature/label `.npz` counts `11000/11000`,
  `bad_paths=0`, H100 CUDA available, FoldScore import works, AF2-medium
  pair-only `3,106,642`, SimplexFold medium `3,106,690`, E48 `3,106,690`,
  and static `simplex_local_neighbor_k=0`.
- E48 completed and the owned Runpod pod was stopped at 2026-05-10
  17:59 EDT. Local returned artifacts were copied under ignored
  `artifacts/nanofold_public_benchmarks/e48_topology_curriculum_s500_c256_m64/`.
  Step 250 reached `val_lddt_ca=0.1002268809825182`, FoldScore
  `0.18271444644778967`, `val_ca_drmsd=14.2001`, and predicted/true
  C-alpha radius of gyration `13.9544 / 15.4034` while
  `simplex_local_neighbor_k=4.0` and `simplex_aux_weight=1.0`. Step 500
  ended at `val_lddt_ca=0.2273553814738989`, FoldScore
  `0.21913115680217743`, `val_ca_drmsd=15.774888902902603`, radius of
  gyration `5.532628566026688 / 15.403406739234924`,
  `simplex_local_neighbor_k=0.0`, and `simplex_aux_weight=0.5`.
- E48 interpretation: the sequence-local scaffold curriculum does not recover
  the stronger early-validation band and still ends with coordinate collapse.
  Move next to the Topotein-inspired outer-edge communication idea, which
  changes inter-cell message passing rather than the selected-neighborhood
  schedule.
- E49 local implementation: added `cell_outer_edge_context` and
  `simplex_outer_edge_context_scale`. For each selected face/tetra, the pass
  gathers selected directed pair edges from vertices in the cell to neighbors
  outside the cell, plus reverse directed states, then updates the higher-rank
  cochain from that pooled edge context. This is the Topotein-style
  outer-edge neighborhood; it differs from E39's shared-boundary face
  averaging.
- E49 local checks: `python -m py_compile minalphafold/simplex.py
  minalphafold/model_config.py scripts/run_nanofold_public_benchmarks.py`
  passed; focused tests for the context mask, adapter effect, runner variant,
  and parameter budget passed (`5 passed`); affected suites
  `tests/test_simplex.py tests/test_nanofold_public_benchmarks.py
  tests/test_trainer.py -q` passed.
- E49 parameter audit: AF2-medium pair-only baseline `3,106,642`,
  SimplexFold medium `3,106,690`, and
  `full_msa_to_face_outer_edge_context` `3,183,282`, within the 5% budget.
  Proposed Runpod gate: same 500-step crop 256 / MSA depth 64 protocol as
  E48/E47, E15 selected coordinate and boundary-distance losses,
  `simplex_aux_weight` annealed `1.0 -> 0.5`, and
  `full_msa_to_face_outer_edge_context`.
- E49 launched on owned Runpod pod `mttz64sa9mhut2` after restart at
  2026-05-10 18:15 EDT from commit `b546c79`. The restart did not preserve
  `/workspace`, so the repo was cloned fresh and public data was restaged.
  Clean launch audit: public train/val/all manifest counts `10000/1000/11000`,
  feature/label `.npz` counts `11000/11000`, `bad_paths=0`, H100 CUDA
  available, FoldScore import works, AF2-medium pair-only `3,106,642`,
  SimplexFold medium `3,106,690`, E49 `3,183,282`, and
  `simplex_outer_edge_context_scale=0.25`.
- E49 completed and the owned Runpod pod was stopped at 2026-05-10
  18:24 EDT. Local returned artifacts were copied under ignored
  `artifacts/nanofold_public_benchmarks/e49_outer_edge_context_s500_c256_m64/`.
  Step 250 reached `val_lddt_ca=0.24210394732654095`, FoldScore
  `0.2118359049782157`, `val_ca_drmsd=15.099262595176697`, and
  predicted/true C-alpha radius of gyration `6.338007271289825 /
  15.403406739234924`. Step 500 ended at
  `val_lddt_ca=0.26945804711431265`, FoldScore `0.24290458485484123`,
  `val_ca_drmsd=14.537729948759079`, and radius of gyration
  `6.785757273435593 / 15.403406739234924`.
- E49 interpretation: directed outer-edge context helps over E47/E48 final
  checkpoints but remains below E22/E25/E30 and far below E15. It is not the
  next main branch by itself. Prefer a topology-grounded expansion/readout
  change that directly counteracts the repeated radius-of-gyration collapse.
- E50 live plan: add a selected-boundary expansion hinge as a zero-parameter
  realization loss. For each selected face/tetra, penalize only contraction
  of predicted C-alpha boundary edges below their true selected boundary
  lengths. This stays in the simplicial/topological motivation because it
  operates on the boundary 1-skeleton induced by selected 2-/3-cells rather
  than on all residue pairs or global radius of gyration.
- E50 implementation in progress: add loss knobs
  `simplex_face_coordinate_expansion_weight`,
  `simplex_tetra_coordinate_expansion_weight`, and
  `simplex_coordinate_expansion_tolerance`; wire them through the trainer and
  NanoFold runner; add `full_msa_to_face_expansion_hinge` as an architecture
  copy of `full_msa_to_face`, leaving the static parameter budget unchanged.
  Planned gate uses expansion weights `0.5/0.5`, the E15 selected coordinate
  and boundary-distance terms, and the usual `simplex_aux_weight 1.0 -> 0.5`
  anneal over steps 250-500.
- E50 local checks: `python -m py_compile minalphafold/simplex.py
  minalphafold/losses.py minalphafold/trainer.py
  scripts/run_nanofold_public_benchmarks.py` passed; focused tests for the
  contraction-only loss, loss override plumbing, runner variant/CLI flags,
  and budget passed; affected suites `tests/test_simplex.py
  tests/test_nanofold_public_benchmarks.py tests/test_trainer.py -q`
  passed; full `python -m pytest -q` passed; `git diff --check` passed.
  `python -m ruff check .` cannot run because ruff is not installed.
  `python -m mypy minalphafold scripts` and `python -m pyright --warnings`
  still fail on the same pre-existing typing/import environment issues noted
  after E49.
- E50 parameter audit: AF2-medium pair-only baseline `3,106,642`,
  SimplexFold medium `3,106,690`, and
  `full_msa_to_face_expansion_hinge` `3,106,690`, only `0.0015%` above
  AF2-medium and within the 5% budget.
- E50 launched on owned Runpod pod `mttz64sa9mhut2` after restart at
  2026-05-10 18:37 EDT from commit `8a126d4`. Clean launch audit after
  restaging public data: public train/val/all manifest counts
  `10000/1000/11000`, feature/label `.npz` counts `11000/11000`,
  encoded-chain `bad_paths=0`, H100 CUDA available, FoldScore import works,
  AF2-medium pair-only `3,106,642`, SimplexFold medium `3,106,690`, E50
  `3,106,690`, and topology flags `(simplicial, faces, tetra, MSA-to-face)`
  all true.
- E50 completed on Runpod. Local returned artifacts were copied under ignored
  `artifacts/nanofold_public_benchmarks/e50_selected_boundary_expansion_s500_c256_m64/`.
  Step 250 reached `val_lddt_ca=0.1592852883040905`, FoldScore
  `0.19877275824546814`, `val_ca_drmsd=14.226054191589355`, and
  predicted/true C-alpha radius of gyration `10.652208685874939 /
  15.403406739234924`. Step 500 ended at
  `val_lddt_ca=0.27311410568654537`, FoldScore `0.2333886418491602`,
  `val_ca_drmsd=14.780930042266846`, and radius of gyration
  `6.60868564248085 / 15.403406739234924`.
- E50 interpretation: the selected-boundary expansion hinge successfully
  pushed early predicted radius toward the true scale, but it did not improve
  early lDDT and the final model still collapsed back near the weak-pilot
  radius band. Reject as a standalone loss. The next idea should use
  expanded boundary geometry inside the selected topology/readout pathway
  rather than applying a stronger auxiliary coordinate-only penalty.
- E51 live plan: run the E50 selected-boundary expansion hinge together with
  `full_msa_to_face_structure_readout`. This keeps the same selected
  2-/3-cell boundary realization objective, but routes simplex pair/single
  readouts into the structure module with `simplex_structure_readout_scale=0.25`.
  The point is to test whether expanded selected-boundary geometry needs to
  be on the atom-placement path, not merely attached as an auxiliary loss.
- E51 launch notes: the stopped H100 pod `mttz64sa9mhut2` could not be
  resumed because its host had no free GPU. A fresh owned H100 pod
  `txeom1sd4r00o9` and then owned A100 pod `8egrtbcrp1n8di` both stayed at
  `uptimeSeconds=0`/SSH not ready and were stopped/deleted. A replacement
  owned A100 SXM pod `ty7dscglwdg847` with explicit `22/tcp` port exposed
  came up cleanly. Clean launch audit after restaging public data: public
  train/val/all manifest counts `10000/1000/11000`, feature/label `.npz`
  counts `11000/11000`, encoded-chain `bad_paths=0`, A100 CUDA available,
  FoldScore import works, AF2-medium pair-only `3,106,642`, SimplexFold
  medium `3,106,690`, E51 `3,106,690`, and
  `simplex_structure_readout_scale=0.25`.
- E51 completed on Runpod and the owned E51 pod was stopped/deleted. Local
  returned artifacts were copied under ignored
  `artifacts/nanofold_public_benchmarks/e51_expansion_structure_readout_s500_c256_m64/`.
  Step 250 reached `val_lddt_ca=0.23751059919595718`, FoldScore
  `0.20890024863183498`, `val_ca_drmsd=14.775567382574081`, and
  predicted/true C-alpha radius of gyration `7.081045240163803 /
  15.40340667963028`. Step 500 ended at
  `val_lddt_ca=0.2272480195388198`, FoldScore `0.22329253144562244`,
  `val_ca_drmsd=15.716134786605835`, and radius of gyration
  `5.762171119451523 / 15.40340667963028`.
- E51 interpretation: broad simplicial structure readout does not rescue the
  selected-boundary expansion hinge. It removes E50's step-250 radius benefit
  and finishes below E50/E49. The next branch should probably stop adding
  auxiliary/readout paths and return to the E15 full-MSA-to-face family with
  an optimization or curriculum change around the existing selected-boundary
  coordinate losses.
- E52 live plan: add selected cell dropout to the E15 `full_msa_to_face`
  family. The new `simplex_cell_dropout` masks selected face/tetra cells
  during training only, before higher-rank message passing and auxiliary
  selected-boundary losses consume the masks. Evaluation uses the full
  selected complex. This is a zero-parameter topological regularizer over
  explicit 2-/3-cell cochains, not a dense coordinate or lDDT-targeted loss.
- E52 proposed Runpod gate: `full_msa_to_face_cell_dropout`,
  `simplex_cell_dropout=0.15`, crop 256, MSA depth 64, 500 steps, E15
  selected coordinate and boundary-distance weights, and
  `simplex_aux_weight` annealed `1.0 -> 0.5` over steps 250-500. Do not write
  E52 to `EXPERIMENT_RESULTS.md` until the Runpod run returns.
- E52 launched on owned temporary Runpod H100 pod `thyay2y2e53fuh`
  (`codex-simplexfold-e52-runpod-20260510`) from commit `a205a93`. Clean
  launch audit: public train/val/all manifest counts `10000/1000/11000`,
  hidden manifest absent, feature/label `.npz` counts `11000/11000`,
  encoded-chain `bad_paths=0`, H100 CUDA available, FoldScore import works,
  AF2-medium pair-only `3,106,642`, SimplexFold medium `3,106,690`, E52
  `3,106,690`, and `simplex_cell_dropout=0.15`.
- E52 completed on Runpod and the temporary pod was stopped/deleted. Local
  returned artifacts were copied under ignored
  `artifacts/nanofold_public_benchmarks/e52_selected_cell_dropout_s500_c256_m64/`.
  Step 250 reached `val_lddt_ca=0.22932657599449158`, FoldScore
  `0.21687721833586693`, `val_ca_drmsd=15.73187729716301`, and
  predicted/true C-alpha radius of gyration `5.546932339668274 /
  15.403406739234924`. Step 500 ended at
  `val_lddt_ca=0.2629628051072359`, FoldScore `0.23007656913250685`,
  `val_ca_drmsd=14.239901304244995`, and radius
  `7.20567774772644 / 15.403406739234924`.
- E52 interpretation: selected cell dropout at `0.15` is too destructive.
  It rebounds somewhat by step 500 but remains below E49/E50 and the older
  E22/E25/E30 early range. Reject this strength and return to the E15
  `full_msa_to_face` scaffold for optimization-scale testing.
- E53 live plan: extend E25's effective-batch-8 test to 1000 optimizer steps
  using the best pre-anneal E09/E15 scaffold: `full_msa_to_face`, crop 256,
  MSA depth 64, `batch_size=1`, `grad_accum_steps=8`, selected coordinate
  weights `1.0/1.0`, and selected boundary-distance weights `0.5/0.5`.
  Evaluate at steps 500 and 1000. Do not write E53 to
  `EXPERIMENT_RESULTS.md` until the Runpod run returns.
- E53 launched on owned temporary Runpod H100 pod `egsopc48v9fjz8`
  (`codex-simplexfold-e53-runpod-20260510`) from commit `26cc074`. Clean
  launch audit: public train/val/all manifest counts `10000/1000/11000`,
  hidden manifest absent, feature/label `.npz` counts `11000/11000`,
  encoded-chain `bad_paths=0`, H100 CUDA available, FoldScore import works,
  AF2-medium pair-only `3,106,642`, SimplexFold medium `3,106,690`, E53
  `3,106,690`, and MSA-to-face enabled.
- E53 completed on Runpod. Local returned artifacts were copied under ignored
  `artifacts/nanofold_public_benchmarks/e53_effective_batch8_s1000_c256_m64/`.
  Step 500 reached `val_lddt_ca=0.28071699384599924`, FoldScore
  `0.25139740761369467`, `val_ca_drmsd=14.983468919992447`, and
  predicted/true C-alpha radius of gyration `6.182499825954437 /
  15.403406739234924`. Step 1000 ended at
  `val_lddt_ca=0.3479556031525135`, FoldScore `0.2728881845250726`,
  `val_ca_drmsd=12.637772142887115`, and radius
  `8.518412441015244 / 15.403406739234924`.
- E53 interpretation: effective batch 8 is promising once it runs past the
  first 500 optimizer steps. It is still below E15's `0.3556` best, but it
  strongly beats E25 and gets close enough to justify an E15-style auxiliary
  anneal continuation before deciding whether to spend on 30k steps.
- E54 launched on the same owned H100 pod `egsopc48v9fjz8`, resuming E53 from
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e53_effective_batch8_s1000_c256_m64/checkpoints/full_msa_to_face_latest.pt`.
  The continuation runs to step 2000 with effective batch 8, selected
  coordinate weights `1.0/1.0`, selected boundary-distance weights `0.5/0.5`,
  and `simplex_aux_weight` ramped from `1.0` to `0.5` over steps 1000-1500.
  Do not write E54 to `EXPERIMENT_RESULTS.md` until the Runpod run returns.
- E54 completed on Runpod. Local returned artifacts were copied under ignored
  `artifacts/nanofold_public_benchmarks/e54_effective_batch8_aux_anneal_s2000_c256_m64/`.
  Step 1500 reached `val_lddt_ca=0.333083800971508`, FoldScore
  `0.2890670131891966`, `val_ca_drmsd=13.365413516759872`, and
  predicted/true C-alpha radius of gyration `7.538277685642242 /
  15.403406739234924` with `simplex_aux_weight=0.5`. Step 2000 recovered to
  `val_lddt_ca=0.35393444262444973`, FoldScore `0.32407183200120926`,
  `val_ca_drmsd=11.933855026960373`, and radius
  `9.240868210792542 / 15.403406739234924`.
- E54 interpretation: the E15-style auxiliary anneal works under effective
  batch 8 after a transient dip. Step 2000 nearly ties E15 lDDT while beating
  E15 FoldScore and dRMSD. Continue one more gate with
  `simplex_aux_weight=0.5` held constant.
- E55 launched on the same owned H100 pod `egsopc48v9fjz8`, resuming E54 from
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e54_effective_batch8_aux_anneal_s2000_c256_m64/checkpoints/full_msa_to_face_latest.pt`.
  The continuation runs to step 3000 with effective batch 8 and constant
  `simplex_aux_weight=0.5`. Do not write E55 to `EXPERIMENT_RESULTS.md` until
  the Runpod run returns.
- E55 completed on Runpod. Local returned artifacts were copied under ignored
  `artifacts/nanofold_public_benchmarks/e55_effective_batch8_aux05_s3000_c256_m64/`.
  Step 2500 reached `val_lddt_ca=0.3423902466893196`, FoldScore
  `0.3378663770854473`, `val_ca_drmsd=11.098472356796265`, and predicted/true
  C-alpha radius of gyration `10.338989078998566 / 15.403406739234924`.
  Step 3000 recovered to `val_lddt_ca=0.36041587218642235`, FoldScore
  `0.34508529864251614`, `val_ca_drmsd=11.327985882759094`, and radius
  `10.050656735897064 / 15.403406739234924`.
- E55 interpretation: this is the new current best and the first returned
  effective-batch-8 run to beat E15's `val_lddt_ca=0.3556`. The target
  `>0.7` is still far away, but the optimizer-regime branch is now clearly
  better than more auxiliary/readout variants.
- E56 launched on the same owned H100 pod `egsopc48v9fjz8`, resuming E55 from
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e55_effective_batch8_aux05_s3000_c256_m64/checkpoints/full_msa_to_face_latest.pt`.
  The continuation runs to step 4000 with effective batch 8 and constant
  `simplex_aux_weight=0.5`. Do not write E56 to `EXPERIMENT_RESULTS.md` until
  the Runpod run returns.
- E56 completed on Runpod and the owned H100 pod `egsopc48v9fjz8` was
  stopped/deleted. Local returned artifacts were copied under ignored
  `artifacts/nanofold_public_benchmarks/e56_effective_batch8_aux05_s4000_c256_m64/`.
  Step 3500 reached `val_lddt_ca=0.3561620619148016`, FoldScore
  `0.3464267496019602`, `val_ca_drmsd=10.71201303601265`, and predicted/true
  C-alpha radius of gyration `10.921668142080307 / 15.403406739234924`.
  Step 4000 ended at `val_lddt_ca=0.35753354616463184`, FoldScore
  `0.3477634973824024`, `val_ca_drmsd=10.98043116927147`, and radius
  `10.319191247224808 / 15.403406739234924`.
- E56 interpretation: constant aux-0.5 continuation after E55 improves
  FoldScore and dRMSD but does not beat E55's lDDT. Do not keep extending
  this exact branch for the lDDT objective without analyzing why E55 is the
  local lDDT peak.
- E57 live plan: resume the E55 checkpoint from
  `artifacts/nanofold_public_benchmarks/e55_effective_batch8_aux05_s3000_c256_m64/checkpoints/full_msa_to_face_latest.pt`
  and continue to step 4000 with effective batch 8, selected coordinate
  weights `1.0/1.0`, selected boundary-distance weights `0.5/0.5`, and
  `simplex_aux_weight=0.75`. This is a selected-simplex auxiliary rewarm from
  the lDDT peak, not a dense metric-loss change. Do not write E57 to
  `EXPERIMENT_RESULTS.md` until the Runpod run returns.
- E57 launched on owned H100 Runpod pod `2lbuhxawih0vzl`
  (`codex-simplexfold-e57-runpod-20260511`) from commit `f62c180`. Clean
  launch audit after copying only public data/code: public train/val/all
  manifest counts `10000/1000/11000`, hidden manifest absent, encoded
  feature/label `.npz` counts `11000/11000`, encoded-chain `bad_paths=0`,
  E55 checkpoint present at
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e55_effective_batch8_aux05_s3000_c256_m64/checkpoints/full_msa_to_face_latest.pt`,
  H100 CUDA available, NanoFold `foldscore_components` import works,
  AF2-medium pair-only `3,106,642`, and E57 `3,106,690` parameters
  (`+0.0015%`).
- E57 remote process: Python PID `758`, log
  `/workspace/SimplexFold/logs/e57_aux075_rewarm_from_e55.log`, artifacts
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e57_aux075_rewarm_from_e55_s4000_c256_m64/`.
- E57 completed on Runpod. Local returned artifacts were copied under ignored
  `artifacts/nanofold_public_benchmarks/e57_aux075_rewarm_from_e55_s4000_c256_m64/`.
  Step 3500 reached `val_lddt_ca=0.33945486322045326`, FoldScore
  `0.35038202442228794`, `val_ca_drmsd=10.685163050889969`, and
  predicted/true C-alpha radius of gyration `11.53536543250084 /
  15.403406739234924`. Step 4000 ended at
  `val_lddt_ca=0.3465292062610388`, FoldScore `0.34945190511643887`,
  `val_ca_drmsd=10.70906126499176`, and radius `10.857420325279236 /
  15.403406739234924`.
- E57 owned Runpod pod `2lbuhxawih0vzl` was stopped and deleted after artifacts
  were copied. A post-delete lookup returned 404, as expected. No other
  Runpod instances were managed.
- E57 interpretation: aux-0.75 rewarming improves the global/FoldScore side of
  the objective, including the best FoldScore observed so far, but it worsens
  local C-alpha lDDT versus E55 and E56. Reject for the primary objective.
  More scalar auxiliary tuning is less promising than a Topotein-style
  architecture change.
- 2026-05-11 PDF reread: both local reference PDFs in `references/papers/`
  hash-match the downloads and were read via full text extraction with
  `pdftotext -layout`. The TDL guide reinforces that topology construction,
  incidence/adjacency operators, and intra-/inter-rank aggregation are model
  choices. Topotein gives the protein-specific path: directed residue
  interactions, outer-edge neighborhoods, edge-centric frames, and deep
  integration of higher-rank cell updates outperform superficial topological
  feature additions.
- E58 live plan: initialize from the E55 checkpoint with variant name
  `full_msa_to_face` but activate `simplex_outer_edge_context_scale=0.25` via
  a model-config CLI override. This keeps checkpoint metadata compatible while
  testing the Topotein-style directed outer-edge context path from the lDDT
  peak checkpoint. Keep selected coordinate weights `1.0/1.0`, selected
  boundary-distance weights `0.5/0.5`, effective batch 8, crop 256, MSA depth
  64, and `simplex_aux_weight=0.5`.
- E58 checkpoint note: directed outer-edge context adds new parameters, so a
  strict model+optimizer resume from E55 would fail. Use
  `--resume-model-weights-only` to load matching E55 tensors, initialize the
  new topology-context tensors fresh, and restart optimizer state while
  preserving the E55 step/history context.
- E58 launched on owned H100 Runpod pod `714wc1nzy3t8qz`
  (`codex-simplexfold-e58-runpod-20260511`) from commit `41af00a`. Clean
  launch audit: public train/val/all manifest counts `10000/1000/11000`,
  hidden manifest absent, feature/label `.npz` counts `11000/11000`,
  encoded missing paths `0`, H100 CUDA available, NanoFold
  `foldscore_components` import works, AF2-medium pair-only `3,106,642`, and
  E58 `3,183,282` parameters (`+2.47%`). Partial checkpoint load from E55
  loaded `1196` matching tensors, initialized `48` new/missing outer-edge
  context tensors, and had `0` shape mismatches.
- E58 remote process: wrapper PID `755`, Python PID `756`, log
  `/workspace/SimplexFold/logs/e58_outer_edge_context_from_e55.log`, artifacts
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e58_outer_edge_context_from_e55_s4000_c256_m64/`.
  The initial log confirms the E55 weights-only load and fresh optimizer.
- E58 step-3500 live note: `val_lddt_ca=0.3418876174837351`, FoldScore
  `0.35072777792811394`, `val_ca_drmsd=10.901953607797623`, and
  predicted/true C-alpha radius of gyration `11.124989807605743 /
  15.403406739234924`. This improves the global/FoldScore side but is well
  below E55's `0.3604` lDDT, matching the E57 tradeoff.
- E58 was stopped early after the step-3500 checkpoint because the primary
  lDDT objective clearly regressed and the outer-edge context run was much
  slower than the base continuation. Local returned artifacts were copied
  under ignored
  `artifacts/nanofold_public_benchmarks/e58_outer_edge_context_from_e55_s4000_c256_m64/`.
  The owned Runpod pod `714wc1nzy3t8qz` was stopped and deleted; a post-delete
  lookup returned 404, as expected. No other Runpod instances were managed.
- E58 interpretation: directed outer-edge context appears to help global
  consistency/FoldScore but harms local C-alpha lDDT when applied at scale
  `0.25` from freshly initialized context modules. Next test should damp or
  schedule this topology-context path rather than strengthen losses.
- E59 live plan: run a 3500-step gate from the E55 checkpoint with the same
  weights-only initialization, but set `simplex_outer_edge_context_scale=0.05`.
  This remains a Topotein-style architecture test while asking whether a weak
  outer-edge context correction can preserve E55's local lDDT.
- E59 launched on owned H100 Runpod pod `n5dtdxgjgk81de`
  (`codex-simplexfold-e59-runpod-20260511`) from commit `6f9750c`. Clean
  launch audit: public train/val/all manifest counts `10000/1000/11000`,
  hidden manifest absent, feature/label `.npz` counts `11000/11000`,
  encoded missing paths `0`, H100 CUDA available, NanoFold
  `foldscore_components` import works, AF2-medium pair-only `3,106,642`, and
  E59 `3,183,282` parameters (`+2.47%`) with
  `simplex_outer_edge_context_scale=0.05`.
- E59 remote process: wrapper PID `431`, Python PID `432`, log
  `/workspace/SimplexFold/logs/e59_outer_edge_context005_from_e55.log`,
  artifacts
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e59_outer_edge_context005_from_e55_s3500_c256_m64/`.
  Do not write E59 to `EXPERIMENT_RESULTS.md` until the Runpod run returns.
- E59 completed on Runpod. Local returned artifacts were copied under ignored
  `artifacts/nanofold_public_benchmarks/e59_outer_edge_context005_from_e55_s3500_c256_m64/`.
  Step 3500 reached `val_lddt_ca=0.34999977238476276`, FoldScore
  `0.35156833939254284`, `val_ca_drmsd=10.950208216905594`, and
  predicted/true C-alpha radius of gyration `11.197810173034668 /
  15.403406739234924`.
- E59 owned Runpod pod `n5dtdxgjgk81de` was stopped and deleted after
  artifacts were copied. A post-delete lookup returned 404, as expected. No
  other Runpod instances were managed.
- E59 interpretation: scale `0.05` recovers much of the lDDT lost by E58 and
  sets the best FoldScore so far, but it still misses E55's `0.3604` lDDT.
  The right next test is not another output loss; it is a scheduled activation
  of the same Topotein-style outer-edge context route so the resumed E55
  checkpoint can adapt to freshly initialized cell-context modules.
- E60 local implementation: added a training-time
  `simplex_outer_edge_context_runtime_scale` ramp that threads through
  `TrainingConfig`, `model_inputs_from_batch`, `AlphaFold2`,
  `SimplicialEvoformer`, and `SimplicialAdapter`. The model-config
  `simplex_outer_edge_context_scale` still allocates the context modules and
  sets validation-time scale; the runtime override gates the training-time
  contribution.
- E60 local checks passed: `python -m py_compile minalphafold/trainer.py
  minalphafold/simplex.py minalphafold/evoformer.py minalphafold/model.py
  scripts/run_nanofold_public_benchmarks.py`; targeted pytest for the new CLI,
  schedule, and adapter gate; and the 46-test slice covering
  `tests/test_nanofold_public_benchmarks.py`, the new adapter gate, the
  existing edge-frame gate, and the AF2-medium budget test. `git diff --check`
  passed. `python -m ruff ...` and `.venv/bin/ruff ...` could not run because
  `ruff` is not installed in the available Python environment or repo
  virtualenv.
- E60 live plan: resume E55 with weights-only initialization, set
  `simplex_outer_edge_context_scale=0.05`, and ramp
  `simplex_outer_edge_context_runtime_scale` from `0.0` to `0.05` across
  steps 3000-3500. Keep effective batch 8, crop 256, MSA depth 64, selected
  coordinate weights `1.0/1.0`, selected boundary-distance weights
  `0.5/0.5`, and `simplex_aux_weight=0.5`. Do not write E60 to
  `EXPERIMENT_RESULTS.md` until the Runpod run returns.
- E60 launched on owned H100 Runpod pod `yzy3zi29gzbfj4`
  (`codex-simplexfold-e60-runpod-20260511`) from commit `ede843b`. Clean
  launch audit after copying only public data/code: public train/val/all
  manifest counts `10000/1000/11000`, hidden manifest/path absent, feature and
  label `.npz` counts `11000/11000`, encoded-chain missing paths `0`, E55
  checkpoint present, H100 CUDA available, NanoFold `foldscore_components`
  import works, AF2-medium pair-only `3,106,642`, and E60 `3,183,282`
  parameters (`+2.47%`). Runtime outer-edge context scale audit:
  step 3000 `0.0`, step 3250 `0.025`, step 3500 `0.05`.
- E60 remote process: launch wrapper PID `1214`, Python PID `1215`, log
  `/workspace/SimplexFold/logs/e60_outer_edge_context_ramp005_from_e55.log`,
  artifacts
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e60_outer_edge_context_ramp005_from_e55_s3500_c256_m64/`.
  Heartbeat `check-simplexfold-e57-runpod` has been retargeted to this E60
  pod and must not touch any other Runpod instance.
- Added evaluation-only selected-complex diagnostics for future runs:
  active face/tetra cell counts, active fractions, boundary-edge mean/max
  reuse degree, and unique boundary-edge fraction. This follows the reference
  paper lesson that topology-aware runs should report properties of the
  constructed complex, not just coordinate metrics. Local checks:
  `python -m py_compile scripts/run_nanofold_public_benchmarks.py`,
  `python -m pytest tests/test_nanofold_public_benchmarks.py`, and
  `git diff --check` passed.
- 2026-05-11 PDF reread for current planning: the two user-provided PDFs are
  saved locally in `references/papers/` and hash-match the Downloads copies.
  I re-extracted both with `pdftotext -layout` and reread the complete text.
  The next experiments should continue to be justified by the topology view:
  alter selected-complex construction, incidence/adjacency message routes,
  outer-edge communication, or selected-cell geometric realization. Avoid
  detached all-pairs or direct lDDT-targeting losses unless the supervision is
  restricted to edges/faces/tetras selected by the learned sparse complex.
  Added the concrete experiment rules to `references/papers/READING_NOTES.md`.
- E61 fallback implementation while E60 is still running: added
  `simplex_edge_frame_message_runtime_scale`, an opt-in training-time ramp for
  the existing edge-frame scalarized selected-boundary message path. The static
  `simplex_edge_frame_message_scale` still allocates the edge-frame MLPs and
  sets validation-time scale; the runtime override gates only training. This
  is a topology/curriculum change, not a new loss: selected face/tetra
  cochains write through their directed boundary-edge frames, following the
  reference-paper edge-centric scalarization idea.
- E61 local checks passed: `python -m py_compile minalphafold/trainer.py
  minalphafold/simplex.py minalphafold/evoformer.py minalphafold/model.py
  scripts/run_nanofold_public_benchmarks.py`; focused runtime/parser/adapter
  tests (`4 passed`); and the affected 48-test slice covering
  `tests/test_nanofold_public_benchmarks.py`, edge-frame runtime gating,
  edge-frame adapter behavior, outer-edge runtime gating, and the edge-frame
  budget test. Do not add E61 to `EXPERIMENT_RESULTS.md` unless a Runpod run
  returns.
- Added selected-boundary geometry diagnostics for future Runpod validation:
  face/tetra boundary-edge length MAE, RMSE, contraction fraction, and
  boundary lDDT, computed only on the selected sparse face/tetra complex. This
  follows the reference-paper note that topology-aware runs should inspect the
  constructed complex, not just the final coordinates. Local checks:
  `python -m py_compile scripts/run_nanofold_public_benchmarks.py`,
  `python -m pytest tests/test_nanofold_public_benchmarks.py::test_simplex_boundary_geometry_metrics_report_selected_edge_errors tests/test_nanofold_public_benchmarks.py::test_simplex_topology_metrics_report_boundary_reuse`,
  and `python -m pytest tests/test_nanofold_public_benchmarks.py`.
- E60 completed on Runpod. Local returned artifacts were copied under ignored
  `artifacts/nanofold_public_benchmarks/e60_outer_edge_context_ramp005_from_e55_s3500_c256_m64/`.
  Step 3500 reached `val_lddt_ca=0.3461837060749531`, FoldScore
  `0.34306286089122295`, `val_ca_drmsd=10.923531711101532`, and
  predicted/true C-alpha radius of gyration `10.852194011211395 /
  15.403406739234924`.
- E60 owned Runpod pod `yzy3zi29gzbfj4` was stopped and deleted after
  artifacts were copied. A post-delete lookup returned 404, as expected. No
  other Runpod instances were managed.
- E60 interpretation: reject. The scheduled `0.0 -> 0.05` directed
  outer-edge context ramp did not preserve E55's `0.3604` lDDT and also lost
  the E59 FoldScore advantage. Pivot to the prepared E61 edge-frame
  boundary-message probe rather than continuing the cell-level outer-edge
  context family.
- E61 owned Runpod pod `h2dvec04rxyoxe`
  (`codex-simplexfold-e61-runpod-20260511`) is running from commit `7823038`.
  A first H100 SXM allocation (`il9j0uq3n7byx0`) never exposed SSH and was
  stopped/deleted; post-delete lookup returned 404. Only these owned pods were
  managed.
- E61 staging note: an initial transfer command began copying the whole local
  `data/manifests` directory, including `hidden_val.txt`; I interrupted that
  transfer, deleted the remote staging tree on the owned pod, and restaged
  only `train.txt`, `val.txt`, `all.txt`, public feature/label NPZ caches, the
  `nanofold` scoring package, and the E55 checkpoint. Final launch audit
  confirmed remote manifest files were exactly `all.txt`, `train.txt`, and
  `val.txt`; `hidden_val.txt` was absent.
- E61 launch audit passed: public train/val/all manifest counts
  `10000/1000/11000`, feature/label `.npz` counts `11000/11000`,
  encoded-chain missing paths `0`, E55 checkpoint present, H100 NVL CUDA
  available, NanoFold `foldscore_components` import works, AF2-medium
  pair-only `3,106,642`, and E61 edge-frame model `3,154,242` parameters
  (`+1.53%`). Runtime edge-frame scale audit: step 3000 `0.0`, step 3250
  `0.025`, step 3500 `0.05`.
- E61 remote process: wrapper PID `2401`, Python PID `2403`, data-worker
  Python PIDs observed under the same command, log
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e61_edge_frame_ramp005_from_e55_s3500_c256_m64/run.log`,
  artifacts
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e61_edge_frame_ramp005_from_e55_s3500_c256_m64/`.
  The log shows it resumed E55 at step 3000/examples 24000, loaded 1196
  matching model tensors, initialized 48 new/missing edge-frame tensors, and
  started a fresh optimizer. Do not add E61 to `EXPERIMENT_RESULTS.md` until
  the Runpod run returns.
- E61 status check while preparing the next fallback: the owned pod is still
  running with active GPU utilization. The remote history currently contains
  the inherited E55 rows through step 3000 and no E61-owned step-3500 row yet.
- E62 fallback implementation: added `simplex_hodge_face_runtime_scale`, a
  training-time ramp for the existing Hodge-style selected-face residual. The
  static `simplex_hodge_face_update_scale` controls validation-time scale; the
  runtime override gates training. This is a zero-parameter topology change:
  selected face states mix through lower shared-boundary-edge adjacency and
  upper selected-tetra co-boundary adjacency.
- E62 local checks passed: `python -m py_compile minalphafold/trainer.py
  minalphafold/simplex.py minalphafold/evoformer.py minalphafold/model.py
  scripts/run_nanofold_public_benchmarks.py`; focused parser/schedule/adapter
  tests (`4 passed`); and the affected 50-test slice covering
  `tests/test_nanofold_public_benchmarks.py`, the Hodge runtime gate, the
  edge-frame and outer-edge runtime gates, the trainer model-input curriculum
  test, and the Hodge zero-parameter budget test. Do not launch E62 unless E61
  returns below the E55/E56 lDDT band.
- E61 completed on Runpod. Local returned artifacts were copied under ignored
  `artifacts/nanofold_public_benchmarks/e61_edge_frame_ramp005_from_e55_s3500_c256_m64/`.
  Step 3500 reached `val_lddt_ca=0.345616951584816`, FoldScore
  `0.3470576610416174`, `val_ca_drmsd=10.773000329732895`, and predicted/true
  C-alpha radius of gyration `11.161340862512589 / 15.40340667963028`.
  Selected-boundary diagnostics show the selected face/tetra boundary lDDT
  remained low (`0.4750` / `0.4610`) with high contraction fractions
  (`0.7473` / `0.7471`).
- E61 owned Runpod pod `h2dvec04rxyoxe` was stopped and deleted after
  artifacts were copied. A post-delete lookup returned 404, as expected. No
  other Runpod instances were managed.
- E61 interpretation: reject. The scheduled edge-frame message path improved
  final dRMSD and global expansion relative to E55 but reduced the primary
  C-alpha lDDT back into the E57/E60 band. Move to the prepared E62 scheduled
  Hodge face residual rather than adding more output-coordinate losses.
- E62 launch attempt 1: owned Runpod pod `f3j3v4qd4f6w8w`
  (`codex-simplexfold-e62-runpod-20260511`) was created but never exposed SSH.
  It was stopped and deleted before any data staging or training. A post-delete
  lookup returned 404. No other Runpod instances were managed.
- E62 launch attempt 2: owned Runpod pod `39s6arzja95amz`
  (`codex-simplexfold-e62-runpod-20260511b`) is running from commit `4517f98`.
  Clean launch audit after copying only public data/code: public train/val/all
  manifest counts `10000/1000/11000`, remote manifest files exactly
  `all.txt`, `train.txt`, and `val.txt`, hidden manifest/path absent,
  feature/label `.npz` counts `11000/11000`, encoded-chain missing paths `0`,
  E55 checkpoint present, H100 NVL CUDA available, NanoFold
  `foldscore_components` import works, AF2-medium pair-only `3,106,642`, and
  E62 Hodge model `3,106,690` parameters (`+0.0015%`). Runtime Hodge scale
  audit: step 3000 `0.0`, step 3250 `0.025`, step 3500 `0.05`.
- E62 remote process: Python PID `2419`, data-worker Python PIDs observed
  under the same command, log
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e62_hodge_face_ramp005_from_e55_s3500_c256_m64/run.log`,
  artifacts
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e62_hodge_face_ramp005_from_e55_s3500_c256_m64/`.
  The log shows it resumed E55 at step 3000/examples 24000, loaded 1196
  matching model tensors, initialized 0 new/missing tensors, and started a
  fresh optimizer. Do not add E62 to `EXPERIMENT_RESULTS.md` until the Runpod
  run returns.
- E62 completed on Runpod. Local returned artifacts were copied under ignored
  `artifacts/nanofold_public_benchmarks/e62_hodge_face_ramp005_from_e55_s3500_c256_m64/`.
  Step 3500 reached `val_lddt_ca=0.34682471491396427`, FoldScore
  `0.34496236592531204`, `val_ca_drmsd=10.901632159948349`, and
  predicted/true C-alpha radius of gyration `10.727833718061447 /
  15.40340667963028`. Selected-boundary diagnostics showed face/tetra
  boundary lDDT `0.482933746650815` / `0.46936048567295074`, with contraction
  fractions `0.7394912205636501` / `0.7400957755744457`.
- E62 owned Runpod pod `39s6arzja95amz` was stopped and deleted after
  artifacts were copied. A post-delete lookup returned 404, as expected. No
  other Runpod instances were managed. The first E62 pod
  `f3j3v4qd4f6w8w` had already been stopped/deleted before data staging.
- E62 interpretation: reject. Hodge face incidence mixing slightly improves
  selected-boundary lDDT relative to E61, but it still reduces the main
  validation C-alpha lDDT below E55's `0.3604`. The next experiment should
  target the selected boundary 1-skeleton directly rather than adding another
  face/tetra message route.
- E63 live plan: resume E55 with `full_msa_to_face` and add small
  selected-boundary lDDT weights on the model-selected face/tetra boundary
  edges only (`0.05`/`0.05`). This is not a generic lDDT loss: the supervision
  is restricted to the boundary edges induced by learned sparse 2-/3-cells,
  so it directly tests whether the explicit simplicial complex can realize
  its own selected 1-skeleton without contraction. Do not add E63 to
  `EXPERIMENT_RESULTS.md` until a Runpod run returns.
- E63 launched on owned Runpod H100 NVL pod `0hm1lpiaqqx21a`
  (`codex-simplexfold-e63-runpod-20260511`) from commit `6bb49f8`. Clean
  launch audit after copying only public data/code: public train/val/all
  manifest counts `10000/1000/11000`, remote manifest files exactly
  `all.txt`, `train.txt`, and `val.txt`, hidden manifest/path absent,
  feature/label `.npz` counts `11000/11000`, encoded-chain missing paths `0`,
  E55 checkpoint present, H100 NVL CUDA available, NanoFold
  `foldscore_components` import works, AF2-medium pair-only `3,106,642`, and
  E63 model `3,106,690` parameters (`+0.0015%`). E63 uses
  `simplex_face_boundary_lddt_weight=0.05` and
  `simplex_tetra_boundary_lddt_weight=0.05` with the E55 selected coordinate
  and selected boundary-distance weights.
- E63 remote process: Python PID `2389`, launch log
  `/workspace/SimplexFold/logs/e63_boundary_lddt005_from_e55.log`, artifacts
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e63_boundary_lddt005_from_e55_s3500_c256_m64/`.
  Initial log shows the runner loaded `train=10000 val=1000 crop=256 msa=64`.
  Heartbeat `check-simplexfold-e57-runpod` has been retargeted to this E63
  pod and must not touch any other Runpod instance.
- E63 completed on Runpod. Local returned artifacts were copied under ignored
  `artifacts/nanofold_public_benchmarks/e63_boundary_lddt005_from_e55_s3500_c256_m64/`,
  including the launch log. Step 3500 reached
  `val_lddt_ca=0.3610654976218939`, FoldScore `0.3576338365674019`,
  `val_ca_drmsd=10.68150195479393`, and predicted/true C-alpha radius of
  gyration `11.430959939956665 / 15.40340667963028`.
- E63 selected-boundary diagnostics improved in the intended direction:
  face/tetra boundary lDDT `0.5208317879587412` / `0.5065008029341698`,
  contraction fractions `0.6897053830325603` / `0.691283892840147`, and
  boundary length MAE `2.784909226000309` / `2.9228954538702965`.
- E63 owned Runpod pod `0hm1lpiaqqx21a` was stopped and deleted after
  artifacts were copied. A post-delete lookup returned 404, as expected. No
  other Runpod instances were managed.
- E63 interpretation: keep for confirmation. The gain over E55 is small
  (`0.3611` versus `0.3604`), but it is the first recent branch to improve
  primary lDDT, FoldScore, dRMSD, radius expansion, and selected-boundary
  realization together. Next run should continue the E63 checkpoint to step
  4000 (`32,000` effective examples at batch 8) before calling the direction
  stable.
- E64 launch attempt 1: owned Runpod H200 NVL pod `4g78gy2fbgl5o7`
  (`codex-simplexfold-e64-runpod-20260511`) was created for the E63
  confirmation run but hit repeated remote `/workspace` input/output errors
  while copying public feature NPZs. No training launched. The pod was
  stopped/deleted and a post-delete lookup returned 404. No other Runpod
  instances were managed.
- E64 launch attempt 2: owned Runpod A100 SXM pod `r64q7czrpsaax4`
  (`codex-simplexfold-e64-runpod-20260511b`) also hit a remote `/workspace`
  input/output error during public feature transfer before any training
  launched. The pod was stopped/deleted and a post-delete lookup returned 404.
  No other Runpod instances were managed.
- E64 launch attempt 3: owned Runpod A100 SXM pod `76h4drrq0mhbxp`
  (`codex-simplexfold-e64-runpod-20260511c`) was launched with
  `volumeInGb=0` and a 160 GB container disk, so `/workspace` is local overlay
  storage rather than the Runpod network volume that failed on attempts 1-2.
  Clean launch audit after copying only public data/code: public train/val/all
  manifest counts `10000/1000/11000`, remote manifest files exactly
  `all.txt`, `train.txt`, and `val.txt`, hidden manifest/path absent,
  feature/label `.npz` counts `11000/11000`, encoded-chain missing paths `0`,
  E63 checkpoint present, A100 SXM CUDA available, NanoFold
  `foldscore_components` import works, AF2-medium pair-only `3,106,642`, and
  E64 model `3,106,690` parameters (`+0.0015%`).
- E64 remote process: Python PID `4466`, launch log
  `/workspace/SimplexFold/logs/e64_boundary_lddt005_from_e63.log`, artifacts
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e64_boundary_lddt005_from_e63_s4000_c256_m64/`.
  The log shows it resumed E63 at step 3500/examples 28000, loaded 1196
  matching model tensors, initialized 0 new/missing tensors, and started a
  fresh optimizer. Heartbeat `check-simplexfold-e57-runpod` has been
  retargeted to this E64 pod and must not touch any other Runpod instance.
  Do not add E64 to `EXPERIMENT_RESULTS.md` until the Runpod run returns.
- E64 launch attempt 3 was treated as stalled before result. After more than
  80 minutes, the process held GPU memory but showed no useful GPU
  utilization, high CPU activity, and no step-4000 history row or
  `results.json`. A separate CUDA matrix-multiply smoke on the same pod
  succeeded, so the B300/A100 class CUDA runtime itself was not the blocker.
  The owned A100 pod `76h4drrq0mhbxp` was stopped/deleted and a post-delete
  lookup returned 404. No artifacts or results were copied, and no E64 result
  was recorded.
- E64 launch attempt 4: owned Runpod B300 pod `ow3ex8z84jypbs`
  (`codex-simplexfold-e64-runpod-20260511d`) is the active confirmation run.
  It uses `volumeInGb=0` with a 160 GB container disk so `/workspace` is local
  overlay storage. Clean launch audit after copying only public data/code:
  public train/val/all manifest counts `10000/1000/11000`, remote manifest
  files exactly `all.txt`, `train.txt`, and `val.txt`, hidden manifest/path
  absent, feature/label `.npz` counts `11000/11000`, encoded-chain missing
  paths `0`, E63 checkpoint present, B300 CUDA available, NanoFold
  `foldscore_components` import works, AF2-medium pair-only `3,106,642`, and
  E64 model `3,106,690` parameters (`+0.0015%`).
- Active E64 B300 remote process: wrapper PID `962`, launch log
  `/workspace/SimplexFold/logs/e64_boundary_lddt005_from_e63.log`, artifacts
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e64_boundary_lddt005_from_e63_s4000_c256_m64/`.
  The log shows it resumed E63 at step 3500/examples 28000, loaded 1196
  matching model tensors, initialized 0 new/missing tensors, and started a
  fresh optimizer. Heartbeat `check-simplexfold-e57-runpod` has been
  retargeted to the B300 pod `ow3ex8z84jypbs` only and must not touch any
  other Runpod instance. Do not add E64 to `EXPERIMENT_RESULTS.md` until this
  Runpod run returns.
- E64 B300 first status check: wrapper PID `962` and child Python processes
  are live, GPU utilization is active, and GPU memory is allocated. The run
  metadata and inherited E63/E55 history rows are present through step 3500,
  but there is not yet a step-4000 validation row or `results.json`.
- E65 local implementation while E64 is running: added schedulable
  selected-boundary lDDT weights for face/tetra boundary 1-skeleton
  realization. New runner/trainer flags:
  `--simplex-face-boundary-lddt-weight-final`,
  `--simplex-tetra-boundary-lddt-weight-final`,
  `--simplex-boundary-lddt-ramp-start-step`, and
  `--simplex-boundary-lddt-ramp-steps`. This is still topology-native because
  it only changes supervision on boundary edges induced by model-selected
  faces/tetras and adds no parameters.
- E65 local checks passed:
  `python -m py_compile minalphafold/trainer.py scripts/run_nanofold_public_benchmarks.py`
  and
  `python -m pytest tests/test_nanofold_public_benchmarks.py tests/test_trainer.py::test_apply_loss_weight_schedule_ramps_research_weights tests/test_trainer.py::test_alphafold_loss_overrides_simplex_coordinate_weights`.
  Do not launch E65 until E64 returns and the concrete direction is selected
  from the step-4000 result.
- E64 B300 status after about 37 minutes: parent Python process and two data
  worker children are still alive, CUDA process ownership is on PID `962`,
  GPU SM utilization remains active, and GPU memory is stable. No
  `results.json`, `results.csv`, checkpoint file, or step-4000 history row has
  been written yet. Treat as active slow training for now rather than the
  previous A100 stall. Heartbeat `check-simplexfold-e57-runpod` is active and
  scoped only to pod `ow3ex8z84jypbs`.
- E64 B300 completed on owned Runpod pod `ow3ex8z84jypbs`. Local returned
  artifacts were copied under ignored
  `artifacts/nanofold_public_benchmarks/e64_boundary_lddt005_from_e63_s4000_c256_m64/`,
  including checkpoints, `results.json`, `results.csv`, `history_full_msa_to_face.json`,
  `eval_details_full_msa_to_face.csv`, `run_metadata.json`, and the launch
  log. Step 4000 reached `val_lddt_ca=0.37385757453739643`, FoldScore
  `0.36337414756417274`, `val_ca_drmsd=10.548106044530869`, and
  predicted/true C-alpha radius of gyration `11.334399700164795 /
  15.403406739234924`.
- E64 selected-boundary diagnostics continued in the intended direction:
  face/tetra boundary lDDT `0.5358305890113115` / `0.5205421168357134`,
  contraction fractions `0.6698742806911469` / `0.6711904853582382`, and
  boundary length MAE `2.758177824318409` / `2.8986342176795006`.
- E64 owned Runpod pod `ow3ex8z84jypbs` was stopped and deleted after
  artifacts were copied. A post-delete lookup returned 404, as expected. No
  other Runpod instances were managed.
- E64 interpretation: keep and continue. The selected-boundary lDDT branch is
  now clearly above the E55/E63 band and improves FoldScore/dRMSD while
  preserving the selected-complex diagnostics. The next run should not raise
  the lDDT pressure blindly; instead, continue from E64 to step 5000, hold the
  selected-boundary lDDT weights at `0.05` through step 4500, then relax them
  to `0.025` over steps 4500-5000.
- E65 launch plan: resume
  `artifacts/nanofold_public_benchmarks/e64_boundary_lddt005_from_e63_s4000_c256_m64/checkpoints/full_msa_to_face_latest.pt`,
  continue `full_msa_to_face` to step 5000 with effective batch 8, keep
  selected face/tetra coordinate weights `1.0`, selected boundary
  coordinate-distance weights `0.5`, `simplex_aux_weight=0.5`, and schedule
  selected-boundary lDDT weights from `0.05` to `0.025` with
  `--simplex-boundary-lddt-ramp-start-step 4500` and
  `--simplex-boundary-lddt-ramp-steps 500`. Decision rule: if step 4500
  improves but step 5000 drops, next test static `0.05`; if both improve,
  continue the relaxed schedule; if both drop, return to architecture changes
  in selected-cell communication.
- E65 launched on owned Runpod B200 pod `21pml3y3hbbbpb`
  (`codex-simplexfold-e65-runpod-20260511`) from commit `d766050`. It uses
  `volumeInGb=0` and a 160 GB container disk so `/workspace` is local overlay
  storage. Only this owned pod should be managed for E65.
- E65 clean launch audit after copying only public data/code: public
  train/val/all manifest counts `10000/1000/11000`, remote manifest files
  exactly `all.txt`, `train.txt`, and `val.txt`, hidden manifest/path absent,
  feature/label `.npz` counts `11000/11000`, encoded-chain missing paths `0`,
  E64 checkpoint present, B200 CUDA available, NanoFold
  `foldscore_components` import works, AF2-medium pair-only `3,106,642`, and
  E65 model `3,106,690` parameters (`+0.0015%`). Schedule audit: face/tetra
  selected-boundary lDDT weights are `0.05` at steps 4000 and 4500,
  `0.0375` at step 4750, and `0.025` at step 5000.
- E65 remote process: wrapper PID `920`, Python PID `922`, data-worker Python
  PIDs `1085` and `1086`, launch log
  `/workspace/SimplexFold/logs/e65_boundary_lddt005_to0025_from_e64.log`,
  artifacts
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e65_boundary_lddt005_to0025_from_e64_s5000_c256_m64/`.
  Initial log shows the runner resumed E64 at step 4000/examples 32000,
  loaded 1196 matching model tensors, initialized 0 new/missing tensors, and
  started a fresh optimizer. CUDA was active under PID `922` with GPU memory
  allocated. Do not add E65 to `EXPERIMENT_RESULTS.md` until this Runpod run
  returns.
- E65 status check: Runpod still reports owned pod `21pml3y3hbbbpb` running.
  Python PID `922` is active with CUDA memory allocated. The E65 artifact
  directory currently contains `run_metadata.json` and
  `history_full_msa_to_face.json`; no `results.json`, `results.csv`,
  checkpoint, or step-4500/5000 row has returned yet. The history contains the
  inherited lineage through E64 step 4000.
- E66 idea recorded while E65 runs: coface-balanced selected-boundary lDDT
  using the existing `--simplex-boundary-degree-normalize` flag. This is a
  topological loss-weighting ablation, not a generic metric hack: selected
  faces/tetras still define the boundary 1-skeleton, but inverse incidence
  degree prevents a repeated undirected edge from dominating just because it
  appears in many selected cofaces. Do not launch E66 until E65 returns and
  the static-versus-relaxed boundary-lDDT question is resolved.
- E65 status check after launch: owned pod `21pml3y3hbbbpb` remains running,
  Python PID `922` is active, CUDA memory is allocated, and GPU utilization is
  nonzero. The E65 artifact directory still has no `results.json`,
  `results.csv`, checkpoint, or step-4500/5000 history row; latest history row
  remains the inherited E64 step 4000. Leave the pod running and continue to
  keep `EXPERIMENT_RESULTS.md` unchanged until a returned result exists.
- E66 readiness check while E65 continues: the runner already reports selected
  boundary-edge mean/max incidence degree and unique-edge fraction, which is
  the diagnostic needed for the coface-balanced boundary-lDDT ablation. Added
  focused test coverage that `--simplex-boundary-degree-normalize` is accepted
  by the benchmark CLI and propagated into the `SimplexGeometryLoss`. Local
  check passed:
  `python -m pytest tests/test_nanofold_public_benchmarks.py::test_topology_margin_args_are_accepted_by_cli_parser tests/test_nanofold_public_benchmarks.py::test_benchmark_loss_builder_applies_topology_margin_config tests/test_nanofold_public_benchmarks.py::test_simplex_topology_metrics_report_boundary_reuse`.
- E65 2026-05-11 19:47Z status check: owned Runpod pod `21pml3y3hbbbpb`
  remains healthy with Python PID `922` active, nonzero B200 utilization, and
  `13244 MiB` CUDA memory allocated. The run still has no `results.json`,
  `results.csv`, checkpoint, or new validation row; latest history remains the
  inherited E64 step 4000 with `val_lddt_ca=0.37385757453739643`. Continue to
  leave `EXPERIMENT_RESULTS.md` unchanged until E65 returns.
- E65 returned on owned Runpod pod `21pml3y3hbbbpb` and was rejected. Step
  4500 reached `val_lddt_ca=0.36452375911176205`, FoldScore
  `0.36604692228138447`, `val_ca_drmsd=10.271232694387436`, and predicted/true
  C-alpha radius `12.000848263502121 / 15.403406739234924`. Step 5000 reached
  `val_lddt_ca=0.368440305814147`, FoldScore `0.3666128497570753`,
  `val_ca_drmsd=10.844526827335358`, and predicted/true C-alpha radius
  `11.787870109081268 / 15.403406739234924`. Selected face/tetra boundary
  lDDT ended at `0.5296795684844255` / `0.514076316729188`, contraction
  fractions at `0.6505701839923859` / `0.6532666180282831`, boundary length
  MAE at `2.8452821373939514` / `2.998050607740879`, mean boundary-edge degree
  at `11.461855471134186` / `76.41236972808838`, and unique-edge fraction at
  `0.08832908659020682` / `0.013249362988531022`.
- E65 artifacts and launch log were copied locally under ignored
  `artifacts/nanofold_public_benchmarks/e65_boundary_lddt005_to0025_from_e64_s5000_c256_m64/`,
  including `results.json`, `results.csv`, `history_full_msa_to_face.json`,
  `eval_details_full_msa_to_face.csv`, `run_metadata.json`, checkpoint, and
  `runpod_launch.log`. The owned E65 pod was stopped and deleted after copying;
  a post-delete lookup returned 404. No other Runpod instances were managed.
- E65 interpretation: reject the `0.05 -> 0.025` selected-boundary lDDT
  schedule. Both returned lDDT points are below E64's `0.37385757453739643`,
  even though FoldScore slightly improves. Next launch E66 from E64 for a
  500-step coface-balanced selected-boundary lDDT gate with static `0.05`
  weights and `--simplex-boundary-degree-normalize`.
- E66 launched on owned Runpod B200 pod `xlvkre8ww4utac`
  (`codex-simplexfold-e66-runpod-20260511`) from commit `c2dce57`, SSH
  `root@38.80.152.146 -p 31156` with
  `/Users/christopherhayduk/.runpod/ssh/RunPod-Key-Go`. The run name is
  `e66_coface_balanced_boundary_lddt005_from_e64_s4500_c256_m64`, log path
  `/workspace/SimplexFold/logs/e66_coface_balanced_boundary_lddt005_from_e64.log`,
  and artifact path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e66_coface_balanced_boundary_lddt005_from_e64_s4500_c256_m64/`.
  Only this owned E66 pod should be managed.
- E66 clean launch audit after copying only public data/code: public
  train/val/all manifest counts `10000/1000/11000`, remote manifest files
  exactly `all.txt`, `train.txt`, and `val.txt`, hidden
  manifest/features/labels absent, feature/label `.npz` counts
  `11000/11000`, E64 checkpoint present, B200 CUDA available, NanoFold
  `foldscore_components` import works, AF2-medium pair-only `3,106,642`, and
  E66 model `3,106,690` parameters (`+0.0015%`). `run_metadata.json` records
  `simplex_boundary_degree_normalize=true`, static selected-boundary lDDT
  weights `0.05` / `0.05`, weights-only resume from E64, crop 256, MSA depth
  64, and no templates.
- E66 remote process after launch: wrapper PID `10344`, Python PID `10346`,
  data-worker Python PIDs `13275` and `13276`. Initial log shows the runner
  resumed E64 at step 4000/examples 32000, loaded 1196 matching model tensors,
  initialized 0 new/missing tensors, and started a fresh optimizer. CUDA is
  active with B200 memory allocated. Heartbeat `check-simplexfold-e57-runpod`
  has been retargeted to E66 pod `xlvkre8ww4utac` only.
- E66 returned on owned Runpod pod `xlvkre8ww4utac` and was rejected. Step
  4500 reached `val_lddt_ca=0.35054648853838444`, FoldScore
  `0.3602386135607958`, `val_ca_drmsd=10.623665452003479`, and predicted/true
  C-alpha radius `11.889198303222656 / 15.40340667963028`. Selected
  face/tetra boundary lDDT ended at `0.5090303029865026` /
  `0.49476016499102116`, contraction fractions at `0.7145102694630623` /
  `0.7135652974247932`, boundary length MAE at `2.8950554355978966` /
  `3.032222591340542`, mean boundary-edge degree at `11.601061284542084` /
  `77.34040784835815`, and unique-edge fraction at `0.08682302490902588` /
  `0.013023453736353881`.
- E66 artifacts and launch log were copied locally under ignored
  `artifacts/nanofold_public_benchmarks/e66_coface_balanced_boundary_lddt005_from_e64_s4500_c256_m64/`,
  including `results.json`, `results.csv`, `history_full_msa_to_face.json`,
  `eval_details_full_msa_to_face.csv`, `run_metadata.json`, checkpoint, and
  `runpod_launch.log`. The owned E66 pod was stopped and deleted after copying;
  a post-delete lookup returned 404. No other Runpod instances were managed.
- E66 interpretation: reject coface-balanced selected-boundary lDDT. It
  lowered main lDDT below E65's step-4500 unbalanced continuation and also
  weakened the topology diagnostics it was meant to clean up. Next test E67:
  weak selected-complex structure readout from the E64 checkpoint using
  `--simplex-structure-readout-scale 0.05` with the E64 selected-boundary
  lDDT/coordinate-loss recipe.
- E67 launched on owned Runpod B200 pod `3en5noqmkkiovz`
  (`codex-simplexfold-e67-runpod-20260511`) from commit `27ddea4`, SSH
  `root@198.13.252.84 -p 55444` with
  `/Users/christopherhayduk/.runpod/ssh/RunPod-Key-Go`. The run name is
  `e67_structure_readout005_from_e64_s4500_c256_m64`, log path
  `/workspace/SimplexFold/logs/e67_structure_readout005_from_e64.log`, and
  artifact path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e67_structure_readout005_from_e64_s4500_c256_m64/`.
  Only this owned E67 pod should be managed.
- E67 clean launch audit after copying only public data/code: public
  train/val/all manifest counts `10000/1000/11000`, remote manifest files
  exactly `all.txt`, `train.txt`, and `val.txt`, hidden
  manifest/features/labels absent, feature/label `.npz` counts
  `11000/11000`, E64 checkpoint present, B200 CUDA available, NanoFold
  `foldscore_components` import works, AF2-medium pair-only `3,106,642`, and
  E67 model `3,106,690` parameters (`+0.0015%`). `run_metadata.json` records
  `simplex_structure_readout_scale=0.05`, static selected-boundary lDDT
  weights `0.05` / `0.05`, weights-only resume from E64, crop 256, MSA depth
  64, and no templates.
- E67 remote process after launch: Python PID `518`, data-worker Python PIDs
  `679` and `680`. Initial log shows the runner resumed E64 at step
  4000/examples 32000, loaded 1196 matching model tensors, initialized 0
  new/missing tensors, and started a fresh optimizer. CUDA is active with B200
  memory allocated. Heartbeat `check-simplexfold-e57-runpod` has been
  retargeted to E67 pod `3en5noqmkkiovz` only.
- E67 returned on owned Runpod pod `3en5noqmkkiovz` and was rejected as a
  continuation branch. Step 4500 reached `val_lddt_ca=0.364693870767951`,
  FoldScore `0.36185639910399914`, `val_ca_drmsd=10.350328981876373`, and
  predicted/true C-alpha radius `11.668777972459793 / 15.403406739234924`.
  Selected face/tetra boundary lDDT ended at `0.530171524733305` /
  `0.5154275149106979`, contraction fractions at `0.6802140511572361` /
  `0.6808506213128567`, boundary length MAE at `2.683264270424843` /
  `2.8166864067316055`, mean boundary-edge degree at `11.859311401844025` /
  `79.06207513809204`, and unique-edge fraction at `0.08480586282465244` /
  `0.012720879423697866`.
- E67 artifacts and launch log were copied locally under ignored
  `artifacts/nanofold_public_benchmarks/e67_structure_readout005_from_e64_s4500_c256_m64/`,
  including `results.json`, `results.csv`, `history_full_msa_to_face.json`,
  `eval_details_full_msa_to_face.csv`, `run_metadata.json`, checkpoint, and
  `runpod_launch.log`. The owned E67 pod was stopped and deleted after copying;
  a post-delete lookup returned 404. No other Runpod instances were managed.
  Heartbeat `check-simplexfold-e57-runpod` was paused because no Runpod job is
  currently active.
- E67 interpretation: reject scale `0.05` for continuation because it remains
  below E64 and loses FoldScore, but keep the geometry lesson. The readout
  path improved dRMSD and selected-boundary length MAE, so next test E68 with
  a damped selected-complex structure readout scale `0.025` from E64.
- E68 launched on owned Runpod B200 pod `qx6oa0jgchz8j8`
  (`codex-simplexfold-e68-runpod-20260511`) from commit `11fc14a`, SSH
  `root@198.13.252.84 -p 35716` with
  `/Users/christopherhayduk/.runpod/ssh/RunPod-Key-Go`. The run name is
  `e68_structure_readout0025_from_e64_s4500_c256_m64`, log path
  `/workspace/SimplexFold/logs/e68_structure_readout0025_from_e64.log`, and
  artifact path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e68_structure_readout0025_from_e64_s4500_c256_m64/`.
  Only this owned E68 pod should be managed.
- E68 clean launch audit after copying only public data/code: public
  train/val/all manifest counts `10000/1000/11000`, remote manifest files
  exactly `all.txt`, `train.txt`, and `val.txt`, hidden
  manifest/features/labels absent, feature/label `.npz` counts
  `11000/11000`, E64 checkpoint present, B200 CUDA available, NanoFold
  `foldscore_components` import works, AF2-medium pair-only `3,106,642`, and
  E68 model `3,106,690` parameters (`+0.0015%`). `run_metadata.json` records
  `simplex_structure_readout_scale=0.025`, static selected-boundary lDDT
  weights `0.05` / `0.05`, weights-only resume from E64, crop 256, MSA depth
  64, and no templates.
- E68 remote process after launch: Python PID `479`, data-worker Python PIDs
  `640` and `641`. Initial log shows the runner resumed E64 at step
  4000/examples 32000, loaded 1196 matching model tensors, initialized 0
  new/missing tensors, and started a fresh optimizer. CUDA is active with B200
  memory allocated. Heartbeat `check-simplexfold-e57-runpod` has been
  retargeted to E68 pod `qx6oa0jgchz8j8` only.
- E68 returned on owned Runpod pod `qx6oa0jgchz8j8` and was rejected. Step
  4500 reached `val_lddt_ca=0.3617166429758072`, FoldScore
  `0.36250696517527103`, `val_ca_drmsd=10.211460411548615`, and predicted/true
  C-alpha radius `11.964475572109222 / 15.403406739234924`. Selected
  face/tetra boundary lDDT ended at `0.524696733802557` /
  `0.5102544575929642`, contraction fractions at `0.6823383867740631` /
  `0.6825277544558048`, boundary length MAE at `2.7149862870573997` /
  `2.8477588519454002`, mean boundary-edge degree at
  `12.109066903591156` / `80.72711277008057`, and unique-edge fraction at
  `0.08305363674494468` / `0.0124580455117417`.
- E68 artifacts and launch log were copied locally under ignored
  `artifacts/nanofold_public_benchmarks/e68_structure_readout0025_from_e64_s4500_c256_m64/`,
  including `results.json`, `results.csv`, `history_full_msa_to_face.json`,
  `eval_details_full_msa_to_face.csv`, `run_metadata.json`, checkpoint, and
  `runpod_launch.log`. The owned E68 pod was stopped and deleted after copying;
  a post-delete lookup returned 404. No other Runpod instances were managed.
  Heartbeat `check-simplexfold-e57-runpod` was paused because no Runpod job is
  currently active.
- E68 interpretation: reject the damped selected-complex structure readout.
  Lowering readout scale from `0.05` to `0.025` improved dRMSD to the best
  value in the E64 continuation family but further reduced C-alpha lDDT. Next
  test E69 from the E64 checkpoint with selected face normal orientation:
  `--simplex-face-normal-weight 0.05` plus the E64 selected-boundary
  lDDT/coordinate-loss recipe. This is justified by the README's oriented
  patch claim for faces and the PDF reread rule that losses should supervise
  realization of the selected sparse complex rather than generic dense output
  geometry.
- E69 launched on owned Runpod B200 pod `eznq63h3uorbrf`
  (`codex-simplexfold-e69-runpod-20260511`) from commit `34a2796`, SSH
  `root@38.80.152.146 -p 31414` with
  `/Users/christopherhayduk/.runpod/ssh/RunPod-Key-Go`. The run name is
  `e69_face_normal005_from_e64_s4500_c256_m64`, log path
  `/workspace/SimplexFold/logs/e69_face_normal005_from_e64.log`, and artifact
  path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e69_face_normal005_from_e64_s4500_c256_m64/`.
  Only this owned E69 pod should be managed.
- E69 clean launch audit after copying only public data/code: public
  train/val/all manifest counts `10000/1000/11000`, remote manifest files
  exactly `all.txt`, `train.txt`, and `val.txt`, hidden
  manifest/features/labels absent, feature/label `.npz` counts
  `11000/11000`, E64 checkpoint present, B200 CUDA available, NanoFold
  `foldscore_components` import works, AF2-medium pair-only `3,106,642`, and
  E69 model `3,106,690` parameters (`+0.0015%`). `run_metadata.json` records
  `simplex_face_normal_weight=0.05`, static selected-boundary lDDT weights
  `0.05` / `0.05`, weights-only resume from E64, crop 256, MSA depth 64, and
  no templates. An interrupted metadata-bearing feature tar left 285
  AppleDouble sidecar `.npz` files; those sidecars were deleted before launch,
  and the final remote feature/label counts were `11000` / `11000`.
- E69 remote process after launch: Python PID `8307`, data-worker Python PIDs
  `11234` and `11235`. Initial log shows the runner resumed E64 at step
  4000/examples 32000, loaded 1196 matching model tensors, initialized 0
  new/missing tensors, and started a fresh optimizer. CUDA is active on the
  B200 with GPU utilization and memory allocated. Heartbeat
  `check-simplexfold-e57-runpod` has been retargeted to E69 pod
  `eznq63h3uorbrf` only.
- E69 returned on owned Runpod pod `eznq63h3uorbrf` and was rejected. Step
  4500 reached `val_lddt_ca=0.3652884252369404`, FoldScore
  `0.36315777339041233`, `val_ca_drmsd=10.583343207836151`, and
  predicted/true C-alpha radius `11.875020861625671 / 15.40340667963028`.
  Selected face/tetra boundary lDDT ended at `0.521033963188529` /
  `0.5058649443089962`, contraction fractions at `0.6824001856148243` /
  `0.6835838556289673`, boundary length MAE at `2.859107196331024` /
  `3.0093840062618256`, mean boundary-edge degree at `11.976542711257935` /
  `79.8436188697815`, and unique-edge fraction at `0.08402037883430959` /
  `0.012603056825146437`. The selected face-normal term was active with
  `val_weighted_simplex_face_normal_loss=0.017673333699349314`.
- E69 artifacts and launch log were copied locally under ignored
  `artifacts/nanofold_public_benchmarks/e69_face_normal005_from_e64_s4500_c256_m64/`,
  including `results.json`, `results.csv`, `history_full_msa_to_face.json`,
  `eval_details_full_msa_to_face.csv`, `run_metadata.json`, checkpoint, and
  `runpod_launch.log`. The owned E69 pod was stopped and deleted after copying;
  a post-delete lookup returned 404. No other Runpod instances were managed.
  Heartbeat `check-simplexfold-e57-runpod` was paused because no Runpod job is
  currently active.
- E69 interpretation: reject selected face-normal orientation as an auxiliary
  continuation from E64. It is topologically justified, but in practice it
  pulled the selected-boundary diagnostics down and did not preserve E64's
  lDDT. Next test E70 from the E64 checkpoint with damped edge-frame boundary
  messages: enable `--simplex-edge-frame-message-scale 0.025` and ramp
  runtime contribution from `0.0` to `0.025` over steps 4000-4500 while
  keeping the E64 selected-boundary lDDT/coordinate-loss recipe. This moves
  orientation-aware higher-rank information through selected boundary-edge
  frames instead of attaching another standalone orientation loss.
- E70 launched on owned Runpod B200 pod `lovgzo4hz2k4fp`
  (`codex-simplexfold-e70-runpod-20260512`) from commit `bf7de3d`, SSH
  `root@38.80.152.146 -p 31403` with
  `/Users/christopherhayduk/.runpod/ssh/RunPod-Key-Go`. The run name is
  `e70_edge_frame0025_from_e64_s4500_c256_m64`, log path
  `/workspace/SimplexFold/logs/e70_edge_frame0025_from_e64.log`, and artifact
  path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e70_edge_frame0025_from_e64_s4500_c256_m64/`.
  Only this owned E70 pod should be managed.
- E70 clean launch audit after copying only public data/code: public
  train/val/all manifest counts `10000/1000/11000`, remote manifest files
  exactly `all.txt`, `train.txt`, and `val.txt`, hidden
  manifest/features/labels absent, feature/label `.npz` counts
  `11000/11000`, no AppleDouble sidecar feature files, E64 checkpoint present,
  B200 CUDA available, NanoFold `foldscore_components` import works,
  AF2-medium pair-only `3,106,642`, and E70 edge-frame model `3,154,242`
  parameters (`+1.53%`). `run_metadata.json` records
  `simplex_edge_frame_message_scale=0.025`, runtime edge-frame scale
  `0.0 -> 0.025`, ramp start step `4000`, ramp steps `500`, weights-only
  resume from E64, crop 256, MSA depth 64, and no templates.
- E70 remote process after launch: Python PID `4346`, data-worker Python PIDs
  `7273` and `7274`. Initial log shows the runner resumed E64 at step
  4000/examples 32000, loaded 1196 matching model tensors, initialized 48
  new/missing edge-frame tensors, and started a fresh optimizer. CUDA is active
  on the B200 with GPU utilization and memory allocated. Heartbeat
  `check-simplexfold-e57-runpod` has been retargeted to E70 pod
  `lovgzo4hz2k4fp` only.
- E70 returned on owned Runpod pod `lovgzo4hz2k4fp` and is kept for a
  stability continuation. Step 4500 reached `val_lddt_ca=0.37417401000857353`,
  FoldScore `0.3653021454811096`, `val_ca_drmsd=10.342539131641388`, and
  predicted/true C-alpha radius `11.481503546237946 / 15.40340667963028`.
  Selected face/tetra boundary lDDT ended at `0.5364893395453691` /
  `0.5214813183993101`, contraction fractions at `0.6665069051086903` /
  `0.6680875010788441`, boundary length MAE at `2.6312515065073967` /
  `2.7605674117803574`, mean boundary-edge degree at `11.896117627620697` /
  `79.30745029449463`, and unique-edge fraction at `0.08471935021707062` /
  `0.012707902532560593`. Runtime edge-frame scale reached `0.025`.
- E70 artifacts and launch log were copied locally under ignored
  `artifacts/nanofold_public_benchmarks/e70_edge_frame0025_from_e64_s4500_c256_m64/`,
  including `results.json`, `results.csv`, `history_full_msa_to_face.json`,
  `eval_details_full_msa_to_face.csv`, `run_metadata.json`, checkpoint, and
  `runpod_launch.log`. The owned E70 pod was intentionally left running for
  the E71 continuation; no other Runpod instances were managed.
- E70 interpretation: keep cautiously. The lDDT gain over E64 is tiny, but it
  is aligned with the selected-boundary diagnostics, unlike E67-E69. Next run
  E71 on the same owned pod from the E70 checkpoint to step 5000, holding
  runtime edge-frame scale at `0.025` and keeping the E64 selected-boundary
  lDDT/coordinate-loss recipe unchanged.
- E71 launched on the same owned Runpod B200 pod `lovgzo4hz2k4fp`
  (`codex-simplexfold-e70-runpod-20260512`) from commit `e201086`, SSH
  `root@38.80.152.146 -p 31403` with
  `/Users/christopherhayduk/.runpod/ssh/RunPod-Key-Go`. The run name is
  `e71_edge_frame0025_from_e70_s5000_c256_m64`, log path
  `/workspace/SimplexFold/logs/e71_edge_frame0025_from_e70.log`, and artifact
  path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e71_edge_frame0025_from_e70_s5000_c256_m64/`.
  Only this owned pod should be managed.
- E71 remote process after launch: Python PID `7593`, data-worker Python PIDs
  `10520` and `10521`. Initial log shows the runner resumed E70 at step
  4500/examples 36000, loaded 1244 matching model tensors, initialized 0
  new/missing tensors, and started a fresh optimizer. `run_metadata.json`
  records `simplex_edge_frame_message_scale=0.025`, runtime edge-frame scale
  `0.025`, weights-only resume from the E70 checkpoint, crop 256, MSA depth 64,
  and no templates. Heartbeat `check-simplexfold-e57-runpod` has been
  retargeted to E71 on pod `lovgzo4hz2k4fp` only.
- E71 returned on owned Runpod pod `lovgzo4hz2k4fp` and is kept for another
  continuation. Step 5000 reached `val_lddt_ca=0.37505385652184486`,
  FoldScore `0.3678866345435381`, `val_ca_drmsd=10.19257652759552`, and
  predicted/true C-alpha radius `11.448325783014297 / 15.40340667963028`.
  Selected face/tetra boundary lDDT ended at `0.5335890911519527` /
  `0.5181191172450781`, contraction fractions at `0.6445319689810276` /
  `0.6458184905350208`, boundary length MAE at `2.791555091738701` /
  `2.9308799281716347`, mean boundary-edge degree at `11.321606159210205` /
  `75.47737550735474`, and unique-edge fraction at `0.08870875050350643` /
  `0.013306312575525964`. Runtime edge-frame scale remained `0.025`.
- E71 artifacts and launch log were copied locally under ignored
  `artifacts/nanofold_public_benchmarks/e71_edge_frame0025_from_e70_s5000_c256_m64/`,
  including `results.json`, `results.csv`, `history_full_msa_to_face.json`,
  `eval_details_full_msa_to_face.csv`, `run_metadata.json`, checkpoint, and
  `runpod_launch.log`. The owned pod remains active for E72; no other Runpod
  instances were managed.
- E71 interpretation: keep cautiously. The branch improves lDDT, FoldScore,
  and dRMSD together, but selected-boundary lDDT softened relative to E70 while
  contraction improved. Launch E72 to step 5500 on the same pod, holding the
  runtime edge-frame scale at `0.025`; stop this branch if selected-boundary
  lDDT keeps eroding or if main lDDT drops back toward the E65/E67/E69 band.
- E72 launched on the same owned Runpod B200 pod `lovgzo4hz2k4fp`
  (`codex-simplexfold-e70-runpod-20260512`) from commit `1d27dd9`, SSH
  `root@38.80.152.146 -p 31403` with
  `/Users/christopherhayduk/.runpod/ssh/RunPod-Key-Go`. The run name is
  `e72_edge_frame0025_from_e71_s5500_c256_m64`, log path
  `/workspace/SimplexFold/logs/e72_edge_frame0025_from_e71.log`, and artifact
  path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e72_edge_frame0025_from_e71_s5500_c256_m64/`.
  Only this owned pod should be managed.
- E72 remote process after launch: Python PID `10825`, data-worker Python PIDs
  `13752` and `13753`. Initial log shows the runner resumed E71 at step
  5000/examples 40000, loaded 1244 matching model tensors, initialized 0
  new/missing tensors, and started a fresh optimizer. `run_metadata.json`
  records `simplex_edge_frame_message_scale=0.025`, runtime edge-frame scale
  `0.025`, weights-only resume from the E71 checkpoint, crop 256, MSA depth 64,
  and no templates. Heartbeat `check-simplexfold-e57-runpod` has been
  retargeted to E72 on pod `lovgzo4hz2k4fp` only.
- E72 returned on owned Runpod pod `lovgzo4hz2k4fp` and is rejected as a
  primary-lDDT continuation. Step 5500 reached
  `val_lddt_ca=0.37177472934126854`, below E71's
  `0.37505385652184486`, while FoldScore improved to
  `0.3721824511885643`, `val_ca_drmsd` improved to
  `10.102694064378738`, and predicted/true C-alpha radius opened to
  `12.087152183055878 / 15.40340667963028`.
- E72 selected-complex diagnostics improved despite the main lDDT regression:
  selected face/tetra boundary lDDT reached `0.5449902731925249` /
  `0.530314240604639`, contraction fractions fell to
  `0.6554896421730518` / `0.6555038057267666`, boundary length MAE ended at
  `2.6295931339263916` / `2.7580906972289085`, mean boundary-edge degree at
  `12.054364442825317` / `80.36243009567261`, and unique-edge fraction at
  `0.08331795553389273` / `0.01249769333008391`.
- E72 artifacts and launch log were copied locally under ignored
  `artifacts/nanofold_public_benchmarks/e72_edge_frame0025_from_e71_s5500_c256_m64/`,
  including `results.json`, `results.csv`, `history_full_msa_to_face.json`,
  `eval_details_full_msa_to_face.csv`, `run_metadata.json`, checkpoint, and
  `runpod_launch.log`. The owned pod remains active for E73; no other Runpod
  instances were managed.
- E72 interpretation: the selected boundary-edge frame route is topologically
  useful but too strong at runtime scale `0.025` after step 5000. Launch E73
  from the E71 checkpoint to step 5500 with the same allocated edge-frame
  modules but runtime scale `0.0125`; keep the E64 selected-boundary
  lDDT/coordinate-loss recipe unchanged.
- E73 launched on the same owned Runpod B200 pod `lovgzo4hz2k4fp`
  (`codex-simplexfold-e70-runpod-20260512`) from commit `bc1b749`, SSH
  `root@38.80.152.146 -p 31403` with
  `/Users/christopherhayduk/.runpod/ssh/RunPod-Key-Go`. The run name is
  `e73_edge_frame00125_from_e71_s5500_c256_m64`, log path
  `/workspace/SimplexFold/logs/e73_edge_frame00125_from_e71.log`, and artifact
  path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e73_edge_frame00125_from_e71_s5500_c256_m64/`.
  Only this owned pod should be managed.
- E73 remote process after launch: Python PID `14028`, data-worker Python PIDs
  `16955` and `16956`. Initial log shows the runner resumed E71 at step
  5000/examples 40000, loaded 1244 matching model tensors, initialized 0
  new/missing tensors, and started a fresh optimizer. `run_metadata.json`
  records `simplex_edge_frame_message_scale=0.025`, runtime edge-frame scale
  `0.0125`, weights-only resume from the E71 checkpoint, crop 256, MSA depth
  64, and no templates. Heartbeat `check-simplexfold-e57-runpod` has been
  retargeted to E73 on pod `lovgzo4hz2k4fp` only.
- E74 local implementation prepared while E73 runs: added
  `--simplex-geometry-distance-weight` to
  `scripts/run_nanofold_public_benchmarks.py`, threaded it through
  `_apply_model_config_overrides`, and recorded `simplex_geometry_distance_weight`
  in result JSON/CSV. This lets the next Runpod branch reduce the recycled
  C-alpha distance prior in `build_simplex_topology` from `0.1` to `0.025`
  without editing config files or changing parameter count.
- E74 local validation passed:
  `python -m pytest tests/test_nanofold_public_benchmarks.py::test_model_config_override_flags_are_accepted_by_cli_parser tests/test_trainer.py::test_simplicial_geometry_selector_weight_adds_no_parameters`.
  This branch should launch only if E73 does not preserve E71's local lDDT;
  update the owned Runpod workspace to the new commit first, because the
  currently running E73 process is still on commit `bc1b749`.
- Runtime-override correction: while preparing E74 I found that
  `scripts/run_nanofold_public_benchmarks.py` trained with runtime simplex
  overrides but `_evaluate` did not pass those overrides, so the initial E73
  process would validate at the static model edge-frame scale rather than the
  intended runtime scale `0.0125`. Fixed `_evaluate` to pass runtime simplex
  overrides, added a scheduled geometry selector override through
  model/evoformer/adapter/trainer plumbing, and expanded focused tests.
- Updated local validation passed:
  `python -m pytest tests/test_nanofold_public_benchmarks.py::test_model_config_override_flags_are_accepted_by_cli_parser tests/test_nanofold_public_benchmarks.py::test_runtime_simplex_message_scales_ramp_and_enter_model_inputs tests/test_nanofold_public_benchmarks.py::test_evaluate_uses_runtime_simplex_overrides_for_validation tests/test_simplex.py::test_build_simplex_topology_geometry_weight_changes_selected_neighbors tests/test_trainer.py::test_simplicial_geometry_selector_weight_adds_no_parameters`;
  `python -m py_compile minalphafold/simplex.py minalphafold/evoformer.py minalphafold/model.py minalphafold/trainer.py scripts/run_nanofold_public_benchmarks.py`;
  `git diff --check`.
- Stop the first E73 process on owned pod `lovgzo4hz2k4fp` and relaunch from
  the fixed commit before recording any E73 result. The first process has no
  returned result yet and should not be added to `EXPERIMENT_RESULTS.md`.
- 2026-05-11 late-session PDF recheck: verified both user-provided PDFs are
  still present in `references/papers/` and hash-match the Downloads copies,
  then re-read the local `pdftotext -layout` extraction. No new generic loss
  direction was promoted. The full-text pass reinforces the current E73/E74
  path: runtime-gated boundary-edge messages and a changed recycled-geometry
  selector prior are topological interventions because they affect the
  selected edge/face/tetra cochains and their incidence-mediated
  communication.
- Fixed E73 relaunch on owned pod `lovgzo4hz2k4fp`: remote py_compile passed
  for `minalphafold/simplex.py`, `minalphafold/evoformer.py`,
  `minalphafold/model.py`, `minalphafold/trainer.py`, and
  `scripts/run_nanofold_public_benchmarks.py`; parser smoke confirmed
  edge-frame runtime scale `0.0125` and geometry selector schedule
  `0.1 -> 0.025`; no benchmark process was active and the B200 was idle before
  launch.
- E73 evalfix launch: run name
  `e73_evalfix_edge_frame00125_from_e71_s5500_c256_m64`, log path
  `/workspace/SimplexFold/logs/e73_evalfix_edge_frame00125_from_e71.log`, and
  artifact path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e73_evalfix_edge_frame00125_from_e71_s5500_c256_m64/`.
  Main Python PID is `17416`; data-worker PIDs are `20343` and `20344`. The
  run resumed E71 at step 5000/examples 40000, loaded 1244 matching model
  tensors, initialized 0 new/missing tensors, and `run_metadata.json` records
  `simplex_edge_frame_message_scale=0.025`,
  `simplex_edge_frame_message_runtime_scale=0.0125`, weights-only resume,
  crop 256, MSA depth 64, and no templates. Heartbeat
  `check-simplexfold-e57-runpod` has been retargeted to the E73 evalfix run
  on pod `lovgzo4hz2k4fp` only.
- E75 local implementation prepared while E73 evalfix runs: added
  zero-parameter `simplex_face_top_k` and `simplex_tetra_top_k` selector caps.
  The topology builder still enumerates candidate faces/tetras inside each
  selected neighbor star, but now can keep only the top-scoring cells per
  anchor by mean selected boundary-edge logit. This is a combinatorial-complex
  selector change, not another coordinate loss: inactive higher-rank cells no
  longer send messages or contribute selected-cell losses while tensor shapes
  remain checkpoint-compatible.
- E75 local validation passed:
  `python -m pytest tests/test_simplex.py::test_build_simplex_topology_cell_topk_caps_active_higher_rank_cells tests/test_nanofold_public_benchmarks.py::test_model_config_override_flags_are_accepted_by_cli_parser tests/test_trainer.py::test_simplicial_cell_topk_selector_adds_no_parameters`;
  `python -m py_compile minalphafold/simplex.py minalphafold/model_config.py scripts/run_nanofold_public_benchmarks.py`.
  Launch only after E73/E74 decision, with an initial cap such as
  `--simplex-face-top-k 24 --simplex-tetra-top-k 48`.
- Synced the E75 source/docs/tests to the owned Runpod workspace
  `/workspace/SimplexFold` while leaving logs/artifacts untouched. Remote
  py_compile passed for `minalphafold/simplex.py`,
  `minalphafold/model_config.py`, and
  `scripts/run_nanofold_public_benchmarks.py`; parser smoke confirmed
  `--simplex-face-top-k 24` and `--simplex-tetra-top-k 48`. E73 evalfix is
  still the only active benchmark process on pod `lovgzo4hz2k4fp`.
- Added `scripts/format_experiment_result_row.py` to format returned
  `results.json` plus optional history into an `EXPERIMENT_RESULTS.md` table
  row. Use `--start-after-step` for resumed continuations so inherited
  checkpoint history, such as E71's step-5000 row inside E73/E74 histories, is
  not counted as the continuation's best validation lDDT. Local tests passed:
  `python -m pytest tests/test_format_experiment_result_row.py`;
  `python -m py_compile scripts/format_experiment_result_row.py`;
  `git diff --check`.
- Synced the result-row formatter to the owned Runpod workspace and verified
  it on the existing E72 artifacts with `--start-after-step 5000`. Cleaned up
  two stray root-level helper files from the first sync attempt; the canonical
  remote paths are now `scripts/format_experiment_result_row.py` and
  `tests/test_format_experiment_result_row.py`.
- E73 evalfix returned on owned pod `lovgzo4hz2k4fp` with a new best:
  step 5500 `val_lddt_ca=0.3807`, FoldScore `0.3720`,
  `val_ca_drmsd=10.0777`, and predicted/true C-alpha radius
  `11.6741 / 15.4034`. It completed with `3,154,242` parameters
  (`+1.53%` versus the AF2-medium baseline), effective batch 8, no templates,
  and runtime edge-frame scale `0.0125`.
- E73 interpretation: keep. Half-scale selected boundary-edge frame messages
  recovered and improved E71's local lDDT while also improving FoldScore and
  dRMSD. It did not fully retain E72's selected-boundary realization gains:
  selected face/tetra boundary lDDT ended at `0.5368` / `0.5213`, boundary
  length MAE at `2.7292` / `2.8698`, and contraction fractions at
  `0.6669` / `0.6692`.
- Copied E73 returned artifacts locally under ignored
  `artifacts/nanofold_public_benchmarks/e73_evalfix_edge_frame00125_from_e71_s5500_c256_m64/`
  and copied the launch log to ignored
  `logs/e73_evalfix_edge_frame00125_from_e71.log`. The local artifact pull
  intentionally excluded the checkpoint directory; the remote checkpoint stays
  available for the next Runpod continuation.
- Used `scripts/format_experiment_result_row.py` with `--start-after-step 5000`
  to add the E73 row to `EXPERIMENT_RESULTS.md`, so the inherited E71 rows in
  E73 history do not count as E73's best.
- Next branch is E76, not E74: continue the E73 checkpoint to step 6000 with
  the same half-scale selected boundary-edge message recipe. Launch E74/E75
  only if E76 turns over or selected-complex diagnostics collapse.
- E76 launched on the same owned Runpod B200 pod `lovgzo4hz2k4fp` with run
  name `e76_edge_frame00125_from_e73_s6000_c256_m64`, log path
  `/workspace/SimplexFold/logs/e76_edge_frame00125_from_e73.log`, and
  artifact path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e76_edge_frame00125_from_e73_s6000_c256_m64/`.
  The launch command found no active benchmark process before starting. Main
  Python PID is `20888`; data-worker PIDs are `23815` and `23816`.
  The runner resumed E73 at step 5500/examples 44000, loaded 1244 matching
  model tensors, initialized 0 new/missing tensors, and started a fresh
  optimizer. GPU activity was confirmed after launch.
- E77 local implementation prepared while E76 runs: added
  zero-parameter `simplex_boundary_message_degree_attenuation`. After selected
  face/tetra boundary messages are scattered and averaged into the pair tensor,
  the pair readout can now be damped by `coface_degree ** attenuation`. This
  changes the selected-complex incidence/message operator and directly targets
  the high boundary-edge reuse diagnostic without adding an output-only metric
  loss or changing parameter count. Defaults preserve current behavior.
- E77 local validation passed:
  `python -m pytest tests/test_simplex.py::test_coface_degree_attenuation_damps_reused_boundary_edges tests/test_simplex.py::test_boundary_message_degree_attenuation_gates_pair_readout_without_single_change tests/test_nanofold_public_benchmarks.py::test_model_config_override_flags_are_accepted_by_cli_parser tests/test_trainer.py::test_simplicial_boundary_message_degree_attenuation_adds_no_parameters`;
  `python -m py_compile minalphafold/simplex.py minalphafold/model_config.py scripts/run_nanofold_public_benchmarks.py`.
  Do not sync this code into the active remote workspace until E76 returns.
- E76 returned on the owned Runpod pod `lovgzo4hz2k4fp`: step 6000
  `val_lddt_ca=0.3712605331093073`, FoldScore `0.37234109826385975`,
  `val_ca_drmsd=10.219059228897095`, predicted/true C-alpha radius
  `12.003583312034607 / 15.40340667963028`, selected face/tetra boundary
  lDDT `0.5340694449841976` / `0.518533306196332`, boundary length MAE
  `2.8649201095104218` / `3.012835741043091`, and contraction fractions
  `0.6487747132778168` / `0.6501699537038803`. Reject E76: the tiny
  FoldScore improvement over E73 does not compensate for losing the primary
  C-alpha lDDT peak and softening selected-boundary diagnostics.
- Copied E76 returned artifacts locally under ignored
  `artifacts/nanofold_public_benchmarks/e76_edge_frame00125_from_e73_s6000_c256_m64/`
  and copied the launch log to ignored
  `logs/e76_edge_frame00125_from_e73.log`. The local artifact pull excluded
  the checkpoint directory; the remote E76 checkpoint remains available but
  E77 should start from the stronger E73 checkpoint.
- Used `scripts/format_experiment_result_row.py` with `--start-after-step 5500`
  to add the E76 row to `EXPERIMENT_RESULTS.md`, so inherited E73 history does
  not count as E76's best validation lDDT.
- Next branch is E77: sync the committed coface-degree attenuation code to the
  owned Runpod workspace, verify remote py_compile/parser support, and launch
  from E73 with `--simplex-boundary-message-degree-attenuation 0.25`.
- Synced the E77 source/tests to the owned Runpod workspace after E76 returned
  and no benchmark process was active. Remote py_compile passed for
  `minalphafold/simplex.py`, `minalphafold/model_config.py`, and
  `scripts/run_nanofold_public_benchmarks.py`; parser smoke confirmed
  attenuation `0.25`.
- E77 launched on the same owned Runpod B200 pod `lovgzo4hz2k4fp` with run
  name `e77_degree_atten025_from_e73_s6000_c256_m64`, log path
  `/workspace/SimplexFold/logs/e77_degree_atten025_from_e73.log`, and
  artifact path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e77_degree_atten025_from_e73_s6000_c256_m64/`.
  Main Python PID is `24587`; data-worker PIDs are `27514` and `27515`.
  The runner resumed E73 at step 5500/examples 44000, loaded 1244 matching
  model tensors, initialized 0 new/missing tensors, and started a fresh
  optimizer. Remote metadata records effective batch 8, crop 256, MSA depth
  64, no templates, runtime edge-frame scale `0.0125`, and
  `simplex_boundary_message_degree_attenuation=0.25`.
- E77 poll on `lovgzo4hz2k4fp`: still running, no `results.json`; GPU active.
  History still contains inherited E73 rows through step 5500 only, so there
  is no E77 result to add to `EXPERIMENT_RESULTS.md`.
- Launched a second owned Runpod pod for a parallel prepared fallback:
  `o1dy17ouv8w5mz` (`codex-simplexfold-e74-runpod-20260512`), H100 SXM,
  image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, 160 GB container
  disk, SSH `root@103.207.149.82 -p 10764`, hourly cost reported by
  `runpodctl` as `$2.99/hr`.
- Initial pod-to-pod workspace streaming from `lovgzo4hz2k4fp` was moving too
  slowly and would have kept reading from the active E77 workspace, so it was
  stopped. The H100 pod was then staged by cloning SimplexFold commit
  `60c82133046f7a3cea393d5567af15ebc07c15b6` and nanoFold-Competition commit
  `96afc8467a108aa8bee3b51cdf4a030cd656a960`, rsyncing only public
  `data/processed_features`, `data/processed_labels`, and `data/manifests`
  from local, and copying only the 34 MB E73 checkpoint from the B200 pod.
- E74 H100 prelaunch verification passed: public manifest counts
  `10000/1000/11000`, processed feature/label NPZ counts `11000/11000`, E73
  checkpoint present with `35,385,519` bytes, remote py_compile passed for
  `minalphafold/simplex.py`, `minalphafold/model_config.py`, and
  `scripts/run_nanofold_public_benchmarks.py`; parser smoke confirmed
  `--simplex-geometry-distance-weight 0.025`; NanoFold FoldScore import works.
- E74 launched on owned pod `o1dy17ouv8w5mz` with run name
  `e74_light_geom0025_from_e73_s6000_c256_m64`, log path
  `/workspace/SimplexFold/logs/e74_light_geom0025_from_e73.log`, and artifact
  path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e74_light_geom0025_from_e73_s6000_c256_m64/`.
  Main Python PID is `850`; data-worker PIDs are `1027` and `1028`. The
  runner resumed E73 at step 5500/examples 44000, loaded 1244 matching model
  tensors, initialized 0 new/missing tensors, and started a fresh optimizer.
  Remote metadata records effective batch 8, crop 256, MSA depth 64, no
  templates, runtime edge-frame scale `0.0125`, and
  `simplex_geometry_distance_weight=0.025`.
- E77 returned on owned pod `lovgzo4hz2k4fp`: step 6000
  `val_lddt_ca=0.3733267541974783`, FoldScore `0.37099136784672737`,
  `val_ca_drmsd=10.128643780946732`, predicted/true C-alpha radius
  `11.863174617290497 / 15.40340667963028`, selected face/tetra boundary
  lDDT `0.5420919340103865` / `0.5265114568173885`, boundary length MAE
  `2.5714316442608833` / `2.7038855478167534`, and contraction fractions
  `0.6467265896499157` / `0.6474730856716633`.
- E77 interpretation: reject as a primary branch. It did exactly what the
  topology diagnostic asked for by improving selected-boundary lDDT and
  boundary length errors relative to E73/E76, but it still lost primary
  C-alpha lDDT versus E73. This suggests the next live test, E74, should move
  upstream to the selected-complex construction prior rather than further
  normalizing the same incidence readout.
- Copied E77 returned artifacts locally under ignored
  `artifacts/nanofold_public_benchmarks/e77_degree_atten025_from_e73_s6000_c256_m64/`
  and copied the launch log to ignored
  `logs/e77_degree_atten025_from_e73.log`. The local artifact pull excluded
  the checkpoint directory.
- Used `scripts/format_experiment_result_row.py` with `--start-after-step 5500`
  to add the E77 row to `EXPERIMENT_RESULTS.md`, so inherited E73 history does
  not count as E77's best validation lDDT.
- Stopped the completed/rejected E77 B200 pod `lovgzo4hz2k4fp` after copying
  and locally verifying the returned artifacts. E74 continues on the only
  active launched pod, `o1dy17ouv8w5mz`.
- 2026-05-12 PDF reference recheck: local copies of
  `references/papers/hands_on_geometric_deep_learning_nodes_to_complexes.pdf`
  and `references/papers/2509.03885v1.pdf` still hash-match the user-provided
  Downloads PDFs. Full-text extraction with `pdftotext -layout` covered 28
  pages and 22 pages, respectively, and about 14.5k extracted words. The
  takeaway for the live branch is unchanged but firmer: prefer changes that
  alter the selected cell complex, incidence operators, outer-edge or
  boundary-edge communication, or selected-cell realization diagnostics. Do
  not promote a generic all-pairs coordinate loss just to chase lDDT.
- E74 returned on owned pod `o1dy17ouv8w5mz`: step 6000
  `val_lddt_ca=0.38410646840929985`, FoldScore `0.3665613066405058`,
  `val_ca_drmsd=10.189340263605118`, predicted/true C-alpha radius
  `11.426604449748993 / 15.403406739234924`, selected face/tetra boundary
  lDDT `0.5408848244696856` / `0.5257674232125282`, boundary length MAE
  `2.514910012483597` / `2.650992676615715`, contraction fractions
  `0.5940733812749386` / `0.5957129746675491`, and boundary-edge mean degree
  `11.593819439411163` / `77.29213094711304`.
- E74 interpretation: keep as the new primary-lDDT leader. It improves E73's
  `0.3807267025113106` local C-alpha lDDT and improves selected-boundary
  lDDT/length/contraction diagnostics, supporting the topology-construction
  hypothesis behind the lighter recycled-geometry selector. FoldScore and
  dRMSD softened versus E73, so continue with a short gate rather than a
  30,000-step commitment.
- Copied E74 returned artifacts locally under ignored
  `artifacts/nanofold_public_benchmarks/e74_light_geom0025_from_e73_s6000_c256_m64/`
  and copied the launch log to ignored
  `logs/e74_light_geom0025_from_e73.log`. The local artifact pull excluded the
  checkpoint directory; the remote E74 checkpoint remains available for the
  continuation.
- Used `scripts/format_experiment_result_row.py` with `--start-after-step 5500`
  to add the E74 row to `EXPERIMENT_RESULTS.md`, so inherited E73 history does
  not count as E74's best validation lDDT.
- E78 launched on the same owned H100 pod `o1dy17ouv8w5mz` with run name
  `e78_light_geom0025_from_e74_s6500_c256_m64`, log path
  `/workspace/SimplexFold/logs/e78_light_geom0025_from_e74.log`, and artifact
  path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e78_light_geom0025_from_e74_s6500_c256_m64/`.
  Remote prelaunch checks found no active Python benchmark process, py_compile
  passed for `minalphafold/simplex.py`, `minalphafold/model_config.py`, and
  `scripts/run_nanofold_public_benchmarks.py`, parser support for
  `--simplex-geometry-distance-weight` was confirmed, and the E74 checkpoint
  was present. Main Python PID is `1969`. The runner resumed E74 at step
  6000/examples 48000, loaded 1244 matching model tensors, initialized 0
  new/missing tensors, and started a fresh optimizer.
- E79 local implementation prepared while E78 runs: added runtime overrides
  and schedules for `simplex_face_top_k` and `simplex_tetra_top_k`. This lets
  a resumed checkpoint start with the full selected higher-rank clique and
  gradually sparsify which face/tetra cochains exist, send messages, and
  contribute selected-cell losses. It is a topology-construction curriculum,
  not a new output-coordinate loss, and it adds no parameters.
- E79 local validation passed:
  `python -m py_compile minalphafold/simplex.py minalphafold/evoformer.py minalphafold/model.py minalphafold/trainer.py scripts/run_nanofold_public_benchmarks.py`;
  `python -m pytest tests/test_nanofold_public_benchmarks.py::test_model_config_override_flags_are_accepted_by_cli_parser tests/test_nanofold_public_benchmarks.py::test_runtime_simplex_message_scales_ramp_and_enter_model_inputs tests/test_nanofold_public_benchmarks.py::test_evaluate_uses_runtime_simplex_overrides_for_validation tests/test_simplex.py::test_simplicial_adapter_runtime_cell_topk_override_caps_active_cells tests/test_simplex.py::test_build_simplex_topology_cell_topk_caps_active_higher_rank_cells tests/test_trainer.py::test_simplicial_cell_topk_selector_adds_no_parameters`.
  Do not sync or launch E79 while E78 is still active; use it only after the
  E78 result is copied, recorded, and interpreted.
- E78 return decision rule: keep/continue only if it matches or beats E74's
  `val_lddt_ca=0.38410646840929985` without a selected-boundary diagnostic
  collapse. If E78 loses local lDDT but keeps or improves boundary geometry,
  launch E79 from the strongest E74/E78 checkpoint rather than another
  half-scale light-geometry continuation. If E78 improves only FoldScore or
  dRMSD, treat that as geometry-side evidence and still avoid a 30,000-step
  confirmation until a branch is plausibly moving toward `val_lddt_ca > 0.7`.
- E78 returned on owned pod `o1dy17ouv8w5mz`: step 6500
  `val_lddt_ca=0.3853302728384733`, FoldScore `0.37175922095775604`,
  `val_ca_drmsd=10.159474521875381`, predicted/true C-alpha radius
  `11.378286242485046 / 15.403406739234924`, selected face/tetra boundary
  lDDT `0.5434355866163969` / `0.5286743398755789`, boundary length MAE
  `2.4519101977348328` / `2.5800751969218254`, and contraction fractions
  `0.6319687962532043` / `0.6327483579516411`.
- E78 interpretation: keep as the new primary-lDDT leader. It improves E74's
  local C-alpha lDDT, FoldScore, dRMSD, selected-boundary lDDT, and selected
  boundary length error under the same topology-construction recipe. The
  contraction fraction regressed versus E74, so continue only as another
  short gate and keep E79's sparse-cell schedule ready if local lDDT turns
  over while selected-boundary diagnostics stay healthy.
- Copied E78 returned artifacts locally under ignored
  `artifacts/nanofold_public_benchmarks/e78_light_geom0025_from_e74_s6500_c256_m64/`
  and copied the launch log to ignored
  `logs/e78_light_geom0025_from_e74.log`. The local artifact pull excluded
  the checkpoint directory; the remote E78 checkpoint remains available for
  continuation.
- Used `scripts/format_experiment_result_row.py` with `--start-after-step
  6000` to add the E78 row to `EXPERIMENT_RESULTS.md`, so inherited E74
  history does not count as E78's best validation lDDT.
- Refreshed local E79 fallback validation after E78 returned:
  `python -m py_compile minalphafold/simplex.py minalphafold/evoformer.py minalphafold/model.py minalphafold/trainer.py scripts/run_nanofold_public_benchmarks.py`;
  `python -m pytest tests/test_nanofold_public_benchmarks.py::test_model_config_override_flags_are_accepted_by_cli_parser tests/test_nanofold_public_benchmarks.py::test_runtime_simplex_message_scales_ramp_and_enter_model_inputs tests/test_nanofold_public_benchmarks.py::test_evaluate_uses_runtime_simplex_overrides_for_validation tests/test_simplex.py::test_simplicial_adapter_runtime_cell_topk_override_caps_active_cells tests/test_simplex.py::test_build_simplex_topology_cell_topk_caps_active_higher_rank_cells tests/test_trainer.py::test_simplicial_cell_topk_selector_adds_no_parameters`.
- E80 launched on the same owned H100 pod `o1dy17ouv8w5mz` with run name
  `e80_light_geom0025_from_e78_s7000_c256_m64`, log path
  `/workspace/SimplexFold/logs/e80_light_geom0025_from_e78.log`, and artifact
  path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e80_light_geom0025_from_e78_s7000_c256_m64/`.
  Remote prelaunch checks found no active Python benchmark process, py_compile
  passed for `minalphafold/simplex.py`, `minalphafold/model_config.py`, and
  `scripts/run_nanofold_public_benchmarks.py`, and the E78 checkpoint was
  present. Main Python PID is `2543`. The launch resumes E78 from step 6500
  to 7000 with the same light-geometry selector, selected-boundary losses, and
  half-scale edge-frame message recipe.
- E80 startup poll at 2026-05-12T06:58:18Z confirmed the benchmark process is
  alive, `results.json` is absent as expected, and the runner resumed from
  the E78 checkpoint at step 6500/examples 52000 with 1244 matching model
  tensors loaded.
- E81 local implementation prepared while E80 runs: added
  `simplex_cell_score_degree_penalty`, a zero-parameter adjustment to capped
  selected-cell scoring. When face/tetra top-k caps are active, candidate
  cells are still scored by selected boundary-edge logits, but cells whose
  boundary edges are already heavily reused across the candidate complex are
  down-ranked by a log-degree penalty. This is a topology-construction change
  to which rank-2/rank-3 cochains exist and send messages, not an output
  coordinate loss.
- E81 local validation passed:
  `python -m py_compile minalphafold/simplex.py minalphafold/model_config.py scripts/run_nanofold_public_benchmarks.py`;
  `python -m pytest tests/test_simplex.py::test_cell_score_degree_penalty_prefers_less_reused_boundary_edges tests/test_simplex.py::test_build_simplex_topology_cell_topk_caps_active_higher_rank_cells tests/test_nanofold_public_benchmarks.py::test_model_config_override_flags_are_accepted_by_cli_parser tests/test_trainer.py::test_simplicial_cell_degree_penalty_adds_no_parameters tests/test_trainer.py::test_simplicial_cell_topk_selector_adds_no_parameters`.
  Do not sync or launch E81 while E80 is active.
- Broader local validation for the prepared E81 code passed after correcting a
  stale outer-edge adapter test assertion that had tried to gate
  `simplex_outer_edge_update_scale` with the unrelated hodge runtime override:
  `python -m pytest tests/test_simplex.py tests/test_nanofold_public_benchmarks.py tests/test_trainer.py`
  reported `151 passed`.
- E80 returned on owned pod `o1dy17ouv8w5mz`: step 7000
  `val_lddt_ca=0.3820213433355093`, FoldScore `0.36820204742252827`,
  `val_ca_drmsd=10.24925634264946`, predicted/true C-alpha radius
  `11.247207999229431 / 15.403406739234924`, selected face/tetra boundary
  lDDT `0.5358894020318985` / `0.5191911403089762`, boundary length MAE
  `2.6559875458478928` / `2.8001304268836975`, contraction fractions
  `0.6268004588782787` / `0.6288906000554562`, boundary-edge mean degree
  `11.145169138908386` / `74.30112886428833`, and boundary unique-edge
  fraction `0.09011029245462714` / `0.013516543868194071`.
- E80 interpretation: reject as a primary branch. It lost E78's local
  C-alpha lDDT and also regressed selected-boundary lDDT and boundary length
  error, although contraction improved slightly. This argues against another
  blind light-geometry continuation and for the prepared sparse-cell
  topology-construction fallback from the stronger E78 checkpoint.
- Copied E80 returned artifacts locally under ignored
  `artifacts/nanofold_public_benchmarks/e80_light_geom0025_from_e78_s7000_c256_m64/`
  and copied the launch log to ignored
  `logs/e80_light_geom0025_from_e78.log`. The local artifact pull excluded
  the checkpoint directory.
- Used `scripts/format_experiment_result_row.py` with `--start-after-step
  6500` to add the E80 row to `EXPERIMENT_RESULTS.md`, so inherited E78
  history does not count as E80's best validation lDDT.
- Synced latest local SimplexFold source/docs/tests to the owned H100
  workspace without deleting remote artifacts, logs, or checkpoints. Remote
  py_compile passed for `minalphafold/simplex.py`,
  `minalphafold/model_config.py`, and
  `scripts/run_nanofold_public_benchmarks.py`; parser smoke confirmed support
  for `--simplex-face-top-k-final`, `--simplex-tetra-top-k-final`, and
  `--simplex-cell-score-degree-penalty`.
- E79 launched on the same owned H100 pod `o1dy17ouv8w5mz` with run name
  `e79_scheduled_topk_from_e78_s7000_c256_m64`, log path
  `/workspace/SimplexFold/logs/e79_scheduled_topk_from_e78.log`, and artifact
  path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e79_scheduled_topk_from_e78_s7000_c256_m64/`.
  Remote prelaunch checks found no active Python benchmark process and the
  E78 checkpoint was present. Main Python PID is `3128`. The launch resumes
  E78 from step 6500 to 7000 with the same light-geometry selector,
  selected-boundary losses, and half-scale edge-frame message recipe, while
  scheduling active face/tetra cell caps from full clique to `24` / `48`.
- E79 startup poll at 2026-05-12T07:41:52Z confirmed the benchmark process is
  alive, `results.json` is absent as expected, and the runner resumed from
  the E78 checkpoint at step 6500/examples 52000 with 1244 matching model
  tensors loaded.
- E79 returned on owned pod `o1dy17ouv8w5mz`: step 7000
  `val_lddt_ca=0.3885485269129276`, FoldScore `0.3728215303272009`,
  `val_ca_drmsd=10.266113311052322`, predicted/true C-alpha radius
  `11.153955072164536 / 15.403406739234924`, selected face/tetra boundary
  lDDT `0.6963267438113689` / `0.6825862638652325`, boundary length MAE
  `1.2635142654180527` / `1.3586284592747688`, contraction fractions
  `0.5353225655853748` / `0.540894154459238`, boundary-edge mean degree
  `12.47058242559433` / `35.61734676361084`, and boundary unique-edge
  fraction `0.080328143582315` / `0.028286503751291676`.
- E79 interpretation: keep as the new primary-lDDT leader and the strongest
  topology-diagnostic run. Sparse selected-cell scheduling beat E78 on local
  C-alpha lDDT and FoldScore while dramatically improving selected-boundary
  realization and reducing tetra boundary-edge reuse. dRMSD softened and
  radius remains under-expanded, so continue with a short sparse-cell gate
  rather than a long confirmation.
- Copied E79 returned artifacts locally under ignored
  `artifacts/nanofold_public_benchmarks/e79_scheduled_topk_from_e78_s7000_c256_m64/`
  and copied the launch log to ignored
  `logs/e79_scheduled_topk_from_e78.log`. The local artifact pull excluded
  the checkpoint directory; the remote E79 checkpoint remains available for
  continuation.
- Used `scripts/format_experiment_result_row.py` with `--start-after-step
  6500` to add the E79 row to `EXPERIMENT_RESULTS.md`, so inherited E78
  history does not count as E79's best validation lDDT.
- E82 launched on the same owned H100 pod `o1dy17ouv8w5mz` with run name
  `e82_sparse_topk_from_e79_s7500_c256_m64`, log path
  `/workspace/SimplexFold/logs/e82_sparse_topk_from_e79.log`, and artifact
  path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e82_sparse_topk_from_e79_s7500_c256_m64/`.
  Remote prelaunch checks found no active Python benchmark process, py_compile
  passed for `minalphafold/simplex.py`, `minalphafold/model_config.py`, and
  `scripts/run_nanofold_public_benchmarks.py`, and the E79 checkpoint was
  present. Main Python PID is `3565`. The launch resumes E79 from step 7000
  to 7500 with sparse face/tetra caps held at `24` / `48`.
- E82 startup poll at 2026-05-12T08:32:38Z confirmed the benchmark process is
  alive, `results.json` is absent as expected, and the runner resumed from
  the E79 checkpoint at step 7000/examples 56000 with 1244 matching model
  tensors loaded.
- 2026-05-12T14:13:20Z PDF reference pass: verified the two user-provided
  papers are saved in `references/papers/` and hash-match the Downloads
  originals. Re-extracted both full texts with `pdftotext -layout`:
  28 pages / about 3.8k words for the TDL guide and 22 pages / about 10.7k
  words for Topotein. Updated `references/papers/READING_NOTES.md` and the
  reference-paper design rules in `EXPERIMENTS.md`. Main design consequence:
  keep E91/E90 focused on selected-complex construction and incidence/
  outer-edge cochain communication; do not add generic lDDT/radius/all-pairs
  losses unless they supervise only the selected sparse complex.
- 2026-05-12T14:20:59Z E91 status poll on owned pod `o1dy17ouv8w5mz`: Python
  PID `6904` is still active, `results.json` is absent, and
  `history_full_msa_to_face.json` still ends at the inherited E86 step-8500
  row with `val_lddt_ca=0.3990174550563097`. No E91 result has returned yet,
  so `EXPERIMENT_RESULTS.md` remains unchanged.
- E91 returned on owned pod `o1dy17ouv8w5mz`: step 9000
  `val_lddt_ca=0.38974714651703835`, FoldScore `0.38195772282779217`,
  `val_ca_drmsd=9.930909246206284`, predicted/true C-alpha radius
  `11.822966694831848 / 15.403406739234924`, selected face/tetra boundary
  lDDT `0.7414444871246815` / `0.7256345339119434`, boundary length MAE
  `1.0578380264341831` / `1.1556165665388107`, and contraction fractions
  `0.6197949759662151` / `0.6163310036063194`.
- E91 interpretation: reject as a primary-lDDT continuation. It improved
  dRMSD and selected-boundary realization over E86, but primary C-alpha lDDT
  fell from E86's `0.3990174550563097` to `0.38974714651703835`. Pivot to
  the directed boundary-readout gate rather than continuing weak outer-edge
  transport.
- Copied E91 returned artifacts locally under ignored
  `artifacts/nanofold_public_benchmarks/e91_weak_outer_edge_from_e86_s9000_c256_m64/`
  and copied the launch log to ignored `logs/e91_weak_outer_edge_from_e86.log`.
  The local artifact pull excluded the checkpoint directory; the remote E91
  checkpoint remains available if needed, but it is not the preferred next
  branch.
- Used `scripts/format_experiment_result_row.py` with `--start-after-step
  8500` to add the E91 row to `EXPERIMENT_RESULTS.md`, so inherited E86
  history does not count as E91's best validation lDDT.
- E87 launch decision: use the cleaner E81 checkpoint rather than the E91
  checkpoint. E91 improved dRMSD and selected-boundary lDDT but lost primary
  lDDT, so the next useful gate should isolate source/target-directed
  simplex-to-pair readout instead of stacking more changes on the regressed
  outer-edge continuation.
- E87 prelaunch checks on owned pod `o1dy17ouv8w5mz`: no active Python
  benchmark process, E81 checkpoint present at
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e81_degree_penalty_from_e82_s8000_c256_m64/checkpoints/full_msa_to_face_latest.pt`,
  remote py_compile passed for `minalphafold/simplex.py`,
  `minalphafold/model_config.py`, `minalphafold/trainer.py`, and
  `scripts/run_nanofold_public_benchmarks.py`; CLI help confirmed support for
  `--simplex-boundary-readout-directionality` and its runtime schedule flags.
- E87 launched on the same owned H100 pod with run name
  `e87_directed_boundary_from_e81_s8500_c256_m64`, log path
  `/workspace/SimplexFold/logs/e87_directed_boundary_from_e81.log`, artifact
  path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e87_directed_boundary_from_e81_s8500_c256_m64/`,
  and Python PID `7573`. Startup poll at `2026-05-12T14:30:13Z` confirmed the
  benchmark process is alive, `results.json` is absent as expected, metadata
  and history files exist, and the runner resumed E81 at step 8000/examples
  64000 with 1244 matching model tensors loaded and 0 new/missing tensors.
- Retargeted the existing heartbeat automation `check-simplexfold-e57-runpod`
  from E91 to E87, keeping the same owned-pod-only restriction and the rule
  that heartbeat must not launch follow-up experiments automatically.
- E87 returned on owned pod `o1dy17ouv8w5mz`: step 8500
  `val_lddt_ca=0.39919308573007584`, FoldScore `0.3831401728093624`,
  `val_ca_drmsd=10.242752134799957`, predicted/true C-alpha radius
  `11.432185918092728 / 15.403406739234924`, selected face/tetra boundary
  lDDT `0.7446220442652702` / `0.7280045710504055`, boundary length MAE
  `1.0540529862046242` / `1.1570463925600052`, contraction fractions
  `0.57584448158741` / `0.5786093026399612`, boundary-edge mean degree
  `11.664112329483032` / `33.11342191696167`, and boundary unique-edge
  fraction `0.08602189490787632` / `0.030495643238529865`.
- E87 interpretation: keep as a tiny new primary-lDDT best and as evidence
  that directed source/target boundary readout is a valid topological
  communication route. Caveat: FoldScore and dRMSD softened versus E86, so do
  not launch a long confirmation. Run one short continuation with directionality
  held at `0.5`; if that turns over, pivot to outer-edge-supported cell scoring.
- Copied E87 returned artifacts locally under ignored
  `artifacts/nanofold_public_benchmarks/e87_directed_boundary_from_e81_s8500_c256_m64/`
  and copied the launch log to ignored `logs/e87_directed_boundary_from_e81.log`.
  The local artifact pull excluded the checkpoint directory; the remote E87
  checkpoint remains available for the short continuation.
- Used `scripts/format_experiment_result_row.py` with `--start-after-step
  8000` to add the E87 row to `EXPERIMENT_RESULTS.md`, so inherited E81
  history does not count as E87's best validation lDDT.
- E92 launch decision: E87 is a tiny new primary-lDDT best and improves
  selected-boundary lDDT/contraction, but FoldScore and dRMSD softened versus
  E86. Run exactly one short continuation from E87 with boundary-readout
  directionality held at `0.5`; if it turns over, pivot to
  outer-edge-supported cell scoring rather than continuing directionality.
- E92 prelaunch checks on owned pod `o1dy17ouv8w5mz`: no active Python
  benchmark process, E87 checkpoint present at
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e87_directed_boundary_from_e81_s8500_c256_m64/checkpoints/full_msa_to_face_latest.pt`,
  remote py_compile passed for `minalphafold/simplex.py`,
  `minalphafold/model_config.py`, `minalphafold/trainer.py`, and
  `scripts/run_nanofold_public_benchmarks.py`; CLI help confirmed support for
  the directionality runtime flags and `--simplex-cell-score-outer-edge-weight`
  for the fallback.
- E92 launched on the same owned H100 pod with run name
  `e92_continue_directed_boundary_from_e87_s9000_c256_m64`, log path
  `/workspace/SimplexFold/logs/e92_continue_directed_boundary_from_e87.log`,
  artifact path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e92_continue_directed_boundary_from_e87_s9000_c256_m64/`,
  and Python PID `8068`. Startup poll at `2026-05-12T15:31:57Z` confirmed the
  benchmark process is alive, `results.json` is absent as expected, metadata
  and history files exist, and the runner resumed E87 at step 8500/examples
  68000 with 1244 matching model tensors loaded and 0 new/missing tensors.
- Retargeted the existing heartbeat automation `check-simplexfold-e57-runpod`
  from E87 to E92, keeping the same owned-pod-only restriction and the rule
  that heartbeat must not launch follow-up experiments automatically.
- 2026-05-12T15:55Z current PDF reread: re-extracted and reread both saved
  reference PDFs from `references/papers/`. The TDL guide keeps E92/E90 framed
  around explicit topology operators: boundary incidence, rank-to-rank
  aggregation, and complex construction. Topotein keeps the protein-specific
  emphasis on directed edges, outer-edge neighborhoods, and edge-centric
  scalarization, while ruling out DSSP/SSE labels for official NanoFold paths.
  This supports E92 as a directed source/target incidence readout test and E90
  as a complex-construction fallback; it does not support a generic dense
  lDDT/radius/all-pairs loss. E92 status at this poll: Python PID `8068` is
  active, `results.json` is absent, and history still ends at inherited E87
  step 8500 with `val_lddt_ca=0.39919308573007584`.
- 2026-05-12T16:09Z E92 status poll on owned pod `o1dy17ouv8w5mz`: Python
  PID `8068` remains active, GPU utilization sampled at `74%`, `results.json`
  is still absent, and the run history/log still end at the inherited E87
  startup state. No `EXPERIMENT_RESULTS.md` update yet because E92 has not
  returned.
- E92 returned on owned pod `o1dy17ouv8w5mz`: step 9000
  `val_lddt_ca=0.39684198051691055`, FoldScore `0.3829442337155342`,
  `val_ca_drmsd=9.961655408143997`, predicted/true C-alpha radius
  `11.736155331134796 / 15.403406739234924`, selected face/tetra boundary
  lDDT `0.7400240190327168` / `0.7230039089918137`, boundary length MAE
  `1.0625056326389313` / `1.173266414552927`, contraction fractions
  `0.5681420378386974` / `0.5686059035360813`, boundary-edge mean degree
  `11.466184198856354` / `32.314730405807495`, and boundary unique-edge
  fraction `0.08755682366733795` / `0.031285135159139124`.
- E92 interpretation: reject as a primary-lDDT continuation. It improved
  dRMSD versus E87, but primary C-alpha lDDT fell below E87's
  `0.39919308573007584` and E86's `0.3990174550563097`, and selected-boundary
  lDDT softened. Pivot to the prepared E90 outer-edge-supported cell scorer
  rather than continuing the directed boundary-readout mechanism.
- Copied E92 returned artifacts locally under ignored
  `artifacts/nanofold_public_benchmarks/e92_continue_directed_boundary_from_e87_s9000_c256_m64/`
  and copied the launch log to ignored
  `logs/e92_continue_directed_boundary_from_e87.log`. The local artifact pull
  excluded the checkpoint directory.
- Used `scripts/format_experiment_result_row.py` with `--start-after-step
  8500` to add the E92 row to `EXPERIMENT_RESULTS.md`, so inherited E87
  history does not count as E92's best validation lDDT.
- E90 launch repair: the first E90 launch attempt exposed stale runtime
  plumbing on the remote checkout. The runner/trainer could create
  `simplex_cell_score_outer_edge_weight_override`, but the model/evoformer
  path did not yet accept and forward it. Added the missing
  `AlphaFold2 -> SimplicialEvoformer -> SimplicialAdapter` override plumbing
  locally, added a focused regression test for the forward signatures, passed
  local py_compile and seven focused pytest cases, synced the patched files to
  the owned pod, and cleared only the failed E90 artifact directory that this
  thread created.
- E90 launched cleanly on the same owned H100 pod with run name
  `e90_outer_edge_score_from_e81_s8500_c256_m64`, log path
  `/workspace/SimplexFold/logs/e90_outer_edge_score_from_e81.log`, artifact
  path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e90_outer_edge_score_from_e81_s8500_c256_m64/`,
  and Python PID `9139`. Startup poll at `2026-05-12T16:41:01Z` confirmed the
  benchmark process is alive, `results.json` is absent as expected, metadata
  and history files exist, and the runner resumed E81 at step 8000/examples
  64000 with 1244 matching model tensors loaded and 0 new/missing tensors.
- Retargeted the existing heartbeat automation `check-simplexfold-e57-runpod`
  from E92 to E90, keeping the same owned-pod-only restriction and the rule
  that heartbeat must not launch follow-up experiments automatically.
- 2026-05-12T16:45Z E90 status poll on owned pod `o1dy17ouv8w5mz`: Python
  PID `9139` is active, GPU utilization sampled at `48%`, `results.json` is
  absent, and history still ends at the inherited E81 step 8000 row with
  `val_lddt_ca=0.39799308963119984`.
- E90 returned on owned pod `o1dy17ouv8w5mz`: step 8500
  `val_lddt_ca=0.3920442685484886`, FoldScore `0.37827762216329575`,
  `val_ca_drmsd=10.040688931941986`, predicted/true C-alpha radius
  `11.524468511343002 / 15.403406739234924`, selected face/tetra boundary
  lDDT `0.7364524565637112` / `0.7197068147361279`, boundary length MAE
  `1.0608491748571396` / `1.1697352267801762`, contraction fractions
  `0.5463919192552567` / `0.5451707523316145`, boundary-edge mean degree
  `11.607987940311432` / `32.68083894252777`, and boundary unique-edge
  fraction `0.0863968789105419` / `0.03085274268859773`.
- E90 interpretation: reject as a primary-lDDT branch. It improved
  selected-boundary contraction versus E81, but primary lDDT fell below E81,
  E86, E87, and E92; FoldScore and selected-boundary lDDT also stayed below
  the E81/E86/E87 leaders. The outer-edge-supported scorer is not useful as a
  standalone construction change. Move next to E88 runtime-gated latent
  segment cells from the E81 checkpoint.
- Copied E90 returned artifacts locally under ignored
  `artifacts/nanofold_public_benchmarks/e90_outer_edge_score_from_e81_s8500_c256_m64/`
  and copied the launch log to ignored `logs/e90_outer_edge_score_from_e81.log`.
  The local artifact pull excluded the checkpoint directory.
- Used `scripts/format_experiment_result_row.py` with `--start-after-step
  8000` to add the E90 row to `EXPERIMENT_RESULTS.md`, so inherited E81
  history does not count as E90's best validation lDDT.
- E88 launch decision: E90 improved selected-boundary contraction but lost
  primary lDDT, so the next useful paper-aligned test is the runtime-gated
  latent segment-cell route rather than another cell-score bonus. This borrows
  Topotein's secondary-structure rank without DSSP/SSE labels by using latent
  contiguous segment cochains built from official features and recycled model
  state.
- E88 prelaunch checks on owned pod `o1dy17ouv8w5mz`: no active Python
  benchmark process, E81 checkpoint present at
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e81_degree_penalty_from_e82_s8000_c256_m64/checkpoints/full_msa_to_face_latest.pt`,
  remote py_compile passed for `minalphafold/simplex.py`,
  `minalphafold/model_config.py`, `minalphafold/evoformer.py`,
  `minalphafold/model.py`, `minalphafold/trainer.py`, and
  `scripts/run_nanofold_public_benchmarks.py`; CLI help confirmed support for
  the segment-cell runtime flags.
- E88 launched on the same owned H100 pod with run name
  `e88_segment_cells_from_e81_s8500_c256_m64`, log path
  `/workspace/SimplexFold/logs/e88_segment_cells_from_e81.log`, artifact path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e88_segment_cells_from_e81_s8500_c256_m64/`,
  and Python PID `9628`. Startup poll at `2026-05-12T17:41:06Z` confirmed the
  benchmark process is alive, `results.json` is absent as expected, metadata
  and history files exist, and the runner resumed E81 at step 8000/examples
  64000 with 1244 matching model tensors loaded and 48 new/missing segment
  tensors initialized.
- Retargeted the existing heartbeat automation `check-simplexfold-e57-runpod`
  from E90 to E88, keeping the same owned-pod-only restriction and the rule
  that heartbeat must not launch follow-up experiments automatically.
- E88 returned on owned pod `o1dy17ouv8w5mz`: step 8500
  `val_lddt_ca=0.38908764719963074`, FoldScore `0.38241876289248466`,
  `val_ca_drmsd=10.198628336191177`, predicted/true C-alpha radius
  `11.50267744064331 / 15.403406739234924`, selected face/tetra boundary
  lDDT `0.7203109562397003` / `0.7046942040324211`, boundary length MAE
  `1.2001516371965408` / `1.30133518576622`, contraction fractions
  `0.6244679093360901` / `0.6275281123816967`, boundary-edge mean degree
  `12.014307677745819` / `35.019731402397156`, and boundary unique-edge
  fraction `0.0835734105056879` / `0.028839249425916206`.
- E88 parameter audit: `parameters=3,282,002`, which is +5.64% versus the
  AF2-medium pair-only baseline `3,106,642` and above the +5% ceiling
  `3,261,974`. The previous segment-cell budget test counted segment cells
  without the edge-frame module combination used in E88, so it missed the
  actual launched profile.
- E88 interpretation: reject. It regressed below E81/E86/E87 on primary
  lDDT and also violated the parameter contract. Do not continue this
  segment-cell branch in its current form. Future topology-module
  combinations should be counted before launch and run with the new runner
  `--max-parameters 3261974` guard.
- Copied E88 returned artifacts locally under ignored
  `artifacts/nanofold_public_benchmarks/e88_segment_cells_from_e81_s8500_c256_m64/`
  and copied the launch log to ignored `logs/e88_segment_cells_from_e81.log`.
  The local artifact pull excluded the checkpoint directory.
- Used `scripts/format_experiment_result_row.py` with `--start-after-step
  8000` to add the E88 row to `EXPERIMENT_RESULTS.md`, so inherited E81
  history does not count as E88's best validation lDDT.
- Added a runner-level `--max-parameters` preflight and a regression test for
  the exact E88 segment-cell plus edge-frame module combination. The E88
  module set now explicitly asserts as over budget locally, while segment
  cells by themselves remain documented as within the 5% cap.
- Next launch decision: because E88 is both lower-lDDT and over budget, move
  to E89 pair-preserving simplex readout from E81. This zero-parameter gate
  keeps selected face/tetra cochain evidence flowing to the pair tensor
  `Z_ij` while damping direct single/residue readout, so it remains a
  topological cochain-routing test rather than a new metric loss.
- E89 prelaunch checks on owned pod `o1dy17ouv8w5mz`: no active Python
  benchmark process, E81 checkpoint present at
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e81_degree_penalty_from_e82_s8000_c256_m64/checkpoints/full_msa_to_face_latest.pt`,
  remote py_compile passed for `scripts/run_nanofold_public_benchmarks.py`,
  CLI help confirmed support for the pair/single runtime gates and
  `--max-parameters`, and the exact E89 instantiated module set counted
  `3,154,242` parameters, under the AF2-medium +5% ceiling `3,261,974`.
  Remote pytest could not be run because `pytest` is not installed on the pod.
- E89 launched on the same owned H100 pod with run name
  `e89_pair_preserving_from_e81_s8500_c256_m64`, log path
  `/workspace/SimplexFold/logs/e89_pair_preserving_from_e81.log`, artifact
  path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e89_pair_preserving_from_e81_s8500_c256_m64/`,
  and Python PID `10400`. Startup poll at `2026-05-12T19:02Z` confirmed the
  benchmark process is alive, metadata exists, `--max-parameters 3261974` is
  recorded, and the runner resumed E81 at step 8000/examples 64000 with 1244
  matching model tensors loaded and 0 new/missing tensors initialized.
- Retargeted the existing heartbeat automation `check-simplexfold-e57-runpod`
  from E88 to E89, keeping the same owned-pod-only restriction and the rule
  that heartbeat must not launch follow-up experiments automatically.
- While E89 runs, queued E93 as the next fallback if E89 rejects: ramp the
  active selected face/tetra caps from `24/48` to `12/24` over the 8000-8500
  gate while keeping degree-penalized scoring and selected-boundary
  realization fixed. This is a zero-parameter filtration of the learned
  sparse cell complex, not a new coordinate or lDDT-style objective.
- E89 live health check at `2026-05-12T19:41:52Z` on owned pod
  `o1dy17ouv8w5mz`: Python PID `10400` was still active after about
  36 minutes, `results.json` was not present yet, GPU memory was allocated
  at about `13,535 MiB`, and the log still showed the clean E81 resume with
  1244 matching tensors loaded and 0 new/missing tensors initialized.
- E89 returned on owned pod `o1dy17ouv8w5mz`: step 8500
  `val_lddt_ca=0.39467536099255085`, FoldScore `0.3861187528818846`,
  `val_ca_drmsd=10.060284554958344`, predicted/true C-alpha radius
  `11.692678928375244 / 15.403406739234924`, selected face/tetra boundary
  lDDT `0.7363793775439262` / `0.7202896736562252`, boundary length MAE
  `1.1024678982794285` / `1.2080971263349056`, contraction fractions
  `0.6316477954387665` / `0.6290897764265537`, boundary-edge mean degree
  `11.755491852760315` / `34.04670584201813`, and boundary unique-edge
  fraction `0.08531751932676811` / `0.0296416131269318`.
- E89 interpretation: reject as a primary-lDDT branch. It stayed within the
  parameter cap at `3,154,242` parameters and improved FoldScore to `0.3861`,
  but primary lDDT fell below E81, E86, and E87. Pair-preserving simplex
  readout is therefore not enough by itself.
- Copied E89 returned artifacts locally under ignored
  `artifacts/nanofold_public_benchmarks/e89_pair_preserving_from_e81_s8500_c256_m64/`
  and copied the launch log to ignored `logs/e89_pair_preserving_from_e81.log`.
  The local artifact pull excluded the checkpoint directory.
- Used `scripts/format_experiment_result_row.py` with `--start-after-step
  8000` to add the E89 row to `EXPERIMENT_RESULTS.md`, so inherited E81
  history does not count as E89's best validation lDDT.
- E93 prelaunch checks on owned pod `o1dy17ouv8w5mz`: no active Python
  benchmark process, E81 checkpoint present, remote py_compile passed for
  `scripts/run_nanofold_public_benchmarks.py`, CLI help confirmed support for
  the face/tetra top-k final ramp flags and `--max-parameters`, and the exact
  E93 instantiated module set counted `3,154,242` parameters, under the
  AF2-medium +5% ceiling `3,261,974`.
- E93 launched on the same owned H100 pod with run name
  `e93_sparse_filtration_from_e81_s8500_c256_m64`, log path
  `/workspace/SimplexFold/logs/e93_sparse_filtration_from_e81.log`, artifact
  path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e93_sparse_filtration_from_e81_s8500_c256_m64/`,
  and Python PID `11069`. Startup poll at `2026-05-12T20:00Z` confirmed the
  benchmark process is alive, metadata exists, `--max-parameters 3261974` is
  recorded, face/tetra top-k ramps are `24/48 -> 12/24`, and the runner
  resumed E81 at step 8000/examples 64000 with 1244 matching model tensors
  loaded and 0 new/missing tensors initialized.
- Retargeted the existing heartbeat automation `check-simplexfold-e57-runpod`
  from E89 to E93, keeping the same owned-pod-only restriction and the rule
  that heartbeat must not launch follow-up experiments automatically.
- E93 live health check at `2026-05-12T20:37:56Z` on owned pod
  `o1dy17ouv8w5mz`: Python PID `11069` was still active after about
  38 minutes, `results.json` was not present yet, GPU memory was allocated
  at about `13,533 MiB`, GPU utilization sampled at `38%`, and the log still
  showed the clean E81 resume with 1244 matching tensors loaded and 0
  new/missing tensors initialized.
- E93 returned on owned pod `o1dy17ouv8w5mz`: step 8500
  `val_lddt_ca=0.3973425030708313`, FoldScore `0.38191401027143`,
  `val_ca_drmsd=10.29494559764862`, predicted/true C-alpha radius
  `11.095172643661499 / 15.403406739234924`, selected face/tetra boundary
  lDDT `0.7896677553653717` / `0.7549491301178932`, boundary length MAE
  `0.818169042468071` / `0.9819679223001003`, contraction fractions
  `0.6130472831428051` / `0.617531418800354`, boundary-edge mean degree
  `9.292783081531525` / `26.090105652809143`, and boundary unique-edge
  fraction `0.10764103507764945` / `0.03869963556465458`.
- E93 interpretation: reject as a primary-lDDT branch. The stricter
  `12/24` filtration strongly cleaned selected-boundary realization and
  reduced boundary-edge reuse, but primary lDDT fell below E81, E86, and E87,
  and predicted C-alpha radius under-expanded further. Treat this as evidence
  that a too-narrow selected complex loses useful higher-rank context.
- Copied E93 returned artifacts locally under ignored
  `artifacts/nanofold_public_benchmarks/e93_sparse_filtration_from_e81_s8500_c256_m64/`
  and copied the launch log to ignored `logs/e93_sparse_filtration_from_e81.log`.
  The local artifact pull excluded the checkpoint directory.
- Used `scripts/format_experiment_result_row.py` with `--start-after-step
  8000` to add the E93 row to `EXPERIMENT_RESULTS.md`, so inherited E81
  history does not count as E93's best validation lDDT.
- E94 launch decision: combine E87's directed source/target boundary readout
  with a gentler version of E93's filtration. Resume E81 from step 8000 to
  8500, preserve E87's incidence-normalized boundary transport, ramp
  boundary-readout directionality `0.0 -> 0.5`, and ramp selected face/tetra
  caps `24/48 -> 18/36`. This remains a zero-parameter topology-construction/
  cochain-routing test, not a generic metric loss.
- E94 prelaunch checks on owned pod `o1dy17ouv8w5mz`: no active Python
  benchmark process, E81 checkpoint present, remote py_compile passed for
  `scripts/run_nanofold_public_benchmarks.py`, CLI help confirmed support for
  boundary-readout directionality, top-k final ramp, boundary-incidence
  normalization, and `--max-parameters`, and the exact E94 instantiated
  module set counted `3,154,242` parameters, under the AF2-medium +5% ceiling
  `3,261,974`.
- E94 launched on the same owned H100 pod with run name
  `e94_moderate_filtration_directed_boundary_from_e81_s8500_c256_m64`, log
  path
  `/workspace/SimplexFold/logs/e94_moderate_filtration_directed_boundary_from_e81.log`,
  artifact path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e94_moderate_filtration_directed_boundary_from_e81_s8500_c256_m64/`,
  and Python PID `11941`. Startup poll at `2026-05-12T21:09:33Z` confirmed
  the benchmark process is alive, metadata exists, `--max-parameters 3261974`
  is recorded, boundary-readout directionality ramp is `0.0 -> 0.5`, face/tetra
  top-k ramps are `24/48 -> 18/36`, and the runner resumed E81 at step
  8000/examples 64000 with 1244 matching model tensors loaded and 0
  new/missing tensors initialized.
- Retargeted the existing heartbeat automation `check-simplexfold-e57-runpod`
  from E93 to E94, keeping the same owned-pod-only restriction and the rule
  that heartbeat must not launch follow-up experiments automatically.
- E94 live health check at `2026-05-12T21:43:08Z` on owned pod
  `o1dy17ouv8w5mz`: Python PID `11941` was still active after about
  34 minutes, `results.json` was not present yet, GPU memory was allocated
  at about `13,513 MiB`, GPU utilization sampled at `34%`, and the log still
  showed the clean E81 resume with 1244 matching tensors loaded and 0
  new/missing tensors initialized.
- E94 returned on owned pod `o1dy17ouv8w5mz`: step 8500
  `val_lddt_ca=0.3913724571466446`, FoldScore `0.37685416638851166`,
  `val_ca_drmsd=10.302786141633987`, predicted/true C-alpha radius
  `11.396026909351349 / 15.403406739234924`, selected face/tetra boundary
  lDDT `0.7599867060780525` / `0.7293643690645695`, boundary length MAE
  `0.9376725740730762` / `1.1067169606685638`, contraction fractions
  `0.5056739300489426` / `0.5157316420227289`, boundary-edge mean degree
  `12.452479362487793` / `29.885388612747192`, and boundary unique-edge
  fraction `0.08070104541502317` / `0.033798065517231766`.
- E94 interpretation: reject. Moderate filtration plus directed boundary
  readout reduced selected-boundary contraction, but primary lDDT fell below
  E81, E86, E87, and E93; selected-boundary lDDT did not approach E93, and
  boundary-edge reuse remained high. Stop pursuing additional top-k
  filtration as the next primary path.
- Copied E94 returned artifacts locally under ignored
  `artifacts/nanofold_public_benchmarks/e94_moderate_filtration_directed_boundary_from_e81_s8500_c256_m64/`
  and copied the launch log to ignored
  `logs/e94_moderate_filtration_directed_boundary_from_e81.log`. The local
  artifact pull excluded the checkpoint directory.
- Used `scripts/format_experiment_result_row.py` with `--start-after-step
  8000` to add the E94 row to `EXPERIMENT_RESULTS.md`, so inherited E81
  history does not count as E94's best validation lDDT.
- E95 launch decision: keep the broader E81 `24/48` sparse complex and
  combine E86's weak directed outer-edge context with E87's directed boundary
  readout. This tests cochain communication through outer-edge neighborhoods
  plus source/target incidence-aware writes back to `Z_ij`, rather than
  changing the selected-cell caps or adding a generic output loss.
- E95 prelaunch checks on owned pod `o1dy17ouv8w5mz`: no active Python
  benchmark process, E81 checkpoint present, remote py_compile passed for
  `scripts/run_nanofold_public_benchmarks.py`, CLI help confirmed support for
  outer-edge context, boundary-readout directionality, boundary-incidence
  normalization, and `--max-parameters`, and the exact E95 instantiated
  module set counted `3,230,834` parameters, under the AF2-medium +5% ceiling
  `3,261,974`.
- E95 launched on the same owned H100 pod with run name
  `e95_outer_edge_directed_boundary_from_e81_s8500_c256_m64`, log path
  `/workspace/SimplexFold/logs/e95_outer_edge_directed_boundary_from_e81.log`,
  artifact path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e95_outer_edge_directed_boundary_from_e81_s8500_c256_m64/`,
  and Python PID `12566`. Startup poll at `2026-05-12T22:10:05Z` confirmed
  the benchmark process is alive, metadata exists, `--max-parameters 3261974`
  is recorded, outer-edge runtime ramp is `0.0 -> 0.025`, boundary-readout
  directionality ramp is `0.0 -> 0.5`, fixed face/tetra caps are `24/48`, and
  the runner resumed E81 at step 8000/examples 64000 with 1244 matching model
  tensors loaded and 48 new/missing outer-edge-context tensors initialized.
- Retargeted the existing heartbeat automation `check-simplexfold-e57-runpod`
  from E94 to E95, keeping the same owned-pod-only restriction and the rule
  that heartbeat must not launch follow-up experiments automatically.
- E95 live health check at `2026-05-12T22:53:43Z` on owned pod
  `o1dy17ouv8w5mz`: Python PID `12566` was still active after about
  44 minutes, `results.json` was not present yet, GPU memory was allocated
  at about `15,205 MiB`, GPU utilization sampled at `79%`, and the log still
  showed the clean E81 resume with 1244 matching tensors loaded and 48
  new/missing outer-edge-context tensors initialized.
- E95 returned on owned pod `o1dy17ouv8w5mz`: step 8500
  `val_lddt_ca=0.39309102669358253`, FoldScore `0.3817453645169735`,
  `val_ca_drmsd=9.998381435871124`, predicted/true C-alpha radius
  `11.715169727802277 / 15.403406739234924`, selected face/tetra boundary
  lDDT `0.7295449376106262` / `0.7139960415661335`, boundary length MAE
  `1.1037089079618454` / `1.204644788056612`, contraction fractions
  `0.6324849687516689` / `0.6339563354849815`, boundary-edge mean degree
  `11.942960619926453` / `34.161340832710266`, and boundary unique-edge
  fraction `0.0840773594868793` / `0.029563833235177802`.
- E95 interpretation: reject as a primary-lDDT branch. Stacking weak
  outer-edge context with directed boundary readout improved dRMSD but
  primary lDDT fell well below E86/E87, and selected-boundary lDDT softened.
  Do not keep stacking these cochain-communication routes as the next main
  path.
- Copied E95 returned artifacts locally under ignored
  `artifacts/nanofold_public_benchmarks/e95_outer_edge_directed_boundary_from_e81_s8500_c256_m64/`
  and copied the launch log to ignored
  `logs/e95_outer_edge_directed_boundary_from_e81.log`. The local artifact
  pull excluded the checkpoint directory.
- Used `scripts/format_experiment_result_row.py` with `--start-after-step
  8000` to add the E95 row to `EXPERIMENT_RESULTS.md`, so inherited E81
  history does not count as E95's best validation lDDT.
- E96 launch decision: resume the E87 checkpoint from step 8500 to 9000 and
  anneal boundary-readout directionality from `0.5` down to `0.25`. E92 showed
  that holding directionality at `0.5` regresses; E96 tests whether directed
  boundary readout is better as a partial cochain-routing curriculum than as a
  fixed setting.
- Re-extracted and read both saved reference PDFs end to end for the current
  request: the TDL guide is 28 pages / 3,799 extracted words, and Topotein is
  22 pages / 10,651 extracted words. The saved copies in `references/papers/`
  hash-match the user-provided Downloads files and remain git-ignored pending
  redistribution-rights confirmation. The reading does not justify a generic
  C-alpha lDDT, radius, or all-pairs distance loss; it supports E96 as a
  directed-incidence cochain-routing curriculum and points to delayed
  edge-centric scalarization or latent selected segment cochains as the next
  paper-aligned architecture branches if E96 regresses.
- Remote E96 preparation on owned pod `o1dy17ouv8w5mz`: the pod checkout was
  dirty from earlier tracked source/doc syncs at commit `60c8213`, so I
  preserved that state in stash
  `codex-pre-e96-remote-dirty-state-20260512T231948Z`, fast-forwarded to
  pushed commit `962be73`, and confirmed the remote status was clean before
  launch.
- E96 prelaunch checks on owned pod `o1dy17ouv8w5mz`: no active Python
  benchmark process, E87 checkpoint present, remote py_compile passed for
  `scripts/run_nanofold_public_benchmarks.py`, `minalphafold/simplex.py`,
  `minalphafold/model_config.py`, and `minalphafold/trainer.py`; CLI help
  confirmed support for boundary-readout directionality runtime flags and
  `--max-parameters`; and the exact E96 instantiated module set counted
  `3,154,242` parameters, under the AF2-medium +5% ceiling `3,261,974`.
- E96 launched on owned pod `o1dy17ouv8w5mz` as
  `e96_anneal_directed_boundary_from_e87_s9000_c256_m64`, log path
  `/workspace/SimplexFold/logs/e96_anneal_directed_boundary_from_e87.log`,
  artifact path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e96_anneal_directed_boundary_from_e87_s9000_c256_m64/`,
  and Python PID `13303`. Startup poll confirmed metadata exists,
  `--max-parameters 3261974` is recorded, boundary-readout directionality
  ramps `0.5 -> 0.25` from step 8500 over 500 steps, fixed face/tetra caps
  are `24/48`, degree penalty is `0.75`, and the runner resumed E87 at step
  8500/examples 68000 with 1244 matching model tensors loaded and 0
  new/missing tensors initialized.
- E96 live health check at `2026-05-12T23:24:40Z` on owned pod
  `o1dy17ouv8w5mz`: Python PID `13303` was alive after about 4 minutes,
  `results.json` was not present yet, GPU memory was allocated at about
  `11,721 MiB`, GPU utilization sampled at `5%`, and the log still showed the
  clean E87 resume with 1244 matching tensors loaded and 0 new/missing tensors
  initialized.
- Next-fallback parameter audit while E96 runs: the E96/E87-style sparse
  edge-frame setup has `3,154,242` parameters; adding
  `simplex_cell_score_outer_edge_weight=0.25` remains `3,154,242`; adding
  latent segment cells without edge-frame modules is within budget at
  `3,234,450`; but segment cells plus edge-frame modules exceed the
  `3,261,974` cap at `3,282,002` for `simplex_c_segment=12` and `3,276,786`
  even for `simplex_c_segment=4`. Therefore, if E96 regresses, the immediate
  paper-aligned fallback should be a zero-parameter outer-edge-supported cell
  scorer, not segment cells stacked onto edge-frame messages.
- E96 health check at `2026-05-12T23:50:43Z` on owned pod
  `o1dy17ouv8w5mz`: Python PID `13303` was alive after about 30 minutes with
  about 5h16m accumulated CPU time. `results.json` and the new latest
  checkpoint were not present yet, and the log/history file mtimes remained
  at the clean E87 resume point. This is still consistent with a quiet
  training slice rather than a failed run: GPU memory was allocated at about
  `13,513 MiB`, and five GPU samples showed utilization between `35%` and
  `83%`.
- E96 returned on owned pod `o1dy17ouv8w5mz`: step 9000
  `val_lddt_ca=0.4043184444308281`, FoldScore `0.38517895340919495`,
  `val_ca_drmsd=10.197308748960495`, predicted/true C-alpha radius
  `11.273331820964813 / 15.403406739234924`, `3,154,242` parameters, and
  `stopped_early=False`. The final runtime boundary-readout directionality
  was `0.25`.
- E96 interpretation: keep as the new primary-lDDT leader. Annealing directed
  boundary readout from `0.5` to `0.25` improved over E87's `0.3992` and
  avoided E92's held-`0.5` regression. It also improved E87's FoldScore and
  dRMSD, though predicted C-alpha radius remains under-expanded.
- Copied E96 returned artifacts locally under ignored
  `artifacts/nanofold_public_benchmarks/e96_anneal_directed_boundary_from_e87_s9000_c256_m64/`
  and copied the launch log to ignored
  `logs/e96_anneal_directed_boundary_from_e87.log`. The local artifact pull
  excluded the checkpoint directory.
- Used `scripts/format_experiment_result_row.py` with `--start-after-step
  8500` to add the E96 row to `EXPERIMENT_RESULTS.md`, so inherited E87
  history does not count as E96's best validation lDDT.
- E98 launch decision: continue the E96 checkpoint from step 9000 to 9500
  with boundary-readout directionality held at `0.25`, keeping the same
  selected sparse complex, degree penalty, selected-boundary realization
  losses, edge-frame message scale, and incidence-normalized transport. If
  E98 regresses below E96, pivot to the queued zero-parameter
  outer-edge-supported cell scorer.
- Remote E98 preparation on owned pod `o1dy17ouv8w5mz`: the pod checkout was
  clean at commit `962be73` and was fast-forwarded to pushed commit `7b0219a`
  before launch. No other Runpod instances were inspected or managed.
- E98 prelaunch checks on owned pod `o1dy17ouv8w5mz`: no active Python
  benchmark process, E96 checkpoint present, remote py_compile passed for
  `scripts/run_nanofold_public_benchmarks.py`, `minalphafold/simplex.py`,
  `minalphafold/model_config.py`, and `minalphafold/trainer.py`; CLI help
  confirmed support for boundary-readout directionality runtime flags and
  `--max-parameters`; and the exact E98 instantiated module set counted
  `3,154,242` parameters, under the AF2-medium +5% ceiling `3,261,974`.
- E98 launched on owned pod `o1dy17ouv8w5mz` as
  `e98_continue_partial_directed_boundary_from_e96_s9500_c256_m64`, log path
  `/workspace/SimplexFold/logs/e98_continue_partial_directed_boundary_from_e96.log`,
  artifact path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e98_continue_partial_directed_boundary_from_e96_s9500_c256_m64/`,
  and Python PID `14318`. Startup poll confirmed metadata exists,
  `--max-parameters 3261974` is recorded, boundary-readout directionality and
  runtime scale are both held at `0.25`, fixed face/tetra caps are `24/48`,
  degree penalty is `0.75`, edge-frame runtime scale is `0.0125`, and the
  runner resumed E96 at step 9000/examples 72000 with 1244 matching model
  tensors loaded and 0 new/missing tensors initialized.
- Retargeted the existing heartbeat automation `check-simplexfold-e57-runpod`
  to E98, keeping the same owned-pod-only restriction and the rule that the
  heartbeat must not launch follow-up experiments automatically.
- E98 live health check at `2026-05-13T00:24:36Z` on owned pod
  `o1dy17ouv8w5mz`: Python PID `14318` was alive after about 2.5 minutes,
  `results.json` was not present yet, GPU memory was allocated at about
  `11,721 MiB`, GPU utilization sampled at `51%`, and the log still showed
  the clean E96 resume with 1244 matching tensors loaded and 0 new/missing
  tensors initialized.
- E98 live health check at `2026-05-13T00:42:29Z` on owned pod
  `o1dy17ouv8w5mz`: Python PID `14318` was alive after about 20 minutes,
  `results.json` was not present yet, GPU memory was allocated at about
  `13,513 MiB`, and the log still showed only the clean E96 resume. This
  matches the quiet training-window behavior seen in earlier continuations.
- E98 deeper health check at `2026-05-13T01:13:10Z` on owned pod
  `o1dy17ouv8w5mz`: Python PID `14318` was alive after about 51 minutes and
  still consuming about 10 CPU cores. `run_metadata.json` and
  `history_full_msa_to_face.json` were present, `results.json` was still
  absent, and the log/history mtimes remained at the clean resume point. Five
  GPU samples showed memory fixed at about `13,513 MiB` with utilization
  `30%`, `57%`, `42%`, `0%`, and `48%`, so the run still appears active
  rather than failed.
- E98 returned on owned pod `o1dy17ouv8w5mz`: step 9500
  `val_lddt_ca=0.3938809931278229`, FoldScore `0.38065901957452297`,
  `val_ca_drmsd=10.045937269926071`, predicted/true C-alpha radius
  `11.586045235395432 / 15.403406739234924`, `3,154,242` parameters, and
  `stopped_early=False`. The fixed runtime boundary-readout directionality
  was `0.25`.
- E98 topology diagnostics: selected face/tetra boundary lDDT
  `0.7355027459561825` / `0.7192892879247665`, boundary length MAE
  `1.0789909288287163` / `1.1812388822436333`, contraction fractions
  `0.5656210742890835` / `0.5662743374705315`, boundary-edge mean degree
  `11.799788415431976` / `33.75571095943451`, and boundary unique-edge
  fraction `0.08504440410307566` / `0.029881862387672056`.
- E98 interpretation: reject as a primary-lDDT branch. Holding the partial
  directed boundary readout at `0.25` improved dRMSD and radius relative to
  E96, but primary lDDT fell well below E96's `0.4043184444308281` and
  FoldScore softened. The selected complex did not collapse, so pivot to the
  queued E97 outer-edge-supported cell scorer that changes face/tetra
  selection rather than continuing boundary-readout pressure.
- Copied E98 returned artifacts locally under ignored
  `artifacts/nanofold_public_benchmarks/e98_continue_partial_directed_boundary_from_e96_s9500_c256_m64/`
  and copied the launch log to ignored
  `logs/e98_continue_partial_directed_boundary_from_e96.log`. The local
  artifact pull excluded the checkpoint directory.
- Used `scripts/format_experiment_result_row.py` with `--start-after-step
  9000` to add the E98 row to `EXPERIMENT_RESULTS.md`, so inherited E96
  history does not count as E98's best validation lDDT.
- E97 launch decision: resume the E96 checkpoint from step 9000 to 9500,
  ramp `simplex_cell_score_outer_edge_weight` from `0.0` to `0.25`, and ramp
  boundary-readout directionality from `0.25` to `0.0`. This is a handoff
  from E96's directed boundary-readout curriculum toward selected-complex
  construction, testing which face/tetra cochains exist rather than adding
  another output-side coordinate objective.
- Remote E97 preparation on owned pod `o1dy17ouv8w5mz`: the pod checkout was
  clean at commit `7b0219a` and was fast-forwarded to pushed commit
  `c34608e` before launch. No other Runpod instances were inspected or
  managed.
- E97 prelaunch checks on owned pod `o1dy17ouv8w5mz`: no active Python
  benchmark process, E96 checkpoint present, remote py_compile passed for
  `scripts/run_nanofold_public_benchmarks.py`, `minalphafold/simplex.py`,
  `minalphafold/model_config.py`, and `minalphafold/trainer.py`; CLI help
  confirmed support for outer-edge cell scoring, boundary-readout
  directionality runtime flags, and `--max-parameters`; and the exact E97
  instantiated module set counted `3,154,242` parameters, under the
  AF2-medium +5% ceiling `3,261,974`.
- E97 launched on owned pod `o1dy17ouv8w5mz` as
  `e97_outer_edge_score_handoff_from_e96_s9500_c256_m64`, log path
  `/workspace/SimplexFold/logs/e97_outer_edge_score_handoff_from_e96.log`,
  artifact path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e97_outer_edge_score_handoff_from_e96_s9500_c256_m64/`,
  and Python PID `15048`. Startup poll confirmed metadata exists,
  `--max-parameters 3261974` is recorded, outer-edge cell scoring ramps
  `0.0 -> 0.25` over steps 9000-9500, boundary-readout directionality ramps
  `0.25 -> 0.0` over steps 9000-9500, fixed face/tetra caps are `24/48`,
  degree penalty is `0.75`, edge-frame runtime scale is `0.0125`, and the
  runner resumed E96 at step 9000/examples 72000 with 1244 matching model
  tensors loaded and 0 new/missing tensors initialized.
- Retargeted the existing heartbeat automation `check-simplexfold-e57-runpod`
  to E97, keeping the same owned-pod-only restriction and the rule that the
  heartbeat must not launch follow-up experiments automatically.
- E97 returned on owned pod `o1dy17ouv8w5mz`: step 9500
  `val_lddt_ca=0.4035918414592743`, FoldScore `0.38672325760126114`,
  `val_ca_drmsd=9.74921926856041`, predicted/true C-alpha radius
  `11.79511296749115 / 15.403406739234924`, `3,154,242` parameters, and
  `stopped_early=False`. The schedule ramped outer-edge cell scoring from
  `0.0` to `0.25` and boundary-readout directionality from `0.25` to `0.0`.
- E97 topology diagnostics: selected face/tetra boundary lDDT
  `0.7487891465425491` / `0.7318084016442299`, contraction fractions
  `0.5646614767611027` / `0.5655763633549213`, and face/tetra outer-edge
  active fractions `1.0` / `1.0`.
- E97 interpretation: near-keep but not a new primary-lDDT leader. It fell
  just below E96's `0.4043184444308281` primary lDDT, but recovered almost all
  of E96's local lDDT after E98's regression and improved FoldScore, dRMSD,
  predicted radius, and selected-boundary lDDT. This supports the
  topology-construction handoff as stabilizing, but not as evidence that the
  current branch is likely to reach `0.7` by 30,000 steps.
- Copied E97 returned artifacts locally under ignored
  `artifacts/nanofold_public_benchmarks/e97_outer_edge_score_handoff_from_e96_s9500_c256_m64/`
  and copied the launch log to ignored
  `logs/e97_outer_edge_score_handoff_from_e96.log`. The local artifact pull
  excluded the checkpoint directory.
- Used `scripts/format_experiment_result_row.py` with `--start-after-step
  9000` to add the E97 row to `EXPERIMENT_RESULTS.md`, so inherited E96
  history does not count as E97's best validation lDDT.
- E99 launch decision: run a diagnostic continuation of the E97 checkpoint
  from step 9500 to 10500 with the E97 final topology settings fixed:
  outer-edge-supported cell scoring at `0.25`, boundary-readout directionality
  runtime scale at `0.0`, fixed `24/48` sparse caps, degree penalty `0.75`,
  selected-boundary realization losses, edge-frame runtime scale `0.0125`,
  and incidence-normalized transport. This is not a 30,000-step confirmation;
  it tests whether the topology-construction handoff has any slope past
  10,000 steps before spending on a long run.
- Remote E99 preparation on owned pod `o1dy17ouv8w5mz`: the pod checkout was
  clean at commit `c34608e` and was fast-forwarded to pushed commit
  `a77ec81` before launch. No other Runpod instances were inspected or
  managed.
- E99 prelaunch checks on owned pod `o1dy17ouv8w5mz`: no active Python
  benchmark process, E97 checkpoint present, remote py_compile passed for
  `scripts/run_nanofold_public_benchmarks.py`, `minalphafold/simplex.py`,
  `minalphafold/model_config.py`, and `minalphafold/trainer.py`; CLI help
  confirmed support for outer-edge cell scoring, boundary-readout
  directionality runtime flags, and `--max-parameters`; and the exact E99
  instantiated module set counted `3,154,242` parameters, under the
  AF2-medium +5% ceiling `3,261,974`.
- E99 launched on owned pod `o1dy17ouv8w5mz` as
  `e99_e97_continuation_s10500_c256_m64`, log path
  `/workspace/SimplexFold/logs/e99_e97_continuation.log`, artifact path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e99_e97_continuation_s10500_c256_m64/`,
  and Python PID `15625`. Startup poll confirmed metadata exists,
  `--max-parameters 3261974` is recorded, fixed outer-edge cell scoring is
  `0.25`, boundary-readout directionality and runtime scale are both `0.0`,
  fixed face/tetra caps are `24/48`, degree penalty is `0.75`, edge-frame
  runtime scale is `0.0125`, and the runner resumed E97 at step
  9500/examples 76000 with 1244 matching model tensors loaded and 0
  new/missing tensors initialized.
- Retargeted the existing heartbeat automation `check-simplexfold-e57-runpod`
  to E99, keeping the same owned-pod-only restriction and the rule that the
  heartbeat must not launch follow-up experiments automatically.
- E99 mid-run check on owned pod `o1dy17ouv8w5mz`: step 10000 reached
  `val_lddt_ca=0.3972`, FoldScore `0.3872`, `val_ca_drmsd=9.8867`, and
  predicted/true C-alpha radius `12.1938 / 15.4034`. This crossed the 10k
  line but did not show the primary-lDDT slope needed for a 30k spend.
- E99 returned on owned pod `o1dy17ouv8w5mz`: step 10500
  `val_lddt_ca=0.4002871196717024`, FoldScore `0.38573700562119484`,
  `val_ca_drmsd=10.15071052312851`, predicted/true C-alpha radius
  `11.38073205947876 / 15.403406739234924`, `3,154,242` parameters, and
  `stopped_early=False`. Fixed outer-edge cell scoring was `0.25`, and fixed
  boundary-readout directionality runtime scale was `0.0`.
- E99 topology diagnostics: selected face/tetra boundary lDDT
  `0.7573754265904427` / `0.7386169098317623`, contraction fractions
  `0.528986806049943` / `0.5271297451108694`, and face/tetra outer-edge
  active fractions `1.0` / `1.0`.
- E99 interpretation: reject as a continuation branch and as a 30k-spend
  signal. The selected complex continued to become cleaner, but primary
  C-alpha lDDT stayed below E96/E97 at both step 10000 and step 10500. The
  next experiment should alter how selected higher-rank states affect the
  residue/pair trunk instead of continuing the same lineage longer.
- Copied E99 returned artifacts locally under ignored
  `artifacts/nanofold_public_benchmarks/e99_e97_continuation_s10500_c256_m64/`
  and copied the launch log to ignored `logs/e99_e97_continuation.log`. The
  local artifact pull excluded the checkpoint directory.
- Used `scripts/format_experiment_result_row.py` with `--start-after-step
  9500` to add the E99 row to `EXPERIMENT_RESULTS.md`, so inherited E97
  history does not count as E99's best validation lDDT.

## 2026-05-13 Bidirectional Simplex-MSA Feedback

- E99 makes a blind continuation unattractive: it crossed 10,000 steps but
  ended below E96/E97 on primary C-alpha lDDT while selected face/tetra
  boundary lDDT kept improving. The next change should make selected
  higher-rank cochains affect the trunk more directly instead of adding
  another output-side loss.
- Prepared E100 locally: added `simplex_msa_feedback_scale`, a runtime ramp
  override, and benchmark-runner plumbing. The path projects the selected
  face/tetra-to-residue 0-simplex readout from `c_s` to `c_m` and adds it to
  the target MSA row in `SimplicialEvoformer`. This completes the missing
  reverse direction in the README's `M <-> Z <-> F <-> U` schematic.
- Local targeted validation passed:
  `python -m py_compile minalphafold/simplex.py minalphafold/evoformer.py minalphafold/model.py minalphafold/model_config.py minalphafold/trainer.py scripts/run_nanofold_public_benchmarks.py`;
  `python -m pytest tests/test_simplex.py::test_simplicial_adapter_can_project_selected_cell_readout_to_msa_feedback tests/test_simplex.py::test_simplicial_evoformer_msa_feedback_updates_target_msa_row tests/test_trainer.py::test_simplicial_runtime_overrides_reach_model_path tests/test_trainer.py::test_simplicial_msa_feedback_stays_within_medium_budget tests/test_nanofold_public_benchmarks.py::test_model_config_override_flags_are_accepted_by_cli_parser tests/test_nanofold_public_benchmarks.py::test_runtime_simplex_message_scales_ramp_and_enter_model_inputs tests/test_nanofold_public_benchmarks.py::test_evaluate_uses_runtime_simplex_overrides_for_validation`
  reported `7 passed`.
- Broader focused validation passed: `python -m pytest tests/test_simplex.py
  tests/test_nanofold_public_benchmarks.py tests/test_trainer.py` reported
  `165 passed`, and `git diff --check` was clean.
- Explicit parameter audit for the E100-style module set counted
  `3,225,090` parameters versus AF2-medium `3,106,642`, leaving `36,884`
  parameters under the +5% ceiling `3,261,974`.
- Intended E100 gate: resume E97 from step 9500 to 10000, allocate
  `--simplex-msa-feedback-scale 0.05`, ramp
  `--simplex-msa-feedback-runtime-scale 0.0 -> 0.05` over steps 9500-10000,
  keep the E97 final topology recipe fixed, and compare against E99's
  step-10000 control (`val_lddt_ca=0.3972`) before spending on any longer run.
- Remote E100 preparation on owned pod `o1dy17ouv8w5mz`: the pod checkout was
  clean at commit `a77ec81` and was fast-forwarded to pushed commit
  `7530c12`. No other Runpod instances were inspected or managed.
- E100 prelaunch checks on owned pod `o1dy17ouv8w5mz`: no active benchmark
  process was running, E97 checkpoint was present, remote py_compile passed
  for `minalphafold/simplex.py`, `minalphafold/evoformer.py`,
  `minalphafold/model.py`, `minalphafold/model_config.py`,
  `minalphafold/trainer.py`, and
  `scripts/run_nanofold_public_benchmarks.py`; CLI help confirmed support for
  MSA-feedback runtime flags and `--max-parameters`; and the exact E100
  module set counted `3,225,090` parameters, under the AF2-medium +5% ceiling
  `3,261,974`.
- E100 launched on owned pod `o1dy17ouv8w5mz` as
  `e100_msa_feedback_from_e97_s10000_c256_m64`, log path
  `/workspace/SimplexFold/logs/e100_msa_feedback_from_e97.log`, artifact path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e100_msa_feedback_from_e97_s10000_c256_m64/`,
  and Python PID `16247`. Startup poll confirmed the process is active and
  the runner resumed E97 at step 9500/examples 76000, loaded 1244 matching
  model tensors, initialized 24 new/missing feedback tensors, and started a
  fresh optimizer.
- Retargeted the existing heartbeat automation `check-simplexfold-e57-runpod`
  to E100, keeping the owned-pod-only restriction and the rule that the
  heartbeat must not launch follow-up experiments automatically.
- E100 live health check at `2026-05-13T05:23:33Z` on owned pod
  `o1dy17ouv8w5mz`: Python PID `16247` was alive after about 2.5 minutes,
  `results.json` was not present yet, and the log still showed the clean E97
  resume with 1244 matching tensors loaded and 24 new/missing feedback tensors
  initialized. This matches the expected quiet training window before the
  step-10000 validation.
- While E100 runs, inspected the remaining simplex communication routes and
  recorded a contingency idea in `EXPERIMENTS.md`: boundary-edge coboundary
  MSA feedback. This would aggregate selected 1-cochain boundary-edge
  messages into incident residues before target-MSA feedback. It is an idea
  only; do not implement or launch while E100 is active.
- E100 live health check at `2026-05-13T05:36:20Z` on owned pod
  `o1dy17ouv8w5mz`: Python PID `16247` was still active after about 15
  minutes, H100 utilization was nonzero (`30%`) with `13.5GB` allocated, and
  `run_metadata.json` plus `history_full_msa_to_face.json` were present.
  `results.json` and step-10000 checkpoints were not present yet, so the run
  remains in progress rather than returned.
- E100 returned on owned pod `o1dy17ouv8w5mz` at step 10000 with effective
  batch size `8`, `3,225,090` parameters, `stopped_early=False`, and
  `val_lddt_ca=0.3935649264603853`. FoldScore was `0.3887240868061781`,
  `val_ca_drmsd=9.969625651836395`, and predicted/true C-alpha radius was
  `11.83768093585968 / 15.403406739234924`.
- E100 topology diagnostics stayed locally strong despite the worse primary
  target: selected face/tetra boundary lDDT was
  `0.7479618825018406` / `0.731723640114069`, and selected face/tetra
  boundary contraction fraction was `0.6223886236548424` /
  `0.6210166253149509`.
- E100 interpretation: reject. The cell-to-residue MSA feedback closed a
  missing route in the README schematic, but it dropped below the E99
  step-10000 control (`0.3972`) and below E96/E97. The next candidate should
  preserve the selected boundary-edge 1-cochain longer by using a
  boundary-edge-to-residue coboundary feedback path rather than feeding a
  collapsed face/tetra residue summary into MSA.
- Copied E100 returned artifacts locally under ignored
  `artifacts/nanofold_public_benchmarks/e100_msa_feedback_from_e97_s10000_c256_m64/`
  and copied the launch log to ignored `logs/e100_msa_feedback_from_e97.log`.
  The local artifact pull excluded the checkpoint directory.
- Used `scripts/format_experiment_result_row.py` with `--start-after-step
  9500` to add the E100 row to `EXPERIMENT_RESULTS.md`, so inherited E97
  history does not count as E100's best validation lDDT.
- Implemented E101 locally as `simplex_boundary_msa_feedback_scale`. The path
  scatters selected boundary-edge updates into source-residue and
  target-residue channels, concatenates the directed 1-cochain-to-0-cochain
  residue summaries, projects them from `2 * c_z` to `c_m`, and feeds them
  through the existing target-MSA feedback hook.
- E101 is topology-native by construction: it does not add an output metric
  loss and does not supervise all residue pairs. It preserves the selected
  face/tetra boundary 1-skeleton longer than E100 before communicating with
  the MSA trunk.
- E101 validation passed locally:
  `python -m py_compile minalphafold/simplex.py minalphafold/model_config.py scripts/run_nanofold_public_benchmarks.py`;
  targeted E101/E100 tests reported `7 passed`; and
  `python -m pytest tests/test_simplex.py tests/test_nanofold_public_benchmarks.py tests/test_trainer.py`
  reported `168 passed`.
- Exact E101 launch-module parameter audit with E97 edge-frame messages
  allocated counted `3,206,722` parameters versus AF2-medium `3,106,642`,
  leaving `55,252` under the +5% ceiling `3,261,974`.
- Remote E101 preparation on owned pod `o1dy17ouv8w5mz`: the pod checkout
  fast-forwarded to commit `9a28688`, remote py_compile passed for
  `minalphafold/simplex.py`, `minalphafold/model_config.py`, and
  `scripts/run_nanofold_public_benchmarks.py`; CLI help confirmed support for
  `--simplex-boundary-msa-feedback-scale`; the E97 checkpoint was present; and
  the exact E101 module set counted `3,206,722` parameters, under the
  AF2-medium +5% ceiling `3,261,974`.
- E101 launched on owned pod `o1dy17ouv8w5mz` as
  `e101_boundary_msa_feedback_from_e97_s10000_c256_m64`, log path
  `/workspace/SimplexFold/logs/e101_boundary_msa_feedback_from_e97.log`,
  artifact path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e101_boundary_msa_feedback_from_e97_s10000_c256_m64/`,
  and Python PID `17228`. Startup poll confirmed the process is active and
  the runner resumed E97 at step 9500/examples 76000, loaded 1244 matching
  model tensors, initialized 24 new/missing boundary-feedback tensors, and
  started a fresh optimizer.
- Retargeted the existing heartbeat automation `check-simplexfold-e57-runpod`
  to E101, keeping the owned-pod-only restriction and the rule that the
  heartbeat must not launch follow-up experiments automatically.
- E101 returned on owned pod `o1dy17ouv8w5mz` at step 10000 with effective
  batch size `8`, `3,206,722` parameters, `stopped_early=False`, and
  `val_lddt_ca=0.39977212622761726`. FoldScore was
  `0.3867220859974623`, `val_ca_drmsd=9.934351950883865`, and
  predicted/true C-alpha radius was
  `11.70958662033081 / 15.403406739234924`.
- E101 topology diagnostics remained strong: selected face/tetra boundary
  lDDT was `0.7555434443056583` / `0.7378766611218452`, boundary length MAE
  was `1.0114886984229088` / `1.1146406307816505`, and selected face/tetra
  contraction fraction was `0.6162960529327393` /
  `0.6145038418471813`.
- E101 interpretation: reject as a 30,000-step spend candidate. It improves
  over E100 and the E99 step-10000 control, but remains below E99 final,
  E97, and E96. Boundary-edge coboundary feedback is better justified than
  E100's collapsed cell-summary feedback, but target-MSA feedback does not
  look like the route that will break the current `0.40` C-alpha lDDT band.
- Copied E101 returned artifacts locally under ignored
  `artifacts/nanofold_public_benchmarks/e101_boundary_msa_feedback_from_e97_s10000_c256_m64/`
  and copied the launch log to ignored
  `logs/e101_boundary_msa_feedback_from_e97.log`. The local artifact pull
  excluded checkpoint directories.
- Used `scripts/format_experiment_result_row.py` with `--start-after-step
  9500` to add the E101 row to `EXPERIMENT_RESULTS.md`, so inherited E97
  history does not count as E101's best validation lDDT.
- 30,000-step candidate assessment after E101: no current branch has earned
  that spend. E96/E97 are the best available lineage, but E99/E100/E101 give
  three near-10k controls that stay near `0.40`; reaching `0.7` by 30,000
  would require a much stronger late-training lDDT slope than any current
  continuation has shown.
- Implemented E102 locally as `simplex_boundary_pair_feedback_scale`. The
  route aggregates selected directed boundary-edge updates into outgoing and
  incoming residue cochains, lifts those endpoint summaries back to pair
  space as `[Z_ij, outgoing_i, incoming_j]`, and applies a learned residual
  pair update. This keeps the feedback target in the pair/edge trunk rather
  than the target MSA row.
- E102 is topology-native: it routes explicit face/tetra cell evidence
  through the selected boundary 1-skeleton into `Z_ij`. It does not add an
  output lDDT/radius loss or any supervision outside the model-selected sparse
  complex.
- E102 validation so far:
  `python -m py_compile minalphafold/simplex.py minalphafold/evoformer.py minalphafold/model.py minalphafold/model_config.py minalphafold/trainer.py scripts/run_nanofold_public_benchmarks.py`;
  targeted tests for the pair-feedback route, CLI plumbing, runtime overrides,
  and parameter budget reported `7 passed`.
- Broader local E102 validation:
  `python -m pytest tests/test_simplex.py tests/test_nanofold_public_benchmarks.py tests/test_trainer.py`
  reported `170 passed`; `git diff --check` passed.
- Exact E102 launch-module parameter audit with E97 settings plus boundary
  pair feedback counted `3,206,882` parameters versus AF2-medium `3,106,642`,
  leaving `55,092` under the +5% ceiling `3,261,974`.
- Intended E102 gate: resume E97 from step 9500 to 10000, keep E97 topology
  settings fixed, allocate `--simplex-boundary-pair-feedback-scale 0.05`, and
  ramp `--simplex-boundary-pair-feedback-runtime-scale 0.0` to `0.025` over
  steps 9500-10000. Keep E100/E101 MSA-feedback routes disabled and compare
  against E99 step 10000, E101, E99 final, E97, and E96.
- Remote E102 preparation on owned pod `o1dy17ouv8w5mz`: the pod checkout
  fast-forwarded to commit `c73b151`, remote py_compile passed for the
  runner/model files, CLI help confirmed support for
  `--simplex-boundary-pair-feedback-*`, the E97 checkpoint was present, and
  the exact E102 module set counted `3,206,882` parameters, under the
  AF2-medium +5% ceiling `3,261,974`.
- E102 launched on owned pod `o1dy17ouv8w5mz` as
  `e102_boundary_pair_feedback_from_e97_s10000_c256_m64`, log path
  `/workspace/SimplexFold/logs/e102_boundary_pair_feedback_from_e97.log`,
  artifact path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e102_boundary_pair_feedback_from_e97_s10000_c256_m64/`,
  bash PID `17895`, and Python PID `17897`. Startup poll confirmed metadata
  exists, `--max-parameters 3261974` is recorded, boundary-pair feedback
  allocates at `0.05` and ramps from `0.0` to `0.025` over steps 9500-10000,
  E100/E101 MSA-feedback routes are disabled, fixed face/tetra caps are
  `24/48`, degree penalty is `0.75`, edge-frame runtime scale is `0.0125`,
  and the runner resumed E97 at step 9500/examples 76000 with 1244 matching
  model tensors loaded and 24 new/missing boundary-pair tensors initialized.
- Retargeted the existing heartbeat automation `check-simplexfold-e57-runpod`
  to E102, keeping the owned-pod-only restriction and the rule that the
  heartbeat must not launch follow-up experiments automatically.
- E102 live performance check on owned pod `o1dy17ouv8w5mz`: after about
  42 minutes, bash PID `17895` and Python PID `17897` were still active,
  GPU memory was about `13.5GB`, and GPU utilization was nonzero, but
  `results.json` was still absent. The local history file still had 20 rows
  with the last row inherited from E97 at step 9500; no new step-10000 row or
  checkpoint had landed yet.
- E102 interpretation so far: this is not a returned result and should not be
  added to `EXPERIMENT_RESULTS.md`, but the dense all-pairs boundary-cochain
  lift is much slower than the earlier sparse boundary routes. If E102 does
  not return cleanly, the next topology-native branch should keep the same
  pair-feedback target but make the route sparse by conditioning selected
  boundary-edge updates on their current pair states before scatter.
- E102 was stopped on the owned pod before any new checkpoint, history row, or
  `results.json` was produced. The remote process tree for bash PID `17895`
  and Python PID `17897` was terminated, and the log ended with `Terminated`.
  This remains an aborted performance diagnostic, not a completed result, so
  `EXPERIMENT_RESULTS.md` should continue to omit E102.
- Implemented E103 locally as `simplex_boundary_pair_gate_scale`. The sparse
  route keeps E102's pair/edge feedback target but avoids the dense `L x L`
  lift: selected face/tetra boundary-edge cochains are concatenated with their
  current pair state `Z_ab`, passed through a learned `tanh` gate, and used to
  modulate the sparse boundary-edge update before incidence normalization and
  scatter. This keeps the change inside the README's explicit selected
  face/tetra complex and its boundary 1-skeleton.
- E103 validation so far:
  `python -m py_compile minalphafold/simplex.py minalphafold/evoformer.py minalphafold/model.py minalphafold/model_config.py minalphafold/trainer.py scripts/run_nanofold_public_benchmarks.py`;
  targeted tests for sparse gate behavior, CLI plumbing, runtime overrides,
  validation propagation, and parameter budget reported `7 passed`.
- Broader local E103 validation:
  `python -m pytest tests/test_simplex.py tests/test_nanofold_public_benchmarks.py tests/test_trainer.py`
  reported `172 passed`.
- Exact E103 launch-module parameter audit with E97 settings plus sparse
  boundary pair gate counted `3,193,762` parameters versus AF2-medium
  `3,106,642`, leaving `68,212` under the +5% ceiling `3,261,974`.
- Intended E103 gate: resume E97 from step 9500 to 10000, keep E97 topology
  settings fixed, allocate `--simplex-boundary-pair-gate-scale 0.05`, and
  ramp `--simplex-boundary-pair-gate-runtime-scale 0.0` to `0.025` over
  steps 9500-10000. Keep E100/E101 MSA-feedback routes and E102 dense
  pair-feedback disabled, then compare against E99 step 10000, E101, E99
  final, E97, and E96.
- Remote E103 preparation on owned pod `o1dy17ouv8w5mz`: no active benchmark
  process was present, the pod checkout fast-forwarded to commit `cf511b8`,
  remote py_compile passed for the runner/model files, CLI help confirmed
  support for `--simplex-boundary-pair-gate-*`, the E97 checkpoint was
  present, and the exact E103 module set counted `3,193,762` parameters
  under the AF2-medium +5% ceiling `3,261,974`.
- E103 launched on owned pod `o1dy17ouv8w5mz` as
  `e103_sparse_boundary_pair_gate_from_e97_s10000_c256_m64`, log path
  `/workspace/SimplexFold/logs/e103_sparse_boundary_pair_gate_from_e97.log`,
  artifact path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e103_sparse_boundary_pair_gate_from_e97_s10000_c256_m64/`,
  and Python PID `18770`. Startup poll confirmed `run_metadata.json` exists,
  `--max-parameters 3261974` is recorded, the sparse boundary-pair gate
  allocates at `0.05` and ramps from `0.0` to `0.025` over steps 9500-10000,
  E100/E101 MSA-feedback routes are disabled, E102 dense pair feedback is
  disabled, fixed face/tetra caps are `24/48`, degree penalty is `0.75`,
  edge-frame runtime scale is `0.0125`, and the runner resumed E97 at step
  9500/examples 76000 with 1244 matching model tensors loaded and 24 new
  sparse-gate tensors initialized.
- Retargeted the existing heartbeat automation `check-simplexfold-e57-runpod`
  to E103, keeping the owned-pod-only restriction and the rule that the
  heartbeat must not launch follow-up experiments automatically.
- E103 live poll at `2026-05-13T09:44:29Z` on owned pod `o1dy17ouv8w5mz`:
  Python PID `18770` was still active after about 37 minutes with GPU memory
  at `14259 MiB`, but `results.json` was absent and the artifact directory
  still contained only `run_metadata.json` and the inherited
  `history_full_msa_to_face.json`. The history remained at 20 rows ending
  with E97 step 9500 (`val_lddt_ca=0.4035918414592743`), so E103 has not
  returned a result and must remain out of `EXPERIMENT_RESULTS.md` for now.
- Implemented E104 locally as `simplex_boundary_metric_gate_scale`. The gate
  reuses the existing selected face/tetra distance heads: per-boundary-edge
  distance-logit entropy becomes a metric-confidence cochain, confident
  selected boundary edges strengthen their pair-trunk transport, and uncertain
  selected boundary edges are damped before incidence normalization/scatter.
  This is topology-native because it acts only on boundary edges induced by
  explicit model-selected 2-/3-cells and uses the complex's own metric heads;
  it adds no parameters and no output-side lDDT/radius loss.
- E104 validation so far:
  `python -m py_compile minalphafold/simplex.py minalphafold/evoformer.py minalphafold/model.py minalphafold/model_config.py minalphafold/trainer.py scripts/run_nanofold_public_benchmarks.py tests/test_simplex.py tests/test_trainer.py tests/test_nanofold_public_benchmarks.py`;
  targeted tests for boundary-metric confidence, adapter gating, CLI/config
  plumbing, runtime override propagation, validation propagation, and
  no-parameter budget behavior reported `8 passed`.
- Broader local E104 validation:
  `python -m pytest tests/test_simplex.py tests/test_nanofold_public_benchmarks.py tests/test_trainer.py`
  reported `175 passed`; `git diff --check` passed.
- Intended E104 gate: do not launch while E103 is active. If E103 returns as
  a reject or is stopped as a confirmed performance failure, resume the
  strongest E96/E97-family checkpoint for a 500-step gate with the E97 sparse
  complex recipe fixed and ramp
  `--simplex-boundary-metric-gate-runtime-scale 0.0` to `0.25`. Compare
  against E99 step 10000, E101, E99 final, E97, and E96; reject unless the
  metric-confidence gate recovers primary C-alpha lDDT toward or above the
  E96/E97 peak without selected-boundary diagnostic collapse.
- E103 live poll after E104 validation: Python PID `18770` remained active
  after about 51 minutes with GPU memory at `14259 MiB` and nonzero
  utilization, but `results.json` was still absent and the history still
  ended at the inherited E97 step-9500 row. This remains an in-flight run,
  not a result.
- E103 returned on owned pod `o1dy17ouv8w5mz`: remote coherence check found
  `results.json`, `results.csv`, eval details, and a new history row at step
  10000. The returned row has effective batch size `8`, `3,193,762`
  parameters, `stopped_early=False`, `val_lddt_ca=0.3980565518140793`,
  FoldScore `0.3909183647483587`, `val_ca_drmsd=9.827525943517685`, and
  predicted/true C-alpha radius `12.048258155584335 / 15.403406739234924`.
- Copied E103 returned artifacts locally under ignored
  `artifacts/nanofold_public_benchmarks/e103_sparse_boundary_pair_gate_from_e97_s10000_c256_m64/`
  and copied the launch log to ignored
  `logs/e103_sparse_boundary_pair_gate_from_e97.log`. The local artifact pull
  excluded checkpoint directories and passed a local JSON/history coherence
  check.
- Added the E103 row to `EXPERIMENT_RESULTS.md` with
  `--start-after-step 9500`, so inherited E97 history does not count as
  E103's best validation lDDT.
- E103 interpretation: reject as a 30k-spend branch. Sparse pair-gated
  boundary cochains improved FoldScore and dRMSD, but primary C-alpha lDDT
  fell below E96, E97, E99 final, and E101. This keeps the plateau diagnosis
  intact and makes E104's no-new-parameter metric-confidence gate the next
  cleaner topology-native probe.
- Remote E104 preparation on owned pod `o1dy17ouv8w5mz`: no active benchmark
  process was present, the pod checkout fast-forwarded to commit `b9aaf9a`,
  remote py_compile passed for the runner/model files, CLI help confirmed
  support for `--simplex-boundary-metric-gate-*`, the E97 checkpoint was
  present, and the exact E104 module set counted `3,154,242` parameters under
  the AF2-medium +5% ceiling `3,261,974`.
- E104 launched on owned pod `o1dy17ouv8w5mz` as
  `e104_boundary_metric_gate_from_e97_s10000_c256_m64`, log path
  `/workspace/SimplexFold/logs/e104_boundary_metric_gate_from_e97.log`,
  artifact path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e104_boundary_metric_gate_from_e97_s10000_c256_m64/`,
  and Python PID `19749`. The run resumes E97 at step 9500, keeps E97's
  sparse selected-complex recipe fixed, disables E100/E101 MSA feedback,
  disables E102 dense pair feedback, disables E103 learned pair gating, and
  ramps `--simplex-boundary-metric-gate-runtime-scale` from `0.0` to `0.25`
  across steps 9500-10000.
- Retargeted the existing heartbeat automation `check-simplexfold-e57-runpod`
  to E104, keeping the owned-pod-only restriction and the rule that the
  heartbeat must not launch follow-up experiments automatically.
- E104 live poll at `2026-05-13T10:19:03Z` on owned pod `o1dy17ouv8w5mz`:
  Python PID `19749` was active with GPU memory at `12573 MiB`; `results.json`
  was absent; the artifact directory contained only `run_metadata.json` and
  the inherited `history_full_msa_to_face.json`; and the history remained at
  20 rows ending with E97 step 9500 (`val_lddt_ca=0.4035918414592743`). E104
  is still in flight and must remain out of `EXPERIMENT_RESULTS.md` until it
  returns and passes remote/local coherence checks.
- E104 live poll at `2026-05-13T10:28:32Z`: Python PID `19749` was still
  active after about 14 minutes with GPU memory at `14365 MiB`, nonzero GPU
  utilization, and `results.json` absent. The log still only shows startup and
  resume lines, so E104 remains in flight.
- Implemented E105 locally as `simplex_boundary_metric_recycling_scale`. The
  path converts selected face/tetra boundary distance logits into AF2 recycling
  distance-bin evidence, scatters that evidence only over selected boundary
  edges, masks inactive pairs to zero, and reuses the existing
  `recycle_linear_d` projection to bias `z_prev` for the next recycling cycle.
  This is a topology-native inter-cycle cochain-memory change and adds no
  parameters.
- E105 validation so far:
  `python -m py_compile minalphafold/simplex.py minalphafold/model.py minalphafold/model_config.py scripts/run_nanofold_public_benchmarks.py tests/test_simplex.py tests/test_trainer.py tests/test_nanofold_public_benchmarks.py`;
  focused tests for sparse recycling-bin scatter, no-new-parameter budget
  behavior, cycle-specific forward behavior, and CLI/config override plumbing
  reported `4 passed`; the broader local slice
  `python -m pytest tests/test_simplex.py tests/test_nanofold_public_benchmarks.py tests/test_trainer.py`
  reported `178 passed`; `git diff --check` passed. A launch-style E105
  parameter audit with E97 topology settings and
  `simplex_boundary_metric_recycling_scale=0.1` counted `3,154,242`
  parameters, under the AF2-medium +5% ceiling.
- E104 live poll at `2026-05-13T10:36Z`: Python PID `19749` was still active
  after about 22 minutes on the owned pod, GPU memory remained allocated at
  `14365 MiB`, and `results.json` was absent. The history still ended at the
  inherited E97 step-9500 row, so E104 remains in flight and stays out of
  `EXPERIMENT_RESULTS.md`.
- E105 follow-up plumbing: add a runtime schedule for
  `simplex_boundary_metric_recycling_scale` before launching it. This keeps
  the same topology-native inter-cycle cochain-memory mechanism but lets a
  resumed E97-family checkpoint ramp selected boundary metric recycling from
  `0.0` to a small value over the 500-step gate instead of switching it on
  abruptly.
- E105 metric-cochain refinement: change the selected boundary metric
  recycling projection from hard expected-distance binning to a soft projection
  of the full face/tetra distance distribution into the AF2 recycling-bin
  basis. This keeps uncertainty in the metric cochain memory instead of
  collapsing every selected boundary edge to one nearest bin.
- E104 live poll at `2026-05-13T10:49:47Z`: Python PID `19749` was still
  active after about 35 minutes on owned pod `o1dy17ouv8w5mz`, with GPU
  memory at `14365 MiB`. `results.json` was still absent, the run directory
  contained only `run_metadata.json` and inherited
  `history_full_msa_to_face.json`, and the history still ended at E97 step
  9500 (`val_lddt_ca=0.4035918414592743`). Keep E104 out of
  `EXPERIMENT_RESULTS.md` until it returns and passes coherence checks.
- E105 launch recipe is now explicit in `EXPERIMENTS.md`: if E104 rejects,
  resume the E97 step-9500 checkpoint to step 10000 with E97 topology settings
  fixed and only ramp soft selected-boundary metric recycling from `0.0` to
  `0.10` over steps 9500-10000.
- E104 delayed live poll at `2026-05-13T10:57:25Z`: Python PID `19749` was
  still active after about 43 minutes, `results.json` was absent, and
  `history_full_msa_to_face.json` still had 20 rows ending at E97 step 9500
  (`val_lddt_ca=0.4035918414592743`). No returned artifact exists yet, so
  `EXPERIMENT_RESULTS.md` remains unchanged.
- E104 bounded watch from `2026-05-13T10:59:08Z` to
  `2026-05-13T11:05:32Z`: four owned-pod polls all showed PID `19749` active,
  `results.json` / `results.csv` absent, and `history_full_msa_to_face.json`
  unchanged since `2026-05-13T10:14:09Z` with 20 rows ending at E97 step 9500.
  Treat E104 as still in flight rather than returned.
- E104 returned on owned pod `o1dy17ouv8w5mz` at `2026-05-13T11:11Z` and was
  pulled locally from
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e104_boundary_metric_gate_from_e97_s10000_c256_m64/`
  into the matching local artifact directory, excluding checkpoints. Local
  coherence checks passed across `results.json`, `results.csv`, and
  `history_full_msa_to_face.json`: one returned row, completed step 10000,
  `parameters=3154242`, `val_lddt_ca=0.39564882032573223`,
  `val_foldscore=0.38540054485201836`, `val_ca_drmsd=10.1848464012146`, and
  C-alpha Rg `11.31956136226654 / 15.403406739234924`. Reject E104: the
  selected face/tetra boundary lDDT diagnostics are high
  (`0.7246271595358849` / `0.7071636132895947`), but primary C-alpha lDDT is
  below E96, E97, E99, E101, and E103, so the metric gate did not convert local
  simplex boundary quality into a better global fold.
- Stopped the owned Runpod pod `o1dy17ouv8w5mz` after the E104 artifacts were
  pulled and locally verified. Runpod reported desired status `EXITED` with
  last status change `2026-05-13T11:13:05Z`.
- Restarted the owned pod `o1dy17ouv8w5mz` at `2026-05-13T11:15:21Z` to
  prepare E105, but the zero-volume pod came back with an empty `/workspace`
  except for `.cache`; `/workspace/SimplexFold` and the E97 checkpoint were no
  longer present. Local searches found no retained E97/E96/E87/E81 checkpoint;
  the strongest retained compatible local checkpoint is E72
  (`e72_edge_frame0025_from_e71_s5500_c256_m64`, step 5500,
  `val_lddt_ca=0.37177472934126854`). Pivot the immediate recycling-memory
  probe to E105a from E72 to step 6000, and compare it against E72/E73/E74/E76
  rather than the later E96-E104 plateau family.
- E105a remote staging on owned pod `o1dy17ouv8w5mz`: SSH moved to
  `root@103.207.149.82 -p 12578` after restart. The pod was restaged from
  scratch by cloning SimplexFold branch
  `codex/simplexfold-topology-e07-boundary-coordinate` at commit `33f4776`,
  cloning nanoFold-Competition at commit `96afc84`, creating
  `/workspace/venv` with system site packages, and installing SimplexFold
  editable there. Public data staging used only `processed_features`,
  `processed_labels`, and public manifests `train.txt`, `val.txt`, and
  `all.txt`; a first tar attempt created macOS `._*` sidecars and was
  discarded, then restaged with `COPYFILE_DISABLE=1 tar --no-xattrs`. Final
  audit before launch: features/labels `11000/11000`, sidecars `0`, manifests
  `10000/1000/11000`, E72 checkpoint present at 34 MB, py_compile passed for
  runner/model files, CLI help exposed the metric-recycling flags and
  `--max-parameters`, FoldScore import worked, and the E105a module set counted
  `3,154,242` parameters under the `3,261,974` cap.
- E105a launched on owned pod `o1dy17ouv8w5mz` at `2026-05-13T11:51Z` with run
  name `e105a_boundary_metric_recycling_from_e72_s6000_c256_m64`, log
  `/workspace/SimplexFold/logs/e105a_boundary_metric_recycling_from_e72.log`,
  artifact path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e105a_boundary_metric_recycling_from_e72_s6000_c256_m64/`,
  and PID `1307`. It resumes the E72 checkpoint at step 5500/examples 44000
  with `--resume-model-weights-only`, keeps the E97-style selected-complex
  recipe fixed, and ramps selected-boundary metric recycling from `0.0` to
  `0.10` over steps 5500-6000. Startup and health poll at
  `2026-05-13T11:52:33Z` showed PID `1307` active, GPU memory `4723 MiB`, GPU
  utilization `44%`, `run_metadata.json` and inherited
  `history_full_msa_to_face.json` present, and a clean resume with `1244`
  matching tensors loaded and `0` new/missing tensors initialized.
- Retargeted heartbeat `check-simplexfold-e57-runpod` to E105a, preserving the
  owned-pod-only scope and the rule that the heartbeat must not launch
  follow-up experiments automatically.
- E105a live check at `2026-05-13T11:58:39Z`: owned pod `o1dy17ouv8w5mz`
  still had PID `1307` active after about 7 minutes. `results.json` was
  absent, `history_full_msa_to_face.json` still had 12 inherited rows ending
  at E72 step 5500 (`val_lddt_ca=0.37177472934126854`), and the log still
  showed only startup/resume lines. Keep E105a out of `EXPERIMENT_RESULTS.md`
  until a returned artifact exists and passes remote/local coherence checks.
- Implemented E106 locally as `simplex_boundary_cochain_recycling_scale`.
  This complements E105: instead of recycling only selected boundary distance
  distributions, it recycles the existing selected face/tetra boundary pair
  cochain itself into `z_prev` between AF2 recycle cycles. The simplex adapter
  exposes `simplex_structure_pair_readout` when cochain recycling is enabled,
  but direct structure readout remains off unless
  `simplex_structure_readout_scale` is set. No new parameters or losses are
  added.
- E106 validation so far:
  `/Users/christopherhayduk/Projects/nanoFold-Competition/.venv/bin/python -m pytest tests/test_trainer.py::test_simplicial_runtime_overrides_reach_model_path tests/test_trainer.py::test_model_inputs_add_training_only_simplex_curricula tests/test_trainer.py::test_simplicial_boundary_cochain_recycling_adds_no_parameters tests/test_trainer.py::test_simplicial_boundary_cochain_recycling_changes_only_recycled_cycles tests/test_nanofold_public_benchmarks.py::test_model_config_override_flags_are_accepted_by_cli_parser tests/test_nanofold_public_benchmarks.py::test_runtime_simplex_message_scales_ramp_and_enter_model_inputs tests/test_nanofold_public_benchmarks.py::test_evaluate_uses_runtime_simplex_overrides_for_validation`
  reported `7 passed`. E106 is queued behind E105a and should not be launched
  automatically while E105a is still running.
- E106 parameter audit: AF2-medium baseline remains `3,106,642` parameters and
  the +5% cap is `3,261,974`. The E106 launch-style module set with E105a
  selected-complex settings and `simplex_boundary_cochain_recycling_scale=0.10`
  counts `3,154,242` parameters, under cap.
- E105a live check at `2026-05-13T12:11:31Z`: owned pod `o1dy17ouv8w5mz`
  still had PID `1307` active after about 20 minutes. `results.json` was
  absent, `history_full_msa_to_face.json` still had 12 inherited rows ending
  at E72 step 5500 (`val_lddt_ca=0.37177472934126854`), and the log still
  showed only startup/resume lines. Continue to treat E105a as in flight.
- E105a live check at `2026-05-13T12:13:33Z`: PID `1307` was still active
  after about 22 minutes on the owned pod. `results.json` and `results.csv`
  were absent, the history still ended at E72 step 5500 with
  `val_lddt_ca=0.37177472934126854` and FoldScore `0.3721824511885643`, and
  the log still only showed startup/resume lines.
- E105a bounded watch at `2026-05-13T12:25:35Z`: PID `1307` was still active
  after about 34 minutes on the owned pod. `results.json` and `results.csv`
  were absent, and the history still ended at E72 step 5500 with
  `val_lddt_ca=0.37177472934126854` and FoldScore `0.3721824511885643`.
- E105a bounded watch at `2026-05-13T12:35:42Z`: PID `1307` was still active
  after about 44 minutes on the owned pod. `results.json` and `results.csv`
  were absent, and `history_full_msa_to_face.json` still had 12 inherited rows
  ending at E72 step 5500. This remains in flight; do not update
  `EXPERIMENT_RESULTS.md` yet.
- E105a returned on owned pod `o1dy17ouv8w5mz` at `2026-05-13T12:47:49Z`.
  Remote coherence passed across `results.json`, `results.csv`,
  `history_full_msa_to_face.json`, and `run_metadata.json`: one result row,
  one CSV row, 13 history rows ending at step 6000, `parameters=3154242`,
  `val_lddt_ca=0.3894216101616621`,
  `val_foldscore=0.37369451113045216`,
  `val_ca_drmsd=10.740996658802032`, and C-alpha Rg
  `10.83685490489006 / 15.403406739234924`.
- Pulled E105a artifacts locally into
  `artifacts/nanofold_public_benchmarks/e105a_boundary_metric_recycling_from_e72_s6000_c256_m64/`,
  excluding checkpoints. Local coherence passed across results/history/CSV and
  confirmed the same step, parameter count, C-alpha lDDT, FoldScore, dRMSD,
  and C-alpha Rg values. E105a is a recovery-branch keep: it improves the
  retained E72/E73/E74/E76 comparison band but remains below the E96 primary
  leader, so use the E105a checkpoint as the handoff for E106 rather than
  spending 30k on E105a directly.
- E106 remote staging on owned pod `o1dy17ouv8w5mz`: fast-forwarded
  `/workspace/SimplexFold` from commit `33f4776` to `54a6635` with
  `git merge --ff-only`, preserving remote artifacts. Verified the E105a
  checkpoint at
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e105a_boundary_metric_recycling_from_e72_s6000_c256_m64/checkpoints/full_msa_to_face_latest.pt`,
  remote py_compile for model/adapter/trainer/runner files, and CLI support
  for `--simplex-boundary-cochain-recycling-scale` plus the runtime schedule
  flags. E106 launch-style parameter audit counted `3,154,242` parameters
  under the `3,261,974` AF2-medium +5% cap.
- E106 launched on owned pod `o1dy17ouv8w5mz` with run name
  `e106_boundary_cochain_recycling_from_e105a_s6500_c256_m64`, PID `2175`,
  log `/workspace/SimplexFold/logs/e106_boundary_cochain_recycling_from_e105a.log`,
  and artifact path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e106_boundary_cochain_recycling_from_e105a_s6500_c256_m64/`.
  It resumes the E105a checkpoint at step 6000/examples 48000 with
  `--resume-model-weights-only`, keeps the E105a selected-complex recipe
  fixed, disables metric recycling during this gate, and ramps
  selected-boundary cochain recycling from `0.0` to `0.10` over steps
  6000-6500. Startup showed `1244` matching tensors loaded and `0` new/missing
  tensors initialized.
- Retargeted heartbeat `check-simplexfold-e57-runpod` to E106, preserving
  owned-pod-only scope and the rule that the heartbeat must not launch
  follow-up experiments automatically.
- E106 startup health poll at `2026-05-13T12:55:09Z`: PID `2175` was active
  after about 1 minute, GPU utilization was `38%` with `11695 MiB` allocated,
  `results.json` was absent, and the inherited history had 13 rows ending at
  E105a step 6000 (`val_lddt_ca=0.3894216101616621`).
- E106 health poll at `2026-05-13T12:57:39Z`: PID `2175` was still active
  after about 4 minutes, GPU utilization was `26%` with `11699 MiB` allocated,
  `results.json` and `results.csv` were absent, and history still had 13 rows
  ending at E105a step 6000. No result artifact exists yet.
- Implemented E107 locally as
  `simplex_boundary_cochain_recycling_metric_gate_scale`, queued only if E106
  rejects. E107 scatters entropy-derived metric confidence from selected
  face/tetra distance heads onto the selected boundary 1-skeleton and gates the
  recycled `simplex_structure_pair_readout` before it is added to `z_prev`.
  This filters inter-cycle cochain memory without changing current-cycle
  edge transport, adding parameters, or adding a new output loss.
- E107 validation so far:
  focused tests for confidence-map scatter, no-new-parameter behavior,
  uncertain-cochain suppression, and runner CLI/config override reported
  `4 passed`; the broader local slice
  `/Users/christopherhayduk/Projects/nanoFold-Competition/.venv/bin/python -m py_compile minalphafold/simplex.py minalphafold/model.py minalphafold/model_config.py scripts/run_nanofold_public_benchmarks.py tests/test_simplex.py tests/test_trainer.py tests/test_nanofold_public_benchmarks.py && /Users/christopherhayduk/Projects/nanoFold-Competition/.venv/bin/python -m pytest tests/test_simplex.py tests/test_trainer.py tests/test_nanofold_public_benchmarks.py`
  reported `183 passed`. A launch-style parameter audit counted `3,154,242`
  parameters under the `3,261,974` cap.
- E106 health poll at `2026-05-13T13:04:17Z`: PID `2175` was active after
  about 10.5 minutes, GPU utilization was `30%` with `13491 MiB` allocated,
  `results.json` and `results.csv` were absent, and history still had 13 rows
  ending at E105a step 6000. Continue to treat E106 as in flight.
- E106 health poll at `2026-05-13T13:20:26Z`: PID `2175` was still active
  after about 26.7 minutes. `results.json` and `results.csv` were still
  absent; `history_full_msa_to_face.json` still had 13 rows ending at E105a
  step 6000 (`val_lddt_ca=0.3894216101616621`, FoldScore
  `0.37369451113045216`, `val_ca_drmsd=10.740996658802032`). A follow-up
  process/GPU check at `2026-05-13T13:20:55Z` showed GPU utilization sampled
  at `0%` with `13491 MiB` allocated while the Python process continued to use
  CPU heavily and still held GPU file descriptors. The remote log still showed
  only startup/resume lines, and the run directory file mtimes had not advanced
  past startup. Treat E106 as in flight for now, but re-check promptly for
  either a returned artifact or signs of a stalled training/evaluation loop.
- E106 health poll at `2026-05-13T13:32:39Z`: PID `2175` remained active after
  about 38.9 minutes. `results.json` and `results.csv` were still absent, and
  history still had 13 rows ending at E105a step 6000. GPU utilization sampled
  at `46%` with `13491 MiB` allocated, so the earlier `0%` sample looks like
  a transient idle point rather than enough evidence of a stall. Continue
  monitoring E106 as an active in-flight gate.
- E106 returned on owned pod `o1dy17ouv8w5mz` at `2026-05-13T13:49:59Z`.
  Remote coherence passed across `results.json`, `results.csv`,
  `history_full_msa_to_face.json`, and `run_metadata.json`: one result row,
  one CSV row, 14 history rows ending at step 6500, `parameters=3154242`
  under the `3261974` cap, and `stopped_early=False`. Metrics:
  `val_lddt_ca=0.3929199054837227`, FoldScore `0.377660034224391`,
  `val_ca_drmsd=10.327910900115967`, and C-alpha Rg
  `11.27131199836731 / 15.403406739234924`.
- Pulled E106 non-checkpoint artifacts locally into
  `artifacts/nanofold_public_benchmarks/e106_boundary_cochain_recycling_from_e105a_s6500_c256_m64/`
  and verified local coherence against the same step, parameter count, lDDT,
  FoldScore, dRMSD, and Rg values. The remote checkpoint
  `checkpoints/full_msa_to_face_latest.pt` exists for continuation.
- E106 decision: keep as recovery-branch evidence, not as a primary 30k spend.
  It improves E105a on primary C-alpha lDDT (`0.3894216101616621` to
  `0.3929199054837227`), FoldScore, dRMSD, and expansion, but remains below
  E96's `0.4043184444308281`. Queue E108 as a clean continuation from the E106
  checkpoint with cochain recycling held at `0.10`; reserve E107's metric gate
  for a stall or regression in the cochain-memory route.
- E108 remote staging on owned pod `o1dy17ouv8w5mz`: fast-forwarded
  `/workspace/SimplexFold` to commit `56150c7`, verified that no active
  benchmark process was present, verified the E106 checkpoint at
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e106_boundary_cochain_recycling_from_e105a_s6500_c256_m64/checkpoints/full_msa_to_face_latest.pt`,
  ran remote py_compile for model/adapter/trainer/runner files, and audited
  the E108 launch-style module set at `3,154,242` parameters under the
  `3,261,974` AF2-medium +5% cap.
- E108 launched on owned pod `o1dy17ouv8w5mz` with run name
  `e108_boundary_cochain_recycling_continue_from_e106_s7000_c256_m64`, PID
  `2881`, log
  `/workspace/SimplexFold/logs/e108_boundary_cochain_recycling_continue_from_e106.log`,
  and artifact path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e108_boundary_cochain_recycling_continue_from_e106_s7000_c256_m64/`.
  It resumes the E106 checkpoint at step 6500/examples 52000 with
  `--resume-model-weights-only`, keeps the E106 selected-complex recipe fixed,
  disables metric recycling, and holds selected-boundary cochain recycling at
  `0.10` instead of ramping it. Startup showed `1244` matching tensors loaded
  and `0` new/missing tensors initialized.
- Retargeted heartbeat `check-simplexfold-e57-runpod` to E108, preserving
  owned-pod-only scope and the rule that the heartbeat must not launch
  follow-up experiments automatically.
- E108 startup health poll at `2026-05-13T13:58:27Z`: PID `2881` was active
  after about 1.75 minutes, GPU utilization was `29%` with `11699 MiB`
  allocated, `results.json` and `results.csv` were absent, and the inherited
  history had 14 rows ending at E106 step 6500
  (`val_lddt_ca=0.3929199054837227`, FoldScore `0.377660034224391`,
  `val_ca_drmsd=10.327910900115967`). Continue to treat E108 as in flight.
- E108 health poll at `2026-05-13T14:10:09Z`: PID `2881` remained active after
  about 13.5 minutes, GPU utilization sampled at `11%` with `13491 MiB`
  allocated, `results.json` and `results.csv` were still absent, and history
  still had 14 rows ending at E106 step 6500. Continue to treat E108 as an
  active in-flight gate.
- E108 health poll at `2026-05-13T14:30:58Z`: PID `2881` remained active after
  about 34.3 minutes, GPU utilization sampled at `5%` with `13491 MiB`
  allocated, `results.json` and `results.csv` were still absent, and history
  still had 14 rows ending at E106 step 6500. Continue to treat E108 as an
  active in-flight gate; heartbeat monitoring is retargeted to this run.
- E108 returned on owned pod `o1dy17ouv8w5mz` at `2026-05-13T14:54:28Z`.
  Remote coherence passed across `results.json`, `results.csv`,
  `history_full_msa_to_face.json`, and `run_metadata.json`: one result row,
  one CSV row, 15 history rows ending at step 7000, `parameters=3154242`
  under the `3261974` cap, and `stopped_early=False`. Metrics:
  `val_lddt_ca=0.38745662942528725`, FoldScore `0.3770501706749201`,
  `val_ca_drmsd=10.616994559764862`, and C-alpha Rg
  `11.311820685863495 / 15.403406739234924`.
- Pulled E108 non-checkpoint artifacts locally into
  `artifacts/nanofold_public_benchmarks/e108_boundary_cochain_recycling_continue_from_e106_s7000_c256_m64/`
  and verified local coherence against the same step, parameter count, lDDT,
  FoldScore, dRMSD, and Rg values.
- E108 decision: reject. Holding selected-boundary cochain recycling at `0.10`
  regressed below E106 on primary C-alpha lDDT (`0.3929199054837227` to
  `0.38745662942528725`), FoldScore, and dRMSD. Launch E107 from the better
  verified E106 checkpoint to test whether metric confidence can suppress
  uncertain recycled boundary cochains.
- E107 remote staging on owned pod `o1dy17ouv8w5mz`: fast-forwarded
  `/workspace/SimplexFold` to commit `3250890`, verified that no active
  benchmark process was present, verified the E106 checkpoint at
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e106_boundary_cochain_recycling_from_e105a_s6500_c256_m64/checkpoints/full_msa_to_face_latest.pt`,
  ran remote py_compile for model/adapter/trainer/runner files, and audited
  the E107 launch-style module set at `3,154,242` parameters under the
  `3,261,974` AF2-medium +5% cap.
- E107 launched on owned pod `o1dy17ouv8w5mz` with run name
  `e107_metric_gated_cochain_recycling_from_e106_s7000_c256_m64`, PID `3424`,
  log `/workspace/SimplexFold/logs/e107_metric_gated_cochain_recycling_from_e106.log`,
  and artifact path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e107_metric_gated_cochain_recycling_from_e106_s7000_c256_m64/`.
  It resumes the E106 checkpoint at step 6500/examples 52000 with
  `--resume-model-weights-only`, keeps the E106 selected-complex/cochain
  recipe fixed, and adds only
  `--simplex-boundary-cochain-recycling-metric-gate-scale 1.0`. Startup showed
  `1244` matching tensors loaded and `0` new/missing tensors initialized.
- Retargeted heartbeat `check-simplexfold-e57-runpod` to E107, preserving
  owned-pod-only scope and the rule that the heartbeat must not launch
  follow-up experiments automatically.
- E107 startup health poll at `2026-05-13T14:59:43Z`: PID `3424` was active
  after about 1.35 minutes, GPU utilization was `60%` with `11695 MiB`
  allocated, `results.json` and `results.csv` were absent, and the inherited
  history had 14 rows ending at E106 step 6500
  (`val_lddt_ca=0.3929199054837227`, FoldScore `0.377660034224391`,
  `val_ca_drmsd=10.327910900115967`). Continue to treat E107 as in flight.
- E107 health poll at `2026-05-13T15:20:48Z`: PID `3424` remained active after
  about 22.5 minutes, GPU utilization sampled at `74%` with `13493 MiB`
  allocated, `results.json` and `results.csv` were still absent, and history
  still had 14 rows ending at E106 step 6500. Continue to treat E107 as an
  active in-flight gate.
- E107 health poll at `2026-05-13T15:51:39Z`: PID `3424` remained active after
  about 53.3 minutes, GPU utilization sampled at `5%` with `13493 MiB`
  allocated, `results.json` and `results.csv` were still absent, and history
  still had 14 rows ending at E106 step 6500. Continue to treat E107 as in
  flight.
- E107 returned on owned pod `o1dy17ouv8w5mz` at `2026-05-13T16:02:01Z`.
  Remote coherence passed across `results.json`, `results.csv`,
  `history_full_msa_to_face.json`, and `run_metadata.json`: one result row,
  one CSV row, 15 history rows ending at step 7000, `parameters=3154242`
  under the `3261974` cap, and `stopped_early=False`. Metrics:
  `val_lddt_ca=0.3867832273244858`, FoldScore `0.3756795562803745`,
  `val_ca_drmsd=10.649024993181229`, and C-alpha Rg
  `11.111636906862259 / 15.403406739234924`.
- Pulled E107 non-checkpoint artifacts locally into
  `artifacts/nanofold_public_benchmarks/e107_metric_gated_cochain_recycling_from_e106_s7000_c256_m64/`
  and verified local coherence against the same step, parameter count, lDDT,
  FoldScore, dRMSD, and Rg values.
- E107 decision: reject. Metric-gating recycled selected-boundary cochains
  regressed below E106 (`0.3929199054837227` to `0.3867832273244858`) and also
  below E108, with worse FoldScore and dRMSD. The next gate should not hold or
  confidence-gate strong cochain memory; queue E109 to anneal the cochain
  memory down from the better verified E106 checkpoint.
- E109 remote staging on owned pod `o1dy17ouv8w5mz`: fast-forwarded
  `/workspace/SimplexFold` to commit `b064c68`, verified that no active
  benchmark process was present, verified the E106 checkpoint at
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e106_boundary_cochain_recycling_from_e105a_s6500_c256_m64/checkpoints/full_msa_to_face_latest.pt`,
  ran remote py_compile for model/adapter/trainer/runner files, and audited
  the E109 launch-style module set at `3,154,242` parameters under the
  `3,261,974` AF2-medium +5% cap.
- E109 launched on owned pod `o1dy17ouv8w5mz` with run name
  `e109_cochain_recycling_anneal_down_from_e106_s7000_c256_m64`, PID `3943`,
  log `/workspace/SimplexFold/logs/e109_cochain_recycling_anneal_down_from_e106.log`,
  and artifact path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e109_cochain_recycling_anneal_down_from_e106_s7000_c256_m64/`.
  It resumes the E106 checkpoint at step 6500/examples 52000 with
  `--resume-model-weights-only`, keeps the E106 selected-complex recipe fixed,
  disables metric gating, and anneals selected-boundary cochain recycling from
  `0.10` to `0.025` over steps 6500-7000. Startup showed `1244` matching
  tensors loaded and `0` new/missing tensors initialized.
- Retargeted heartbeat `check-simplexfold-e57-runpod` to E109, preserving
  owned-pod-only scope and the rule that the heartbeat must not launch
  follow-up experiments automatically.
- E109 startup health poll at `2026-05-13T16:07:34Z`: PID `3943` was active
  after about 1.25 minutes, GPU utilization was `45%` with `8195 MiB`
  allocated, `results.json` and `results.csv` were absent, and the inherited
  history had 14 rows ending at E106 step 6500
  (`val_lddt_ca=0.3929199054837227`, FoldScore `0.377660034224391`,
  `val_ca_drmsd=10.327910900115967`). Continue to treat E109 as in flight.
- E109 health poll at `2026-05-13T16:40:10Z`: PID `3943` remained active after
  about 33.8 minutes, GPU utilization sampled at `1%` with `13491 MiB`
  allocated, `results.json` and `results.csv` were still absent, and history
  still had 14 rows ending at E106 step 6500. Continue to treat E109 as an
  active in-flight gate; heartbeat monitoring is retargeted to this run.
- E109 health poll at `2026-05-13T16:57:35Z`: PID `3943` remained active after
  about 51.3 minutes, GPU utilization sampled at `27%` with `13491 MiB`
  allocated, `results.json` and `results.csv` were still absent, and history
  still had 14 rows ending at E106 step 6500. Continue to treat E109 as in
  flight.
- E109 returned on owned pod `o1dy17ouv8w5mz` at `2026-05-13T17:08:05Z`.
  Remote coherence passed across `results.json`, `results.csv`,
  `history_full_msa_to_face.json`, and `run_metadata.json`: one result row,
  one CSV row, 15 history rows ending at step 7000, `parameters=3154242`
  under the `3261974` cap, and `stopped_early=False`. Metrics:
  `val_lddt_ca=0.39087833277881145`, FoldScore `0.3797789867967367`,
  `val_ca_drmsd=10.329187154769897`, and C-alpha Rg
  `11.550284147262573 / 15.403406739234924`.
- Pulled E109 non-checkpoint artifacts locally into
  `artifacts/nanofold_public_benchmarks/e109_cochain_recycling_anneal_down_from_e106_s7000_c256_m64/`
  and verified local coherence against the same step, parameter count, lDDT,
  FoldScore, dRMSD, and Rg values.
- E109 decision: partial recovery but reject as a continuation candidate.
  Annealing cochain memory from `0.10` to `0.025` improves over E107/E108 and
  improves FoldScore over E106, but primary C-alpha lDDT remains below E106
  (`0.39087833277881145` versus `0.3929199054837227`). Queue E110 as the
  limiting release-to-zero ablation from the better verified E106 checkpoint.
- E110 remote staging on owned pod `o1dy17ouv8w5mz`: fast-forwarded
  `/workspace/SimplexFold` to commit `9359d30`, verified that no active
  benchmark process was present, verified the E106 checkpoint at
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e106_boundary_cochain_recycling_from_e105a_s6500_c256_m64/checkpoints/full_msa_to_face_latest.pt`,
  ran remote py_compile for model/adapter/trainer/runner files, and audited
  the E110 launch-style module set at `3,154,242` parameters under the
  `3,261,974` AF2-medium +5% cap.
- E110 launched on owned pod `o1dy17ouv8w5mz` with run name
  `e110_cochain_recycling_release_from_e106_s7000_c256_m64`, PID `4513`,
  log `/workspace/SimplexFold/logs/e110_cochain_recycling_release_from_e106.log`,
  and artifact path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e110_cochain_recycling_release_from_e106_s7000_c256_m64/`.
  It resumes the E106 checkpoint at step 6500/examples 52000 with
  `--resume-model-weights-only`, keeps the E106 selected-complex recipe fixed,
  disables metric gating, and anneals selected-boundary cochain recycling from
  `0.10` to `0.0` over steps 6500-7000. Startup showed `1244` matching
  tensors loaded and `0` new/missing tensors initialized.
- Retargeted heartbeat `check-simplexfold-e57-runpod` to E110, preserving
  owned-pod-only scope and the rule that the heartbeat must not launch
  follow-up experiments automatically.
- E110 startup health poll at `2026-05-13T17:17:23Z`: PID `4513` was active,
  `results.json` was absent, and the inherited history had 14 rows ending at
  E106 step 6500 (`val_lddt_ca=0.3929199054837227`, FoldScore
  `0.377660034224391`, `val_ca_drmsd=10.327910900115967`, C-alpha Rg
  `11.27131199836731 / 15.403406739234924`). Continue to treat E110 as an
  active in-flight gate; do not add it to `EXPERIMENT_RESULTS.md` until final
  artifacts return and pass coherence checks.
- E110 follow-up health poll at `2026-05-13T17:18:53Z`: PID `4513` remained
  active, GPU utilization sampled at `3%` with `11699 MiB` allocated,
  `results.json` was still absent, and history still had 14 rows ending at
  E106 step 6500. Continue to treat E110 as in flight.
- Implemented E111 locally as `simplex_structure_pair_readout_scale`, queued
  but not launched while E110 is running. The route is a pair-only
  structure-module bias from the selected boundary 1-cochain: the simplex
  adapter emits `simplex_structure_pair_readout` without broad
  `simplex_structure_readout_scale`, and `AlphaFold2` RMS-normalizes that
  cochain before adding it only to the pair representation consumed by IPA.
  This intentionally avoids a new loss, direct coordinate supervision, residue
  0-cochain structure readout, or persistent cochain recycle state.
- E111 validation so far:
  `/Users/christopherhayduk/Projects/nanoFold-Competition/.venv/bin/python -m py_compile minalphafold/model.py minalphafold/simplex.py minalphafold/model_config.py scripts/run_nanofold_public_benchmarks.py`;
  `/Users/christopherhayduk/Projects/nanoFold-Competition/.venv/bin/python -m pytest tests/test_simplex.py::test_simplicial_adapter_can_emit_pair_only_structure_readout tests/test_trainer.py::test_simplicial_structure_pair_readout_adds_no_parameters tests/test_trainer.py::test_simplicial_structure_pair_readout_forward_uses_private_pair_cochain tests/test_nanofold_public_benchmarks.py::test_model_config_override_flags_are_accepted_by_cli_parser`
  reported `4 passed`; E111 launch-style parameter audit with the E110
  selected-complex recipe and `simplex_structure_pair_readout_scale=0.05`
  counted `3,154,242` parameters under the `3,261,974` AF2-medium +5% cap.
- E111 broader focused validation:
  `/Users/christopherhayduk/Projects/nanoFold-Competition/.venv/bin/python -m pytest tests/test_simplex.py::test_simplicial_adapter_can_emit_structure_readout_from_selected_cells tests/test_simplex.py::test_simplicial_adapter_can_emit_pair_only_structure_readout tests/test_trainer.py::test_simplicial_structure_readout_adds_no_parameters tests/test_trainer.py::test_simplicial_structure_pair_readout_adds_no_parameters tests/test_trainer.py::test_simplicial_structure_readout_forward_keeps_internal_tensors_private tests/test_trainer.py::test_simplicial_structure_pair_readout_forward_uses_private_pair_cochain tests/test_trainer.py::test_simplicial_boundary_cochain_recycling_changes_only_recycled_cycles tests/test_trainer.py::test_metric_gated_boundary_cochain_recycling_suppresses_uncertain_recycled_cochains tests/test_nanofold_public_benchmarks.py::test_model_config_override_flags_are_accepted_by_cli_parser tests/test_nanofold_public_benchmarks.py::test_runtime_simplex_message_scales_ramp_and_enter_model_inputs tests/test_nanofold_public_benchmarks.py::test_evaluate_uses_runtime_simplex_overrides_for_validation`
  reported `11 passed`.
- E110 health poll at `2026-05-13T17:27:56Z`: PID `4513` remained active, GPU
  utilization sampled at `32%` with `13491 MiB` allocated, `results.json` was
  still absent, and history still had 14 rows ending at E106 step 6500.
- E110 process-detail poll at `2026-05-13T17:29:41Z`: PID `4513` remained
  active after about 13.3 minutes with process state `Rl`, high CPU activity,
  and GPU memory still allocated. The E110 log still contained only the clean
  E106 resume/startup lines, and recent files were limited to
  `run_metadata.json` plus inherited `history_full_msa_to_face.json`. Treat
  this as an active first 500-step gate rather than a returned run.
- E110 health poll at `2026-05-13T17:32:19Z`: PID `4513` remained active, GPU
  utilization sampled at `48%` with `13491 MiB` allocated, `results.json` and
  `results.csv` were absent, and history still had 14 rows ending at E106 step
  6500. Continue to treat E110 as in flight.
- E110 health poll at `2026-05-13T17:33:36Z`: PID `4513` remained active after
  about 17.3 minutes, process state `Rl` with high CPU activity, GPU memory
  `13491 MiB`, `results.json` and `results.csv` absent, and history still at
  the inherited E106 step-6500 row. Continue monitoring without intervention.
- E110 health poll at `2026-05-13T17:35:15Z`: PID `4513` remained active after
  about 19.0 minutes, process state `Rl` with high CPU activity, GPU
  utilization sampled at `55%` with `13491 MiB` allocated, `results.json` and
  `results.csv` absent, and history still at E106 step 6500. Continue treating
  E110 as an active first post-resume gate.
- E110 health poll at `2026-05-13T17:36:55Z`: PID `4513` remained active after
  about 20.7 minutes, process state `Rl` with high CPU activity, GPU
  utilization sampled at `33%` with `13491 MiB` allocated, `results.json` and
  `results.csv` absent, and history still at E106 step 6500. Continue
  monitoring; no remote checkout changes while the process is active.
- E110 health poll at `2026-05-13T17:38:22Z`: PID `4513` remained active after
  about 22.0 minutes, process state `Rl` with high CPU activity, GPU
  utilization sampled at `41%` with `13491 MiB` allocated, `results.json` and
  `results.csv` absent, and history still at E106 step 6500.
- E110 health poll at `2026-05-13T17:43:31Z`: PID `4513` remained active after
  about 27.3 minutes, process state `Rl` with high CPU activity, GPU
  utilization sampled at `20%` with `13491 MiB` allocated, `results.json` and
  `results.csv` absent, history still at E106 step 6500, and the log still
  showed only startup/resume lines. Continue waiting for the first post-resume
  eval/write point.
- E110 health poll at `2026-05-13T17:49:31Z`: PID `4513` remained active after
  about 33.3 minutes, process state `Sl` with high CPU activity, GPU
  utilization sampled at `37%` with `13491 MiB` allocated, `results.json` and
  `results.csv` absent, and history still at E106 step 6500. Continue treating
  E110 as in flight.
- E110 health poll at `2026-05-13T18:00:38Z`: PID `4513` remained active after
  about 44.4 minutes, process state `Sl` with high CPU activity, GPU
  utilization sampled at `56%` with `13491 MiB` allocated, `results.json` and
  `results.csv` absent, and history still at E106 step 6500.
- E110 returned on owned pod `o1dy17ouv8w5mz` at the next check. Remote
  coherence passed across `results.json`, `results.csv`,
  `history_full_msa_to_face.json`, and `run_metadata.json`: one result row,
  one CSV row, 15 history rows ending at step 7000, `completed_steps=7000`,
  `parameters=3154242` under the `3261974` cap, `effective_batch_size=8`, and
  `stopped_early=False`. Metrics: `val_lddt_ca=0.38161011785268784`,
  FoldScore `0.37879121489822865`, `val_ca_drmsd=10.373755931854248`, and
  C-alpha Rg `11.778071075677872 / 15.403406739234924`.
- Pulled E110 non-checkpoint artifacts locally into
  `artifacts/nanofold_public_benchmarks/e110_cochain_recycling_release_from_e106_s7000_c256_m64/`
  and verified local coherence against the same step, parameter count,
  effective batch size, stopped-early flag, lDDT, FoldScore, dRMSD, and Rg
  values.
- E110 decision: reject. Fully releasing selected-boundary cochain recycling
  from `0.10` to `0.0` regressed below E106 (`0.3929199054837227`) and below
  E107/E108/E109 on primary C-alpha lDDT. The cochain-memory schedule family
  is not the next 30k candidate; launch E111 from the better verified E106
  checkpoint rather than from E110.
- E111 remote staging on owned pod `o1dy17ouv8w5mz`: fast-forwarded
  `/workspace/SimplexFold` to commit `a17b4c3`; py_compile passed for
  `minalphafold/simplex.py`, `minalphafold/model.py`,
  `minalphafold/model_config.py`, `minalphafold/trainer.py`, and
  `scripts/run_nanofold_public_benchmarks.py`; verified the E106 resume
  checkpoint at
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e106_boundary_cochain_recycling_from_e105a_s6500_c256_m64/checkpoints/full_msa_to_face_latest.pt`.
- E111 launch-style parameter audit counted `3,154,242` parameters under the
  `3,261,974` cap with the selected-complex module set and
  `simplex_structure_pair_readout_scale=0.05`.
- E111 launched on owned pod `o1dy17ouv8w5mz` with run name
  `e111_pair_only_structure_cochain_from_e106_s7000_c256_m64`, log
  `/workspace/SimplexFold/logs/e111_pair_only_structure_cochain_from_e106.log`,
  and PID file
  `/workspace/SimplexFold/logs/e111_pair_only_structure_cochain_from_e106.pid`.
  The run resumes the verified E106 checkpoint at step 6500 with cochain
  recycling released to `0.0`, `simplex_structure_pair_readout_scale=0.05`,
  crop 256, MSA depth 64, four cycles, and effective batch size 8.
- E111 startup health poll at `2026-05-13T18:20:11Z`: PID `5411` was active,
  the log showed clean artifact creation, train/val counts `10000/1000`,
  resume from E106 at step 6500/examples 52000, `1244` matching tensors loaded,
  and `0` new/missing tensors initialized with a fresh optimizer.
- Retargeted heartbeat `check-simplexfold-e57-runpod` to E111, preserving the
  owned-pod-only rule for pod `o1dy17ouv8w5mz`, the 30-minute interval, and the
  no-automatic-follow-up-launch rule.
- E111 follow-up health poll at `2026-05-13T18:22:09Z`: PID `5411` remained
  active after about 2.2 minutes with process state `Rl`, GPU memory
  `11719 MiB` allocated, `results.json` and `results.csv` absent, and history
  still at the inherited E106 step-6500 row. Continue treating E111 as in
  flight.
- E111 mid-run health poll at `2026-05-13T18:45:11Z`: PID `5411` remained
  active after about 25.1 minutes with process state `Sl`, GPU memory
  `13511 MiB` allocated, `results.json` and `results.csv` absent, history
  still at the inherited E106 step-6500 row, and the log still showing only
  clean startup/resume lines. Continue waiting for the step-7000 write point
  without changing the remote checkout.
- E111 returned on owned pod `o1dy17ouv8w5mz`. Remote coherence passed across
  `results.json`, `results.csv`, `history_full_msa_to_face.json`,
  `eval_details_full_msa_to_face.csv`, and `run_metadata.json`: one result
  row, one CSV row, 15 history rows ending at step 7000, 16 eval-detail rows,
  `completed_steps=7000`, `parameters=3154242` under the `3261974` cap,
  `effective_batch_size=8`, and `stopped_early=False`. Metrics:
  `val_lddt_ca=0.39201722107827663`, FoldScore `0.37592183239758015`,
  `val_ca_drmsd=10.419719159603119`, and C-alpha Rg
  `11.342422276735306 / 15.403406739234924`.
- Pulled E111 non-checkpoint artifacts locally into
  `artifacts/nanofold_public_benchmarks/e111_pair_only_structure_cochain_from_e106_s7000_c256_m64/`
  and verified local coherence against the same step, parameter count,
  effective batch size, stopped-early flag, lDDT, FoldScore, dRMSD, Rg, and
  eval-detail row count values.
- E111 decision: reject. The pair-only selected-boundary cochain structure bias
  recovered most of E110's release-to-zero drop but stayed below E106
  (`0.39201722107827663` versus `0.3929199054837227`) and softened FoldScore
  and dRMSD. Queue E112 as the only remaining structure-bias calibration:
  rerun from the verified E106 checkpoint with
  `simplex_structure_pair_readout_scale=0.025`.
- E112 remote staging on owned pod `o1dy17ouv8w5mz`: fast-forwarded
  `/workspace/SimplexFold` to commit `e07d71c`; verified no active benchmark
  process; py_compile passed for `minalphafold/simplex.py`,
  `minalphafold/model.py`, `minalphafold/model_config.py`,
  `minalphafold/trainer.py`, and `scripts/run_nanofold_public_benchmarks.py`;
  verified the E106 resume checkpoint; launch-style parameter audit counted
  `3,154,242` parameters under the `3,261,974` cap with
  `simplex_structure_pair_readout_scale=0.025`.
- E112 launched on owned pod `o1dy17ouv8w5mz` with run name
  `e112_half_pair_only_structure_cochain_from_e106_s7000_c256_m64`, log
  `/workspace/SimplexFold/logs/e112_half_pair_only_structure_cochain_from_e106.log`,
  and PID file
  `/workspace/SimplexFold/logs/e112_half_pair_only_structure_cochain_from_e106.pid`.
  Startup at `2026-05-13T19:20:41Z` showed PID `6212`, clean artifact
  creation, train/val counts `10000/1000`, resume from E106 at step
  6500/examples 52000, `1244` matching tensors loaded, and `0` new/missing
  tensors initialized with a fresh optimizer.
- Retargeted heartbeat `check-simplexfold-e57-runpod` to E112, preserving the
  owned-pod-only rule for pod `o1dy17ouv8w5mz`, the 30-minute interval, and the
  no-automatic-follow-up-launch rule.
- E112 mid-run health poll at `2026-05-13T19:54:47Z`: PID `6212` remained
  active after about 34.4 minutes with process state `Rl`, GPU memory
  `13511 MiB` allocated, `results.json` and `results.csv` absent, history
  still at the inherited E106 step-6500 row, and the log still showing clean
  startup/resume lines only. Continue waiting for the step-7000 write point
  without changing the remote checkout.
- E112 returned on owned pod `o1dy17ouv8w5mz`. Remote coherence passed across
  `results.json`, `results.csv`, `history_full_msa_to_face.json`,
  `eval_details_full_msa_to_face.csv`, and `run_metadata.json`: one result
  row, one CSV row, 15 history rows ending at step 7000, 16 eval-detail rows,
  `completed_steps=7000`, `parameters=3154242` under the `3261974` cap,
  `effective_batch_size=8`, and `stopped_early=False`. Metrics:
  `val_lddt_ca=0.38734209537506104`, FoldScore `0.3793369084596634`,
  `val_ca_drmsd=10.389029681682587`, and C-alpha Rg
  `11.570150882005692 / 15.403406739234924`.
- Pulled E112 non-checkpoint artifacts locally into
  `artifacts/nanofold_public_benchmarks/e112_half_pair_only_structure_cochain_from_e106_s7000_c256_m64/`
  and verified local coherence against the same step, parameter count,
  effective batch size, stopped-early flag, lDDT, FoldScore, dRMSD, Rg, and
  eval-detail row count values.
- E112 decision: reject. The half-scale pair-only selected-boundary cochain
  structure bias fell below E106 and E111 on primary C-alpha lDDT, despite a
  small FoldScore recovery over E111. Leave the structure-bias route and queue
  E113: directed boundary-readout annealing from the verified E106 checkpoint,
  with cochain recycling released to `0.0` and no structure pair bias.
- E113 remote staging on owned pod `o1dy17ouv8w5mz`: fast-forwarded
  `/workspace/SimplexFold` to commit `9d2b23c`; verified no active benchmark
  process; py_compile passed for `minalphafold/simplex.py`,
  `minalphafold/model.py`, `minalphafold/model_config.py`,
  `minalphafold/trainer.py`, and `scripts/run_nanofold_public_benchmarks.py`;
  verified the E106 resume checkpoint; launch-style parameter audit counted
  `3,154,242` parameters under the `3,261,974` cap with directed boundary
  readout enabled.
- E113 launched on owned pod `o1dy17ouv8w5mz` with run name
  `e113_anneal_directed_boundary_from_e106_s7000_c256_m64`, log
  `/workspace/SimplexFold/logs/e113_anneal_directed_boundary_from_e106.log`,
  and PID file
  `/workspace/SimplexFold/logs/e113_anneal_directed_boundary_from_e106.pid`.
  Startup at `2026-05-13T20:17:41Z` showed PID `6887`, clean artifact
  creation, train/val counts `10000/1000`, resume from E106 at step
  6500/examples 52000, `1244` matching tensors loaded, and `0` new/missing
  tensors initialized with a fresh optimizer. Directionality is annealed from
  `0.5` to `0.25` over steps 6500-7000, with cochain recycling released to
  `0.0` and structure pair readout disabled.
- Retargeted heartbeat `check-simplexfold-e57-runpod` to E113, preserving the
  owned-pod-only rule for pod `o1dy17ouv8w5mz`, the 30-minute interval, and the
  no-automatic-follow-up-launch rule.
- E113 mid-run health poll at `2026-05-13T20:50:24Z`: PID `6887` remained
  active after about 33.1 minutes with process state `Rl`, GPU memory
  `13689 MiB` allocated, `results.json` and `results.csv` absent, history
  still at the inherited E106 step-6500 row, and no failure trace visible in
  the startup log. Continue waiting for the step-7000 write point without
  changing the remote checkout.
- E113 follow-up health poll at `2026-05-13T21:00:15Z`: PID `6887` remained
  active after about 42.1 minutes with process state `Sl`, GPU memory
  `13689 MiB` allocated, `results.json` and `results.csv` absent, history
  still at the inherited E106 step-6500 row, and no failure trace visible.
  Continue waiting.
- While E113 runs, audited the next latent segment-cell branch against the
  AF2-medium +5% cap (`3,261,974`). With the current selected sparse-complex
  settings, latent contiguous segment cochains fit only if the learned
  edge-frame message modules are disabled: `c_segment=12`,
  `simplex_segment_cell_scale=0.05`, and edge-frame disabled gives
  `3,234,450` parameters, but even `c_segment=1` with edge-frame enabled gives
  `3,274,830`, which exceeds the cap. If E113 fails, the segment-cell route is
  still topology-native, but it must be framed as a trade of local
  edge-frame scalarization for persistent contiguous 1-/2-dimensional
  segment cochains rather than as a stacked add-on.
- Prepared E114 locally as a zero-parameter alternative to learned latent
  segment cells: `simplex_cell_score_segment_weight` rewards candidate
  selected face/tetra cells whose boundary edges are supported by contiguous
  sequence-segment cochains before the existing face/tetra top-k mask. This
  preserves the successful edge-frame/directed-incidence modules while still
  changing the active simplicial complex. Initial validation passed:
  `python -m py_compile minalphafold/simplex.py minalphafold/model_config.py scripts/run_nanofold_public_benchmarks.py`;
  `python -m pytest tests/test_simplex.py::test_cell_score_segment_weight_prefers_sequence_supported_cells tests/test_trainer.py::test_simplicial_cell_segment_score_adds_no_parameters tests/test_nanofold_public_benchmarks.py::test_model_config_override_flags_are_accepted_by_cli_parser`
  reported `3 passed`.
- Broader local validation for E114 also passed:
  `python -m pytest tests/test_simplex.py tests/test_nanofold_public_benchmarks.py tests/test_trainer.py`
  reported `188 passed`.
- E113 returned on owned pod `o1dy17ouv8w5mz`. Remote coherence passed across
  `results.json`, `results.csv`, `history_full_msa_to_face.json`,
  `eval_details_full_msa_to_face.csv`, and `run_metadata.json`: one result
  row, one CSV row, 15 history rows ending at step 7000, 16 eval-detail rows,
  `completed_steps=7000`, `parameters=3154242` under the `3261974` cap,
  `effective_batch_size=8`, and `stopped_early=False`. Metrics:
  `val_lddt_ca=0.395872812718153`, FoldScore `0.3774686213582754`,
  `val_ca_drmsd=10.630483537912369`, and C-alpha Rg
  `11.165981888771057 / 15.403406739234924`.
- Pulled E113 non-checkpoint artifacts locally into
  `artifacts/nanofold_public_benchmarks/e113_anneal_directed_boundary_from_e106_s7000_c256_m64/`
  and verified local coherence against the same step, parameter count,
  effective batch size, stopped-early flag, lDDT, FoldScore, dRMSD, Rg, and
  eval-detail row count values.
- E113 decision: keep only as a recovery-branch handoff. It improves over
  E106 on primary C-alpha lDDT (`0.395872812718153` versus
  `0.3929199054837227`) but remains below E96/E97 and worsens dRMSD. Launch
  E114 from the E113 checkpoint rather than spending 30k on E113 as-is.
- E114 remote staging on owned pod `o1dy17ouv8w5mz`: fast-forwarded
  `/workspace/SimplexFold` to commit `221df89`; verified no active benchmark
  process; py_compile passed for `minalphafold/simplex.py`,
  `minalphafold/model.py`, `minalphafold/model_config.py`,
  `minalphafold/trainer.py`, and `scripts/run_nanofold_public_benchmarks.py`;
  verified the E113 resume checkpoint; launch-style parameter audit counted
  `3,154,242` parameters under the `3,261,974` cap with
  `simplex_cell_score_segment_weight=0.25`.
- E114 launched on owned pod `o1dy17ouv8w5mz` with run name
  `e114_segment_supported_filtration_from_e113_s7500_c256_m64`, log
  `/workspace/SimplexFold/logs/e114_segment_supported_filtration_from_e113.log`,
  and PID file
  `/workspace/SimplexFold/logs/e114_segment_supported_filtration_from_e113.pid`.
  Startup showed PID `7698`, clean artifact creation, resume from E113 at step
  7000/examples 56000, `1244` matching tensors loaded, and `0` new/missing
  tensors initialized with a fresh optimizer. A first health poll confirmed
  the metadata records `steps=7500`,
  `simplex_cell_score_segment_weight=0.25`, directed boundary readout held at
  `0.25`, `max_parameters=3261974`, and no result files yet.
- Retargeted heartbeat `check-simplexfold-e57-runpod` to E114, preserving the
  owned-pod-only rule for pod `o1dy17ouv8w5mz`, the 30-minute interval, and the
  no-automatic-follow-up-launch rule.
- E114 mid-run health poll: PID `7698` remained active after about 30.3
  minutes with process state `Sl`, GPU memory `13689 MiB` allocated,
  `results.json` and `results.csv` absent, and history still at the inherited
  E113 step-7000 row. Continue waiting for the step-7500 validation write
  without changing the remote checkout.
- E114 returned on owned pod `o1dy17ouv8w5mz`. Remote coherence passed across
  `results.json`, `results.csv`, `history_full_msa_to_face.json`,
  `eval_details_full_msa_to_face.csv`, and `run_metadata.json`: one result
  row, one CSV row, 16 history rows ending at step 7500, 16 eval-detail rows,
  `completed_steps=7500`, `parameters=3154242` under the `3261974` cap,
  `effective_batch_size=8`, `simplex_cell_score_segment_weight=0.25`, and
  `stopped_early=False`. Metrics: `val_lddt_ca=0.381430733948946`, FoldScore
  `0.37934823520481586`, `val_ca_drmsd=10.612294971942902`, and C-alpha Rg
  `11.858259320259094 / 15.403406739234924`.
- Pulled E114 non-checkpoint artifacts locally into
  `artifacts/nanofold_public_benchmarks/e114_segment_supported_filtration_from_e113_s7500_c256_m64/`
  and verified local coherence against the same step, parameter count,
  effective batch size, stopped-early flag, segment-support weight, lDDT,
  FoldScore, dRMSD, Rg, and eval-detail row count values.
- E114 decision: reject. Segment-supported filtration improved FoldScore
  (`0.3793` versus E113 `0.3775`), dRMSD (`10.6123` versus `10.6305`),
  C-alpha expansion, and selected-boundary contraction, but primary C-alpha
  lDDT fell sharply below E113 and E106. Launch E115 as a no-segment
  continuation control from the same E113 checkpoint before trying a weaker
  segment-support scale.
- E115 remote staging on owned pod `o1dy17ouv8w5mz`: fast-forwarded
  `/workspace/SimplexFold` to commit `2ccb433`; verified no active benchmark
  process; py_compile passed for `minalphafold/simplex.py`,
  `minalphafold/model.py`, `minalphafold/model_config.py`,
  `minalphafold/trainer.py`, and `scripts/run_nanofold_public_benchmarks.py`;
  verified the E113 resume checkpoint; launch-style parameter audit counted
  `3,154,242` parameters under the `3,261,974` cap with segment-supported
  scoring disabled.
- E115 launched on owned pod `o1dy17ouv8w5mz` with run name
  `e115_no_segment_control_from_e113_s7500_c256_m64`, log
  `/workspace/SimplexFold/logs/e115_no_segment_control_from_e113.log`, and PID
  file `/workspace/SimplexFold/logs/e115_no_segment_control_from_e113.pid`.
  Startup showed PID `8296`, clean artifact creation, resume from E113 at step
  7000/examples 56000, `1244` matching tensors loaded, and `0` new/missing
  tensors initialized with a fresh optimizer. A first health poll confirmed
  metadata records `steps=7500`, `max_parameters=3261974`, directed boundary
  readout held at `0.25`, `simplex_cell_score_segment_weight=None`, and no
  result files yet.
- Retargeted heartbeat `check-simplexfold-e57-runpod` to E115, preserving the
  owned-pod-only rule for pod `o1dy17ouv8w5mz`, the 30-minute interval, and the
  no-automatic-follow-up-launch rule.
- E115 health poll at `2026-05-13T22:41:18Z`: PID `8296` remained active on
  the owned pod, `results.json`, `results.csv`, and
  `eval_details_full_msa_to_face.csv` were still absent, and history remained
  at the inherited E113 step-7000 row (`val_lddt_ca=0.395872812718153`).
- Implemented the queued E116 global selected-complex context candidate
  locally while E115 runs. The new `simplex_global_context_scale` path pools
  active face/tetra cochains into a protein-level selected-complex summary and
  routes that summary back into active face/tetra states before boundary-edge
  readout. This is a topology-native architecture change rather than an
  output-side metric loss.
- E116 local validation so far: `python -m py_compile minalphafold/simplex.py
  minalphafold/model_config.py scripts/run_nanofold_public_benchmarks.py` and
  `python -m pytest
  tests/test_simplex.py::test_global_context_adapter_routes_selected_complex_summary_back_to_cells
  tests/test_trainer.py::test_simplicial_global_context_stays_inside_af2_medium_budget
  tests/test_nanofold_public_benchmarks.py::test_model_config_override_flags_are_accepted_by_cli_parser`
  both passed. Launch-style parameter audit for the E113/E115 sparse recipe
  plus global context counted `3,201,970` parameters under the `3,261,974`
  cap.
- E115 health poll at `2026-05-13T22:50:42Z`: PID `8296` remained active
  after about `22:40` elapsed, GPU memory was still allocated, `results.json`,
  `results.csv`, and `eval_details_full_msa_to_face.csv` were absent, and
  history still had 15 rows ending at the inherited E113 step-7000 row. Do not
  launch E116 or touch the remote checkout until E115 either returns coherent
  results or shows a real failure.
- E115 returned on owned pod `o1dy17ouv8w5mz`. Remote coherence passed across
  `results.json`, `results.csv`, `history_full_msa_to_face.json`,
  `eval_details_full_msa_to_face.csv`, and `run_metadata.json`: one result
  row, one CSV row, 16 history rows ending at step 7500, 16 eval-detail rows,
  `completed_steps=7500`, `parameters=3154242` under the `3261974` cap,
  `effective_batch_size=8`, `simplex_cell_score_segment_weight=0.0`, and
  `stopped_early=False`. Metrics: `val_lddt_ca=0.3820176888257265`,
  FoldScore `0.37714015133678913`, `val_ca_drmsd=10.377011388540268`,
  C-alpha Rg `11.570684283971786 / 15.403406739234924`, selected
  face/tetra boundary lDDT `0.721814651042223 / 0.7055758349597454`, and
  face/tetra contraction `0.6876201741397381 / 0.6860276646912098`.
- Pulled E115 non-checkpoint artifacts locally into
  `artifacts/nanofold_public_benchmarks/e115_no_segment_control_from_e113_s7500_c256_m64/`
  and the log into `logs/e115_no_segment_control_from_e113.log`. Local
  coherence passed against the same step, parameter count, effective batch,
  stopped-early flag, disabled segment support, history/eval row counts, and
  primary metrics.
- E115 decision: reject. The matched no-segment continuation fell below E113
  and E106 and nearly matched E114's primary-lDDT drop, so E114 was not only a
  segment-support failure. The E113 recovery lineage is unstable; do not launch
  E116 from E113/E115 blindly. Audit retained checkpoints on the owned pod
  before deciding whether E116 should run from a stronger compatible source.
- Owned-pod checkpoint audit after E115: no active benchmark process was found
  on pod `o1dy17ouv8w5mz`. Retained checkpoints exist for E72 and E105a-E115,
  but no E96/E97-family checkpoint remains on the pod. The strongest retained
  stable source is therefore E106; E116 can be considered only as a short
  E106-sourced recovery test unless a stronger checkpoint is restored.
- Paused heartbeat `check-simplexfold-e57-runpod` because no SimplexFold
  benchmark is active. Stopped owned pod `o1dy17ouv8w5mz` after local
  verification and checkpoint audit; Runpod reported desired status `EXITED`.
- Restarted owned pod `o1dy17ouv8w5mz` to continue the E116 probe. As after
  E104, the zero-volume restart returned with an empty `/workspace`; the
  remote SimplexFold checkout, NanoFold checkout, and E105a-E115 checkpoints
  were gone. Local retained checkpoints only go up through E72, so the
  immediate E116 launch source changed from E106 to E72.
- Restaged the owned pod from scratch for E116. Cloned SimplexFold branch
  `codex/simplexfold-topology-e07-boundary-coordinate` at commit `7ba7bc9`,
  cloned nanoFold-Competition at commit `96afc846`, recreated
  `/workspace/venv` with system site packages, and installed SimplexFold
  editable. Rsync was too slow on the many-file public data tree, so the final
  transfer used a single Runpod croc archive
  `/tmp/simplexfold_e116_stage.tgz` containing public `processed_features`,
  `processed_labels`, `manifests`, and the local E72 checkpoint. Remote audit
  confirmed feature/label NPZ counts `11000/11000`, manifest rows
  `10000/1000/11000`, zero `._*` sidecars, E72 checkpoint present, py_compile
  passed for model/runner files, CLI support for `--simplex-global-context-scale`,
  and launch-style parameter count `3,201,970 <= 3,261,974`.
- E116 launched on owned pod `o1dy17ouv8w5mz` as
  `e116_global_context_from_e72_s6000_c256_m64`, PID `1566`, log
  `/workspace/SimplexFold/logs/e116_global_context_from_e72.log`, and artifact
  path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e116_global_context_from_e72_s6000_c256_m64/`.
  It resumes the local E72 checkpoint at step 5500/examples 44000 with
  `--resume-model-weights-only`, keeps the E105a sparse-complex recipe fixed,
  disables metric/cochain recycling, and enables
  `--simplex-global-context-scale 0.10`. Startup showed `1244` matching
  tensors loaded and `48` new/missing tensors initialized for the
  global-context modules with a fresh optimizer.
- E116 startup health poll: PID `1566` remained active after about 42 seconds,
  GPU memory was allocated, metadata recorded `steps=6000`,
  `effective_batch_size=8`, `max_parameters=3261974`,
  `simplex_global_context_scale=0.1`, `simplex_face_top_k=24`, and
  `simplex_tetra_top_k=48`. The inherited history had 12 rows ending at E72
  step 5500 and no `results.json` yet.
- E116 health poll after about 13.5 minutes: PID `1566` remained active with
  GPU utilization around `41%` and `13589 MiB` allocated. The run metadata and
  inherited history were still present, history still had 12 rows ending at
  E72 step 5500, and `results.json`, `results.csv`, and
  `eval_details_full_msa_to_face.csv` were not written yet. Continue waiting
  for the step-6000 validation result; do not touch the remote checkout or
  launch a follow-up while this run is active.
- E116 health poll after about 66 minutes: PID `1566` remained active, with
  `13589 MiB` GPU memory allocated but no result files yet. A process-state
  check showed state `Rl`, about `1049%` CPU, 178 threads, and 41 file
  descriptors, so the run appears compute-active rather than dead. Artifact
  mtimes still show only startup `run_metadata.json` and inherited
  `history_full_msa_to_face.json`; continue waiting for the first step-6000
  write.
- E116 returned on owned pod `o1dy17ouv8w5mz`. Remote coherence passed after
  correcting the eval-detail expectation: one result row, 13 history rows
  ending at step 6000, 1000 final eval-detail rows, `completed_steps=6000`,
  `parameters=3201970` under the `3261974` cap, `effective_batch_size=8`,
  `simplex_global_context_scale=0.1`, and `stopped_early=False`. Metrics:
  `val_lddt_ca=0.4095440942049027`, FoldScore `0.3880571389496326`,
  `val_ca_drmsd=11.29642592227459`, C-alpha Rg
  `11.591795643806458 / 16.30911695623398`, selected face/tetra boundary
  lDDT `0.7231793713569641 / 0.7094700435996055`, and selected face/tetra
  contraction `0.601930175870657 / 0.5994533261656761`.
- Pulled E116 non-checkpoint artifacts locally into
  `artifacts/nanofold_public_benchmarks/e116_global_context_from_e72_s6000_c256_m64/`
  and the log into `logs/e116_global_context_from_e72.log`. Local coherence
  passed against the same step, result row, history row, eval-detail row,
  parameter count, effective batch, stopped-early flag, global-context scale,
  and primary metrics.
- E116 decision: keep as the new primary-lDDT leader. It beats E96's
  `0.4043` despite starting from the weaker E72 checkpoint, so the selected
  global-complex cochain is the first topology-native change to improve the
  main metric beyond the previous plateau. It is still far below `0.7`; launch
  E117 as a matched 500-step continuation from the E116 checkpoint before any
  longer 30k spend.
- Committed and pushed the E116 result tracker update as `9548d5f`
  (`Record E116 global context result`). The owned pod was still running, with
  no active SimplexFold benchmark process and the E116 checkpoint present.
- E117 launched on owned pod `o1dy17ouv8w5mz` as
  `e117_global_context_continue_from_e116_s6500_c256_m64`, PID `13554`, log
  `/workspace/SimplexFold/logs/e117_global_context_continue_from_e116.log`,
  and artifact path
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e117_global_context_continue_from_e116_s6500_c256_m64/`.
  The remote checkout fast-forwarded to commit `9548d5f` before launch. E117
  resumes the E116 checkpoint at step 6000/examples 48000 with
  `--resume-model-weights-only`, keeps the same selected sparse-complex/global
  context recipe, and targets step 6500 at effective batch 8.
- E117 startup health poll: PID `13554` was compute-active after about
  31 seconds, metadata recorded `steps=6500`, `effective_batch_size=8`,
  `max_parameters=3261974`, `simplex_global_context_scale=0.1`,
  `simplex_face_top_k=24`, `simplex_tetra_top_k=48`, and
  `simplex_boundary_incidence_normalization=1.0`. The inherited history had
  13 rows ending at step 6000 and no `results.json` yet.
- E117 health poll after about 3 minutes: PID `13554` remained active on the
  owned pod, still running from the E116 checkpoint with no `results.json`,
  `results.csv`, eval-detail file, or checkpoint written yet. Artifact files
  are still limited to `run_metadata.json` and inherited
  `history_full_msa_to_face.json`; continue waiting for the step-6500
  validation write and do not launch a follow-up while this process is active.
- E117 health poll after about 4.5 minutes: PID `13554` remained active with
  no final result artifacts yet. The remote checkout is still at the E116
  result commit `9548d5f` plus the editable-install `minalphafold.egg-info/`
  dirt only; this is acceptable for the already-launched E117 run.
- E118 local implementation prepared while E117 runs. Added default-off
  `simplex_vertex_star_context_scale`, which reuses the existing selected
  global-complex MLPs but interpolates their context toward residue
  vertex-star cochains pooled by incidence from selected face/tetra states.
  This is topology-native: selected cells communicate through their incident
  residues before returning to active cells and boundary edges. It adds no
  parameters and should be checkpoint-compatible with E116/E117 when enabled.
- E118 local validation passed:
  `python -m py_compile minalphafold/simplex.py minalphafold/model_config.py scripts/run_nanofold_public_benchmarks.py`;
  `python -m pytest tests/test_simplex.py::test_vertex_star_cell_mean_pools_incident_selected_cells tests/test_simplex.py::test_vertex_star_context_routes_incident_cell_summary_without_extra_parameters tests/test_trainer.py::test_simplicial_vertex_star_context_adds_no_parameters tests/test_nanofold_public_benchmarks.py::test_model_config_override_flags_are_accepted_by_cli_parser`
  reported `4 passed`; and
  `python -m pytest tests/test_simplex.py tests/test_nanofold_public_benchmarks.py::test_model_config_override_flags_are_accepted_by_cli_parser tests/test_trainer.py::test_simplicial_global_context_stays_inside_af2_medium_budget tests/test_trainer.py::test_simplicial_vertex_star_context_adds_no_parameters`
  reported `66 passed`.
- E118 lint note: full
  `/Users/christopherhayduk/Projects/nanoFold-Competition/.venv/bin/ruff check ...`
  over the touched files is blocked by pre-existing style/import findings in
  `simplex.py`, `run_nanofold_public_benchmarks.py`, and tests. The narrower
  undefined-name gate
  `/Users/christopherhayduk/Projects/nanoFold-Competition/.venv/bin/ruff check --select F821,F822,F823 minalphafold/model_config.py minalphafold/simplex.py scripts/run_nanofold_public_benchmarks.py tests/test_nanofold_public_benchmarks.py tests/test_simplex.py tests/test_trainer.py`
  passed.
- E117 health poll after about 12 minutes: PID `13554` remained active on the
  owned pod, GPU memory was allocated (`13593 MiB`) and utilization sampled at
  `43%`. No `results.json` yet, and the artifact directory still contains
  only the inherited history plus run metadata. Continue waiting; do not
  launch E118 while E117 is active.
- E117 health poll after about 15.5 minutes: PID `13554` remained active on
  the owned pod. A sampled GPU utilization check happened to read `0%`, but
  the process was still running with high CPU and `13593 MiB` GPU memory
  allocated. No result artifacts yet; continue waiting for the step-6500
  validation write.
- E117 health poll after about 17 minutes: PID `13554` remained active on the
  owned pod with state `Rl`, about `1014%` CPU, 178 threads, and `13593 MiB`
  GPU memory allocated. No result artifacts yet; artifact mtimes still show
  only the inherited history and run metadata from launch.
- Updated heartbeat `check-simplexfold-e57-runpod` to keep monitoring only
  owned pod `o1dy17ouv8w5mz`, record E117 before any follow-up, and consider
  the prepared E118 vertex-star selected-complex context only after E117
  returns. The heartbeat should launch E118 only if E117 remains stable near
  or above the E96/E116 lDDT band; if E117 collapses, it should not launch a
  follow-up.
- E117 health poll after about 19 minutes: PID `13554` remained active on the
  owned pod with state `Rl`, about `1014%` CPU, 178 threads, and sampled GPU
  utilization `53%` with `13593 MiB` allocated. No `results.json`,
  `results.csv`, eval-detail file, or new checkpoint yet; this matches the
  long compute-only phase seen in E116. Keep waiting and do not launch E118.
- E117 health poll after about 21 minutes: PID `13554` remained active on the
  owned pod with state `Rl`, about `1014%` CPU, 178 threads, and `13593 MiB`
  GPU memory allocated. No `results.json`, `results.csv`, or eval-detail file
  yet; remote git dirt remains only the editable-install
  `minalphafold.egg-info/`. Continue waiting and leave E118 parked.
- E117 health poll after about 22 minutes: PID `13554` remained active on the
  owned pod with state `Sl`, about `1014%` CPU, 178 threads, sampled GPU
  utilization `28%`, and `13593 MiB` GPU memory allocated. No result files,
  eval-detail file, or `full_msa_to_face_latest.pt` checkpoint yet. Leave the
  process alone and keep E118 parked until E117 returns and is recorded.
- E117 health poll after about 24 minutes: PID `13554` remained active on the
  owned pod with state `Rl`, about `1015%` CPU, 178 threads, sampled GPU
  utilization `31%`, and `13593 MiB` GPU memory allocated. Still no result
  files, eval-detail file, or checkpoint. Continue waiting for the first
  step-6500 validation write.
- E117 health poll after about 25.5 minutes: PID `13554` remained active on
  the owned pod with state `Sl`, about `1015%` CPU, 178 threads, sampled GPU
  utilization `31%`, and `13593 MiB` GPU memory allocated. Still no result
  files, eval-detail file, or checkpoint; keep waiting and do not launch E118.
- E117 health poll after about 37.5 minutes: PID `13554` remained active on
  the owned pod with state `Sl`, about `1017%` CPU, 178 threads, sampled GPU
  utilization `47%`, and `13593 MiB` GPU memory allocated. The remote checkout
  is still at E117 launch commit `9548d5f` with only editable-install
  `minalphafold.egg-info/` dirt. The artifact directory still contains only
  `run_metadata.json` and inherited `history_full_msa_to_face.json`; no
  `results.json`, `results.csv`, eval-detail file, or checkpoint has returned.
- Clarified the E118 follow-up gate in `PLAN.md`, `EXPERIMENTS.md`, and the
  active heartbeat. If E117 is the source checkpoint, E118 should target the
  next 500-step gate at step 7000 with
  `--simplex-vertex-star-context-scale 1.0`; step 6500 is only correct when
  falling back to the E116 step-6000 checkpoint.
- E119 local implementation prepared while E117 runs. Added default-off
  `simplex_edge_star_context_scale`, which reuses the E116 global-context
  MLPs but interpolates their context toward selected boundary-edge star
  cochains pooled from incident face/tetra states. This is the edge-star
  analogue of E118's residue vertex-star route and stays inside the
  topological cochain view: selected cells communicate through their shared
  1-skeleton before returning to active cells and boundary-edge readout. It
  adds no parameters and should remain checkpoint-compatible with E116/E117.
- E119 local validation passed:
  `python -m py_compile minalphafold/simplex.py minalphafold/model_config.py scripts/run_nanofold_public_benchmarks.py`;
  `python -m pytest tests/test_simplex.py::test_edge_star_cell_mean_pools_cells_through_boundary_edges tests/test_simplex.py::test_edge_star_context_routes_boundary_edge_summary_without_extra_parameters tests/test_trainer.py::test_simplicial_edge_star_context_adds_no_parameters tests/test_nanofold_public_benchmarks.py::test_model_config_override_flags_are_accepted_by_cli_parser`
  reported `4 passed`;
  `python -m pytest tests/test_simplex.py tests/test_nanofold_public_benchmarks.py::test_model_config_override_flags_are_accepted_by_cli_parser tests/test_trainer.py::test_simplicial_global_context_stays_inside_af2_medium_budget tests/test_trainer.py::test_simplicial_vertex_star_context_adds_no_parameters tests/test_trainer.py::test_simplicial_edge_star_context_adds_no_parameters`
  reported `69 passed`; and
  `python -m pytest tests/test_simplex.py tests/test_nanofold_public_benchmarks.py tests/test_trainer.py`
  reported `196 passed`;
  `/Users/christopherhayduk/Projects/nanoFold-Competition/.venv/bin/ruff check --select F821,F822,F823 minalphafold/model_config.py minalphafold/simplex.py scripts/run_nanofold_public_benchmarks.py tests/test_nanofold_public_benchmarks.py tests/test_simplex.py tests/test_trainer.py`
  passed.
- Added standalone trainer CLI consistency for
  `--simplex-global-context-scale`, `--simplex-vertex-star-context-scale`,
  and `--simplex-edge-star-context-scale`, so the non-benchmark trainer can
  instantiate the same E116/E118/E119 context routes without requiring a
  custom TOML profile. Validation passed:
  `python -m py_compile minalphafold/trainer.py`;
  `python -m pytest tests/test_trainer.py::test_trainer_cli_accepts_simplex_star_context_overrides tests/test_trainer.py::test_simplicial_edge_star_context_adds_no_parameters tests/test_nanofold_public_benchmarks.py::test_model_config_override_flags_are_accepted_by_cli_parser`
  reported `3 passed`;
  `/Users/christopherhayduk/Projects/nanoFold-Competition/.venv/bin/ruff check --select F821,F822,F823 minalphafold/trainer.py tests/test_trainer.py`
  passed; and
  `python -m pytest tests/test_simplex.py tests/test_nanofold_public_benchmarks.py tests/test_trainer.py`
  reported `197 passed`.
- E117 health poll after about 55 minutes: PID `13554` remained active on the
  owned pod with state `Rl`, about `1008%` CPU, 178 threads, `13593 MiB` GPU
  memory allocated, and sampled GPU utilization `0%`. The remote checkout is
  still at E117 launch commit `9548d5f` with only editable-install
  `minalphafold.egg-info/` dirt. The artifact directory still contains only
  `run_metadata.json` and inherited `history_full_msa_to_face.json`; no
  `results.json`, `results.csv`, eval-detail file, or checkpoint has returned.
- Updated heartbeat `check-simplexfold-e57-runpod` to acknowledge that E119
  edge-star selected-complex context is prepared, but to keep E119 parked
  until E117 is recorded and E118 has either run or been explicitly skipped.
  The heartbeat should still launch at most E118 after E117, and only if E117
  stays stable near or above the E96/E116 lDDT band.
- E117 health poll at `2026-05-14T02:56:20Z`: PID `13554` remained active on
  owned pod `o1dy17ouv8w5mz` after about 66.7 minutes, with state `Rl`, about
  `1053%` CPU, 178 threads, `13593 MiB` GPU memory allocated, and sampled GPU
  utilization `33%`. The remote checkout is still at E117 launch commit
  `9548d5f` with only editable-install `minalphafold.egg-info/` dirt. The
  artifact directory still has `run_metadata.json` and inherited
  `history_full_msa_to_face.json`; no `results.json`, `results.csv`,
  eval-detail file, or checkpoint has returned. Keep E118 and E119 parked.
- E118/E119 readiness audit while E117 runs: config, benchmark runner CLI,
  standalone trainer CLI, metadata/results fields, docs, and tests all still
  reference the selected-complex global, vertex-star, and edge-star context
  flags. Fresh validation passed:
  `python -m py_compile minalphafold/simplex.py minalphafold/model_config.py minalphafold/trainer.py scripts/run_nanofold_public_benchmarks.py`;
  `python -m pytest tests/test_simplex.py::test_vertex_star_cell_mean_pools_incident_selected_cells tests/test_simplex.py::test_edge_star_cell_mean_pools_cells_through_boundary_edges tests/test_simplex.py::test_vertex_star_context_routes_incident_cell_summary_without_extra_parameters tests/test_simplex.py::test_edge_star_context_routes_boundary_edge_summary_without_extra_parameters tests/test_nanofold_public_benchmarks.py::test_model_config_override_flags_are_accepted_by_cli_parser tests/test_trainer.py::test_trainer_cli_accepts_simplex_star_context_overrides tests/test_trainer.py::test_simplicial_global_context_stays_inside_af2_medium_budget tests/test_trainer.py::test_simplicial_vertex_star_context_adds_no_parameters tests/test_trainer.py::test_simplicial_edge_star_context_adds_no_parameters`
  reported `9 passed`.
- 2026-05-14T03:09:21Z local prep while E117 runs: added runtime ramp support
  for the E118/E119 star-context routes. The new training/benchmark CLI flags
  are `--simplex-vertex-star-context-runtime-scale`,
  `--simplex-vertex-star-context-runtime-scale-final`,
  `--simplex-vertex-star-context-runtime-scale-ramp-start-step`,
  `--simplex-vertex-star-context-runtime-scale-ramp-steps`,
  `--simplex-edge-star-context-runtime-scale`,
  `--simplex-edge-star-context-runtime-scale-final`,
  `--simplex-edge-star-context-runtime-scale-ramp-start-step`, and
  `--simplex-edge-star-context-runtime-scale-ramp-steps`. This lets a resumed
  E118/E119 gate anneal from E116's protein-level selected-complex cochain
  toward vertex-star or edge-star incidence cochains, instead of abruptly
  changing the selected-complex routing at the checkpoint boundary. This adds
  no parameters and stays in the topological cochain/inter-incidence view.
- Star-context ramp validation passed:
  `python -m py_compile minalphafold/simplex.py minalphafold/evoformer.py minalphafold/model.py minalphafold/trainer.py scripts/run_nanofold_public_benchmarks.py`;
  `python -m pytest tests/test_simplex.py::test_star_context_runtime_overrides_gate_context_route tests/test_simplex.py::test_vertex_star_context_routes_incident_cell_summary_without_extra_parameters tests/test_simplex.py::test_edge_star_context_routes_boundary_edge_summary_without_extra_parameters tests/test_trainer.py::test_model_inputs_add_training_only_simplex_curricula tests/test_trainer.py::test_trainer_cli_accepts_simplex_star_context_overrides tests/test_nanofold_public_benchmarks.py::test_model_config_override_flags_are_accepted_by_cli_parser tests/test_nanofold_public_benchmarks.py::test_runtime_simplex_message_scales_ramp_and_enter_model_inputs tests/test_nanofold_public_benchmarks.py::test_evaluate_uses_runtime_simplex_overrides_for_validation`
  reported `8 passed`; `python -m pytest tests/test_simplex.py tests/test_nanofold_public_benchmarks.py tests/test_trainer.py`
  reported `198 passed`; and
  `/Users/christopherhayduk/Projects/nanoFold-Competition/.venv/bin/ruff check --select F821,F822,F823 minalphafold/simplex.py minalphafold/evoformer.py minalphafold/model.py minalphafold/trainer.py scripts/run_nanofold_public_benchmarks.py tests/test_simplex.py tests/test_nanofold_public_benchmarks.py tests/test_trainer.py`
  passed.
- 2026-05-14T03:10:51Z heartbeat `check-simplexfold-e57-runpod` updated to
  launch any post-E117 E118 gate with the new vertex-star runtime ramp after
  fast-forwarding the remote checkout to the current branch tip. If E117 is
  the source checkpoint, E118 should target step 7000 and ramp
  `--simplex-vertex-star-context-runtime-scale` from `0.0` to `1.0` over
  steps 6500-7000; if falling back to E116 step 6000, target step 6500 and
  ramp over steps 6000-6500. E119 remains parked.
- 2026-05-14T03:16:50Z E117 returned on owned pod `o1dy17ouv8w5mz`.
  Remote coherence passed before pulling artifacts: `results.json`,
  `results.csv`, `history_full_msa_to_face.json`,
  `eval_details_full_msa_to_face.csv`, `run_metadata.json`, and the source
  checkpoint were present; `completed_steps=6500`, `effective_batch_size=8`,
  `parameters=3,201,970 <= 3,261,974`,
  `simplex_global_context_scale=0.1`, `stopped_early=False`, 1000 eval-detail
  rows, and history ended at step 6500. Final metrics:
  `val_lddt_ca=0.4151`, FoldScore `0.3927`, `val_ca_drmsd=11.2046`, C-alpha
  Rg `11.5744 / 16.3091`, selected face/tetra boundary lDDT
  `0.7342 / 0.7174`, and selected face/tetra contraction
  `0.6359 / 0.6340`.
- Pulled E117 non-checkpoint artifacts and log locally. Local coherence passed
  with the same checks and confirmed no checkpoint directory was pulled. E117
  is the new primary-lDDT leader and is stable enough to launch E118, but it is
  still not a 30k candidate because the branch remains in the low-0.4 lDDT
  band rather than breaking toward the 0.7 target.
- Updated `EXPERIMENT_RESULTS.md`, `PLAN.md`, and `EXPERIMENTS.md` to record
  E117 as final and to make E118 the next active gate. The E118 launch should
  fast-forward the remote checkout to the current branch tip, resume from the
  E117 step-6500 checkpoint, target step 7000, keep
  `--simplex-global-context-scale 0.10` and
  `--simplex-vertex-star-context-scale 1.0`, and ramp
  `--simplex-vertex-star-context-runtime-scale` from `0.0` to `1.0` over
  steps 6500-7000.
- 2026-05-14T03:22:55Z launched E118 on owned pod `o1dy17ouv8w5mz` after
  fast-forwarding the remote checkout to commit `3013d4f`. Run name:
  `e118_vertex_star_context_from_e117_s7000_c256_m64`; PID `26269`; log
  `/workspace/SimplexFold/logs/e118_vertex_star_context_from_e117.log`;
  artifacts
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e118_vertex_star_context_from_e117_s7000_c256_m64/`.
  The source checkpoint is the returned E117 step-6500 checkpoint. Startup
  confirmed resume at step 6500/examples 52000, `1292` matching tensors loaded,
  and `0` new/missing tensors. Metadata confirmed `steps=7000`,
  `effective_batch_size=8`, `max_parameters=3261974`,
  `simplex_global_context_scale=0.1`,
  `simplex_vertex_star_context_scale=1.0`, and the vertex-star runtime ramp
  from `0.0` to `1.0` over steps 6500-7000. E119 remains parked until E118
  returns or is explicitly skipped.
- 2026-05-14T04:58:51Z E118 returned on owned pod `o1dy17ouv8w5mz`.
  Remote coherence passed before pulling artifacts: `results.json`,
  `results.csv`, `history_full_msa_to_face.json`,
  `eval_details_full_msa_to_face.csv`, `run_metadata.json`, and the source
  checkpoint were present; `completed_steps=7000`, `effective_batch_size=8`,
  `parameters=3,201,970 <= 3,261,974`,
  `simplex_global_context_scale=0.1`,
  `simplex_vertex_star_context_scale=1.0`, vertex-star runtime ramp `0.0` to
  `1.0` over steps 6500-7000, `stopped_early=False`, 1000 eval-detail rows,
  and history ended at step 7000. Final metrics: `val_lddt_ca=0.4190`,
  FoldScore `0.3955`, `val_ca_drmsd=11.2342`, C-alpha Rg
  `11.3877 / 16.3091`, selected face/tetra boundary lDDT
  `0.7404 / 0.7238`, and selected face/tetra contraction
  `0.6120 / 0.6125`.
- Pulled E118 non-checkpoint artifacts and log locally. Local coherence passed
  with the same checks and confirmed no checkpoint directory was pulled. E118
  is the new primary-lDDT leader and improves E117, so it is stable enough to
  launch E119 from the returned step-7000 checkpoint. It remains far below the
  0.7 target and is still not a 30k candidate.
- Updated `EXPERIMENT_RESULTS.md`, `PLAN.md`, and `EXPERIMENTS.md` to record
  E118 as final and to make E119 the next active gate. The E119 launch should
  fast-forward the remote checkout to the current branch tip, resume from the
  E118 step-7000 checkpoint, target step 7500, keep
  `--simplex-global-context-scale 0.10`,
  `--simplex-edge-star-context-scale 1.0`, and ramp
  `--simplex-edge-star-context-runtime-scale` from `0.0` to `1.0` over
  steps 7000-7500. Do not also enable vertex-star context in this first
  edge-star gate.
- 2026-05-14T05:03:44Z launched E119 on owned pod `o1dy17ouv8w5mz` after
  fast-forwarding the remote checkout to commit `91ebe38`. Run name:
  `e119_edge_star_context_from_e118_s7500_c256_m64`; PID `38461`; log
  `/workspace/SimplexFold/logs/e119_edge_star_context_from_e118.log`;
  artifacts
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e119_edge_star_context_from_e118_s7500_c256_m64/`.
  The source checkpoint is the returned E118 step-7000 checkpoint. Startup
  confirmed resume at step 7000/examples 56000, `1292` matching tensors loaded,
  and `0` new/missing tensors. Metadata confirmed `steps=7500`,
  `effective_batch_size=8`, `max_parameters=3261974`,
  `simplex_global_context_scale=0.1`, `simplex_edge_star_context_scale=1.0`,
  `simplex_vertex_star_context_scale=null`, and the edge-star runtime ramp
  from `0.0` to `1.0` over steps 7000-7500.
- 2026-05-14T06:43:39Z E119 returned on owned pod `o1dy17ouv8w5mz`.
  Remote coherence passed before pulling artifacts after interpreting the
  result row's runtime fields as ramp start `0.0` and final `1.0`:
  `results.json`, `results.csv`, `history_full_msa_to_face.json`,
  `eval_details_full_msa_to_face.csv`, `run_metadata.json`, and the checkpoint
  were present; `completed_steps=7500`, `effective_batch_size=8`,
  `parameters=3,201,970 <= 3,261,974`,
  `simplex_global_context_scale=0.1`,
  `simplex_edge_star_context_scale=1.0`,
  `simplex_vertex_star_context_scale=0.0`, edge-star runtime ramp `0.0` to
  `1.0` over steps 7000-7500, `stopped_early=False`, 1000 eval-detail rows,
  and history ended at step 7500. Final metrics: `val_lddt_ca=0.4181`,
  FoldScore `0.3957`, `val_ca_drmsd=11.0494`, C-alpha Rg
  `11.8732 / 16.3091`, selected face/tetra boundary lDDT
  `0.7428 / 0.7275`, and selected face/tetra contraction
  `0.6430 / 0.6391`.
- Pulled E119 non-checkpoint artifacts and log locally. Local coherence passed
  with the same checks and confirmed no checkpoint directory was pulled. E119
  improves FoldScore, dRMSD, expansion, and selected-boundary diagnostics
  versus E118, but primary C-alpha lDDT regressed from E118's `0.4190` to
  `0.4181`. Reject E119 as a 30k candidate and record the star-context family
  as still stuck in the low-0.4 lDDT band.
- Updated `EXPERIMENT_RESULTS.md`, `PLAN.md`, and `EXPERIMENTS.md` to record
  E119 as final. The next experiment should be a new topology-native
  local-to-global bridge that changes how selected face/tetra cochains affect
  pair/edge geometry before structure readout; do not launch a longer E118 or
  E119 continuation unless a short gate first breaks out of the low-0.4 band.
- Preserved the E118 and E119 restart checkpoints locally under ignored
  artifact paths before stopping the owned pod:
  `artifacts/nanofold_public_benchmarks/e118_vertex_star_context_from_e117_s7000_c256_m64/checkpoints/full_msa_to_face_latest.pt`
  and
  `artifacts/nanofold_public_benchmarks/e119_edge_star_context_from_e118_s7500_c256_m64/checkpoints/full_msa_to_face_latest.pt`.
  Both are about 35 MB and remain outside git. Paused the E119 heartbeat and
  stopped only the owned Runpod pod `o1dy17ouv8w5mz` after verifying no Python
  benchmark process was active.
- Prepared E120 as a mixed star-context short gate rather than a longer E118
  or E119 continuation. The run should resume the E118 step-7000 checkpoint,
  keep `--simplex-vertex-star-context-scale 1.0` with runtime scale `1.0`,
  allocate edge-star context with `--simplex-edge-star-context-scale 1.0`, and
  ramp `--simplex-edge-star-context-runtime-scale` from `0.0` to `0.5` over
  steps 7000-7500. This tests whether the edge-star route's better
  pair-interface packing signal can help without replacing the vertex-star
  route that currently gives the best primary C-alpha lDDT.
- 2026-05-14T07:08:41Z launched E120 on owned pod `o1dy17ouv8w5mz` after
  restarting and restaging the zero-volume workspace. Current SSH endpoint:
  `103.207.149.82:10779` with key
  `/Users/christopherhayduk/.runpod/ssh/RunPod-Key-Go`. Run name:
  `e120_mixed_star_context_from_e118_s7500_c256_m64`; PID `1274`; log
  `/workspace/SimplexFold/logs/e120_mixed_star_context_from_e118.log`;
  artifacts
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e120_mixed_star_context_from_e118_s7500_c256_m64/`.
  The first launch was stopped immediately because the data-only restage left
  `nanofold` unavailable for FoldScore component imports. The clean launch
  cloned `nanoFold-Competition` source at commit `96afc84`, set
  `PYTHONPATH=/workspace/nanoFold-Competition-src`, and no longer emitted the
  missing-`nanofold` warning.
- E120 remote prelaunch audit passed: SimplexFold commit `4e75148`, clean
  remote status, `11000/11000` feature/label NPZs, manifest rows
  `10000/1000/11000`, zero `._*` sidecars, restored E118 checkpoint present,
  py_compile passed, and launch-style parameters were `3,201,970` under the
  `3,261,974` cap. Startup health confirmed resume from E118 at step
  7000/examples 56000, `1292` matching tensors loaded, `0` new/missing tensors
  initialized, `effective_batch_size=8`, `simplex_global_context_scale=0.1`,
  `simplex_vertex_star_context_scale=1.0`, vertex-star runtime scale `1.0`,
  `simplex_edge_star_context_scale=1.0`, and edge-star runtime ramp `0.0` to
  `0.5` over steps 7000-7500.
- E120 health at `2026-05-14T07:17:28Z`: PID `1274` still active, no
  `results.json`, and log remains in the expected quiet compute phase after
  clean E118 resume. Do not touch the run or launch follow-up while E120 is
  active.
- Implemented E121 locally as default-off
  `simplex_pre_triangle_update_scale`. The change reuses the existing
  SimplicialAdapter before the AF2 pair triangle stack inside each enabled
  SimplicialEvoformer block, so selected face/tetra cochains can write to
  `Z_ij` before triangle multiplication/attention globalizes pair evidence.
  This adds no loss and no parameters; it is meant to address the recurring
  gap between high selected-boundary diagnostics and low global C-alpha lDDT.
- E121 validation passed:
  `python -m py_compile minalphafold/evoformer.py minalphafold/model_config.py minalphafold/trainer.py scripts/run_nanofold_public_benchmarks.py`;
  `python -m pytest tests/test_simplex.py::test_pre_triangle_simplex_update_changes_evoformer_block_outputs_without_new_state tests/test_trainer.py::test_trainer_cli_accepts_simplex_star_context_overrides tests/test_trainer.py::test_simplicial_pre_triangle_update_adds_no_parameters tests/test_nanofold_public_benchmarks.py::test_model_config_override_flags_are_accepted_by_cli_parser`
  reported `4 passed`;
  `python -m pytest tests/test_simplex.py tests/test_nanofold_public_benchmarks.py tests/test_trainer.py`
  reported `200 passed`;
  `/Users/christopherhayduk/Projects/nanoFold-Competition/.venv/bin/ruff check --select F821,F822,F823 minalphafold/evoformer.py minalphafold/model_config.py minalphafold/trainer.py scripts/run_nanofold_public_benchmarks.py tests/test_simplex.py tests/test_trainer.py tests/test_nanofold_public_benchmarks.py`
  passed. Park E121 until E120 returns and is recorded.
- E120 health at `2026-05-14T07:25:45Z`: PID `1274` still active on owned
  pod `o1dy17ouv8w5mz`, no `results.json`, artifacts still limited to
  `history_full_msa_to_face.json` and `run_metadata.json`, and history still
  ends at inherited step 7000 with `val_lddt_ca=0.4190087939500809`. A
  spot-check of `nvidia-smi` at `2026-05-14T07:26:43Z` showed the H100 memory
  allocated (`13599 / 81559` MiB) while the process was in a high-CPU quiet
  training phase. Continue monitoring only this pod; do not touch other
  Runpod instances.
- 30k candidate assessment while E120 is active: no returned candidate has
  earned a 30,000-step spend yet. The best returned primary metric is E118's
  `0.4190` at step 7000, so reaching `0.7` by step 30000 would require about
  `+0.281` absolute validation C-alpha lDDT, or roughly `+0.0122` per 1000
  steps over the remaining 23000 steps. Recent star-context gains are below
  that slope and noisy: E116 -> E118 gained about `+0.0095` across 1000 steps,
  then E119 improved several geometric diagnostics while slightly regressing
  primary lDDT. Treat E120 as a short gate only; if it does not beat E118 and
  leave the low-0.4 band, the next topology-native test is the parked E121
  pre-triangle simplex update rather than a longer continuation.
- E120 health at `2026-05-14T07:57:49Z`: still active on owned pod
  `o1dy17ouv8w5mz` with PID `1274`, elapsed `49:08`, accumulated process CPU
  time `07:45:27`, and no `results.json`. The history file still ends at step
  7000. This is slower than earlier short gates, but a non-invasive health
  snapshot showed continuing process reads and live H100 activity with
  utilization samples between `29%` and `52%` around 13.6 GiB allocated. Treat
  the run as long-running rather than stalled; leave it alive and let the
  heartbeat or next manual poll catch the 7500-step return.
- E120 health at `2026-05-14T08:34:17Z`: still active on owned pod
  `o1dy17ouv8w5mz` with PID `1274`, elapsed `01:25:36`, accumulated process
  CPU time `14:59:05`, and no `results.json`. Code inspection explains why the
  artifact directory can remain quiet until the end: after resuming from step
  7000, this benchmark loop only logs/evaluates/checkpoints on `step % 500 == 0`
  or final step, so the next expected history row is step 7500. The mixed
  vertex/edge-star route is also computationally heavier than the isolated
  vertex-star route because `edge_star_cell_mean` materializes dense `L x L`
  boundary-edge star cochains for both face and tetra states before gathering
  selected boundary edges back into cells. Treat E120 as a slow active gate,
  not a failed launch; do not stop or replace it unless the process exits
  without coherent step-7500 artifacts.
- Implemented a local sparse edge-star context path for future gates. The new
  `boundary_edge_star_context` computes the same selected boundary-edge star
  means that the dense `edge_star_cell_mean` path would gather, but it only
  gathers requested target cell boundary edges instead of materializing a full
  `L x L` cochain tensor. This is topology-equivalent to the E119/E120
  boundary-edge-star route, adds no parameters and no losses, and does not
  affect the already-running E120 remote checkout. Validation passed:
  `python -m py_compile minalphafold/simplex.py`;
  `python -m pytest tests/test_simplex.py::test_edge_star_cell_mean_pools_cells_through_boundary_edges tests/test_simplex.py::test_boundary_edge_star_context_matches_dense_edge_star_gather tests/test_simplex.py::test_edge_star_context_routes_boundary_edge_summary_without_extra_parameters tests/test_simplex.py::test_star_context_runtime_overrides_gate_context_route`
  reported `4 passed`;
  `python -m pytest tests/test_simplex.py` reported `68 passed`;
  `/Users/christopherhayduk/Projects/nanoFold-Competition/.venv/bin/ruff check --select F821,F822,F823 minalphafold/simplex.py tests/test_simplex.py`
  passed.
- 2026-05-14T08:43Z E120 returned on owned pod `o1dy17ouv8w5mz`. Remote
  coherence passed before pulling artifacts: `results.json`, `results.csv`,
  `history_full_msa_to_face.json`, `eval_details_full_msa_to_face.csv`,
  `run_metadata.json`, and
  `checkpoints/full_msa_to_face_latest.pt` were present;
  `completed_steps=7500`, `effective_batch_size=8`,
  `parameters=3,201,970 <= 3,261,974`, `stopped_early=False`, 1000
  eval-detail rows, history ended at step 7500,
  `simplex_global_context_scale=0.1`,
  `simplex_vertex_star_context_scale=1.0`, vertex-star runtime scale `1.0`,
  `simplex_edge_star_context_scale=1.0`, and the edge-star runtime ramp was
  `0.0` to `0.5` over steps 7000-7500. Final metrics:
  `val_lddt_ca=0.4248`, FoldScore `0.3983`, `val_ca_drmsd=11.1450`,
  C-alpha Rg `11.4973 / 16.3091`, selected face/tetra boundary lDDT
  `0.7548 / 0.7383`, and selected face/tetra boundary contraction
  `0.5845 / 0.5826`.
- Pulled E120 artifacts, log, and the step-7500 checkpoint locally. Local
  coherence passed with the same checks. E120 is the new primary-lDDT and
  FoldScore leader, and it shows that mixed vertex/edge-star cochain routing
  can improve the topology-native branch. It is still not a 30,000-step
  candidate: the model remains in the low-0.4 C-alpha lDDT band, C-alpha Rg
  is still under-expanded, and reaching `0.7` would require roughly `+0.275`
  absolute lDDT. Keep the E120 checkpoint as the strongest restart point, but
  use E121 pre-triangle simplex injection as the next short-gate idea rather
  than extending E120 blindly.
- Stopped only the owned Runpod pod `o1dy17ouv8w5mz` after verifying no
  benchmark Python process was active and preserving the E120 checkpoint
  locally. Paused the E120 heartbeat, then restarted the same owned pod for
  the next gate; the zero-volume workspace came back empty, so the remote was
  restaged from local source with public train/val feature and label caches
  only. Remote prelaunch audit for E121 passed: SimplexFold commit `34ef7d7`,
  clean status, `11000/11000` feature/label NPZs, manifest rows
  `10000/1000/11000`, zero `._*` sidecars, the E120 checkpoint present, and
  launch-style parameters `3,201,970 <= 3,261,974`.
- 2026-05-14T08:54Z launched E121 on owned Runpod pod `o1dy17ouv8w5mz` using
  SSH endpoint `103.207.149.82:19763` and key
  `/Users/christopherhayduk/.runpod/ssh/RunPod-Key-Go`. Run name:
  `e121_pre_triangle_from_e120_s8000_c256_m64`; PID `1051`; log
  `/workspace/SimplexFold/logs/e121_pre_triangle_from_e120.log`; artifacts
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e121_pre_triangle_from_e120_s8000_c256_m64/`.
  The run resumes the E120 step-7500 checkpoint to step 8000 with the E120
  mixed selected-complex recipe fixed and adds
  `--simplex-pre-triangle-update-scale 0.25`. Startup verification confirmed
  resume at step 7500/examples 60000, `1292` matching tensors loaded, `0`
  new/missing tensors, `effective_batch_size=8`,
  `simplex_global_context_scale=0.1`,
  `simplex_vertex_star_context_scale=1.0`, vertex-star runtime scale `1.0`,
  `simplex_edge_star_context_scale=1.0`, edge-star runtime scale `0.5`, and
  the intended pre-triangle scale `0.25`.
- E121 health poll after about 3 minutes: PID `1051` remained active on the
  owned pod and `results.json` was still absent. The remote checkout is at
  commit `34ef7d7`, which is fine for the active run because the later
  `a6b5b86` commit only recorded launch docs. Continue waiting and do not
  launch follow-up while E121 is active.
- Prepared a default-off E122 pair-only pre-triangle fallback locally while
  E121 runs. The new `simplex_pre_triangle_single_update_scale` keeps E121's
  behavior by default with `-1.0`, but allows a follow-up gate to set the
  pre-triangle single scale to `0.0` while keeping
  `simplex_pre_triangle_update_scale=0.25` for pair/edge `Z_ij` injection.
  This keeps the topology-native claim focused on selected face/tetra
  boundary cochains entering the AF2 triangle stack, adds no parameters, and
  avoids adding any new loss. Validation passed:
  `python -m py_compile minalphafold/evoformer.py minalphafold/model_config.py minalphafold/trainer.py scripts/run_nanofold_public_benchmarks.py`;
  `python -m pytest tests/test_simplex.py::test_pre_triangle_simplex_update_changes_evoformer_block_outputs_without_new_state tests/test_simplex.py::test_pre_triangle_simplex_update_can_run_pair_only tests/test_trainer.py::test_trainer_cli_accepts_simplex_star_context_overrides tests/test_trainer.py::test_simplicial_pre_triangle_update_adds_no_parameters tests/test_nanofold_public_benchmarks.py::test_model_config_override_flags_are_accepted_by_cli_parser`
  reported `5 passed`.
- 2026-05-14T09:18Z E121 health poll: PID `1051` is still active on owned
  Runpod pod `o1dy17ouv8w5mz`, no `results.json` yet, and the log still shows
  the expected post-resume quiet training phase from step 7500. Do not sync
  new local code into the remote checkout while E121 is running.
- Prepared E123 locally as a ramped pair-only pre-triangle fallback. The new
  runtime override fields schedule `simplex_pre_triangle_update_scale` and
  `simplex_pre_triangle_single_update_scale` through the trainer and NanoFold
  public benchmark runner, then route those per-step values into
  `AlphaFold2.forward` and `SimplicialEvoformer.forward`. The intended follow-up
  is topology-native and zero-parameter: gradually let selected face/tetra
  boundary cochains update pair `Z_ij` before AF2 triangle operations, while
  optionally holding the pre-triangle single update at zero. Awaiting tests
  before commit.
- E123 local validation passed:
  `python -m py_compile minalphafold/evoformer.py minalphafold/model.py minalphafold/trainer.py scripts/run_nanofold_public_benchmarks.py`;
  focused runtime/pre-triangle tests reported `9 passed`;
  `python -m pytest tests/test_simplex.py tests/test_nanofold_public_benchmarks.py tests/test_trainer.py`
  reported `202 passed`;
  `/Users/christopherhayduk/Projects/nanoFold-Competition/.venv/bin/ruff check --select F821,F822,F823 minalphafold/evoformer.py minalphafold/model.py minalphafold/trainer.py scripts/run_nanofold_public_benchmarks.py tests/test_simplex.py tests/test_trainer.py tests/test_nanofold_public_benchmarks.py`
  passed. E123 is ready to launch only after E121b returns.
- 2026-05-14 full PDF reread completed while E121 was active. The saved
  PDFs in `references/papers/` still hash-match the Downloads originals and
  remain intentionally git-ignored. The current takeaways for the E121/E123
  family are: pre-triangle simplex injection is justified as inter-rank
  cochain communication into the pair 1-skeleton before triangle reasoning;
  the next paper-derived architecture idea should be selected-complex
  edge-centric scalarization onto boundary/outer-edge frames; and dense
  C-alpha/radius/all-pairs losses remain out of scope unless restricted to
  realization of the model-selected sparse complex.
- E121 health poll after about 26 minutes: PID `1051` remains active on the
  owned Runpod pod, `results.json` is still absent, and
  `history_full_msa_to_face.json` still ends at the inherited E120 step-7500
  row (`val_lddt_ca=0.4248279729783535`). This is expected until E121 reaches
  its step-8000 eval/checkpoint boundary. Leave the run alone.
- E121 failed before a new validation point. Remote PID `1051` exited with no
  `results.json`, no step-8000 history row, and no new checkpoint. The log
  traceback ended in `torch.utils.checkpoint.CheckpointError`: checkpoint
  recomputation saw selected-complex packed tensors with different metadata,
  including shapes `[4399, 1]` in the original forward and `[4403, 1]` during
  recomputation. Pulled the failed log plus `history_full_msa_to_face.json`
  and `run_metadata.json` locally for audit. This is an implementation failure
  of the activation-checkpointed dynamic pre-triangle path, not an experiment
  result; do not update the leader.
- Implemented the E121b corrective fix locally: active pre-triangle simplex
  blocks run eagerly during training instead of through activation
  checkpointing. This keeps the scientific intervention unchanged while
  avoiding recomputation over variable-size selected-cell tensors. Validation
  passed: `python -m py_compile minalphafold/model.py`; focused
  pre-triangle/checkpoint tests reported `5 passed`; ruff undefined-name check
  over `minalphafold/model.py tests/test_trainer.py` passed.
- Broader E121b validation passed: `python -m pytest tests/test_simplex.py tests/test_nanofold_public_benchmarks.py tests/test_trainer.py`
  reported `203 passed`; `python -m py_compile minalphafold/evoformer.py minalphafold/model.py minalphafold/trainer.py scripts/run_nanofold_public_benchmarks.py`
  passed; the full touched-file ruff undefined-name check passed; and
  `git diff --check` passed.
- 2026-05-14T10:03Z E121b health poll: owned Runpod pod `o1dy17ouv8w5mz`
  is running remote commit `a598e32` with a clean worktree. Active run
  `e121b_pre_triangle_eager_from_e120_s8000_c256_m64` has PID `1845`; no
  `results.json` exists yet; `history_full_msa_to_face.json` still has 16
  rows and ends at the inherited E120 step-7500 row
  (`val_lddt_ca=0.4248279729783535`, FoldScore `0.3983173668086529`,
  `val_ca_drmsd=11.145049806624652`). The heartbeat
  `check-simplexfold-e57-runpod` is active and already points to E121b with
  the owned-pod-only scope and remote/local coherence gates. Leave E121b
  running and do not launch E122/E123 until it returns.
- 2026-05-14T10:11Z prepared E124 locally while E121b runs. New default-off
  knob: `simplex_boundary_edge_frame_gate_scale`. The adapter scalarizes each
  selected face boundary edge in its directed edge frame, concatenates that
  oriented geometry with the face cochain and current boundary `Z_ij`, and
  gates only the selected face-to-edge message before scatter. This is a
  topology-native cochain communication change, not a new lDDT loss. A first
  face+tetra version exceeded the AF2-medium +5% cap (`3,275,762 > 3,261,974`),
  so the parked version is face-boundary-only and audits at `3,239,522`
  parameters under the cap. Validation passed:
  `python -m py_compile minalphafold/simplex.py minalphafold/model_config.py minalphafold/trainer.py scripts/run_nanofold_public_benchmarks.py`;
  focused E124/parser/budget tests reported `7 passed`, including a
  tetra-enabled regression case for the parked face-boundary gate;
  `python -m pytest tests/test_simplex.py tests/test_nanofold_public_benchmarks.py tests/test_trainer.py`
  reported `206 passed`; the touched-file ruff undefined-name check passed;
  and `git diff --check` passed. Do not launch E124 while E121b is active;
  consider it only after E121b/E123 clarify whether pre-triangle/cochain
  communication is worth another branch.
- 2026-05-14T10:14Z E121b health poll: PID `1845` is still active on owned
  pod `o1dy17ouv8w5mz`; `results.json` is absent and history still ends at
  the inherited E120 step-7500 row. The process appears to be actively
  training rather than wedged: `nvidia-smi` reported H100 GPU utilization
  `38%`, memory `42,875 / 81,559` MiB, and power draw `179.76 W`; the Python
  process showed high CPU while the log/history mtimes remained at launch
  time. Keep waiting for the step-8000 eval/checkpoint boundary.
- 2026-05-14T10:25Z E121b still has no `results.json`; PID `1845` remains
  active on the owned H100 and history still ends at E120 step 7500. While
  leaving the remote undisturbed, tightened the local launch decision notes:
  E123 should be the first fallback if E121b is weak, using pair-only
  pre-triangle injection ramped from `0.0` to `0.25` over steps 7500-8000;
  E124 should remain a later face boundary-edge-frame gate at scale `0.05`
  unless the pre-triangle family gives a reason to spend that extra budget.
- 2026-05-14T10:30Z 30k-candidate assessment while E121b is active: there is
  still no returned branch that deserves a 30,000-step confirmation. E120 is
  the best returned score at `val_lddt_ca=0.4248`, so reaching `0.7` by step
  30000 would require a qualitative local-to-global assembly improvement, not
  just a smoother continuation. Treat E121b/E123 as plausible short gates
  because pre-triangle simplex cochain injection is topology-native and gives
  AF2 triangle reasoning earlier access to selected face/tetra evidence. Use
  a stricter spend gate: below `0.45` at step 8000 means reject as another
  low-0.4 branch; `0.45`-`0.50` needs another short confirmation; above
  `0.50` with coherent FoldScore/dRMSD improvement is the first credible
  30k-candidate signal. A tiny `0.005`-style gain over E120 is not enough.
- 2026-05-14T10:44Z added a reusable returned-artifact verifier at
  `scripts/verify_nanofold_benchmark_artifacts.py` so future E121b/E123
  result handoffs can check the same coherence gates before editing
  `EXPERIMENT_RESULTS.md`. The verifier checks required result, metadata,
  history, eval-detail, and checkpoint files; expected step, effective batch
  size, parameter cap, stopped-early state, eval-detail row count, history
  endpoint, metadata key/value pairs, and finite primary metrics. Validation
  passed: `python -m py_compile scripts/verify_nanofold_benchmark_artifacts.py tests/test_verify_nanofold_benchmark_artifacts.py`;
  `python -m pytest tests/test_verify_nanofold_benchmark_artifacts.py`
  reported `3 passed`; ruff undefined-name check passed; `git diff --check`
  passed; and the verifier accepted the known-good E120 artifact directory
  with `completed_steps=7500`, effective batch size `8`, `1000` eval-detail
  rows, `3,201,970` parameters under cap, and history ending at step `7500`.
- 2026-05-14T10:54Z E121b health poll: PID `1845` is still alive on the owned
  Runpod pod after about `54:37` elapsed. `results.json` is absent, artifacts
  remain limited to `run_metadata.json` and `history_full_msa_to_face.json`,
  and history still ends at the inherited E120 step-7500 row
  (`val_lddt_ca=0.4248279729783535`, FoldScore `0.3983173668086529`,
  `val_ca_drmsd=11.145049806624652`). GPU utilization was active at `35%`
  with `42,875 / 81,559` MiB allocated. Leave E121b running; do not launch
  E123/E124 until the step-8000 result is coherently returned.
- 2026-05-14T10:58Z tightened the returned-artifact verifier to require a
  matching `results.csv` row for the requested variant and to optionally
  enforce the total `results.csv` row count. This closes a small gap where
  `results.json` could be coherent while the CSV tracker artifact was merely
  present. Validation passed: `python -m py_compile scripts/verify_nanofold_benchmark_artifacts.py tests/test_verify_nanofold_benchmark_artifacts.py`;
  `python -m pytest tests/test_verify_nanofold_benchmark_artifacts.py`
  reported `5 passed`; ruff undefined-name check passed; `git diff --check`
  passed; and the verifier accepted the known-good E120 artifact with
  `--expected-results-rows 1`. The E121b heartbeat now includes that stricter
  `--expected-results-rows 1` gate before any results-table update.
- 2026-05-14T11:10Z E121b remains active on the owned Runpod pod after about
  `01:11:07` elapsed. `results.json` is still absent, remote artifacts remain
  `run_metadata.json` and `history_full_msa_to_face.json`, and the history
  endpoint is still the inherited E120 step-7500 row. GPU utilization was
  active at `48%` with `42,875 / 81,559` MiB allocated. The log still shows
  only startup/resume lines, so treat this as a slow in-flight eager
  pre-triangle gate rather than a returned or failed experiment. Do not launch
  E123/E124 while this PID remains alive.
- 2026-05-14T11:33Z E121b returned coherently on the owned Runpod pod at step
  8000. Remote coherence passed: no active benchmark process, required result,
  metadata, history, eval-detail, and checkpoint files present, one
  `results.csv` row, 1000 eval-detail rows, history ending at step 8000,
  `effective_batch_size=8`, `parameters=3,201,970 <= 3,261,974`, and
  `stopped_early=False`. The raw returned score was
  `val_lddt_ca=0.4222809443175793`, below E120's `0.4248279729783535` and
  below the `0.45` short-gate threshold.
- 2026-05-14T12:20Z repaired the missing FoldScore field for E121b without
  changing the scientific result. The run log showed `FoldScore components
  unavailable: No module named 'nanofold'`, so the owned pod's public
  NanoFold package was synced and a post-hoc full validation evaluation was
  run from the saved E121b checkpoint. Recomputed C-alpha lDDT was
  `0.42227677324414253`, within `4.2e-6` of the returned value, and supplied
  `val_foldscore=0.4007264577746391`. Pulled the amended artifacts locally;
  `scripts/verify_nanofold_benchmark_artifacts.py` passed for E121b with the
  expected step, batch size, parameter cap, result/eval/history counts,
  checkpoint, metadata, and finite metrics. Record E121b as rejected and make
  E123 the next queued Runpod gate.
- 2026-05-14T12:25Z launched E123 on the owned Runpod pod `o1dy17ouv8w5mz`
  as `e123_ramped_pair_pre_triangle_from_e120_s8000_c256_m64`, PID `21142`.
  Remote checkout fast-forwarded cleanly to `5fce7f1`, no active benchmark
  process was present, the E120 checkpoint existed, py_compile passed for the
  relevant model/trainer/runner files, and the launch-style parameter audit
  counted `3,201,970 <= 3,261,974`. Startup health confirmed a clean
  weights-only resume at step 7500/examples 60000 with `1292` matching tensors
  loaded and `0` new/missing tensors. Metadata confirms the intended
  pair-only pre-triangle ramp: static pair scale `0.25`, static single scale
  `0.0`, pair runtime `0.0 -> 0.25` over steps 7500-8000, and single runtime
  held at `0.0`.
- 2026-05-14T13:20Z E123 remains active on owned Runpod pod
  `o1dy17ouv8w5mz` after about `54:57` elapsed. `results.json` and
  `results.csv` are still absent, artifacts remain limited to
  `run_metadata.json` and `history_full_msa_to_face.json`, and the history
  endpoint is still the inherited E120 step-7500 row
  (`val_lddt_ca=0.4248279729783535`, FoldScore `0.3983173668086529`,
  `val_ca_drmsd=11.145049806624652`). The process is CPU-active, so treat
  this as an in-flight ramped pair-only pre-triangle gate rather than a
  returned or failed experiment. Do not launch E124 while this PID remains
  active.
- 2026-05-14T14:04Z E123 is still in flight on the owned pod after about
  `01:39:24` elapsed. PID `21142` remains active, no `results.json`,
  `results.csv`, eval-detail file, or run checkpoint exists yet, and history
  still ends at the inherited E120 step-7500 row. The process remains
  CPU-active, so continue treating this as a slow active run. Do not pull,
  score, stop the pod, or launch E124 until E123 either returns coherent
  artifacts or fails with an exited PID/log traceback.
- 2026-05-14T14:16Z E123 returned coherently on the owned Runpod pod at step
  8000. Remote coherence passed with no active benchmark process, all required
  result/metadata/history/eval-detail/checkpoint files present, one
  `results.csv` row, 1000 eval-detail rows, history ending at step 8000,
  `effective_batch_size=8`, `parameters=3,201,970 <= 3,261,974`,
  `stopped_early=False`, finite `val_lddt_ca`, `val_foldscore`, and
  `val_ca_drmsd`, plus the intended pair-only pre-triangle runtime metadata.
  Pulled artifacts and the checkpoint locally, and the local verifier passed.
  Result: `val_lddt_ca=0.42700926753878593`, FoldScore
  `0.3992369193732738`, `val_ca_drmsd=11.192668638288975`, C-alpha Rg
  `11.470016014099121 / 16.30911695623398`, selected face/tetra boundary
  lDDT `0.7446703624725342` / `0.7280488775372506`. Treat this as a tiny new
  primary-lDDT leader but reject it as a 30k candidate because it remains
  below the `0.45` short gate and softens selected-boundary diagnostics versus
  E120. E124 is now the next short topology-communication probe to consider.
- 2026-05-14T14:21Z launched E124 on the owned Runpod pod `o1dy17ouv8w5mz`
  as `e124_face_edge_frame_gate_from_e120_s8000_c256_m64`, PID `33724`.
  Remote checkout fast-forwarded cleanly to `7cefa48`, no active benchmark
  process was present, the E120 checkpoint existed, py_compile passed for the
  relevant simplex/config/trainer/runner files, and the launch-style
  parameter audit counted `3,239,522 <= 3,261,974`. Startup health confirmed
  a clean weights-only resume at step 7500/examples 60000 with `1292`
  matching tensors loaded and `24` new gate tensors initialized. Metadata
  confirms the intended face boundary-edge-frame gate: scale `0.05`,
  boundary readout directionality/runtime `0.25`, edge-frame message runtime
  `0.0125`, global context `0.1`, vertex-star context `1.0`, edge-star
  context `0.5`, and sparse caps `24 / 48`.
- 2026-05-14T14:27Z E124 health check: PID `33724` is still active on the
  owned pod with the intended command. The artifact directory exists and
  contains `run_metadata.json` plus `history_full_msa_to_face.json`, but
  `results.json` has not been written yet. The history is still at inherited
  pre-step-8000 validation rows, so there is no returned E124 metric to record
  in `EXPERIMENT_RESULTS.md`.
- 2026-05-14T14:30Z E124 health check: PID `33724` remains active on the
  owned pod. The final artifacts are still absent: no `results.json`, no
  `results.csv`, no eval details, and no checkpoint. The history still ends at
  step 7500 with inherited `val_lddt_ca=0.4248279729783535`.
- 2026-05-14T14:31Z prepared E125 locally while E124 runs:
  `simplex_boundary_edge_frame_gate_runtime_scale` adds a runtime ramp for the
  same oriented face boundary-edge-frame gate used by E124. This keeps the
  change inside the selected-complex/topological view: selected face cochains
  still communicate through directed boundary 1-simplices with edge-frame
  scalarized geometry; the new knob only lets that gate turn on as a
  curriculum. Py_compile passed for the simplex/evoformer/model/trainer/runner
  slice, and focused tests passed (`6 passed`) for adapter runtime gating,
  trainer CLI parsing, AF2-medium budget preservation, benchmark CLI parsing,
  model-input plumbing, and validation-time runtime overrides.
- 2026-05-14T14:42Z E124 slow in-flight check: PID `33724` remains active on
  the owned pod after about 21 minutes elapsed. H100 memory remains allocated
  (`13673 MiB`), but instantaneous GPU utilization sampled at `0%`; the
  process is still CPU-active. No final artifacts exist yet (`results.json`,
  `results.csv`, eval details, and checkpoint all absent), and history still
  ends at inherited step 7500. Keep monitoring; do not pull, stop, or launch a
  follow-up until E124 either returns coherent artifacts or clearly fails.
- 2026-05-14T14:54Z E124 remains in flight on the owned pod after about
  33 minutes elapsed. GPU utilization sampled at `21%` with `13673 MiB`
  allocated, so the run still appears alive rather than frozen. No final
  artifacts exist yet and history still ends at step 7500; continue monitoring
  without pulling, stopping, or launching a follow-up.
- 2026-05-14T15:11Z E124 is still in flight on the owned pod. PID `33724`
  remains alive, H100 memory is still allocated (`13673 MiB`), and the
  latest instantaneous GPU utilization sample was `0%`. Required final
  artifacts remain absent (`results.json`, `results.csv`, eval details, and
  checkpoint), and history still ends at inherited E120 step 7500 with
  `val_lddt_ca=0.4248279729783535`. Treat this as a slow active run, not a
  returned result; do not update `EXPERIMENT_RESULTS.md`, stop the pod, or
  launch E125 until E124 returns coherent artifacts or exits with a clear
  failure.
- 2026-05-14T15:46Z E124 remains an active slow run after a 25-minute
  monitor window. PID `33724` stayed alive across six checks, H100 memory
  remained allocated at `13673 MiB`, and sampled GPU utilization ranged from
  `0%` to `68%`. The final result bundle is still absent (`results.json`,
  `results.csv`, eval details, and checkpoint all missing), so there is no
  returned metric to record. Continue waiting under the owned-pod-only
  heartbeat; do not stop the pod or launch E125 while E124 is still alive.
- 2026-05-14T15:58Z E124 returned coherently on the owned Runpod pod at step
  8000. The process exited, all required result/metadata/history/eval-detail/
  checkpoint files were present, `results.csv` had one row, eval details had
  1000 rows, history ended at step 8000, `effective_batch_size=8`,
  `parameters=3,239,522 <= 3,261,974`, `stopped_early=False`, and the
  intended E124 metadata was present. Artifacts, checkpoint, and log were
  pulled locally; `scripts/verify_nanofold_benchmark_artifacts.py` passed with
  the expected step, batch, result rows, eval rows, history endpoint,
  parameter cap, stopped-early state, and metadata. Result:
  `val_lddt_ca=0.42803398206830023`, FoldScore `0.39794732853770254`,
  `val_ca_drmsd=11.252948240101338`, C-alpha Rg
  `11.307547686100007 / 16.30911695623398`, selected face/tetra boundary
  lDDT `0.758343939781189` / `0.7405965490937233`, and selected face/tetra
  contraction `0.5614195781946182` / `0.561115251004696`. Decision: tiny new
  primary-lDDT leader but reject as a 30k candidate because it remains below
  the `0.45` short-gate threshold and worsens FoldScore/dRMSD versus E123/E120.
  E124's stronger selected-boundary diagnostics justify E125 only as a short
  smooth-on topology-curriculum probe, not as a long spend.
- 2026-05-14T16:03Z launched E125 on the owned Runpod pod `o1dy17ouv8w5mz`
  as `e125_ramped_boundary_edge_frame_gate_from_e120_s8000_c256_m64`, PID
  `46151`. Remote checkout fast-forwarded to `f37fcdd`, no active benchmark
  process was present, the E120 checkpoint existed, and py_compile passed for
  the simplex/evoformer/model/trainer/runner slice. Startup confirmed a clean
  weights-only resume from E120 at step 7500/examples 60000 with `1292`
  matching tensors loaded, `24` new gate tensors initialized, and a fresh
  optimizer. Metadata confirms `steps=8000`, `effective_batch_size=8`,
  `max_parameters=3261974`, `simplex_boundary_edge_frame_gate_scale=0.05`,
  `simplex_boundary_edge_frame_gate_runtime_scale=0.0`,
  `simplex_boundary_edge_frame_gate_runtime_scale_final=0.05`,
  ramp start step `7500`, ramp steps `500`, global context `0.1`,
  vertex-star runtime `1.0`, edge-star runtime `0.5`, and sparse caps
  `24 / 48`. E125 should be treated as a short topology-curriculum probe;
  do not launch a follow-up or stop the pod until E125 returns coherent
  artifacts or clearly fails.
- 2026-05-14T16:34Z E125 remains active after a 25-minute monitor window.
  PID `46151` stayed alive across six checks, H100 memory remained allocated
  at `13673 MiB` after startup, and sampled GPU utilization ranged from `0%`
  to `59%`. The final result bundle is still absent (`results.json`,
  `results.csv`, eval details, and checkpoint all missing), while history
  remains at the inherited E120 step-7500 state. Continue monitoring under
  the E125 heartbeat; do not update `EXPERIMENT_RESULTS.md`, stop the pod, or
  launch another follow-up while E125 is still alive.
- 2026-05-14T17:02Z E125 remains active after another 25-minute monitor
  window. PID `46151` stayed alive across six checks, H100 memory remained
  allocated at `13673 MiB`, and sampled GPU utilization ranged from `0%` to
  `46%`. The final result bundle remains absent (`results.json`,
  `results.csv`, eval details, and checkpoint all missing), while history is
  still the inherited E120 step-7500 row. Treat this as a slow active run;
  continue monitoring and do not stop the pod or launch any follow-up while
  E125 is still alive.
- 2026-05-14T17:30Z E125 remains active after a third monitor window. PID
  `46151` stayed alive across six checks, H100 memory remained allocated at
  `13673 MiB`, and sampled GPU utilization ranged from `0%` to `55%`. No
  final artifacts exist yet (`results.json`, `results.csv`, eval details, and
  checkpoint all missing), and history still ends at the inherited E120
  step-7500 row. Continue treating E125 as slow active training; leave the pod
  running and do not update final results until coherent artifacts return.
- 2026-05-14T17:42Z E125 returned coherently on the owned Runpod pod at step
  8000. The process exited, all required result/metadata/history/eval-detail/
  checkpoint files were present, `results.csv` had one row, eval details had
  1000 rows, history ended at step 8000, `effective_batch_size=8`,
  `parameters=3,239,522 <= 3,261,974`, `stopped_early=False`, and the
  intended ramped E125 metadata was present. Artifacts, checkpoint, and log
  were pulled locally; `scripts/verify_nanofold_benchmark_artifacts.py` passed
  with the expected step, batch, result rows, eval rows, history endpoint,
  parameter cap, stopped-early state, and metadata. Result:
  `val_lddt_ca=0.42745678713917734`, FoldScore `0.39857429602742195`,
  `val_ca_drmsd=11.316084318935872`, C-alpha Rg
  `11.29976204776764 / 16.30911695623398`, selected face/tetra boundary lDDT
  `0.7486708375811577` / `0.7314348567426204`, and selected face/tetra
  contraction `0.5641601893305779` / `0.565635446369648`. Decision: reject as
  a 30k candidate. The smoother ramp improved FoldScore slightly versus E124
  but reduced primary C-alpha lDDT, dRMSD, selected-boundary lDDT, and
  contraction. Do not continue the plain boundary-edge-frame schedule family
  without a new topology mechanism.
- 2026-05-14T17:47Z after E125 remote coherence, local artifact pull, and
  local verifier passed, the E125 heartbeat was paused and the owned Runpod
  pod `o1dy17ouv8w5mz` was stopped. A final idle check found no active
  benchmark process and confirmed the required E125 artifact files were still
  present on the pod before stopping. No follow-up run is active.
- 2026-05-14T18:01Z implemented E126 as a default-off sparse simplex
  triangle-attention bias. This is an architecture change, not a new output
  loss: selected face cochains and tetra boundary-face cochains now emit
  sparse per-head logits for AF2 triangle attention on represented triples,
  so persistent higher-order cells can steer triangle consistency directly.
  The projections are zero-initialized for checkpoint compatibility and add
  only `1,216` parameters to the E120 selected-complex profile
  (`3,203,186 <= 3,261,974`). Local validation passed:
  `python -m py_compile minalphafold/embedders.py minalphafold/simplex.py minalphafold/evoformer.py minalphafold/model.py minalphafold/model_config.py minalphafold/trainer.py scripts/run_nanofold_public_benchmarks.py`
  and
  `python -m pytest tests/test_simplex.py::test_simplex_adapter_emits_sparse_triangle_attention_bias tests/test_simplex.py::test_triangle_attention_uses_sparse_simplex_bias tests/test_trainer.py::test_trainer_cli_accepts_simplex_star_context_overrides tests/test_trainer.py::test_simplicial_triangle_attention_bias_stays_inside_medium_budget tests/test_trainer.py::test_triangle_attention_bias_runs_evoformer_block_eagerly tests/test_nanofold_public_benchmarks.py::test_model_config_override_flags_are_accepted_by_cli_parser`
  with `6 passed`; the broader
  `python -m pytest tests/test_simplex.py tests/test_trainer.py tests/test_nanofold_public_benchmarks.py`
  slice passed with `211 passed`; Ruff undefined-name checks passed; and
  `git diff --check` passed. Queue E126 as
  `e126_triangle_attention_bias_from_e120_s8000_c256_m64` from the E120
  checkpoint, not as a blind continuation; reject it unless it clears the
  low-0.4 band and preferably the `0.45` short-gate threshold.
- 2026-05-14T18:57Z launched E126 on the owned Runpod pod only:
  `o1dy17ouv8w5mz` (`codex-simplexfold-e74-runpod-20260512`), SSH
  `root@103.207.149.82:10677` with
  `/Users/christopherhayduk/.runpod/ssh/RunPod-Key-Go`. After the pod restart,
  `/workspace` was empty, so the remote workspace was restaged from scratch:
  SimplexFold was cloned at commit `206fdc5` on
  `codex/simplexfold-topology-e07-boundary-coordinate`; only public NanoFold
  assets were copied (`11000` processed features, `11000` processed labels,
  manifests with `10000 / 1000 / 11000` train/val/all chains, and
  `nanofold/`); the E120 checkpoint was restored at
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e120_mixed_star_context_from_e118_s7500_c256_m64/checkpoints/full_msa_to_face_latest.pt`.
  Prelaunch audit passed: modified Python files compiled remotely,
  `nanofold.metrics.foldscore_components` imported, the model parameter audit
  returned `3,203,186 <= 3,261,974`, hidden/sidecar scan was empty, GPU was
  `NVIDIA H100 80GB HBM3`, and no benchmark process was already active.
- E126 command:
  `python scripts/run_nanofold_public_benchmarks.py --nanofold-root /workspace/nanoFold-Competition --model-config simplexfold_medium_param_matched --variants full_msa_to_face --run-name e126_triangle_attention_bias_from_e120_s8000_c256_m64 --steps 8000 --crop-size 256 --msa-depth 64 --extra-msa-depth 0 --batch-size 1 --grad-accum-steps 8 --learning-rate 0.001 --n-cycles 4 --eval-every 500 --checkpoint-every 500 --max-parameters 3261974 --resume-from-checkpoint /workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e120_mixed_star_context_from_e118_s7500_c256_m64/checkpoints/full_msa_to_face_latest.pt --resume-model-weights-only --simplex-face-coordinate-weight 1.0 --simplex-face-coordinate-distance-weight 0.5 --simplex-face-boundary-lddt-weight 0.05 --simplex-tetra-coordinate-weight 1.0 --simplex-tetra-coordinate-distance-weight 0.5 --simplex-tetra-boundary-lddt-weight 0.05 --simplex-geometry-distance-weight 0.025 --simplex-face-top-k 24 --simplex-tetra-top-k 48 --simplex-cell-score-degree-penalty 0.75 --simplex-cell-score-outer-edge-weight 0.25 --simplex-edge-frame-message-scale 0.025 --simplex-edge-frame-message-runtime-scale 0.0125 --simplex-boundary-readout-directionality 0.25 --simplex-boundary-readout-directionality-runtime-scale 0.25 --simplex-boundary-incidence-normalization 1.0 --simplex-global-context-scale 0.1 --simplex-vertex-star-context-scale 1.0 --simplex-edge-star-context-scale 1.0 --simplex-vertex-star-context-runtime-scale 1.0 --simplex-edge-star-context-runtime-scale 0.5 --simplex-triangle-attention-bias-scale 0.05`.
  It started as PID `1120`, writing
  `/workspace/SimplexFold/logs/e126_triangle_attention_bias_from_e120.log`
  and artifacts under
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e126_triangle_attention_bias_from_e120_s8000_c256_m64/`.
  Initial log health is clean: `train=10000 val=1000 crop=256 msa=64`,
  resumed E120 at `step=7500 examples=60000`, loaded `1292` matching tensors,
  initialized `16` new/missing tensors, and started a fresh optimizer. Current
  `run_metadata.json` confirms `effective_batch_size=8`,
  `max_parameters=3261974`, `simplex_triangle_attention_bias_scale=0.05`,
  the intended star-context/readout/runtime scales, and
  `resume_model_weights_only=True`; the top-level `parameters` field is not
  written until result materialization, so final verification should use the
  returned result row plus the explicit parameter audit. No final result files
  exist yet, so `EXPERIMENT_RESULTS.md` intentionally remains unchanged.
- Retargeted heartbeat `check-simplexfold-e57-runpod` to E126 and reactivated
  it on a 30-minute interval. The heartbeat is scoped to pod
  `o1dy17ouv8w5mz` only, must not inspect or manage any other Runpod pod, must
  pull and verify artifacts locally before stopping the owned pod, and must not
  launch a follow-up automatically.
- 2026-05-14T19:08Z prepared E127 locally while E126 is still active and
  resultless. The owned pod still shows PID `1120` running for E126, the log
  remains at the clean startup/resume messages, and the only E126 artifacts are
  `history_full_msa_to_face.json` plus `run_metadata.json`; no final result
  files exist, so `EXPERIMENT_RESULTS.md` remains unchanged. E127 adds
  `simplex_triangle_attention_value_scale`, a default-off architecture hook
  that lets selected face cochains and tetra boundary-face cochains scatter
  sparse value residuals into the starting-/ending-node AF2 triangle-attention
  pair updates for represented triples. This is the value-side companion to
  E126's logit-bias path, not a new lDDT or coordinate loss. The E120
  selected-complex profile with both `simplex_triangle_attention_bias_scale=0.05`
  and `simplex_triangle_attention_value_scale=0.025` audits at
  `3,215,346 <= 3,261,974` parameters. Local validation passed:
  `python -m py_compile minalphafold/embedders.py minalphafold/simplex.py minalphafold/evoformer.py minalphafold/model.py minalphafold/model_config.py minalphafold/trainer.py scripts/run_nanofold_public_benchmarks.py`
  and
  `python -m pytest tests/test_simplex.py::test_simplex_adapter_emits_sparse_triangle_attention_bias tests/test_simplex.py::test_simplex_adapter_emits_sparse_triangle_attention_value tests/test_simplex.py::test_triangle_attention_uses_sparse_simplex_bias tests/test_simplex.py::test_triangle_attention_uses_sparse_simplex_value tests/test_trainer.py::test_trainer_cli_accepts_simplex_star_context_overrides tests/test_trainer.py::test_simplicial_triangle_attention_bias_stays_inside_medium_budget tests/test_trainer.py::test_simplicial_triangle_attention_value_stays_inside_medium_budget tests/test_trainer.py::test_triangle_attention_bias_runs_evoformer_block_eagerly tests/test_trainer.py::test_triangle_attention_value_runs_evoformer_block_eagerly tests/test_nanofold_public_benchmarks.py::test_model_config_override_flags_are_accepted_by_cli_parser`
  with `10 passed`. Keep E127 parked until E126 returns and is
  pulled/verified; launch only if the E126 result suggests triangle-attention
  cochain routing is directionally useful but too weak.
- 2026-05-14T19:12Z E126 remains active on the owned Runpod pod. PID `1120`
  stayed alive through a four-sample monitor window from `19:11:41Z` to
  `19:12:11Z`; process CPU stayed around `755%`, H100 memory stayed at
  `42899 MiB`, and sampled GPU utilization ranged from `35%` to `57%`.
  The log still only contains clean startup/resume lines, `history_full_msa_to_face.json`
  has not advanced past its inherited startup timestamp, and no final result
  files exist yet. Treat this as slow active training; do not pull, update
  `EXPERIMENT_RESULTS.md`, stop the pod, or launch E127 while E126 is still
  alive.
- 2026-05-14T19:19Z E126 is still active on the same owned Runpod pod. PID
  `1120` has been running for about 24 minutes with process CPU around `754%`;
  the H100 sample reported `42899 MiB` allocated and `43%` GPU utilization.
  Required final artifacts remain absent (`results.json`, `results.csv`,
  eval details, and checkpoint), while the history file still ends at the
  inherited E120 step-7500 validation row (`val_lddt_ca=0.4248279729783535`).
  Continue monitoring; leave `EXPERIMENT_RESULTS.md` unchanged until E126
  returns a coherent final or early-stop result bundle.
- 2026-05-14T19:23Z-19:26Z E126 stayed alive through a six-sample monitor
  window on the owned Runpod pod. PID `1120` remained active, process CPU
  stayed around `751%`-`755%`, H100 memory stayed allocated at `42899 MiB`,
  and sampled GPU utilization ranged from `0%` to `60%`. No final result
  bundle exists yet (`results.json`, `results.csv`, eval details, and
  checkpoint all absent), and history still ends at inherited E120 step 7500.
  Treat this as slow active training, not a failed or returned run.
- 2026-05-14T19:28Z-19:38Z E126 remained active through a longer ten-sample
  monitor window. PID `1120` stayed alive with process CPU around
  `758%`-`763%`, H100 memory stayed allocated at `42899 MiB`, and sampled GPU
  utilization ranged from `0%` to `65%`. The required final artifacts are
  still absent and `history_full_msa_to_face.json` still ends at the inherited
  E120 step-7500 row. Keep waiting for the coherent step-8000 bundle before
  pulling, updating `EXPERIMENT_RESULTS.md`, stopping the pod, or launching
  E127.
- 2026-05-14T19:40Z-19:48Z E126 continued running on the owned pod. PID
  `1120` remained active with process CPU around `751%`-`758%`, H100 memory
  stayed allocated at `42899 MiB`, and sampled GPU utilization ranged from
  `0%` to `54%`. No step-8000 artifacts have been written yet: `results.json`,
  `results.csv`, eval details, and checkpoint are all still absent, and
  history still ends at inherited E120 step 7500. Leave the job running.
- 2026-05-14T19:50Z-20:02Z E126 remained active across a twelve-sample
  monitor window. PID `1120` stayed alive, process CPU stayed around
  `744%`-`748%`, H100 memory remained allocated at `42899 MiB`, and sampled
  GPU utilization ranged from `0%` to `58%`. The final result bundle is still
  absent and history remains unchanged at the inherited E120 step-7500 row.
  Continue waiting; do not pull, stop, or launch a follow-up until E126 writes
  coherent step-8000 artifacts or exits with a clear failure.
- 2026-05-14T20:05Z-20:27Z E126 remained active across a twenty-sample
  early-break watch. PID `1120` stayed alive, process CPU rose from about
  `742%` to `800%`, H100 memory stayed allocated at `42899 MiB`, and sampled
  GPU utilization ranged from `0%` to `81%`. The run is now slower than the
  E124/E125 short gates but still shows real CPU/GPU activity. No final
  artifacts exist yet and history still ends at inherited E120 step 7500, so
  leave the job running and keep `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-14T20:30Z non-disruptive E126 health probe: PID `1120` was still
  active after about 96 minutes elapsed, with `178` threads, process CPU around
  `808%`-`809%`, RSS changing from about `2.08 GiB` to `2.23 GiB` over the
  probe, H100 memory still allocated at `42899 MiB`, and sampled GPU
  utilization at `55%` then `0%`. `/proc/1120/io` read counters advanced over
  30 seconds while write counters and the log stayed unchanged. Interpretation:
  the process is still doing training/evaluation work before the step-8000
  validation/writeout, not an exited run waiting to be collected.
- 2026-05-14T20:52Z E126 returned coherently on the owned Runpod pod at step
  8000. The benchmark process exited and the required remote result bundle was
  present: `results.json`, `results.csv`, history, eval details, metadata, and
  `checkpoints/full_msa_to_face_latest.pt`. Remote coherence passed with one
  result row, 1000 eval-detail rows, history ending at step 8000,
  `effective_batch_size=8`, `parameters=3,203,186 <= 3,261,974`,
  `stopped_early=False`, and the intended triangle-attention-bias metadata.
  Artifacts, checkpoint, and log were pulled locally; local
  `scripts/verify_nanofold_benchmark_artifacts.py` passed with the expected
  step, batch, row counts, history endpoint, parameter cap, stopped-early
  state, and metadata.
- E126 result: `val_lddt_ca=0.4254467885494232`, FoldScore
  `0.39922703403234483`, `val_ca_drmsd=11.122689092040062`, C-alpha Rg
  `11.661468877792359 / 16.30911695623398`, selected face/tetra boundary
  lDDT `0.7477853461503983` / `0.7309605762958526`, and selected face/tetra
  contraction `0.5891976501345635` / `0.5890026765465737`.
  Decision: reject as a 30k candidate. Sparse simplex triangle-attention
  logit bias slightly improved FoldScore/dRMSD versus E120/E124/E125, but it
  reduced primary C-alpha lDDT below E124/E125, softened selected-boundary
  geometry versus E124, and stayed below the `0.45` short-gate threshold.
  Keep E127 parked; do not launch it automatically from this weak
  primary-lDDT signal.
- After local verification and documentation were committed/pushed, the owned
  Runpod pod `o1dy17ouv8w5mz` was stopped. The E126 heartbeat automation
  `check-simplexfold-e57-runpod` was paused. No follow-up Runpod experiment is
  active.
- 2026-05-14T21:00Z prepared E128 as the next topology-native short gate.
  Rationale: E124 is still the primary-lDDT leader via oriented selected face
  boundary-edge-frame transport, while E126 improved FoldScore/dRMSD but
  reduced primary lDDT. E128 resumes the E124 checkpoint and adds a much
  weaker sparse simplex triangle-attention bias (`0.0125`) so selected
  face/tetra cochains can weakly influence represented residue triples without
  overwhelming E124's boundary-realization signal. Local checkpoint
  availability for E124 was confirmed, and the combined profile audits at
  `3,240,738 <= 3,261,974` parameters. Reject unless it beats E124 primary
  C-alpha lDDT and moves toward the `0.45` short-gate threshold.
- 2026-05-14T21:29Z launched E128 on the owned Runpod pod
  `o1dy17ouv8w5mz` only, after restart on SSH `root@103.207.149.82:10704`.
  Because `/workspace` was empty after the restart, the SimplexFold repo was
  recloned at commit `199d0ad`, public NanoFold manifests/code were restored,
  and public `processed_features`/`processed_labels` were transferred via
  `runpodctl` after raw SSH file streaming proved too slow. Prelaunch audit
  passed: `processed_features=11000`, `processed_labels=11000`,
  `train=10000`, `val=1000`, `all=11000`, hidden/private/salt path count
  `0`, H100 `NVIDIA H100 80GB HBM3`, FoldScore import ok, no active benchmark
  process, and E124 checkpoint present at
  `36577263` bytes. Remote model audit for the E128 architecture returned
  `3,240,738 <= 3,261,974` parameters.
- E128 process details: run
  `e128_damped_triangle_bias_from_e124_s8500_c256_m64`, PID `1523`, log
  `logs/e128_damped_triangle_bias_from_e124.log`, artifact directory
  `artifacts/nanofold_public_benchmarks/e128_damped_triangle_bias_from_e124_s8500_c256_m64`.
  Startup log confirms resume from the E124 checkpoint at step 8000/examples
  64000, `1316` matching tensors loaded, `16` new/missing tensors initialized,
  and a fresh optimizer. Initial history currently ends at inherited E124 step
  8000 with `val_lddt_ca=0.42803398206830023`; wait for the coherent step-8500
  bundle before pulling or updating `EXPERIMENT_RESULTS.md`.
- 2026-05-14T21:33Z E128 was still active on the owned pod. PID `1523`
  had about `04:51` elapsed time, process CPU stayed around `755%`-`768%`,
  RSS was about `2.0 GiB`, and the process had `178` threads. H100 memory was
  allocated at `38489 MiB`, with sampled GPU utilization ranging from `42%` to
  `95%`. The artifact directory still contained only `run_metadata.json` and
  `history_full_msa_to_face.json`; history still ended at inherited step 8000,
  and the log remained at startup. Interpretation: active short-gate training
  before the step-8500 validation/writeout, not a completed result.
- 2026-05-14T21:35Z E128 remained active on the owned pod. PID `1523` had
  about `06:47` elapsed time, process CPU stayed around `770%`-`774%`, RSS was
  about `2.0 GiB`, and the process still had `178` threads. H100 memory was
  still allocated at `38489 MiB`, with sampled GPU utilization ranging from
  `48%` to `100%`. The artifact directory still had only the two startup files,
  the log was unchanged after startup, and history still ended at inherited
  step 8000. Leave the job running; do not pull, stop, or launch another run
  until E128 writes a coherent step-8500 bundle or exits.
- 2026-05-14T21:36Z-21:45Z bounded E128 watch on the owned pod: PID `1523`
  stayed alive from about `08:17` through `17:18` elapsed, process CPU stayed
  around `758%`-`773%`, RSS stayed about `2.0 GiB`, and the process kept `178`
  threads. H100 memory stayed allocated at `43353 MiB`, with sampled GPU
  utilization ranging from `19%` to `95%`. The artifact directory still had
  only `run_metadata.json` and `history_full_msa_to_face.json`, history still
  ended at inherited step 8000, and no `results.csv` or checkpoint was present.
  Interpretation: active training/evaluation before the step-8500 writeout;
  leave the run alone and let the heartbeat catch the coherent result bundle.
- 2026-05-14T21:47Z-21:58Z longer E128 watch on the owned pod: PID `1523`
  remained alive from about `19:41` through `30:42` elapsed, process CPU stayed
  around `755%`-`762%`, RSS stayed about `2.0 GiB`, and the process kept `178`
  threads. H100 memory stayed allocated at `43353 MiB`; sampled GPU
  utilization ranged from `0%` to `63%` while CPU stayed high. The artifact
  directory still had only the two startup files, history still ended at
  inherited step 8000, and neither `results.csv` nor
  `checkpoints/full_msa_to_face_latest.pt` was present. Continue to wait; do
  not pull, stop, or launch a follow-up until the coherent step-8500 bundle
  appears or the process exits with a clear failure.
- 2026-05-14T22:14Z E128 was still active on the owned pod. PID `1523` had
  about `46:03` elapsed time, process CPU stayed around `754%`-`755%`, RSS was
  about `2.0 GiB`, and the process kept `178` threads. H100 memory remained at
  `43353 MiB`, sampled GPU utilization moved from `51%` to `64%`, and
  `/proc/1523/io` read counters advanced over a 30-second probe. The artifact
  directory still had only the two startup files, history still ended at
  inherited step 8000, and the log remained at startup. Treat as active work
  before writeout, not a completed or failed run.
- 2026-05-14T22:15Z-22:29Z E128 remained active across a 15-sample watch on
  the owned pod. PID `1523` stayed alive from about `47:39` through `01:01:40`
  elapsed, process CPU stayed around `750%`-`754%`, RSS stayed about `2.0 GiB`,
  and the process kept `178` threads. H100 memory stayed allocated at
  `43353 MiB`, with sampled GPU utilization ranging from `17%` to `63%`.
  Artifacts were still limited to `run_metadata.json` and
  `history_full_msa_to_face.json`; history still ended at inherited step 8000,
  and no `results.csv` or checkpoint was present. Continue waiting for the
  coherent step-8500 bundle; do not stop the pod or launch a follow-up.
- 2026-05-14T22:32Z-22:51Z E128 remained active across a 20-sample watch on
  the owned pod. PID `1523` stayed alive from about `01:04:28` through
  `01:23:30` elapsed, process CPU stayed around `746%`-`775%`, RSS stayed
  about `2.0 GiB`, and the process kept `178` threads. H100 memory stayed
  allocated at `43353 MiB`, with sampled GPU utilization ranging from `0%` to
  `100%`. The artifact directory still only had `run_metadata.json` and
  `history_full_msa_to_face.json`; history still ended at inherited step 8000,
  with no `results.csv` and no checkpoint. Continue waiting for a coherent
  step-8500 result; do not stop the pod or launch a follow-up.
- 2026-05-14T22:55Z E128 remained active on the owned pod. PID `1523` had
  about `01:27:00` elapsed time, process CPU stayed around `782%`-`784%`, RSS
  was about `2.0 GiB`, and the process kept `178` threads. H100 memory stayed
  allocated at `43353 MiB`; sampled GPU utilization moved from `44%` to `0%`
  during the probe. Over 60 seconds, process CPU time advanced from `11:12:44`
  to `11:22:45`, `/proc/1523/io` read counters advanced, and syscall read
  counts advanced, while write counters and log size stayed unchanged. The
  artifact directory still only had the two startup files and history still
  ended at inherited step 8000. Interpretation: active computation before
  writeout, not an idle or completed run.
- 2026-05-14T23:26Z E128 returned coherently on the owned pod
  `o1dy17ouv8w5mz`. Remote verification found all required files
  (`results.json`, `results.csv`, history, eval details, run metadata, and
  checkpoint), one result row, 1000 eval-detail rows, history ending at step
  8500, `completed_steps=8500`, `effective_batch_size=8`,
  `stopped_early=False`, and `parameters=3,240,738 <= 3,261,974`. Local
  artifacts were pulled into
  `artifacts/nanofold_public_benchmarks/e128_damped_triangle_bias_from_e124_s8500_c256_m64/`,
  and `scripts/verify_nanofold_benchmark_artifacts.py` passed with the same
  step, batch, row-count, parameter-cap, stopped-early, checkpoint, and
  metadata checks. Result: `val_lddt_ca=0.4311057258844376`, FoldScore
  `0.4025340421795845`, `val_ca_drmsd=11.004606088757514`, C-alpha Rg
  `11.719762571811676 / 16.30911695623398`, selected face/tetra boundary
  lDDT `0.7559379814267159` / `0.7384514356851578`, and selected face/tetra
  contraction `0.5746216677725315` / `0.5738915434479713`. Decision: keep as
  the new primary-lDDT and FoldScore leader, but reject as a 30k candidate
  because it remains below the `0.45` short-gate threshold and far below the
  `0.7` target.
- After local verification and documentation were committed/pushed, the owned
  Runpod pod `o1dy17ouv8w5mz` was stopped. The E128 heartbeat automation
  `check-simplexfold-e57-runpod` was paused. No follow-up Runpod experiment is
  active.
- 2026-05-14T23:37Z prepared E129 as the next topology-native short gate.
  Rationale: E128 made weak simplex triangle-attention bias useful only after
  pairing it with E124's oriented face boundary-edge-frame gate. E129 therefore
  keeps the E128 route fixed and adds a tiny sparse triangle-attention value
  residual (`0.0125`) so selected face/tetra cochains can provide content, not
  only logit routing, on represented triples. This changes what AF2 triangle
  attention propagates through `Z_ij`, rather than adding a generic output
  loss or increasing the attention-bias scale. Local parameter audit:
  `3,252,898 <= 3,261,974`. Targeted tests passed:
  `python -m pytest tests/test_simplex.py::test_simplex_adapter_emits_sparse_triangle_attention_value tests/test_simplex.py::test_triangle_attention_uses_sparse_simplex_value tests/test_trainer.py::test_simplicial_triangle_attention_value_stays_inside_medium_budget tests/test_trainer.py::test_triangle_attention_value_runs_evoformer_block_eagerly tests/test_nanofold_public_benchmarks.py::test_model_config_override_flags_are_accepted_by_cli_parser`.
  Reject unless it beats E128 primary C-alpha lDDT and moves toward the `0.45`
  short-gate threshold.
- 2026-05-14T23:55Z launched E129 on the owned Runpod pod
  `o1dy17ouv8w5mz` only, after restart on SSH `root@103.207.149.82:16326`.
  The remote SimplexFold checkout is commit `c825faf`, and the remote
  nanoFold checkout is commit `96afc84`. Public data staging audit passed
  after removing macOS AppleDouble sidecars from the transfer archive:
  `processed_features=11000`, `processed_labels=11000`, train/val/all rows
  `10000/1000/11000`, data/artifact hidden/private/salt path count `0`,
  FoldScore import ok, H100 `NVIDIA H100 80GB HBM3`, and launch-style E129
  parameter count `3,252,898 <= 3,261,974`. Run
  `e129_triangle_value_from_e128_s9000_c256_m64` started with PID `1033`, log
  `logs/e129_triangle_value_from_e128.log`, and artifact directory
  `artifacts/nanofold_public_benchmarks/e129_triangle_value_from_e128_s9000_c256_m64`.
  Startup confirmed resume from the E128 checkpoint at step 8500/examples
  68000, `1332` matching tensors loaded, `16` new/missing value-residual
  tensors initialized, and a fresh optimizer. Initial health at 23:55Z: PID
  `1033`, elapsed `00:19`, process CPU `774%`, RSS about `2.0 GiB`, H100
  memory `18665 MiB`, two startup files, and history ending at inherited step
  8500.
- 2026-05-14T23:56Z-2026-05-15T00:00Z E129 remained active across the initial
  five-sample watch on the owned pod. PID `1033` stayed alive from about
  `01:38` through `05:38` elapsed, process CPU stayed around `731%`-`784%`,
  RSS stayed about `2.0 GiB`, and the process kept `178` threads. H100 memory
  ramped from `29295 MiB` to `38823 MiB`, with sampled GPU utilization between
  `42%` and `59%`. The artifact directory still had only the two startup
  files, history still ended at inherited step 8500, and no `results.csv` or
  checkpoint was present. Interpretation: normal early training before the
  step-9000 writeout.
- 2026-05-15T00:03Z-00:12Z E129 remained active across a 10-sample watch on
  the owned pod. PID `1033` stayed alive from about `08:54` through `17:56`
  elapsed, process CPU stayed around `720%`-`733%`, RSS stayed about
  `2.0 GiB`, and the process kept `178` threads. H100 memory stayed allocated
  at `43687 MiB`, with sampled GPU utilization ranging from `0%` to `89%`.
  The artifact directory still had only `run_metadata.json` and
  `history_full_msa_to_face.json`, history still ended at inherited step 8500,
  and no `results.csv` or checkpoint was present. Interpretation: active
  training/evaluation before the step-9000 writeout; leave the run alone and
  let the heartbeat catch the coherent result bundle.
- 2026-05-15T00:15Z-00:29Z E129 remained active on the owned pod. A bounded
  watch plus a final status probe found PID `1033` alive from about `20:28`
  through `34:19` elapsed, process CPU around `720%`-`726%`, RSS about
  `2.0 GiB`, and `178` threads. H100 memory stayed allocated at `43687 MiB`,
  with sampled GPU utilization ranging from `0%` to `96%`; the final probe
  sampled `46%`. The artifact directory still contained only
  `run_metadata.json` and `history_full_msa_to_face.json`, history still ended
  at the inherited E128 step 8500 row, and no `results.csv` or checkpoint was
  present. Interpretation: E129 is still computing before the step-9000
  writeout; do not pull, stop, or launch a follow-up until the coherent result
  bundle appears or the process exits with a clear failure.
- 2026-05-15T00:32Z-00:51Z E129 remained active through a 20-sample bounded
  watch on the owned pod. PID `1033` stayed alive from about `37:30` through
  `56:32` elapsed, process CPU stayed around `716%`-`727%`, RSS stayed around
  `1.98`-`2.08 GiB`, and the process kept `178` threads. H100 memory stayed
  allocated at `43687 MiB`, with sampled GPU utilization ranging from `0%` to
  `99%`. The artifact directory still had only `run_metadata.json` and
  `history_full_msa_to_face.json`; history still ended at the inherited E128
  step 8500 row (`val_lddt_ca=0.4311057258844376`,
  `val_foldscore=0.4025340421795845`), with no `results.csv` and no
  checkpoint. Interpretation: active compute/evaluation before the step-9000
  writeout; leave the job running and let the heartbeat catch the coherent
  bundle.
- 2026-05-15T00:55Z-01:19Z E129 remained active through a 25-sample bounded
  watch on the owned pod. PID `1033` stayed alive from about `59:59` through
  `01:24:01` elapsed, process CPU moved from about `715%` to `748%`, RSS stayed
  about `1.98`-`2.12 GiB`, and the process kept `178` threads. H100 memory
  remained allocated at `43687 MiB`, with sampled GPU utilization ranging from
  `0%` to `67%`. Artifacts were still limited to `run_metadata.json` and
  `history_full_msa_to_face.json`; history still ended at the inherited step
  8500 E128 row (`val_lddt_ca=0.4311057258844376`,
  `val_foldscore=0.4025340421795845`), with no `results.csv` and no checkpoint.
  Interpretation: long but active compute/evaluation before the step-9000
  writeout; do not intervene, pull, stop, or launch a follow-up until a
  coherent result bundle appears or the process exits with a clear failure.
- 2026-05-15T01:22Z-01:36Z E129 remained active through a 15-sample bounded
  watch on the owned pod. PID `1033` stayed alive from about `01:27:13`
  through `01:41:15` elapsed, process CPU rose from about `758%` to `794%`,
  RSS stayed around `2.0 GiB` with brief samples up to about `2.52 GiB`, and
  the process kept `178` threads. H100 memory stayed allocated at `43687 MiB`,
  with sampled GPU utilization ranging from `0%` to `65%`. The artifact
  directory still contained only `run_metadata.json` and
  `history_full_msa_to_face.json`; history still ended at inherited step 8500,
  with no `results.csv` and no checkpoint. Interpretation: the run is still
  active but has not reached a coherent step-9000 writeout; continue waiting
  and keep the heartbeat responsible for the next probe.
- 2026-05-15T01:40Z-01:41Z E129 deeper health check on the owned pod:
  PID `1033` stayed alive from `01:44:59` to `01:45:59` elapsed, process CPU
  rose from `14:03:01` to `14:13:52`, process CPU utilization was about
  `802%`-`805%`, RSS stayed around `2.0 GiB`, and `178` threads remained open.
  `/proc/1033/io` read counters advanced (`rchar` `716310923` -> `719790080`,
  `syscr` `128260` -> `128817`) while write counters stayed unchanged. CUDA
  file descriptors remained open and H100 memory stayed at `43687 MiB`. The
  only artifact mtimes were the launch-time `run_metadata.json` and
  `history_full_msa_to_face.json`. Interpretation: the long quiet period is
  active compute/read-side work before writeout, not an idle completed bundle;
  keep waiting and do not intervene.
- 2026-05-15T01:43Z-01:50Z E129 remained active through an 8-sample bounded
  watch on the owned pod. PID `1033` stayed alive from `01:48:50` through
  `01:55:51` elapsed, process CPU rose from `14:42:20` to `15:55:19`, process
  CPU utilization stayed around `810%`-`824%`, RSS stayed about `2.0 GiB`, and
  `178` threads remained open. H100 memory stayed allocated at `43687 MiB`,
  with sampled GPU utilization ranging from `0%` to `58%`. The artifact
  directory still contained only `run_metadata.json` and
  `history_full_msa_to_face.json`; history still ended at inherited step 8500,
  with no `results.csv` and no checkpoint. Interpretation: still active,
  still no coherent step-9000 bundle; let the heartbeat continue monitoring.
- 2026-05-15T01:53Z E129 returned coherently on the owned pod
  `o1dy17ouv8w5mz`. Remote verification passed with required files
  (`results.json`, `results.csv`, history, eval details, run metadata, and
  checkpoint), one result row, 1000 eval-detail rows, history ending at step
  9000, `completed_steps=9000`, `effective_batch_size=8`,
  `stopped_early=False`, and `parameters=3,252,898 <= 3,261,974`. Local
  artifacts and the log were pulled into
  `artifacts/nanofold_public_benchmarks/e129_triangle_value_from_e128_s9000_c256_m64/`
  and `logs/e129_triangle_value_from_e128.log`; local
  `scripts/verify_nanofold_benchmark_artifacts.py` passed with the same step,
  batch, row-count, parameter-cap, stopped-early, checkpoint, and metadata
  checks. Result: `val_lddt_ca=0.4302854641973972`, FoldScore
  `0.39839958867430686`, `val_ca_drmsd=11.225015842318534`, C-alpha Rg
  `11.272140373706817 / 16.30911695623398`, selected face/tetra boundary
  lDDT `0.7584963102340698` / `0.7400881498157978`, and selected face/tetra
  contraction `0.5353929928243161` / `0.5351497863531113`. Decision: reject
  E129 as a 30k candidate and do not continue this exact value-residual path;
  it improves local selected-boundary diagnostics but regresses primary
  C-alpha lDDT, FoldScore, dRMSD, and expansion versus E128.
- After local verification, documentation, commit, and push, the owned Runpod
  pod `o1dy17ouv8w5mz` was stopped at 2026-05-15T01:58Z. The E129 heartbeat
  automation `check-simplexfold-e57-runpod` was paused. No follow-up Runpod
  experiment is active.
- 2026-05-15T02:10Z Prepared E130 locally without launching Runpod. E130 adds
  `simplex_boundary_hodge_readout_scale`, a parameter-neutral Hodge-style
  double-centering step on the selected boundary-edge 1-cochain before it is
  written into `Z_ij`. This directly targets the E128/E129 diagnosis: local
  selected face/tetra boundary lDDT is strong, but the global C-alpha trace is
  still under-assembled. The hook subtracts source/target residue vertex-star
  means and restores the global selected-boundary mean, so it is a
  simplicial/topological boundary-cochain assembly mechanism rather than a
  generic lDDT/radius/coordinate loss. Candidate launch is documented as E130
  from the E128 checkpoint with E129's triangle-attention value residual
  disabled and `--simplex-boundary-hodge-readout-scale 0.25`. Local validation
  passed: py_compile for modified modules, 5 targeted pytest checks including
  the no-parameter audit (`3,240,738` unchanged from E128), the broader
  `tests/test_simplex.py tests/test_trainer.py tests/test_nanofold_public_benchmarks.py`
  slice (`218 passed`), focused ruff checks via `../../.venv/bin/ruff`, and
  `git diff --check`.
- 2026-05-15T02:14Z-02:36Z Runpod launch logistics for E130: three owned
  pods were created but never became usable, then were stopped/deleted without
  touching any unrelated Runpod instances. Failed owned pods:
  `jhe7fevfkoxr4i` (`codex-simplexfold-e130-runpod-20260515`, H100 custom
  image, SSH refused / uptime stayed 0), `n1vlthzsvpa5l1`
  (`codex-simplexfold-e130b-runpod-20260515`, H100 custom image, pod never
  ready / uptime stayed 0), and `us8gyfkar6u28s`
  (`codex-simplexfold-e130-a100-runpod-20260515`, A100 custom image, pod
  never ready / uptime stayed 0).
- 2026-05-15T02:39Z E130 switched to owned pod `c67fbk189vnvfp`
  (`codex-simplexfold-e130-template-runpod-20260515`) from the official
  `runpod-torch-v280` template, image
  `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, A100-SXM4-80GB. Remote
  sanity check passed: Python `3.12.3`, torch `2.8.0+cu128`, CUDA available,
  `/workspace` mounted, SimplexFold at branch
  `codex/simplexfold-topology-e07-boundary-coordinate` commit `bd5ced6`, and
  NanoFold at commit `96afc84`.
- 2026-05-15T02:48Z-03:00Z Public data staging note: `/workspace` refused
  ownership changes and was slow for many small files, so the public NanoFold
  features and labels were copied onto the pod's local container disk under
  `/root/nanofold_data/` and symlinked into
  `/workspace/nanoFold-Competition/data/processed_features` and
  `/workspace/nanoFold-Competition/data/processed_labels`. Coherence audit
  passed with `11000` feature files, `11000` label files, manifest counts
  `train=10000`, `val=1000`, `all=11000`, E128 checkpoint present, and no
  hidden/private/salt/AppleDouble paths under the staged data or artifacts.
- 2026-05-15T03:02Z E130 launched on owned pod `c67fbk189vnvfp` as
  `e130_hodge_boundary_readout_from_e128_s9000_c256_m64`, PID `4224`, log
  `/workspace/SimplexFold/logs/e130_hodge_boundary_readout.log`. It resumes
  E128 from step `8500` with weights only, effective batch size `8`, crop
  `256`, MSA depth `64`, sparse caps `24 / 48`, E128's selected-complex
  recipe, E129's triangle-attention value residual disabled, and
  `--simplex-boundary-hodge-readout-scale 0.25`.
- E130 startup verification passed: runner saw `train=10000`, `val=1000`,
  resumed from the E128 checkpoint at `step=8500` and `examples=68000`, loaded
  `1332` matching model tensors, initialized `0` new/missing tensors, started
  a fresh optimizer, and allocated about `17681 MiB` on the A100. Remote
  parameter audit remained `3,240,738 <= 3,261,974`. Leave E130 running until
  a coherent step-9000 result bundle appears, then pull, locally verify,
  update `EXPERIMENT_RESULTS.md`, commit/push, and stop only this owned pod.
- 2026-05-15T03:04Z-03:05Z E130 health check on owned pod `c67fbk189vnvfp`:
  PID `4224` stayed active about 5 minutes after launch, process CPU time
  advanced from `00:48:35` to `01:01:20` over one wall-clock minute, CPU
  utilization stayed around `1192%`-`1207%`, `/proc/4224/io` read counters
  advanced (`rchar` `84688609` -> `85452340`, `syscr` `5621` -> `5735`), and
  A100 memory stayed allocated around `17885`-`19131 MiB`. Artifacts still
  contain only `run_metadata.json` and inherited `history_full_msa_to_face.json`
  ending at E128 step `8500`; no `results.json`, eval details, or checkpoint
  have returned yet. Interpretation: active early compute, not a coherent
  result bundle. Keep monitoring and do not update `EXPERIMENT_RESULTS.md`
  until the step-9000 bundle exists and verifies.
- 2026-05-15T03:08Z Prepared E131 locally while E130 runs; no Runpod launch.
  E131 adds `simplex_boundary_edge_star_readout_scale`, a parameter-neutral
  edge-star diffusion step on the selected boundary-edge 1-cochain after the
  Hodge-centered E130 readout and before the pair update. This stays inside
  the simplicial view: selected face/tetra cells write to their boundary
  1-skeleton, and the boundary cochain is locally smoothed through residue
  edge-stars rather than supervised with an output-side C-alpha or distance
  metric. Candidate use only if E130 returns weakly but keeps the
  boundary-cochain stabilization route plausible; do not launch before E130 is
  remotely and locally verified. Local focused validation passed:
  py_compile for modified modules plus targeted tests for the edge-star
  readout helper, adapter effect, CLI override, parameter audit, and NanoFold
  runner parser (`5 passed`).
- 2026-05-15T03:13Z E131 broader local validation passed:
  `python -m pytest tests/test_simplex.py tests/test_trainer.py tests/test_nanofold_public_benchmarks.py`
  reported `220 passed`; focused ruff checks for undefined-name/syntax-risk
  rules passed; `git diff --check` passed. E130 remained active on owned pod
  `c67fbk189vnvfp` at the same time with no `results.json`, `results.csv`,
  eval-details file, or checkpoint yet, so E131 remains a parked fallback and
  `EXPERIMENT_RESULTS.md` remains unchanged.
- 2026-05-15T03:16Z-03:17Z E130 bounded health sample on owned pod
  `c67fbk189vnvfp`: PID `4224` stayed active, process CPU time advanced from
  `03:19:17` to `03:32:00` over one wall-clock minute, CPU utilization stayed
  around `1238%`-`1239%`, RSS moved from about `1.85 GiB` to `1.92 GiB`,
  `/proc/4224/io` read counters advanced (`rchar` `92238968` -> `92942446`,
  `syscr` `6968` -> `7087`), and A100 memory remained allocated at
  `24033 MiB`. No result bundle was present immediately before this sample:
  history still ended at inherited E128 step `8500` and no `results.json`,
  `results.csv`, eval-details file, or checkpoint existed. Interpretation:
  active compute, not a returned or idle bundle; leave the run undisturbed.
- 2026-05-15T03:30Z E130 status refresh on the same owned pod: `runpodctl pod
  get c67fbk189vnvfp` showed the pod still in desired `RUNNING` status, and
  SSH inspection showed PID `4224` still running the intended
  `e130_hodge_boundary_readout_from_e128_s9000_c256_m64` command. The remote
  artifact directory still had only `run_metadata.json` and inherited
  `history_full_msa_to_face.json`, with history length `18` ending at E128
  step `8500` and no `results.json`, `results.csv`, eval-details file, or
  checkpoint yet. The log still only shows launch/startup lines. Decision:
  keep waiting for a coherent step-9000 bundle; do not update
  `EXPERIMENT_RESULTS.md`, stop the pod, or launch E131 until E130 returns or
  exits with a clear failure.
- 2026-05-15 reference PDF check: the two user-provided PDFs remain saved in
  `references/papers/` and hash-match the Downloads originals
  (`hands_on_geometric_deep_learning_nodes_to_complexes.pdf` SHA-256
  `11a87bfc6867cec432a2f9b8068212997e14acd5a2f0653944ed3ca17e3e3c60`;
  `2509.03885v1.pdf` SHA-256
  `676fd6764bb8a1788a6fbcf7a59edf831c23dd7f5661672a8b265ff397f9e4a7`).
  `references/papers/READING_NOTES.md` already records full-text reading
  passes and the current design rule: prefer selected-complex construction,
  incidence, Hodge/co-boundary, outer-edge, and cochain-communication changes
  over generic output-side lDDT or coordinate losses.
- 2026-05-15T03:33Z E130 health sample on owned pod `c67fbk189vnvfp`:
  PID `4224` remained active with elapsed time `33:27`, process CPU time
  `06:55:16`, `%CPU=1240`, RSS about `1.9 GiB`, and `194` threads. IO
  counters had advanced to `rchar=102062187`, `syscr=8964`, and `wchar=189351`;
  the A100 still had `38227 MiB` allocated. The artifact check immediately
  before this health sample still showed no `results.json`, `results.csv`,
  eval-details file, or checkpoint, and history still ended at inherited E128
  step `8500`. Interpretation: the process remains active but has not reached
  a coherent step-9000 writeout, so leave E130 running under the heartbeat and
  do not launch the parked E131 fallback yet.
- 2026-05-15T03:42Z E130 remained active on the owned pod with no returned
  bundle: history still ended at E128 step `8500`, no `results.json`,
  `results.csv`, eval-details file, or checkpoint existed, and PID `4224`
  showed elapsed `42:16`, CPU time `08:45:14`, `%CPU=1242`, and RSS about
  `1.9 GiB`. The pod should remain running under the heartbeat.
- 2026-05-15T03:45Z Prepared E132 locally; no Runpod launch. E132 adds runtime
  schedules for `simplex_boundary_hodge_readout_scale` and
  `simplex_boundary_edge_star_readout_scale`. This is not a new loss or metric
  hack: it lets a resumed checkpoint receive the same selected-boundary
  1-cochain operations from E130/E131 gradually, after face/tetra boundary
  messages scatter to selected edges and before `Z_ij` is updated. Use only if
  E130/E131 make the boundary-cochain route plausible but look unstable. Local
  focused validation passed: py_compile for `simplex.py`, `evoformer.py`,
  `model.py`, `trainer.py`, and `run_nanofold_public_benchmarks.py`, plus six
  targeted pytest checks covering adapter overrides, trainer schedule inputs,
  runner parser support, runner validation-time overrides, and model-input
  plumbing.
- 2026-05-15T03:53Z E132 broader local validation passed:
  `python -m pytest tests/test_simplex.py tests/test_trainer.py tests/test_nanofold_public_benchmarks.py`
  reported `220 passed`; focused ruff checks for undefined-name/syntax-risk
  rules passed; `git diff --check` passed. A final owned-pod check at the
  same time showed E130 still active with no `results.json`, `results.csv`,
  eval-details file, or checkpoint and history still ending at step `8500`,
  so E132 remains parked and `EXPERIMENT_RESULTS.md` remains unchanged.
- 2026-05-15T04:05Z E130 remained active on owned pod `c67fbk189vnvfp`:
  PID `4224` had elapsed `01:05:06`, `%CPU=1245`, RSS about `1.8 GiB`, and
  the remote log still showed only launch/startup lines. The artifact
  directory still had no `results.json`, `results.csv`, eval-details file, or
  step-9000 checkpoint, so no result was pulled and `EXPERIMENT_RESULTS.md`
  remains unchanged.
- 2026-05-15T04:05Z Prepared E133 locally; no Runpod launch. E133 adds runtime
  schedules for sparse simplex triangle-attention bias and value payloads.
  This is a topological communication change rather than a metric hack:
  selected face/tetra cochains modulate AF2 triangle attention through sparse
  ordered triples, and the runtime schedule controls only how strongly that
  cochain path is activated in a resumed checkpoint. Static nonzero
  triangle-attention scales still allocate the projection modules; runtime
  zero suppresses the aux payload. Local validation so far: py_compile for
  `simplex.py`, `evoformer.py`, `model.py`, `trainer.py`, and the NanoFold
  runner passed, `git diff --check` passed, and focused pytest for triangle
  runtime aux gating plus trainer/runner runtime override plumbing reported
  `6 passed`.
- 2026-05-15T04:07Z E133 broader validation passed:
  `python -m pytest tests/test_simplex.py tests/test_trainer.py tests/test_nanofold_public_benchmarks.py`
  reported `221 passed`; focused ruff checks for undefined-name/syntax-risk
  rules passed; `git diff --check` passed. E133 remains a parked fallback
  while E130 is active.
- 2026-05-15T04:10Z E130 continued active on owned pod `c67fbk189vnvfp`.
  `runpodctl pod get c67fbk189vnvfp` reported desired status `RUNNING`; SSH
  inspection showed PID `4224` still running the intended
  `e130_hodge_boundary_readout_from_e128_s9000_c256_m64` command with elapsed
  time `01:10:15`, `%CPU=1246`, and RSS about `1.9 GiB`. Remote artifacts
  still contained only `history_full_msa_to_face.json` and `run_metadata.json`;
  history length remained `18` ending at inherited E128 step `8500` with
  `val_lddt_ca=0.4311057258844376`. No `results.json`, `results.csv`,
  eval-details file, or step-9000 checkpoint existed, so leave the pod running
  and do not update `EXPERIMENT_RESULTS.md` yet.
- 2026-05-15T04:20Z E130 continued active on the same owned pod: PID `4224`
  had elapsed `01:19:53`, `%CPU=1247`, RSS about `1.8 GiB`, and the remote
  artifact directory still had only `history_full_msa_to_face.json` and
  `run_metadata.json`. No `results.json`, `results.csv`, eval-details file, or
  checkpoint existed, so no result was pulled and `EXPERIMENT_RESULTS.md`
  remains unchanged.
- 2026-05-15T04:20Z Began E134 local candidate while E130 runs; no Runpod
  launch. E134 adds `simplex_boundary_edge_star_residual_scale`, a
  parameter-neutral high-pass transform of the selected boundary-edge
  1-cochain. It computes each selected edge's deviation from the average of
  its source and target edge-star means before the pair update. This is a
  topological/cochain-route change, not a new C-alpha or lDDT loss: it tests
  whether the explicit face/tetra boundary signal needs its local contrast
  component preserved rather than only Hodge-centering or edge-star smoothing.
- 2026-05-15T04:24Z E134 local validation passed: py_compile for the modified
  SimplexFold modules and NanoFold runner, five focused pytest checks covering
  the residual helper, adapter effect, trainer parser, zero-parameter audit,
  and benchmark-runner parser, the broader
  `tests/test_simplex.py tests/test_trainer.py tests/test_nanofold_public_benchmarks.py`
  slice (`223 passed`), focused ruff undefined-name/syntax-risk checks, and
  `git diff --check`.
- 2026-05-15T04:25Z E130 remained active on owned pod `c67fbk189vnvfp` with
  PID `4224`, elapsed `01:25:26`, `%CPU=1246`, RSS about `1.9 GiB`, and still
  only `history_full_msa_to_face.json` plus `run_metadata.json` in the remote
  artifact directory. No result bundle was present, so leave E130 running and
  keep `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-15T04:30Z Began E135 local candidate while E130 runs; no Runpod
  launch. E135 adds runtime scheduling for
  `simplex_boundary_edge_star_residual_scale`, so E134's high-pass selected
  boundary 1-cochain can ramp into a resumed checkpoint rather than switch on
  abruptly. This keeps the change inside the simplex boundary/cochain pathway
  and adds no parameters or output-side loss. Validation passed: py_compile for
  `simplex.py`, `evoformer.py`, `model.py`, `trainer.py`, and the NanoFold
  runner; five focused runtime-plumbing pytest checks; the broader
  `tests/test_simplex.py tests/test_trainer.py tests/test_nanofold_public_benchmarks.py`
  slice (`223 passed`); focused ruff undefined-name/syntax-risk checks; and
  `git diff --check`.
- 2026-05-15T04:35Z-04:36Z Bounded E130 health sample on owned pod
  `c67fbk189vnvfp`: PID `4224` remained active, elapsed time advanced from
  `01:35:35` to `01:36:35`, process CPU time advanced from `19:51:12` to
  `20:03:59`, `%CPU` stayed at `1246`, thread count stayed at `194`, RSS
  stayed near `1.9 GiB`, and `/proc/4224/io` read counters advanced
  (`rchar` `139719710` -> `140114623`, `syscr` `15864` -> `15959`). A100
  memory remained allocated at `43091 MiB` though the instantaneous GPU
  utilization sample was `0%`. The artifact directory still had only
  `history_full_msa_to_face.json` and `run_metadata.json`; no result bundle,
  eval-details file, or checkpoint existed. Interpretation: active but not
  returned; leave the run undisturbed and keep `EXPERIMENT_RESULTS.md`
  unchanged.
- 2026-05-15T04:44Z E130 refresh on owned pod `c67fbk189vnvfp`: PID `4224`
  remained active with elapsed `01:44:54`, `%CPU=1246`, RSS about `1.8 GiB`,
  and the expected E130 command line. Remote artifacts still contained only
  `history_full_msa_to_face.json` and `run_metadata.json`; the log still
  showed launch/startup lines only. No `results.json`, `results.csv`,
  eval-details file, or step-9000 checkpoint existed, so no result was pulled
  and `EXPERIMENT_RESULTS.md` remains unchanged.
- 2026-05-15T04:45Z Prepared E136 local candidate while E130 runs; no Runpod
  launch. E136 adds `simplex_boundary_oriented_cochain_scale` plus runtime
  scheduling so the selected face/tetra boundary readout can be blended toward
  an oriented 1-cochain difference before Hodge/readout stabilization and
  before updating `Z_ij`. This is a topology-native boundary-cochain change:
  it subtracts reverse selected-edge content when both directions exist while
  preserving one-way selected directed edges. It adds no output-side lDDT,
  radius, all-pairs distance, or coordinate loss. Local validation passed:
  py_compile for `simplex.py`, `evoformer.py`, `model.py`, `trainer.py`, and
  the NanoFold runner; nine focused pytest checks covering the oriented
  cochain helper, adapter effect, runtime signatures, trainer parser/input
  plumbing, parameter audit, benchmark parser, and benchmark validation-time
  override path; the broader
  `tests/test_simplex.py tests/test_trainer.py tests/test_nanofold_public_benchmarks.py`
  slice (`225 passed`); focused ruff undefined-name/syntax-risk checks; and
  `git diff --check`.
- 2026-05-15T04:50Z E130 remained active on owned pod `c67fbk189vnvfp` after
  E136 validation: PID `4224` had elapsed `01:50:47`, `%CPU=1246`, RSS about
  `1.8 GiB`, and the remote artifact directory still contained only
  `history_full_msa_to_face.json` and `run_metadata.json`. No result bundle,
  eval-details file, or checkpoint existed, so `EXPERIMENT_RESULTS.md` remains
  unchanged and E130 should continue under the heartbeat.
- 2026-05-15T04:53Z-04:54Z Bounded E130 health sample on owned pod
  `c67fbk189vnvfp`: PID `4224` remained active, elapsed time advanced from
  `01:53:16` to `01:54:16`, process CPU time advanced from `23:32:33` to
  `23:45:00`, `%CPU` stayed at `1247`, RSS stayed near `1.9 GiB`, and
  `/proc/4224/io` read counters advanced (`rchar` `149909441` -> `150573956`,
  `syscr` `17812` -> `17917`). A100 memory remained allocated at
  `43091 MiB`, and instantaneous GPU utilization sampled `4%` then `59%`.
  The artifact directory still had only `history_full_msa_to_face.json` and
  `run_metadata.json`; no result bundle, eval-details file, or checkpoint
  existed. Interpretation: active compute, not returned or idle.
- 2026-05-15T04:56Z E130 refresh on owned pod `c67fbk189vnvfp`: PID `4224`
  remained active with elapsed `01:56:35`, process CPU time `1-00:13:45`,
  `%CPU=1246`, RSS about `1.9 GiB`, and still only
  `history_full_msa_to_face.json` plus `run_metadata.json` in the remote
  artifact directory. No result bundle existed, so no pull or results update
  was possible.
- 2026-05-15T04:57Z Began E137 local candidate while E130 runs; no Runpod
  launch. E137 adds `simplex_boundary_face_cyclic_readout_scale` plus runtime
  scheduling. It changes the directed face-to-pair boundary readout from the
  face slot convention `(i,j)`, `(i,k)`, `(j,k)` toward the actual oriented
  2-simplex boundary cycle `(i->j, j->k, k->i)`. This is a topology-native
  boundary-operator change on selected face cochains, not an output-side
  lDDT/radius/distance/coordinate loss. Focused validation so far:
  py_compile for modified runtime modules and runner, `git diff --check`, and
  nine targeted pytest checks covering cyclic boundary helpers, adapter
  override behavior, runtime signatures, trainer parser/input plumbing,
  parameter audit, benchmark parser, and benchmark validation-time override
  path. Broader validation also passed:
  `python -m pytest tests/test_simplex.py tests/test_trainer.py tests/test_nanofold_public_benchmarks.py`
  reported `227 passed`, focused ruff undefined-name/syntax-risk checks
  passed, and `git diff --check` remained clean.
- 2026-05-15T05:06Z E130 refresh on owned pod `c67fbk189vnvfp`: PID `4224`
  remained active with elapsed `02:06:11`, process CPU time `1-02:13:02`,
  `%CPU=1246`, RSS about `1.9 GiB`, and `194` threads. Remote artifacts still
  contained only `history_full_msa_to_face.json` and `run_metadata.json`.
  History length remained `18`, ending at inherited E128 step `8500` with
  `val_lddt_ca=0.4311057258844376`. No `results.json`, `results.csv`,
  eval-details file, or checkpoint existed, so no result was pulled and
  `EXPERIMENT_RESULTS.md` remains unchanged.
- 2026-05-15T05:12Z E130 remained active on the owned Runpod pod
  `c67fbk189vnvfp`: PID `4224` had elapsed `02:11:41` in the latest process
  sample, process CPU time advanced past `1-03:22:33`, `%CPU` stayed near
  `1247`, RSS stayed near `1.9 GiB`, and the A100 allocation remained at
  `43091 MiB`. The log still
  contained only startup/resume lines, and the artifact directory still had
  only `history_full_msa_to_face.json` plus `run_metadata.json`; history
  length stayed `18`, ending at the inherited E128 step `8500`. I also cloned
  the current pushed branch into a separate remote checkout,
  `/workspace/SimplexFold_next`, at commit
  `8debc35bfdf33fd0be7119613103870188f0f034` and verified syntax with
  `python3 -m py_compile` for the modified runtime modules and NanoFold
  runner. This separate checkout leaves the active `/workspace/SimplexFold`
  E130 process untouched while preparing E136/E137-style parked candidates for
  later launch only after E130 returns and is documented. No result was pulled
  and `EXPERIMENT_RESULTS.md` remains unchanged.
- 2026-05-15T05:25Z E130 is still active but now unusually slow relative to
  recent 500-step gates. PID `4224` remained alive on owned pod
  `c67fbk189vnvfp` with elapsed `02:25:34`, process CPU time
  `1-06:15:26`, `%CPU=1247`, RSS about `1.9 GiB`, and `194` threads. Remote
  introspection showed one Python process with many runnable CPU threads, an
  open CUDA context, no child dataloader processes, and no available
  `py-spy`/`gdb`/`strace`; this looks like an expensive active PyTorch path
  rather than a missing-output post-processing failure. The remote artifact
  directory still contained only `history_full_msa_to_face.json` and
  `run_metadata.json`; history length stayed `18`, ending at inherited E128
  step `8500` with `val_lddt_ca=0.4311057258844376`; `results.json` remained
  missing. E128 and E129 each returned their comparable 500-step gates in just
  under two hours, so if E130 crosses roughly three hours without a history or
  result writeout, treat the static Hodge readout as a runtime-failed branch
  and pivot to a ramped/sparser boundary-cochain candidate rather than
  spending another identical long gate. No result was pulled and
  `EXPERIMENT_RESULTS.md` remains unchanged.
- 2026-05-15T05:27Z Prepared E138 as a launch-recipe-only fallback while E130
  continues; no Runpod launch. E138 reuses the already-implemented E137
  face-cyclic boundary readout but removes E130's static Hodge double-centering
  from the recipe. Rationale: if E130's terminal state is a runtime failure,
  the next short gate should still test the topology-native boundary
  orientation hypothesis without repeating the slow vertex-star Hodge path.
  The candidate keeps E128's oriented face gate and damped simplex
  triangle-attention bias, then ramps only
  `simplex_boundary_face_cyclic_readout_scale` from `0.0` to `0.5` over
  steps `8500`-`9000`. It adds no parameters and no output-side loss. Launch
  only after E130 either returns or is explicitly stopped and documented.
- 2026-05-15T05:31Z E138 fallback validation on the local branch tip passed
  while E130 continued running. Checks: `python -m py_compile` for
  `simplex.py`, `evoformer.py`, `model.py`, `trainer.py`, and the NanoFold
  runner; the nine focused E137/E138 parser/runtime tests covering cyclic
  boundary helpers, adapter effect, runtime signatures, trainer parser/input
  plumbing, parameter audit, benchmark parser, and benchmark validation-time
  override path (`9 passed`); and focused ruff undefined-name/syntax-risk
  checks. This does not change `EXPERIMENT_RESULTS.md` because no Runpod
  result has returned.
- 2026-05-15T05:35Z Fast-forwarded the separate remote checkout
  `/workspace/SimplexFold_next` on owned pod `c67fbk189vnvfp` from `8debc35`
  to `3778ea4` and reran remote `python3 -m py_compile` for the modified
  runtime modules plus the NanoFold runner. This checkout is not the active
  E130 working tree and does not affect PID `4224`; it only keeps a clean
  branch tip ready for an E138-style launch if E130 times out or returns
  unusably. The E128 checkpoint and NanoFold checkout were present. At the
  same sample, E130 remained active with elapsed `02:34:57`, process CPU time
  `1-08:13:37`, `%CPU=1247`, and no result writeout yet.
- 2026-05-15T06:00Z E130 reached the documented runtime-failure cutoff on the
  same owned pod: elapsed `03:00:16`, process CPU time `1-13:27:46`,
  `%CPU=1246`, history still length `18` ending at inherited E128 step
  `8500`, and no `results.json`, `results.csv`, eval-details file, or
  checkpoint. I pulled the timeout trace locally under ignored paths
  (`logs/e130_hodge_boundary_readout.log` plus E130 `run_metadata.json` and
  `history_full_msa_to_face.json`), then terminated only E130 PID `4224`.
  Because the pod/data/checkpoint were already staged and owned by this
  thread, I reused pod `c67fbk189vnvfp` for the next Runpod experiment rather
  than touching any other instance.
- 2026-05-15T06:01Z Launched E138 on owned pod `c67fbk189vnvfp` from the
  separate `/workspace/SimplexFold_next` checkout, PID `24980`, log
  `/workspace/SimplexFold_next/logs/e138_no_hodge_face_cyclic_boundary.log`,
  artifacts
  `/workspace/SimplexFold_next/artifacts/nanofold_public_benchmarks/e138_no_hodge_face_cyclic_boundary_from_e128_s9000_c256_m64`.
  The launch fast-forwarded the checkout to `56aa623`, reran remote
  `python3 -m py_compile`, resumed the E128 checkpoint at step `8500` /
  examples `68000`, loaded `1332` matching tensors, initialized `0`
  new/missing tensors, and started a fresh optimizer. Startup metadata records
  effective batch size `8`, max parameters `3261974`, Hodge readout disabled,
  face-cyclic readout scale `0.5`, and damped simplex triangle-attention bias
  `0.0125`. No E138 result has returned yet.
- 2026-05-15T06:06Z E138 startup health sample on owned pod
  `c67fbk189vnvfp`: PID `24980` remained active with elapsed `00:05:05`,
  process CPU time `01:01:58`, `%CPU=1217`, RSS about `1.9 GiB`, and
  `194` threads. Remote artifacts contained `run_metadata.json` and
  `history_full_msa_to_face.json`; history length was still `18`, ending at
  inherited E128 step `8500` with `val_lddt_ca=0.4311057258844376`. No
  `results.json`, `results.csv`, eval-details file, or checkpoint existed
  yet, which is normal for the first few minutes of this 500-step resumed
  gate. Leave E138 running under the heartbeat.
- 2026-05-15T06:18Z Bounded E138 watch on owned pod `c67fbk189vnvfp`:
  PID `24980` remained active through samples at elapsed `00:07:03`,
  `00:12:04`, and `00:17:04`. The final sample had process CPU time
  `03:33:30`, `%CPU=1250`, RSS about `1.9 GiB`, and `194` threads. The
  artifact directory still contained only `run_metadata.json` and
  `history_full_msa_to_face.json`; history length remained `18`, ending at
  inherited E128 step `8500` with `val_lddt_ca=0.4311057258844376`. No
  `results.json`, `results.csv`, eval-details file, or checkpoint exists yet.
  Interpretation: normal early active compute, not a returned result.
- 2026-05-15T06:20Z Prepared E139 as a launch-recipe-only fallback while E138
  runs; no Runpod launch. E139 reuses the already implemented oriented
  boundary-cochain readout from E136, but removes E130's Hodge
  double-centering from the recipe. If E138's face-cyclic 2-simplex boundary
  route returns flat but coherent, E139 will test orientation after face/tetra
  cochains have been pooled onto the selected boundary 1-skeleton by ramping
  `simplex_boundary_oriented_cochain_scale` from `0.0` to `0.25` over steps
  `8500`-`9000`. It adds no parameters and no output-side loss. Launch only
  after E138 returns and is documented.
- 2026-05-15T06:21Z E139 fallback validation passed locally while E138
  continued running. Checks: `python -m py_compile` for `simplex.py`,
  `evoformer.py`, `model.py`, `trainer.py`, and the NanoFold runner; the nine
  focused E136/E139 parser/runtime tests covering oriented boundary-cochain
  helpers, adapter effect, runtime signatures, trainer parser/input plumbing,
  parameter audit, benchmark parser, and benchmark validation-time override
  path (`9 passed`); focused ruff undefined-name/syntax-risk checks; and
  `git diff --check`. No Runpod launch and no `EXPERIMENT_RESULTS.md` update.
- 2026-05-15T06:29Z E138 remained active on owned pod `c67fbk189vnvfp`:
  PID `24980` had elapsed `00:28:02`-`00:28:03`, process CPU time was about
  `05:51`, `%CPU=1251`, RSS about `1.9 GiB`, and `194` threads. Remote
  artifacts still contained only `run_metadata.json` and
  `history_full_msa_to_face.json`; history length remained `18`, ending at
  inherited E128 step `8500` with `val_lddt_ca=0.4311057258844376`. No
  `results.json`, `results.csv`, eval-details file, or checkpoint exists yet,
  so there is nothing to pull or add to `EXPERIMENT_RESULTS.md`.
- 2026-05-15T06:34Z E138 remained active on the same owned pod with PID
  `24980`, elapsed `00:32:53`, process CPU time `06:51:43`, `%CPU=1251`,
  RSS about `1.9 GiB`, and `194` threads. Remote artifacts were still limited
  to `run_metadata.json` and inherited `history_full_msa_to_face.json`;
  history length stayed `18`, ending at E128 step `8500` with
  `val_lddt_ca=0.4311057258844376`, FoldScore `0.4025340421795845`,
  `val_ca_drmsd=11.004606088757514`, and C-alpha Rg
  `11.719762571811676 / 16.30911695623398`. No result bundle, eval details,
  or checkpoint existed yet, so `EXPERIMENT_RESULTS.md` stays unchanged.
- 2026-05-15T06:36Z Prepared E140 as a docs-only parked fallback; no Runpod
  launch. E140 would use the existing selected face/tetra coordinate-expansion
  terms at small weight as a selected-complex realization probe, not as a
  generic radius/all-pairs/lDDT shortcut. Rationale: E128's selected
  face/tetra boundary lDDT is already high (`0.7559` / `0.7385`), but global
  C-alpha Rg is still collapsed (`11.7198 / 16.3091`). If E138/E139 return
  coherent but remain flat, E140 can test whether the selected 2- and 3-cells
  need a small anti-collapse constraint on their own boundary 1-skeleton
  before the pair trunk can assemble a less compact backbone. Launch only
  after E138 returns and E139 is either returned or explicitly skipped.
- 2026-05-15T06:39Z Validated the parked E140 plumbing locally; no Runpod
  launch. A parameter audit of `simplexfold_medium_param_matched` counted
  `3,106,690 <= 3,261,974`, and focused tests covering the expansion-hinge
  no-parameter guarantee, AlphaFold loss coordinate-expansion overrides,
  NanoFold benchmark loss builder, variant parser acceptance, and topology
  preservation all passed (`5 passed in 1.31s`). This confirms E140 can be
  launched later using existing selected-complex loss plumbing if the active
  orientation/readout branches return flat.
- 2026-05-15T06:42Z E138 was still active on owned pod `c67fbk189vnvfp`, but
  the runtime profile now looks CPU-bound: six repeated `nvidia-smi` samples
  from `06:41:57`-`06:42:24Z` showed `0%` GPU utilization and `0%` GPU memory
  utilization while PID `24980` held about `38.2 GiB` of A100 memory and used
  about `1249%` CPU. The process remained alive with elapsed `00:40:42`,
  CPU time `08:28:42`, RSS about `1.9 GiB`, and `194` threads. The artifact
  directory still contained only `run_metadata.json` and the inherited
  `history_full_msa_to_face.json`; no result bundle, eval details, or
  checkpoint existed. Treat this as an active warning, not a returned result;
  if the same no-write state persists to the E130-style cutoff, classify E138
  as a runtime-failed branch before launching E139.
- 2026-05-15T06:45Z E138 still had no writeout, but the next health sample
  showed active GPU work again rather than a persistent stall: PID `24980`
  was alive at elapsed `00:43:14`, CPU time `09:01:27`, `%CPU=1252`, RSS about
  `1.9 GiB`, and `nvidia-smi` reported `58%` GPU utilization with `38.2 GiB`
  allocated. Artifacts were unchanged (`run_metadata.json` plus inherited
  history ending at step `8500`, no result/eval/checkpoint files). Keep E138
  running under the heartbeat; do not classify it failed solely from the
  earlier zero-utilization burst.
- 2026-05-15T06:47Z Updated heartbeat automation
  `check-simplexfold-e57-runpod` to reflect the E138 state nuance and next
  action. The automation remains scoped only to owned pod `c67fbk189vnvfp`.
  It should leave E138 running while active before the roughly 3-hour no-write
  cutoff, verify/pull/document/commit/push if E138 returns, and only if E138
  reaches the E130-style no-write cutoff or fails without a coherent result
  should it document E138 as runtime failed, terminate only the E138 process,
  and launch the already documented E139 no-Hodge oriented boundary-cochain
  fallback on the same healthy pod.
- 2026-05-15T06:50Z E138 remained active and below cutoff on owned pod
  `c67fbk189vnvfp`: PID `24980` had elapsed `00:48:01`, CPU time `10:01:16`,
  `%CPU=1251`, RSS about `1.9 GiB`, and instantaneous GPU utilization again
  sampled at `0%` while `38.2 GiB` remained allocated. Artifacts were still
  only `run_metadata.json` plus inherited history ending at step `8500`; no
  result, eval-detail, or checkpoint files existed. While leaving E138
  running, I expanded the E139 documentation with a full same-pod launch
  skeleton and validated the documented flags through the local
  `parse_args` path: run name accepted, variant `full_msa_to_face`, oriented
  cochain final runtime scale `0.25`, and effective batch size `8`.
- 2026-05-15T07:02Z E138 remained active and below cutoff on the same owned
  pod: PID `24980` had elapsed `01:00:16`, CPU time `12:35:02`, `%CPU=1252`,
  RSS about `1.8 GiB`, and no new result bundle, eval details, or checkpoint.
  Remote artifacts were still `run_metadata.json` plus inherited history
  ending at E128 step `8500` with `val_lddt_ca=0.4311057258844376`; no
  `EXPERIMENT_RESULTS.md` update is appropriate.
- 2026-05-15T07:07Z Prepared E141 locally while E138 continues; no Runpod
  launch. E141 is the signed version of E138's face-cyclic boundary readout:
  it keeps the selected 2-simplex boundary cycle but applies the oriented
  incidence sign, scattering face `[i,j,k]` updates as `(ij, jk, -ik -> ki)`.
  This is a parameter-neutral simplicial cochain readout change, not an
  output-side C-alpha lDDT/radius/distance loss. Local validation passed:
  `python -m py_compile` for `simplex.py`, `evoformer.py`, `model.py`,
  `trainer.py`, and the NanoFold runner; the nine focused signed-boundary
  parser/runtime tests (`9 passed`); and the broader
  `tests/test_simplex.py tests/test_nanofold_public_benchmarks.py
  tests/test_trainer.py` slice (`228 passed`). Focused ruff
  undefined-name/syntax-risk checks and `git diff --check` also passed.
  Parameter audit counted `3,106,690` parameters with or without
  `simplex_boundary_signed_face_cyclic_readout_scale=0.25`, so the operator
  adds no parameters and remains under the `3,261,974` cap. The documented
  E141 launch flags parse with effective batch size `8` and signed
  face-cyclic runtime final scale `0.25`. E141 should stay parked until E138
  returns or is terminally documented; if the current heartbeat has already
  launched E139 by then, do not interrupt that fallback solely to run E141
  first.
- 2026-05-15T07:10Z Staged E141 on the owned Runpod pod without disturbing
  the active E138 checkout. Cloned the pushed branch tip into
  `/workspace/SimplexFold_e141` at commit `d5cb9d9` and ran remote
  `python3 -m py_compile` for `simplex.py`, `evoformer.py`, `model.py`,
  `trainer.py`, and the NanoFold runner successfully. No Runpod experiment
  was launched from this checkout; E138 PID `24980` continues to own the
  active GPU run from `/workspace/SimplexFold_next`.
- 2026-05-15T07:13Z Added a full same-pod launch skeleton for E141 in
  `EXPERIMENTS.md`, using `/workspace/SimplexFold_e141` so the active E138
  tree stays untouched. Parser validation accepted the documented command
  shape with run name
  `e141_signed_face_cyclic_boundary_from_e128_s9000_c256_m64`, variant
  `full_msa_to_face`, effective batch size `8`, max-parameter cap `3261974`,
  signed static scale `0.25`, and signed runtime final scale `0.25`.
- 2026-05-15T07:17Z Rechecked E141 remote readiness while leaving E138
  untouched. `/workspace/SimplexFold_e141` is clean on commit `d5cb9d9`, the
  NanoFold checkout exists at `/workspace/nanoFold-Competition`, and the E128
  resume checkpoint exists at
  `/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e128_damped_triangle_bias_from_e124_s8500_c256_m64/checkpoints/full_msa_to_face_latest.pt`.
  The staged E141 parser path still reports effective batch size `8`,
  max-parameter cap `3261974`, and signed runtime final scale `0.25`. No
  experiment was launched.
- 2026-05-15T07:38Z E138 remained active and below cutoff on owned pod
  `c67fbk189vnvfp`: PID `24980` had elapsed `01:36:37`, process CPU time
  `20:13:14`, `%CPU=1255`, and `38.2 GiB` of A100 memory allocated. The
  artifact directory still contained only `run_metadata.json` plus inherited
  `history_full_msa_to_face.json`; no result bundle, eval-details file, or
  checkpoint existed yet. Leave E138 running under the heartbeat.
- 2026-05-15T07:39Z Prepared E142 locally while E138 continues; no Runpod
  launch. E142 adds a signed upper-coboundary face update through selected
  tetra cofaces: anchored tetra faces `(i,j,k)`, `(i,j,l)`, and `(i,k,l)` use
  oriented boundary signs `[-, +, -]`, so sibling face messages are aligned by
  `sign(current) * sign(sibling)` before scattering a coboundary delta back to
  the face cochain. This is a parameter-neutral simplicial cochain update, not
  an output-side C-alpha lDDT/radius/distance loss. Local validation passed:
  `python -m py_compile` for `simplex.py`, `evoformer.py`, `model.py`,
  `trainer.py`, and the NanoFold runner; the nine focused signed tetra
  coboundary parser/runtime tests (`9 passed`); and the broader
  `tests/test_simplex.py tests/test_nanofold_public_benchmarks.py
  tests/test_trainer.py` slice (`230 passed`). Focused ruff
  undefined-name/syntax-risk checks passed. Parameter audit counted
  `3,106,690` parameters with or without
  `simplex_signed_tetra_coboundary_scale=0.25`, so the operator adds no
  parameters and remains under the `3,261,974` cap. Keep E142 parked until
  the active E138 result or terminal failure is documented.
- 2026-05-15T07:43Z Staged E142 on the owned Runpod pod without disturbing
  the active E138 checkout. Cloned the pushed branch into
  `/workspace/SimplexFold_e142`, reset it to commit `410589d`, confirmed a
  clean checkout, and ran remote `python3 -m py_compile` for `simplex.py`,
  `evoformer.py`, `model.py`, `trainer.py`, and the NanoFold runner. Remote
  parser validation accepted the full E142 launch command with run name
  `e142_signed_tetra_coboundary_from_e128_s9000_c256_m64`, variant
  `full_msa_to_face`, effective batch size `8`, max-parameter cap `3261974`,
  signed static scale `0.25`, and signed runtime final scale `0.25`. The
  NanoFold checkout and E128 resume checkpoint were verified present. No
  Runpod experiment was launched from this checkout.
- 2026-05-15T07:53Z E138 remained active and below cutoff on owned pod
  `c67fbk189vnvfp`: PID `24980` had elapsed `01:51:56`, process CPU time
  `23:26:38`, and the artifact directory still contained only
  `run_metadata.json` plus inherited `history_full_msa_to_face.json`. No
  result bundle, eval-details file, or checkpoint existed yet.
- 2026-05-15T07:55Z Prepared E143 locally while E138 continues; no Runpod
  launch. E143 adds a signed blend on the learned `tetra_to_face` readout
  before selected tetra messages scatter back into the maintained anchored
  faces. It uses the same oriented tetra boundary signs `[-, +, -]` as E142,
  but acts on the learned tetra-to-face cochain readout rather than on the
  parameter-neutral sibling-face coboundary residual. This is a
  parameter-neutral simplicial incidence change, not an output-side
  C-alpha lDDT/radius/distance loss. Local validation passed: `python -m
  py_compile` for `simplex.py`, `evoformer.py`, `model.py`, `trainer.py`, and
  the NanoFold runner; the nine focused signed tetra-to-face parser/runtime
  tests (`9 passed`); and the broader `tests/test_simplex.py
  tests/test_nanofold_public_benchmarks.py tests/test_trainer.py` slice
  (`232 passed`). Focused ruff undefined-name/syntax-risk checks passed.
  Parameter audit counted `3,106,690` parameters with or without
  `simplex_signed_tetra_to_face_scale=0.25`, so the operator adds no
  parameters and remains under the `3,261,974` cap. Local parser validation
  accepted the full E143 launch command with effective batch size `8`,
  max-parameter cap `3261974`, signed static scale `0.25`, and signed runtime
  final scale `0.25`.
- 2026-05-15T08:00Z Staged E143 on the owned Runpod pod without disturbing
  the active E138 checkout. Cloned the pushed branch into
  `/workspace/SimplexFold_e143`, reset it to commit `f931bed`, confirmed a
  clean checkout, and ran remote `python3 -m py_compile` for `simplex.py`,
  `evoformer.py`, `model.py`, `trainer.py`, and the NanoFold runner. Remote
  parser validation accepted the full E143 launch command with run name
  `e143_signed_tetra_to_face_from_e128_s9000_c256_m64`, variant
  `full_msa_to_face`, effective batch size `8`, max-parameter cap `3261974`,
  signed static scale `0.25`, and signed runtime final scale `0.25`. The
  NanoFold checkout and E128 resume checkpoint were verified present. No
  Runpod experiment was launched from this checkout.
- 2026-05-15T08:00Z E138 remained active and below cutoff on owned pod
  `c67fbk189vnvfp`: PID `24980` had elapsed `01:58:42`, process CPU time
  `1-00:51:31`, and `43.1 GiB` of A100 memory allocated. The artifact
  directory still contained only `run_metadata.json` plus inherited
  `history_full_msa_to_face.json`; no result bundle, eval-details file, or
  checkpoint existed yet. Leave E138 running.
- 2026-05-15T08:12Z Staged E139 on the owned Runpod pod without disturbing
  the active E138 checkout. Cloned the pushed branch into
  `/workspace/SimplexFold_e139`, confirmed a clean checkout at commit
  `83deaf0`, and ran remote `python3 -m py_compile` for `simplex.py`,
  `evoformer.py`, `model.py`, `trainer.py`, and the NanoFold runner. Remote
  parser validation accepted the full E139 launch command with run name
  `e139_no_hodge_oriented_boundary_from_e128_s9000_c256_m64`, variant
  `full_msa_to_face`, effective batch size `8`, max-parameter cap `3261974`,
  oriented static scale `0.25`, and oriented runtime final scale `0.25`. The
  NanoFold checkout and E128 resume checkpoint were verified present. No
  Runpod experiment was launched from this checkout. At the same remote check,
  E138 PID `24980` remained active at elapsed `02:10:33`.
- 2026-05-15T08:15Z E138 remained active and below cutoff on the same owned
  pod: PID `24980` had elapsed `02:13:21`, GPU memory remained allocated at
  about `43.1 GiB`, and the artifact directory still contained only
  `run_metadata.json` plus inherited step-8500 history. No result bundle,
  eval-details file, or checkpoint existed, so leave E138 running.
- 2026-05-15T08:18Z E128 validation-detail diagnostic from the local
  `eval_details_full_msa_to_face.csv`: over 1000 validation chains, C-alpha
  lDDT correlates with predicted C-alpha Rg (`r=0.5958`), selected face
  boundary lDDT (`r=0.5969`), and selected tetra boundary lDDT (`r=0.6270`),
  while dRMSD correlation is weak (`r=-0.0693`). The bottom 100 chains average
  `lddt_ca=0.3404`, Rg ratio `0.6919`, face/tetra boundary lDDT
  `0.6859` / `0.6633`; the top 100 average `lddt_ca=0.6381`, Rg ratio
  `0.8252`, and face/tetra boundary lDDT `0.8206` / `0.8115`. Interpretation:
  the target is not purely impossible for this parameter budget on favorable
  chains, but the branch still needs topology-to-global-assembly improvements
  that raise the low tail and expansion, not a generic output-side lDDT hack.
- 2026-05-15T08:44Z E138 was still active but below the documented no-write
  cutoff on owned pod `c67fbk189vnvfp`: PID `24980` had elapsed `02:42:51`,
  artifact files were still only `run_metadata.json` and inherited
  `history_full_msa_to_face.json`, history length remained `18` ending at
  E128 step `8500`, and no result bundle, eval details, or checkpoint existed.
  Left E138 running until the cutoff instead of stopping early.
- 2026-05-15T09:02Z E138 reached the documented runtime-failure cutoff on the
  same owned pod. PID `24980` had elapsed `03:00:56`, process CPU time was
  `1-13:54:06`, the log had not changed since startup, artifact mtimes were
  still `2026-05-15T06:01Z` / `06:02Z`, history remained at inherited E128
  step `8500`, and no `results.json`, `results.csv`,
  `eval_details_full_msa_to_face.csv`, or checkpoint existed. Preserved the
  E138 trace locally under
  `artifacts/nanofold_public_benchmarks/e138_no_hodge_face_cyclic_boundary_from_e128_s9000_c256_m64/`
  plus `logs/e138_no_hodge_face_cyclic_boundary.log`, then terminated only
  E138 PID `24980`.
- 2026-05-15T09:03Z Launched E139 on owned pod `c67fbk189vnvfp` from the
  separate staged checkout `/workspace/SimplexFold_e139` as
  `e139_no_hodge_oriented_boundary_from_e128_s9000_c256_m64`. Remote
  `python3 -m py_compile` passed before launch. The run resumes the E128
  checkpoint from step `8500` with weights only, effective batch size `8`,
  crop `256`, MSA depth `64`, sparse caps `24 / 48`, E128's
  selected-complex recipe, no Hodge readout, no E138 face-cyclic readout, and
  oriented boundary-cochain readout ramped from `0.0` to `0.25` over
  steps `8500`-`9000`.
- 2026-05-15T09:04Z E139 startup verification passed. The runner wrote
  `run_metadata.json` and inherited `history_full_msa_to_face.json`, saw
  `train=10000`, `val=1000`, resumed from the E128 checkpoint at
  `step=8500` / `examples=68000`, loaded `1332` matching model tensors, and
  initialized `0` new/missing tensors. The actual remote Python process is
  PID `42517`; it had GPU memory allocated and was active. The printed launch
  PID `42514` was the wrapper shell, not the training process.
- 2026-05-15T09:09Z E139 remained active on owned pod `c67fbk189vnvfp`:
  Python PID `42517` had elapsed `00:05:52`, process CPU time `01:12:17`,
  and the expected E139 command line. The artifact directory contained
  startup `run_metadata.json` plus inherited `history_full_msa_to_face.json`
  ending at E128 step `8500`; no result bundle, eval-details file, or
  checkpoint existed yet. Leave E139 running under the updated heartbeat.
- 2026-05-15T09:13Z-09:15Z E139 remained active and still below cutoff on the
  same owned pod. Python PID `42517` had elapsed `00:09:58`, process CPU time
  `02:04:00`, and the artifact/log files were unchanged from startup. A
  six-sample utilization check showed `19.2 GiB` GPU memory allocated but GPU
  utilization mostly `0%` (`3%` in one sample), while the Python process used
  roughly `12.4` CPU cores and had `194` threads. This matches the quiet
  CPU-heavy/no-write pattern seen in E130/E138, but E139 is far below the
  documented no-write cutoff, so leave it running. `py-spy` is installed on
  the pod, but attaching to PID `42517` failed with container ptrace
  `Permission denied`; no process state was changed.
- 2026-05-15T09:25Z Added future-run heartbeat instrumentation to the NanoFold
  public benchmark runner. New checkouts will write `status_<variant>.json`
  at startup, checkpoint/log/finish events, roughly every minute after step
  progress, and at the start of each training step with the active gradient
  accumulation microbatch. This is an experiment-ops change to make slow
  topology gates diagnosable when a single step is CPU-heavy; it does not alter
  model parameters, losses, training data, evaluation, or any active Runpod
  process. The current E139 run is unaffected because it is running an older
  staged checkout.
- 2026-05-15T09:31Z Refreshed the parked fallback Runpod checkouts without
  touching active E139. `/workspace/SimplexFold_e141`,
  `/workspace/SimplexFold_e142`, and `/workspace/SimplexFold_e143` all
  fast-forwarded cleanly to heartbeat commit `fd65f74`, and remote
  `python3 -m py_compile` passed for `simplex.py`, `evoformer.py`, `model.py`,
  `trainer.py`, and `scripts/run_nanofold_public_benchmarks.py` in each tree.
  Remote parser validation still accepted the documented E141, E142, and E143
  launch recipes with effective batch size `8`, max-parameter cap `3261974`,
  static scale `0.25`, and runtime final scale `0.25` for the candidate-specific
  signed face-cyclic, signed tetra-coboundary, or signed tetra-to-face flag.
  E139 remained active during this refresh and no new experiment was launched.
- 2026-05-15T09:35Z Staged E140 on the owned Runpod pod without touching
  active E139. Cloned `/workspace/SimplexFold_e140` at commit `b0a7806`,
  remote `python3 -m py_compile` passed for `simplex.py`, `evoformer.py`,
  `model.py`, `trainer.py`, and `scripts/run_nanofold_public_benchmarks.py`,
  and remote parser validation accepted the full E140 launch recipe with
  effective batch size `8`, max-parameter cap `3261974`, selected face/tetra
  coordinate-expansion weights `0.05`, and expansion tolerance `0.05`. The
  full E128-style architecture audit counted `3,240,738` parameters, still
  under the `3,261,974` cap. E140 remains a parked selected-complex
  realization probe; no Runpod experiment was launched.
- 2026-05-15T09:38Z E139 remained active and below the no-write cutoff on
  owned pod `c67fbk189vnvfp`: Python PID `42517` had elapsed `00:34:37`,
  process CPU time was `07:11:48`, the artifact directory still contained only
  startup `run_metadata.json` and inherited `history_full_msa_to_face.json`,
  history length remained `18` ending at E128 step `8500`
  (`val_lddt_ca=0.4311057258844376`, FoldScore `0.4025340421795845`), and no
  `results.json`, `results.csv`, eval details, checkpoint, or status file
  existed. GPU memory was about `38.2 GiB` with `0%` utilization. Leave E139
  running.
- 2026-05-15T09:40Z Staged E144 on the owned Runpod pod without touching
  active E139. E144 uses the existing no-Hodge selected boundary edge-star
  residual readout: a parameter-neutral projection of the selected boundary
  1-cochain away from source/target residue edge-star common modes, not a
  metric-side loss. Cloned `/workspace/SimplexFold_e144`, remote
  `python3 -m py_compile` passed for `simplex.py`, `evoformer.py`, `model.py`,
  `trainer.py`, and `scripts/run_nanofold_public_benchmarks.py`, and remote
  parser validation accepted the full E144 launch recipe with effective batch
  size `8`, max-parameter cap `3261974`, static residual scale `0.25`, and
  runtime final scale `0.25`. The full E128-style architecture audit counted
  `3,240,738` parameters, still under the cap. No Runpod experiment was
  launched.
- 2026-05-15T09:45Z E139 remained active and below the no-write cutoff on
  owned pod `c67fbk189vnvfp`: Python PID `42517` had elapsed `00:42:06`,
  process CPU time was `08:41:53`, and the artifact directory still contained
  only startup `run_metadata.json` plus inherited history ending at E128 step
  `8500` (`val_lddt_ca=0.4311057258844376`, FoldScore
  `0.4025340421795845`). No `results.json`, `results.csv`,
  `eval_details_full_msa_to_face.csv`, checkpoint, or status file existed.
  GPU memory was about `38.2 GiB` and utilization sampled at `12%`. Leave E139
  running while it remains below the documented cutoff.
- 2026-05-15T09:48Z Retired E139 early as an infeasible short gate on the
  owned pod. A final check at `09:47Z` showed Python PID `42517` still in the
  same no-write first-step state after `00:44:14` elapsed and `09:08:39`
  process CPU time: no `results.json`, `results.csv`,
  `eval_details_full_msa_to_face.csv`, checkpoint, or status file existed, and
  history still ended at inherited E128 step `8500`. Preserved the remote trace
  locally under
  `artifacts/nanofold_public_benchmarks/e139_no_hodge_oriented_boundary_from_e128_s9000_c256_m64/`
  plus `logs/e139_no_hodge_oriented_boundary.log`, then stopped only E139
  PIDs `42517` and `42514`. GPU memory returned to zero. Interpretation:
  oriented selected-boundary cochain readout is not a viable short-gate path in
  this implementation; pivot to E140, which targets the observed collapsed
  selected-complex realization diagnostic instead of adding another boundary
  readout.
- 2026-05-15T09:51Z Launched E140 on owned pod `c67fbk189vnvfp` from
  `/workspace/SimplexFold_e140` as
  `e140_selected_boundary_expansion_from_e128_s9000_c256_m64`. The checkout was
  fast-forwarded to pushed commit `050b954`, remote `python3 -m py_compile`
  passed for `simplex.py`, `evoformer.py`, `model.py`, `trainer.py`, and the
  NanoFold benchmark runner, and the run uses effective batch size `8`, crop
  `256`, MSA depth `64`, no extra MSA/templates, `n_cycles=4`, and max
  parameters `3261974`. E140 resumes the E128 checkpoint at step `8500` with
  weights only, keeps the E128 selected-complex recipe, and adds only
  selected face/tetra coordinate-expansion weights `0.05` with tolerance
  `0.05`.
- 2026-05-15T09:52Z E140 startup verification passed. Remote log shows
  `train=10000`, `val=1000`, resumed E128 at `step=8500` /
  `examples=68000`, loaded `1332` matching model tensors, and initialized `0`
  new/missing tensors. The runner wrote `run_metadata.json`, inherited
  `history_full_msa_to_face.json` ending at step `8500`
  (`val_lddt_ca=0.4311057258844376`), and wrote the new
  `status_full_msa_to_face.json`; the status heartbeat reported active step
  `8501`, microbatch `1`, and effective batch size `8`. The active E140 Python
  PID is `55949`; GPU memory was about `18.0 GiB` at the startup check.
- 2026-05-15T09:57Z E140 remained active and was making step progress on the
  owned pod: PID `55949` had elapsed `00:05:34`, process CPU time was
  `01:08:06`, `status_full_msa_to_face.json` reported active step `8503`,
  microbatch `4`, and effective batch size `8`, while history still correctly
  ended at inherited E128 step `8500` until the next evaluation/checkpoint.
  GPU memory was about `18.8 GiB`. This confirms the new heartbeat is needed
  to distinguish real step progress from an apparently quiet pre-eval run.
- 2026-05-15T09:59Z Corrected the interpretation of E130/E138/E139. E140's
  heartbeat shows that a resumed 500-step gate can make real step progress for
  many minutes while history still ends at inherited step `8500`; the next
  history/result write is not expected until the step-9000 evaluation. The old
  no-history cutoff therefore overcalled earlier stopped branches as runtime
  failures. Reclassify E130/E138/E139 as stopped pre-eval with no scored result,
  not as evidence that Hodge, face-cyclic, or oriented boundary-cochain
  readouts are architecturally bad. Future long-running gates should rely on
  `status_full_msa_to_face.json` heartbeat progress before any stop decision.
- 2026-05-15T10:05Z E140 remained active on owned pod `c67fbk189vnvfp`: PID
  `55949` had elapsed `00:13:37`, process CPU time was `02:49:22`, and the
  status heartbeat reported active step `8507`, microbatch `4`,
  `stopped_early=false`, and effective batch size `8`. History still correctly
  ends at inherited E128 step `8500` until the next step-9000 evaluation; no
  `results.json` exists yet. GPU memory was about `19.7 GiB`. Leave E140
  running.
- 2026-05-15T10:07Z Evaluated whether to launch a parallel E141 short gate
  while E140 continues. The prepared workspace is small (`32M` NanoFold,
  `100M` base SimplexFold, `105M` E141 checkout), so a second pod would be
  practical. Runpod capacity attempts for a same-image A100-SXM4-80GB secure
  pod and an RTX 5090 pod both failed with no deployable capacity/resources.
  No new pod was created. Keep only owned pod `c67fbk189vnvfp` running E140 and
  retry parallel capacity later only if useful.
- 2026-05-15T10:09Z E140 remained active on owned pod `c67fbk189vnvfp`: PID
  `55949` had elapsed `00:17:58`, process CPU time was `03:44:16`, and the
  status heartbeat reported active step `8509`, microbatch `4`,
  `stopped_early=false`, and effective batch size `8`. History still ends at
  inherited E128 step `8500` and no result/eval-detail files exist yet, as
  expected before the step-9000 evaluation. GPU memory was about `24.2 GiB`.
  A same-image H100 NVL secure pod request for parallel E141 also failed with
  no available requested capacity. No new pod was created; leave E140 running.
- 2026-05-15T10:12Z E140 still active on the owned pod: PID `55949` had elapsed
  `00:20:53`, process CPU time was `04:20:29`, and the status heartbeat
  reported active step `8511`, microbatch `1`, `stopped_early=false`, and
  effective batch size `8`. History still correctly ends at inherited E128 step
  `8500`, so the current scored best remains E128 (`val_lddt_ca=0.4311057258844376`).
  No `results.json` or eval-detail CSV exists yet; wait for the step-9000
  evaluation before judging E140.
- 2026-05-15T10:16Z Launched a second owned Runpod A100 pod for a parallel
  E141 gate after capacity became available:
  `5ox436mhzej7j4` (`codex-simplexfold-e141-parallel-runpod-20260515`), SSH
  `root@216.81.248.113 -p 13046`, image
  `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, A100-SXM4-80GB at
  `$1.49/hr`. This is the only new pod launched for E141; continue to avoid
  inspecting or managing unrelated Runpod instances.
- 2026-05-15T10:30Z E141 pod setup completed with public-only data staging:
  cloned SimplexFold branch `codex/simplexfold-topology-e07-boundary-coordinate`
  at commit `208193a89d366191b9f9166ac1dbff0211a2f051`, cloned NanoFold at
  commit `96afc8467a108aa8bee3b51cdf4a030cd656a960`, copied public NanoFold
  data to `/root/nanofold_data/`, removed macOS sidecar files, and symlinked
  the checkout data paths to the container-disk copies. Remote counts were
  `11000` feature `.npz` files, `11000` label `.npz` files,
  `preprocess_meta.json` present, and zero hidden sidecar files. The E128
  resume checkpoint exists at
  `/workspace/SimplexFold_e141/artifacts/nanofold_public_benchmarks/e128_damped_triangle_bias_from_e124_s8500_c256_m64/checkpoints/full_msa_to_face_latest.pt`
  with size `36627247` bytes.
- 2026-05-15T10:33Z Launched E141 from the separate owned pod and checkout
  `/workspace/SimplexFold_e141` as
  `e141_signed_face_cyclic_boundary_from_e128_s9000_c256_m64`. Remote
  `python3 -m py_compile` passed for `minalphafold/simplex.py`,
  `minalphafold/evoformer.py`, `minalphafold/model.py`,
  `minalphafold/trainer.py`, and the NanoFold benchmark runner. Parser
  validation accepted the launch command with effective batch size `8`,
  max-parameter cap `3261974`, E128 checkpoint present, and signed face-cyclic
  runtime final scale `0.25`. The active E141 Python PID is `576`; the wrapper
  shell was stopped after verifying PID `576` remained reparented to PID `1`.
  Startup log currently contains artifact/train/val lines and `run_metadata.json`;
  no heartbeat/history/result files have appeared yet.
- 2026-05-15T10:35Z Updated heartbeat automation
  `check-simplexfold-e57-runpod` to monitor only the two owned active pods:
  E140 pod `c67fbk189vnvfp` and E141 pod `5ox436mhzej7j4`. E140 remained
  active at PID `55949` with elapsed `00:44:17`, process CPU time `09:13:13`,
  status active step `8523`, microbatch `1`, `stopped_early=false`, and
  effective batch size `8`; history still ends at E128 step `8500` and no
  result/eval-detail files exist yet.
- 2026-05-15T10:37Z E141 startup verification passed. PID `576` remained
  active and reparented to PID `1`, `history_full_msa_to_face.json` was
  inherited from E128 ending at step `8500`, and
  `status_full_msa_to_face.json` reported active step `8501`, microbatch `1`,
  `stopped_early=false`, and effective batch size `8`. No result or
  eval-detail files exist yet, as expected before the step-9000 evaluation.
- 2026-05-15T10:39Z Both owned active runs are coherent and pre-eval. E140 on
  pod `c67fbk189vnvfp` had PID `55949`, elapsed `00:47:51`, process CPU time
  `09:57:45`, status active step `8525`, microbatch `1`, effective batch size
  `8`, and `stopped_early=false`; history still ends at E128 step `8500` and
  no result/eval-detail files exist. E141 on pod `5ox436mhzej7j4` had PID
  `576`, elapsed `00:05:57`, process CPU time `00:44:30`, status active step
  `8502`, microbatch `1`, effective batch size `8`, and
  `stopped_early=false`; history also still ends at E128 step `8500` with no
  result/eval-detail files. Leave both running.
- 2026-05-15T10:49Z Rechecked only the two owned active Runpod pods. E140 on
  `c67fbk189vnvfp` remains active at PID `55949`, elapsed `00:53:10`, process
  CPU time `11:03:43`, status active step `8528`, completed step `8527`,
  microbatch `1`, effective batch size `8`, and `stopped_early=false`. E141 on
  `5ox436mhzej7j4` remains active at PID `576`, elapsed `00:11:16`, process
  CPU time `01:53:23`, status active step `8504`, completed step `8503`,
  microbatch `1`, effective batch size `8`, and `stopped_early=false`. Both
  runs still have only inherited history through E128 step `8500`; neither has
  `results.json`, `results.csv`, `eval_details_full_msa_to_face.csv`, or a new
  checkpoint yet.
- 2026-05-15T10:49Z Confirmed the user-provided PDFs are saved in the repo at
  `references/papers/hands_on_geometric_deep_learning_nodes_to_complexes.pdf`
  and `references/papers/2509.03885v1.pdf`. Extracted and read full text from
  both PDFs. The general TDL guide reinforces the framing around cochains,
  incidence matrices, filtrations, and inter-neighborhood aggregation. The
  Topotein paper is the more experiment-actionable source: its Protein
  Combinatorial Complex and TCPNet design argue for persistent multi-rank
  protein states, directed incidence/adjacency operators, edge-centric
  scalarization, and outer-edge neighborhoods that let higher-rank cells
  communicate through external directed edges. It also explicitly cautions
  that shallow higher-rank feature addition without dedicated update
  mechanisms can hurt, which matches our negative readout-only and weak
  feedback branches. The immediate experiment consequence is to keep E140/E141
  running, then consider an E145 outer-neighborhood transport candidate only if
  they return below threshold; the existing trainable
  `simplex_outer_edge_context_scale` hook matches this idea but exceeds the
  parameter cap when added to the full E128 recipe unless another
  parameterized path is removed or a lower-rank/parameter-neutral variant is
  implemented.
- 2026-05-15T10:58Z Staged the parameter-neutral E145 outer-neighborhood
  transport hook locally. New model config/CLI field:
  `simplex_outer_edge_residual_context_scale`. The path pools directed
  external pair edges for each selected face/tetra cell, separates symmetric
  and oriented outer-edge context, folds that context into the active
  face/tetra cochain width without parameters, RMS-matches it to the current
  cell state, and gates it with the existing cell gate. This directly follows
  the Topotein outer-edge-neighborhood idea while avoiding the parameter cost
  of the trainable `simplex_outer_edge_context_scale` MLPs. Focused
  verification passed: `python -m py_compile minalphafold/simplex.py
  minalphafold/model_config.py minalphafold/trainer.py
  scripts/run_nanofold_public_benchmarks.py`; targeted pytest for the new
  outer-edge delta, adapter behavior, parameter audit, trainer CLI, and
  NanoFold runner config override (`5 passed`). Parameter audit for the full
  E128-style recipe plus E145 scale `0.25` remains `3,240,738`, with `21,236`
  parameters of headroom under the AF2-medium+5% cap.
- 2026-05-15T11:05Z Rechecked only the two owned active Runpod pods after the
  30k-candidate assessment. E140 on pod `c67fbk189vnvfp` remains active at
  PID `55949`, elapsed `01:13:19`, process CPU `1251%`, status active step
  `8538`, completed step `8537`, effective batch size `8`, and
  `stopped_early=false`. E141 on pod `5ox436mhzej7j4` remains active at PID
  `576`, elapsed `00:31:24`, process CPU `1179%`, status active step `8511`,
  completed step `8510`, effective batch size `8`, and
  `stopped_early=false`. Both runs remain pre-eval: inherited history still
  ends at E128 step `8500`, with no `results.json`,
  `eval_details_full_msa_to_face.csv`, or new checkpoint yet. Current
  candidate read: E140 and especially the staged E145 are plausible
  topology-native routes to test because they target the local-to-global
  assembly/Rg-collapse gap, but no branch should receive a 30k confirmation
  spend unless a short gate first clears about `0.45` C-alpha lDDT with
  coherent FoldScore, dRMSD, and C-alpha radius.
- 2026-05-15T11:08Z Re-audited the returned E128 eval-detail CSV while
  E140/E141 continue. The failure is strongly length/global-assembly shaped,
  not simply a missing local simplex metric. Mean C-alpha lDDT by length bin:
  `<80`: `0.5096`, `80-119`: `0.4546`, `120-159`: `0.4171`,
  `160-219`: `0.3979`, `>=220`: `0.3919`. The corresponding mean selected
  boundary lDDT stays nearly flat around `0.744-0.753`, while mean predicted
  C-alpha Rg ratio drops from `0.9126` for `<80` residues to `0.6229` for
  `>=220` residues. Across all 1000 chains, primary lDDT correlates with mean
  selected boundary lDDT (`r=0.6128`) and predicted C-alpha Rg (`r=0.5958`),
  but only weakly with Rg ratio (`r=0.2133`) and not meaningfully with dRMSD
  (`r=-0.0693`). The top 100 chains average length `82.9`, boundary lDDT
  `0.8160`, and lDDT `0.6381`; the bottom 100 average length `176.6`,
  boundary lDDT `0.6746`, and lDDT `0.3404`. There are also 75 chains with
  high selected-boundary lDDT (`>=0.75`) but low global lDDT (`<0.4`), so a
  local-boundary-only objective is unlikely to solve the gap by itself. This
  supports prioritizing E140/E145-style selected-complex realization and
  outer-neighborhood communication over generic C-alpha lDDT/Rg losses.
- 2026-05-15T11:15Z Added reusable aggregate eval-detail analyzer
  `scripts/analyze_nanofold_eval_details.py` plus tests. The helper computes
  length-bin metrics, lDDT strata, correlations against primary C-alpha lDDT,
  Rg summaries, selected-boundary diagnostics, boundary/outer-edge degree
  summaries, and the high-boundary/low-global subset without printing chain
  identifiers. Focused validation passed: `python -m py_compile
  scripts/analyze_nanofold_eval_details.py`; `python -m pytest
  tests/test_analyze_nanofold_eval_details.py` (`3 passed`); and
  `../../.venv/bin/ruff check scripts/analyze_nanofold_eval_details.py
  tests/test_analyze_nanofold_eval_details.py`. A real E128 smoke reproduced
  the documented aggregate signal: `1000` rows, mean lDDT `0.4311`, length-bin
  lDDT from `0.5096` (`<80`) to `0.3919` (`>=220`), and 75
  high-boundary/low-global rows.
- 2026-05-15T11:18Z Rechecked only the two owned active pods. E140 remains
  pre-eval at active step `8544`, completed step `8543`, elapsed `01:25:36`,
  with inherited history still ending at E128 step `8500` and no
  `results.json`, eval-details CSV, or new checkpoint. E141 remains pre-eval
  at active step `8515`, completed step `8514`, elapsed `00:43:42`, also with
  no returned result artifacts. Both jobs are CPU-heavy with large A100 memory
  allocations (`~38 GiB`); a short utilization sample saw E141 spike to `66%`
  GPU once while E140 did not spike in that window. Treat the slow heartbeat as
  expected for these gates and continue waiting for the step-9000 evaluation
  unless the process exits or status stops advancing for a much longer window.
- 2026-05-15T11:23Z Runtime triage for future gates: E140/E141 were launched
  without `--num-workers`, so they inherited the benchmark runner default
  `0`. Both owned A100 pods report `128` CPU cores, while the active jobs are
  CPU-heavy and only burst the GPU intermittently. Do not mutate E140/E141
  mid-flight, but for the next short gate use a cautious startup smoke with a
  small DataLoader worker count, e.g. `--num-workers 4`, and verify the usual
  run metadata/status before letting it continue. This is a throughput knob,
  not an architecture/loss/data change.
- 2026-05-15T11:26Z Added a focused dataloader regression test backing the
  future `--num-workers` runtime knob:
  `test_build_dataloader_worker_count_preserves_first_training_batch` compares
  the first seeded training batch for `num_workers=0` and `num_workers=2`.
  Validation passed: the new test alone (`1 passed`), the three nearby
  dataloader feature tests (`3 passed`), `git diff --check`, and
  `ruff check --select F821,F822,F823 tests/test_trainer.py`. Full-file ruff
  still flags pre-existing import-order/unused-import/zip-style cleanup in
  `tests/test_trainer.py`, so this change keeps the verification scoped to the
  new guard and syntax/name errors.
- 2026-05-15T11:31Z Optimized the staged E145 parameter-free
  outer-neighborhood path before launch. `_fold_feature_channels` no longer
  loops once per target channel; it pads, reshapes, sums, and divides by the
  exact per-offset counts, preserving the old offset-mean behavior including
  zero-filled tail channels. Added
  `test_fold_feature_channels_matches_offset_mean_reference` and reran the
  E145-focused simplex tests (`3 passed`), `tests/test_simplex.py` (`95
  passed`), `python -m py_compile minalphafold/simplex.py tests/test_simplex.py`,
  `ruff check --select F821,F822,F823 minalphafold/simplex.py
  tests/test_simplex.py`, `ruff check --select I001 tests/test_simplex.py`, and
  `git diff --check`. A local CPU smoke for a representative
  `[1, 256, 48, 128] -> 32` fold matched the reference exactly and measured
  about `0.95 ms` vectorized versus `10.54 ms` for the prior loop.
- 2026-05-15T11:36Z Added artifact-level auditability for the future
  `--num-workers` throughput knob. The NanoFold benchmark runner now records
  `num_workers` in live status JSON, final `results.json`/`results.csv`, and
  `run_metadata.json`, so a returned E145-style gate can prove whether it used
  the worker-count path. Focused validation passed:
  `python -m pytest
  tests/test_nanofold_public_benchmarks.py::test_num_workers_guardrail_is_accepted_by_cli_parser
  tests/test_nanofold_public_benchmarks.py::test_run_status_payload_tracks_live_progress`
  (`2 passed`), `python -m py_compile
  scripts/run_nanofold_public_benchmarks.py tests/test_nanofold_public_benchmarks.py`,
  and a `_write_csv` smoke showing the `num_workers` column.
- 2026-05-15T11:42Z Rechecked only the two owned active Runpod pods. E140 is
  still pre-eval at active step `8556`, completed step `8555`, elapsed
  `01:50:09`, with inherited history ending at E128 step `8500` and no
  `results.json`, eval-details CSV, or new checkpoint. E141 is still pre-eval
  at active step `8525`, completed step `8524`, elapsed `01:08:15`, also with
  no returned result artifacts. Both status files still show effective batch
  size `8` and `stopped_early=false`; continue waiting rather than treating
  either as scored evidence.
- 2026-05-15T11:45Z Rechecked only the owned E140/E141 endpoints plus Runpod
  GPU capacity. E140 advanced to active step `8558`, completed step `8557`,
  elapsed `01:52:40`; E141 advanced to active step `8526`, completed step
  `8525`, elapsed `01:10:45`. Neither has a result bundle, eval-details CSV,
  or checkpoint yet. `runpodctl gpu list --include-unavailable` showed no
  currently available 80GB-class secure GPU (`A100`, `H100`, `H200`, `B200`,
  `B300` all unavailable), while the available 24GB-class GPUs are too small
  for the crop-256 full gate that has been allocating about `38 GiB`. Do not
  launch E145 on undersized hardware; wait for E140/E141 or 80GB-class
  capacity.
- 2026-05-15T11:47Z Updated the active heartbeat automation
  `check-simplexfold-e57-runpod` so it continues monitoring only the owned
  E140/E141 pods, verifies and pulls any coherent returned result before
  documenting `EXPERIMENT_RESULTS.md`, and treats E145 as a parked candidate
  only after E140/E141 return below the `0.45` short-gate threshold or fail
  coherently. The heartbeat now explicitly forbids launching E145 on
  undersized 24GB/32GB hardware and requires suitable 80GB-class capacity plus
  the audited `--num-workers 4` runtime knob for any future E145 short gate.
- 2026-05-15T11:59Z Added runtime-ramp plumbing for the E145
  parameter-free outer-edge residual context path. The new
  `simplex_outer_edge_residual_context_runtime_scale*` flags thread through the
  NanoFold runner, trainer schedules, model-input overrides, AlphaFold2,
  Evoformer, and the SimplicialAdapter so the resumed E128 checkpoint can ramp
  the Topotein-style external-edge cochain update from `0.0` to `0.25` across
  steps `8500`-`9000`. This is still a topology-native cochain transport
  change and adds zero parameters. Validation passed: py_compile for the
  touched model/trainer/runner/test files, the focused E145/runtime plumbing
  pytest set (`11 passed`), and the E128-style parameter audit remains
  `3,240,738 <= 3,261,974`. The heartbeat automation was updated to require
  these ramp flags for any future E145 launch.
- 2026-05-15T12:05Z Added `scripts/summarize_nanofold_run_status.py`, a
  read-only local artifact summarizer for live or returned benchmark run
  directories. It reports active/returned/partial/startup state from
  `status_*.json`, `results.json`, eval-detail row count, history last step,
  and checkpoint presence without surfacing chain-level eval details. Focused
  validation passed: `python -m pytest
  tests/test_summarize_nanofold_run_status.py` (`3 passed`),
  `python -m py_compile scripts/summarize_nanofold_run_status.py
  tests/test_summarize_nanofold_run_status.py`, and
  `ruff check --select F821,F822,F823` on the new script/test.
- 2026-05-15T12:15Z Extended the local run-status summarizer with live
  throughput and ETA fields derived from the runner's existing
  `start_step`, `completed_step`, `target_steps`, and
  `elapsed_seconds_total` heartbeat fields. This is monitoring-only: it does
  not change training behavior, result scoring, data access, or Runpod
  launches. Focused validation passed: `python -m pytest
  tests/test_summarize_nanofold_run_status.py` (`4 passed`),
  `python -m py_compile scripts/summarize_nanofold_run_status.py
  tests/test_summarize_nanofold_run_status.py`, and
  `../../.venv/bin/ruff check --select F821,F822,F823
  scripts/summarize_nanofold_run_status.py
  tests/test_summarize_nanofold_run_status.py`.
- 2026-05-15T12:19Z Corrected the local ETA helper before relying on it for
  decisions. E140/E141 heartbeats showed `elapsed_seconds_total` is inherited
  from the resumed checkpoint, so it is useful as total training time but not
  as a run-local throughput denominator. The summarizer now estimates active
  run rate from the status-file mtime minus `run_metadata.json` mtime and
  labels that source explicitly. Focused validation passed again:
  `python -m pytest tests/test_summarize_nanofold_run_status.py` (`4 passed`),
  `python -m py_compile scripts/summarize_nanofold_run_status.py
  tests/test_summarize_nanofold_run_status.py`, and
  `../../.venv/bin/ruff check --select F821,F822,F823
  scripts/summarize_nanofold_run_status.py
  tests/test_summarize_nanofold_run_status.py`.
- 2026-05-15T12:23Z Added a future-proof runner heartbeat field,
  `elapsed_seconds_run`, so future launched gates report run-local wall time
  directly instead of forcing the summarizer to infer it from file mtimes.
  The summarizer now prefers `elapsed_seconds_run` and falls back to
  status/metadata mtime deltas for older active runs such as E140/E141.
  This remains monitoring-only and does not affect training, scoring, or the
  sealed NanoFold data path. Focused validation passed:
  `python -m pytest tests/test_summarize_nanofold_run_status.py
  tests/test_nanofold_public_benchmarks.py::test_run_status_payload_tracks_live_progress`
  (`6 passed`), `python -m py_compile
  scripts/run_nanofold_public_benchmarks.py
  scripts/summarize_nanofold_run_status.py
  tests/test_summarize_nanofold_run_status.py
  tests/test_nanofold_public_benchmarks.py`, and
  `../../.venv/bin/ruff check --select F821,F822,F823
  scripts/run_nanofold_public_benchmarks.py
  scripts/summarize_nanofold_run_status.py
  tests/test_summarize_nanofold_run_status.py
  tests/test_nanofold_public_benchmarks.py`.
- 2026-05-15T12:29Z Tightened returned-artifact verification for future
  `--num-workers 4` gates. `scripts/verify_nanofold_benchmark_artifacts.py`
  now accepts `--expected-num-workers` and checks the result row,
  `run_metadata.json`, and optional `status_<variant>.json` agree on expected
  worker count; it also cross-checks expected effective batch size against
  metadata/status when those fields are present. This is verification-only and
  does not touch the live E140/E141 jobs. Focused validation passed:
  `python -m pytest tests/test_verify_nanofold_benchmark_artifacts.py`
  (`8 passed`), `python -m py_compile
  scripts/verify_nanofold_benchmark_artifacts.py
  tests/test_verify_nanofold_benchmark_artifacts.py`, and
  `../../.venv/bin/ruff check --select F821,F822,F823
  scripts/verify_nanofold_benchmark_artifacts.py
  tests/test_verify_nanofold_benchmark_artifacts.py`.
- 2026-05-15T12:34Z Added an exact E145 launch-recipe regression test:
  `test_e145_outer_residual_context_recipe_matches_documented_gate`. The test
  locks the documented run name, step target `9000`, effective batch size `8`,
  crop `256`, MSA depth `64`, no extra MSA/templates, parameter cap
  `3261974`, `--num-workers 4`, parameter-free residual outer-edge context
  scale `0.25`, and the runtime ramp from `0.0` at step `8500` to `0.25` at
  step `9000`. Focused validation passed:
  `python -m pytest
  tests/test_nanofold_public_benchmarks.py::test_e145_outer_residual_context_recipe_matches_documented_gate`
  (`1 passed`), `python -m py_compile
  tests/test_nanofold_public_benchmarks.py`, `../../.venv/bin/ruff check
  --select F821,F822,F823 tests/test_nanofold_public_benchmarks.py`, and
  `git diff --check`.
- 2026-05-15T12:40Z Tightened the `EXPERIMENT_RESULTS.md` row formatter for
  returned multi-variant `results.json` files. `scripts/format_experiment_result_row.py`
  now accepts `--variant` and selects exactly one matching result row instead
  of blindly formatting the first row. This keeps the final-results handoff
  aligned with the verifier/summarizer behavior and reduces manual row-editing
  risk once E140/E141/E145 return. Focused validation passed:
  `python -m pytest tests/test_format_experiment_result_row.py` (`3 passed`),
  `python -m py_compile scripts/format_experiment_result_row.py
  tests/test_format_experiment_result_row.py`, `../../.venv/bin/ruff check
  --select F821,F822,F823 scripts/format_experiment_result_row.py
  tests/test_format_experiment_result_row.py`, and `git diff --check`.
- 2026-05-15T12:47Z Improved the local active-run summarizer for long resumed
  gates such as E140/E141. `scripts/summarize_nanofold_run_status.py` now
  includes the last aggregate validation row from `history_<variant>.json` and
  falls back to that `val_lddt_ca` when an active run has no returned
  `results.json` yet. This is monitoring-only and does not read chain-level
  eval details or affect training/scoring. Focused validation passed:
  `python -m pytest tests/test_summarize_nanofold_run_status.py` (`5 passed`),
  `python -m py_compile scripts/summarize_nanofold_run_status.py
  tests/test_summarize_nanofold_run_status.py`, and `../../.venv/bin/ruff
  check --select F821,F822,F823 scripts/summarize_nanofold_run_status.py
  tests/test_summarize_nanofold_run_status.py`.
- 2026-05-15T12:50Z Rechecked only the two owned active Runpod pods by
  snapshotting their small status/history/metadata artifacts into `/tmp` and
  running the local summarizer. Neither run has returned a result bundle,
  eval-details CSV, or checkpoint. E140 is active at completed step `8591`
  with estimated rate `30.4` steps/hour and ETA `13.5h`; E141 is active at
  completed step `8549` with estimated rate `21.3` steps/hour and ETA `21.2h`.
  Both snapshots correctly show inherited history ending at E128 step `8500`
  with aggregate `val_lddt_ca=0.4311057258844376`, so the best scored result
  remains E128 until a new eval row returns.
- 2026-05-15T12:52Z Tightened returned-result handoff documentation before
  E140/E141 finish. `scripts/verify_nanofold_benchmark_artifacts.py` now
  parses `--metadata key=null` / `key=none` as JSON `null`, so recipe checks
  can prove disabled topology knobs as well as enabled ones. `EXPERIMENTS.md`
  now contains exact E140 and E141 verifier command templates with
  run-name/model-config/recipe metadata expectations, and explicitly omits
  `--expected-num-workers` for those two older launches because they predate
  worker-count metadata.
- 2026-05-15T12:55Z Added the full parked E145 launch skeleton and returned
  verifier template to `EXPERIMENTS.md`, and pointed `PLAN.md` at that exact
  handoff path. The template keeps the full E128 selected-complex recipe,
  adds only the parameter-free outer-edge residual context ramp from `0.0` to
  `0.25` over steps `8500`-`9000`, and includes `--num-workers 4` plus
  returned-artifact verification with `--expected-num-workers 4`. This is a
  launch-safety/documentation update only; E145 remains parked until E140/E141
  return below the short-gate threshold or fail coherently.
- 2026-05-15T12:57Z Fixed a result-row formatting handoff issue before
  E140/E141 return. `scripts/format_experiment_result_row.py` now accepts both
  historical `val_pred_ca_rg` / `val_true_ca_rg` keys and canonical
  `val_ca_pred_rg` / `val_ca_true_rg` keys when filling the C-alpha Rg column,
  with a regression test based on the E128-style metric names. Without this,
  a coherent returned result could have been verified correctly but still
  formatted with a blank Rg column in `EXPERIMENT_RESULTS.md`.
- 2026-05-15T12:59Z Updated the heartbeat automation
  `check-simplexfold-e57-runpod` in place. It still monitors only the owned
  E140 pod `c67fbk189vnvfp` and E141 pod `5ox436mhzej7j4`, but now points the
  returned-result handoff at the exact E140/E141 verifier templates in
  `EXPERIMENTS.md`, the variant-aware row formatter, the E145 full launch and
  verifier template, and latest pushed branch tip `9fe602a`. The schedule
  remains `FREQ=MINUTELY;INTERVAL=30`.
- 2026-05-15T13:01Z Added launch-recipe regression tests for the two active
  gates. `test_e140_selected_boundary_expansion_recipe_matches_running_gate`
  locks E140's selected-boundary coordinate-expansion recipe, default
  `num_workers=0`, and absence of signed face-cyclic readout.
  `test_e141_signed_face_cyclic_recipe_matches_running_gate` locks E141's
  signed face-cyclic static scale and `8500`-to-`9000` runtime ramp while
  confirming coordinate expansion stays disabled. Focused validation passed:
  the E140/E141/E145 recipe-guard pytest set (`3 passed`), `py_compile` on
  `tests/test_nanofold_public_benchmarks.py`, ruff `F821/F822/F823` on the
  same test file, and `git diff --check`.
- 2026-05-15T13:04Z Updated heartbeat automation
  `check-simplexfold-e57-runpod` in place after the E140/E141 recipe-guard
  commit. The heartbeat no longer pins a specific branch SHA; it now tells the
  worker to use the latest pushed
  `codex/simplexfold-topology-e07-boundary-coordinate` commit, which includes
  the active/parked launch-recipe guards, exact verifier templates,
  variant-aware row formatter, and C-alpha Rg key compatibility. Owned-pod
  scope and the 30-minute schedule are unchanged.
- 2026-05-15T13:06Z Added `scripts/upsert_experiment_result_row.py`, a small
  helper that mechanically inserts or replaces one Markdown table row in
  `EXPERIMENT_RESULTS.md`. This reduces manual row-editing risk after a
  returned run is verified, while still leaving the `Last updated` and
  `Best validation C-alpha lDDT` header text as an explicit post-verification
  decision. Focused validation passed: `python -m pytest
  tests/test_upsert_experiment_result_row.py` (`3 passed`), `py_compile` on
  the helper/test, ruff `F821/F822/F823` on the helper/test, and
  `git diff --check`.
- 2026-05-15T13:11Z Added
  `scripts/refresh_experiment_results_summary.py`, which refreshes the
  `EXPERIMENT_RESULTS.md` date and best-score summary from the Markdown table
  after verified rows are inserted. It ignores nonnumeric stopped-pre-eval
  rows and treats a short-gate score above `0.7` as still needing the
  30,000-step confirmation. Focused validation passed: `python -m pytest
  tests/test_refresh_experiment_results_summary.py` (`4 passed`),
  `py_compile` on the helper/test, ruff `F821/F822/F823` on the helper/test,
  and `git diff --check`.
- 2026-05-15T13:14Z Added `scripts/record_experiment_result.py`, a thin
  post-verification wrapper that formats one returned artifact directory,
  upserts the corresponding `EXPERIMENT_RESULTS.md` table row, and refreshes
  the best-score summary header. This does not replace the artifact verifier
  or eval-detail analyzer; it is the final recording step after those pass.
  Focused validation passed: `python -m pytest
  tests/test_record_experiment_result.py` (`2 passed`), `py_compile` on the
  helper/test, ruff `F821/F822/F823` on the helper/test, and `git diff --check`.
- 2026-05-15T13:16Z Updated heartbeat automation
  `check-simplexfold-e57-runpod` in place after the verified-result recording
  wrapper landed. The returned-result handoff now says to verify artifacts
  with the exact E140/E141 templates, run the eval-detail analyzer, then call
  `scripts/record_experiment_result.py` to update `EXPERIMENT_RESULTS.md`.
  Owned-pod scope and the 30-minute schedule are unchanged.
- 2026-05-15T13:19Z Dry-ran the full returned-result handoff on the existing
  local E128 artifact bundle without modifying tracked files. The artifact
  verifier passed with `completed_steps=8500`, effective batch size `8`,
  parameters `3,240,738 <= 3,261,974`, `1000` eval rows, history ending at
  step `8500`, and `stopped_early=false`. The eval-detail analyzer reproduced
  the known E128 diagnostics (`mean val_lddt_ca=0.4311`, length/global-assembly
  failure, and high-boundary/low-global subset). The
  `scripts/record_experiment_result.py` dry-run on a temporary copy of
  `EXPERIMENT_RESULTS.md` preserved the E128 best-summary header and generated
  the expected table row with C-alpha Rg intact.
- 2026-05-16T15:33Z E145 is now active on owned Runpod pod `723hbew2jrvxjx`
  (`root@195.26.233.76 -p 31813`, A100 SXM 80GB, image
  `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`). The remote checkout is
  `/workspace/SimplexFold_e145` at commit
  `780017258110abc95640d9d4b6c0cdf723b71dc8`; NanoFold is staged at
  `/workspace/nanoFold-Competition` commit
  `96afc8467a108aa8bee3b51cdf4a030cd656a960`. Public data staging was
  verified before launch with `11000` feature NPZs, `11000` label NPZs,
  train/val manifest counts `10000 / 1000`, `preprocess_meta.json` present,
  and no macOS `._*` sidecars. The E128 checkpoint was present at
  `/workspace/SimplexFold_e145/artifacts/nanofold_public_benchmarks/e128_damped_triangle_bias_from_e124_s8500_c256_m64/checkpoints/full_msa_to_face_latest.pt`
  with size `35M`.
- 2026-05-16T15:33Z E145 launch/runtime coherence check passed. Run name:
  `e145_outer_residual_context_from_e128_s9000_c256_m64`; log:
  `/workspace/SimplexFold_e145/logs/e145_outer_residual_context.log`;
  artifact directory:
  `/workspace/SimplexFold_e145/artifacts/nanofold_public_benchmarks/e145_outer_residual_context_from_e128_s9000_c256_m64`.
  The launch wrapper PID is `345`; the active trainer parent is PID `347`,
  with DataLoader worker children observed under the same command. Remote
  metadata confirms `steps=9000`, `effective_batch_size=8`, `num_workers=4`,
  `max_parameters=3261974`, crop `256`, MSA depth `64`,
  `simplex_outer_edge_residual_context_scale=0.25`, runtime scale `0.0`,
  runtime final `0.25`, ramp start `8500`, and ramp length `500`. The status
  heartbeat at `2026-05-16T15:33:11Z` showed `completed_step=8525`, active
  step `8526`, active microbatch `1 / 8`, `stopped_early=false`, inherited
  history ending at E128 step `8500`, and no `results.json`, result CSV,
  eval-detail CSV, or new checkpoint yet. Leave E145 running and update
  `EXPERIMENT_RESULTS.md` only after a scored result or documented terminal
  no-score outcome.
- 2026-05-16T15:37Z E145 remains live and coherent on owned pod
  `723hbew2jrvxjx`. The status heartbeat showed `completed_step=8545`,
  active step `8546`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=4`, `stopped_early=false`, and last train loss
  `4.621459543704987`. The process list still shows the launch wrapper PID
  `345`, trainer parent PID `347`, and DataLoader worker children under the
  same benchmark command. Artifact listing still contains only
  `run_metadata.json`, inherited `history_full_msa_to_face.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or new checkpoint exists yet. Do not update
  `EXPERIMENT_RESULTS.md` until E145 returns a scored bundle or terminal
  no-score outcome.
- 2026-05-16T15:39Z E145 remains live and pre-eval on owned pod
  `723hbew2jrvxjx`. The status heartbeat advanced to `completed_step=8556`,
  active step `8557`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=4`, `stopped_early=false`, and last train loss
  `4.648955553770065`. Artifact inventory is still pre-return:
  `run_metadata.json`, inherited `history_full_msa_to_face.json`, and
  `status_full_msa_to_face.json` only; no `results.json`, result CSV,
  eval-detail CSV, or new checkpoint. Leave the run active and continue to
  judge it only after the step-9000 returned bundle or a documented terminal
  no-score outcome.
- 2026-05-16T15:41Z E145 remains live and pre-eval on owned pod
  `723hbew2jrvxjx`. The status heartbeat advanced to `completed_step=8567`,
  active step `8568`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=4`, `stopped_early=false`, and last train loss
  `4.645114898681641`. The process list still shows the launch wrapper,
  trainer parent, and DataLoader workers for the E145 command. Artifact
  inventory remains pre-return with only `run_metadata.json`, inherited
  `history_full_msa_to_face.json`, and `status_full_msa_to_face.json`; no
  `results.json`, result CSV, eval-detail CSV, or new checkpoint exists yet.
  Keep the run active and continue to withhold `EXPERIMENT_RESULTS.md` updates
  until a scored bundle or terminal no-score outcome exists.
- 2026-05-16T15:42Z E145 remains live and pre-eval on owned pod
  `723hbew2jrvxjx`. The status heartbeat advanced to `completed_step=8575`,
  active step `8576`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=4`, `stopped_early=false`, and last train loss
  `4.8212475180625916`. The remote artifact directory still contains only
  `run_metadata.json`, inherited `history_full_msa_to_face.json`, and
  `status_full_msa_to_face.json`; no result bundle, eval-detail CSV, or new
  checkpoint has been written. Continue monitoring; do not record a final
  result until the step-9000 artifact bundle exists or the run reaches a
  documented terminal no-score state.
- 2026-05-16T15:44Z E145 remains live and pre-eval on owned pod
  `723hbew2jrvxjx`. The status heartbeat advanced to `completed_step=8585`,
  active step `8586`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=4`, `stopped_early=false`, and last train loss
  `4.467980057001114`. The run is still before the step-9000 evaluation
  boundary: the artifact directory contains only `run_metadata.json`,
  inherited `history_full_msa_to_face.json`, and `status_full_msa_to_face.json`.
  No `results.json`, result CSV, eval-detail CSV, or new checkpoint exists yet,
  so `EXPERIMENT_RESULTS.md` remains unchanged.
- 2026-05-16T15:46Z E145 remains live and pre-eval on owned pod
  `723hbew2jrvxjx`. The status heartbeat advanced to `completed_step=8594`,
  active step `8595`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=4`, `stopped_early=false`, and last train loss
  `4.804234325885773`. The process list still shows the E145 launch wrapper,
  trainer parent, and DataLoader workers. Artifact inventory remains
  pre-return: `run_metadata.json`, inherited `history_full_msa_to_face.json`,
  and `status_full_msa_to_face.json` only; no `results.json`, result CSV,
  eval-detail CSV, or new checkpoint exists yet.
- 2026-05-16T15:48Z E145 remains live and pre-eval on owned pod
  `723hbew2jrvxjx`. The status heartbeat advanced to `completed_step=8604`,
  active step `8605`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=4`, `stopped_early=false`, and last train loss
  `4.80533841252327`. Artifact inventory remains pre-return with only
  `run_metadata.json`, inherited `history_full_msa_to_face.json`, and
  `status_full_msa_to_face.json`; no result bundle, eval-detail CSV, or new
  checkpoint exists yet.
- 2026-05-16T15:50Z E145 remains live and pre-eval on owned pod
  `723hbew2jrvxjx`. The status heartbeat advanced to `completed_step=8614`,
  active step `8615`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=4`, `stopped_early=false`, and last train loss
  `4.382141292095184`. Artifact inventory remains pre-return with only
  `run_metadata.json`, inherited `history_full_msa_to_face.json`, and
  `status_full_msa_to_face.json`; no result bundle, eval-detail CSV, or new
  checkpoint exists yet.
- 2026-05-16T15:52Z E145 remains live and pre-eval on owned pod
  `723hbew2jrvxjx`. The status heartbeat advanced to `completed_step=8631`,
  active step `8632`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=4`, `stopped_early=false`, and last train loss
  `4.926614582538605`. The process list still shows the launch wrapper PID
  `345`, trainer parent PID `347`, and DataLoader workers for the E145 command.
  Artifact inventory remains pre-return with only `run_metadata.json`,
  inherited `history_full_msa_to_face.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or new checkpoint exists yet, so `EXPERIMENT_RESULTS.md`
  remains unchanged.
- 2026-05-16T15:54Z E145 remains live and pre-eval on owned pod
  `723hbew2jrvxjx`. The status heartbeat advanced to `completed_step=8642`,
  active step `8643`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=4`, `stopped_early=false`, and last train loss
  `4.646269261837006`. The launch wrapper, trainer parent, and DataLoader
  workers are still alive under the E145 command. Artifact inventory remains
  pre-return with only `run_metadata.json`, inherited
  `history_full_msa_to_face.json`, and `status_full_msa_to_face.json`; no
  `results.json`, result CSV, eval-detail CSV, or new checkpoint exists yet.
- 2026-05-16T15:56Z E145 remains live and pre-eval on owned pod
  `723hbew2jrvxjx`. The status heartbeat advanced to `completed_step=8651`,
  active step `8652`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=4`, `stopped_early=false`, and last train loss
  `4.731453955173492`. The process tree still shows the launch wrapper PID
  `345`, trainer parent PID `347`, and DataLoader workers under the E145
  command. Artifact inventory remains pre-return with only `run_metadata.json`,
  inherited `history_full_msa_to_face.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or new checkpoint exists yet.
- 2026-05-16T15:58Z E145 remains live and pre-eval on owned pod
  `723hbew2jrvxjx`. The status heartbeat advanced to `completed_step=8659`,
  active step `8660`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=4`, and `stopped_early=false`; the latest status now reports
  `last_train_loss=NaN`, so monitor the next heartbeat for recurrence before
  treating it as a terminal training failure. The E145 launch wrapper, trainer
  parent, and DataLoader workers are still alive. Artifact inventory remains
  pre-return with only `run_metadata.json`, inherited
  `history_full_msa_to_face.json`, and `status_full_msa_to_face.json`; no
  `results.json`, result CSV, eval-detail CSV, or new checkpoint exists yet.
- 2026-05-16T15:59Z E145 remains live and pre-eval on owned pod
  `723hbew2jrvxjx`. The status heartbeat advanced to `completed_step=8669`,
  active step `8670`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=4`, `stopped_early=false`, and last train loss returned to a
  finite `4.631755948066711` after the previous one-heartbeat `NaN`. The
  process tree remains alive under the E145 command. Artifact inventory is
  still pre-return with only `run_metadata.json`, inherited
  `history_full_msa_to_face.json`, and `status_full_msa_to_face.json`; no
  `results.json`, result CSV, eval-detail CSV, or new checkpoint exists yet.
- 2026-05-16T16:01Z E145 remains live and pre-eval on owned pod
  `723hbew2jrvxjx`. The status heartbeat advanced to `completed_step=8678`,
  active step `8679`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=4`, `stopped_early=false`, and finite last train loss
  `4.4152931571006775`. The E145 process tree is still alive. Artifact
  inventory remains pre-return with only `run_metadata.json`, inherited
  `history_full_msa_to_face.json`, and `status_full_msa_to_face.json`; no
  `results.json`, result CSV, eval-detail CSV, or new checkpoint exists yet.
- 2026-05-16T16:03Z E145 remains live and pre-eval on owned pod
  `723hbew2jrvxjx`. The status heartbeat advanced to `completed_step=8687`,
  active step `8688`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=4`, `stopped_early=false`, and finite last train loss
  `4.535571336746216`. The launch wrapper, trainer parent, and DataLoader
  workers remain alive. Artifact inventory remains pre-return with only
  `run_metadata.json`, inherited `history_full_msa_to_face.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or new checkpoint exists yet.
- 2026-05-16T16:04Z E145 remains live and pre-eval on owned pod
  `723hbew2jrvxjx`. The status heartbeat advanced to `completed_step=8696`,
  active step `8697`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=4`, `stopped_early=false`, and finite last train loss
  `4.824461460113525`. The launch wrapper, trainer parent, and DataLoader
  workers remain alive. Artifact inventory remains pre-return with only
  `run_metadata.json`, inherited `history_full_msa_to_face.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or new checkpoint exists yet.
- 2026-05-16T16:06Z E145 remains live and pre-eval on owned pod
  `723hbew2jrvxjx`. The status heartbeat advanced to `completed_step=8705`,
  active step `8706`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=4`, `stopped_early=false`, and finite last train loss
  `4.360813558101654`. The launch wrapper, trainer parent, and DataLoader
  workers remain alive. Artifact inventory remains pre-return with only
  `run_metadata.json`, inherited `history_full_msa_to_face.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or new checkpoint exists yet.
- 2026-05-16T16:08Z E145 remains live and pre-eval on owned pod
  `723hbew2jrvxjx`. The status heartbeat advanced to `completed_step=8715`,
  active step `8716`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=4`, `stopped_early=false`, and finite last train loss
  `4.7247925996780396`. The launch wrapper, trainer parent, and DataLoader
  workers remain alive. Artifact inventory remains pre-return with only
  `run_metadata.json`, inherited `history_full_msa_to_face.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or new checkpoint exists yet.
- 2026-05-16T16:10Z E145 remains live and pre-eval on owned pod
  `723hbew2jrvxjx`. The status heartbeat advanced to `completed_step=8725`,
  active step `8726`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=4`, `stopped_early=false`, and finite last train loss
  `4.62440425157547`. The launch wrapper, trainer parent, and DataLoader
  workers remain alive. Artifact inventory remains pre-return with only
  `run_metadata.json`, inherited `history_full_msa_to_face.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or new checkpoint exists yet.
- 2026-05-16T16:12Z E145 remains live and pre-eval on owned pod
  `723hbew2jrvxjx`. The status heartbeat advanced to `completed_step=8735`,
  active step `8736`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=4`, `stopped_early=false`, and finite last train loss
  `4.214575558900833`. The launch wrapper, trainer parent, and DataLoader
  workers remain alive. Artifact inventory remains pre-return with only
  `run_metadata.json`, inherited `history_full_msa_to_face.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or new checkpoint exists yet.
- 2026-05-16T16:13Z E145 remains live and pre-eval on owned pod
  `723hbew2jrvxjx`. The status heartbeat advanced to `completed_step=8744`,
  active step `8745`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=4`, `stopped_early=false`, and finite last train loss
  `4.624148011207581`. The launch wrapper, trainer parent, and DataLoader
  workers remain alive. Artifact inventory remains pre-return with only
  `run_metadata.json`, inherited `history_full_msa_to_face.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or new checkpoint exists yet.
- 2026-05-16T16:15Z E145 remains live and pre-eval on owned pod
  `723hbew2jrvxjx`. The status heartbeat advanced to `completed_step=8753`,
  active step `8754`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=4`, `stopped_early=false`, and finite last train loss
  `4.48009380698204`. The launch wrapper, trainer parent, and DataLoader
  workers remain alive. Artifact inventory remains pre-return with only
  `run_metadata.json`, inherited `history_full_msa_to_face.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or new checkpoint exists yet.
- 2026-05-16T16:19Z E145 remains live and pre-eval on owned pod
  `723hbew2jrvxjx`. The status heartbeat advanced to `completed_step=8775`,
  active step `8776`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=4`, `stopped_early=false`, and finite last train loss
  `4.795517325401306`. The launch wrapper, trainer parent, and DataLoader
  workers remain alive under the E145 command. Artifact inventory remains
  pre-return with only `run_metadata.json`, inherited
  `history_full_msa_to_face.json`, and `status_full_msa_to_face.json`; no
  `results.json`, result CSV, eval-detail CSV, or new checkpoint exists yet.
- 2026-05-16T16:26Z E145 remained live and pre-eval on owned pod
  `723hbew2jrvxjx`. The status heartbeat advanced to `completed_step=8810`,
  active step `8811`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=4`, `stopped_early=false`, and finite last train loss
  `4.816759765148163`. The process tree remained alive, and artifact inventory
  still contained only `run_metadata.json`, inherited
  `history_full_msa_to_face.json`, and `status_full_msa_to_face.json`.
- 2026-05-16T16:31Z E145 remained live and pre-eval on owned pod
  `723hbew2jrvxjx`. The status heartbeat advanced to `completed_step=8837`,
  active step `8838`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=4`, `stopped_early=false`, and finite last train loss
  `4.865641355514526`. The artifact directory still had no result bundle,
  eval-detail CSV, result CSV, or E145 checkpoint.
- 2026-05-16T16:42Z E145 remained live and pre-eval on owned pod
  `723hbew2jrvxjx`. The status heartbeat advanced to `completed_step=8890`,
  active step `8891`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=4`, `stopped_early=false`, and finite last train loss
  `4.470302224159241`. The process tree remained alive and the artifact
  inventory remained pre-return.
- 2026-05-16T16:52Z E145 remained live and pre-eval on owned pod
  `723hbew2jrvxjx`. The status heartbeat advanced to `completed_step=8944`,
  active step `8945`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=4`, `stopped_early=false`, and finite last train loss
  `4.386955678462982`. The artifact directory still contained only
  `run_metadata.json`, inherited `history_full_msa_to_face.json`, and
  `status_full_msa_to_face.json`.
- 2026-05-16T17:07Z E145 reached the final-step watch region on owned pod
  `723hbew2jrvxjx`: status reported `completed_step=8999`, active step
  `9000`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=4`, `stopped_early=false`, and last train loss
  `4.847704619169235`. Status mtime was `2026-05-16T17:03:26Z`, artifact
  inventory still had no `results.json`, result CSV, eval-detail CSV, or E145
  checkpoint, and the process tree remained alive.
- 2026-05-16T17:15Z-17:18Z E145 remained in final-step watch state rather
  than returning a result. Status stayed at `completed_step=8999`, active step
  `9000`, active microbatch `1 / 8`, with status mtime still
  `2026-05-16T17:03:26Z`; artifact inventory was unchanged. The trainer PID
  `347` remained alive and CPU-active, increasing from about `348%` to
  `402%` CPU across samples, while `nvidia-smi` sampled `0%` GPU utilization
  with `29137 / 81920` MiB allocated. Treat this as a watch condition, not a
  terminal no-score outcome yet; leave the owned pod running and do not update
  `EXPERIMENT_RESULTS.md`.
- 2026-05-16T17:21Z-17:35Z E145 still had no scored return and no new
  artifact movement. Status remained fixed at `completed_step=8999`, active
  step `9000`, active microbatch `1 / 8`, with status mtime
  `2026-05-16T17:03:26Z`; artifact inventory still contained only
  `run_metadata.json`, inherited `history_full_msa_to_face.json`, and
  `status_full_msa_to_face.json`. The trainer PID `347` stayed alive and
  increasingly CPU-active (`452%` at 17:21Z, `646%` at 17:35Z), while GPU
  utilization stayed sampled at `0%` with `29137 / 81920` MiB allocated. Local
  code inspection showed the status file is written at `microbatch_start` and
  then again only after batch load, device transfer, model forward, loss, and
  backward complete for that microbatch, so the unchanged `1 / 8` status means
  the run is still inside the first final-step microbatch path, not in final
  eval/checkpoint writing. `py-spy`, `strace`, and `gdb` were not installed on
  the pod, so only low-risk `ps`, `/proc`, and `nvidia-smi` diagnostics were
  used. Continue treating this as a final-step watch condition rather than a
  scored or terminal result; leave the owned pod running for another sample
  window and keep `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T17:52Z E145 remained in the same final-step watch state on
  owned pod `723hbew2jrvxjx`. `runpodctl pod get` still reported the owned pod
  as `RUNNING`. The status file stayed at `completed_step=8999`, active step
  `9000`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=4`, `stopped_early=false`, and last train loss
  `4.847704619169235`, with status mtime still `2026-05-16T17:03Z`.
  Artifact inventory remained only `history_full_msa_to_face.json`,
  `run_metadata.json`, and `status_full_msa_to_face.json`; no `results.json`,
  result CSV, eval-detail CSV, or E145 checkpoint exists yet. The launcher
  PID `345` and trainer PID `347` were still alive, with trainer CPU-active
  at roughly `839%` and `nvidia-smi` sampling `0%` GPU utilization with
  `29137 / 81920` MiB allocated. Leave the owned pod running; this is still
  not a scored result and does not warrant an `EXPERIMENT_RESULTS.md` update.
- 2026-05-16T17:57Z E145 was sampled again before committing the local
  tracker/runtime-prep update. Artifact inventory and status were unchanged:
  still no result bundle, result CSV, eval-detail CSV, or E145 checkpoint;
  status remained at `completed_step=8999`, active step `9000`, active
  microbatch `1 / 8`. The trainer PID `347` remained alive and CPU-active at
  roughly `886%`, with GPU utilization again sampled at `0%` and
  `29137 / 81920` MiB allocated. Continue the watch; do not classify this as
  terminal no-score yet without stronger evidence.
- 2026-05-16T17:59Z-18:01Z E145 was classified as a terminal final-step
  stall/no-score outcome on owned pod `723hbew2jrvxjx`. A one-minute interval
  diagnostic from `17:59:50Z` to `18:00:50Z` showed the status file mtime and
  size frozen at `2026-05-16T17:03:26Z` / `681` bytes, artifact inventory
  still limited to `history_full_msa_to_face.json`, `run_metadata.json`, and
  `status_full_msa_to_face.json`, trainer CPU time advancing from `23:07:17`
  to `23:29:42`, and GPU utilization sampled at `0%` with
  `29137 / 81920` MiB allocated. Preserved the trace locally under ignored
  `artifacts/runpod_traces/e145_stalled_20260516T1800Z/`, including the
  artifact directory, log, process sample, GPU sample, and artifact inventory.
  Stopped only the owned E145 launcher/trainer PIDs `345` and `347`; no
  benchmark process or GPU compute app remained afterward.
- 2026-05-16T18:54Z-18:57Z E146 was launched on the same owned pod
  `723hbew2jrvxjx` from clean checkout `/workspace/SimplexFold_e146` at
  commit `f610b81`. E146 is the same outer-neighborhood selected-cell
  transport gate as E145, but with the exact `cell_outer_edge_context`
  memory-reduction code that reduces outgoing and incoming external-edge
  means before concatenation. Public NanoFold data are symlinked under
  `/workspace/nanoFold-Competition/data`; `find -L` confirmed `11000`
  feature NPZs and `11000` label NPZs, and manifests are `10000 / 1000`
  train/val rows. Remote `py_compile` passed for the model, trainer, and
  benchmark runner modules. Launch wrapper PID is `6334`, trainer PID is
  `6336`, run name is
  `e146_outer_residual_context_exact_from_e128_s9000_c256_m64`, log is
  `/workspace/SimplexFold_e146/logs/e146_outer_residual_context_exact.log`,
  and artifact directory is
  `/workspace/SimplexFold_e146/artifacts/nanofold_public_benchmarks/e146_outer_residual_context_exact_from_e128_s9000_c256_m64`.
  Startup status at `18:57Z` showed resume from E128 step `8500`, `1332`
  matching tensors loaded, `0` new/missing tensors, `completed_step=8512`,
  active step `8513`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=4`, finite last train loss `4.638599187135696`, and GPU
  utilization sampled at `52%` with `22359 / 81920` MiB allocated.
- 2026-05-16T18:58Z E146 heartbeat remained healthy on owned pod
  `723hbew2jrvxjx`: status advanced to `completed_step=8523`, active step
  `8524`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=4`, and finite last train loss `4.705839097499847`. Trainer
  PID `6336` remained alive and CPU-active, and GPU utilization sampled at
  `32%` with `22361 / 81920` MiB allocated. No returned result bundle yet;
  keep `EXPERIMENT_RESULTS.md` unchanged for E146 until a scored or terminal
  outcome exists.
- 2026-05-16T19:01Z E146 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8534`, active step `8535`, active
  microbatch `1 / 8`, `effective_batch_size=8`, `num_workers=4`, and finite
  last train loss `5.0033299922943115`. The status file mtime updated at
  `19:01Z`; artifact inventory still contained only startup metadata,
  inherited history, and live status. Trainer PID `6336` remained alive with
  GPU utilization sampled at `43%` and `22363 / 81920` MiB allocated. Continue
  monitoring only E146; no scored result exists yet.
- 2026-05-16T19:01Z-19:02Z E146 continued to advance on owned pod
  `723hbew2jrvxjx`. A one-minute interval moved from `completed_step=8539`,
  active step `8540`, GPU `83%`, to `completed_step=8544`, active step
  `8545`, GPU `57%`, with memory steady at `22363 / 81920` MiB. History still
  ends at inherited E128 step `8500`, and no `results.json`, result CSV,
  eval-detail CSV, or E146 checkpoint exists yet. This is coherent training
  progress, not a returned result.
- 2026-05-16T19:04Z E146 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8553`, active step `8554`, active
  microbatch `1 / 8`, `effective_batch_size=8`, `num_workers=4`, and finite
  last train loss `4.587962657213211`. The artifact directory still contained
  only `history_full_msa_to_face.json`, `run_metadata.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or E146 checkpoint exists yet. Trainer PID `6336` remained
  alive with GPU utilization sampled at `100%` and `23191 / 81920` MiB
  allocated. Continue monitoring E146 only.
- 2026-05-16T19:04Z follow-up sample after the previous commit confirmed E146
  continued advancing on owned pod `723hbew2jrvxjx`: status reached
  `completed_step=8558`, active step `8559`, active microbatch `1 / 8`,
  `effective_batch_size=8`, `num_workers=4`, and finite last train loss
  `4.505498677492142`. GPU utilization sampled at `36%` with
  `23191 / 81920` MiB allocated. No returned result bundle exists yet, so
  `EXPERIMENT_RESULTS.md` remains unchanged.
- 2026-05-16T19:08Z E146 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8572`, active step `8573`, active
  microbatch `1 / 8`, `effective_batch_size=8`, `num_workers=4`, and finite
  last train loss `4.71696263551712`. Artifact inventory remained pre-return
  with only `history_full_msa_to_face.json`, `run_metadata.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or E146 checkpoint exists yet. Trainer PID `6336` remained
  alive and GPU utilization sampled at `47%` with `23191 / 81920` MiB
  allocated. Continue watching E146; do not update `EXPERIMENT_RESULTS.md`
  for this in-flight state.
- 2026-05-16T19:10Z E146 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8583`, active step `8584`, active
  microbatch `1 / 8`, `effective_batch_size=8`, `num_workers=4`, and finite
  last train loss `4.644069284200668`. Artifact inventory remained pre-return
  with no `results.json`, result CSV, eval-detail CSV, or E146 checkpoint.
  Trainer PID `6336` remained alive and GPU utilization sampled at `56%` with
  `23191 / 81920` MiB allocated. Continue monitoring only E146.
- 2026-05-16T19:12Z E146 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8593`, active step `8594`, active
  microbatch `1 / 8`, `effective_batch_size=8`, `num_workers=4`, and finite
  last train loss `4.511488437652588`. Artifact inventory remained pre-return
  with no `results.json`, result CSV, eval-detail CSV, or E146 checkpoint.
  Trainer PID `6336` remained alive and GPU utilization sampled at `41%` with
  `23191 / 81920` MiB allocated. Continue monitoring only E146.
- 2026-05-16T19:16Z E146 remained coherent on owned pod `723hbew2jrvxjx`.
  `runpodctl pod get` still reported the owned pod as `RUNNING`. Status
  advanced to `completed_step=8615`, active step `8616`, active microbatch
  `1 / 8`, `effective_batch_size=8`, `num_workers=4`, `stopped_early=false`,
  and finite last train loss `4.3651726841926575`. Artifact inventory
  remained pre-return with only `history_full_msa_to_face.json`,
  `run_metadata.json`, and `status_full_msa_to_face.json`; no `results.json`,
  result CSV, eval-detail CSV, or E146 checkpoint exists yet. Trainer PID
  `6336` remained alive and CPU-active, and GPU utilization sampled at `67%`
  with `23191 / 81920` MiB allocated. The existing 30-minute SimplexFold
  heartbeat automation was updated from stale E145 process/artifact paths to
  E146's PID, log, and artifact paths. Continue monitoring only E146 and keep
  `EXPERIMENT_RESULTS.md` unchanged until a scored or terminal outcome exists.
- 2026-05-16T19:20Z E146 continued advancing on owned pod
  `723hbew2jrvxjx`. Status reached `completed_step=8632`, active step `8633`,
  active microbatch `1 / 8`, `effective_batch_size=8`, `num_workers=4`,
  `stopped_early=false`, and finite last train loss `4.637099087238312`.
  Artifact inventory remained pre-return with only inherited history,
  run metadata, and live status; no `results.json`, result CSV, eval-detail
  CSV, or E146 checkpoint exists yet. Trainer PID `6336` remained alive and
  CPU-active, with GPU utilization sampled at `65%` and `23191 / 81920` MiB
  allocated. Continue monitoring only E146; this is still in-flight training,
  not a scored or terminal outcome.
- 2026-05-16T19:21Z E146 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8643`, active step `8644`, active
  microbatch `1 / 8`, `effective_batch_size=8`, `num_workers=4`,
  `stopped_early=false`, and finite last train loss `4.746176332235336`.
  Artifact inventory still contained only `history_full_msa_to_face.json`,
  `run_metadata.json`, and `status_full_msa_to_face.json`; no `results.json`,
  result CSV, eval-detail CSV, or E146 checkpoint exists yet. Trainer PID
  `6336` remained alive and CPU-active, with GPU utilization sampled at `68%`
  and `23191 / 81920` MiB allocated. Continue monitoring only E146; do not
  update `EXPERIMENT_RESULTS.md` until returned artifacts or a terminal
  no-score outcome exists.
- 2026-05-16T19:23Z E146 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8652`, active step `8653`, active
  microbatch `1 / 8`, `effective_batch_size=8`, `num_workers=4`,
  `stopped_early=false`, and finite last train loss `4.717605918645859`.
  Artifact inventory remained pre-return with only `history_full_msa_to_face.json`,
  `run_metadata.json`, and `status_full_msa_to_face.json`; no `results.json`,
  result CSV, eval-detail CSV, or E146 checkpoint exists yet. Trainer PID
  `6336` remained alive and CPU-active, with GPU utilization sampled at `51%`
  and `23191 / 81920` MiB allocated. Continue monitoring only E146.
- 2026-05-16T19:25Z E146 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8662`, active step `8663`, active
  microbatch `1 / 8`, `effective_batch_size=8`, `num_workers=4`,
  `stopped_early=false`, and finite last train loss `4.63549017906189`.
  Artifact inventory remained pre-return with only `history_full_msa_to_face.json`,
  `run_metadata.json`, and `status_full_msa_to_face.json`; no `results.json`,
  result CSV, eval-detail CSV, or E146 checkpoint exists yet. Trainer PID
  `6336` remained alive and CPU-active, with GPU utilization sampled at `61%`
  and `23191 / 81920` MiB allocated. Continue monitoring only E146 and keep
  `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T19:26Z E146 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8669`, active step `8670`, active
  microbatch `1 / 8`, `effective_batch_size=8`, `num_workers=4`,
  `stopped_early=false`, and finite last train loss `4.637627840042114`.
  Artifact inventory remained pre-return with only `history_full_msa_to_face.json`,
  `run_metadata.json`, and `status_full_msa_to_face.json`; no `results.json`,
  result CSV, eval-detail CSV, or E146 checkpoint exists yet. Trainer PID
  `6336` remained alive and CPU-active, with GPU utilization sampled at `60%`
  and `23191 / 81920` MiB allocated. Continue monitoring only E146.
- 2026-05-16T19:28Z E146 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8680`, active step `8681`, active
  microbatch `1 / 8`, `effective_batch_size=8`, `num_workers=4`,
  `stopped_early=false`, and finite last train loss `4.630786418914795`.
  Artifact inventory remained pre-return with only `history_full_msa_to_face.json`,
  `run_metadata.json`, and `status_full_msa_to_face.json`; no `results.json`,
  result CSV, eval-detail CSV, or E146 checkpoint exists yet. Trainer PID
  `6336` remained alive and CPU-active, with GPU utilization sampled at `71%`
  and `23191 / 81920` MiB allocated. Continue monitoring only E146.
- 2026-05-16T19:30Z E146 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8690`, active step `8691`, active
  microbatch `1 / 8`, `effective_batch_size=8`, `num_workers=4`,
  `stopped_early=false`, and finite last train loss `4.488826632499695`.
  Artifact inventory remained pre-return with only `history_full_msa_to_face.json`,
  `run_metadata.json`, and `status_full_msa_to_face.json`; no `results.json`,
  result CSV, eval-detail CSV, or E146 checkpoint exists yet. Trainer PID
  `6336` remained alive and CPU-active, with GPU utilization sampled at `45%`
  and `23191 / 81920` MiB allocated. Continue monitoring only E146.
- 2026-05-16T19:32Z E146 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8698`, active step `8699`, active
  microbatch `1 / 8`, `effective_batch_size=8`, `num_workers=4`,
  `stopped_early=false`, and finite last train loss `4.579521864652634`.
  Artifact inventory remained pre-return with only `history_full_msa_to_face.json`,
  `run_metadata.json`, and `status_full_msa_to_face.json`; no `results.json`,
  result CSV, eval-detail CSV, or E146 checkpoint exists yet. Trainer PID
  `6336` remained alive and CPU-active, with GPU utilization sampled at `42%`
  and `23191 / 81920` MiB allocated. Continue monitoring only E146.
- 2026-05-16T19:33Z E146 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8707`, active step `8708`, active
  microbatch `1 / 8`, `effective_batch_size=8`, `num_workers=4`,
  `stopped_early=false`, and finite last train loss `4.230406492948532`.
  Artifact inventory remained pre-return with only `history_full_msa_to_face.json`,
  `run_metadata.json`, and `status_full_msa_to_face.json`; no `results.json`,
  result CSV, eval-detail CSV, or E146 checkpoint exists yet. Trainer PID
  `6336` remained alive and CPU-active, with GPU utilization sampled at `39%`
  and `23191 / 81920` MiB allocated. Continue monitoring only E146.
- 2026-05-16T19:35Z E146 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8716`, active step `8717`, active
  microbatch `1 / 8`, `effective_batch_size=8`, `num_workers=4`,
  `stopped_early=false`, and finite last train loss `4.781200468540192`.
  Artifact inventory remained pre-return with only `history_full_msa_to_face.json`,
  `run_metadata.json`, and `status_full_msa_to_face.json`; no `results.json`,
  result CSV, eval-detail CSV, or E146 checkpoint exists yet. Trainer PID
  `6336` remained alive and CPU-active, with GPU utilization sampled at `67%`
  and `23191 / 81920` MiB allocated.
- 2026-05-16T19:36Z Post-E146 queue check: do not launch a successor while
  E146 is still coherent. If E146 returns below the `0.45` short gate or
  becomes terminal no-score, the next prepared topology-native short gate is
  E142 signed tetra coboundary, followed by E143 signed tetra-to-face readout
  and then E144 no-Hodge edge-star residual. This queue remains within the
  simplicial/topological motivation: E142/E143 test oriented tetra boundary
  incidence, and E144 tests a residual selected boundary 1-cochain. None adds
  a generic metric-side loss.
- 2026-05-16T19:39Z Current-pod E142 readiness was refreshed without touching
  the active E146 trainer. Initial checks showed `/workspace/SimplexFold_e142`
  and the older `/workspace/SimplexFold/.../e128.../full_msa_to_face_latest.pt`
  path were absent on owned pod `723hbew2jrvxjx`. The E128 checkpoint does
  exist at
  `/workspace/SimplexFold_e145/artifacts/nanofold_public_benchmarks/e128_damped_triangle_bias_from_e124_s8500_c256_m64/checkpoints/full_msa_to_face_latest.pt`.
  Staged `/workspace/SimplexFold_e142` by cloning `/workspace/SimplexFold_e146`
  at commit `f610b81`; remote `py_compile` passed for the model, trainer, and
  benchmark runner files. Updated the E142 launch recipe docs to use the
  current-pod checkpoint path.
- 2026-05-16T19:45Z E146 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8767`, active step `8768`, active
  microbatch `1 / 8`, `effective_batch_size=8`, `num_workers=4`,
  `stopped_early=false`, and finite last train loss `4.68983581662178`.
  Artifact inventory remained pre-return with only `history_full_msa_to_face.json`,
  `run_metadata.json`, and `status_full_msa_to_face.json`; no `results.json`,
  result CSV, eval-detail CSV, or E146 checkpoint exists yet. Trainer PID
  `6336` remained alive and CPU-active, with GPU utilization sampled at `44%`
  and `23191 / 81920` MiB allocated. Continue monitoring only E146 and keep
  `EXPERIMENT_RESULTS.md` unchanged until a scored or terminal outcome exists.
- 2026-05-16T19:51Z E146 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8798`, active step `8799`, active
  microbatch `1 / 8`, `effective_batch_size=8`, `num_workers=4`,
  `stopped_early=false`, and finite last train loss `4.607700049877167`.
  Artifact inventory remained pre-return with only `history_full_msa_to_face.json`,
  `run_metadata.json`, and `status_full_msa_to_face.json`; no `results.json`,
  result CSV, eval-detail CSV, or E146 checkpoint exists yet. Trainer PID
  `6336` remained alive and CPU-active, with GPU utilization sampled at `99%`
  and `23191 / 81920` MiB allocated. Continue monitoring only E146.
- 2026-05-16T19:52Z Added local final-step status instrumentation for
  subsequent runs. The runner now forces status phases through final-step
  batch readiness, forward start/done, backward start/done, evaluation,
  checkpointing, and final history writes. This changes observability only; it
  does not change the model, losses, data, parameter count, or active E146.
  Pull this branch tip into `/workspace/SimplexFold_e142` before launching a
  successor so any repeated final-step stall is attributable to a specific
  operation.
- 2026-05-16T19:54Z The parked `/workspace/SimplexFold_e142` checkout on
  owned pod `723hbew2jrvxjx` was fast-forwarded from GitHub to commit
  `62fda26`, which includes the final-step status instrumentation. Its
  `origin` still points at the local E146 checkout, so `git status` reports
  the branch as ahead of `origin`; the added `github` remote is the source
  used for the fast-forward. Remote `py_compile` passed for
  `scripts/run_nanofold_public_benchmarks.py` and
  `tests/test_nanofold_public_benchmarks.py`.
- 2026-05-16T19:54Z E146 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8811`, active step `8812`, active
  microbatch `1 / 8`, `effective_batch_size=8`, `num_workers=4`,
  `stopped_early=false`, and finite last train loss `4.893148183822632`.
  Artifact inventory remained pre-return with only `history_full_msa_to_face.json`,
  `run_metadata.json`, and `status_full_msa_to_face.json`; no `results.json`,
  result CSV, eval-detail CSV, or E146 checkpoint exists yet. Trainer PID
  `6336` remained alive and CPU-active, with GPU utilization sampled at `71%`
  and `23191 / 81920` MiB allocated. Continue monitoring only E146.
- 2026-05-16T19:57Z E146 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8823`, active step `8824`, active
  microbatch `1 / 8`, `effective_batch_size=8`, `num_workers=4`,
  `stopped_early=false`, and finite last train loss `4.689121246337891`.
  Artifact inventory remained pre-return with only `history_full_msa_to_face.json`,
  `run_metadata.json`, and `status_full_msa_to_face.json`; no `results.json`,
  result CSV, eval-detail CSV, or E146 checkpoint exists yet. Trainer PID
  `6336` remained alive and CPU-active, with GPU utilization sampled at `41%`
  and `23191 / 81920` MiB allocated. Continue monitoring only E146 and keep
  `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T19:59Z E146 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8835`, active step `8836`, active
  microbatch `1 / 8`, `effective_batch_size=8`, `num_workers=4`,
  `stopped_early=false`, and finite last train loss `4.422989010810852`.
  Artifact inventory remained pre-return with only `history_full_msa_to_face.json`,
  `run_metadata.json`, and `status_full_msa_to_face.json`; no `results.json`,
  result CSV, eval-detail CSV, or E146 checkpoint exists yet. Trainer PID
  `6336` remained alive and CPU-active, with GPU utilization sampled at `69%`
  and `23191 / 81920` MiB allocated. Continue monitoring only E146 and keep
  `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T20:02Z E146 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8851`, active step `8852`, active
  microbatch `1 / 8`, `effective_batch_size=8`, `num_workers=4`,
  `stopped_early=false`, and finite last train loss `4.623487234115601`.
  Artifact inventory remained pre-return with only `history_full_msa_to_face.json`,
  `run_metadata.json`, and `status_full_msa_to_face.json`; no `results.json`,
  result CSV, eval-detail CSV, or E146 checkpoint exists yet. Trainer PID
  `6336` remained alive and CPU-active, with GPU utilization sampled at `42%`
  and `23191 / 81920` MiB allocated. Continue monitoring only E146 and keep
  `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T20:05Z E146 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8864`, active step `8865`, active
  microbatch `1 / 8`, `effective_batch_size=8`, `num_workers=4`,
  `stopped_early=false`, and finite last train loss `4.701645314693451`.
  Artifact inventory remained pre-return with only `history_full_msa_to_face.json`,
  `run_metadata.json`, and `status_full_msa_to_face.json`; no `results.json`,
  result CSV, eval-detail CSV, or E146 checkpoint exists yet. Trainer PID
  `6336` remained alive and CPU-active, with GPU utilization sampled at `42%`
  and `23191 / 81920` MiB allocated. Continue monitoring only E146 and keep
  `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T20:07Z E146 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8874`, active step `8875`, active
  microbatch `1 / 8`, `effective_batch_size=8`, `num_workers=4`,
  `stopped_early=false`, and finite last train loss `4.15495491027832`.
  Artifact inventory remained pre-return with only `history_full_msa_to_face.json`,
  `run_metadata.json`, and `status_full_msa_to_face.json`; no `results.json`,
  result CSV, eval-detail CSV, or E146 checkpoint exists yet. Trainer PID
  `6336` remained alive and CPU-active, with GPU utilization sampled at `66%`
  and `23191 / 81920` MiB allocated. Continue monitoring only E146 and keep
  `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T20:09Z E146 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8884`, active step `8885`, active
  microbatch `1 / 8`, `effective_batch_size=8`, `num_workers=4`,
  `stopped_early=false`, and finite last train loss `4.620166629552841`.
  Artifact inventory remained pre-return with only `history_full_msa_to_face.json`,
  `run_metadata.json`, and `status_full_msa_to_face.json`; no `results.json`,
  result CSV, eval-detail CSV, or E146 checkpoint exists yet. Trainer PID
  `6336` remained alive and CPU-active, with GPU utilization sampled at `65%`
  and `23191 / 81920` MiB allocated. Continue monitoring only E146 and keep
  `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T20:38Z E146 was classified as a terminal final-step stall on
  owned Runpod pod `723hbew2jrvxjx`. It had reached `completed_step=8999`,
  active step `9000`, active microbatch `1 / 8`, `effective_batch_size=8`,
  and `num_workers=4`, then stopped updating `status_full_msa_to_face.json`
  at `2026-05-16T20:32Z`. Repeated samples through
  `2026-05-16T20:38:21Z` showed no `results.json`, result CSV,
  eval-detail CSV, checkpoint, or new step-9000 history row; status and
  artifact inventory stayed unchanged, trainer CPU time continued advancing,
  and GPU utilization sampled `0%`. Preserved the trace locally under ignored
  `artifacts/runpod_traces/e146_stalled_20260516T2038Z/`, then stopped only
  the owned E146 process tree. Treat E146 as a runtime stall/no-score, not
  scored evidence against the exact outer-neighborhood architecture.
- 2026-05-16T21:08Z Launched E142 signed tetra coboundary face update on the
  same owned Runpod pod from `/workspace/SimplexFold_e142` at commit
  `62fda26`. Run name:
  `e142_signed_tetra_coboundary_from_e128_s9000_c256_m64`; launch wrapper PID
  `13260`, trainer PID `13262`; artifact directory:
  `/workspace/SimplexFold_e142/artifacts/nanofold_public_benchmarks/e142_signed_tetra_coboundary_from_e128_s9000_c256_m64`.
  E142 resumed the E128 checkpoint at step `8500`, loaded `1332` matching
  model tensors, initialized `0` new/missing tensors, and started a fresh
  optimizer. Latest heartbeat at `2026-05-16T21:09:44Z` showed
  `completed_step=8515`, active step `8516`, active microbatch `1 / 8`,
  `effective_batch_size=8`, `num_workers=0`, `stopped_early=false`, finite
  last train loss `4.59290337562561`, GPU utilization sampled at `40%`, and
  `38235 / 81920` MiB allocated. No returned result bundle exists yet, so
  keep `EXPERIMENT_RESULTS.md` unchanged for E142 until it returns or reaches
  a documented terminal state.
- 2026-05-16T21:14Z E142 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced from the earlier `8524` sample to `completed_step=8533`,
  active step `8534`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=0`, `stopped_early=false`, and finite last train loss
  `4.655693680047989`. Artifact inventory remained pre-return with only
  `history_full_msa_to_face.json`, `run_metadata.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or E142 checkpoint exists yet. Trainer PID `13262`
  remained alive and CPU-active, with GPU memory allocated at
  `38237 / 81920` MiB. The instantaneous GPU-utilization sample was `0%`,
  but step advancement confirms coherent training rather than a stall.
  Continue monitoring only E142 and keep `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T21:16Z E142 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8542`, active step `8543`, active
  microbatch `1 / 8`, `effective_batch_size=8`, `num_workers=0`,
  `stopped_early=false`, and finite last train loss `4.6126130521297455`.
  Artifact inventory remained pre-return with only
  `history_full_msa_to_face.json`, `run_metadata.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or E142 checkpoint exists yet. Trainer PID `13262`
  remained alive and CPU-active, with GPU memory allocated at
  `38237 / 81920` MiB. The instantaneous GPU-utilization sample was again
  `0%`, but status has continued advancing, so this is still in-flight
  training. Continue monitoring only E142 and keep `EXPERIMENT_RESULTS.md`
  unchanged.
- 2026-05-16T21:18Z E142 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8549`, active step `8550`, active
  microbatch `1 / 8`, `effective_batch_size=8`, `num_workers=0`,
  `stopped_early=false`, and finite last train loss `4.851902902126312`.
  Artifact inventory remained pre-return with only
  `history_full_msa_to_face.json`, `run_metadata.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or E142 checkpoint exists yet. Trainer PID `13262`
  remained alive and CPU-active, with GPU utilization sampled at `19%` and
  `43101 / 81920` MiB allocated. Continue monitoring only E142 and keep
  `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T21:20Z E142 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8558`, active step `8559`, active
  microbatch `1 / 8`, `effective_batch_size=8`, `num_workers=0`,
  `stopped_early=false`, and finite last train loss `4.533531099557877`.
  Artifact inventory remained pre-return with only
  `history_full_msa_to_face.json`, `run_metadata.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or E142 checkpoint exists yet. Trainer PID `13262`
  remained alive and CPU-active, with GPU memory allocated at
  `43101 / 81920` MiB. Continue monitoring only E142 and keep
  `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T21:23Z E142 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8566`, active step `8567`, active
  microbatch `1 / 8`, `effective_batch_size=8`, `num_workers=0`,
  `stopped_early=false`, and finite last train loss `4.844849765300751`.
  Artifact inventory remained pre-return with only
  `history_full_msa_to_face.json`, `run_metadata.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or E142 checkpoint exists yet. Trainer PID `13262`
  remained alive and CPU-active, with GPU memory allocated at
  `43101 / 81920` MiB. Continue monitoring only E142 and keep
  `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T21:25Z E142 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8574`, active step `8575`, active
  microbatch `1 / 8`, `effective_batch_size=8`, `num_workers=0`,
  `stopped_early=false`, and finite last train loss `4.62242066860199`.
  Artifact inventory remained pre-return with only
  `history_full_msa_to_face.json`, `run_metadata.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or E142 checkpoint exists yet. Trainer PID `13262`
  remained alive and CPU-active, with GPU utilization sampled at `55%` and
  `43101 / 81920` MiB allocated. Continue monitoring only E142 and keep
  `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T21:27Z E142 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8582`, active step `8583`, active
  microbatch `1 / 8`, `effective_batch_size=8`, `num_workers=0`,
  `stopped_early=false`, and finite last train loss `4.55370169878006`.
  Artifact inventory remained pre-return with only
  `history_full_msa_to_face.json`, `run_metadata.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or E142 checkpoint exists yet. Trainer PID `13262`
  remained alive and CPU-active, with GPU utilization sampled at `62%` and
  `43101 / 81920` MiB allocated. Continue monitoring only E142 and keep
  `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T21:31Z E142 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8597`, active step `8598`, phase
  `microbatch_forward_start`, active microbatch `1 / 8`,
  `effective_batch_size=8`, `num_workers=0`, `stopped_early=false`, and
  finite last train loss `4.855421006679535`. Artifact inventory remained
  pre-return with only `history_full_msa_to_face.json`,
  `run_metadata.json`, and `status_full_msa_to_face.json`; no
  `results.json`, result CSV, eval-detail CSV, or E142 checkpoint exists yet.
  Trainer PID `13262` remained alive and CPU-active. Continue monitoring only
  E142 and keep `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T21:32Z E142 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8604`, active step `8605`, phase
  `microbatch_forward_start`, active microbatch `1 / 8`,
  `effective_batch_size=8`, `num_workers=0`, `stopped_early=false`, and
  finite last train loss `4.8286319971084595`. Artifact inventory remained
  pre-return with only `history_full_msa_to_face.json`,
  `run_metadata.json`, and `status_full_msa_to_face.json`; no
  `results.json`, result CSV, eval-detail CSV, or E142 checkpoint exists yet.
  Trainer PID `13262` remained alive and CPU-active. GPU memory remained
  allocated at `43101 / 81920` MiB; the instantaneous utilization sample was
  `0%`, but status advancement confirms this is still in-flight training
  rather than a stall. Continue monitoring only E142 and keep
  `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T21:35Z E142 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8614`, active step `8615`, phase
  `microbatch_done`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=0`, `stopped_early=false`, and finite last train loss
  `4.385830670595169`. Artifact inventory remained pre-return with only
  `history_full_msa_to_face.json`, `run_metadata.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or E142 checkpoint exists yet. Trainer PID `13262`
  remained alive and CPU-active, with GPU utilization sampled at `30%` and
  `43101 / 81920` MiB allocated. Continue monitoring only E142 and keep
  `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T21:37Z E142 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8622`, active step `8623`, phase
  `microbatch_forward_start`, active microbatch `1 / 8`,
  `effective_batch_size=8`, `num_workers=0`, `stopped_early=false`, and
  finite last train loss `4.7341970801353455`. Artifact inventory remained
  pre-return with only `history_full_msa_to_face.json`,
  `run_metadata.json`, and `status_full_msa_to_face.json`; no
  `results.json`, result CSV, eval-detail CSV, or E142 checkpoint exists yet.
  Trainer PID `13262` remained alive and CPU-active, with GPU memory
  allocated at `43101 / 81920` MiB. The instantaneous GPU-utilization sample
  was `0%`, but status advancement from the prior heartbeat confirms this is
  still in-flight training. Continue monitoring only E142 and keep
  `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T21:39Z E142 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8629`, active step `8630`, phase
  `microbatch_forward_start`, active microbatch `1 / 8`,
  `effective_batch_size=8`, `num_workers=0`, `stopped_early=false`, and
  finite last train loss `4.442022949457169`. Artifact inventory remained
  pre-return with only `history_full_msa_to_face.json`,
  `run_metadata.json`, and `status_full_msa_to_face.json`; no
  `results.json`, result CSV, eval-detail CSV, or E142 checkpoint exists yet.
  Trainer PID `13262` remained alive and CPU-active, with GPU utilization
  sampled at `19%` and `43101 / 81920` MiB allocated. Continue monitoring
  only E142 and keep `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T21:41Z E142 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8636`, active step `8637`, phase
  `microbatch_done`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=0`, `stopped_early=false`, and finite last train loss
  `4.674022167921066`. Artifact inventory remained pre-return with only
  `history_full_msa_to_face.json`, `run_metadata.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or E142 checkpoint exists yet. Trainer PID `13262`
  remained alive and CPU-active, with GPU utilization sampled at `58%` and
  `43101 / 81920` MiB allocated. Continue monitoring only E142 and keep
  `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T21:43Z E142 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8645`, active step `8646`, phase
  `microbatch_done`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=0`, `stopped_early=false`, and finite last train loss
  `4.435737878084183`. Artifact inventory remained pre-return with only
  `history_full_msa_to_face.json`, `run_metadata.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or E142 checkpoint exists yet. Trainer PID `13262`
  remained alive and CPU-active, with GPU utilization sampled at `43%` and
  `43101 / 81920` MiB allocated. Continue monitoring only E142 and keep
  `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T21:45Z E142 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8653`, active step `8654`, phase
  `microbatch_done`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=0`, `stopped_early=false`, and finite last train loss
  `4.380232751369476`. Artifact inventory remained pre-return with only
  `history_full_msa_to_face.json`, `run_metadata.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or E142 checkpoint exists yet. Trainer PID `13262`
  remained alive and CPU-active, with GPU utilization sampled at `31%` and
  `43101 / 81920` MiB allocated. Continue monitoring only E142 and keep
  `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T21:47Z E142 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8661`, active step `8662`, phase
  `microbatch_done`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=0`, `stopped_early=false`, and finite last train loss
  `4.802636921405792`. Artifact inventory remained pre-return with only
  `history_full_msa_to_face.json`, `run_metadata.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or E142 checkpoint exists yet. Trainer PID `13262`
  remained alive and CPU-active, with GPU utilization sampled at `55%` and
  `43101 / 81920` MiB allocated. Continue monitoring only E142 and keep
  `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T21:48Z E142 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8668`, active step `8669`, phase
  `microbatch_forward_start`, active microbatch `1 / 8`,
  `effective_batch_size=8`, `num_workers=0`, `stopped_early=false`, and
  finite last train loss `4.778805673122406`. Artifact inventory remained
  pre-return with only `history_full_msa_to_face.json`,
  `run_metadata.json`, and `status_full_msa_to_face.json`; no
  `results.json`, result CSV, eval-detail CSV, or E142 checkpoint exists yet.
  Trainer PID `13262` remained alive and CPU-active, with GPU utilization
  sampled at `74%` and `43101 / 81920` MiB allocated. Continue monitoring
  only E142 and keep `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T21:51Z E142 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8676`, active step `8677`, phase
  `microbatch_done`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=0`, `stopped_early=false`, and finite last train loss
  `4.334850788116455`. Artifact inventory remained pre-return with only
  `history_full_msa_to_face.json`, `run_metadata.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or E142 checkpoint exists yet. Trainer PID `13262`
  remained alive and CPU-active, with GPU utilization sampled at `60%` and
  `43101 / 81920` MiB allocated. Continue monitoring only E142 and keep
  `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T21:53Z E142 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8686`, active step `8687`, phase
  `microbatch_done`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=0`, `stopped_early=false`, and finite last train loss
  `4.690335512161255`. Artifact inventory remained pre-return with only
  `history_full_msa_to_face.json`, `run_metadata.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or E142 checkpoint exists yet. Trainer PID `13262`
  remained alive and CPU-active, with GPU utilization sampled at `93%` and
  `43101 / 81920` MiB allocated. Continue monitoring only E142 and keep
  `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T21:58Z E142 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8709`, active step `8710`, phase
  `microbatch_done`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=0`, `stopped_early=false`, and finite last train loss
  `4.954083979129791`. Artifact inventory remained pre-return with only
  `history_full_msa_to_face.json`, `run_metadata.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or E142 checkpoint exists yet. Trainer PID `13262`
  remained alive and CPU-active, with GPU utilization sampled at `35%` and
  `43101 / 81920` MiB allocated. Continue monitoring only E142 and keep
  `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T22:01Z E142 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8719`, active step `8720`, phase
  `microbatch_done`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=0`, `stopped_early=false`, and finite last train loss
  `4.720953702926636`. Artifact inventory remained pre-return with only
  `history_full_msa_to_face.json`, `run_metadata.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or E142 checkpoint exists yet. Trainer PID `13262`
  remained alive and CPU-active, with GPU utilization sampled at `46%` and
  `43101 / 81920` MiB allocated. Continue monitoring only E142 and keep
  `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T22:03Z E142 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8729`, active step `8730`, phase
  `microbatch_start`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=0`, `stopped_early=false`, and finite last train loss
  `4.749753475189209`. Artifact inventory remained pre-return with only
  `history_full_msa_to_face.json`, `run_metadata.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or E142 checkpoint exists yet. Trainer PID `13262`
  remained alive and CPU-active, with GPU utilization sampled at `26%` and
  `43101 / 81920` MiB allocated. Continue monitoring only E142 and keep
  `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T22:05Z E142 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8737`, active step `8738`, phase
  `microbatch_done`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=0`, `stopped_early=false`, and finite last train loss
  `4.317967385053635`. Artifact inventory remained pre-return with only
  `history_full_msa_to_face.json`, `run_metadata.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or E142 checkpoint exists yet. Trainer PID `13262`
  remained alive and CPU-active. GPU utilization sampled at `0%` with
  `43101 / 81920` MiB allocated, but the step advance from the prior
  heartbeat confirms in-flight training rather than a no-progress stall.
  Continue monitoring only E142 and keep `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T22:07Z E142 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8746`, active step `8747`, phase
  `microbatch_forward_start`, active microbatch `1 / 8`,
  `effective_batch_size=8`, `num_workers=0`, `stopped_early=false`, and
  finite last train loss `4.792213648557663`. Artifact inventory remained
  pre-return with only `history_full_msa_to_face.json`, `run_metadata.json`,
  and `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or E142 checkpoint exists yet. Trainer PID `13262`
  remained alive and CPU-active, with GPU utilization sampled at `43%` and
  `43101 / 81920` MiB allocated. Continue monitoring only E142 and keep
  `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T22:10Z E142 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8755`, active step `8756`, phase
  `microbatch_done`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=0`, `stopped_early=false`, and finite last train loss
  `4.567559540271759`. Artifact inventory remained pre-return with only
  `history_full_msa_to_face.json`, `run_metadata.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or E142 checkpoint exists yet. Trainer PID `13262`
  remained alive and CPU-active, with GPU utilization sampled at `39%` and
  `43101 / 81920` MiB allocated. Continue monitoring only E142 and keep
  `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T22:12Z E142 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8764`, active step `8765`, phase
  `microbatch_done`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=0`, `stopped_early=false`, and finite last train loss
  `4.5877596735954285`. Artifact inventory remained pre-return with only
  `history_full_msa_to_face.json`, `run_metadata.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or E142 checkpoint exists yet. Trainer PID `13262`
  remained alive and CPU-active. GPU utilization sampled at `0%` with
  `43101 / 81920` MiB allocated, but the step advance from the prior
  heartbeat confirms in-flight training rather than a no-progress stall.
  Continue monitoring only E142 and keep `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T22:14Z E142 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8773`, active step `8774`, phase
  `microbatch_start`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=0`, `stopped_early=false`, and finite last train loss
  `4.65733402967453`. Artifact inventory remained pre-return with only
  `history_full_msa_to_face.json`, `run_metadata.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or E142 checkpoint exists yet. Trainer PID `13262`
  remained alive and CPU-active. GPU utilization sampled at `0%` with
  `43101 / 81920` MiB allocated, but the step advance from the prior
  heartbeat confirms in-flight training rather than a no-progress stall.
  Continue monitoring only E142 and keep `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T22:16Z E142 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8781`, active step `8782`, phase
  `microbatch_done`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=0`, `stopped_early=false`, and finite last train loss
  `4.519021540880203`. Artifact inventory remained pre-return with only
  `history_full_msa_to_face.json`, `run_metadata.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or E142 checkpoint exists yet. Trainer PID `13262`
  remained alive and CPU-active, with GPU utilization sampled at `67%` and
  `43101 / 81920` MiB allocated. Continue monitoring only E142 and keep
  `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T22:18Z E142 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8790`, active step `8791`, phase
  `microbatch_forward_start`, active microbatch `1 / 8`,
  `effective_batch_size=8`, `num_workers=0`, `stopped_early=false`, and
  finite last train loss `4.281107842922211`. Artifact inventory remained
  pre-return with only `history_full_msa_to_face.json`, `run_metadata.json`,
  and `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or E142 checkpoint exists yet. Trainer PID `13262`
  remained alive and CPU-active, with GPU utilization sampled at `62%` and
  `43101 / 81920` MiB allocated. Continue monitoring only E142 and keep
  `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T22:24Z E142 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8811`, active step `8812`, phase
  `microbatch_forward_start`, active microbatch `1 / 8`,
  `effective_batch_size=8`, `num_workers=0`, `stopped_early=false`, and
  finite last train loss `4.881737112998962`. Artifact inventory remained
  pre-return with only `history_full_msa_to_face.json`, `run_metadata.json`,
  and `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or E142 checkpoint exists yet. Trainer PID `13262`
  remained alive and CPU-active, with GPU utilization sampled at `47%` and
  `43101 / 81920` MiB allocated. Continue monitoring only E142 and keep
  `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T22:26Z E142 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8821`, active step `8822`, phase
  `microbatch_done`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=0`, `stopped_early=false`, and finite last train loss
  `4.2976531982421875`. Artifact inventory remained pre-return with only
  `history_full_msa_to_face.json`, `run_metadata.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or E142 checkpoint exists yet. Trainer PID `13262`
  remained alive and CPU-active, with GPU utilization sampled at `45%` and
  `43101 / 81920` MiB allocated. Continue monitoring only E142 and keep
  `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T22:28Z E142 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8830`, active step `8831`, phase
  `microbatch_done`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=0`, `stopped_early=false`, and finite last train loss
  `4.699853092432022`. Artifact inventory remained pre-return with only
  `history_full_msa_to_face.json`, `run_metadata.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or E142 checkpoint exists yet. Trainer PID `13262`
  remained alive and CPU-active. GPU utilization sampled at `0%` with
  `43101 / 81920` MiB allocated, but the step advance from the prior
  heartbeat confirms in-flight training rather than a no-progress stall.
  Continue monitoring only E142 and keep `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T22:30Z E142 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8838`, active step `8839`, phase
  `microbatch_done`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=0`, `stopped_early=false`, and finite last train loss
  `4.1288584768772125`. Artifact inventory remained pre-return with only
  `history_full_msa_to_face.json`, `run_metadata.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or E142 checkpoint exists yet. Trainer PID `13262`
  remained alive and CPU-active, with GPU utilization sampled at `61%` and
  `43101 / 81920` MiB allocated. Continue monitoring only E142 and keep
  `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T22:32Z E142 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8849`, active step `8850`, phase
  `microbatch_done`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=0`, `stopped_early=false`, and finite last train loss
  `4.840060532093048`. Artifact inventory remained pre-return with only
  `history_full_msa_to_face.json`, `run_metadata.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or E142 checkpoint exists yet. Trainer PID `13262`
  remained alive and CPU-active, with GPU utilization sampled at `14%` and
  `43101 / 81920` MiB allocated. Continue monitoring only E142 and keep
  `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T22:35Z E142 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8858`, active step `8859`, phase
  `microbatch_done`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=0`, `stopped_early=false`, and finite last train loss
  `4.616480529308319`. Artifact inventory remained pre-return with only
  `history_full_msa_to_face.json`, `run_metadata.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or E142 checkpoint exists yet. Trainer PID `13262`
  remained alive and CPU-active, with GPU utilization sampled at `53%` and
  `43101 / 81920` MiB allocated. Continue monitoring only E142 and keep
  `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T22:37Z E142 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8868`, active step `8869`, phase
  `microbatch_done`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=0`, `stopped_early=false`, and finite last train loss
  `5.023931503295898`. Artifact inventory remained pre-return with only
  `history_full_msa_to_face.json`, `run_metadata.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or E142 checkpoint exists yet. Trainer PID `13262`
  remained alive and CPU-active. GPU utilization sampled at `9%` with
  `43101 / 81920` MiB allocated, but the step advance from the prior
  heartbeat confirms in-flight training rather than a no-progress stall.
  Continue monitoring only E142 and keep `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T22:40Z E142 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8880`, active step `8881`, phase
  `microbatch_done`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=0`, `stopped_early=false`, and finite last train loss
  `4.564347505569458`. Artifact inventory remained pre-return with only
  `history_full_msa_to_face.json`, `run_metadata.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or E142 checkpoint exists yet. Trainer PID `13262`
  remained alive and CPU-active, with GPU utilization sampled at `42%` and
  `43101 / 81920` MiB allocated. Continue monitoring only E142 and keep
  `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T22:42Z E142 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8890`, active step `8891`, phase
  `microbatch_done`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=0`, `stopped_early=false`, and finite last train loss
  `4.503138780593872`. Artifact inventory remained pre-return with only
  `history_full_msa_to_face.json`, `run_metadata.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or E142 checkpoint exists yet. Trainer PID `13262`
  remained alive and CPU-active, with GPU utilization sampled at `51%` and
  `43101 / 81920` MiB allocated. Continue monitoring only E142 and keep
  `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T22:48Z E142 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8911`, active step `8912`, phase
  `microbatch_done`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=0`, `stopped_early=false`, and finite last train loss
  `4.870556473731995`. Artifact inventory remained pre-return with only
  `history_full_msa_to_face.json`, `run_metadata.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or E142 checkpoint exists yet. Trainer PID `13262`
  remained alive and CPU-active, with GPU utilization sampled at `19%` and
  `43101 / 81920` MiB allocated. Continue monitoring only E142 and keep
  `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T22:51Z E142 remained coherent on owned pod `723hbew2jrvxjx`.
  Status advanced to `completed_step=8926`, active step `8927`, phase
  `microbatch_done`, active microbatch `1 / 8`, `effective_batch_size=8`,
  `num_workers=0`, `stopped_early=false`, and finite last train loss
  `4.475143253803253`. Artifact inventory remained pre-return with only
  `history_full_msa_to_face.json`, `run_metadata.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or E142 checkpoint exists yet. Trainer PID `13262`
  remained alive and CPU-active, with GPU utilization sampled at `47%` and
  `43101 / 81920` MiB allocated. Continue monitoring only E142 and keep
  `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-16T23:24Z E142 reached the final validation watch state on owned
  pod `723hbew2jrvxjx`. Status advanced through the final microbatches to
  `completed_step=9000`, active step `9000`, phase `evaluating`, effective
  batch size `8`, `num_workers=0`, `stopped_early=false`, and finite last
  train loss `4.721518278121948`. The latest status mtime was
  `2026-05-16T23:09:53Z`; artifact inventory remained pre-return with only
  `history_full_msa_to_face.json`, `run_metadata.json`, and
  `status_full_msa_to_face.json`. There is still no `results.json`, result
  CSV, eval-detail CSV, or E142 checkpoint. Trainer PID `13262` remained
  alive and strongly CPU-active, GPU utilization sampled at `0%`, and GPU
  memory remained allocated at `43101 / 81920` MiB. Treat this as
  final-validation/eval-watch, not as a scored result or terminal no-score
  outcome yet; continue monitoring only E142 and keep `EXPERIMENT_RESULTS.md`
  unchanged.
- 2026-05-16T23:44Z E142 remained in final validation watch on owned pod
  `723hbew2jrvxjx`. Status stayed at `completed_step=9000`, active step
  `9000`, phase `evaluating`, with status mtime still
  `2026-05-16T23:09:53Z`; artifact inventory was unchanged and still lacked
  `results.json`, result CSV, eval-detail CSV, and checkpoint files. Trainer
  PID `13262` remained alive and strongly CPU-active, with elapsed time
  `02:39:29` and accumulated process CPU time `1-10:06:22`; GPU utilization
  sampled at `0%` with `43101 / 81920` MiB allocated. `py-spy`, `gdb`,
  `pstack`, and `strace` were not available on the pod for low-risk stack
  inspection. Continue treating this as CPU-active eval-watch rather than a
  scored result or terminal no-score outcome; keep `EXPERIMENT_RESULTS.md`
  unchanged.
- 2026-05-17T00:22Z E142 still has no returned result bundle, but a fresh
  one-minute interval argues against a terminal stop. Status remained
  `completed_step=9000`, active step `9000`, phase `evaluating`, with status
  mtime `2026-05-16T23:09:53Z`; artifact inventory remained only
  `status_full_msa_to_face.json`, `history_full_msa_to_face.json`, and
  `run_metadata.json`, with no `results.json`, result CSV, eval-detail CSV,
  or checkpoint. Trainer PID `13262` stayed runnable, elapsed time advanced
  from `03:16:30` to `03:17:31`, CPU time advanced from `2-00:36:57` to
  `2-01:01:20`, `rchar` increased from `665906345` to `666515962`,
  `read_bytes` increased from `5853184` to `6291456`, and GPU utilization
  rebounded from `0%` to `45%` with `43101 / 81920` MiB allocated. Keep E142
  alive as slow final evaluation, do not update `EXPERIMENT_RESULTS.md`, and
  do not launch E143 while this liveness evidence persists.
- 2026-05-17T00:31Z E142 remained a live CPU-bound final-evaluation watch
  after a five-minute interval on owned pod `723hbew2jrvxjx`. Artifact
  inventory was unchanged from `00:25:54Z` through `00:30:56Z`, still only
  `run_metadata.json`, `history_full_msa_to_face.json`, and
  `status_full_msa_to_face.json`; status mtime stayed
  `2026-05-16T23:09:53Z`, with no `results.json`, result CSV, eval-detail
  CSV, or checkpoint. Trainer PID `13262` stayed runnable with `194` threads,
  elapsed time advanced from `03:20:38` to `03:25:40`, process CPU time
  advanced from `2-02:15:11` to `2-04:14:45`, `%CPU` rose from `1502` to
  `1524`, RSS stayed in the `1.7GB` band, `rchar` increased from
  `668927868` to `672125469`, and `read_bytes` increased from `8159232` to
  `10559488`. GPU utilization sampled `0%` throughout with
  `43101 / 81920` MiB allocated. The active E142 checkout writes
  `eval_details_full_msa_to_face.csv` only after `_evaluate` finishes all
  validation examples, and it lacks the newer `active_eval_batch` status
  counters. Keep `EXPERIMENT_RESULTS.md` unchanged and do not launch E143
  while this CPU-active liveness pattern persists.
- 2026-05-17T00:36Z E142 still had no returned bundle, but the process
  remained CPU-active. A two-minute interval from `00:33:59Z` to `00:35:59Z`
  showed unchanged artifact inventory and status mtime, trainer PID `13262`
  runnable with `194` threads, process CPU time advancing from `2-05:26:42`
  to `2-06:14:08`, `rchar` increasing from `674351075` to `675899001`, and
  `read_bytes` increasing from `12296192` to `13533184`. GPU utilization
  sampled `0%` at both ends with `43101 / 81920` MiB allocated. Continue
  treating E142 as live CPU-bound final evaluation; keep
  `EXPERIMENT_RESULTS.md` unchanged and do not launch E143 yet.
- 2026-05-17T00:40Z E142 remained live but still unreturned. The log tail was
  unchanged after startup/resume messages. A two-minute interval from
  `00:38:16Z` to `00:40:16Z` again showed unchanged artifact inventory and
  status mtime, trainer PID `13262` runnable with `194` threads, process CPU
  time advancing from `2-07:08:09` to `2-07:55:43`, `rchar` increasing from
  `677613445` to `678778624`, and `read_bytes` increasing from `14843904` to
  `15691776`. GPU utilization sampled `0%` at both ends with
  `43101 / 81920` MiB allocated. Continue treating E142 as live CPU-bound
  final evaluation; keep `EXPERIMENT_RESULTS.md` unchanged and do not launch
  E143 yet.
- 2026-05-17T00:45Z E142 remained live and unreturned. A two-minute interval
  from `00:42:39Z` to `00:44:39Z` showed unchanged artifact inventory and
  status mtime, trainer PID `13262` runnable with `194` threads, process CPU
  time advancing from `2-08:52:08` to `2-09:40:45`, `rchar` increasing from
  `680757709` to `681768617`, and `read_bytes` increasing from `17276928` to
  `17915904`. GPU utilization sampled `0%` at both ends with
  `43101 / 81920` MiB allocated. Continue treating E142 as live CPU-bound
  final evaluation; keep `EXPERIMENT_RESULTS.md` unchanged and do not launch
  E143 yet.
- 2026-05-17T00:49Z E142 remained live and unreturned on owned pod
  `723hbew2jrvxjx`. A two-minute interval from `00:47:05Z` to `00:49:05Z`
  showed unchanged artifact inventory and status mtime, trainer PID `13262`
  runnable with `194` threads, process CPU time advancing from `2-10:37:25`
  to `2-11:25:56`, `rchar` increasing from `683398826` to `684623116`, and
  `read_bytes` increasing from `19181568` to `20189184`. GPU utilization
  sampled `74%` at the start and `0%` at the end, with `43101 / 81920` MiB
  allocated. Continue treating E142 as live final evaluation; keep
  `EXPERIMENT_RESULTS.md` unchanged and do not launch E143 yet.
- 2026-05-17T00:53Z E142 remained live and unreturned. Artifact inventory
  still contained only `run_metadata.json`, `history_full_msa_to_face.json`,
  and `status_full_msa_to_face.json`. A two-minute interval from `00:51:19Z`
  to `00:53:19Z` showed trainer PID `13262` runnable with `194` threads,
  process CPU time advancing from `2-12:18:32` to `2-13:06:00`, `rchar`
  increasing from `686498648` to `687954460`, and `read_bytes` increasing
  from `21590016` to `22769664`. GPU utilization sampled `0%` at both ends
  with `43101 / 81920` MiB allocated. Continue treating E142 as live
  CPU-bound final evaluation; keep `EXPERIMENT_RESULTS.md` unchanged and do
  not launch E143 yet.
- 2026-05-17T00:59Z E142 remained live and unreturned. A two-minute interval
  from `00:56:42Z` to `00:58:42Z` showed unchanged artifact inventory and
  status mtime, trainer PID `13262` runnable with `194` threads, process CPU
  time advancing from `2-14:26:23` to `2-15:14:45`, `rchar` increasing from
  `689784920` to `691436288`, and `read_bytes` increasing from `24068096` to
  `25382912`. GPU utilization sampled `0%` at both ends with
  `43101 / 81920` MiB allocated. Continue treating E142 as live CPU-bound
  final evaluation; keep `EXPERIMENT_RESULTS.md` unchanged and do not launch
  E143 yet.
- 2026-05-17T01:05Z E142 remained live and unreturned on owned pod
  `723hbew2jrvxjx`. A two-minute interval from `01:01:52Z` to `01:03:52Z`
  showed unchanged returned artifacts, trainer PID `13262` runnable with
  `194` threads, process CPU time advancing from `2-16:30:35` to
  `2-17:18:20`, `rchar` increasing from `692847647` to `694284520`, and
  `read_bytes` increasing from `26337280` to `27439104`. A fresh `01:04:50Z`
  check still showed status `phase=evaluating`, completed step `9000`, no
  `results.json`, no `eval_details_full_msa_to_face.csv`, and no checkpoint
  directory; the trainer was still runnable with CPU time `2-17:41:28`, and
  GPU utilization sampled `76%` with `43101 / 81920` MiB allocated. Continue
  treating E142 as live final evaluation; keep `EXPERIMENT_RESULTS.md`
  unchanged and do not launch E143 yet.
- 2026-05-17T01:11Z E142 still had no returned bundle. The owned Runpod pod
  metadata matched `723hbew2jrvxjx` / `codex-simplexfold-e145-runpod-20260516`
  with desired status `RUNNING`; no other pods were queried or managed. A
  direct `01:10:40Z` artifact check showed only `run_metadata.json`,
  `history_full_msa_to_face.json`, and `status_full_msa_to_face.json`; there
  was still no `results.json`, no `eval_details_full_msa_to_face.csv`, and no
  checkpoint directory. Status remained `phase=evaluating`, completed step
  `9000`, active step `9000`, and last history step `8500`. A two-minute
  interval from `01:08:15Z` to `01:10:15Z` showed trainer PID `13262`
  runnable with `194` threads, process CPU time advancing from `2-19:00:05`
  to `2-19:47:25`, `rchar` increasing from `697308085` to `698802022`, and
  `read_bytes` increasing from `29544448` to `30666752`. GPU utilization
  sampled `0%` at both ends with `43101 / 81920` MiB allocated. Continue
  treating this as live final evaluation because CPU and read counters are
  still moving; keep `EXPERIMENT_RESULTS.md` unchanged and leave E143 parked.
- 2026-05-17T01:19Z E142 remained live and unreturned after a five-minute
  owned-pod interval. Artifact inventory stayed unchanged with only
  `run_metadata.json`, `history_full_msa_to_face.json`, and
  `status_full_msa_to_face.json`; there was still no `results.json`,
  eval-details CSV, result CSV, or checkpoint directory. Trainer PID `13262`
  stayed runnable with `194` threads, elapsed time advanced from `04:08:46`
  to `04:13:46`, process CPU time advanced from `2-21:17:13` to
  `2-23:15:06`, `rchar` increased from `701424405` to `704710751`, and
  `read_bytes` increased from `32661504` to `35041280`. GPU utilization
  sampled `0%` at both ends with `43101 / 81920` MiB allocated. The local
  runner confirms eval details are written only after `_evaluate` finishes, so
  unchanged artifacts plus strong CPU/read movement is still live-eval
  evidence, not a terminal no-score trigger. Keep `EXPERIMENT_RESULTS.md`
  unchanged and do not launch E143 yet.
- 2026-05-17T01:27Z E142 remained live and unreturned after another
  five-minute owned-pod interval. The artifact directory still contained only
  `run_metadata.json`, `history_full_msa_to_face.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV, eval-details
  CSV, or checkpoint directory existed. Trainer PID `13262` stayed runnable
  with `194` threads, elapsed time advanced from `04:16:49` to `04:21:49`,
  process CPU time advanced from `3-00:27:33` to `3-02:26:00`, `rchar`
  increased from `707304171` to `710850664`, and `read_bytes` increased from
  `37109760` to `39784448`. GPU utilization moved from `0%` to `18%` with
  `43101 / 81920` MiB allocated. Keep classifying E142 as live final
  evaluation; leave `EXPERIMENT_RESULTS.md` unchanged and keep E143 parked.
- 2026-05-17T01:33Z E142 remained live and unreturned. A direct owned-pod
  check at `01:29:19Z` still showed only `run_metadata.json`,
  `history_full_msa_to_face.json`, and `status_full_msa_to_face.json`, with
  status still `phase=evaluating`, completed step `9000`, active step `9000`,
  and last history step `8500`. A three-minute interval from `01:30:00Z` to
  `01:33:00Z` showed trainer PID `13262` runnable with `194` threads, process
  CPU time advancing from `3-03:34:49` to `3-04:45:33`, `rchar` increasing
  from `712761930` to `714913106`, and `read_bytes` increasing from
  `41177088` to `42741760`. GPU utilization sampled `73%` at the start and
  `0%` at the end with `43101 / 81920` MiB allocated. A thread snapshot showed
  the main Python thread runnable and many Python worker/library threads
  active in `futex_wait_queue`; the log tail remained only startup/resume
  lines. Continue treating E142 as live slow final evaluation; do not update
  `EXPERIMENT_RESULTS.md` or launch E143 yet.
- 2026-05-17T01:39Z E142 remained live and unreturned. A direct owned-pod
  check at `01:35:34Z` still showed only the three live files and status
  `phase=evaluating`, completed step `9000`, active step `9000`, last history
  step `8500`. Over the three-minute interval ending `01:38:58Z`, trainer PID
  `13262` stayed runnable with `194` threads, elapsed time advanced from
  `04:30:42` to `04:33:42`, process CPU time advanced from `3-05:54:28` to
  `3-07:05:56`, `rchar` increased from `717851764` to `720111462`, and
  `read_bytes` increased from `45080576` to `46891008`. GPU utilization
  sampled `57%` then `0%` with `43101 / 81920` MiB allocated. Continue
  treating this as live slow final evaluation; leave `EXPERIMENT_RESULTS.md`
  unchanged and keep E143 parked.
- 2026-05-17T01:45Z E142 remained live and unreturned on owned pod
  `723hbew2jrvxjx`. A direct `01:41:48Z` check still showed the same three
  live files and status `phase=evaluating`; no result bundle or checkpoint
  directory existed. Over the three-minute interval ending `01:45:15Z`,
  trainer PID `13262` stayed runnable with `194` threads, elapsed time
  advanced from `04:36:59` to `04:39:59`, process CPU time advanced from
  `3-08:23:11` to `3-09:33:56`, `rchar` increased from `722551744` to
  `724367292`, and `read_bytes` increased from `48775168` to `50077696`. GPU
  utilization sampled `0%` at both ends with `43101 / 81920` MiB allocated.
  Keep E142 classified as live slow final evaluation; do not update
  `EXPERIMENT_RESULTS.md` or launch E143 yet.
- 2026-05-17T02:16Z E142 remained live and unreturned on owned pod
  `723hbew2jrvxjx`. The artifact directory still contained only
  `run_metadata.json`, `history_full_msa_to_face.json`, and
  `status_full_msa_to_face.json`; there was still no `results.json`, result
  CSV, eval-detail CSV, or checkpoint directory. Status stayed
  `phase=evaluating`, completed step `9000`, active step `9000`, last history
  step `8500`, `effective_batch_size=8`, `num_workers=0`, and
  `stopped_early=false`. Over the one-minute interval from `02:15:29Z` to
  `02:16:29Z`, trainer PID `13262` remained runnable with `194` threads,
  process CPU time advanced from `3-21:27:19` to `3-21:52:04`, `rchar`
  increased from `744629933` to `745133394`, `read_bytes` increased from
  `64999424` to `65376256`, and RSS increased from `1808596` to `1951028`
  KiB. GPU utilization sampled `0%` at both ends with `43101 / 81920` MiB
  allocated. Keep E142 classified as live slow final evaluation; leave
  `EXPERIMENT_RESULTS.md` unchanged and keep E143 parked until E142 returns or
  reaches a documented terminal no-score state.
- 2026-05-17T02:20Z E142 still had no returned bundle, but remained live on
  owned pod `723hbew2jrvxjx`. The required scored-return files were still
  absent (`results.json`, `results.csv`, `eval_details_full_msa_to_face.csv`,
  and `checkpoints/full_msa_to_face_latest.pt`), and status stayed
  `phase=evaluating`, completed step `9000`, active step `9000`, last history
  step `8500`. Over the one-minute interval from `02:19:41Z` to `02:20:41Z`,
  trainer PID `13262` remained runnable with `194` threads, process CPU time
  advanced from `3-23:08:41` to `3-23:32:04`, `rchar` increased from
  `747677563` to `748631881`, `read_bytes` increased from `67428352` to
  `68206592`, and RSS increased from `1777536` to `1804240` KiB. GPU
  utilization sampled `38%` then `0%` with `43101 / 81920` MiB allocated.
  Keep E142 in eval-watch, leave `EXPERIMENT_RESULTS.md` unchanged, and do
  not launch E143 while these liveness signals continue.
- 2026-05-17T02:24Z E142 remained live and unreturned on owned pod
  `723hbew2jrvxjx`. Status was still `phase=evaluating`, completed step
  `9000`, active step `9000`, last history step `8500`,
  `effective_batch_size=8`, `num_workers=0`, and `stopped_early=false`; the
  scored-return files (`results.json`, `results.csv`,
  `eval_details_full_msa_to_face.csv`, and
  `checkpoints/full_msa_to_face_latest.pt`) were still absent. Over the
  one-minute interval from `02:22:58Z` to `02:23:58Z`, trainer PID `13262`
  stayed runnable with `194` threads, process CPU time advanced from
  `4-00:26:43` to `4-00:50:19`, `rchar` increased from `750499254` to
  `750943816`, and `read_bytes` increased from `69652480` to `69959680`. GPU
  utilization sampled `0%` at both ends with `43101 / 81920` MiB allocated.
  This remains a live slow final-evaluation state; keep
  `EXPERIMENT_RESULTS.md` unchanged and leave E143 parked.
- 2026-05-17T02:27Z E142 remained live and unreturned on owned pod
  `723hbew2jrvxjx`. Status stayed `phase=evaluating`, completed step `9000`,
  active step `9000`, last history step `8500`, `effective_batch_size=8`,
  `num_workers=0`, and `stopped_early=false`; `results.json`, `results.csv`,
  `eval_details_full_msa_to_face.csv`, and
  `checkpoints/full_msa_to_face_latest.pt` were still absent. Over the
  one-minute interval from `02:26:26Z` to `02:27:26Z`, trainer PID `13262`
  stayed runnable with `194` threads, process CPU time advanced from
  `4-01:49:45` to `4-02:12:49`, `rchar` increased from `752722453` to
  `753406526`, `read_bytes` increased from `71344128` to `71880704`, and RSS
  increased from `1798888` to `1885280` KiB. GPU utilization sampled `68%`
  then `0%` with `43101 / 81920` MiB allocated. Continue treating E142 as a
  live slow final evaluation; do not update `EXPERIMENT_RESULTS.md` or launch
  E143 while these liveness signals persist.
- 2026-05-17T02:31Z E142 returned coherently on owned pod `723hbew2jrvxjx`.
  During the `02:29:59Z` to `02:31:00Z` interval, status switched from
  `phase=evaluating` to `phase=finished`, trainer PID `13262` exited, GPU
  memory dropped from `43101 / 81920` MiB to `1 / 81920` MiB, and the scored
  bundle appeared. Remote and local artifact checks confirmed `results.json`,
  `results.csv`, `history_full_msa_to_face.json`,
  `eval_details_full_msa_to_face.csv`, `run_metadata.json`,
  `status_full_msa_to_face.json`, and
  `checkpoints/full_msa_to_face_latest.pt`.
- 2026-05-17T02:31Z E142 verification passed after pulling artifacts/logs
  locally: completed step `9000`, history last step `9000`,
  `effective_batch_size=8`, `num_workers=0`, `stopped_early=false`, `1000`
  eval-detail rows, and `3,240,738` parameters under the `3,261,974` cap.
  Goal audit failed as expected for a short gate below the full goal:
  `val_lddt_ca=0.4210`, `9000 < 30000` confirmation steps, target `0.7` not
  reached. Eval-detail analysis showed mean FoldScore `0.4015`, dRMSD
  `10.7533`, predicted/true C-alpha Rg `12.1970 / 16.3091`, boundary mean
  lDDT `0.7462`, and a high-boundary/low-global subset of `126 / 490`
  high-boundary rows. Decision: reject E142 as a continuation or 30k candidate
  because it fell below E128's `0.4311` primary lDDT and below the `0.45`
  short-gate threshold; launch the parked E143 signed tetra-to-face readout
  after result/docs are committed and no active SimplexFold training remains.
- 2026-05-17T02:34Z Launched E143 signed tetra-to-face boundary readout on
  owned Runpod pod `723hbew2jrvxjx` after committing the verified E142 result.
  Checkout: `/workspace/SimplexFold_e143`; run name:
  `e143_signed_tetra_to_face_from_e128_s9000_c256_m64`; trainer PID `95692`;
  log: `/workspace/SimplexFold_e143/logs/e143_signed_tetra_to_face.log`;
  artifact directory:
  `/workspace/SimplexFold_e143/artifacts/nanofold_public_benchmarks/e143_signed_tetra_to_face_from_e128_s9000_c256_m64`.
  The launch resumed the E128 checkpoint at step `8500`, loaded `1332`
  matching tensors, initialized `0` new/missing tensors, and started a fresh
  optimizer. A post-launch status check at `02:34:13Z` showed
  `completed_step=8501`, active step `8502`, active microbatch `1 / 8`,
  `effective_batch_size=8`, `num_workers=0`, `stopped_early=false`, finite
  last train loss `4.276291459798813`, trainer PID `95692` runnable with
  `194` threads, and GPU utilization `60%` with `18089 / 81920` MiB
  allocated. Treat E143 as the only active SimplexFold run; keep
  `EXPERIMENT_RESULTS.md` unchanged until a scored bundle or terminal no-score
  state exists.
- 2026-05-17T02:38Z E143 remained coherent on owned pod `723hbew2jrvxjx`.
  A direct check still showed only `run_metadata.json`,
  `history_full_msa_to_face.json`, and `status_full_msa_to_face.json`; there
  was no `results.json`, result CSV, eval-detail CSV, or checkpoint yet. Over
  the one-minute interval from `02:36:36Z` to `02:37:36Z`, status advanced
  from completed step `8511`, active step `8512`, microbatch `1 / 8`, to
  completed step `8515`, active step `8516`, microbatch `1 / 8`, with
  `effective_batch_size=8`, `num_workers=0`, `stopped_early=false`, and
  finite train losses. Trainer PID `95692` stayed runnable with `194` threads,
  process CPU time advanced from `00:27:43` to `00:37:31`, `rchar` increased
  from `95551699` to `100018226`, and GPU memory rose from
  `28929 / 81920` to `38445 / 81920` MiB. Keep E143 running and keep
  `EXPERIMENT_RESULTS.md` unchanged until a scored bundle or documented
  terminal no-score state exists.
- 2026-05-17T02:39Z E143 remained coherent on owned pod `723hbew2jrvxjx`.
  Status had advanced to completed step `8523`, active step `8524`,
  microbatch `1 / 8`, with `effective_batch_size=8`, `num_workers=0`,
  `stopped_early=false`, and finite last train loss `4.731101334095001`.
  Artifact inventory remained pre-return with only `run_metadata.json`,
  `history_full_msa_to_face.json`, and `status_full_msa_to_face.json`; there
  was still no `results.json`, result CSV, eval-detail CSV, or checkpoint.
  Trainer PID `95692` stayed runnable with `194` threads, elapsed time
  `00:05:56`, process CPU time `00:55:18`, and GPU memory
  `38445 / 81920` MiB allocated. Keep E143 running and keep
  `EXPERIMENT_RESULTS.md` unchanged until a scored bundle or documented
  terminal no-score state exists.
- 2026-05-17T02:42Z E143 remained coherent on owned pod `723hbew2jrvxjx`.
  Status had advanced to completed step `8533`, active step `8534`,
  microbatch `1 / 8`, with `effective_batch_size=8`, `num_workers=0`,
  `stopped_early=false`, and finite last train loss `4.656118214130402`.
  Artifact inventory remained pre-return with only `run_metadata.json`,
  `history_full_msa_to_face.json`, and `status_full_msa_to_face.json`; no
  `results.json`, result CSV, eval-detail CSV, or checkpoint existed yet.
  Trainer PID `95692` stayed runnable with `194` threads, elapsed time
  `00:08:32`, process CPU time `01:20:45`, and GPU memory
  `38447 / 81920` MiB allocated. Keep E143 running and keep
  `EXPERIMENT_RESULTS.md` unchanged until a scored bundle or documented
  terminal no-score state exists.
- 2026-05-17T02:46Z E143 was still coherent and in-flight on the same owned
  pod. Status had advanced to completed step `8548`, active step `8549`,
  microbatch `1 / 8`, with `effective_batch_size=8`, `num_workers=0`,
  `stopped_early=false`, and finite last train loss `4.672388464212418`.
  Artifact inventory was still pre-return: `run_metadata.json`,
  `history_full_msa_to_face.json`, and `status_full_msa_to_face.json` exist,
  but there is no `results.json`, result CSV, eval-detail CSV, or checkpoint.
  Trainer PID `95692` remained runnable with `194` threads, elapsed time
  `00:12:25`, process CPU time `02:00:29`, GPU utilization sampled `0%`, and
  GPU memory `43311 / 81920` MiB allocated. Keep E143 running; the current
  results audit still reports E128 as best at `val_lddt_ca=0.4311`, zero
  short gates at or above `0.45`, and zero `30000`-step confirmations above
  `0.7`.
- 2026-05-17T02:49Z E143 continued to advance. A one-minute read-only interval
  on owned pod `723hbew2jrvxjx` moved from completed step `8559`, active step
  `8560`, phase `microbatch_backward_start`, to completed step `8562`, active
  step `8563`, phase `microbatch_done`. Required return artifacts were still
  absent: no `results.json`, result CSV, eval-detail CSV, or checkpoint. PID
  `95692` stayed alive with `194` threads; GPU utilization sampled `32%` then
  `43%`, with `43311 / 81920` MiB allocated. Keep E143 running and leave
  `EXPERIMENT_RESULTS.md` unchanged until a scored bundle or documented
  terminal no-score state exists.
- 2026-05-17T02:52Z E143 remained live and training. Direct status at
  `02:51:01Z` showed completed step `8569`, active step `8570`, finite last
  train loss `4.768990516662598`, PID `95692` alive, and no returned bundle.
  A follow-up one-minute interval moved from completed step `8570`, active
  step `8571`, to completed step `8574`, active step `8575`; required return
  artifacts were still absent. GPU utilization sampled `0%` then `73%`, with
  `43311 / 81920` MiB allocated. Keep E143 running and leave
  `EXPERIMENT_RESULTS.md` unchanged until a scored bundle or documented
  terminal no-score state exists.
- 2026-05-17T02:54Z E143 was still training coherently. Status had advanced to
  completed step `8581`, active step `8582`, phase `microbatch_done`, with
  finite last train loss `4.26901239156723`, `effective_batch_size=8`,
  `num_workers=0`, and `stopped_early=false`. PID `95692` remained runnable
  with `194` threads, GPU utilization sampled `60%`, and GPU memory stayed at
  `43311 / 81920` MiB. Required return artifacts were still absent: no
  `results.json`, result CSV, eval-detail CSV, or checkpoint. Keep E143
  running and leave `EXPERIMENT_RESULTS.md` unchanged until a scored bundle or
  documented terminal no-score state exists.
- 2026-05-17T02:56Z E143 remained coherent and pre-return. Status advanced to
  completed step `8589`, active step `8590`, phase `microbatch_done`, finite
  last train loss `4.8124619126319885`, `effective_batch_size=8`,
  `num_workers=0`, and `stopped_early=false`. PID `95692` was still alive
  with `194` threads; GPU utilization sampled `53%` with `43311 / 81920` MiB
  allocated. Required result artifacts were still absent, so keep E143
  running and keep `EXPERIMENT_RESULTS.md` unchanged.
- 2026-05-17T02:58Z E143 remained coherent and pre-return. Status advanced to
  completed step `8596`, active step `8597`, phase `microbatch_done`, finite
  last train loss `4.003231644630432`, `effective_batch_size=8`,
  `num_workers=0`, and `stopped_early=false`. PID `95692` remained alive with
  `194` threads; GPU utilization sampled `58%` with `43311 / 81920` MiB
  allocated. No returned bundle existed yet: no `results.json`, result CSV,
  eval-detail CSV, or checkpoint.
- 2026-05-17T02:59Z E143 remained live and pre-return. Status advanced to
  completed step `8602`, active step `8603`, phase `microbatch_done`, finite
  last train loss `4.4618119597435`, `effective_batch_size=8`,
  `num_workers=0`, and `stopped_early=false`. PID `95692` remained alive with
  `194` threads, process CPU time `04:11:03`, and GPU memory `43311 / 81920`
  MiB allocated. No returned bundle existed yet: no `results.json`, result
  CSV, eval-detail CSV, or checkpoint.
- 2026-05-17T03:01Z E143 remained live and pre-return. Status advanced to
  completed step `8609`, active step `8610`, phase `microbatch_done`, finite
  last train loss `4.475048989057541`, `effective_batch_size=8`,
  `num_workers=0`, and `stopped_early=false`. PID `95692` remained alive with
  `194` threads and process CPU time `04:28:10`. No returned bundle existed
  yet: no `results.json`, result CSV, eval-detail CSV, or checkpoint.
- 2026-05-17T03:07Z E143 continued coherent training on owned pod
  `723hbew2jrvxjx`. Status advanced to completed step `8635`, active step
  `8636`, phase `microbatch_done`, finite last train loss `4.806308209896088`,
  `effective_batch_size=8`, `num_workers=0`, and `stopped_early=false`.
  Trainer PID `95692` remained alive with `194` threads, process CPU time
  `05:30:17`, GPU utilization sampled `36%`, and GPU memory stayed at
  `43311 / 81920` MiB. The artifact directory still had only
  `run_metadata.json`, `history_full_msa_to_face.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or checkpoint existed yet. Keep E143 running and keep
  `EXPERIMENT_RESULTS.md` unchanged until a scored bundle or documented
  terminal no-score state exists.
- 2026-05-17T03:10Z E143 remained coherent and pre-return on owned pod
  `723hbew2jrvxjx`. Status advanced to completed step `8648`, active step
  `8649`, phase `microbatch_forward_start`, finite last train loss
  `4.389040514826775`, `effective_batch_size=8`, `num_workers=0`, and
  `stopped_early=false`. Trainer PID `95692` remained alive with `194`
  threads, process CPU time `05:55:53`, GPU utilization sampled `32%`, and
  GPU memory stayed at `43311 / 81920` MiB. Required returned artifacts were
  still absent: no `results.json`, result CSV, eval-detail CSV, or checkpoint.
  Leave E143 running and keep `EXPERIMENT_RESULTS.md` unchanged until a scored
  bundle or documented terminal no-score state exists.
- 2026-05-17T03:11Z E143 continued training coherently on owned pod
  `723hbew2jrvxjx`. Status advanced to completed step `8654`, active step
  `8655`, phase `microbatch_done`, finite last train loss `4.755711793899536`,
  `effective_batch_size=8`, `num_workers=0`, and `stopped_early=false`.
  Trainer PID `95692` remained alive with `194` threads, process CPU time
  `06:08:50`, GPU utilization sampled `42%`, and GPU memory stayed at
  `43311 / 81920` MiB. The run remains pre-return: no `results.json`, result
  CSV, eval-detail CSV, or checkpoint exists yet. Keep E143 running and keep
  `EXPERIMENT_RESULTS.md` unchanged until a scored bundle or documented
  terminal no-score state exists.
- 2026-05-17T03:14Z E143 remained live on owned pod `723hbew2jrvxjx`.
  Status advanced to completed step `8667`, active step `8668`, phase
  `microbatch_start`, finite last train loss `4.531794726848602`,
  `effective_batch_size=8`, `num_workers=0`, and `stopped_early=false`.
  Trainer PID `95692` remained alive with `194` threads, process CPU time
  `06:36:41`, GPU memory stayed at `43311 / 81920` MiB, and the latest status
  mtime was current at `2026-05-17T03:14:24Z`. Required returned artifacts
  were still absent: no `results.json`, result CSV, eval-detail CSV, or
  checkpoint. Continue monitoring E143 only; keep `EXPERIMENT_RESULTS.md`
  unchanged until a scored bundle or documented terminal no-score state exists.
- 2026-05-17T03:16Z E143 remained coherent and pre-return on owned pod
  `723hbew2jrvxjx`. Status advanced to completed step `8673`, active step
  `8674`, phase `microbatch_done`, finite last train loss `4.2225044667720795`,
  `effective_batch_size=8`, `num_workers=0`, and `stopped_early=false`.
  Trainer PID `95692` remained alive with `194` threads, process CPU time
  `06:51:24`, GPU utilization sampled `45%`, and GPU memory stayed at
  `43311 / 81920` MiB. The artifact directory still had no `results.json`,
  result CSV, eval-detail CSV, or checkpoint. Keep E143 running and keep
  `EXPERIMENT_RESULTS.md` unchanged until a scored bundle or documented
  terminal no-score state exists.
- 2026-05-17T03:17Z E143 continued advancing on owned pod `723hbew2jrvxjx`.
  Status reached completed step `8680`, active step `8681`, phase
  `microbatch_done`, finite last train loss `4.598262071609497`,
  `effective_batch_size=8`, `num_workers=0`, and `stopped_early=false`.
  Trainer PID `95692` remained alive with `194` threads, process CPU time
  `07:06:34`, GPU memory stayed at `43311 / 81920` MiB, and the status mtime
  was current at `2026-05-17T03:17:30Z`. Required returned artifacts were
  still absent: no `results.json`, result CSV, eval-detail CSV, or checkpoint.
  Continue monitoring E143 only and keep `EXPERIMENT_RESULTS.md` unchanged
  until a scored bundle or documented terminal no-score state exists.
- 2026-05-17T03:19Z E143 stayed live and pre-return on owned pod
  `723hbew2jrvxjx`. Status reached completed step `8686`, active step `8687`,
  phase `microbatch_done`, finite last train loss `4.698548197746277`,
  `effective_batch_size=8`, `num_workers=0`, and `stopped_early=false`.
  Trainer PID `95692` remained alive with `194` threads, process CPU time
  `07:21:47`, GPU utilization sampled `31%`, and GPU memory stayed at
  `43311 / 81920` MiB. The artifact directory still had no `results.json`,
  result CSV, eval-detail CSV, or checkpoint. Continue monitoring E143 only;
  do not update `EXPERIMENT_RESULTS.md` until a scored bundle or documented
  terminal no-score state exists.
- 2026-05-17T03:20Z E143 remained live and advancing on owned pod
  `723hbew2jrvxjx`. Status reached completed step `8693`, active step `8694`,
  phase `microbatch_done`, finite last train loss `4.576408952474594`,
  `effective_batch_size=8`, `num_workers=0`, and `stopped_early=false`.
  Trainer PID `95692` remained alive with `194` threads, process CPU time
  `07:35:07`, GPU utilization sampled `43%`, and GPU memory stayed at
  `43311 / 81920` MiB. The artifact directory still had no `results.json`,
  result CSV, eval-detail CSV, or checkpoint. Keep E143 running and keep
  `EXPERIMENT_RESULTS.md` unchanged until a scored bundle or documented
  terminal no-score state exists.
- 2026-05-17T03:22Z E143 continued coherent training on owned pod
  `723hbew2jrvxjx`. Status reached completed step `8702`, active step `8703`,
  phase `microbatch_done`, finite last train loss `4.584864497184753`,
  `effective_batch_size=8`, `num_workers=0`, and `stopped_early=false`.
  Trainer PID `95692` remained alive with `194` threads, process CPU time
  `07:55:48`, GPU utilization sampled `36%`, and GPU memory stayed at
  `43311 / 81920` MiB. No returned files exist yet: no `results.json`, result
  CSV, eval-detail CSV, or checkpoint. Keep E143 running and keep
  `EXPERIMENT_RESULTS.md` unchanged until a scored bundle or documented
  terminal no-score state exists.
- 2026-05-17T03:27Z E143 remained coherent and pre-return on owned pod
  `723hbew2jrvxjx`. Status reached completed step `8720`, active step `8721`,
  phase `microbatch_done`, finite last train loss `4.711500942707062`,
  `effective_batch_size=8`, `num_workers=0`, and `stopped_early=false`.
  Trainer PID `95692` remained alive with `194` threads, process CPU time
  `08:36:07`, GPU memory stayed allocated at `43311 / 81920` MiB, and the
  status mtime was current at `2026-05-17T03:27:13Z`. Required returned
  artifacts were still absent: no `results.json`, result CSV,
  eval-detail CSV, or checkpoint. Continue monitoring only E143 and keep
  `EXPERIMENT_RESULTS.md` unchanged until a scored bundle or documented
  terminal no-score state exists.
- 2026-05-17T03:29Z E143 continued advancing on owned pod `723hbew2jrvxjx`.
  Status reached completed step `8729`, active step `8730`, phase
  `microbatch_forward_start`, finite last train loss `4.767255365848541`,
  `effective_batch_size=8`, `num_workers=0`, and `stopped_early=false`.
  Trainer PID `95692` remained alive with `194` threads, elapsed time
  `00:55:45`, process CPU time `08:55:33`, and GPU memory allocated at
  `43311 / 81920` MiB. Required returned artifacts were still absent:
  no `results.json`, result CSV, eval-detail CSV, or checkpoint. Keep E143
  running and keep `EXPERIMENT_RESULTS.md` unchanged until a scored bundle or
  documented terminal no-score state exists.
- 2026-05-17T03:31Z E143 remained coherent and pre-return on owned pod
  `723hbew2jrvxjx`. Status reached completed step `8735`, active step `8736`,
  phase `microbatch_done`, finite last train loss `4.221947491168976`,
  `effective_batch_size=8`, `num_workers=0`, and `stopped_early=false`.
  Trainer PID `95692` remained alive with `194` threads, elapsed time
  `00:57:29`, process CPU time `09:10:37`, GPU utilization sampled `58%`,
  and GPU memory stayed at `43311 / 81920` MiB. Required returned artifacts
  were still absent: no `results.json`, result CSV, eval-detail CSV, or
  checkpoint. Continue monitoring only E143 and keep `EXPERIMENT_RESULTS.md`
  unchanged until a scored bundle or documented terminal no-score state exists.
- 2026-05-17T03:33Z E143 continued coherent training on owned pod
  `723hbew2jrvxjx`. Status reached completed step `8744`, active step `8745`,
  phase `microbatch_done`, finite last train loss `4.612494707107544`,
  `effective_batch_size=8`, `num_workers=0`, and `stopped_early=false`.
  Trainer PID `95692` remained alive with `194` threads, elapsed time
  `00:59:37`, process CPU time `09:31:04`, and GPU memory stayed at
  `43311 / 81920` MiB. The status mtime was current at
  `2026-05-17T03:33:10Z`; no returned bundle existed yet: no `results.json`,
  result CSV, eval-detail CSV, or checkpoint. Keep E143 running and keep
  `EXPERIMENT_RESULTS.md` unchanged until a scored bundle or documented
  terminal no-score state exists.
- 2026-05-17T03:35Z E143 continued coherent training on owned pod
  `723hbew2jrvxjx`. Status reached completed step `8751`, active step `8752`,
  phase `microbatch_done`, finite last train loss `4.747511059045792`,
  `effective_batch_size=8`, `num_workers=0`, and `stopped_early=false`.
  Trainer PID `95692` remained alive with `194` threads, elapsed time
  `01:01:35`, process CPU time `09:50:16`, GPU utilization sampled `38%`,
  and GPU memory stayed at `43311 / 81920` MiB. The status mtime was current
  at `2026-05-17T03:35:01Z`; no returned bundle existed yet:
  no `results.json`, result CSV, eval-detail CSV, or checkpoint. Keep E143
  running and keep `EXPERIMENT_RESULTS.md` unchanged until a scored bundle or
  documented terminal no-score state exists.
- 2026-05-17T03:37Z E143 continued coherent training on owned pod
  `723hbew2jrvxjx`. Status reached completed step `8759`, active step `8760`,
  phase `microbatch_done`, finite last train loss `4.701133549213409`,
  `effective_batch_size=8`, `num_workers=0`, and `stopped_early=false`.
  Trainer PID `95692` remained alive with `194` threads, elapsed time
  `01:03:31`, process CPU time `10:06:42`, and GPU memory stayed at
  `43311 / 81920` MiB. The status mtime was current at
  `2026-05-17T03:36:55Z`; no returned bundle existed yet: no `results.json`,
  result CSV, eval-detail CSV, or checkpoint. Keep E143 running and keep
  `EXPERIMENT_RESULTS.md` unchanged until a scored bundle or documented
  terminal no-score state exists.
- 2026-05-17T03:39Z E143 continued coherent training on owned pod
  `723hbew2jrvxjx`. Status reached completed step `8768`, active step `8769`,
  phase `microbatch_done`, finite last train loss `4.346406996250153`,
  `effective_batch_size=8`, `num_workers=0`, and `stopped_early=false`.
  Trainer PID `95692` remained alive with `194` threads, elapsed time
  `01:05:36`, process CPU time `10:26:51`, and GPU memory stayed at
  `43311 / 81920` MiB. The status mtime was current at
  `2026-05-17T03:39:01Z`; no returned bundle existed yet: no `results.json`,
  result CSV, eval-detail CSV, or checkpoint. Keep E143 running and keep
  `EXPERIMENT_RESULTS.md` unchanged until a scored bundle or documented
  terminal no-score state exists.
- 2026-05-17T03:41Z E143 continued coherent training on owned pod
  `723hbew2jrvxjx`. Status reached completed step `8776`, active step `8777`,
  phase `microbatch_done`, finite last train loss `4.79453057050705`,
  `effective_batch_size=8`, `num_workers=0`, and `stopped_early=false`.
  Trainer PID `95692` remained alive with `194` threads, elapsed time
  `01:07:33`, process CPU time `10:44:38`, GPU utilization sampled `44%`,
  and GPU memory stayed at `43311 / 81920` MiB. The status mtime was current
  at `2026-05-17T03:41:05Z`; no returned bundle existed yet:
  no `results.json`, result CSV, eval-detail CSV, or checkpoint. Keep E143
  running and keep `EXPERIMENT_RESULTS.md` unchanged until a scored bundle or
  documented terminal no-score state exists.
- 2026-05-17T03:43Z E143 continued coherent training on owned pod
  `723hbew2jrvxjx`. Status reached completed step `8785`, active step `8786`,
  phase `microbatch_done`, finite last train loss `4.392650783061981`,
  `effective_batch_size=8`, `num_workers=0`, and `stopped_early=false`.
  Trainer PID `95692` remained alive with `194` threads, elapsed time
  `01:09:50`, process CPU time `11:04:42`, GPU utilization sampled `72%`,
  and GPU memory stayed at `43311 / 81920` MiB. The status mtime was current
  at `2026-05-17T03:43:14Z`; no returned bundle existed yet:
  no `results.json`, result CSV, eval-detail CSV, or checkpoint. Keep E143
  running and keep `EXPERIMENT_RESULTS.md` unchanged until a scored bundle or
  documented terminal no-score state exists.
- 2026-05-17T03:45Z E143 continued coherent training on owned pod
  `723hbew2jrvxjx`. Status reached completed step `8795`, active step `8796`,
  phase `microbatch_done`, finite last train loss `4.653806537389755`,
  `effective_batch_size=8`, `num_workers=0`, and `stopped_early=false`.
  Trainer PID `95692` remained alive with `194` threads, elapsed time
  `01:12:04`, process CPU time `11:24:57`, and GPU memory stayed at
  `43311 / 81920` MiB. The status mtime was current at
  `2026-05-17T03:45:36Z`; no returned bundle existed yet: no `results.json`,
  result CSV, eval-detail CSV, or checkpoint. Keep E143 running and keep
  `EXPERIMENT_RESULTS.md` unchanged until a scored bundle or documented
  terminal no-score state exists.
- 2026-05-17T03:48Z E143 continued coherent training on owned pod
  `723hbew2jrvxjx`. Status reached completed step `8807`, active step `8808`,
  phase `microbatch_done`, active microbatch `1 / 8`, finite last train loss
  `4.538805782794952`, `effective_batch_size=8`, `num_workers=0`, and
  `stopped_early=false`. Trainer PID `95692` remained alive with `194`
  threads, elapsed time `01:15:06`, process CPU time `11:51:42`, GPU
  utilization sampled `45%`, and GPU memory stayed at `43311 / 81920` MiB.
  The status mtime was current at `2026-05-17T03:48:27Z`; no returned bundle
  existed yet: no `results.json`, result CSV, eval-detail CSV, or checkpoint.
  Keep E143 running and keep `EXPERIMENT_RESULTS.md` unchanged until a scored
  bundle or documented terminal no-score state exists.
- 2026-05-17T03:50Z E143 continued coherent training on owned pod
  `723hbew2jrvxjx`. Status reached completed step `8816`, active step `8817`,
  phase `microbatch_done`, active microbatch `1 / 8`, finite last train loss
  `4.434696048498154`, `effective_batch_size=8`, `num_workers=0`, and
  `stopped_early=false`. Trainer PID `95692` remained alive with `194`
  threads, elapsed time `01:17:23`, process CPU time `12:10:35`, GPU
  utilization sampled `41%`, and GPU memory stayed at `43311 / 81920` MiB.
  The status mtime was current at `2026-05-17T03:50:45Z`; no returned bundle
  existed yet: no `results.json`, result CSV, eval-detail CSV, or checkpoint.
  Keep E143 running and keep `EXPERIMENT_RESULTS.md` unchanged until a scored
  bundle or documented terminal no-score state exists.
- 2026-05-17T03:52Z E143 continued coherent training on owned pod
  `723hbew2jrvxjx`. Status reached completed step `8823`, active step `8824`,
  phase `microbatch_done`, active microbatch `1 / 8`, finite last train loss
  `4.688283085823059`, `effective_batch_size=8`, `num_workers=0`, and
  `stopped_early=false`. Trainer PID `95692` remained alive with `194`
  threads, elapsed time `01:19:12`, process CPU time `12:27:16`, GPU
  utilization sampled `25%`, and GPU memory stayed at `43311 / 81920` MiB.
  The status mtime was current at `2026-05-17T03:52:36Z`; no returned bundle
  existed yet: no `results.json`, result CSV, eval-detail CSV, or checkpoint.
  Keep E143 running and keep `EXPERIMENT_RESULTS.md` unchanged until a scored
  bundle or documented terminal no-score state exists.
- 2026-05-17T03:54Z E143 continued coherent training on owned pod
  `723hbew2jrvxjx`. Status reached completed step `8831`, active step `8832`,
  phase `microbatch_done`, active microbatch `1 / 8`, finite last train loss
  `4.1300424337387085`, `effective_batch_size=8`, `num_workers=0`, and
  `stopped_early=false`. Trainer PID `95692` remained alive with `194`
  threads, elapsed time `01:21:17`, process CPU time `12:45:54`, GPU
  utilization sampled `58%`, and GPU memory stayed at `43311 / 81920` MiB.
  The status mtime was current at `2026-05-17T03:54:35Z`; no returned bundle
  existed yet: no `results.json`, result CSV, eval-detail CSV, or checkpoint.
  Keep E143 running and keep `EXPERIMENT_RESULTS.md` unchanged until a scored
  bundle or documented terminal no-score state exists.
- 2026-05-17T03:56Z E143 continued coherent training on owned pod
  `723hbew2jrvxjx`. Status reached completed step `8840`, active step `8841`,
  phase `microbatch_start`, active microbatch `1 / 8`, finite last train loss
  `4.405416905879974`, `effective_batch_size=8`, `num_workers=0`, and
  `stopped_early=false`. Trainer PID `95692` remained alive with `194`
  threads, elapsed time `01:23:13`, process CPU time `13:02:20`, and GPU
  memory stayed at `43311 / 81920` MiB. The status mtime was current at
  `2026-05-17T03:56:45Z`; no returned bundle existed yet: no `results.json`,
  result CSV, eval-detail CSV, or checkpoint. Keep E143 running and keep
  `EXPERIMENT_RESULTS.md` unchanged until a scored bundle or documented
  terminal no-score state exists.
- 2026-05-17T03:58Z E143 continued coherent training on owned pod
  `723hbew2jrvxjx`. Status reached completed step `8848`, active step `8849`,
  phase `microbatch_done`, active microbatch `1 / 8`, finite last train loss
  `4.6501113176345825`, `effective_batch_size=8`, `num_workers=0`, and
  `stopped_early=false`. Trainer PID `95692` remained alive with `194`
  threads, elapsed time `01:25:03`, process CPU time `13:19:58`, and GPU
  memory stayed at `43311 / 81920` MiB. The status mtime was current at
  `2026-05-17T03:58:34Z`; no returned bundle existed yet: no `results.json`,
  result CSV, eval-detail CSV, or checkpoint. Keep E143 running and keep
  `EXPERIMENT_RESULTS.md` unchanged until a scored bundle or documented
  terminal no-score state exists.
- 2026-05-17T04:00Z E143 continued coherent training on owned pod
  `723hbew2jrvxjx`. Status reached completed step `8856`, active step `8857`,
  phase `microbatch_done`, active microbatch `1 / 8`, finite last train loss
  `4.357317119836807`, `effective_batch_size=8`, `num_workers=0`, and
  `stopped_early=false`. Trainer PID `95692` remained alive with `194`
  threads, elapsed time `01:27:00`, process CPU time `13:36:54`, and GPU
  memory stayed at `43311 / 81920` MiB. The status mtime was current at
  `2026-05-17T04:00:33Z`; no returned bundle existed yet: no `results.json`,
  result CSV, eval-detail CSV, or checkpoint. Keep E143 running and keep
  `EXPERIMENT_RESULTS.md` unchanged until a scored bundle or documented
  terminal no-score state exists.
- 2026-05-17T04:02Z E143 continued coherent training on owned pod
  `723hbew2jrvxjx`. Status reached completed step `8862`, active step `8863`,
  phase `microbatch_done`, active microbatch `1 / 8`, finite last train loss
  `4.3537485003471375`, `effective_batch_size=8`, `num_workers=0`, and
  `stopped_early=false`. Trainer PID `95692` remained alive with `194`
  threads, elapsed time `01:28:40`, process CPU time `13:52:19`, GPU
  utilization sampled `45%`, and GPU memory stayed at `43311 / 81920` MiB.
  The status mtime was current at `2026-05-17T04:01:58Z`; no returned bundle
  existed yet: no `results.json`, result CSV, eval-detail CSV, or checkpoint.
  Keep E143 running and keep `EXPERIMENT_RESULTS.md` unchanged until a scored
  bundle or documented terminal no-score state exists.
- 2026-05-17T04:04Z E143 continued coherent training on owned pod
  `723hbew2jrvxjx`. Status reached completed step `8870`, active step `8871`,
  phase `microbatch_done`, active microbatch `1 / 8`, finite last train loss
  `4.657688796520233`, `effective_batch_size=8`, `num_workers=0`, and
  `stopped_early=false`. Trainer PID `95692` remained alive with `194`
  threads, elapsed time `01:30:32`, process CPU time `14:07:26`, and GPU
  memory stayed at `43311 / 81920` MiB. The status mtime was current at
  `2026-05-17T04:04:04Z`; no returned bundle existed yet: no `results.json`,
  result CSV, eval-detail CSV, or checkpoint. Keep E143 running and keep
  `EXPERIMENT_RESULTS.md` unchanged until a scored bundle or documented
  terminal no-score state exists.
- 2026-05-17T04:05Z E143 continued coherent training on owned pod
  `723hbew2jrvxjx`. Status reached completed step `8877`, active step `8878`,
  phase `microbatch_forward_start`, active microbatch `1 / 8`, finite last
  train loss `4.674767941236496`, `effective_batch_size=8`, `num_workers=0`,
  and `stopped_early=false`. Trainer PID `95692` remained alive with `194`
  threads, elapsed time `01:32:14`, process CPU time `14:22:13`, GPU
  utilization sampled `63%`, and GPU memory stayed at `43311 / 81920` MiB.
  The status mtime was current at `2026-05-17T04:05:46Z`; no returned bundle
  existed yet: no `results.json`, result CSV, eval-detail CSV, or checkpoint.
  Keep E143 running and keep `EXPERIMENT_RESULTS.md` unchanged until a scored
  bundle or documented terminal no-score state exists.
- 2026-05-17T04:09Z E143 continued coherent training on owned pod
  `723hbew2jrvxjx`. Status reached completed step `8891`, active step `8892`,
  phase `microbatch_done`, active microbatch `1 / 8`, finite last train loss
  `4.797743558883667`, `effective_batch_size=8`, `num_workers=0`, and
  `stopped_early=false`. Trainer PID `95692` remained alive with `194`
  threads, elapsed time `01:35:43`, process CPU time `14:54:36`, GPU memory
  stayed at `43311 / 81920` MiB, and the GPU utilization sample was `0%`.
  The status mtime was current at `2026-05-17T04:09:15Z`; no returned bundle
  existed yet: no `results.json`, result CSV, eval-detail CSV, or checkpoint.
  Keep E143 running and keep `EXPERIMENT_RESULTS.md` unchanged until a scored
  bundle or documented terminal no-score state exists.
- 2026-05-17T05:21Z E143 completed training to target step `9000` and entered
  full validation evaluation on owned pod `723hbew2jrvxjx`. Status showed phase
  `evaluating`, active eval batch `221 / 1000`, active eval examples `221`,
  completed step `9000`, `effective_batch_size=8`, `num_workers=0`,
  `stopped_early=false`, and last train loss `4.728216648101807`. Trainer PID
  `95692` remained alive with `194` threads, elapsed time `02:47:55`, process
  CPU time `1-12:33:06`, GPU memory `43311 / 81920` MiB, and status mtime
  `2026-05-17T05:21:04Z`. The artifact tree still contained only
  `history_full_msa_to_face.json`, `run_metadata.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or checkpoint existed yet. Keep E143 running and keep
  `EXPERIMENT_RESULTS.md` unchanged until a coherent scored bundle or
  documented terminal no-score state exists.
- 2026-05-17T07:12Z E143 remained coherent in full validation evaluation on
  owned pod `723hbew2jrvxjx`. Status showed phase `evaluating`, active eval
  batch `763 / 1000`, active eval examples `763`, completed step `9000`,
  `effective_batch_size=8`, `num_workers=0`, `stopped_early=false`, and last
  train loss `4.728216648101807`. Trainer PID `95692` remained alive with
  `194` threads, elapsed time `04:39:11`, process CPU time `3-08:07:47`, GPU
  memory `43311 / 81920` MiB, and status mtime
  `2026-05-17T07:11:46Z`. The artifact tree still contained only
  `history_full_msa_to_face.json`, `run_metadata.json`, and
  `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or checkpoint existed yet. Keep E143 running and keep
  `EXPERIMENT_RESULTS.md` unchanged until a coherent scored bundle or
  documented terminal no-score state exists.
- 2026-05-17T16:22Z E143 had returned coherently on owned pod
  `723hbew2jrvxjx`; remote status was `finished`, completed step `9000`,
  `effective_batch_size=8`, `num_workers=0`, `stopped_early=false`, and the
  required files were present: `results.json`, `results.csv`,
  `history_full_msa_to_face.json`, `eval_details_full_msa_to_face.csv`,
  `run_metadata.json`, `status_full_msa_to_face.json`, and
  `checkpoints/full_msa_to_face_latest.pt`. The artifact directory and log
  were pulled locally. Local verification passed with `completed_steps=9000`,
  `eval_rows=1000`, `history_last_step=9000`, `parameters=3240738`, and
  `val_lddt_ca=0.43106790351867674`. The goal audit correctly failed because
  `0.4311` is below `0.7` and `9000` is below the required `30000`
  confirmation steps. Eval-detail analysis showed mean `lddt_ca=0.4311`,
  FoldScore `0.4003`, dRMSD `11.0003`, mean boundary lDDT `0.7504`, and `79`
  high-boundary / low-global examples. Record E143 as returned but rejected;
  do not spend 30k.
- 2026-05-17T16:27Z Launched E144 no-Hodge edge-star residual boundary
  readout on owned pod `723hbew2jrvxjx` from `/workspace/SimplexFold_e144`
  after E143 was verified, recorded, and rejected as a below-threshold short
  gate. Run name:
  `e144_no_hodge_edge_star_residual_from_e128_s9000_c256_m64`; trainer PID
  `172579`; log:
  `/workspace/SimplexFold_e144/logs/e144_no_hodge_edge_star_residual.log`;
  artifact directory:
  `/workspace/SimplexFold_e144/artifacts/nanofold_public_benchmarks/e144_no_hodge_edge_star_residual_from_e128_s9000_c256_m64`.
  The launch briefly produced a duplicate Python process with PID `175937`
  against the same run directory. A trace was preserved locally under
  `artifacts/runpod_traces/e144_duplicate_launch_20260517T1629Z/`, then only
  the later duplicate process group was stopped. The original launch PID
  `172579` remained coherent: status advanced to `completed_step=8509`, active
  step `8510`, active microbatch `1 / 8`, phase `microbatch_done`,
  `effective_batch_size=8`, `num_workers=0`, `stopped_early=false`, finite
  last train loss `4.5722126960754395`, status mtime
  `2026-05-17T16:29:57Z`, GPU utilization `63%`, and GPU memory
  `28669 / 81920` MiB. Keep E144 running and keep `EXPERIMENT_RESULTS.md`
  unchanged until a scored bundle or documented terminal no-score state exists.
- 2026-05-17T16:33Z E144 continued coherent training on owned pod
  `723hbew2jrvxjx`. Status advanced to completed step `8522`, active step
  `8523`, active microbatch `1 / 8`, phase `microbatch_done`,
  `effective_batch_size=8`, `num_workers=0`, `stopped_early=false`, finite
  last train loss `4.637174069881439`, and status mtime
  `2026-05-17T16:33:07Z`. Active trainer PID `172579` remained alive with
  `194` threads, elapsed time `00:07:18`, process CPU time `00:57:36`, GPU
  utilization sampled `45%`, and GPU memory `38189 / 81920` MiB. The artifact
  tree still had only `history_full_msa_to_face.json`, `run_metadata.json`,
  and `status_full_msa_to_face.json`; no `results.json`, result CSV,
  eval-detail CSV, or checkpoint existed yet. Keep E144 running and keep
  `EXPERIMENT_RESULTS.md` unchanged until a scored bundle or documented
  terminal no-score state exists.
- 2026-05-17T16:40Z E144 continued coherent training on owned pod
  `723hbew2jrvxjx`. Status advanced to completed step `8551`, active step
  `8552`, active microbatch `1 / 8`, phase `microbatch_done`, finite last
  train loss `4.8796935975551605`, and active trainer PID `172579` remained
  alive with `194` threads, elapsed time `00:14:40`, and process CPU time
  `02:07:08`. No returned bundle existed yet: no `results.json`, result CSV,
  eval-detail CSV, or checkpoint. Keep E144 running and keep
  `EXPERIMENT_RESULTS.md` unchanged until a scored bundle or documented
  terminal no-score state exists.
- 2026-05-17T16:54Z E144 continued coherent training on owned pod
  `723hbew2jrvxjx`. Status advanced to completed step `8607`, active step
  `8608`, active microbatch `1 / 8`, phase `microbatch_done`, finite last
  train loss `4.753740459680557`, and active trainer PID `172579` remained
  alive with `194` threads, elapsed time `00:28:56`, process CPU time
  `04:22:22`, and GPU memory `43055 / 81920` MiB. No returned bundle existed
  yet: no `results.json`, result CSV, eval-detail CSV, or checkpoint. Keep
  E144 running and keep `EXPERIMENT_RESULTS.md` unchanged until a scored bundle
  or documented terminal no-score state exists.
- 2026-05-17T17:05Z E144 continued coherent training on owned pod
  `723hbew2jrvxjx`. Status advanced to completed step `8650`, active step
  `8651`, active microbatch `1 / 8`, phase `microbatch_done`, finite last
  train loss `4.492419868707657`, and active trainer PID `172579` remained
  alive with `194` threads, elapsed time `00:39:47`, and process CPU time
  `06:06:47`. No returned bundle existed yet: no `results.json`, result CSV,
  eval-detail CSV, or checkpoint. Keep E144 running and keep
  `EXPERIMENT_RESULTS.md` unchanged until a scored bundle or documented
  terminal no-score state exists.
- 2026-05-17T17:18Z E144 continued coherent training on owned pod
  `723hbew2jrvxjx`. Status advanced to completed step `8701`, active step
  `8702`, active microbatch `1 / 8`, phase `microbatch_done`, finite last
  train loss `4.872242033481598`, and active trainer PID `172579` remained
  alive with `194` threads, elapsed time `00:52:18`, and process CPU time
  `08:06:52`. No returned bundle existed yet: no `results.json`, result CSV,
  eval-detail CSV, or checkpoint. Keep E144 running and keep
  `EXPERIMENT_RESULTS.md` unchanged until a scored bundle or documented
  terminal no-score state exists.
- 2026-05-17T17:31Z E144 continued coherent training on owned pod
  `723hbew2jrvxjx`. Status advanced to completed step `8751`, active step
  `8752`, active microbatch `1 / 8`, phase `microbatch_done`, finite last
  train loss `4.749566555023193`, and active trainer PID `172579` remained
  alive with `194` threads, elapsed time `01:05:19`, and process CPU time
  `10:12:33`. No returned bundle existed yet: no `results.json`, result CSV,
  eval-detail CSV, or checkpoint. Keep E144 running and keep
  `EXPERIMENT_RESULTS.md` unchanged until a scored bundle or documented
  terminal no-score state exists.
- 2026-05-17T17:43Z E144 continued coherent training on owned pod
  `723hbew2jrvxjx`. Status advanced to completed step `8800`, active step
  `8801`, active microbatch `1 / 8`, phase `microbatch_done`, finite last
  train loss `4.427721321582794`, and active trainer PID `172579` remained
  alive with `194` threads, elapsed time `01:17:33`, and process CPU time
  `12:07:23`. No returned bundle existed yet: no `results.json`, result CSV,
  eval-detail CSV, or checkpoint. Keep E144 running and keep
  `EXPERIMENT_RESULTS.md` unchanged until a scored bundle or documented
  terminal no-score state exists.
- 2026-05-17T17:56Z E144 continued coherent training on owned pod
  `723hbew2jrvxjx`. Status advanced to completed step `8853`, active step
  `8854`, active microbatch `1 / 8`, phase `microbatch_done`, finite last
  train loss `4.633165746927261`, and active trainer PID `172579` remained
  alive with `194` threads, elapsed time `01:30:50`, and process CPU time
  `14:08:50`. No returned bundle existed yet: no `results.json`, result CSV,
  eval-detail CSV, or checkpoint. Keep E144 running and keep
  `EXPERIMENT_RESULTS.md` unchanged until a scored bundle or documented
  terminal no-score state exists.
- 2026-05-17T18:08Z E144 continued coherent training on owned pod
  `723hbew2jrvxjx`. Status advanced to completed step `8900`, active step
  `8901`, active microbatch `1 / 8`, phase `microbatch_done`, finite last
  train loss `4.522779315710068`, and active trainer PID `172579` remained
  alive with `194` threads, elapsed time `01:42:41`, and process CPU time
  `15:57:54`. No returned bundle existed yet: no `results.json`, result CSV,
  eval-detail CSV, or checkpoint. Keep E144 running and keep
  `EXPERIMENT_RESULTS.md` unchanged until a scored bundle or documented
  terminal no-score state exists.
- 2026-05-17T19:23Z E144 reached the final validation sweep on owned pod
  `723hbew2jrvxjx`. Status showed completed step `9000`, active step `9000`,
  phase `evaluating`, active eval batch `243 / 1000`, effective batch size
  `8`, `num_workers=0`, `stopped_early=false`, finite last train loss
  `4.741060197353363`, and a fresh status mtime at `2026-05-17T19:23:09Z`.
  Trainer PID `172579` remained alive with elapsed time `02:57:25`, process
  CPU time `1-15:04:33`, and active running state. The artifact directory had
  history, run metadata, and status only; no `results.json`, result CSV,
  eval-detail CSV, or checkpoint existed. Keep E144 running and keep
  `EXPERIMENT_RESULTS.md` unchanged until a scored bundle or documented
  terminal no-score state exists.
- 2026-05-17T19:34Z E144 continued the final validation sweep on owned pod
  `723hbew2jrvxjx`. Status showed completed step `9000`, active step `9000`,
  phase `evaluating`, active eval batch `303 / 1000`, effective batch size
  `8`, `num_workers=0`, `stopped_early=false`, finite last train loss
  `4.741060197353363`, and a fresh status mtime at `2026-05-17T19:34:22Z`.
  Trainer PID `172579` remained alive with elapsed time `03:08:56`, process
  CPU time `1-19:32:09`, and active running state. The artifact directory
  still had history, run metadata, and status only; no `results.json`, result
  CSV, eval-detail CSV, or checkpoint existed. Keep E144 running and keep
  `EXPERIMENT_RESULTS.md` unchanged until a scored bundle or documented
  terminal no-score state exists.
- 2026-05-17T19:54Z E144 continued the final validation sweep on owned pod
  `723hbew2jrvxjx`. Status showed completed step `9000`, active step `9000`,
  phase `evaluating`, active eval batch `400 / 1000`, effective batch size
  `8`, `num_workers=0`, `stopped_early=false`, finite last train loss
  `4.741060197353363`, and a fresh status mtime at `2026-05-17T19:54:27Z`.
  Trainer PID `172579` remained alive with elapsed time `03:28:45`, process
  CPU time `2-03:16:10`, and active running state. The artifact directory
  still had history, run metadata, and status only; no `results.json`, result
  CSV, eval-detail CSV, or checkpoint existed. Keep E144 running and keep
  `EXPERIMENT_RESULTS.md` unchanged until a scored bundle or documented
  terminal no-score state exists.
- 2026-05-17T20:17Z E144 continued the final validation sweep on owned pod
  `723hbew2jrvxjx`. Status showed completed step `9000`, active step `9000`,
  phase `evaluating`, active eval batch `504 / 1000`, effective batch size
  `8`, `num_workers=0`, `stopped_early=false`, finite last train loss
  `4.741060197353363`, and a fresh status mtime at `2026-05-17T20:16:34Z`.
  Trainer PID `172579` remained alive with elapsed time `03:51:32`, process
  CPU time `2-12:11:19`, and active running state. The artifact directory
  still had history, run metadata, and status only; no `results.json`, result
  CSV, eval-detail CSV, or checkpoint existed. Keep E144 running and keep
  `EXPERIMENT_RESULTS.md` unchanged until a scored bundle or documented
  terminal no-score state exists.
- 2026-05-17T20:36Z E144 continued the final validation sweep on owned pod
  `723hbew2jrvxjx`. Status showed completed step `9000`, active step `9000`,
  phase `evaluating`, active eval batch `600 / 1000`, effective batch size
  `8`, `num_workers=0`, `stopped_early=false`, finite last train loss
  `4.741060197353363`, and a fresh status mtime at `2026-05-17T20:35:34Z`.
  Trainer PID `172579` remained alive with elapsed time `04:10:29`, process
  CPU time `2-19:37:03`, and active running state. The artifact directory
  still had history, run metadata, and status only; no `results.json`, result
  CSV, eval-detail CSV, or checkpoint existed. Keep E144 running and keep
  `EXPERIMENT_RESULTS.md` unchanged until a scored bundle or documented
  terminal no-score state exists.
- 2026-05-17T20:57Z E144 continued the final validation sweep on owned pod
  `723hbew2jrvxjx`. Status showed completed step `9000`, active step `9000`,
  phase `evaluating`, active eval batch `704 / 1000`, effective batch size
  `8`, `num_workers=0`, `stopped_early=false`, finite last train loss
  `4.741060197353363`, and a fresh status mtime at `2026-05-17T20:56:39Z`.
  Trainer PID `172579` remained alive with elapsed time `04:31:25`, process
  CPU time `3-03:48:30`, and active running state. The artifact directory
  still had history, run metadata, and status only; no `results.json`, result
  CSV, eval-detail CSV, or checkpoint existed. Keep E144 running and keep
  `EXPERIMENT_RESULTS.md` unchanged until a scored bundle or documented
  terminal no-score state exists.
