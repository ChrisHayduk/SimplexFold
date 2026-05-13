# SimplexFold Experiment Notes

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
