# SimplexFold Experiment Notes

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
