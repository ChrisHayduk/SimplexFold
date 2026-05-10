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
- `EXPERIMENT_RESULTS.md` is now the durable returned-results tracker. Keep
  live planning and in-flight notes here, and add a row to results only after
  a Runpod experiment returns a final or early-stop validation point.
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
