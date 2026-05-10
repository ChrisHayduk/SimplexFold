# SimplexFold NanoFold Experiments

## Goal

Improve AF2-medium-matched SimplexFold on NanoFold public validation toward
`val_lddt_ca > 0.7` while keeping parameters within 5% of the AF2-medium
pair-only baseline.

## Fixed Evaluation Contract

- Dataset: NanoFold public train/validation manifests from
  `${NANOFOLD_ROOT}/data/manifests`.
- Execution: experiment runs happen on Runpod CUDA pods. Local execution is
  reserved for unit tests, lint checks, and non-evidential smoke debugging.
- Templates: disabled with `--max-templates 0`.
- Baseline model budget: `configs/medium.toml` with simplex disabled,
  3,106,642 parameters.
- Primary SimplexFold profile: `configs/simplexfold_medium_param_matched.toml`,
  3,106,690 parameters.
- Validation target: `val_lddt_ca`.
- Control variants: `no_simplex`, `faces`, `full`.

## Experiment Queue

### E00: Matched Short-Run Baseline

Purpose: measure current behavior before changing architecture or loss.

Command template:

```bash
NANOFOLD_ROOT=/workspace/nanoFold-Competition
python scripts/run_nanofold_public_benchmarks.py \
  --nanofold-root "${NANOFOLD_ROOT}" \
  --model-config simplexfold_medium_param_matched \
  --variants no_simplex faces full \
  --train-limit 256 \
  --val-limit 64 \
  --steps 1000 \
  --eval-every 100 \
  --max-val-batches 16 \
  --crop-size 128 \
  --msa-depth 32 \
  --extra-msa-depth 0 \
  --max-templates 0 \
  --device cuda \
  --run-name e00_runpod_baseline
```

Decision rule: use this as the Runpod control for short-run iteration. It is
still not enough to claim the 0.7 target, but it is the comparison point for
E01 and later changes.

### E01: Balanced Topology Contact Supervision

Status: implemented and queued for Runpod.

Hypothesis: the learned contact/topology logits drive first-pass neighbor
selection. A class-balanced contact loss should make selected neighbors more
contact-like without adding parameters or introducing non-simplicial
supervision.

Mechanism: modify `SimplexGeometryLoss` so positive and negative contact
terms are normalized separately, then averaged. This keeps the supervision on
the topology scorer that builds `N(i)` for sparse faces/tetras.

Decision rule: keep if targeted tests pass, parameter count remains within
budget, and a short NanoFold run does not regress `full` relative to E00.

Runpod pilot:

```bash
bash scripts/run_runpod_e01_pilot.sh
```

This compares `full` on main commit `a299438` against E01 commit `6c20faa`
with identical Runpod CUDA settings.

Note: the pilot currently defaults to fp32 because a Runpod BF16 smoke exposed
an activation-checkpoint recomputation metadata mismatch in the simplex trunk.

Validation:

- `pytest tests/test_simplex.py::test_simplex_contact_loss_balances_contacts_and_non_contacts`
- `pytest tests/test_simplex.py::test_simplex_geometry_loss_adds_distance_and_consistency_terms`
- `pytest tests/test_simplex.py::test_tiny_alphafold2_profile_emits_simplex_training_tensors`
- `pytest tests/test_trainer.py::test_simplexfold_medium_param_matched_matches_af2_medium_budget`
- `pytest tests/test_trainer.py::test_load_model_config_selects_requested_profile`

### Rejected: Generic C-alpha Distance Loss

Reason: an all-pairs C-alpha distance preservation loss is metric-aligned but
not specifically simplicial. It could help a pair-only AF2 trunk in the same
way and would blur the intended claim.

### E02: Row-Wise Topology Neighborhood Loss

Status: Runpod pilot completed.

Hypothesis: the sparse complex depends on each anchor's selected neighbor star
`N(i)`. A row-wise contact-neighborhood objective should better train the
topology logits for the actual top-k selector than independent pairwise BCE.

Mechanism: for each residue with at least one true contact, treat true contact
neighbors as the target distribution over that residue's valid neighbors and
apply cross entropy to `simplex_contact_logits[i, :]`. This is a direct
supervision signal for the simplicial complex construction step, not an
all-pairs metric-fitting loss.

Decision rule: keep only if Runpod `full` improves over E01/control on best
interim and final `val_lddt_ca` without changing parameter count.

Validation:

- `pytest tests/test_simplex.py::test_simplex_topology_neighborhood_loss_targets_anchor_neighbors`
- E01 targeted tests and budget tests.

### E03: Warm-Started Simplex Boundary Messages

Status: implemented and under Runpod evaluation.

Hypothesis: the current simplex adapter starts as an identity because the
face/tetra residual projections into pair and single states are zero
initialized. That is conservative, but it delays the effect of explicit
higher-order cells on the structure trunk. Warm-starting the simplex boundary
message projections should let faces and tetras influence pair/single
representations immediately while keeping the parameter count unchanged.

Mechanism: initialize the face-to-edge, face-to-single, face-to-tetra,
tetra-to-face, tetra-to-edge, and tetra-to-single MLP output projections with
the standard LeCun initializer instead of the zero "final" initializer.

Decision: keep as the leading candidate. The Runpod pilot improved both final
and best-interim `val_lddt_ca` relative to the main/full control, E01, and
E02.

Scaled follow-up: a 3000-step Runpod pilot at crop 256 / MSA depth 64 reached
best `val_lddt_ca=0.1219` and final `val_lddt_ca=0.1009`. This confirms the
direction is useful but not sufficient for the 0.7 target.

### E04: Selected-Simplex Coordinate Realization Loss

Status: launched on Runpod.

Hypothesis: the pilots show persistent under-expansion of predicted C-alpha
geometry. The model may learn useful face/tetra latent geometry while the
structure module still collapses the realized coordinates. Supervising the
predicted structure only on selected sparse cells should push the realized
geometry to match the topological complex without becoming a generic all-pairs
distance loss.

Mechanism: use `simplex_face_indices` and `simplex_tetra_indices` to compare
predicted C-alpha face areas and tetra signed/absolute volume plus radius of
gyration against the true coordinates for the same selected cells. These terms
are attached to `SimplexGeometryLoss` and add no parameters.

Decision rule: keep if a Runpod pilot improves final `val_lddt_ca` over E03
and reduces the predicted/true radius-of-gyration gap.

Diagnostic result: the 1000-step Runpod pilot improved final `val_lddt_ca` to
`0.2200`, the best result so far, but predicted C-alpha radius of gyration was
still much smaller than the true structures.

Scaled follow-up: the 3000-step Runpod pilot at crop 256 / MSA depth 64
reached best `val_lddt_ca=0.2394` at step 1500 and final
`val_lddt_ca=0.1985`. This confirms the selected-coordinate realization loss
is the strongest direction so far, but it still does not hold the structure
open enough for the 0.7 target.

### E05: Tuned Selected-Cell Coordinate Realization

Status: completed on Runpod.

Hypothesis: E04 shows that selected simplex coordinate realization is the
right topological pressure, but the default face/tetra coordinate weights are
too weak to fix under-expansion at the larger crop/MSA setting.

Mechanism: expose `--simplex-face-coordinate-weight` and
`--simplex-tetra-coordinate-weight` so Runpod experiments can strengthen only
the realized geometry terms attached to selected face/tetra cells. This keeps
the objective mediated by `simplex_face_indices` and `simplex_tetra_indices`;
it is not a generic all-pairs C-alpha loss.

Initial Runpod test: running `full` with the E04 scaled protocol and stronger
coordinate weights: `--simplex-face-coordinate-weight 0.5` and
`--simplex-tetra-coordinate-weight 0.5`.

Decision rule: keep only if `val_lddt_ca` and the predicted/true
radius-of-gyration ratio improve over E04 scaled without destabilizing the
base AF2 loss.

Result: keep. The `0.5` selected-coordinate weights reached final
`val_lddt_ca=0.2948` and final FoldScore `0.2647`, improving over E04 scaled
final `0.1985` and best `0.2394`.

### E06: Strong Selected-Cell Coordinate Realization

Status: completed on Runpod.

Hypothesis: E05 improved the full scaled curve without destabilizing training,
so the coordinate-realization pressure may still be too weak relative to the
base AF2 losses.

Mechanism: run the same scaled protocol with
`--simplex-face-coordinate-weight 1.0` and
`--simplex-tetra-coordinate-weight 1.0`, keeping the pressure restricted to
the selected sparse face/tetra cells.

Decision rule: keep only if it improves over E05 on final `val_lddt_ca` or
materially improves predicted C-alpha radius of gyration without a large
FoldScore regression.

Result: keep as the current reference. The `1.0` selected-coordinate weights
reached final and best `val_lddt_ca=0.3127`, final FoldScore `0.2511`,
final `val_ca_drmsd=14.5496`, and final predicted/true C-alpha radius of
gyration `7.1388 / 15.7622`.

### E07: Selected Simplex Boundary Coordinate Realization

Status: completed on Runpod.

Hypothesis: E04-E06 show that selected face/tetra realization helps, but the
model is still under-expanding the realized structure. Area, volume, and
radius-of-gyration constraints can leave boundary edge lengths too loose.

Mechanism: add coordinate-distance realization losses only on the boundary
edges of the selected sparse face/tetra cells. For selected faces, compare
predicted C-alpha lengths for `(i,j)`, `(i,k)`, and `(j,k)` to the true
lengths. For selected tetras, compare the six boundary edges. This is the
metric 1-skeleton induced by `simplex_face_indices` and
`simplex_tetra_indices`; it is not a generic all-pairs C-alpha loss.

Initial Runpod test: use the same scaled protocol as E05/E06, keep the E06
selected area/volume/Rg weights with `--simplex-face-coordinate-weight 1.0`
and `--simplex-tetra-coordinate-weight 1.0`, and add
`--simplex-face-coordinate-distance-weight 0.5` plus
`--simplex-tetra-coordinate-distance-weight 0.5`.

Decision rule: keep only if final `val_lddt_ca`, dRMSD, or predicted/true
radius-of-gyration ratio improves over E05/E06 without collapsing FoldScore.

Result: keep. The `0.5` selected-boundary coordinate-distance weights reached
best `val_lddt_ca=0.3247` at step 2000 and final `val_lddt_ca=0.3187`.
This improves over E06 on best lDDT, final FoldScore, dRMSD, and
predicted/true radius-of-gyration, while staying within the same parameter
budget.

### E08: Strong Selected Boundary Coordinate Realization

Status: stopped early on Runpod.

Hypothesis: E07 improves the whole curve, especially dRMSD and radius of
gyration, so the selected-boundary distance pressure may still be slightly
underweighted.

Mechanism: keep the E07 protocol and raise only the selected-boundary
coordinate-distance weights from `0.5` to `1.0`. The signal remains limited
to the boundary edges induced by selected face/tetra cells.

Decision rule: keep only if it improves E07 best/final `val_lddt_ca` or
retains similar lDDT while further improving dRMSD and predicted/true radius
of gyration.

Result: reject. The step-500 validation point regressed to
`val_lddt_ca=0.2636`, with predicted C-alpha radius of gyration back down to
`6.0681`. The doubled boundary-distance weight appears to overconstrain or
slow the useful structure expansion.

### E09: Full Simplex With MSA-to-Face Moment

Status: completed on Runpod.

Hypothesis: the explicit face states should be able to receive third-order
evolutionary signals from the MSA, not only pair/single and recycled geometry
signals. This is the `MSA <-> sparse face tensor` part of the SimplexFold
README diagram.

Mechanism: add a `full_msa_to_face` benchmark variant that keeps faces and
tetras enabled while turning on the existing low-rank MSA-to-face moment.
This adds no parameters because the MSA-to-face projections are already in
the SimplexFold adapter; it changes only whether that topological pathway is
active.

Initial Runpod test: use the E07 loss weights, but run
`--variants full_msa_to_face`.

Decision rule: keep only if it improves E07 best/final `val_lddt_ca` or
improves early dRMSD/Rg without destabilizing FoldScore.

Result: keep. The `full_msa_to_face` variant reached final and best
`val_lddt_ca=0.3429`, final FoldScore `0.2689`, and final
`val_ca_drmsd=12.9189`. It is the strongest Runpod result so far and supports
the README's claim that MSA information can usefully interact with explicit
face states.

### E10: Warm-Started MSA-to-Face Moment

Status: stopped early on Runpod and reverted.

Hypothesis: E09 improved the final score even though the MSA-to-face residual
projection still used zero final initialization. As with E03's boundary
message warm start, the explicit MSA-to-face topological path may need a
non-zero start to influence selected face states early enough.

Mechanism: initialize the low-rank MSA-to-face projection with the standard
SimplexMLP final initializer instead of the zero "final" initializer. This
adds no parameters and only changes the active `full_msa_to_face` path.

Initial Runpod test: repeat E09 with the E07 loss weights and
`--variants full_msa_to_face`.

Decision rule: keep only if it improves E09 best/final `val_lddt_ca` or
improves early convergence without losing final FoldScore.

Result: reject. The step-500 validation point fell to
`val_lddt_ca=0.2232` and FoldScore `0.2190`, even though predicted radius of
gyration rose to `10.5325`. The warm MSA-to-face path appears to inject too
much noisy face information early, so the branch tip restores the zero final
initializer used by E09.

### E11: Long-Range Full MSA-to-Face Simplex Topology

Status: stopped early on Runpod.

Hypothesis: E09 is the strongest result, but the sparse complex still starts
from a topology selector with a very strong local-neighborhood bias. That can
leave too few nonlocal contacts for face/tetra cells to express global fold
constraints.

Mechanism: add a `full_msa_to_face_long` benchmark variant that keeps the E09
full face/tetra/MSA-to-face path and adds a positive long-separation topology
bias for residue pairs with sequence separation at least 16. This changes only
the sparse complex construction, not parameter count or data.

Initial Runpod test: repeat E09 with E07/E09 selected-coordinate and
selected-boundary loss weights, but run
`--variants full_msa_to_face_long`.

Decision rule: keep only if it improves E09 best/final `val_lddt_ca` or
improves radius-of-gyration/global geometry without a FoldScore regression.

Result: reject. The step-500 validation point regressed to
`val_lddt_ca=0.2288`, FoldScore `0.2244`, and predicted C-alpha radius of
gyration `6.0858`. The explicit long-range bias appears to choose harmful
nonlocal cells before the topology scorer is reliable.

### E12: Continue Best Full MSA-to-Face Run

Status: completed on Runpod.

Hypothesis: E09 improved through the final 3000-step checkpoint, so the best
topology/loss stack may still be undertrained rather than architecture-limited
at the current protocol length.

Mechanism: resume E09's `full_msa_to_face` checkpoint from step 3000 and
continue the same Runpod protocol to step 6000. This does not change
parameters or data; it tests whether the current SimplexFold stack can keep
improving under the same sparse face/tetra/MSA objective.

Decision rule: keep as the new reference if it improves E09 final
`val_lddt_ca=0.3429` without a severe FoldScore regression.

Result: keep as a modest new reference, but not a breakthrough. The continued
run reached best `val_lddt_ca=0.3472` at step 5000 and final
`val_lddt_ca=0.3449` at step 6000. Final FoldScore improved to `0.2856`,
final `val_ca_drmsd` improved to `11.7918`, and final predicted/true
C-alpha radius of gyration improved to `9.8828 / 15.7622`. The improvement
supports continued training of the E09 stack, but the score remains far from
the `0.7` target.

### E13: Mixed Local/Global Simplex Neighbor Scaffold

Status: stopped early on Runpod.

Hypothesis: E11 failed because a blunt long-range bias selected noisy
nonlocal cells before the topology scorer was reliable. The sparse complex
may need a small guaranteed local scaffold for sequence-neighbor manifold
continuity, while reserving the remaining neighbor slots for learned/global
edges.

Mechanism: add `simplex_local_neighbor_k`, a zero-parameter topology selector
knob. When it is positive, the first slots in each anchor's neighbor list are
the nearest valid sequence neighbors; the remaining slots are selected by the
normal learned contact/recycled-geometry score. The new
`full_msa_to_face_mixed` benchmark variant keeps the E09 face/tetra/MSA path,
sets `simplex_local_neighbor_k=4`, and sets `simplex_local_bias=0.0` so the
other 8 slots are not swallowed by the old local bias.

Decision rule: start with an early Runpod check and keep only if it beats
E11's step-500 regression and approaches or exceeds E09/E12 lDDT without a
FoldScore collapse. This is a topological selector change, not a loss hack:
it changes which vertices are allowed to form persistent face/tetra cells.

Result: reject. The step-500 point was `val_lddt_ca=0.2371`, FoldScore
`0.2238`, `val_ca_drmsd=15.3413`, and predicted/true C-alpha radius of
gyration `6.2290 / 15.4034`. It is only slightly better than E11's failed
long-bias run and much worse than the E09/E12 early curve. Turning off the
broad local bias makes the learned/global slots too noisy early.

### E14: Soft Mixed Local/Global Simplex Neighbor Scaffold

Status: completed on Runpod.

Hypothesis: E13's hard handoff from 4 local slots to 8 unbiased learned/global
slots was too abrupt. The sparse complex may still need the same guaranteed
local scaffold, but the remaining learned slots should keep a reduced local
bias so they do not become arbitrary nonlocal cells before the topology scorer
is reliable.

Mechanism: add `full_msa_to_face_mixed_soft`, which keeps the E13
`simplex_local_neighbor_k=4` local scaffold but uses `simplex_local_bias=2.0`
instead of `0.0`. This is still a zero-parameter topological selector
ablation: it changes how vertices are admitted into selected face/tetra cells.

Decision rule: launch the same early Runpod check and keep only if the step-500
point recovers toward E09/E12 while retaining better nonlocal capacity than
the default local-biased selector.

Result: reject. The run partially recovered from E13 but did not beat E09/E12:
best `val_lddt_ca=0.3264` at step 2000 and final `val_lddt_ca=0.3015`.
Final dRMSD and predicted radius of gyration improved relative to some earlier
runs, but lDDT and FoldScore remained below the E09/E12 stack. The result
suggests that hand-crafted mixed local/global selector variants are less
useful than the default local-biased selector plus learned contact scoring.

### E15: E12 Continuation With Simplex Auxiliary Anneal

Status: completed on Runpod.

Hypothesis: E12 improved global geometry and FoldScore through step 6000, but
lDDT peaked earlier at step 5000. The selected simplex realization losses may
be useful as an early/mid training scaffold, then slightly overconstrain the
structure head once the face/tetra states have learned a geometry prior.

Mechanism: resume the best `full_msa_to_face` E12 checkpoint at step 6000 and
continue to step 9000. Keep the selected face/tetra coordinate and
boundary-distance weights unchanged, but ramp the overall `simplex_aux_weight`
from `1.0` to `0.5` over steps 6000-7000. This is still a simplicial objective
curriculum: the only annealed signal is the auxiliary pressure on selected
face/tetra/contact topology terms.

Decision rule: keep if it improves E12 best `val_lddt_ca=0.3472` or improves
final lDDT while preserving the E12 FoldScore/dRMSD gains.

Result: keep as the new reference. The run reached best and final
`val_lddt_ca=0.3556` at step 9000, final FoldScore `0.3025`,
final `val_ca_drmsd=12.3527`, and predicted/true C-alpha radius of gyration
`9.0217 / 15.7622`. This is the strongest result so far and supports the
idea that selected simplex realization is valuable as a scaffold, but should
be relaxed after the face/tetra states have learned useful geometry.

### E16: Deeper Simplex Auxiliary Anneal

Status: stopped early on Runpod.

Hypothesis: E15 improved after reducing `simplex_aux_weight` from `1.0` to
`0.5`, and its best point was the final checkpoint. A further conservative
anneal may continue to help the structure module optimize lDDT while keeping
the selected simplex losses active enough to preserve the topological
inductive bias.

Mechanism: resume E15 at step 9000 and continue to step 12000, ramping
`simplex_aux_weight` from `0.5` to `0.25` over steps 9000-10000. Keep the
selected face/tetra coordinate and boundary-distance weights unchanged.

Decision rule: keep if it improves E15 final `val_lddt_ca=0.3556` or preserves
lDDT while improving FoldScore/dRMSD.

Result: reject for the lDDT objective. The deeper anneal reached
`val_lddt_ca=0.3506` at step 9500, `0.3400` at step 10000, and `0.3438` at
step 10500, so it did not recover the E15 best. It briefly improved FoldScore
to `0.3062`, but the lower auxiliary pressure appears to trade away C-alpha
lDDT.

### E17: Continue E15 With Constant Simplex Auxiliary Weight

Status: stopped early on Runpod.

Hypothesis: E15's `simplex_aux_weight=0.5` final checkpoint improved lDDT and
FoldScore together, while E16's deeper anneal hurt lDDT. The best next test is
not lower auxiliary pressure, but more training at the E15 scaffold strength.

Mechanism: resume E15 at step 9000 and continue to step 12000 with
`simplex_aux_weight=0.5` held constant. Keep the selected face/tetra
coordinate and boundary-distance weights unchanged.

Decision rule: keep if it improves E15 final `val_lddt_ca=0.3556`.

Result: reject for the lDDT objective. The run nearly tied E15 at step 9500
with `val_lddt_ca=0.3554` and improved FoldScore to `0.3041`; step 10000 had
`val_lddt_ca=0.3541` and FoldScore `0.3094`. Later checkpoints drifted down
to `0.3454` at step 10500 and `0.3441` at step 11000, so the run was stopped.
More training at `simplex_aux_weight=0.5` improves aggregate FoldScore, but it
does not break the C-alpha lDDT plateau.

### E18: Simplex-Only Topology Capacity Within 5% Budget

Status: completed on Runpod.

Hypothesis: E15/E17 may be plateauing because the persistent higher-order
state is too narrow, not because the AF2-style trunk needs more generic
capacity. The exact-match profile leaves almost the full 5% allowance unused.
Spending that slack only on face/tetra channels and the MSA-to-face adapter is
a direct test of whether explicit 2-/3-simplex state helps when it has enough
room to store selected patch and packing geometry.

Mechanism: add `simplexfold_medium_topology_plus`, preserving all medium
MSA/pair/template/structure dimensions and layer counts while increasing
`simplex_c_face` from 24 to 28, `simplex_c_tetra` from 12 to 14,
`simplex_hidden_dim` from 80 to 87, and `simplex_msa_to_face_rank` from 12 to
16. The profile has 3,256,126 parameters, which is 4.81% above the AF2-medium
baseline and below the 5% cap. Run it with the E09/E15 topology-mediated stack:
`full_msa_to_face` plus selected face/tetra coordinate and boundary-distance
losses.

Decision rule: compare the 3000-step curve against E09's 3000-step result
(`val_lddt_ca=0.3429`). Keep and continue with the E15 anneal schedule only if
the early lDDT/FoldScore curve improves or reaches the same lDDT with better
global geometry.

Result: reject as a replacement for E09/E15. E18 briefly beat the E09 curve at
step 2000 (`val_lddt_ca=0.3324` versus E09's `0.3283`) and had a better
FoldScore/dRMSD at that point, but final 3000-step performance fell short:
`val_lddt_ca=0.3350`, FoldScore `0.2655`, and `val_ca_drmsd=13.4524`, versus
E09's final `0.3429`, `0.2689`, and `12.9189`. The larger simplex state is
not harmful in the way the failed selector variants were, but extra face/tetra
capacity alone does not break the lDDT plateau.

### E19: Selected Simplex Boundary lDDT Realization

Status: stopped early on Runpod.

Hypothesis: E18 improved the mid-run geometry but not final local C-alpha
lDDT, which suggests the selected complex can carry useful scale/packing
information while still failing to realize local boundary distances under the
tolerances that lDDT rewards. A dense all-pairs lDDT loss would be independent
of the SimplexFold hypothesis. A loss on only selected face/tetra boundary
edges is different: it makes the learned 2-/3-simplex complex realize the
metric on its own 1-skeleton.

Mechanism: add optional `simplex_face_boundary_lddt_weight` and
`simplex_tetra_boundary_lddt_weight`. For each selected face or tetra, compute
a differentiable lDDT-style tolerance loss on its boundary C-alpha edges,
restricted to local true distances below 15 A and averaged over the usual
0.5/1/2/4 A tolerance levels. Boundary edges selected by many cells naturally
receive more weight through topological multiplicity. Run the E09/E15
`full_msa_to_face` stack with the existing selected coordinate-distance
weights plus modest selected-boundary lDDT weights.

Decision rule: stop early if step 500 falls into the failed-selector band.
Continue only if the curve improves on E09's lDDT/FoldScore trajectory without
substantially worsening dRMSD/Rg. Promote only if it beats E15's current best
`val_lddt_ca=0.3556` after continuation.

Result: reject. With face/tetra boundary-lDDT weights of `0.25`, step 500
fell to `val_lddt_ca=0.2832`, FoldScore `0.2448`, `val_ca_drmsd=14.4789`,
and predicted/true C-alpha radius of gyration `7.1624 / 15.4034`. This is
below E09 at the same point (`0.2928`) and does not justify continuation.

### E20: Lower-Weight Selected Boundary lDDT

Status: stopped early on Runpod.

Hypothesis: E19 may have failed because the selected-boundary lDDT term was
too strong relative to the existing selected coordinate-distance terms. A
fivefold smaller weight could preserve the topological metric-realization
signal without overwhelming early structure formation.

Mechanism: rerun the E19 setup with
`simplex_face_boundary_lddt_weight=0.05` and
`simplex_tetra_boundary_lddt_weight=0.05`, keeping the E09/E15 coordinate and
boundary-distance weights unchanged.

Decision rule: continue only if the step-500 point recovers to the E09/E18
early band.

Result: reject. Step 500 collapsed to `val_lddt_ca=0.2364`,
FoldScore `0.2447`, `val_ca_drmsd=15.5881`, and predicted/true C-alpha
radius of gyration `5.6076 / 15.4034`. The selected-boundary lDDT formulation
is topologically motivated, but it is empirically harmful in this form.

### E21: Stronger Simplex Boundary Messages

Status: stopped early on Runpod.

Hypothesis: E15 shows that selected simplex realization helps, but E18 shows
that simply widening face/tetra states is not enough. The useful signal may be
losing strength when scattered back into the AF2 pair and single streams,
especially because each selected edge/residue averages messages over many
incident cells before the structure module sees them.

Mechanism: add zero-parameter `simplex_pair_update_scale` and
`simplex_single_update_scale` knobs, defaulting to `1.0`, and a
`full_msa_to_face_strong_messages` variant that keeps the E09 topology
(`full_msa_to_face`) but scales both simplex-to-pair and simplex-to-single
residual updates by `1.5`. This tests the architectural simplex-to-trunk
coupling directly without changing the AF2 trunk, adding parameters, or
introducing a new loss.

Decision rule: compare the 3000-step curve against E09. Continue only if the
scaled messages improve lDDT or match lDDT with better FoldScore/dRMSD.

Result: reject. Step 500 fell into a stronger collapse band than E09/E18:
`val_lddt_ca=0.2315`, FoldScore `0.2328`, `val_ca_drmsd=15.1343`, and
predicted/true C-alpha radius of gyration `6.3715 / 15.4034`. Amplifying the
selected face/tetra messages makes the higher-order pathway dominate early
structure formation in a harmful way.

### E22: Damped Simplex Boundary Messages

Status: stopped early on Runpod.

Hypothesis: E21 shows the selected simplex messages are not merely too weak;
amplifying them collapses global scale. The complementary topological question
is whether the explicit 2-/3-simplex states should act as a gentler scaffold
whose geometric signal is integrated by the AF2 pair/single streams rather
than injected at full residual strength.

Mechanism: add `full_msa_to_face_damped_messages`, preserving the E09 selected
complex and MSA-to-face construction but setting `simplex_pair_update_scale`
and `simplex_single_update_scale` to `0.5`. This changes only how strongly
learned face/tetra boundary messages are coupled back to residues and pairs;
it adds no parameters and introduces no new loss.

Decision rule: stop early if step 500 remains below E09's early lDDT band.
Continue if damping recovers or improves lDDT while increasing predicted
C-alpha radius of gyration toward the true value.

Result: reject. Step 500 recovered from E21 but did not improve over the
useful early baselines: `val_lddt_ca=0.2917`, FoldScore `0.2458`,
`val_ca_drmsd=14.4541`, and predicted/true C-alpha radius of gyration
`6.6487 / 15.4034`. Damping both pair and single simplex residuals is safer
than amplifying them, but it does not make the selected complex more useful
than E09.

### E23: Edge-Biased Simplex Messages

Status: stopped early on Runpod.

Hypothesis: E21 and E22 suggest that the direct simplex-to-single update is a
likely source of early coordinate-scale collapse, while the pair stream is the
natural 1-skeleton boundary representation that face/tetra cells should refine.
The higher-order states may be useful if their signal is routed mainly through
pair/edge geometry before the structure module converts it into coordinates.

Mechanism: add `full_msa_to_face_edge_messages`, keeping the E09 selected
complex and MSA-to-face path but setting `simplex_pair_update_scale=1.5` and
`simplex_single_update_scale=0.5`. This biases learned face/tetra boundary
messages toward pair/edge updates and dampens direct residue-state forcing. It
adds no parameters and no new objective.

Decision rule: use the same step-500 gate. Continue only if the edge-biased
coupling beats E09/E22 on lDDT or recovers a substantially better radius of
gyration without losing FoldScore.

Result: reject. Step 500 was worse than both E09 and the fully damped E22
coupling: `val_lddt_ca=0.2509`, FoldScore `0.2355`,
`val_ca_drmsd=15.0561`, and predicted/true C-alpha radius of gyration
`6.2181 / 15.4034`. Biasing the simplex pathway toward pair updates while
damping single updates does not preserve global scale.

### E24: Degree-Normalized Simplex Boundary Realization

Status: stopped early on Runpod.

Hypothesis: E19-E23 suggest the problem is not simply too little lDDT pressure
or too weak simplex-to-trunk messages. The selected boundary losses currently
count an edge once for every incident selected face or tetra. That topological
multiplicity can over-weight local/high-degree boundary edges and encourage a
compact structure that satisfies repeated short constraints while failing
global scale. Normalizing by undirected boundary-edge incidence should make
the selected complex supervise its 1-skeleton more evenly.

Mechanism: add an opt-in `simplex_boundary_degree_normalize` loss flag. When
enabled, selected face/tetra boundary distance, boundary lDDT, and distance
head losses are weighted by inverse undirected edge degree within the selected
cell complex. The supervision remains restricted to boundary edges induced by
`simplex_face_indices` and `simplex_tetra_indices`; no dense all-pairs loss or
new parameters are added.

Decision rule: run the E09/E15 stack with selected coordinate and boundary
distance weights, plus degree normalization. Continue past step 500 only if
the run improves on the E09/E22 early band or materially improves predicted
C-alpha radius of gyration without sacrificing FoldScore.

Result: reject. Step 500 reached `val_lddt_ca=0.2724`, FoldScore `0.2383`,
`val_ca_drmsd=14.1528`, and predicted/true C-alpha radius of gyration
`7.2673 / 15.4034`. Degree normalization opened the structure more than the
message-scale failures, but the lDDT/FoldScore regression relative to E09 and
E22 does not justify continuation.

### E25: Effective-Batch-8 Optimization Pilot

Status: completed on Runpod.

Hypothesis: the current best topology-mediated stack was developed with
effective batch size 1, while the target confirmation run should use effective
batch size 8 for 30,000 optimizer steps. Some of the observed lDDT volatility
and late-run collapse may be optimizer noise rather than a purely architectural
failure. A small effective-batch-8 pilot should test whether the E09/E15 stack
has a better early trajectory under the intended optimization regime.

Mechanism: run `full_msa_to_face` with the E09 selected coordinate and
boundary-distance weights, `batch_size=1`, and `grad_accum_steps=8`. This does
not change parameters or losses; it tests whether the existing simplicial
architecture benefits from the larger effective batch required for the final
30k-step confirmation.

Decision rule: first run a small 500-step gate. Continue to longer effective
batch-8 training only if the validation point improves over the E09/E15 early
band or substantially improves global geometry without losing FoldScore.

Result: reject. The 500-step effective-batch-8 gate reached
`val_lddt_ca=0.2946`, FoldScore `0.2466`, `val_ca_drmsd=14.3073`, and
predicted/true C-alpha radius of gyration `7.6818 / 15.7622`. It rebounded
from the step-250 validation point (`val_lddt_ca=0.2637`), but still did not
improve over the useful E09/E15 early band or the current E15 best. Effective
batch size alone is not enough to break the topology-mediated lDDT plateau.

### E26: MSA-to-Face Two-Skeleton Stabilization

Status: completed on Runpod.

Hypothesis: the E09/E15 stack may be asking the model to learn persistent
3-simplex packing cells before the selected triangular face geometry is
reliable. A protein's sparse 2-skeleton still carries explicit three-residue
patch constraints, normals, areas, and boundary edges, while removing the
noisier four-residue volume/packing pathway.

Mechanism: run the existing `msa_to_face` variant, which enables explicit
faces and the low-rank MSA-to-face moment while disabling tetra states. Keep
the E09 selected face coordinate and boundary-distance weights; tetra loss
weights are harmless because no tetra tensors are produced. This is an
architecture ablation inside the simplicial view, not a generic loss change.

Decision rule: use a 500-step Runpod gate. Continue only if the face-only
2-skeleton improves over the E09/E15 early band or gives a clearly better
global geometry/FoldScore tradeoff.

Result: reject. The official-metrics rerun reached `val_lddt_ca=0.2517` at
step 250 and final `val_lddt_ca=0.2489`, FoldScore `0.2214`,
`val_ca_drmsd=15.8143`, and predicted/true C-alpha radius of gyration
`5.9651 / 15.7622`. Removing tetra states worsened both local lDDT and global
scale, so the full 3-simplex pathway remains preferable despite its plateau.

### E27: No Recycled-Coordinate Topology Feedback

Status: completed on Runpod.

Hypothesis: SimplexFold's README motivates recycling geometry back into the
simplex trunk, but the current runs repeatedly show under-expanded predicted
coordinates. If those collapsed coordinates participate in neighbor selection
and simplex geometry features, recycling may reinforce a poor sparse complex.

Mechanism: add `full_msa_to_face_no_recycled_topology`, which keeps faces,
tetras, the MSA-to-face moment, and the E09/E15 selected-coordinate losses,
but sets `simplex_use_recycled_geometry=false`. The selected complex is then
driven by learned MSA/pair topology rather than predicted coordinate feedback.
This is a topology-construction ablation inside the SimplexFold view, not a
dense metric loss.

Decision rule: use a 500-step Runpod gate. Continue only if disabling
recycled-coordinate topology improves over the E09/E15 early band or improves
global scale without sacrificing FoldScore.

Result: reject. Step 250 reached only `val_lddt_ca=0.2317`, FoldScore
`0.2169`, `val_ca_drmsd=15.5788`, and predicted/true C-alpha radius of
gyration `5.7226 / 15.4034`. Final step 500 was `val_lddt_ca=0.2369`,
FoldScore `0.2354`, `val_ca_drmsd=16.3061`, and radius of gyration
`5.7967 / 15.7622`. Removing recycled-coordinate topology worsens both the
local and global geometry, so the failed plateau is not simply caused by
coordinate feedback contaminating cell selection.

### E28: Training-Only Simplex Topology Teacher Forcing

Status: completed on Runpod.

Hypothesis: the repeated collapse across E21-E27 suggests the sparse simplex
selector may be too noisy before the structure module has learned useful
geometry. If the face/tetra states are initialized on a plausible training
complex, they may learn the topological patch/packing roles described in the
README before being asked to select cells from noisy MSA/pair logits.

Mechanism: add opt-in `simplex_topology_teacher_forcing_*` training knobs.
During training batches only, `model_inputs_from_batch` can pass true C-alpha
coordinates and masks into the model. The simplex adapter linearly blends the
learned topology logits with `-d_true_ca(i,j)` for neighbor selection, then
anneals that teacher weight back to the learned selector. True coordinates are
not used as face/tetra geometry features, and validation/inference do not pass
teacher coordinates. This uses train labels only to scaffold the topology
construction step, not to add a dense all-pairs metric objective, and adds no
parameters.

Initial Runpod gate: `full_msa_to_face` with the E09 selected coordinate and
boundary-distance weights, `--simplex-topology-teacher-forcing-weight 1.0`,
`--simplex-topology-teacher-forcing-weight-final 0.0`,
`--simplex-topology-teacher-forcing-ramp-start-step 250`, and
`--simplex-topology-teacher-forcing-ramp-steps 250`.

Decision rule: run a 500-step Runpod gate. Continue only if the learned
selector handoff improves over the E09/E15 early band without making
FoldScore or global scale worse.

Result: reject. Full teacher forcing reached only `val_lddt_ca=0.1560` at
step 250 while the teacher weight was still 1.0. After annealing to zero, the
final step 500 validation recovered only to `val_lddt_ca=0.2398`, FoldScore
`0.2222`, `val_ca_drmsd=15.5485`, and predicted/true C-alpha radius of
gyration `6.1752 / 15.7622`. Replacing the selector with true-distance
topology is too disruptive for early validation geometry.

### E29: Soft Simplex Topology Teacher Forcing

Status: completed on Runpod.

Hypothesis: E28 may have failed because full true-distance topology
overrode the learned MSA/pair selector instead of stabilizing it. A small
teacher-distance blend could bias the sparse complex toward plausible local
cells while preserving learned topology information.

Mechanism: use the E28 implementation but reduce
`simplex_topology_teacher_forcing_weight` from `1.0` to `0.25`, then anneal
to `0.0` from step 250 to step 500. Keep the E09 selected coordinate and
boundary-distance weights and the `full_msa_to_face` architecture.

Decision rule: use the same 500-step Runpod gate. Continue only if the soft
teacher blend recovers above the E09/E15 early band or improves global scale
without losing FoldScore.

Result: reject. Step 250 improved over E28's full-teacher collapse but still
reached only `val_lddt_ca=0.2161`, FoldScore `0.2196`,
`val_ca_drmsd=16.0005`, and predicted/true C-alpha radius of gyration
`5.5601 / 15.4034`. Final step 500 was `val_lddt_ca=0.2451`, FoldScore
`0.2169`, `val_ca_drmsd=15.4451`, and radius of gyration
`6.7226 / 15.7622`. A softer true-distance prior is less damaging than full
teacher forcing, but it remains well below the E09/E15 early band.

### E30: Simplex Coupling Warmup

Status: completed on Runpod.

Hypothesis: teacher-forcing the selected complex did not help, while previous
message-scale experiments showed strong coupling collapses and damping only
partially recovers. The persistent simplex states may need to learn their
selected patch/packing geometry before their boundary messages are allowed to
drive pair and single representations.

Mechanism: add an opt-in training schedule for simplex residual update
coupling. During training only, `model_inputs_from_batch` can pass scale
overrides to the simplex adapter so the selected face/tetra states keep their
auxiliary realization losses active while their residual messages into pair
and single streams ramp back to full strength. Validation and inference keep
the configured model coupling. This is an architectural curriculum for how the
learned 2-/3-simplex cells communicate with the AF2 1-skeleton and
0-skeleton, not a new metric loss.

Initial Runpod gate: `full_msa_to_face` with the E09 selected coordinate and
boundary-distance weights, `--simplex-update-scale 0.0`,
`--simplex-update-scale-final 1.0`,
`--simplex-update-scale-ramp-start-step 250`, and
`--simplex-update-scale-ramp-steps 250`.

Decision rule: implement narrowly and run a 500-step Runpod gate. Continue
only if the warmup improves over the E09/E15 early band or preserves lDDT
while materially improving global scale/FoldScore.

Result: reject. Step 250, before coupling had ramped up, reached
`val_lddt_ca=0.2411`, FoldScore `0.2168`, `val_ca_drmsd=15.4721`, and
predicted/true C-alpha radius of gyration `5.8110 / 15.4034`. After the
scale ramp reached `1.0`, final step 500 improved global shape to
`val_ca_drmsd=13.9247` and radius of gyration `8.9047 / 16.3091`, but local
accuracy remained low at `val_lddt_ca=0.2854` with FoldScore `0.2405`. The
warmup supports the idea that simplex coupling can help global expansion, but
it does not break the lDDT plateau.

### E31: Damped Simplex Coupling Warmup

Status: completed on Runpod.

Hypothesis: E30 suggests that ramping simplex boundary messages into the AF2
trunk can open the predicted structure, but full `1.0` coupling is too heavy
and still underperforms the damped static-message pilot. A smaller coupling
target may preserve the global-expansion benefit while reducing disruption to
residue-local accuracy.

Mechanism: reuse the E30 training-only coupling schedule, but ramp
`simplex_update_scale` from `0.0` to `0.5` instead of `1.0`. Keep selected
face/tetra coordinate and boundary-distance losses active. This remains a
simplex communication curriculum: persistent 2-/3-simplex states learn their
patch/packing geometry, then write a damped boundary message into pair/single
states.

Decision rule: run the same 500-step Runpod gate. Continue only if final
`val_lddt_ca` exceeds E22/E25's early band or if it preserves comparable
lDDT while improving dRMSD/FoldScore enough to justify a longer run.

Result: reject. Step 250 reached `val_lddt_ca=0.2422`, FoldScore `0.2133`,
`val_ca_drmsd=14.8018`, and predicted/true C-alpha radius of gyration
`6.7519 / 15.4034`. Final step 500, after the damped ramp reached `0.5`,
reached only `val_lddt_ca=0.2578`, FoldScore `0.2332`,
`val_ca_drmsd=14.7889`, and radius of gyration `8.9024 / 16.3091`. Damping
preserved the global expansion seen in E30, but local lDDT and FoldScore
worsened, so coupling warmups are not the next promising direction.

### E32: Topology Capacity With Auxiliary Anneal

Status: stopped early on Runpod.

Hypothesis: E18 showed that spending the allowed parameter headroom on
simplex-only face/tetra capacity was competitive, while E15 showed that
annealing the selected simplex auxiliary scaffold to `0.5` gave the best
validation lDDT so far. Combining those two topological levers may help the
higher-order states store patch/packing geometry without over-constraining
the structure module late in training.

Mechanism: run `simplexfold_medium_topology_plus` with `full_msa_to_face`,
selected face/tetra coordinate weights, and selected boundary-distance
weights. Add a short auxiliary-loss anneal from `simplex_aux_weight=1.0` to
`0.5`, keeping the signal attached to selected face/tetra cells rather than a
dense all-pairs metric.

Decision rule: use a 500-step Runpod gate first. Continue only if it beats the
E18/E25 early band or materially improves FoldScore/dRMSD without losing
lDDT.

Result: reject. The run was stopped early at step 250 after reaching only
`val_lddt_ca=0.2545`, FoldScore `0.2059`, `val_ca_drmsd=14.2821`, and
predicted/true C-alpha radius of gyration `7.2877 / 15.4034`. The
topology-plus profile remained within the 5% AF2-medium parameter budget
(`3,256,126` versus AF2-medium `3,106,642`), but combining extra persistent
face/tetra capacity with E15-style auxiliary annealing did not recover the
E18/E25 early band and weakened FoldScore. Capacity plus auxiliary anneal is
therefore not the next scaling path.

### E33: Simplicial Structure Readout

Status: stopped early on Runpod.

Hypothesis: E30/E31 showed that simplex boundary messages can improve global
expansion but disrupt local lDDT, while E32 showed that adding persistent
face/tetra capacity does not help if the states still influence the model
mainly through the same trunk-level residual path. The higher-order cells may
be learning useful local patch/packing information but writing it into the
wrong place.

Mechanism: add a small gated readout from selected face/tetra states into the
structure input. Pool each learned face/tetra state back onto its boundary
residues and boundary edges, then inject a scaled copy of that boundary
summary into the structure module's single/pair conditioning. The first
implementation is zero-parameter: it reuses the adapter's existing
selected-cell boundary summaries and gates, with
`simplex_structure_readout_scale=0.25` only in the
`full_msa_to_face_structure_readout` benchmark variant. This keeps the change
simplicial: the model realizes coordinates from the boundary summaries of its
own sparse 2-/3-simplex complex rather than adding a generic dense coordinate
loss or widening the AF2 trunk.

Decision rule: keep the AF2-medium trunk fixed and stay within the 5%
parameter budget. First run a short Runpod gate using the E09 selected
coordinate and boundary-distance losses. Continue only if the new readout
preserves at least the E22/E25 early lDDT band while improving FoldScore or
global scale.

Local validation: `full_msa_to_face_structure_readout` leaves the parameter
count unchanged at `3,106,690`, versus AF2-medium `3,106,642`, and remains
within the 5% budget. Focused tests cover adapter readout emission, benchmark
variant wiring, train-step behavior, simplex curricula, and the parameter
budget.

Result: reject. The corrected Runpod launch reached only
`val_lddt_ca=0.2405` at step 250, with FoldScore `0.2108`,
`val_ca_drmsd=14.8467`, and predicted/true C-alpha radius of gyration
`6.9826 / 15.4034`. This is below the E22/E25 early band and does not improve
FoldScore enough to continue. The result suggests that adding structure
readout on top of the usual repeated simplex residual updates still perturbs
the trunk more than it helps coordinate realization.

### E34: Readout-Only Simplicial Sidecar

Status: stopped early on Runpod.

Hypothesis: E33 did not isolate the new readout path because the simplex
adapter still wrote its usual boundary residuals into the pair and single
streams inside every enabled Evoformer block. If those residual perturbations
are the source of the local-lDDT damage seen in E21-E33, a sidecar mode may
let persistent face/tetra states learn patch/packing summaries and pass them
only to the structure module.

Mechanism: reuse the E33 structure readout but set
`simplex_pair_update_scale=0.0` and `simplex_single_update_scale=0.0`. Keep
the selected face/tetra auxiliary losses active, and inject only a small
readout summary into the final structure input. The benchmark variant
`full_msa_to_face_structure_readout_only` uses
`simplex_structure_readout_scale=0.5` while adding no parameters. This keeps
the topology explicit while separating "learn higher-order cells" from
"rewrite the AF2 trunk after every block."

Decision rule: run a short Runpod gate. Continue only if readout-only
recovers at least the E22/E25 early lDDT band or materially improves
FoldScore/global scale without a further local-accuracy drop.

Result: reject. Step 250 reached only `val_lddt_ca=0.2426`, FoldScore
`0.2103`, `val_ca_drmsd=14.9311`, and predicted/true C-alpha radius of
gyration `6.5743 / 15.4034`. Removing the repeated simplex residual writes
did not recover local accuracy, so the face+tetra readout itself is not yet a
useful structure-conditioning signal.

### E35: Face-Only Structure Sidecar

Status: stopped early on Runpod.

Hypothesis: E33/E34 both used face and tetra summaries in the structure
readout. The tetra summaries may still be too noisy early in training and may
inject four-body packing signals before the model has learned reliable
two-simplex patches. A face-only sidecar would ask whether the learned
2-skeleton can provide a gentler local patch signal to the structure module.

Mechanism: reuse the readout-only sidecar but disable tetra construction:
`simplex_use_tetra=false`, `simplex_pair_update_scale=0.0`,
`simplex_single_update_scale=0.0`, and nonzero
`simplex_structure_readout_scale`. The benchmark variant
`face_structure_readout_only` keeps MSA-to-face active, uses
`simplex_structure_readout_scale=0.5`, and adds no parameters. Keep selected
face coordinate and boundary-distance supervision active. This remains
simplicial because the structure module sees boundary summaries from selected
learned faces, not a dense all-pairs metric loss.

Decision rule: run a short Runpod gate. Continue only if the face-only
readout recovers above the E33/E34 band and approaches the E22/E25 early
range without worsening FoldScore.

Result: reject. Step 250 reached `val_lddt_ca=0.2406`, FoldScore `0.2062`,
`val_ca_drmsd=13.0352`, and predicted/true C-alpha radius of gyration
`9.1316 / 15.4034`. The larger predicted radius suggests face-only readout
can pass some expansion signal, but local lDDT and FoldScore remained in the
same weak band as E33/E34. The readout family is therefore not the next
scaling path until the selected 2-simplex states themselves become more
reliable.

### E36: Topology Margin Selector

Status: stopped early on Runpod.

Hypothesis: E33-E35 suggest that pooling selected face/tetra states into the
structure module is not useful until the sparse complex itself is more
reliable. The topology selector currently gets balanced contact BCE and a
row-wise positive-neighborhood cross entropy, but the top-k neighbor star can
still be polluted by high-scoring hard non-contacts. A margin term on the
simplex contact logits should make the 1-skeleton cleaner before faces and
tetras are built.

Mechanism: add an optional hard-negative margin loss to
`SimplexGeometryLoss`. For each anchor residue with at least one true contact,
compare the positive contact-neighborhood logit energy against the highest
non-contact logits in that row. Penalize hard non-contacts that sit within a
configurable margin. This is not an lDDT-shaped or all-pairs coordinate loss:
it only trains the logits that construct the sparse 1-skeleton used by the
explicit 2-/3-simplex states.

Decision rule: run a 500-step Runpod gate on the E09 selected-coordinate stack
with MSA-to-face enabled. Continue only if the margin selector recovers above
the weak E33-E35 readout band and approaches the E22/E25 early range without
damaging FoldScore.

Result: reject. The first Runpod launch was stopped as invalid before
validation because the benchmark runner had not wired the new margin override
into its local `AlphaFoldLoss` construction path. After fixing that path and
verifying `simplex_topology_margin_weight=0.05` in the runner, the corrected
run reached only `val_lddt_ca=0.1286` at step 250, with FoldScore `0.1857`,
`val_ca_drmsd=13.5268`, and predicted/true C-alpha radius of gyration
`13.1096 / 15.4034`. The margin term did separate training contact logits,
but it appears to over-constrain the selector and damages local structure
quality. Do not continue hard topology-margin weighting in this form.

### E37: Selected Face Normal Orientation

Status: stopped early on Runpod.

Hypothesis: the strongest SimplexFold runs came from selected face/tetra
coordinate realization, but the face realization terms only supervise edge
lengths and area. The README motivation explicitly assigns oriented patch
information to 2-simplices. A selected-face normal term may give each learned
face state a real orientation target without adding a generic all-pairs
coordinate loss.

Mechanism: add an optional selected-face normal loss. For each selected face
`(i, j, k)`, compute its C-alpha normal, express that normal in the local
N-CA-C backbone frame at each boundary residue, and compare predicted versus
true local normal directions. Expressing normals in residue-local frames keeps
the loss invariant to global rigid motion while still supervising the
orientation of the learned 2-simplex boundary. The term adds no parameters
and is active only when `--simplex-face-normal-weight` is nonzero.

Decision rule: run a 500-step Runpod gate on the E09 selected-coordinate stack
with MSA-to-face enabled and a small face-normal weight. Continue only if the
oriented-face signal improves over the E33-E36 weak band and approaches the
E22/E25 early range without losing FoldScore.

Result: reject. Step 250 reached `val_lddt_ca=0.2464`, FoldScore `0.2109`,
`val_ca_drmsd=14.9943`, and predicted/true C-alpha radius of gyration
`6.4679 / 15.4034`. The face-normal term was active
(`val_weighted_simplex_face_normal_loss=0.0501`), but it did not recover
above the E33-E35 readout band or approach the E22/E25 early range. The
selected-face orientation signal is not useful at this weight on the current
E09 coordinate stack.

### E38: Selected Simplex Shape Realization

Status: stopped early on Runpod.

Hypothesis: E37 adds orientation pressure to selected faces, but the core
SimplexFold motivation is stronger: explicit faces and tetras should act like
learned local geometric cells. The existing selected-coordinate terms supervise
area, volume/radius, and boundary distances separately. A rigid local cell
realization term may help the model learn each selected 2-/3-simplex as one
coherent shape instead of optimizing disconnected scalars.

Mechanism: add optional selected face/tetra shape losses. For each selected
face `(i, j, k)` or tetra `(i, j, k, l)`, gather predicted and true C-alpha
vertices, center the cell, align predicted to true by a proper Kabsch rotation,
and score normalized vertex RMSD. The term is globally rigid-motion invariant,
preserves tetra chirality by disallowing reflection in the alignment, adds no
parameters, and only supervises cells chosen by the sparse simplex topology.

Decision rule: run a 500-step Runpod gate on the E09 selected-coordinate stack
with MSA-to-face enabled and small face/tetra shape weights. Continue only if
the local rigid-cell signal improves over the E33-E37 weak band and approaches
the E22/E25 early range without losing FoldScore.

First launch note: the initial E38 Runpod launch was stopped as invalid before
validation after losses became `NaN` by step 50. The corrected E38r2 patch
keeps the proper Kabsch alignment as a cell-local frame solve but computes the
rotation without backpropagating through SVD, then backpropagates only through
the aligned predicted vertices.

Result: reject. The corrected E38r2 run was stable but weak at the first
validation point: step 250 reached `val_lddt_ca=0.2402`, FoldScore `0.2113`,
`val_ca_drmsd=14.9614`, and predicted/true C-alpha radius of gyration
`6.6367 / 15.4034`. The selected-cell shape losses were active, but the run
did not improve over E37 or approach the E22/E25 early range. This supports the
PDF-derived lesson that additional scalar realization terms are less promising
than improving explicit cell communication and local-frame message use.

### E39: Outer-Edge Cell Communication

Status: stopped early on Runpod.

Hypothesis: Topotein argues that higher-rank protein cells need dedicated
neighborhoods; simply attaching a cell feature to an otherwise pairwise model
can be counterproductive. SimplexFold currently builds explicit selected faces
and tetras, but the most useful cell-to-cell information may flow through
boundary edges that leave one local cell and enter another, analogous to
Topotein's outer-edge neighborhoods for secondary-structure cells.

Mechanism: add a zero-parameter `simplex_outer_edge_update_scale` path and a
`full_msa_to_face_outer_edge` benchmark variant. The adapter scatters each
selected face state to its three undirected boundary edges, gathers the average
state of other selected faces incident on those edges, and applies a gated
face-state residual before pair/single readout. Keep the construction based
only on official MSA features, selected topology logits, and recycled
coordinates. Do not use DSSP or external secondary-structure labels.

Local validation: focused tests pass for shared-edge communication, CLI
variant parsing, and zero-parameter behavior. Parameter audit gives AF2-medium
`3,106,642`, SimplexFold `3,106,690`, and E39 outer-edge `3,106,690`, within
the 5% AF2-medium budget.

Decision rule: keep if explicit inter-cell communication improves the early
validation band without increasing parameters outside the AF2-medium budget.

Result: reject. E39 was stable but did not break the weak E33-E38 band. The
first validation point at step 250 reached `val_lddt_ca=0.2460`, FoldScore
`0.2163`, `val_ca_drmsd=14.7805`, and predicted/true C-alpha radius of
gyration `6.7531 / 15.4034`. The small FoldScore and dRMSD nudge over E38 is
not enough to justify continuing this branch.

### E40: Edge-Frame Scalarized Simplex Messages

Status: stopped early on Runpod.

Hypothesis: Topotein's TCP module gets mileage from edge-centric local frames:
vector information from higher-rank cells is scalarized in frames associated
with directed edges. SimplexFold already has selected face normals and
tetrahedral geometry, but most of this information is used as scalar targets or
raw geometric features. Projecting face/tetra orientation signals into boundary
edge frames may make higher-order messages more usable by the AF2-style pair
stream.

Mechanism: add `simplex_edge_frame_message_scale` and a
`full_msa_to_face_edge_frame_messages` benchmark variant. For each selected
face boundary edge, construct a directed edge frame from recycled C-alpha
coordinates and the recycled residue frame, scalarize the opposite vertex and
face normal in that frame, and feed those scalars plus the face state to a
learned edge-specific pair update. For each selected tetra boundary edge,
scalarize the two opposite vertices, their local plane normal, inter-opposite
angle, and signed volume in the same edge frame before a learned tetra-to-pair
update. This remains a topological architecture change because the frames and
messages live on the selected 1-skeleton boundaries of explicit 2-/3-cells.

Local validation: focused tests pass for rigid-transform invariance of the
edge-frame features, pair-readout changes, CLI variant parsing, and budget.
Parameter audit gives AF2-medium `3,106,642`, SimplexFold `3,106,690`, and
E40 edge-frame `3,154,242`, within the 5% AF2-medium budget.

Decision rule: keep the parameter change near zero. Continue only if the
frame-scalarized cell messages beat the E33-E38 weak band and preserve
FoldScore.

Result: reject. E40 was stable but worse than E39 at the first validation
point: step 250 reached `val_lddt_ca=0.2350`, FoldScore `0.2139`,
`val_ca_drmsd=15.2338`, and predicted/true C-alpha radius of gyration
`6.3502 / 15.4034`. The added edge-frame readout did not make the selected
cell messages more useful in this short gate.

### E41: Latent Rank-2 Segment Cells

Status: stopped early on Runpod.

Hypothesis: Topotein's strongest protein-specific inductive bias is a rank-2
cell for secondary-structure-like groups. NanoFold official runs cannot use
external DSSP or template-derived labels, but SimplexFold could learn latent
rank-2 segment cells from sequence-local windows and recycled geometry. These
would complement sparse triangular faces: faces capture three-residue patches,
while segment cells capture longer contiguous local topology.

Mechanism: add `simplex_segment_cell_scale`, `simplex_segment_radius`, and
`simplex_c_segment`, plus a `full_msa_to_face_segment_cells` benchmark
variant. Each residue owns one latent contiguous segment cell over
`i - r ... i + r`. The cell state is initialized from the anchor single state,
the masked pooled single states in the window, masked anchor-to-window pair
features, and invariant recycled C-alpha geometry summaries for the segment.
Selected faces gather the three incident segment states at their vertices and
receive a gated residual before the existing face-to-edge and face-to-single
readouts. This keeps the change inside the simplicial/topological view: it
adds another learned rank-2 cochain and an incidence map from segment cells to
selected triangular faces.

Local validation: segment-cell indexing respects sequence masks, segment
geometry features are rigid-transform invariant, enabling the segment path
changes adapter outputs, the CLI accepts the new variant, and the budget test
keeps the model within 5% of AF2-medium. Parameter audit gives AF2-medium
`3,106,642`, SimplexFold `3,106,690`, and E41 segment cells `3,234,450`,
below the 5% cap of `3,261,974`.

Decision rule: run a 500-step Runpod gate at crop 256 / MSA depth 64 with the
current selected-coordinate stack. Continue only if the first validation point
breaks out of the weak E33-E40 band or shows a meaningful improvement in
FoldScore/dRMSD/radius-of-gyration behavior.

Result: reject. E41 was stable but weak at the first validation point: step
250 reached `val_lddt_ca=0.2393`, FoldScore `0.2125`,
`val_ca_drmsd=15.2012`, and predicted/true C-alpha radius of gyration
`6.2747 / 15.4034`. The selected-coordinate terms were active, but the latent
segment cells did not improve over E39/E40 or the older E22/E25/E30 early
range.

### E42: Damped Hodge Face Residual

Status: stopped early on Runpod.

Hypothesis: E39's lower face adjacency through shared boundary edges was too
weak on its own, and E40/E41 added side channels that did not improve the pair
stream. A better topological intervention is to update selected face cochains
with a small Hodge-style residual before face/tetra states write back into the
AF2 pair and single streams. The residual combines lower adjacency through
shared selected boundary edges with upper adjacency through selected tetra
cofaces.

Mechanism: add `simplex_hodge_face_update_scale` and the
`full_msa_to_face_hodge_residual` benchmark variant. The lower term reuses the
shared-boundary-edge face delta from E39. The upper term gathers the three
selected anchored faces incident on each selected tetra, averages sibling
face states within each coface, scatters those co-boundary messages back to
face slots with degree normalization, and applies a gated face-state residual.
This adds no parameters and does not add a new loss; it changes message
passing on the selected 2-/3-cell complex itself.

Local validation: focused tests pass for tetra co-boundary averaging, adapter
output changes without parameter growth, CLI parser acceptance, and zero
parameter budget. Parameter audit gives AF2-medium `3,106,642`, SimplexFold
`3,106,690`, and E42 Hodge residual `3,106,690`, within the 5% AF2-medium
budget.

Decision rule: run a 500-step Runpod gate at crop 256 / MSA depth 64 with the
current selected-coordinate stack. Continue only if the first validation point
beats the E33-E41 weak band or at least recovers the stronger E22/E25/E30
early range while preserving FoldScore and radius-of-gyration behavior.

Result: reject. E42 modestly improved over E33-E41 at the first validation
point but did not recover the stronger E22/E25/E30 early range. Step 250
reached `val_lddt_ca=0.2545`, FoldScore `0.2112`, `val_ca_drmsd=14.7096`,
and predicted/true C-alpha radius of gyration `6.7897 / 15.4034`. The run
continued through step 450 and entered the final full-validation path, but no
new validation row was written after an extended wait, so the process was
stopped and the pod was shut down. Treat the zero-parameter Hodge residual as
a weak positive diagnostic, not a keep.

### E43: Hodge Residual With Auxiliary Anneal

Status: completed on Runpod.

Hypothesis: E42 shows the Hodge face residual is a mild positive architectural
prior, while E15 shows that the selected-simplex auxiliary scaffold should be
relaxed after it has helped face/tetra states learn geometry. Combining the
zero-parameter Hodge residual with an E15-style auxiliary anneal may preserve
the pair/single boundary-message benefit without overconstraining the
structure module late in the short gate.

Mechanism: run `full_msa_to_face_hodge_residual` with the same selected
face/tetra coordinate and boundary-distance weights as E09/E15/E42. Add a
short overall auxiliary-weight ramp from `simplex_aux_weight=1.0` to `0.5`
over steps 250-500. This keeps the change in the topological view because the
annealed terms are still the selected face/tetra/contact topology scaffold,
not a dense lDDT-targeting loss.

Decision rule: run a 500-step Runpod gate at crop 256 / MSA depth 64 with
bounded 16-batch validation at the intermediate and final checkpoints. Keep
only if the final/best validation point improves over E42 and approaches the
E22/E25/E30 early range without a FoldScore or radius-of-gyration collapse.

Result: reject. E43 verified that the auxiliary-weight anneal was applied to
the selected-simplex scaffold: `simplex_aux_weight` was `1.0` through step
250, then ramped to `0.5` by step 500. Validation improved during the anneal
from `val_lddt_ca=0.2388`, FoldScore `0.2120`, and
`val_ca_drmsd=15.0913` at step 250 to `val_lddt_ca=0.2492`, FoldScore
`0.2232`, and `val_ca_drmsd=15.1139` at step 500, with predicted/true
C-alpha radius of gyration `6.1772 / 15.4034`. This did not beat E42 and did
not approach the E22/E25/E30 early range. The result argues that the Hodge
face residual is not enough on its own; the next iteration should improve
how the sparse complex is constructed or realized before adding more
downstream structure conditioning.

### E44: Soft Flag-Complex Closure

Status: completed on Runpod.

Hypothesis: the selected higher-order complex is currently too permissive.
Each residue selects a top-k neighbor list, then every neighbor pair becomes a
face and every neighbor triplet becomes a tetra. In a flag-complex view,
however, a filled triangle or tetrahedron should be trusted only when its
boundary edges are also plausible. Weighting selected cell masks by their
boundary-edge topology probabilities may suppress noisy open faces/tetras
without adding a generic pair-distance loss or widening the AF2 trunk.

Mechanism: add `simplex_boundary_closure_weight` and
`simplex_boundary_closure_temperature`. During topology construction, gather
the learned boundary-edge scores for each selected face/tetra, convert them
to edge probabilities, take the geometric mean over the boundary 1-skeleton,
and blend that closure score into the selected cell mask. The new
`full_msa_to_face_flag_closure` variant enables MSA-to-face messages with a
soft closure weight of `0.5` and temperature `1.0`. This is a zero-parameter
change to the sparse cell complex itself.

Decision rule: run a 500-step Runpod gate at crop 256 / MSA depth 64 with the
same selected face/tetra coordinate and boundary-distance weights used by
E09/E15. Use the E15-style auxiliary anneal from `1.0` to `0.5` over steps
250-500, because E43 confirmed that annealing improves the final checkpoint
within a run even when the architecture is weak. Continue only if E44 beats
E42/E43 and moves back toward the E22/E25/E30 early range.

Result: reject. E44's soft flag-complex gate improved over E43 at the first
checkpoint but not over E42: step 250 reached `val_lddt_ca=0.2449`,
FoldScore `0.2105`, `val_ca_drmsd=14.8883`, and predicted/true C-alpha
radius of gyration `6.6400 / 15.4034`. After the auxiliary anneal, step 500
fell to `val_lddt_ca=0.2111`, FoldScore `0.2241`, `val_ca_drmsd=16.1468`,
and radius of gyration `5.0536 / 15.4034`. This suggests the closure gate
suppressed noisy higher-order cells, but also weakened the sparse complex too
much for late structure expansion. If revisited, it should be staged or made
temperature/weight-ramped rather than applied at fixed strength from step 1.

### E45: Light Soft Flag-Complex Closure

Status: completed on Runpod.

Hypothesis: E44 may have failed because the closure gate was too strong from
step 1, not because flag-complex closure is a bad topological prior. A much
lighter blend should preserve most face/tetra messages while still gently
downweighting open cells whose learned boundary 1-skeleton is implausible.

Mechanism: add `full_msa_to_face_flag_closure_soft`, identical to E44 except
`simplex_boundary_closure_weight=0.1` instead of `0.5`. This keeps the same
zero-parameter flag-complex construction but reduces the maximum early mask
suppression.

Decision rule: run the same 500-step Runpod gate and E15-style auxiliary
anneal as E44. Continue only if the lighter closure beats E44's step-250
checkpoint and does not collapse the final checkpoint radius/lDDT.

Result: reject. E45 confirmed that reducing closure strength helps relative
to E44 but does not make the flag-complex gate competitive. Step 250 reached
`val_lddt_ca=0.2477`, FoldScore `0.2112`, `val_ca_drmsd=14.8438`, and
predicted/true C-alpha radius of gyration `6.4528 / 15.4034`, slightly above
E44's first checkpoint but still below E42. Step 500 reached
`val_lddt_ca=0.2273`, FoldScore `0.1992`, `val_ca_drmsd=14.9228`, and
radius of gyration `7.3539 / 15.4034`. The lighter gate avoided E44's severe
late radius collapse but still degraded final structure quality, so closure
should not remain a fixed mask in the main path.

### E46: Expanded Selected Complex

Status: completed on Runpod.

Hypothesis: the default selected complex may under-cover local packing
constraints. With `simplex_neighbor_k=12`, each anchor can form 66 selected
faces and 220 selected tetras before masking. Increasing to 14 expands that
to 91 faces and 364 tetras, giving the same learned face/tetra channels more
candidate 2-/3-cells without adding parameters or changing the AF2 trunk.

Mechanism: add `full_msa_to_face_expanded_complex`, identical to the strong
E09/E15 MSA-to-face stack except `simplex_neighbor_k=14`. This is a
topological architecture ablation: it changes the sparsity and coface
coverage of the learned complex, not the scalar loss target.

Decision rule: run a 500-step Runpod gate at crop 256 / MSA depth 64 with the
same selected face/tetra coordinate and boundary-distance weights as E15 and
the same auxiliary anneal from `1.0` to `0.5` over steps 250-500. Continue
only if expanded coverage recovers the E22/E25/E30 early range or improves
FoldScore/radius behavior without a compute or stability penalty.

Result: reject. E46 reached its best checkpoint at step 250 with
`val_lddt_ca=0.2517`, FoldScore `0.2069`, `val_ca_drmsd=14.4515`, and
predicted/true C-alpha radius of gyration `7.1049 / 15.4034`. The final
step-500 checkpoint fell to `val_lddt_ca=0.2327`, FoldScore `0.2215`,
`val_ca_drmsd=15.5059`, and radius of gyration `5.7840 / 15.4034`. Expanding
the selected complex modestly improved over E43-E45 early but did not recover
the stronger E22/E25/E30 range and still collapsed late, so simple K-expansion
should not remain in the main path.
