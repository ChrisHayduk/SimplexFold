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

## Reference-Paper Design Rules

The local PDFs in `references/papers/` were re-read from full-text extraction
on 2026-05-12. They sharpen the filter for future ideas:

- Treat selected neighbor graphs, sparse faces/tetras, incidence relations,
  and outer-edge neighborhoods as model variables, not only diagnostics.
- Prefer changes that alter cochain communication across ranks: edge-to-face,
  face/tetra-to-boundary-edge, outer-edge exchange, or gated selected-cell
  readout.
- Keep realization losses attached to the selected sparse complex. Generic
  all-pairs coordinate losses remain out of scope unless they are recast as
  supervision of the model-selected cell complex.
- Do not require DSSP, SSE labels, templates, external structures, pretrained
  weights, or external MSA/template retrieval in official NanoFold paths.
- When a complex-construction change returns, report topology-aware
  diagnostics alongside lDDT: selected-cell counts, boundary-edge reuse,
  selected boundary lDDT, boundary length error, and contraction fraction.

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

Status: completed on Runpod.

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

### E47: Auxiliary Flag-Closure Curriculum

Status: completed on Runpod.

Hypothesis: E44/E45 may have failed because closure was applied directly to
the selected cell masks, suppressing face/tetra message passing before the
topology scorer had learned useful edges. The topological prior is still
sound: a filled face or tetra should be a stronger coordinate-realization
target when its boundary 1-skeleton is itself locally plausible.

Mechanism: add a training-only `simplex_cell_closure_weight` loss knob. For
each selected face or tetra, compute a soft flag-closure score from the true
C-alpha boundary-edge distances and blend that score into only the selected
coordinate-realization masks. The learned geometry heads, topology/contact
losses, and message-passing masks remain unchanged. The benchmark variant
`full_msa_to_face_aux_closure` is architecturally identical to
`full_msa_to_face`; the intervention is the scheduled auxiliary realization
weight, not a new parameterized module.

Decision rule: run a 500-step Runpod gate at crop 256 / MSA depth 64 with the
E15 selected coordinate and boundary-distance weights, `simplex_aux_weight`
annealed from `1.0` to `0.5`, and `simplex_cell_closure_weight` ramped from
`0.0` to `0.5` over steps 250-500. Continue only if the run recovers above
the weak E43-E46 band and approaches the E22/E25/E30 early range without
late radius collapse.

Result: reject. E47's pre-ramp step-250 checkpoint reached
`val_lddt_ca=0.2466`, FoldScore `0.2070`, `val_ca_drmsd=14.5113`, and
predicted/true C-alpha radius of gyration `7.1693 / 15.4034`. After the
closure ramp reached `0.5`, the final step-500 checkpoint fell to
`val_lddt_ca=0.2262`, FoldScore `0.2182`, `val_ca_drmsd=15.7332`, and radius
of gyration `5.5581 / 15.4034`. Auxiliary-only flag closure avoids directly
suppressing messages, but it still weakens the same early band and does not
prevent late collapse.

### E48: Adaptive Local-to-Global Topology Curriculum

Status: completed on Runpod.

Hypothesis: recent failures are not caused by too little closure or too few
static selected cells. The sparse complex may need a training curriculum for
its neighborhood operator: start with a small local manifold scaffold so early
faces/tetras are coherent, then anneal back to the learned/global E09 selector
once pair/MSA topology has begun to stabilize.

Mechanism: add training-time overrides for the local selected-neighbor
scaffold. The implementation schedules `simplex_local_neighbor_k` from `4`
to `0` over the 500-step gate while keeping the E09/E15
`full_msa_to_face_topology_curriculum` pathway, selected face/tetra
coordinate realization, and selected boundary-distance losses. Evaluation and
inference keep the static model config, so this is a training curriculum on
the selected-complex construction operator, not a dense coordinate metric.

Local validation: py_compile passed for the changed model/trainer/runner
files; focused tests passed for adapter local-slot overrides, schedule math,
training-only model inputs, benchmark variant parsing, and parameter budget.
Affected suites `tests/test_simplex.py`, `tests/test_nanofold_public_benchmarks.py`,
and `tests/test_trainer.py` passed. Parameter audit: AF2-medium pair-only
`3,106,642`, SimplexFold medium `3,106,690`, and E48 topology curriculum
`3,106,690`.

Decision rule: run a 500-step Runpod gate at crop 256 / MSA depth 64 with
the E15 auxiliary anneal and compare against E47 and the stronger E22/E25/E30
early range. Continue only if the run recovers the early lDDT/FoldScore band
without late radius collapse.

Result: reject. The pre-ramp step-250 checkpoint reached only
`val_lddt_ca=0.1002`, FoldScore `0.1827`, and predicted/true C-alpha radius
of gyration `13.9544 / 15.4034`. After the neighborhood curriculum annealed
`simplex_local_neighbor_k` from `4` to `0` and the simplex auxiliary weight
from `1.0` to `0.5`, the final step-500 checkpoint recovered to
`val_lddt_ca=0.2274`, FoldScore `0.2191`, `val_ca_drmsd=15.7749`, and radius
of gyration `5.5326 / 15.4034`. The local scaffold did not recover the
stronger E22/E25/E30 early band and still ended with strong coordinate
collapse.

### E49: Outer-Edge Selected-Cell Communication

Status: completed on Runpod.

Hypothesis: selected face/tetra cells should communicate through boundary
edges that connect one selected cell to another, preserving multiple
edge-level geometric relationships instead of collapsing cell pairs into
coarse superedges. This follows Topotein's outer-edge neighborhoods and keeps
the model in the combinatorial-complex view.

Mechanism: build an incidence-aware selected-cell message pass where a face
or tetra gathers messages from boundary edges that leave its anchor/coface
and enter another selected cell. Normalize by edge incidence and return the
updated higher-rank state to the ordinary pair/single residual path. Start
with a zero-parameter or very small-parameter residual before spending the
remaining 5% budget.

Implementation: add `simplex_outer_edge_context_scale`, a directed
outer-edge context pass distinct from the older shared-boundary face averaging
variant. For each selected face/tetra cell, gather the selected topology
edges that originate from a cell vertex and terminate outside the cell, plus
the reverse directed pair states. The pooled outgoing/incoming pair context
updates the face or tetra cochain before the ordinary boundary readout.
This follows Topotein's `B^{2->0} B^{0->1} - B^{2->1}` idea more closely
than requiring two selected cells to share an existing boundary edge.

Local validation: focused tests passed for the outer-edge context exclusion
rule, adapter behavior, runner variant parsing, and parameter headroom.
Affected suites `tests/test_simplex.py`, `tests/test_nanofold_public_benchmarks.py`,
and `tests/test_trainer.py` passed. Parameter audit: AF2-medium pair-only
`3,106,642`, SimplexFold medium `3,106,690`, and E49
`3,183,282` parameters.

Decision rule: run a 500-step Runpod gate at crop 256 / MSA depth 64 with
the E15 selected coordinate and boundary-distance losses plus auxiliary
annealing. Continue only if the outer-edge context recovers at least the
E22/E25/E30 early-validation band without worsening the final radius
collapse.

Result: reject. The step-250 checkpoint reached `val_lddt_ca=0.2421`,
FoldScore `0.2118`, `val_ca_drmsd=15.0993`, and predicted/true C-alpha
radius of gyration `6.3380 / 15.4034`. The final step-500 checkpoint improved
to `val_lddt_ca=0.2695`, FoldScore `0.2429`, `val_ca_drmsd=14.5377`, and
radius of gyration `6.7858 / 15.4034`. Directed outer-edge context is better
than E47/E48 final and roughly comparable to several weak pilots, but it does
not recover the E22/E25/E30 early band or approach E15.

### E50: Selected Boundary Expansion Hinge

Status: completed on Runpod.

Hypothesis: the repeated low-radius checkpoints are a failure to realize the
selected simplicial complex, not just a bad global coordinate scale. The
model selects faces and tetrahedra, predicts coordinates, and already tries
to match selected boundary distances symmetrically. A one-sided realization
constraint may help the complex avoid degenerate contractions while leaving
over-expanded early structures free to be corrected by FAPE, lDDT-style
terms, and selected distance losses.

Mechanism: add zero-parameter face and tetra expansion losses. For every
selected face/tetra boundary edge, compute the log-scaled true and predicted
C-alpha edge lengths and apply a smooth one-sided hinge only when the
predicted boundary edge is shorter than the true edge. The loss acts only on
edges induced by selected higher-rank cells, so it is a boundary-realization
objective for the learned 2-/3-complex rather than a generic all-pairs
radius or lDDT hack.

Implementation: add `simplex_face_coordinate_expansion_weight`,
`simplex_tetra_coordinate_expansion_weight`, and
`simplex_coordinate_expansion_tolerance`, plus the runner variant
`full_msa_to_face_expansion_hinge`. The variant is architecturally identical
to `full_msa_to_face`, so it should remain at the SimplexFold medium budget
of `3,106,690` parameters.

Local validation: py_compile passed for the changed model/trainer/runner
files. Focused tests passed for the one-sided loss, AlphaFoldLoss override
plumbing, runner variant parsing, CLI flags, loss builder forwarding, and
zero-parameter budget. Affected suites `tests/test_simplex.py`,
`tests/test_nanofold_public_benchmarks.py`, and `tests/test_trainer.py`
passed, and full `python -m pytest -q` passed. Parameter audit: AF2-medium
pair-only `3,106,642`, SimplexFold medium `3,106,690`, and E50
`3,106,690` parameters.

Decision rule: run a 500-step Runpod gate at crop 256 / MSA depth 64 with
the E15 selected coordinate and boundary-distance losses, expansion weights
`0.5/0.5`, and `simplex_aux_weight` annealed from `1.0` to `0.5`. Continue
only if the run improves over E49 while increasing predicted C-alpha radius
toward the validation target geometry instead of producing the usual
late-collapse signature.

Result: reject. The step-250 checkpoint had the desired expansion-side
effect but weak validation accuracy: `val_lddt_ca=0.1593`, FoldScore
`0.1988`, `val_ca_drmsd=14.2261`, and predicted/true C-alpha radius of
gyration `10.6522 / 15.4034`. By step 500, lDDT recovered to
`val_lddt_ca=0.2731`, FoldScore `0.2334`, `val_ca_drmsd=14.7809`, and
radius `6.6087 / 15.4034`. The hinge helped early expansion but did not
prevent final collapse or beat E49. This suggests the next branch should not
just strengthen coordinate expansion; it needs to make the expanded selected
boundary geometry feed back into the learned topology/readout path.

### E51: Expansion Hinge With Structure Readout

Status: completed on Runpod.

Hypothesis: E50 failed because the expansion objective supervised selected
simplex boundaries without putting that expanded boundary information on the
path that the structure module uses to place atoms. If the same selected
face/tetra states contribute pair/single readouts to the structure module,
the contraction penalty may become a useful topological realization signal
instead of a side loss that the trunk can ignore.

Mechanism: reuse the existing `full_msa_to_face_structure_readout` variant
and run it with the E50 face/tetra expansion hinge plus the E15 selected
coordinate and boundary-distance losses. The readout keeps the explicit
2-/3-cell boundary messages in the topology stream and injects their
pair/single summaries before the structure module.

Decision rule: run a 500-step Runpod gate at crop 256 / MSA depth 64 with
the E50 expansion weights `0.5/0.5`, E15 auxiliary annealing, and
`simplex_structure_readout_scale=0.25`. Continue only if the final lDDT and
radius both improve over E50 and the run approaches or beats the E49/E22
pilot band.

Result: reject. The step-250 checkpoint reached `val_lddt_ca=0.2375`,
FoldScore `0.2089`, `val_ca_drmsd=14.7756`, and predicted/true C-alpha
radius of gyration `7.0810 / 15.4034`. The final step-500 checkpoint fell to
`val_lddt_ca=0.2272`, FoldScore `0.2233`, `val_ca_drmsd=15.7161`, and radius
`5.7622 / 15.4034`. Adding structure readout erased E50's early expansion
effect and underperformed E50, E49, and the stronger E22/E25/E30 pilots.
Do not pursue this combination without a more selective gate or a different
readout mechanism.

### E52: Selected Cell Dropout

Status: completed on Runpod.

Hypothesis: E15 is still the best branch, and E16/E17 suggest that simply
continuing or further relaxing its auxiliary scaffold does not break the
lDDT plateau. The selected complex may be too brittle: early learned topology
chooses a fixed set of face/tetra cells, and the trunk can over-rely on noisy
higher-order cells that later encourage collapsed geometry. Randomly thinning
the selected 2-/3-cell cochains during training should make the pair/single
trunk learn from many subcomplexes of the same selected boundary graph.

Mechanism: add `simplex_cell_dropout` and the variant
`full_msa_to_face_cell_dropout`. During training only, the adapter randomly
drops selected face and tetra masks before face/tetra message passing,
auxiliary heads, and selected-boundary losses consume them. Evaluation uses
the full selected complex. This is a zero-parameter topological regularizer:
the stochasticity acts on explicit cells in the learned complex, not on dense
residue pairs or global coordinate scale.

Decision rule: run a 500-step Runpod gate at crop 256 / MSA depth 64 with
the E15 selected coordinate and boundary-distance losses and
`simplex_aux_weight` annealed from `1.0` to `0.5`. Continue only if the run
recovers the E22/E25/E30 early band while preserving FoldScore/radius better
than E49-E51. If it only matches the weak post-E43 pilot band, reject and
return to longer E15/effective-batch-8 optimization rather than adding more
auxiliary losses.

Result: reject. Step 250 reached `val_lddt_ca=0.2293`, FoldScore `0.2169`,
`val_ca_drmsd=15.7319`, and predicted/true C-alpha radius of gyration
`5.5469 / 15.4034`. Step 500 recovered to `val_lddt_ca=0.2630`, FoldScore
`0.2301`, `val_ca_drmsd=14.2399`, and radius `7.2057 / 15.4034`, but this
remains below E49/E50 final and far below the stronger E22/E25/E30 early
band. Cell dropout is too destructive at this strength and should not be the
next main branch.

### E53: Longer Effective-Batch-8 E15 Scaffold

Status: completed on Runpod.

Hypothesis: E25 showed that effective batch 8 is operationally viable but did
not improve the 500-step point. The final objective, however, requires a
30,000-step effective-batch-8 confirmation, and E09/E15-style curves often
improve substantially after the first 500 optimizer steps. Before adding more
architecture or auxiliary losses, run a longer effective-batch-8 gate on the
best selected simplex scaffold.

Mechanism: use `full_msa_to_face` with the E09/E15 selected face/tetra
coordinate and boundary-distance realization losses, `batch_size=1`, and
`grad_accum_steps=8`. This changes only optimizer batch statistics; the
architecture remains the persistent face/tetra complex with MSA-to-face
updates and selected boundary realization.

Decision rule: run 1000 optimizer steps at crop 256 / MSA depth 64, evaluating
at steps 500 and 1000. Continue toward a longer effective-batch-8 run only if
step 1000 improves clearly over E25 step 500 (`val_lddt_ca=0.2946`) or shows
substantially better FoldScore/radius without losing lDDT.

Result: keep for continuation. Step 500 reached `val_lddt_ca=0.2807`,
FoldScore `0.2514`, `val_ca_drmsd=14.9835`, and predicted/true C-alpha
radius of gyration `6.1825 / 15.4034`, below E25 lDDT but with better
FoldScore. Step 1000 rebounded to `val_lddt_ca=0.3480`, FoldScore `0.2729`,
`val_ca_drmsd=12.6378`, and radius `8.5184 / 15.4034`. This is the strongest
short effective-batch-8 result so far and close enough to E15 to justify an
E15-style auxiliary anneal continuation.

### E54: Effective-Batch-8 Auxiliary Anneal Continuation

Status: completed on Runpod.

Hypothesis: E53 shows that effective-batch-8 training catches up after 1000
optimizer steps, while E15 showed that reducing selected-simplex auxiliary
pressure from `1.0` to `0.5` can improve lDDT once the scaffold has learned a
useful geometry prior. Applying the E15 anneal to the E53 checkpoint should
test whether the target optimizer regime can match or beat E15 earlier.

Mechanism: resume E53 at step 1000 and continue to step 2000 with
`batch_size=1`, `grad_accum_steps=8`, `full_msa_to_face`, and the same
selected coordinate/boundary-distance losses. Ramp only `simplex_aux_weight`
from `1.0` to `0.5` over steps 1000-1500. This remains a selected
simplex-realization curriculum rather than a generic coordinate loss.

Decision rule: keep and extend toward 3000+ effective-batch-8 steps only if
the continuation beats E53's `val_lddt_ca=0.3480` or materially improves
FoldScore/dRMSD without losing lDDT. If it falls below the E53 plateau, do
not spend on a 30k confirmation yet.

Result: keep for continuation. Step 1500 fell to `val_lddt_ca=0.3331` while
the auxiliary weight reached `0.5`, but FoldScore improved to `0.2891`.
Holding the annealed scaffold to step 2000 recovered lDDT to `0.3539`,
FoldScore to `0.3241`, `val_ca_drmsd` to `11.9339`, and predicted/true
C-alpha radius of gyration to `9.2409 / 15.4034`. This nearly ties E15's best
lDDT and exceeds E15's FoldScore, so the effective-batch-8 path should
continue with `simplex_aux_weight=0.5`.

### E55: Effective-Batch-8 Aux-0.5 Continuation

Status: completed on Runpod.

Hypothesis: E54 shows the E15-style auxiliary anneal initially disrupts lDDT
but then recovers to near-best validation while improving FoldScore and
dRMSD. Continuing with `simplex_aux_weight=0.5` may now let the structure
module consolidate the selected higher-order scaffold under the target
effective-batch-8 optimizer regime.

Mechanism: resume E54 at step 2000 and continue to step 3000 with
`full_msa_to_face`, `batch_size=1`, `grad_accum_steps=8`, selected
coordinate weights `1.0/1.0`, selected boundary-distance weights `0.5/0.5`,
and constant `simplex_aux_weight=0.5`.

Decision rule: keep and extend only if step 2500 or 3000 beats E15's
`val_lddt_ca=0.3556` or preserves comparable lDDT with clearly better
FoldScore/dRMSD. If lDDT drifts down as in E17, stop the continuation branch.

Result: keep as the new current best. Step 2500 dipped to
`val_lddt_ca=0.3424`, but FoldScore improved to `0.3379`, dRMSD to
`11.0985`, and predicted/true C-alpha radius of gyration to
`10.3390 / 15.4034`. Step 3000 recovered to `val_lddt_ca=0.3604`, beating
E15's best `0.3556`, with FoldScore `0.3451`, `val_ca_drmsd=11.3280`, and
radius `10.0507 / 15.4034`. This makes the effective-batch-8 aux-0.5
continuation the leading branch.

### E56: Effective-Batch-8 Aux-0.5 To 4000

Status: completed on Runpod.

Hypothesis: E55 beat E15 at step 3000 while preserving the improved
FoldScore/global-geometry trend. Continuing one more 1000-step block should
test whether the branch is still climbing or beginning the E17-style lDDT
drift.

Mechanism: resume E55 at step 3000 and continue to step 4000 with the same
`full_msa_to_face` architecture, effective batch 8, selected coordinate and
boundary-distance losses, and constant `simplex_aux_weight=0.5`.

Decision rule: keep only if step 3500 or 4000 improves over E55's
`val_lddt_ca=0.3604` or preserves lDDT within noise while continuing to
improve FoldScore/dRMSD. If lDDT falls back toward E54/E17 behavior, stop and
analyze the E55 checkpoint before longer training.

Result: stop for the lDDT objective. Step 3500 reached `val_lddt_ca=0.3562`,
FoldScore `0.3464`, `val_ca_drmsd=10.7120`, and radius
`10.9217 / 15.4034`. Step 4000 reached `val_lddt_ca=0.3575`, FoldScore
`0.3478`, `val_ca_drmsd=10.9804`, and radius `10.3192 / 15.4034`. This
continues improving aggregate/global geometry but does not beat E55's
`val_lddt_ca=0.3604`, so longer constant-aux-0.5 continuation should pause.
The next branch should analyze or resume from the E55 checkpoint rather than
spending immediately on a 30k confirmation.

### E57: Aux-0.75 Rewarm From E55

Status: completed on Runpod.

Hypothesis: E56 shows that continuing E55 with constant
`simplex_aux_weight=0.5` improves FoldScore and dRMSD but does not preserve
the lDDT peak. A modest selected-simplex auxiliary rewarm to `0.75` may keep
the higher-order face/tetra realization constraints active enough to preserve
local C-alpha agreement while still allowing the structure module to use the
better global geometry learned by the effective-batch-8 branch.

Mechanism: resume the E55 checkpoint at step 3000 and continue to step 4000
with `full_msa_to_face`, `batch_size=1`, `grad_accum_steps=8`, selected
coordinate weights `1.0/1.0`, selected boundary-distance weights `0.5/0.5`,
and constant `simplex_aux_weight=0.75`. This keeps the architecture and losses
inside the selected residue-edge-face-tetra complex; it is not a generic dense
lDDT hack.

Decision rule: keep only if step 3500 or 4000 beats E55's
`val_lddt_ca=0.3604` or beats E56's FoldScore/dRMSD while staying at least
within small noise of E55's lDDT. If the rewarm worsens lDDT without a clear
geometry tradeoff, return to architecture-level topology changes rather than
more scalar loss tuning.

Result: reject for the lDDT objective. Step 3500 reached
`val_lddt_ca=0.3395`, FoldScore `0.3504`, `val_ca_drmsd=10.6852`, and
predicted/true C-alpha radius of gyration `11.5354 / 15.4034`. Step 4000
recovered only to `val_lddt_ca=0.3465`, with FoldScore `0.3495`,
`val_ca_drmsd=10.7091`, and radius `10.8574 / 15.4034`. The stronger
selected-simplex auxiliary pressure improved global/FoldScore behavior but
damaged local C-alpha lDDT relative to E55. This argues against more scalar
auxiliary rewarming and points back to architecture-level topology changes.

### E58: Resume-Compatible Outer-Edge Context From E55

Status: stopped early on Runpod.

Hypothesis: the reference PDFs, especially Topotein, argue that effective
protein TDL requires dedicated cross-rank and within-rank communication over
the constructed complex. E57 shows that simply increasing selected-cell loss
pressure is not enough. Resuming E55 while activating directed outer-edge
context should let selected face/tetra cochains exchange information through
boundary/interior pair edges, preserving edge-level geometry rather than
collapsing higher-rank communication into another scalar loss.

Mechanism: initialize from the E55 checkpoint at step 3000 using the checkpoint
variant name `full_msa_to_face`, effective batch 8, selected coordinate
weights `1.0/1.0`, selected boundary-distance weights `0.5/0.5`, and constant
`simplex_aux_weight=0.5`. Activate the existing directed outer-edge context
architecture with a model-config override,
`--simplex-outer-edge-context-scale 0.25`, so the checkpoint remains
resume-compatible with E55 while the next forward passes use Topotein-style
outer-edge communication.
Because this architecture adds context-path parameters relative to E55, use
`--resume-model-weights-only`: matching E55 tensors are loaded, new tensors are
initialized fresh, and optimizer state is restarted while training continues
from the E55 step count.

Decision rule: keep only if step 3500 or 4000 beats E55's
`val_lddt_ca=0.3604`, or preserves lDDT while materially improving FoldScore
or dRMSD. Reject if it follows E57's pattern of better global geometry with
worse local lDDT.

Launch: E58 is running on owned Runpod H100 pod `714wc1nzy3t8qz` from commit
`41af00a`. Launch audit passed: public train/val/all counts are
`10000/1000/11000`, no hidden manifest was staged, feature/label cache counts
are `11000/11000`, encoded missing paths are `0`, FoldScore import works,
CUDA reports `NVIDIA H100 80GB HBM3`, and parameters are `3,183,282`
(+2.47% versus AF2-medium pair-only `3,106,642`). Do not write E58 to
`EXPERIMENT_RESULTS.md` until the Runpod run returns.

Result: reject after the step-3500 validation point. E58 reached
`val_lddt_ca=0.3419`, FoldScore `0.3507`, `val_ca_drmsd=10.9020`, and
predicted/true C-alpha radius of gyration `11.1250 / 15.4034`. It gives the
best FoldScore so far, but the primary lDDT is far below E55 and follows
E57's pattern of better global geometry with damaged local agreement. The run
was stopped early at the step-3500 checkpoint and the owned pod was deleted.

### E59: Damped Outer-Edge Context From E55

Status: completed on Runpod.

Hypothesis: E58 shows directed outer-edge context has useful global-geometry
signal, but a scale of `0.25` is too disruptive when the context modules are
freshly initialized from an E55 weight checkpoint. A much smaller context
scale may let selected face/tetra cochains receive Topotein-style outer-edge
information as a weak correction while preserving E55's local C-alpha lDDT.

Mechanism: initialize from the E55 checkpoint with `--resume-model-weights-only`
and variant name `full_msa_to_face`, but set
`--simplex-outer-edge-context-scale 0.05`. Run only to step 3500 as a gate
under the same effective-batch-8, crop 256, MSA depth 64, selected coordinate
weights `1.0/1.0`, selected boundary-distance weights `0.5/0.5`, and
`simplex_aux_weight=0.5` settings.

Decision rule: keep only if the step-3500 lDDT beats or stays very close to
E55's `0.3604` while preserving E58's FoldScore/dRMSD improvement. Reject if
it remains in the E57/E58 lDDT band.

Launch: E59 ran on owned Runpod H100 pod `n5dtdxgjgk81de` from commit
`6f9750c`. Launch audit passed: public train/val/all counts were
`10000/1000/11000`, no hidden manifest was staged, feature/label cache counts
were `11000/11000`, encoded missing paths were `0`, FoldScore import worked,
CUDA reported `NVIDIA H100 80GB HBM3`, and parameters were `3,183,282`
(+2.47% versus AF2-medium pair-only `3,106,642`).

Result: reject for the primary objective. E59 completed at step 3500 with
`val_lddt_ca=0.3500`, FoldScore `0.3516`, `val_ca_drmsd=10.9502`, and
predicted/true C-alpha radius of gyration `11.1978 / 15.4034`. The weaker
outer-edge context path improves substantially over E58's lDDT and sets the
best FoldScore so far, but it still remains below E55's `0.3604` lDDT.
This suggests the context path is useful but still needs a less abrupt
integration into the selected cell complex.

### E60: Scheduled Damped Outer-Edge Context From E55

Status: completed on Runpod.

Hypothesis: E58 and E59 show that Topotein-style outer-edge context improves
global FoldScore, but switching freshly initialized context modules on at a
fixed scale still disrupts local C-alpha agreement. A runtime scale ramp from
zero to the same damped `0.05` endpoint should let the resumed E55 checkpoint
adapt to the new cochain communication route over the 3000-3500 gate.

Mechanism: add a training-time
`simplex_outer_edge_context_runtime_scale` schedule. The model config
`simplex_outer_edge_context_scale` still allocates the face/tetra outer-edge
context modules and controls validation-time scale, while the runtime
override gates the contribution during training. E60 should use
`--simplex-outer-edge-context-scale 0.05`,
`--simplex-outer-edge-context-runtime-scale 0.0`,
`--simplex-outer-edge-context-runtime-scale-final 0.05`,
`--simplex-outer-edge-context-runtime-scale-ramp-start-step 3000`, and
`--simplex-outer-edge-context-runtime-scale-ramp-steps 500`.

Decision rule: keep only if the step-3500 lDDT beats or stays very close to
E55's `0.3604` while preserving E59's FoldScore/dRMSD improvement. Reject if
it remains below the E55/E56 lDDT band.

Launch: E60 ran on owned Runpod H100 pod `yzy3zi29gzbfj4` from commit
`ede843b`. Launch audit passed: public train/val/all counts are
`10000/1000/11000`, no hidden manifest/path was staged, feature/label cache
counts are `11000/11000`, encoded missing paths are `0`, FoldScore import
works, CUDA reports `NVIDIA H100 80GB HBM3`, and parameters are `3,183,282`
(+2.47% versus AF2-medium pair-only `3,106,642`). The runtime scale audit
confirmed `0.0` at step 3000, `0.025` at step 3250, and `0.05` at step 3500.

Result: reject. E60 completed at step 3500 with
`val_lddt_ca=0.3462`, FoldScore `0.3431`, `val_ca_drmsd=10.9235`, and
predicted/true C-alpha radius of gyration `10.8522 / 15.4034`. The runtime
ramp did not preserve E55's `0.3604` lDDT and gave up the FoldScore advantage
seen in E59. Artifacts were copied locally, and the owned E60 pod was stopped
and deleted after the returned result was recorded.

Validation:

- `python -m py_compile minalphafold/trainer.py minalphafold/simplex.py minalphafold/evoformer.py minalphafold/model.py scripts/run_nanofold_public_benchmarks.py`
- `python -m pytest tests/test_nanofold_public_benchmarks.py::test_model_config_override_flags_are_accepted_by_cli_parser tests/test_nanofold_public_benchmarks.py::test_outer_edge_context_runtime_scale_ramps_and_enters_model_inputs tests/test_simplex.py::test_outer_edge_context_runtime_scale_gates_context_path`
- `python -m pytest tests/test_nanofold_public_benchmarks.py tests/test_simplex.py::test_outer_edge_context_runtime_scale_gates_context_path tests/test_simplex.py::test_edge_frame_message_scale_changes_pair_readout_within_adapter tests/test_trainer.py::test_simplexfold_medium_param_matched_matches_af2_medium_budget`

### E61: Scheduled Edge-Frame Boundary Messages From E55

Status: completed on Runpod.

Hypothesis: the reference PDFs and Topotein both point to edge-centric frames
as the clean way to keep higher-rank cell geometry useful without collapsing
everything into a coarse cell summary. E40 tested this path from scratch and
too early, where it was weak. A more relevant test is resume-compatible:
start from E55, allocate the edge-frame message modules at a small endpoint,
and ramp their contribution during the 3000-3500 continuation so the pair
stream can adapt gradually.

Mechanism: add a training-time
`simplex_edge_frame_message_runtime_scale` schedule. The model-config
`simplex_edge_frame_message_scale` still allocates the face/tetra
boundary-edge frame MLPs and controls validation-time scale; the runtime
override gates the training-time contribution. A planned E61 gate should use
`full_msa_to_face`, `--simplex-edge-frame-message-scale 0.05`,
`--simplex-edge-frame-message-runtime-scale 0.0`,
`--simplex-edge-frame-message-runtime-scale-final 0.05`,
`--simplex-edge-frame-message-runtime-scale-ramp-start-step 3000`, and
`--simplex-edge-frame-message-runtime-scale-ramp-steps 500` from the E55
checkpoint if E60 is rejected.

Instrumentation: add selected-boundary geometry metrics to validation for
future runs. For selected faces and tetras, the runner now reports
boundary-edge length MAE/RMSE, contraction fraction, and boundary lDDT on the
model's own selected sparse complex. These are diagnostics only; they do not
change training loss or official scoring.

Decision rule: keep only if the step-3500 lDDT beats or stays very close to
E55's `0.3604`; otherwise reject and use the diagnostics to decide whether
the selected complex is under-realized or over-coupled.

Launch: E61 ran on owned Runpod H100 NVL pod `h2dvec04rxyoxe` from commit
`7823038`. Launch audit passed after restaging only public assets:
public train/val/all counts are `10000/1000/11000`, no hidden manifest/path is
staged, feature/label cache counts are `11000/11000`, encoded missing paths
are `0`, the E55 checkpoint is present, FoldScore import works, CUDA reports
`NVIDIA H100 NVL`, and the edge-frame model has `3,154,242` parameters
(+1.53% versus AF2-medium pair-only `3,106,642`). The runtime scale audit
confirmed `0.0` at step 3000, `0.025` at step 3250, and `0.05` at step 3500.
The run resumed E55 at step 3000 with weights-only loading; the newly added
edge-frame tensors were initialized fresh as expected.

Result: reject. E61 completed at step 3500 with `val_lddt_ca=0.3456`,
FoldScore `0.3471`, `val_ca_drmsd=10.7730`, and predicted/true C-alpha radius
of gyration `11.1613 / 15.4034`. The edge-frame schedule improved dRMSD and
global expansion relative to E55, but it did not preserve E55's local lDDT.
Artifacts were copied locally, and the owned E61 pod was stopped and deleted
after the returned result was recorded.

Validation:

- `python -m py_compile minalphafold/trainer.py minalphafold/simplex.py minalphafold/evoformer.py minalphafold/model.py scripts/run_nanofold_public_benchmarks.py`
- `python -m pytest tests/test_nanofold_public_benchmarks.py::test_model_config_override_flags_are_accepted_by_cli_parser tests/test_nanofold_public_benchmarks.py::test_runtime_simplex_message_scales_ramp_and_enter_model_inputs tests/test_simplex.py::test_edge_frame_message_runtime_scale_gates_pair_readout tests/test_simplex.py::test_edge_frame_message_scale_changes_pair_readout_within_adapter`
- `python -m pytest tests/test_nanofold_public_benchmarks.py tests/test_simplex.py::test_edge_frame_message_runtime_scale_gates_pair_readout tests/test_simplex.py::test_edge_frame_message_scale_changes_pair_readout_within_adapter tests/test_simplex.py::test_outer_edge_context_runtime_scale_gates_context_path tests/test_trainer.py::test_simplicial_edge_frame_messages_stay_within_medium_budget`
- `python -m py_compile scripts/run_nanofold_public_benchmarks.py`
- `python -m pytest tests/test_nanofold_public_benchmarks.py::test_simplex_boundary_geometry_metrics_report_selected_edge_errors tests/test_nanofold_public_benchmarks.py::test_simplex_topology_metrics_report_boundary_reuse`
- `python -m pytest tests/test_nanofold_public_benchmarks.py`

### E62: Scheduled Hodge Face Residual From E55

Status: completed on Runpod.

Hypothesis: the reference material emphasizes that topological networks should
move information through incidence maps across ranks. E42 tested the static
Hodge face residual too early from scratch and failed. A resume-compatible
version should test the same lower/upper face adjacency after E55 has already
learned the selected complex: ramp lower adjacency through shared boundary
edges plus upper adjacency through selected tetra cofaces over steps
3000-3500.

Mechanism: add a training-time `simplex_hodge_face_runtime_scale` schedule.
The model config `simplex_hodge_face_update_scale` controls validation-time
scale, while the runtime override gates the contribution during training. The
path adds no parameters because it reuses selected face states, shared
boundary-edge averaging, and tetra co-boundary averaging. A planned E62 gate
should use `full_msa_to_face`, `--simplex-hodge-face-update-scale 0.05`,
`--simplex-hodge-face-runtime-scale 0.0`,
`--simplex-hodge-face-runtime-scale-final 0.05`,
`--simplex-hodge-face-runtime-scale-ramp-start-step 3000`, and
`--simplex-hodge-face-runtime-scale-ramp-steps 500` from the E55 checkpoint if
E61 is rejected.

Decision rule: keep only if the step-3500 lDDT beats or stays very close to
E55's `0.3604`; otherwise reject and use selected-boundary diagnostics to
decide whether adjacency mixing is over-smoothing the sparse complex.

Launch: E62 ran on owned Runpod H100 NVL pod `39s6arzja95amz` from
commit `4517f98`. A first replacement pod, `f3j3v4qd4f6w8w`, never exposed
SSH and was stopped/deleted before any data staging or training. Final launch
audit on `39s6arzja95amz` passed: public train/val/all counts are
`10000/1000/11000`, no hidden manifest/path is staged, feature/label cache
counts are `11000/11000`, encoded missing paths are `0`, the E55 checkpoint
is present, FoldScore import works, CUDA reports `NVIDIA H100 NVL`, and the
Hodge model has `3,106,690` parameters (+0.0015% versus AF2-medium pair-only
`3,106,642`). The runtime scale audit confirmed `0.0` at step 3000, `0.025`
at step 3250, and `0.05` at step 3500. The run resumed E55 at step 3000 with
weights-only loading and initialized `0` new/missing tensors, as expected for
a zero-parameter Hodge residual.

Result: reject. E62 completed at step 3500 with
`val_lddt_ca=0.3468`, FoldScore `0.3450`, `val_ca_drmsd=10.9016`, and
predicted/true C-alpha radius of gyration `10.7278 / 15.4034`. The selected
face/tetra boundary lDDT diagnostics were `0.4829` / `0.4694`, slightly above
E61's edge-frame run, but the main validation C-alpha lDDT stayed well below
E55's `0.3604`. Artifacts were copied locally, and the owned E62 pod was
stopped and deleted after the returned result was recorded.

Validation:

- `python -m py_compile minalphafold/trainer.py minalphafold/simplex.py minalphafold/evoformer.py minalphafold/model.py scripts/run_nanofold_public_benchmarks.py`
- `python -m pytest tests/test_nanofold_public_benchmarks.py::test_model_config_override_flags_are_accepted_by_cli_parser tests/test_nanofold_public_benchmarks.py::test_runtime_simplex_message_scales_ramp_and_enter_model_inputs tests/test_simplex.py::test_hodge_face_adapter_scale_changes_outputs_without_new_parameters tests/test_trainer.py::test_model_inputs_add_training_only_simplex_curricula`
- `python -m pytest tests/test_nanofold_public_benchmarks.py tests/test_simplex.py::test_hodge_face_adapter_scale_changes_outputs_without_new_parameters tests/test_simplex.py::test_edge_frame_message_runtime_scale_gates_pair_readout tests/test_simplex.py::test_outer_edge_context_runtime_scale_gates_context_path tests/test_trainer.py::test_model_inputs_add_training_only_simplex_curricula tests/test_trainer.py::test_simplicial_hodge_face_update_adds_no_parameters`

### E63: Selected-Boundary lDDT Curriculum From E55

Status: completed on Runpod.

Hypothesis: E61 and E62 both improve aspects of global geometry while leaving
the learned selected complex with weak boundary distance preservation: selected
face/tetra boundary lDDT remains below `0.5`, and roughly three quarters of
selected boundary edges are contracted. The next topology-native objective is
not a generic all-pairs lDDT loss. It should supervise only the boundary
1-skeleton induced by the model-selected face and tetra cells, asking the
explicit simplicial complex to realize its own selected edges more faithfully.

Mechanism: resume E55 with `full_msa_to_face`, keep the selected
coordinate/selected boundary-distance weights from E55, and add a small
selected-boundary lDDT loss on both faces and tetras. Start with a conservative
static weight gate unless a loss-specific ramp is implemented first:
`--simplex-face-boundary-lddt-weight 0.05` and
`--simplex-tetra-boundary-lddt-weight 0.05`.

Decision rule: keep only if the step-3500 validation lDDT beats or stays very
close to E55's `0.3604` while improving selected-boundary lDDT and contraction
diagnostics. Reject if it behaves like E19/E20 and trades local C-alpha lDDT
for a narrow auxiliary gain.

Launch: E63 ran on owned Runpod H100 NVL pod `0hm1lpiaqqx21a` from
commit `6bb49f8`. Launch audit passed after staging only public assets:
public train/val/all counts are `10000/1000/11000`, remote manifest files are
exactly `all.txt`, `train.txt`, and `val.txt`, no hidden manifest/path is
present, feature/label cache counts are `11000/11000`, encoded missing paths
are `0`, the E55 checkpoint is present, FoldScore import works, CUDA reports
`NVIDIA H100 NVL`, and the model has `3,106,690` parameters (+0.0015% versus
AF2-medium pair-only `3,106,642`).

Result: keep for confirmation. E63 completed at step 3500 with
`val_lddt_ca=0.3611`, FoldScore `0.3576`, `val_ca_drmsd=10.6815`, and
predicted/true C-alpha radius of gyration `11.4310 / 15.4034`. It is only a
small lDDT gain over E55's `0.3604`, but it also improves FoldScore and the
selected-boundary diagnostics: face/tetra boundary lDDT rose to
`0.5208` / `0.5065`, while contraction fractions fell to
`0.6897` / `0.6913`. Artifacts were copied locally, and the owned E63 pod was
stopped and deleted after the returned result was recorded.

### E64: E63 Confirmation To 32k Examples

Status: completed on Runpod.

Hypothesis: E63 is the first tested branch to improve the primary C-alpha
lDDT while also improving selected-boundary realization. Because the margin is
small and E56 showed that plain E55 continuation to 4000 steps regresses
lDDT, the next test should continue E63's selected-boundary lDDT objective to
step 4000, i.e. 32,000 effective training examples at batch 8.

Mechanism: resume E63 from its step-3500 checkpoint, keep
`simplex_face_boundary_lddt_weight=0.05` and
`simplex_tetra_boundary_lddt_weight=0.05`, and run the same public validation
gate to step 4000.

Decision rule: keep only if the step-4000 lDDT stays above E55/E63's band or
improves selected-boundary realization without the E56-style lDDT regression.

Launch: E64 ran on owned Runpod B300 pod `ow3ex8z84jypbs` from commit
`b12093d`. Earlier owned E64 attempts failed before returning a result:
H200 NVL pod `4g78gy2fbgl5o7` and A100 SXM pod `r64q7czrpsaax4` hit remote
`/workspace` network-volume I/O errors while copying public feature NPZs, and
A100 SXM pod `76h4drrq0mhbxp` used local storage but became CPU-bound with no
step-4000 artifact after more than an hour. All three were stopped/deleted.
The successful B300 pod used `volumeInGb=0` and a 160 GB container disk so
`/workspace` was local overlay storage. Launch audit passed: public
train/val/all counts were `10000/1000/11000`, remote manifest files were
exactly `all.txt`, `train.txt`, and `val.txt`, no hidden manifest/path was
present, feature/label cache counts were `11000/11000`, encoded missing paths
were `0`, the E63 checkpoint was present, FoldScore import worked, CUDA
reported `NVIDIA B300 SXM6 AC`, and the model had `3,106,690` parameters
(+0.0015% versus AF2-medium pair-only `3,106,642`).

Result: keep and continue. E64 completed at step 4000 with
`val_lddt_ca=0.3739`, FoldScore `0.3634`, `val_ca_drmsd=10.5481`, and
predicted/true C-alpha radius of gyration `11.3344 / 15.4034`. Selected
face/tetra boundary lDDT was `0.5358` / `0.5205`, contraction fractions were
`0.6699` / `0.6712`, and boundary length MAE was
`2.7582` / `2.8986`. Artifacts were copied locally, and the owned E64 pod was
stopped and deleted after the returned result was recorded.

### E65: Scheduled Selected-Boundary lDDT Weight

Status: completed on Runpod and rejected.

Hypothesis: E63 is the first branch where selected-boundary realization and
primary C-alpha lDDT improve together. A static `0.05` selected-boundary lDDT
weight may either need to persist for confirmation or relax after the selected
complex has crossed the `0.5` boundary-lDDT diagnostic threshold. Making the
face/tetra boundary-lDDT weights schedulable lets us test that topology-native
curriculum directly instead of changing unrelated coordinate losses.

Mechanism: add optional
`--simplex-face-boundary-lddt-weight-final`,
`--simplex-tetra-boundary-lddt-weight-final`,
`--simplex-boundary-lddt-ramp-start-step`, and
`--simplex-boundary-lddt-ramp-steps`. These only schedule the loss weights for
the selected face/tetra boundary 1-skeleton; they add no parameters and do not
turn the objective into a dense all-pairs lDDT loss.

Planned launch: resume E64 from its step-4000 checkpoint and continue
`full_msa_to_face` to step 5000. Keep the selected face/tetra coordinate
weights at `1.0`, selected boundary coordinate-distance weights at `0.5`,
and `simplex_aux_weight=0.5`. Start the selected-boundary lDDT weights at
`0.05`, hold through step 4500, then ramp to `0.025` by step 5000:
`--simplex-face-boundary-lddt-weight-final 0.025`,
`--simplex-tetra-boundary-lddt-weight-final 0.025`,
`--simplex-boundary-lddt-ramp-start-step 4500`, and
`--simplex-boundary-lddt-ramp-steps 500`.

Launch: E65 ran on owned Runpod B200 pod `21pml3y3hbbbpb` from commit
`d766050`. The pod uses `volumeInGb=0` and a 160 GB container disk so
`/workspace` is local overlay storage. Clean launch audit after copying only
public data/code: public train/val/all manifest counts are `10000/1000/11000`,
remote manifest files are exactly `all.txt`, `train.txt`, and `val.txt`,
hidden manifest/path absent, feature/label `.npz` counts `11000/11000`,
encoded missing paths `0`, E64 checkpoint present, B200 CUDA available,
NanoFold `foldscore_components` import works, AF2-medium pair-only
`3,106,642`, and E65 model `3,106,690` parameters (`+0.0015%`). Schedule
audit confirmed face/tetra selected-boundary lDDT weights of `0.05` at steps
4000 and 4500, `0.0375` at step 4750, and `0.025` at step 5000. The run
resumed E64 at step 4000 with weights-only loading, loaded 1196 matching model
tensors, initialized 0 new/missing tensors, and started a fresh optimizer.

Decision rule: if step 4500 improves but step 5000 drops, next test a static
`0.05` continuation from E64. If both step 4500 and step 5000 improve,
continue the relaxed schedule. If both drop, reject this scheduling family and
return to architecture changes in selected-cell communication.

Result: reject. E65 step 4500 reached `val_lddt_ca=0.3645`, FoldScore
`0.3660`, `val_ca_drmsd=10.2712`, and predicted/true C-alpha radius
`12.0008 / 15.4034`. The final step 5000 point reached
`val_lddt_ca=0.3684`, FoldScore `0.3666`, `val_ca_drmsd=10.8445`, and
predicted/true C-alpha radius `11.7879 / 15.4034`. This improves FoldScore
slightly relative to E64 but loses the primary C-alpha lDDT at both returned
points, so the relaxed selected-boundary lDDT schedule is not the next branch.
Artifacts were copied locally, and the owned E65 pod was stopped and deleted;
a post-delete lookup returned 404.

Validation:

- `python -m py_compile minalphafold/trainer.py scripts/run_nanofold_public_benchmarks.py`
- `python -m pytest tests/test_nanofold_public_benchmarks.py tests/test_trainer.py::test_apply_loss_weight_schedule_ramps_research_weights tests/test_trainer.py::test_alphafold_loss_overrides_simplex_coordinate_weights`

### E66: Coface-Balanced Selected-Boundary lDDT

Status: completed on Runpod and rejected.

Hypothesis: E63/E64 show that selected-boundary lDDT improves the learned
boundary 1-skeleton, but the same undirected residue edge can appear many
times across selected faces and tetra cofaces. Without coface balancing, the
loss is partly a cell-multiplicity-weighted objective rather than a clean
1-cochain objective on selected boundary edges. Normalizing by undirected
boundary-edge incidence degree should ask each selected edge to contribute
once to realization, while still letting face/tetra cells define which edges
exist.

Mechanism: use the existing `--simplex-boundary-degree-normalize` flag with
the E64/E65 selected-boundary lDDT protocol. This applies inverse incidence
weights through `_boundary_degree_weights` to selected face/tetra boundary
edge losses, including the selected-boundary lDDT terms. It adds no
parameters and remains topological because the weights are computed from the
selected sparse cell complex.

Planned launch: resume E64 from its step-4000 checkpoint and run a 500-step
gate to step 4500 with static selected-boundary lDDT weights `0.05`, selected
face/tetra coordinate weights `1.0`, selected boundary coordinate-distance
weights `0.5`, `simplex_aux_weight=0.5`, and
`--simplex-boundary-degree-normalize`. This directly compares coface-balanced
boundary realization against E65's unbalanced static step-4500 point before
spending on a longer continuation.

Launch: E66 ran on owned Runpod B200 pod `xlvkre8ww4utac`
(`codex-simplexfold-e66-runpod-20260511`) from commit `c2dce57`. Clean launch
audit after copying only public data/code: public train/val/all manifest
counts are `10000/1000/11000`, remote manifest files are exactly `all.txt`,
`train.txt`, and `val.txt`, hidden manifest/features/labels are absent,
feature/label `.npz` counts are `11000/11000`, E64 checkpoint present, B200
CUDA available, NanoFold `foldscore_components` import works, AF2-medium
pair-only has `3,106,642` parameters, and E66 model has `3,106,690`
parameters (`+0.0015%`). `run_metadata.json` records
`simplex_boundary_degree_normalize=true`, static face/tetra
selected-boundary lDDT weights `0.05` / `0.05`, weights-only resume from E64,
crop 256, MSA depth 64, and no templates.

Decision rule: keep only if coface balancing improves or preserves
`val_lddt_ca` while reducing selected-boundary contraction and avoiding a
FoldScore/dRMSD regression. Reject if it repeats E24's early-run behavior and
weakens the selected-boundary signal.

Result: reject. E66 completed at step 4500 with
`val_lddt_ca=0.3505`, FoldScore `0.3602`, `val_ca_drmsd=10.6237`, and
predicted/true C-alpha radius `11.8892 / 15.4034`. Selected face/tetra
boundary lDDT dropped to `0.5090` / `0.4948`, contraction fractions rose to
`0.7145` / `0.7136`, and boundary length MAE rose to `2.8951` / `3.0322`.
Coface balancing reduced repeated-edge weighting but weakened the actual
selected-boundary realization signal, so it is not the next branch. Artifacts
were copied locally, and the owned E66 pod was stopped and deleted; a
post-delete lookup returned 404.

### E67: Weak Selected-Complex Structure Readout

Status: completed on Runpod and rejected as a continuation branch.

Hypothesis: E65 and E66 show that changing selected-boundary loss weighting
does not preserve the E64 lDDT peak. The next topology-native test should
change communication instead: route a small selected face/tetra cochain summary
into the structure module so higher-rank cell state participates in coordinate
realization, rather than only supervising final boundary edges.

Mechanism: use the existing simplicial structure readout path with
`--simplex-structure-readout-scale 0.05` while keeping the E64 selected-boundary
lDDT and coordinate-realization losses. This adds no new parameters beyond the
already budgeted SimplexFold modules and keeps the intervention on selected
higher-order cells.

Planned launch: resume E64 from its step-4000 checkpoint and run a 500-step
gate to step 4500 with static selected-boundary lDDT weights `0.05`, selected
face/tetra coordinate weights `1.0`, selected boundary coordinate-distance
weights `0.5`, `simplex_aux_weight=0.5`, and
`--simplex-structure-readout-scale 0.05`.

Launch: E67 ran on owned Runpod B200 pod `3en5noqmkkiovz`
(`codex-simplexfold-e67-runpod-20260511`) from commit `27ddea4`. Clean launch
audit after copying only public data/code: public train/val/all manifest
counts are `10000/1000/11000`, remote manifest files are exactly `all.txt`,
`train.txt`, and `val.txt`, hidden manifest/features/labels are absent,
feature/label `.npz` counts are `11000/11000`, E64 checkpoint present, B200
CUDA available, NanoFold `foldscore_components` import works, AF2-medium
pair-only has `3,106,642` parameters, and E67 model has `3,106,690`
parameters (`+0.0015%`). `run_metadata.json` records
`simplex_structure_readout_scale=0.05`, static face/tetra selected-boundary
lDDT weights `0.05` / `0.05`, weights-only resume from E64, crop 256, MSA
depth 64, and no templates.

Decision rule: keep only if the step-4500 lDDT improves over E65's unbalanced
continuation (`0.3645`) without a large FoldScore/dRMSD or selected-boundary
diagnostic regression. Continue only if it approaches or exceeds E64's
`0.3739`; reject if it behaves like the earlier broad readout sidecars.

Result: reject as a continuation branch. E67 completed at step 4500 with
`val_lddt_ca=0.3647`, FoldScore `0.3619`, `val_ca_drmsd=10.3503`, and
predicted/true C-alpha radius `11.6688 / 15.4034`. It barely improved over
E65's step-4500 lDDT but stayed below E64 and regressed FoldScore. The useful
signal is geometric: selected face/tetra boundary length MAE improved to
`2.6833` / `2.8167`, better than E64/E65, while boundary lDDT stayed close to
E65 at `0.5302` / `0.5154`. Artifacts were copied locally, and the owned E67
pod was stopped and deleted; a post-delete lookup returned 404.

### E68: Damped Selected-Complex Structure Readout

Status: completed on Runpod and rejected.

Hypothesis: E67 shows the selected-complex structure readout carries useful
geometry signal but couples too strongly into the structure module at scale
`0.05`. Halving the readout scale may preserve local C-alpha lDDT while
retaining the dRMSD and boundary-length benefits.

Mechanism: rerun the E67 topology communication path with
`--simplex-structure-readout-scale 0.025`, keeping the E64 selected-boundary
lDDT and coordinate-realization losses unchanged.

Planned launch: resume E64 from its step-4000 checkpoint and run a 500-step
gate to step 4500 with static selected-boundary lDDT weights `0.05`, selected
face/tetra coordinate weights `1.0`, selected boundary coordinate-distance
weights `0.5`, `simplex_aux_weight=0.5`, and
`--simplex-structure-readout-scale 0.025`.

Launch: E68 ran on owned Runpod B200 pod `qx6oa0jgchz8j8`
(`codex-simplexfold-e68-runpod-20260511`) from commit `11fc14a`. Clean launch
audit after copying only public data/code: public train/val/all manifest
counts are `10000/1000/11000`, remote manifest files are exactly `all.txt`,
`train.txt`, and `val.txt`, hidden manifest/features/labels are absent,
feature/label `.npz` counts are `11000/11000`, E64 checkpoint present, B200
CUDA available, NanoFold `foldscore_components` import works, AF2-medium
pair-only has `3,106,642` parameters, and E68 model has `3,106,690`
parameters (`+0.0015%`). `run_metadata.json` records
`simplex_structure_readout_scale=0.025`, static face/tetra selected-boundary
lDDT weights `0.05` / `0.05`, weights-only resume from E64, crop 256, MSA
depth 64, and no templates.

Decision rule: keep only if step-4500 lDDT improves over E67 and E65 while
preserving E67's dRMSD or selected-boundary length improvements. Continue only
if it approaches or exceeds E64; otherwise reject the structure-readout scale
family and move to a different selected-cell communication mechanism.

Result: reject. E68 completed at step 4500 with
`val_lddt_ca=0.3617`, FoldScore `0.3625`, `val_ca_drmsd=10.2115`, and
predicted/true C-alpha radius `11.9645 / 15.4034`. Selected face/tetra
boundary lDDT was `0.5247` / `0.5103`, contraction fractions were
`0.6823` / `0.6825`, and boundary length MAE was
`2.7150` / `2.8478`. The damped readout improved dRMSD relative to E67 but
lost more local C-alpha lDDT, so the structure-readout scale family is not the
next continuation branch. Artifacts were copied locally, and the owned E68 pod
was stopped and deleted; a post-delete lookup returned 404.

### E69: Selected Face Normal Orientation

Status: completed on Runpod and rejected.

Hypothesis: the README's face interpretation includes oriented local patches
with area, angles, and normal direction. E64's selected-boundary lDDT improves
the selected complex's 1-skeleton, but face orientation may still be too weak
for the structure module to realize coherent 2-simplex patches. Supervising
normal orientation only on model-selected faces should strengthen the learned
2-skeleton without becoming a generic dense output metric.

Mechanism: add `--simplex-face-normal-weight 0.05` to the E64 selected-boundary
recipe. The existing loss compares predicted and true selected-face normals in
residue-local frames, is global-rotation invariant, adds no parameters, and is
attached only to `simplex_face_indices`.

Planned launch: resume E64 from its step-4000 checkpoint and run a 500-step
gate to step 4500 with static selected-boundary lDDT weights `0.05`, selected
face/tetra coordinate weights `1.0`, selected boundary coordinate-distance
weights `0.5`, `simplex_aux_weight=0.5`, and
`--simplex-face-normal-weight 0.05`.

Launch: E69 ran on owned Runpod B200 pod `eznq63h3uorbrf`
(`codex-simplexfold-e69-runpod-20260511`) from commit `34a2796`. Clean launch
audit after copying only public data/code: public train/val/all manifest
counts are `10000/1000/11000`, remote manifest files are exactly `all.txt`,
`train.txt`, and `val.txt`, hidden manifest/features/labels are absent,
feature/label `.npz` counts are `11000/11000`, E64 checkpoint present, B200
CUDA available, NanoFold `foldscore_components` import works, AF2-medium
pair-only has `3,106,642` parameters, and E69 model has `3,106,690`
parameters (`+0.0015%`). `run_metadata.json` records
`simplex_face_normal_weight=0.05`, static face/tetra selected-boundary lDDT
weights `0.05` / `0.05`, weights-only resume from E64, crop 256, MSA depth
64, and no templates. Do not add E69 to `EXPERIMENT_RESULTS.md` until the
Runpod run returns.

Decision rule: keep only if step-4500 lDDT improves over E65/E67 and does not
badly regress FoldScore, dRMSD, or selected-boundary diagnostics. Continue only
if it approaches or exceeds E64; reject if it repeats the early E37 behavior
at this stronger E64 checkpoint.

Result: reject. E69 completed at step 4500 with
`val_lddt_ca=0.3653`, FoldScore `0.3632`, `val_ca_drmsd=10.5833`, and
predicted/true C-alpha radius `11.8750 / 15.4034`. The selected face-normal
term was active (`val_weighted_simplex_face_normal_loss=0.0177`), but the run
stayed below E64 and did not improve selected-complex realization: selected
face/tetra boundary lDDT fell to `0.5210` / `0.5059`, contraction fractions
were `0.6824` / `0.6836`, and boundary length MAE was
`2.8591` / `3.0094`. Artifacts were copied locally, and the owned E69 pod was
stopped and deleted; a post-delete lookup returned 404.

### E70: Damped Edge-Frame Boundary Messages

Status: completed on Runpod and kept for continuation.

Hypothesis: E69 suggests that supervising selected face orientation as a
separate auxiliary target weakens the boundary geometry. The orientation-aware
signal may need to move through the selected-complex communication path
instead. Edge-frame scalarized messages let selected face/tetra cochains write
geometry-aware information through boundary-edge frames, matching the
topological view of cochains exchanging information across ranks.

Mechanism: enable the existing edge-frame message module at a smaller scale
than E61 and ramp the runtime contribution from `0.0` to `0.025` during the
E64 continuation. This adds budgeted edge-frame MLP parameters but stays
within the 5% AF2-medium allowance and keeps the intervention on selected
face/tetra boundary-edge communication rather than adding a generic dense
coordinate loss.

Planned launch: resume E64 from its step-4000 checkpoint and run a 500-step
gate to step 4500 with static selected-boundary lDDT weights `0.05`, selected
face/tetra coordinate weights `1.0`, selected boundary coordinate-distance
weights `0.5`, `simplex_aux_weight=0.5`,
`--simplex-edge-frame-message-scale 0.025`,
`--simplex-edge-frame-message-runtime-scale 0.0`,
`--simplex-edge-frame-message-runtime-scale-final 0.025`,
`--simplex-edge-frame-message-runtime-scale-ramp-start-step 4000`, and
`--simplex-edge-frame-message-runtime-scale-ramp-steps 500`.

Launch: E70 ran on owned Runpod B200 pod `lovgzo4hz2k4fp`
(`codex-simplexfold-e70-runpod-20260512`) from commit `bf7de3d`. Clean launch
audit after copying only public data/code: public train/val/all manifest
counts are `10000/1000/11000`, remote manifest files are exactly `all.txt`,
`train.txt`, and `val.txt`, hidden manifest/features/labels are absent,
feature/label `.npz` counts are `11000/11000`, no AppleDouble sidecar feature
files are present, E64 checkpoint present, B200 CUDA available, NanoFold
`foldscore_components` import works, AF2-medium pair-only has `3,106,642`
parameters, and E70 edge-frame model has `3,154,242` parameters (`+1.53%`).
`run_metadata.json` records `simplex_edge_frame_message_scale=0.025`, runtime
edge-frame scale `0.0 -> 0.025`, ramp start step `4000`, ramp steps `500`,
weights-only resume from E64, crop 256, MSA depth 64, and no templates.

Decision rule: keep only if step-4500 lDDT improves over E65/E67/E69 and does
not lose E64's selected-boundary lDDT/contraction diagnostics. Continue only if
it approaches or exceeds E64; reject if it repeats E61's local-lDDT regression.

Result: keep for a stability continuation. E70 completed at step 4500 with
`val_lddt_ca=0.3742`, FoldScore `0.3653`, `val_ca_drmsd=10.3425`, and
predicted/true C-alpha radius `11.4815 / 15.4034`. It is only a tiny lDDT
gain over E64, but the topological diagnostics also improved: selected
face/tetra boundary lDDT reached `0.5365` / `0.5215`, contraction fractions
fell to `0.6665` / `0.6681`, and boundary length MAE improved to
`2.6313` / `2.7606`. Artifacts were copied locally; the owned pod remains
active for the E71 continuation.

### E71: Continue Damped Edge-Frame Boundary Messages

Status: completed on Runpod and kept for continuation.

Hypothesis: E70 is a small but coherent improvement, so the edge-frame
boundary-message path may be the first architecture route that improves both
main lDDT and selected-complex diagnostics after E64. The next question is
whether that improvement survives another 500 steps or is just a single
checkpoint fluctuation.

Mechanism: continue from the E70 step-4500 checkpoint to step 5000 on the same
edge-frame architecture, holding runtime edge-frame contribution at `0.025`
instead of ramping. Keep the E64/E70 selected-boundary lDDT and
coordinate-realization recipe unchanged.

Planned launch: resume
`e70_edge_frame0025_from_e64_s4500_c256_m64/checkpoints/full_msa_to_face_latest.pt`
and run a 500-step continuation to step 5000 with
`--simplex-edge-frame-message-scale 0.025`,
`--simplex-edge-frame-message-runtime-scale 0.025`,
static selected-boundary lDDT weights `0.05`, selected face/tetra coordinate
weights `1.0`, selected boundary coordinate-distance weights `0.5`, and
`simplex_aux_weight=0.5`.

Launch: E71 ran on the same owned Runpod B200 pod `lovgzo4hz2k4fp`
(`codex-simplexfold-e70-runpod-20260512`) from commit `e201086`, reusing the
clean public-data environment staged for E70. `run_metadata.json` records
`simplex_edge_frame_message_scale=0.025`, runtime edge-frame scale `0.025`,
weights-only resume from the E70 step-4500 checkpoint, crop 256, MSA depth 64,
and no templates. The launch log shows the runner resumed E70 at step
4500/examples 36000, loaded 1244 matching model tensors, initialized 0
new/missing tensors, and started a fresh optimizer.

Decision rule: keep only if step 5000 improves or preserves E70's lDDT and
selected-boundary diagnostics. Reject if lDDT drops back into the E65/E67/E69
band despite the improved edge-frame boundary geometry.

Result: keep for another continuation. E71 completed at step 5000 with
`val_lddt_ca=0.3751`, FoldScore `0.3679`, `val_ca_drmsd=10.1926`, and
predicted/true C-alpha radius `11.4483 / 15.4034`. This improves E70 on the
main validation metrics, but selected-boundary lDDT softened to
`0.5336` / `0.5181` and boundary length MAE rose to
`2.7916` / `2.9309`. Continue once more, but treat a further boundary
diagnostic regression as a warning even if lDDT inches upward.

### E72: Continue Edge-Frame Boundary Messages To 5500

Status: completed on Runpod and rejected.

Hypothesis: E70-E71 are the first continuation sequence after E64 to improve
lDDT, FoldScore, and dRMSD together. A 5500-step gate tests whether the
edge-frame path is a stable direction or whether the selected-boundary
diagnostics are beginning to decouple from the main metrics.

Mechanism: continue from the E71 step-5000 checkpoint on the same edge-frame
architecture, holding runtime edge-frame contribution at `0.025` and keeping
the E64 selected-boundary lDDT and coordinate-realization recipe unchanged.

Planned launch: resume
`e71_edge_frame0025_from_e70_s5000_c256_m64/checkpoints/full_msa_to_face_latest.pt`
and run a 500-step continuation to step 5500 with
`--simplex-edge-frame-message-scale 0.025`,
`--simplex-edge-frame-message-runtime-scale 0.025`,
static selected-boundary lDDT weights `0.05`, selected face/tetra coordinate
weights `1.0`, selected boundary coordinate-distance weights `0.5`, and
`simplex_aux_weight=0.5`.

Launch: E72 is running on the same owned Runpod B200 pod `lovgzo4hz2k4fp`
(`codex-simplexfold-e70-runpod-20260512`) from commit `1d27dd9`, reusing the
clean public-data environment staged for E70/E71. `run_metadata.json` records
`simplex_edge_frame_message_scale=0.025`, runtime edge-frame scale `0.025`,
weights-only resume from the E71 step-5000 checkpoint, crop 256, MSA depth 64,
and no templates. The launch log shows the runner resumed E71 at step
5000/examples 40000, loaded 1244 matching model tensors, initialized 0
new/missing tensors, and started a fresh optimizer.

Decision rule: keep only if step 5500 improves or preserves E71's lDDT,
FoldScore, and dRMSD without further eroding selected-boundary lDDT. Reject if
the run shows main metric drift at the expense of selected-complex realization.

Result: reject as a primary-lDDT continuation. E72 completed at step 5500 with
`val_lddt_ca=0.3718`, below E71's `0.3751`, even though FoldScore improved to
`0.3722`, `val_ca_drmsd` improved to `10.1027`, and predicted/true C-alpha
radius opened to `12.0872 / 15.4034`. The selected-complex diagnostics also
improved: selected face/tetra boundary lDDT reached `0.5450` / `0.5303`,
contraction fractions fell to `0.6555` / `0.6555`, and boundary length MAE
ended at `2.6296` / `2.7581`. The interpretation is a genuine
main-metric/topological-diagnostic split: full-strength edge-frame messages
keep improving selected-boundary realization and global geometry, but they no
longer preserve local C-alpha lDDT.

### E73: Half-Scale Edge-Frame Boundary Messages From E71

Status: completed on owned Runpod pod `lovgzo4hz2k4fp`.

Hypothesis: E72 shows that the selected boundary-edge frame route is not
empty signal; it improves FoldScore, dRMSD, radius, and boundary lDDT. The
failure mode is likely over-coupling from higher-rank face/tetra cochains back
into boundary edges after step 5000. A half-scale runtime gate should test the
same topological communication path without pushing the structure module as
hard toward global expansion.

Mechanism: resume the E71 step-5000 checkpoint, keep the same allocated
edge-frame message modules (`--simplex-edge-frame-message-scale 0.025`) but
set the runtime contribution to `0.0125`. Keep the E64 selected-boundary
lDDT and coordinate-realization recipe unchanged. This is not a new generic
metric loss; it is a damping test for the selected face/tetra boundary-edge
cochain exchange.

Planned launch: resume
`e71_edge_frame0025_from_e70_s5000_c256_m64/checkpoints/full_msa_to_face_latest.pt`
and run a 500-step continuation to step 5500 with
`--simplex-edge-frame-message-scale 0.025`,
`--simplex-edge-frame-message-runtime-scale 0.0125`, static
selected-boundary lDDT weights `0.05`, selected face/tetra coordinate weights
`1.0`, selected boundary coordinate-distance weights `0.5`, and
`simplex_aux_weight=0.5`.

Decision rule: keep only if step 5500 improves or preserves E71's primary
`val_lddt_ca` while retaining E72's improved selected-boundary realization.
Reject if it remains below E71, and then stop this edge-frame continuation
family in favor of changing the selected complex construction itself.

Launch: the accepted E73 run is
`e73_evalfix_edge_frame00125_from_e71_s5500_c256_m64`, running on the same
owned Runpod B200 pod `lovgzo4hz2k4fp`
(`codex-simplexfold-e70-runpod-20260512`) from the fixed runner snapshot
corresponding to commit `7f83b2e`. It reuses the clean public-data environment
staged for E70-E72. The first launch from commit `bc1b749` was stopped before
any result because it trained with runtime simplex overrides but would have
validated with static simplex settings.

`run_metadata.json` for the evalfix run records
`simplex_edge_frame_message_scale=0.025`, runtime edge-frame scale `0.0125`,
weights-only resume from the E71 step-5000 checkpoint, crop 256, MSA depth 64,
and no templates. The launch log shows the runner resumed E71 at step
5000/examples 40000, loaded 1244 matching model tensors, initialized 0
new/missing tensors, and started a fresh optimizer. The heartbeat has been
retargeted to the evalfix artifact path.

Result: keep as the new primary reference. E73 completed at step 5500 with
`val_lddt_ca=0.3807`, FoldScore `0.3720`, `val_ca_drmsd=10.0777`, and
predicted/true C-alpha radius `11.6741 / 15.4034`. This improves E71's local
lDDT (`0.3751`) while preserving the E72 FoldScore/dRMSD direction. The
selected-complex diagnostics are mixed: selected face/tetra boundary lDDT
ended at `0.5368` / `0.5213`, below E72's `0.5450` / `0.5303`, with boundary
length MAE `2.7292` / `2.8698` and contraction fractions
`0.6669` / `0.6692`. Half-scale edge-frame cochain exchange is therefore a
better local-lDDT setting than E72, but it does not solve boundary-edge reuse
or fully recover E72's selected-boundary realization gains.

### E76: Continue Half-Scale Edge-Frame Boundary Messages To 6000

Status: completed and rejected.

Hypothesis: E73 is the first edge-frame continuation past E71 to improve
local C-alpha lDDT, FoldScore, and dRMSD together. Continuing the same
half-scale boundary-edge message route for another 500 optimizer steps tests
whether the improvement is a real trajectory or just a single checkpoint
fluctuation.

Mechanism: resume the E73 step-5500 checkpoint and keep the same
selected-boundary lDDT, selected coordinate-realization, and half-scale
edge-frame boundary-message recipe. This remains a selected-complex
communication experiment: the changed signal is still the face/tetra
boundary-edge cochain exchange, not a generic dense output-coordinate loss.

Planned launch: resume
`e73_evalfix_edge_frame00125_from_e71_s5500_c256_m64/checkpoints/full_msa_to_face_latest.pt`
and run to step 6000 with `--simplex-edge-frame-message-scale 0.025`,
`--simplex-edge-frame-message-runtime-scale 0.0125`, static selected-boundary
lDDT weights `0.05`, selected face/tetra coordinate weights `1.0`, selected
boundary coordinate-distance weights `0.5`, `simplex_aux_weight=0.5`, crop
256, MSA depth 64, no templates, and effective batch 8.

Decision rule: keep only if step 6000 improves or preserves E73's
`val_lddt_ca=0.3807` without a serious selected-boundary diagnostic collapse.
If local lDDT turns over, stop the half-scale edge-frame continuation branch
and move to a topology operator or selected-complex construction change.

Launch: E76 ran as
`e76_edge_frame00125_from_e73_s6000_c256_m64` on the same owned Runpod B200
pod. The launch resumed
`e73_evalfix_edge_frame00125_from_e71_s5500_c256_m64/checkpoints/full_msa_to_face_latest.pt`
at step 5500/examples 44000, loaded 1244 matching tensors, initialized 0
new/missing tensors, and started a fresh optimizer. The log path is
`/workspace/SimplexFold/logs/e76_edge_frame00125_from_e73.log`, and the
artifact path is
`/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e76_edge_frame00125_from_e73_s6000_c256_m64/`.

Result: step 6000 reached `val_lddt_ca=0.3713`, FoldScore `0.3723`,
`val_ca_drmsd=10.2191`, and predicted/true C-alpha radius
`12.0036 / 15.4034`. Selected face/tetra boundary lDDT ended at `0.5341` /
`0.5185`, boundary length MAE at `2.8649` / `3.0128`, and contraction
fractions at `0.6488` / `0.6502`.

Interpretation: reject. The tiny FoldScore gain does not compensate for the
loss of local C-alpha lDDT from E73's `0.3807`, and the selected-boundary
diagnostics also softened relative to the stronger checkpoints. Stop the
plain half-scale edge-frame continuation branch and test an incidence
operator change next.

### E77: Coface-Degree-Attenuated Boundary Messages

Status: completed and rejected.

Hypothesis: E70-E76 improved lDDT only after the model learned to pass
geometry through selected face/tetra boundary edges, but the diagnostics still
show very high boundary-edge reuse. The current pair readout averages
messages per edge, yet a heavily reused boundary edge can still act as a noisy
high-order bottleneck. Damping the readout by selected coface degree should
reduce over-coupling on those dense incidence edges while preserving the same
selected cell complex.

Mechanism: add a zero-parameter
`simplex_boundary_message_degree_attenuation` model knob. After face/tetra
messages are scattered and averaged into the pair tensor, the pair readout is
divided by `coface_degree ** attenuation`. `0.0` exactly preserves current
behavior. A small value such as `0.25` is the first candidate because it
weakens reused boundary edges without deleting cells or changing the topology
selector. This is an incidence-normalized cochain communication change, not a
generic coordinate loss.

Launch plan: sync the implementation to the owned Runpod workspace, resume the
stronger E73 checkpoint, and run a 500-step gate to step 6000 with the E76
recipe plus `--simplex-boundary-message-degree-attenuation 0.25`. Compare
against E73/E76 on primary `val_lddt_ca` and against E72/E73 on
selected-boundary diagnostics.

Launch: E77 ran as `e77_degree_atten025_from_e73_s6000_c256_m64` on the
owned Runpod B200 pod. Remote py_compile passed for
`minalphafold/simplex.py`, `minalphafold/model_config.py`, and
`scripts/run_nanofold_public_benchmarks.py`, and parser smoke confirmed
`--simplex-boundary-message-degree-attenuation 0.25`. The launch resumed
`e73_evalfix_edge_frame00125_from_e71_s5500_c256_m64/checkpoints/full_msa_to_face_latest.pt`
at step 5500/examples 44000, loaded 1244 matching tensors, initialized 0
new/missing tensors, and started a fresh optimizer. Main Python PID is
`24587`; data-worker PIDs are `27514` and `27515`. The log path is
`/workspace/SimplexFold/logs/e77_degree_atten025_from_e73.log`, and the
artifact path is
`/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e77_degree_atten025_from_e73_s6000_c256_m64/`.

Result: step 6000 reached `val_lddt_ca=0.3733`, FoldScore `0.3710`,
`val_ca_drmsd=10.1286`, and predicted/true C-alpha radius
`11.8632 / 15.4034`. Selected face/tetra boundary lDDT improved to `0.5421` /
`0.5265`, boundary length MAE improved to `2.5714` / `2.7039`, and
contraction fractions improved to `0.6467` / `0.6475`.

Interpretation: reject as a primary branch but keep the diagnostic. Coface
degree attenuation made the selected sparse complex realize its own boundary
edges more cleanly, but primary local C-alpha lDDT remained below E73's
`0.3807`. The next active test should move upstream to selected-complex
construction rather than further normalizing the same boundary-edge readout.

Validation:

- `python -m pytest tests/test_simplex.py::test_coface_degree_attenuation_damps_reused_boundary_edges tests/test_simplex.py::test_boundary_message_degree_attenuation_gates_pair_readout_without_single_change tests/test_nanofold_public_benchmarks.py::test_model_config_override_flags_are_accepted_by_cli_parser tests/test_trainer.py::test_simplicial_boundary_message_degree_attenuation_adds_no_parameters`
- `python -m py_compile minalphafold/simplex.py minalphafold/model_config.py scripts/run_nanofold_public_benchmarks.py`

### E74: Light Recycled-Geometry Topology Selector

Status: completed on owned Runpod pod `o1dy17ouv8w5mz`.

Hypothesis: E72 improved selected-boundary lDDT, FoldScore, dRMSD, and global
radius while reducing primary local lDDT. One possible failure mode is that,
after coordinates open up, the simplex neighbor selector's default recycled
geometry term (`simplex_geometry_distance_weight=0.1`) over-biases the sparse
complex toward the model's current metric geometry. Reducing that weight should
let learned pair/contact topology retain more control over which residues
become selected face/tetra cochains.

Mechanism: add runner overrides for `simplex_geometry_distance_weight` and
its optional final/ramp settings, then test `0.025` from the E71/E73
checkpoint family or schedule `0.1 -> 0.025` over the continuation. This
changes the cell-complex construction step itself: the selected face/tetra
incidence relations are built from a softer blend of learned pair topology and
recycled C-alpha geometry. It adds no parameters and does not introduce a
generic output-coordinate loss.

Launch: E74 ran as `e74_light_geom0025_from_e73_s6000_c256_m64` on a
second owned Runpod H100 pod. The pod was created from the same PyTorch image,
then staged with the current SimplexFold branch, NanoFold public manifests,
features, labels, and the E73 checkpoint. Verification before launch confirmed
public manifest counts `10000/1000/11000`, feature/label NPZ counts
`11000/11000`, E73 checkpoint size `35,385,519` bytes, remote py_compile for
the runner/model files, parser support for `--simplex-geometry-distance-weight
0.025`, and NanoFold FoldScore import. The launch resumed
`e73_evalfix_edge_frame00125_from_e71_s5500_c256_m64/checkpoints/full_msa_to_face_latest.pt`
at step 5500/examples 44000, loaded 1244 matching tensors, initialized 0
new/missing tensors, and started a fresh optimizer. The log path is
`/workspace/SimplexFold/logs/e74_light_geom0025_from_e73.log`, and the
artifact path is
`/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e74_light_geom0025_from_e73_s6000_c256_m64/`.

Result: keep as the new primary-lDDT leader and continue it. E74 reached
`val_lddt_ca=0.3841`, FoldScore `0.3666`, `val_ca_drmsd=10.1893`, and
predicted/true C-alpha radius `11.4266 / 15.4034`. It improved selected
face/tetra boundary lDDT to `0.5409` / `0.5258`, boundary length MAE to
`2.5149` / `2.6510`, and contraction fraction to `0.5941` / `0.5957`.
This supports the paper-derived hypothesis that topology construction matters,
though the softer FoldScore/dRMSD means this is a primary-lDDT branch rather
than a fully balanced geometry solution.

Validation:

- `python -m pytest tests/test_nanofold_public_benchmarks.py::test_model_config_override_flags_are_accepted_by_cli_parser tests/test_nanofold_public_benchmarks.py::test_runtime_simplex_message_scales_ramp_and_enter_model_inputs tests/test_nanofold_public_benchmarks.py::test_evaluate_uses_runtime_simplex_overrides_for_validation tests/test_simplex.py::test_build_simplex_topology_geometry_weight_changes_selected_neighbors tests/test_trainer.py::test_simplicial_geometry_selector_weight_adds_no_parameters`

### E78: Continue Light Recycled-Geometry Selector

Status: completed on owned Runpod pod `o1dy17ouv8w5mz`.

Hypothesis: E74 improved primary C-alpha lDDT and selected-boundary
diagnostics over E73. A short continuation can test whether the lighter
recycled-geometry topology prior keeps improving, or whether it only gives a
single-step local-lDDT bump while FoldScore/dRMSD soften.

Mechanism: resume E74's checkpoint from step 6000 to step 6500 with the same
`simplex_geometry_distance_weight=0.025`, half-scale edge-frame message
runtime scale `0.0125`, selected-boundary lDDT weights `0.05`, selected
coordinate weights `1.0`, and selected boundary-distance weights `0.5`.
This remains a topology-construction continuation rather than a new loss.

Launch: E78 ran as `e78_light_geom0025_from_e74_s6500_c256_m64`.
Remote prelaunch checks confirmed no active Python benchmark process,
successful py_compile for the simplex/model-config/runner files, parser
support for the geometry selector flag, and the E74 checkpoint present. Main
Python PID was `1969`. The log path is
`/workspace/SimplexFold/logs/e78_light_geom0025_from_e74.log`, and the
artifact path is
`/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e78_light_geom0025_from_e74_s6500_c256_m64/`.

Result: keep. E78 reached `val_lddt_ca=0.3853`, FoldScore `0.3718`,
`val_ca_drmsd=10.1595`, and predicted/true C-alpha radius
`11.3783 / 15.4034`. It improved over E74 on the primary metric, FoldScore,
dRMSD, selected face/tetra boundary lDDT (`0.5434` / `0.5287`), and selected
boundary length MAE (`2.4519` / `2.5801`). Selected-boundary contraction
worsened to `0.6320` / `0.6327`, so continue only as a short gate and keep
watching whether improved local geometry is coming with over-contracted
selected edges.

### E80: Continue E78 Light-Geometry Selector

Status: completed on owned Runpod pod `o1dy17ouv8w5mz`.

Hypothesis: E78 showed the light recycled-geometry topology selector can
continue improving local C-alpha lDDT while also recovering FoldScore and
dRMSD. Another 500-step continuation tests whether this is a real slope or a
small checkpoint fluctuation.

Mechanism: resume E78's checkpoint from step 6500 to step 7000 with the same
`simplex_geometry_distance_weight=0.025`, half-scale edge-frame message
runtime scale `0.0125`, selected-boundary lDDT weights `0.05`, selected
coordinate weights `1.0`, and selected boundary-distance weights `0.5`.
This remains a topology-construction continuation rather than a new generic
coordinate loss.

Launch: E80 ran as `e80_light_geom0025_from_e78_s7000_c256_m64`.
Remote prelaunch checks confirmed no active Python benchmark process,
successful py_compile for the simplex/model-config/runner files, and the E78
checkpoint present. Main Python PID was `2543`. The log path is
`/workspace/SimplexFold/logs/e80_light_geom0025_from_e78.log`, and the
artifact path is
`/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e80_light_geom0025_from_e78_s7000_c256_m64/`.

Result: reject. E80 reached `val_lddt_ca=0.3820`, FoldScore `0.3682`,
`val_ca_drmsd=10.2493`, and predicted/true C-alpha radius
`11.2472 / 15.4034`. This is below E78 on the primary metric, FoldScore, and
dRMSD. Selected face/tetra boundary lDDT fell to `0.5359` / `0.5192`, and
selected boundary length MAE worsened to `2.6560` / `2.8001`. Contraction
improved slightly to `0.6268` / `0.6289`, but not enough to justify another
blind continuation. Pivot to the scheduled sparse selected-cell complex from
the stronger E78 checkpoint.

### E75: Sparse Selected Higher-Rank Cell Complex

Status: implemented locally and planned only if E78 turns over.

Hypothesis: the current selector picks a sparse residue neighbor star, but then
instantiates the full clique of faces and tetrahedra inside that star. The
E70-E72 diagnostics show very high boundary-edge reuse, especially for tetra
cells, which suggests the higher-rank complex may be too dense even when the
residue-neighbor graph is sparse. A combinatorial-complex view does not require
every possible face/tetra closure to exist. Keeping only the highest-scoring
rank-2 and rank-3 cells should make the persistent cochains more selective and
reduce noisy boundary-edge averaging.

Mechanism: add zero-parameter `simplex_face_top_k` and `simplex_tetra_top_k`
selector caps. `build_simplex_topology` still forms the local candidate
neighbor-star combinations, then scores each face/tetra by the mean of its
selected boundary-edge logits and masks out lower-scoring cells per anchor.
The tensor shapes stay unchanged for checkpoint compatibility, but inactive
cells stop contributing to face/tetra updates, selected-boundary losses, and
diagnostics. This changes the active cell complex itself; it is not an output
metric loss.

Planned launch if needed: from the strongest available E73/E74 checkpoint,
run a 500-step gate with the E64 selected-boundary lDDT/coordinate-realization
recipe, edge-frame modules available, and a first cap such as
`--simplex-face-top-k 24 --simplex-tetra-top-k 48`. Compare against E74/E78 on
primary `val_lddt_ca` and against E72/E77 on selected-boundary diagnostics.

Validation:

- `python -m pytest tests/test_simplex.py::test_build_simplex_topology_cell_topk_caps_active_higher_rank_cells tests/test_nanofold_public_benchmarks.py::test_model_config_override_flags_are_accepted_by_cli_parser tests/test_trainer.py::test_simplicial_cell_topk_selector_adds_no_parameters`
- `python -m py_compile minalphafold/simplex.py minalphafold/model_config.py scripts/run_nanofold_public_benchmarks.py`

### E79: Scheduled Sparse Selected Higher-Rank Cell Complex

Status: running on owned Runpod pod `o1dy17ouv8w5mz`.

Hypothesis: E75's sparse selected-cell complex is the right topology-native
response to high boundary-edge reuse, but switching directly from the full
neighbor-star clique to a small face/tetra cap may shock a resumed checkpoint.
A schedule can preserve the learned E74/E78 geometry while gradually reducing
which rank-2 and rank-3 cochains exist and send messages.

Mechanism: add runtime overrides for `simplex_face_top_k` and
`simplex_tetra_top_k`, with optional final values and ramp windows. For
example, `--simplex-face-top-k 0 --simplex-face-top-k-final 24` plus
`--simplex-tetra-top-k 0 --simplex-tetra-top-k-final 48` starts from the full
selected-cell clique and linearly introduces the E75 cap. This changes the
active cell complex during training without changing parameter count.

Launch: E79 is running as `e79_scheduled_topk_from_e78_s7000_c256_m64`.
After E80 returned and the H100 pod was idle, local source/docs/tests were
synced to `/workspace/SimplexFold/` without deleting remote artifacts, logs,
or checkpoints. Remote py_compile passed, parser smoke confirmed support for
the scheduled top-k flags and `simplex_cell_score_degree_penalty`, and the E78
checkpoint was present. Main Python PID is `3128`. The log path is
`/workspace/SimplexFold/logs/e79_scheduled_topk_from_e78.log`, and the
artifact path is
`/workspace/SimplexFold/artifacts/nanofold_public_benchmarks/e79_scheduled_topk_from_e78_s7000_c256_m64/`.
Do not add E79 to `EXPERIMENT_RESULTS.md` until it returns.

Decision rule after return: compare E79 against E78 and E80. Keep only if the
scheduled sparse-cell construction improves or preserves primary lDDT while
improving selected-boundary lDDT/length or boundary-edge reuse. If E79
improves topology diagnostics but loses primary lDDT, use E81's
degree-penalized sparse-cell scoring as the next fallback rather than
returning to plain light-geometry continuation.

Validation:

- `python -m pytest tests/test_nanofold_public_benchmarks.py::test_model_config_override_flags_are_accepted_by_cli_parser tests/test_nanofold_public_benchmarks.py::test_runtime_simplex_message_scales_ramp_and_enter_model_inputs tests/test_nanofold_public_benchmarks.py::test_evaluate_uses_runtime_simplex_overrides_for_validation tests/test_simplex.py::test_simplicial_adapter_runtime_cell_topk_override_caps_active_cells tests/test_simplex.py::test_build_simplex_topology_cell_topk_caps_active_higher_rank_cells tests/test_trainer.py::test_simplicial_cell_topk_selector_adds_no_parameters`
- `python -m py_compile minalphafold/simplex.py minalphafold/evoformer.py minalphafold/model.py minalphafold/trainer.py scripts/run_nanofold_public_benchmarks.py`

### E81: Degree-Penalized Sparse Cell Scoring

Status: implemented locally and planned only if the sparse-cell fallback is
needed after E80.

Hypothesis: E78 improved selected-boundary lDDT and boundary length error, but
its selected-boundary contraction fraction rose. The topology metrics have
also repeatedly shown high boundary-edge reuse, especially for tetra cells.
When E75/E79 cap the higher-rank complex, a pure boundary-logit score can
still spend many face/tetra cochains on the same overrepresented boundary
edges. A degree penalty should make the selected complex cover more distinct
boundary edges without adding parameters or supervising output coordinates.

Mechanism: add `simplex_cell_score_degree_penalty` to the selected-cell
top-k scorer. For each candidate face/tetra, compute how often its undirected
boundary edges appear across the candidate complex, subtract a log-degree
penalty from the boundary-edge logit score, then apply the existing
per-anchor face/tetra top-k mask. This changes which rank-2 and rank-3
cochains exist and send messages; it does not change tensor shapes,
checkpoint compatibility, or parameter count.

Planned launch if needed: combine with E79's scheduled caps from the strongest
E78/E80 checkpoint, for example
`--simplex-face-top-k 0 --simplex-face-top-k-final 24`,
`--simplex-tetra-top-k 0 --simplex-tetra-top-k-final 48`, and
`--simplex-cell-score-degree-penalty 0.75`. Compare against E79 on primary
`val_lddt_ca`, selected boundary lDDT/length, contraction fraction, and
boundary unique-edge fraction.

Validation:

- `python -m py_compile minalphafold/simplex.py minalphafold/model_config.py scripts/run_nanofold_public_benchmarks.py`
- `python -m pytest tests/test_simplex.py::test_cell_score_degree_penalty_prefers_less_reused_boundary_edges tests/test_simplex.py::test_build_simplex_topology_cell_topk_caps_active_higher_rank_cells tests/test_nanofold_public_benchmarks.py::test_model_config_override_flags_are_accepted_by_cli_parser tests/test_trainer.py::test_simplicial_cell_degree_penalty_adds_no_parameters tests/test_trainer.py::test_simplicial_cell_topk_selector_adds_no_parameters`
