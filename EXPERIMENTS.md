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
