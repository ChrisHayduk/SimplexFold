## 2026-05-15 Operating Plan Update: E140 Failed, E141 Active

Current returned best remains E128 at `val_lddt_ca=0.4311` at step `8500`,
well below the `0.7` goal and below the `0.45` short-gate threshold for a
credible 30k-step spend. E140 has now failed pre-eval on owned Runpod pod
`c67fbk189vnvfp`: the trace was preserved locally and the pod was stopped
afterward, but it produced no
`results.json`, eval-detail CSV, result CSV, checkpoint, or new history row.
The log ends with `OSError: [Errno 5] Input/output error` while writing
`status_full_msa_to_face.json`, so treat E140 as an infrastructure failure
with no scored architectural evidence. E141 remains active on owned Runpod pod
`5ox436mhzej7j4`; its history still correctly ends at inherited E128 step
`8500`, with no result or eval-detail bundle yet.

The two user-provided PDFs are saved for later reference:

- `references/papers/hands_on_geometric_deep_learning_nodes_to_complexes.pdf`
- `references/papers/2509.03885v1.pdf`

Full-text extraction/read-through on 2026-05-15 supports the current
topology-native direction. The general topological-deep-learning guide
emphasizes cochains, incidence maps, inter-neighborhood aggregation across
topological domains, filtrations, and pooling/unspooling. `Topotein:
Topological Deep Learning for Protein Representation Learning` is more
directly actionable: it argues that effective protein TDL needs persistent
multi-rank states, incident/adjacent neighborhood operators, and especially
outer-edge neighborhoods that let higher-rank cells communicate through
external directed edges without echoing messages only inside the same cell.
It also warns that superficial higher-rank features without dedicated update
mechanisms can be counterproductive.

Near-term implication: E141 should finish before launching another branch
because it is already running and directly probes signed oriented boundary
incidence. The next parked idea, if E141 fails to clear the short-gate
threshold, remains E145: a Topotein-style outer-neighborhood
transport probe. The clean version is not a generic metric loss; it is a
topological architecture question: selected face/tetra cochains should update
through directed residue-to-external-edge neighborhoods before their boundary
signal reaches the pair trunk. The trainable `simplex_outer_edge_context_scale`
hook was too expensive on top of the full E128 recipe, so E145 uses a new
parameter-free `simplex_outer_edge_residual_context_scale` path that folds
directed external pair-edge context into face/tetra cochain channels. Parameter
audit: E128-style recipe with E145 scale `0.25` remains `3,240,738 <= 3,261,974`.
The launch should use the runtime override ramp from `0.0` to `0.25` across
steps `8500`-`9000`, so the resumed E128 checkpoint is not hit with an abrupt
new outer-neighborhood cochain update.

Next actions:

1. Keep monitoring only the owned E141 pod `5ox436mhzej7j4` through its
   `status_full_msa_to_face.json` heartbeat. E140 has no live Python process
   and no scored result.
2. When E141 returns, verify and pull artifacts, update
   `EXPERIMENT_RESULTS.md`, commit/push, then decide whether the returned
   score justifies a continuation or pivot. Use the exact E140/E141
   recipe-metadata verifier templates now recorded in `EXPERIMENTS.md`.
   After verifier and eval-detail analysis pass, record the row/header update
   with `scripts/record_experiment_result.py`, which composes the formatter,
   row upsert, and summary-refresh helpers. Review the refreshed
   `EXPERIMENT_RESULTS.md` text against the verified metrics before
   committing. For future `--num-workers 4` launches, include
   `--expected-num-workers 4` in `scripts/verify_nanofold_benchmark_artifacts.py`
   verification.
3. If E141 remains below `0.45` or fails coherently, launch only a short E145
   gate around the ramped parameter-free outer-neighborhood selected-cell
   transport; avoid spending 30,000 steps unless a short gate clears `0.45`
   with coherent FoldScore, dRMSD, and C-alpha Rg. Use the full E145 launch and verifier
   templates now recorded in `EXPERIMENTS.md`, including `--num-workers 4`
   and returned-artifact `--expected-num-workers 4`.
4. For the next launched short gate, run a startup smoke with a small
   DataLoader worker count such as `--num-workers 4`. E140/E141 inherited the
   runner default `0`, and both owned A100 pods have 128 CPU cores while the
   jobs are CPU-heavy. Treat this as a runtime-throughput knob only; do not
   change active runs mid-flight.

## 2026-05-15 Operating Plan Update: E140 Active With Heartbeat

Current returned best remains E128 at `val_lddt_ca=0.4311` at step `8500`,
well below the `0.7` goal and below the `0.45` short-gate threshold for a
credible 30k-step spend. E130, E138, and E139 were all stopped before their
first new evaluation/checkpoint row. With E140's new status heartbeat, we now
know a 500-step continuation can run for many hours before writing the next
history row, so those branches should be treated as pre-eval stopped runs, not
as scored architectural failures.

E139 tested a no-Hodge oriented boundary-cochain readout from E128 and was
stopped on the same owned pod after `00:45:26` elapsed with only startup files
and inherited step-8500 history. Its trace was preserved locally and only E139
PIDs `42517` / `42514` were stopped. Because that checkout did not have live
step heartbeats, the lack of a new history row is not enough evidence to reject
the oriented boundary-cochain architecture.

Active gate: E140 is running from the separate `/workspace/SimplexFold_e140`
checkout on the same owned pod as
`e140_selected_boundary_expansion_from_e128_s9000_c256_m64`, Python PID
`55949`. E140 keeps the E128 selected-complex recipe and adds only the existing
small selected-boundary coordinate-expansion terms for model-selected
face/tetra boundary edges. Startup verification passed: it resumed E128 at
step `8500` / `examples=68000`, wrote `run_metadata.json`, inherited history,
and wrote the live `status_full_msa_to_face.json` heartbeat showing active
step `8501`, microbatch `1`, and effective batch size `8`.

This is the most direct probe of the current bottleneck: local selected
face/tetra boundary lDDT is already high (`0.7559` / `0.7385`), while predicted
C-alpha Rg remains too compact (`11.7198 / 16.3091`). E140 should tell us
whether the learned sparse complex can realize its own 1-skeleton without
collapsing before the structure module assembles global geometry.

Next actions:

1. Monitor E140 through the owned pod and its `status_full_msa_to_face.json`
   heartbeat.
2. When E140 returns, verify and pull artifacts, update
   `EXPERIMENT_RESULTS.md`, commit/push, then decide whether the next gate
   should continue selected-boundary realization or pivot back to the signed
   incidence queue.
3. Do not spend 30,000 steps on any branch until a short gate clears `0.45`,
   with FoldScore, dRMSD, and C-alpha Rg remaining coherent.

E128 validation-detail audit: the top 100 chains by C-alpha lDDT average
`0.6381`, so the current architecture can approach the target on favorable
examples, but the lower tail is still collapsed. Across all 1000 validation
chains, C-alpha lDDT correlates with predicted C-alpha radius (`r=0.5958`),
selected face boundary lDDT (`r=0.5969`), and selected tetra boundary lDDT
(`r=0.6270`). This supports the current topology-native queue: improve how
high-quality local face/tetra cochains assemble global C-alpha geometry rather
than adding an output-side metric shortcut.

## Superseded Plan: E139 No-Hodge Oriented Boundary-Cochain Readout

This section is retained for provenance. E139 has now been stopped as a
pre-eval branch, and E140 is the active gate described in the operating plan
above.

Current status after E138: the best returned validation C-alpha lDDT remains
E128 at `0.4311` at step 8500. E129 resumed the verified E128 checkpoint and
added a tiny sparse triangle-attention value residual, but returned lower at
step 9000 with `val_lddt_ca=0.4303`, FoldScore `0.3984`, and
`val_ca_drmsd=11.2250`. E130 then tested static Hodge-centered selected
boundary readout from E128 but was stopped before its first new eval after
roughly three hours with no new history row, result bundle, eval details, or
checkpoint. E138 then removed Hodge and tested face-cyclic boundary readout
from E128, but it also reached the roughly three-hour cutoff with no step-9000
bundle and was stopped pre-eval.

The planned E139 short gate ran on the same owned Runpod pod
`c67fbk189vnvfp` from the separate `/workspace/SimplexFold_e139` checkout as
`e139_no_hodge_oriented_boundary_from_e128_s9000_c256_m64`, Python PID
`42517`. It resumed E128 at step `8500`, kept E128's oriented face gate and
damped simplex triangle-attention bias, removed E130's Hodge readout and
E138's face-cyclic readout, and ramped only the selected boundary
oriented-cochain readout from `0.0` to `0.25` over steps `8500`-`9000`.
Startup verification passed: the runner saw `train=10000`, `val=1000`, crop
`256`, MSA depth `64`, resumed the E128 checkpoint at `step=8500` /
`examples=68000`, loaded `1332` matching model tensors, initialized `0`
new/missing tensors, and recorded `effective_batch_size=8` with
`max_parameters=3261974`. It was later stopped before the first new eval row;
E140's heartbeat evidence showed this stop was over-aggressive.

E129 tested this topology-native route:

```text
selected F_ijk / boundary faces of U_ijkl
        -> sparse triangle-attention logit bias on ordered triples
        -> sparse triangle-attention value residual on represented triples
        -> AF2 triangle attention propagates cochain content through Z_ij
        -> structure module reads globally updated pair geometry
```

The result is useful evidence but a negative branch decision. The value
residual improved selected face/tetra boundary lDDT (`0.7585` / `0.7401`) and
reduced selected-boundary contraction relative to E128, so the explicit
higher-rank cells are still learning local boundary geometry. However, global
assembly got worse: primary C-alpha lDDT, FoldScore, dRMSD, and C-alpha
expansion all regressed. This reinforces the current diagnosis: simply
injecting more selected-complex content into AF2 triangle attention is not yet
enough to turn good local simplex geometry into an accurate global backbone.

Next branch criteria: do not spend 30,000 steps on E129 or on a plain
triangle-attention value/bias scale-up. The next topology-native short gate
should change how local selected-complex geometry is reconciled before global
readout, not merely add more value content. Prefer a mechanism that preserves
E128's successful oriented boundary-edge realization while adding a
stability/normalization path for global assembly, and require a clear break
above the `0.45` short-gate threshold before any longer-run consideration.

E130 pre-eval diagnosis: Hodge-centered selected-boundary readout remains a
topology-native question, but the branch was stopped before it produced a
score. It treated the
face/tetra boundary messages as a sparse boundary-edge 1-cochain on the
selected complex, double-centered that cochain over source and target residue
vertex stars, and blended the centered cochain back into the pair readout
before `Z_ij` was updated. This did not add parameters or an output-side
lDDT/radius/coordinate loss. The static Hodge gate ran for roughly three
hours without writing a step-9000 history row, result bundle, eval details, or
checkpoint, but without heartbeat evidence we cannot distinguish slow training
from failure.

```text
selected F_ijk / U_ijkl
        -> oriented boundary-edge-frame gate
        -> weak triangle-attention bias
        -> selected boundary 1-cochain readout
        -> Hodge-style vertex-star double centering
        -> Z_ij / structure module
```

E138 pre-eval diagnosis: no-Hodge face-cyclic boundary readout remains a
topology-native question, but the branch was stopped before it produced a
score. It kept E128's
oriented face gate and damped triangle-attention bias, removed
`simplex_boundary_hodge_readout_scale`, and tested whether learned
2-simplex cochains write through the oriented boundary cycle
`(i->j, j->k, k->i)` before updating `Z_ij`. The run crossed the earlier
three-hour cutoff with only inherited E128 history, but after E140 we should
not use that absence of a 500-step history row as architectural evidence.

E139 was the no-Hodge oriented boundary-cochain fallback. E138 tested
orientation at the level of each selected face's 2-simplex boundary cycle;
E139 instead tests the induced selected boundary 1-cochain after face/tetra
scatter by subtracting reverse selected-edge common mode,
`cochain(i,j) - cochain(j,i)`, while preserving one-way directed boundary
edges. This kept the change in the explicit simplicial boundary pathway and
avoided the Hodge double-centering path, but the run was stopped before
producing a scored step-9000 point.

Second parked fallback if E138/E139 are coherent but still too collapsed:
E140 selected-boundary realization anti-collapse. This is the only loss-side
candidate currently worth keeping in queue because it remains attached to the
explicit selected complex: it supervises the realization of model-selected
face/tetra boundary edges, not the full C-alpha radius, all-pairs distance
matrix, or validation lDDT directly. The motivation is the current E128
diagnostic split: selected face/tetra boundary lDDT is already high
(`0.7559` / `0.7385`), but predicted C-alpha Rg is still much smaller than
truth (`11.7198 / 16.3091`). A small selected-boundary expansion term could
test whether local higher-rank cells are being realized too compactly before
their geometry reaches the structure module. Now launched as E140; keep it as a topology realization probe, not
as a generic score-chasing loss.

Parallel candidate now launched while E140 runs: E141 signed face-cyclic
boundary readout. E138's face-cyclic readout preserves the directed
2-simplex boundary cycle, but it is unsigned. The actual oriented boundary of
face `[i,j,k]` is `[j,k] - [i,k] + [i,j]`, so the `(i,k)` boundary slot should
enter the reverse `(k,i)` edge with a negative incidence coefficient. E141
adds a parameter-neutral signed version of the face-cyclic readout with the
same runtime-ramp pattern. It is the more faithful topology test of the
E138 idea, not a metric-side loss. It is now running on a separate owned A100
Runpod pod, leaving E140 undisturbed on its original pod.
The parked E141/E142/E143 Runpod checkouts have been fast-forwarded to
`fd65f74` so future launches inherit the live `status_<variant>.json`
heartbeat; their candidate-specific topology flags and effective batch size
`8` were rechecked by the remote parser.
E140 was staged as `/workspace/SimplexFold_e140`, then launched after E139 was
documented as stopped pre-eval. Remote py_compile passed, the full E128-style
launch recipe parses with effective batch size `8`, and the architecture audit
remains within cap at `3,240,738 <= 3,261,974`.
E144 is also staged as `/workspace/SimplexFold_e144`: it is a no-Hodge
boundary edge-star residual readout from the E128 checkpoint. This projects
the selected boundary 1-cochain away from residue edge-star common modes
without adding parameters, losses, or dense all-pairs supervision. It should
remain parked behind E140 and the signed-boundary candidates.

Earlier, E120 became the primary-lDDT leader at `val_lddt_ca=0.4248` at step 7500.
It continued the selected-complex global-context family by combining the best
E118 residue vertex-star route with a half-strength boundary-edge-star
correction. Remote and local coherence passed: `completed_steps=7500`, 1000
eval-detail rows, history ending at step 7500, `effective_batch_size=8`,
`parameters=3,201,970` under the `3,261,974` cap,
`simplex_global_context_scale=0.1`,
`simplex_vertex_star_context_scale=1.0`, vertex-star runtime scale `1.0`,
`simplex_edge_star_context_scale=1.0`, and the intended edge-star runtime ramp
from `0.0` to `0.5` over steps 7000-7500.

E119 tested the boundary-edge-star analogue from E118 and returned at step
7500 with `val_lddt_ca=0.4181`, FoldScore `0.3957`, `val_ca_drmsd=11.0494`,
and C-alpha Rg `11.8732 / 16.3091`. Remote and local coherence passed:
`completed_steps=7500`, 1000 eval-detail rows, history ending at step 7500,
`effective_batch_size=8`, `parameters=3,201,970` under the `3,261,974` cap,
`simplex_global_context_scale=0.1`, `simplex_edge_star_context_scale=1.0`,
vertex-star context disabled, and the intended edge-star runtime ramp from
`0.0` to `1.0` over steps 7000-7500.

Reject E119 as a 30,000-step candidate. Boundary-edge-star routing improved
FoldScore, dRMSD, C-alpha expansion, and selected face/tetra boundary lDDT
(`0.7428` / `0.7275`) versus E118, but the primary C-alpha lDDT slipped below
E118. E120 fixed that specific regression and improved the leader to `0.4248`,
but the family remains in the low-0.4 band. To reach `0.7` from the current
best near `0.425`, the model still needs roughly `+0.275` validation C-alpha
lDDT, so this is not a "train longer and hope" situation.

The next branch should stay topology-native and address the same diagnosis
more directly: explicit higher-rank cells learn good local selected-boundary
geometry, but the main trunk still fails to assemble a globally accurate
C-alpha trace. Prefer a mechanism that changes how selected face/tetra
cochains influence pair/edge geometry before the structure module, with a
clear short gate before any 30,000-step spend. Do not launch a longer E118 or
E119 continuation unless a new gate first breaks out of the low-0.4 lDDT band.

E120 returned as a short gate on owned Runpod pod `o1dy17ouv8w5mz`. It was not
a blind continuation and not a new loss: it resumed the E118 checkpoint at
step 7000, kept the winning vertex-star route at `1.0`, and ramped a partial
edge-star context from `0.0` to `0.5` over steps 7000-7500. Because the
adapter composes star contexts by interpolation, this tested a mixed
selected-complex context:

```text
selected F_ijk / U_ijkl -> residue vertex-star cochains
                         -> partial boundary-edge-star correction
                         -> active F_ijk / U_ijkl
                         -> selected boundary edges Z_ij
```

The hypothesis is narrow: E119 improved FoldScore, dRMSD, expansion, and
selected-boundary diagnostics but lost primary lDDT, while E118 had the best
primary lDDT. A half-strength edge-star pull might add E119's pair-interface
packing signal without overwriting the vertex-star assembly route. E120 did
beat E118 on primary `val_lddt_ca` and improved FoldScore, but it did not
break out of the low-0.4 band, so do not spend 30,000 steps on this branch as
is.

The active E120 run was launched before the local sparse edge-star performance
refactor. Future star-context gates should use the local branch's
`boundary_edge_star_context` path, which gathers the same boundary-edge star
cochains for selected target edges without materializing the dense `L x L`
star tensor. This is a topology-equivalent implementation improvement: it
keeps the selected boundary 1-skeleton communication pattern, adds no
parameters or losses, and is meant to make E121 or any E120 retry less
expensive.

The next prepared fallback is E121 pre-triangle simplex injection. The repeated
failure mode is that local selected-boundary geometry improves while global
C-alpha assembly stays weak. E121 keeps the same selected complex but lets a
scaled face/tetra boundary cochain update `Z_ij` before the pair triangle
multiplication/attention stack inside each enabled Evoformer block. That gives
AF2's own triangle machinery a chance to propagate simplex evidence globally
within the same block, rather than waiting for later blocks or the structure
module to assemble it. The hook is default-off and parameter-neutral. Because
E120 returned as a better but still low-0.4 leader, E121 is the next
topology-native short gate candidate, not an automatic 30k run.

E121 ran on owned Runpod pod `o1dy17ouv8w5mz`. It resumed E120's
step-7500 checkpoint and keeps the mixed selected-complex recipe fixed:
`simplex_global_context_scale=0.1`, vertex-star context `1.0`, edge-star
context `1.0` with runtime scale `0.5`, fixed sparse caps `24 / 48`,
boundary incidence normalization `1.0`, boundary readout directionality
`0.25`, and edge-frame runtime message scale `0.0125`. The only architecture
change is `--simplex-pre-triangle-update-scale 0.25`, which reuses the
existing simplex adapter before AF2 triangle updates. Launch metadata confirms
`steps=8000`, `effective_batch_size=8`, `parameters=3,201,970 <= 3,261,974`,
and a clean weights-only resume from the E120 checkpoint with all `1292`
model tensors loaded. The run failed before producing a new validation point:
PyTorch activation checkpointing recomputed the dynamic pre-triangle selected
complex with different packed tensor lengths, so no E121 metric should be
treated as returned.

Corrective plan: rerun the same scientific gate as E121b after the local fix
that executes active pre-triangle simplex blocks eagerly during training. This
does not change the experimental hypothesis, parameters, loss, selected
complex, or launch recipe; it only avoids activation-checkpoint recomputation
over variable-size selected-cell tensors.

E121b returned on the owned Runpod pod `o1dy17ouv8w5mz` as
`e121b_pre_triangle_eager_from_e120_s8000_c256_m64`, with the same E120
checkpoint, step-8000 target, effective batch size 8, and parameter count
`3,201,970 <= 3,261,974`. Remote and local coherence passed after a post-hoc
FoldScore repair from the saved checkpoint: `completed_steps=8000`, one
result row, 1000 eval-detail rows, history ending at step 8000,
`stopped_early=False`, and the expected E120/E121b metadata. The result is a
reject: `val_lddt_ca=0.4223`, FoldScore `0.4007`,
`val_ca_drmsd=11.1491`, and C-alpha Rg `11.8330 / 16.3091`. It is below
E120's `0.4248` primary lDDT and below the `0.45` short-gate threshold, so do
not spend 30,000 steps on abrupt eager pre-triangle injection.

E123 returned on the owned Runpod pod `o1dy17ouv8w5mz` as
`e123_ramped_pair_pre_triangle_from_e120_s8000_c256_m64`. Remote and local
coherence passed: `completed_steps=8000`, one result row, 1000 eval-detail
rows, history ending at step 8000, `effective_batch_size=8`,
`parameters=3,201,970 <= 3,261,974`, `stopped_early=False`, and the intended
pair-only runtime ramp metadata. The result is a tiny new primary-lDDT leader:
`val_lddt_ca=0.4270`, FoldScore `0.3992`, `val_ca_drmsd=11.1927`, and
C-alpha Rg `11.4700 / 16.3091`. It beats E120's `0.4248` C-alpha lDDT but
does not clear the `0.45` short-gate threshold, worsens dRMSD, and softens
selected face/tetra boundary geometry (`0.7447` / `0.7280` boundary lDDT and
`0.6505` / `0.6494` contraction). Do not spend 30,000 steps on E123.

E124 face boundary-edge-frame gating is now returned. This is not a
generic output loss: it asks whether a learned 2-simplex can communicate more
usefully through its own oriented boundary 1-simplices by gating selected
face-to-edge messages with edge-frame scalarized face geometry. Keep it as a
500-step topology-communication probe from the E120 checkpoint at small scale.
It returned at step 8000 with `val_lddt_ca=0.4280`, FoldScore `0.3979`,
`val_ca_drmsd=11.2529`, and C-alpha Rg `11.3075 / 16.3091`. Remote and local
coherence passed with `parameters=3,239,522 <= 3,261,974`,
`effective_batch_size=8`, one result row, 1000 eval-detail rows, and history
ending at step 8000. E124 is a tiny new primary-lDDT leader over E123
(`0.4270`) and E120 (`0.4248`), and it improves selected face/tetra boundary
lDDT (`0.7583` / `0.7406`) plus contraction (`0.5614` / `0.5611`), but it
remains below the `0.45` short-gate threshold and worsens FoldScore/dRMSD
versus E123/E120. Do not spend 30,000 steps on E124.

E125 has now returned:
`simplex_boundary_edge_frame_gate_runtime_scale` lets the same oriented
face-boundary edge-frame gate ramp from `0.0` to the configured gate scale
instead of switching on abruptly when resuming from E120. This stays inside the
same selected-complex view as E124. The selected face cochain still writes
through its directed boundary 1-simplices using edge-frame scalarized geometry;
the only change is treating that inter-rank gate as a topology curriculum.
Because E124 improved local selected-complex geometry but worsened global
FoldScore/dRMSD, E125 should test whether a smoother handoff preserves the
local boundary signal without over-contracting the global C-alpha trace. It
returned at step 8000 with `val_lddt_ca=0.4275`, FoldScore `0.3986`,
`val_ca_drmsd=11.3161`, and C-alpha Rg `11.2998 / 16.3091`. Remote and local
coherence passed with `parameters=3,239,522 <= 3,261,974`,
`effective_batch_size=8`, one result row, 1000 eval-detail rows, and history
ending at step 8000. The ramp slightly improves FoldScore versus E124 but
reduces primary C-alpha lDDT, dRMSD, selected-boundary lDDT, and contraction.
Do not spend 30,000 steps on E125 or another plain boundary-edge-frame
schedule.

The pair/edge-trunk direction remains the most relevant backlog. E100 showed
that collapsed cell-to-residue MSA feedback is too blunt; E101 showed that
preserving directed boundary-edge incidence helps relative to E100 but still
does not beat the E96/E97 leaders. A future candidate should therefore use
selected boundary-edge cochains as incidence-aware pair/edge bias or gating,
so the explicit face/tetra complex changes how pair geometry is updated
before structure readout instead of asking MSA feedback to carry that signal.

E102 tested that pair/edge feedback target by lifting selected boundary-edge
cochains densely back to all `L x L` pairs. It was stopped as a performance
failure before returning a new result: after roughly 42 minutes on the owned
H100 pod it had not produced a new history row beyond the inherited E97
step-9500 row, and `results.json` was absent. Do not add E102 to
`EXPERIMENT_RESULTS.md`; it is an aborted implementation diagnostic, not a
completed experiment.

E103 keeps E102's topological claim but removes the dense all-pairs lift. It
adds `simplex_boundary_pair_gate_scale`: each selected boundary-edge cochain
induced by explicit face/tetra cells is modulated by a learned gate conditioned
on that same edge's current pair state `Z_ab` before the edge update is
scattered back through the selected 1-skeleton. This is still a simplicial
architecture change, not an output-side lDDT hack: higher-order cells alter
pair geometry through their boundary edges, while computation stays on the
sparse selected complex.

E103 returned at step 10000 with `val_lddt_ca=0.3981`, FoldScore `0.3909`,
and `val_ca_drmsd=9.8275`. Reject it as a primary-lDDT branch: the sparse
pair gate improved FoldScore/dRMSD, but it fell below E96, E97, E99 final,
and E101 on the target C-alpha lDDT. This is useful evidence that direct
learned pair-conditioned boundary-edge modulation is not the missing 30k
candidate.

E104 tested a selected-boundary metric-confidence gate and is now rejected:
it returned `val_lddt_ca=0.3956`, below E96, E97, E99, E101, and E103,
despite strong selected face/tetra boundary lDDT (`0.7246` / `0.7072`).
That result sharpens the diagnosis: the explicit simplex complex can learn
local boundary metrics, but those local metrics are not yet being assembled
into a better full-chain C-alpha geometry.

E105a tested selected-boundary metric recycling. The
plateau evidence suggests that the selected complex learns reasonable local
boundary geometry, but the main trunk only recycles the final coordinate
prediction. E105/E105a reuses the existing
face/tetra distance heads, maps their selected boundary-edge distance
distributions softly into the AF2 recycling distance-bin basis, scatters that
evidence only onto the selected boundary 1-skeleton, and adds it as a
no-new-parameter bias to `z_prev` for the next recycle cycle. This is not a
new output loss; it changes the inter-cycle cochain memory so explicit
higher-rank metric states can influence the next pair-trunk pass before the
structure module. Use the runtime recycling-scale ramp rather than an abrupt
static turn-on when resuming from E97/E72.

Checkpoint caveat: E104 artifacts were pulled with checkpoint directories
excluded, and restarting the zero-volume Runpod pod cleared `/workspace`, so
the E97/E96-family checkpoint is no longer available locally or remotely.
The strongest retained compatible checkpoint was E72 at step 5500. E105a ran
from E72 to step 6000 on the owned Runpod pod `o1dy17ouv8w5mz` and returned
`val_lddt_ca=0.3894`, FoldScore `0.3737`, and `val_ca_drmsd=10.7410`.
It improves the retained E72/E73/E74/E76 recovery band but remains below the
E96 primary leader, so it is a recovery handoff rather than a 30k candidate by
itself.

E106 tested selected-boundary cochain recycling. E104 showed that local
face/tetra boundary metrics can be strong without improving global C-alpha
lDDT, and E105/E105a tested recycling only the distance-distribution view of
that selected 1-skeleton. E106 recycles the learned selected-boundary pair
cochain itself: the existing simplex face/tetra-to-boundary readout is
detached, masked to valid residue pairs, and added to `z_prev` between AF2
recycle cycles. This keeps the intervention in the README's simplex view
because persistent rank-2/rank-3 cells influence the next pair-trunk pass
through their boundary 1-cochain. It adds no parameters and no new loss.

E106 returned at step 6500 with `val_lddt_ca=0.3929`, FoldScore `0.3777`,
`val_ca_drmsd=10.3279`, and predicted/true C-alpha radius
`11.2713 / 15.4034`. This improves E105a's `0.3894` C-alpha lDDT and also
improves FoldScore, dRMSD, and expansion, so the cochain-memory route is not
an immediate reject. It remains below E96's `0.4043`, so it is still a
recovery-branch signal rather than a 30k candidate by itself.

E108 continued E106 from the verified step-6500 checkpoint to step 7000 with
the same selected-complex recipe and selected-boundary cochain recycling held
at `0.10`. It returned `val_lddt_ca=0.3875`, FoldScore `0.3771`,
`val_ca_drmsd=10.6170`, and predicted/true C-alpha radius
`11.3118 / 15.4034`. Reject E108: holding the raw cochain memory did not
maintain the E106 improvement and regressed below E106 on primary C-alpha
lDDT, FoldScore, and dRMSD.

E107 tested metric-gated selected-boundary cochain recycling from the better
verified E106 checkpoint. It returned `val_lddt_ca=0.3868`, FoldScore
`0.3757`, `val_ca_drmsd=10.6490`, and predicted/true C-alpha radius
`11.1116 / 15.4034`. Reject E107: confidence-gating the recycled cochain fell
below both E106 and E108, so the failure is not simply that uncertain selected
cell cochains were recycled too strongly.

E109 tested cochain-memory anneal-down from the verified E106 checkpoint. It
returned `val_lddt_ca=0.3909`, FoldScore `0.3798`, `val_ca_drmsd=10.3292`,
and predicted/true C-alpha radius `11.5503 / 15.4034`. This partially
recovered the E107/E108 regression and improved FoldScore over E106, but it
still did not beat E106 on primary C-alpha lDDT.

E110 tested full cochain-memory release from the verified E106 checkpoint.
It resumed E106 and annealed selected-boundary cochain recycling from `0.10`
to `0.0` over steps 6500-7000. It returned `val_lddt_ca=0.3816`,
FoldScore `0.3788`, `val_ca_drmsd=10.3738`, and predicted/true C-alpha
radius `11.7781 / 15.4034`. Reject E110: full release to zero fell below
E106, E107, E108, and E109 on primary C-alpha lDDT. The failure is therefore
not just residual cochain memory at validation; the current cochain-recycling
family is not a 30k candidate.

E111 tested an RMS-normalized pair-only structure-module readout from the
selected boundary 1-cochain. It returned `val_lddt_ca=0.3920`, FoldScore
`0.3759`, `val_ca_drmsd=10.4197`, and predicted/true C-alpha radius
`11.3424 / 15.4034`. Reject E111 as a primary branch: it recovered most of
E110's release-to-zero drop, but it remained below E106's `0.3929` on primary
C-alpha lDDT and also softened FoldScore and dRMSD. The local selected-boundary
diagnostics stayed strong, so the failure is not loss of the explicit complex;
the structure-module pair bias is likely too strong or still aimed too late in
the trunk.

E112 tested the same pair-only structure bias at half scale. It returned
`val_lddt_ca=0.3873`, FoldScore `0.3793`, `val_ca_drmsd=10.3890`, and
predicted/true C-alpha radius `11.5702 / 15.4034`. Reject E112: lowering the
structure-module pair bias worsened primary C-alpha lDDT below both E106 and
E111. The structure-bias route is therefore not the next 30k candidate.

E113 reintroduced directed boundary readout from the verified E106 checkpoint
and annealed it from `0.5` to `0.25` over the 6500-7000 gate. It returned
`val_lddt_ca=0.3959`, FoldScore `0.3775`, `val_ca_drmsd=10.6305`, and
predicted/true C-alpha radius `11.1660 / 15.4034`. Keep it as a
recovery-branch handoff because it beats E106/E111/E112 on primary C-alpha
lDDT, but do not treat it as a 30k candidate: it remains below the E96/E97
band and worsens dRMSD.

E114 tested segment-supported sparse-cell filtration from the E113 step-7000
checkpoint to step 7500. The branch stayed in the README's
simplicial/topological view by changing which selected face/tetra cells exist,
not by adding an output-side metric loss. It returned `val_lddt_ca=0.3814`,
FoldScore `0.3793`, `val_ca_drmsd=10.6123`, and predicted/true C-alpha radius
`11.8583 / 15.4034`. Reject E114 as a primary branch: segment-supported
filtration improved FoldScore, dRMSD, expansion, and contraction, but it
damaged primary C-alpha lDDT and softened selected-boundary lDDT/length
diagnostics.

E115 then ran the clean no-segment continuation control from the same E113
checkpoint to step 7500. It returned `val_lddt_ca=0.3820`, FoldScore `0.3771`,
`val_ca_drmsd=10.3770`, and predicted/true C-alpha radius
`11.5707 / 15.4034`. Reject E115: it fell below E113 and E106 and nearly
matched E114's primary-lDDT drop, so E114's failure was not mainly the new
segment-supported scorer. The E113 recovery branch itself is not stable enough
for another local-filtration tweak or a blind 30,000-step spend.

E116 is implemented as a stronger topology-native branch rather than another
local filtration tweak. The motivation is the same failure pattern seen from
E96 through E115: selected face/tetra boundary diagnostics can become strong
while global C-alpha assembly remains near `0.40`. E116 adds
`simplex_global_context_scale`, a selected-complex global cochain: each
SimplicialAdapter pools only active face and tetra states into a protein-level
summary, then routes that summary back into the active face/tetra cells before
their boundary-edge readout. This keeps the intervention inside the README's
`Z_ij <-> F_ijk <-> U_ijkl` view; it is not an output-side lDDT, radius, or
all-pairs distance loss.

The E116 launch-style parameter audit for the E72 sparse recipe plus global
context is `3,201,970`, still under the AF2-medium +5% cap of `3,261,974`.
E116 returned `val_lddt_ca=0.4095`, FoldScore `0.3881`,
`val_ca_drmsd=11.2964`, and predicted/true C-alpha radius
`11.5918 / 16.3091`. The selected face/tetra boundary lDDT stayed strong at
`0.7232 / 0.7095`, while contraction fractions improved to roughly
`0.602 / 0.599`. Continue this route with E117 rather than pivoting: the
mechanism directly addresses the local-to-global gap and finally moved the
primary metric.

E128 is the current short-gate leader after combining E124's oriented
face-boundary-edge-frame gate with a damped sparse simplex triangle-attention
bias. It returned at step 8500 with `val_lddt_ca=0.4311`, FoldScore `0.4025`,
`val_ca_drmsd=11.0046`, and predicted/true C-alpha radius
`11.7198 / 16.3091`, with local verification passing at
`3,240,738 <= 3,261,974` parameters. This keeps supporting the central
topological hypothesis that explicit selected face/tetra cochains help when
they communicate through boundary 1-simplices and represented triangles. It is
still not a 30,000-step candidate: the gain over E124 is small, the run has not
cleared the `0.45` short-gate threshold, and the target remains `>0.7`. The
next plan should focus on translating strong selected-complex geometry into
global C-alpha assembly, not just making triangle-attention bias stronger.

## Historical Plan Context

E44-E52 show that closure masks, broad structure readouts, stronger auxiliary
expansion, and selected-cell dropout do not break the C-alpha lDDT plateau.
E53 then showed that the strongest branch, E15's `full_msa_to_face` selected
simplex realization scaffold, can catch up under effective batch 8 if it is
allowed to run past the first 500 optimizer steps.

E53 reached `val_lddt_ca=0.3480` at step 1000, E54 recovered to
`val_lddt_ca=0.3539` at step 2000 after the auxiliary anneal, and E55 reached
the new best `val_lddt_ca=0.3604` at step 3000 while improving FoldScore to
`0.3451`. E56 continued the same checkpoint lineage to step 4000 and improved
FoldScore/dRMSD further, but its best `val_lddt_ca=0.3575` stayed below E55.
E57 then tried a selected-simplex auxiliary rewarm at `0.75`; it improved the
global/FoldScore side but reduced lDDT to `0.3465`. E58-E62 tested directed
outer-edge, boundary-frame, and Hodge-style incidence routes from the E55
checkpoint. These runs improved pieces of global or selected-complex geometry
but did not preserve E55's lDDT peak.

E63 resumed E55 and added a conservative selected-boundary lDDT objective
only on boundary edges induced by the model-selected face/tetra cells. It
reached `val_lddt_ca=0.3611`, FoldScore `0.3576`, `val_ca_drmsd=10.6815`,
and predicted/true C-alpha radius `11.4310 / 15.4034`. The topological
diagnostics moved in the intended direction too: selected face/tetra boundary
lDDT rose to `0.5208` / `0.5065`, and contraction fractions fell to roughly
`0.69`.

E64 became the selected-boundary lDDT reference. It continued E63 to step 4000 with the
same selected-boundary lDDT weights and reached `val_lddt_ca=0.3739`,
FoldScore `0.3634`, `val_ca_drmsd=10.5481`, and predicted/true C-alpha
radius `11.3344 / 15.4034`. This confirms that the selected-boundary lDDT
direction is not just a one-checkpoint fluctuation, though the target remains
far away.

E65 tested whether the selected-boundary lDDT pressure could relax after E64.
It continued E64 to step 5000, holding face/tetra boundary-lDDT weights at
`0.05` through step 4500 and then ramping to `0.025`. Reject this schedule:
step 4500 reached `val_lddt_ca=0.3645`, and step 5000 reached
`val_lddt_ca=0.3684`, both below E64. FoldScore improved slightly to
`0.3666`, but the primary C-alpha lDDT target did not survive the
continuation.

E66 tested coface-balanced selected-boundary lDDT from E64 by enabling
`--simplex-boundary-degree-normalize`. Reject it: the step-4500 result dropped
to `val_lddt_ca=0.3505`, FoldScore `0.3602`, and selected face/tetra boundary
lDDT `0.5090` / `0.4948`. Boundary contraction also worsened to about `0.714`,
so inverse coface-degree weighting weakens the selected-boundary signal rather
than making it cleaner.

E67 tested a weak selected-complex structure readout from E64 with
`--simplex-structure-readout-scale 0.05`. Reject it as a continuation branch:
it reached `val_lddt_ca=0.3647`, essentially tied with E65 step 4500 and still
below E64, while FoldScore fell to `0.3619`. It did improve dRMSD to `10.3503`
and boundary length MAE to `2.6833` / `2.8167`, so the readout path has useful
geometry signal but needs less coupling to avoid hurting local lDDT.

E68 damped that selected-complex structure readout to `0.025`. Reject it too:
step 4500 reached `val_lddt_ca=0.3617`, FoldScore `0.3625`,
`val_ca_drmsd=10.2115`, and predicted/true C-alpha radius
`11.9645 / 15.4034`. This is the best dRMSD in the E64 continuation family,
but it further reduces the primary lDDT and confirms that weaker structure
readout is not enough to preserve local C-alpha agreement.

E69 tested selected face normal orientation from E64 with
`--simplex-face-normal-weight 0.05`. Reject it: step 4500 reached
`val_lddt_ca=0.3653`, FoldScore `0.3632`, `val_ca_drmsd=10.5833`, and
predicted/true C-alpha radius `11.8750 / 15.4034`. The normal term was active,
but selected face/tetra boundary lDDT fell to `0.5210` / `0.5059`, so
orientation supervision alone weakens the selected-complex boundary geometry
instead of improving the E64 peak.

E70 then became the lDDT leader by a very small margin. It continued E64 with
damped edge-frame boundary messages and reached `val_lddt_ca=0.3742`,
FoldScore `0.3653`, `val_ca_drmsd=10.3425`, and predicted/true C-alpha radius
`11.4815 / 15.4034`. The selected-boundary diagnostics also improved over E64:
face/tetra boundary lDDT rose to `0.5365` / `0.5215`, contraction fractions
fell to `0.6665` / `0.6681`, and boundary length MAE fell to
`2.6313` / `2.7606`.

E71 continued the same edge-frame path to step 5000 and improved the primary
curve again: `val_lddt_ca=0.3751`, FoldScore `0.3679`,
`val_ca_drmsd=10.1926`, and predicted/true C-alpha radius
`11.4483 / 15.4034`. This is still far from the `0.7` target, but it is the
first post-E64 continuation family to improve lDDT, FoldScore, and dRMSD
together. The caveat is that selected-boundary lDDT softened from E70 to
`0.5336` / `0.5181`, so the next continuation should watch for divergence
between main metrics and selected-complex diagnostics.

E72 continued E71 to step 5500 with the same runtime edge-frame scale `0.025`.
Reject it as a primary-lDDT continuation: `val_lddt_ca` fell to `0.3718`,
below E71, even though FoldScore rose to `0.3722`, dRMSD improved to
`10.1027`, predicted/true C-alpha radius opened to `12.0872 / 15.4034`, and
selected face/tetra boundary lDDT improved to `0.5450` / `0.5303`. The
selected complex is becoming more geometrically coherent, but the full-strength
edge-frame cochain exchange appears to be shifting the model toward global
expansion at the expense of local C-alpha lDDT.

Do not spend on a blind 30,000-step continuation yet, and do not keep turning
the scalar auxiliary knob. The full reread of the reference PDFs in
`references/papers/` points back to the core topological claim: the model
should improve by changing the cell complex and its multi-rank message routes,
or by supervising realization of the selected sparse complex, not by attaching
generic metric pressure to the output coordinates.

E60 tested that idea by scheduling the damped directed outer-edge context from
`0.0` to `0.05` over steps 3000-3500. It completed at
`val_lddt_ca=0.3462`, FoldScore `0.3431`, `val_ca_drmsd=10.9235`, and
predicted/true C-alpha radius `10.8522 / 15.4034`. Reject it: the schedule did
not preserve the E55/E56 lDDT band and also gave up E59's FoldScore gain.

E61 shifted from cell-level outer-edge summaries to boundary-edge
scalarization. It completed at step 3500 with
`val_lddt_ca=0.3456`, FoldScore `0.3471`, `val_ca_drmsd=10.7730`, and
predicted/true C-alpha radius `11.1613 / 15.4034`. Reject it: the scheduled
edge-frame message path improved global expansion and dRMSD relative to E55
but moved lDDT back into the E57/E60 band. The launch used:
`--simplex-edge-frame-message-scale 0.05`,
`--simplex-edge-frame-message-runtime-scale 0.0`,
`--simplex-edge-frame-message-runtime-scale-final 0.05`,
`--simplex-edge-frame-message-runtime-scale-ramp-start-step 3000`, and
`--simplex-edge-frame-message-runtime-scale-ramp-steps 500`. The launch audit
passed with public train/val/all counts `10000/1000/11000`, hidden manifest
absent, feature/label NPZ counts `11000/11000`, encoded missing paths `0`,
FoldScore import working, and `3,154,242` parameters (`+1.53%` versus
AF2-medium).

The runner should keep `EXPERIMENT_RESULTS.md` only for returned Runpod
results. Do not launch a 30,000-step confirmation until a branch clears the
lDDT target direction under effective batch 8.

The runner should also keep reporting selected-boundary diagnostics for future
runs: face/tetra boundary-edge length MAE/RMSE, contraction fraction, boundary
lDDT, selected-cell counts, and boundary-edge reuse. These are diagnostics of
the learned sparse complex, not training objectives.

E73 reran the E71-to-5500 continuation with the same edge-frame architecture
but half the runtime cochain-exchange scale (`0.0125`). It returned a new best:
`val_lddt_ca=0.3807`, FoldScore `0.3720`, `val_ca_drmsd=10.0777`, and
predicted/true C-alpha radius `11.6741 / 15.4034`. This beats E71's
`0.3751` local lDDT and slightly improves E72's FoldScore/dRMSD trend.

The caveat is diagnostic: E73's selected face/tetra boundary lDDT
`0.5368` / `0.5213` is below E72's `0.5450` / `0.5303`, and boundary-edge
reuse remains high. The half-scale edge-frame route therefore preserves local
C-alpha lDDT better than E72, but it does not solve the dense/reused selected
complex problem.

E76 continued E73 from step 5500 to step 6000 with the same half-scale
selected boundary-edge message recipe. Reject it as a primary-lDDT
continuation: step 6000 fell to `val_lddt_ca=0.3713`, below E73's `0.3807`.
FoldScore edged up to `0.3723`, but `val_ca_drmsd` worsened to `10.2191`,
predicted/true C-alpha radius moved to `12.0036 / 15.4034`, and selected
face/tetra boundary lDDT softened to `0.5341` / `0.5185`.

E77 tested coface-degree attenuation on the selected boundary-edge readout.
Reject it as a primary-lDDT branch: step 6000 reached `val_lddt_ca=0.3733`,
below E73's `0.3807`, and FoldScore softened to `0.3710`. The diagnostic is
still useful: selected face/tetra boundary lDDT improved to `0.5421` /
`0.5265`, and boundary length MAE improved to `2.5714` / `2.7039`. Degree
attenuation therefore cleans up selected-boundary realization but is not
enough by itself to preserve local C-alpha agreement.

E74 tested the prepared topology-construction alternative: keep the E73/E77
loss and edge-frame recipe, but reduce the recycled-geometry selector weight
to `0.025` so learned pair/contact topology has more control over which sparse
face/tetra cochains exist. Keep it as the new primary-lDDT leader. Step 6000
reached `val_lddt_ca=0.3841`, above E73's `0.3807`, with selected face/tetra
boundary lDDT improving to `0.5409` / `0.5258` and boundary length MAE
improving to `2.5149` / `2.6510`. The caveat is that FoldScore softened to
`0.3666` and dRMSD to `10.1893`, so the branch still needs continuation and
diagnostic monitoring rather than a 30k-step commitment.

E78 continued E74 to step 6500 with the same light-geometry selector,
selected boundary-edge losses, and half-scale edge-frame message recipe. Keep
it as the new primary-lDDT leader: `val_lddt_ca=0.3853`, FoldScore `0.3718`,
`val_ca_drmsd=10.1595`, and predicted/true C-alpha radius
`11.3783 / 15.4034`. Selected face/tetra boundary lDDT improved to
`0.5434` / `0.5287`, and boundary length MAE improved to `2.4519` /
`2.5801`. Boundary contraction rose to `0.6320` / `0.6327`, so the selected
complex is cleaner by lDDT/length but still over-contracts a large fraction
of its boundary edges.

E80 continued E78 to step 7000 with the same light-geometry
topology-construction recipe. Reject it as a primary branch:
`val_lddt_ca=0.3820`, FoldScore `0.3682`, and `val_ca_drmsd=10.2493`, all
below E78. Selected face/tetra boundary lDDT also fell to `0.5359` /
`0.5192`, and boundary length MAE worsened to `2.6560` / `2.8001`. This
confirms that the E78 gain was a local peak rather than a reason to keep
blindly continuing the light-geometry selector.

E79 became the new best by changing the active higher-rank cell complex rather
than adding output-side pressure. It resumed E78 from step 6500 to 7000 and
scheduled the selected complex from the full neighbor-star clique toward `24`
face cells and `48` tetra cells per anchor. Keep it: `val_lddt_ca=0.3885`,
FoldScore `0.3728`, and selected face/tetra boundary lDDT `0.6963` /
`0.6826`. Selected boundary length MAE improved sharply to `1.2635` /
`1.3586`, tetra boundary-edge mean degree dropped to `35.6`, and tetra
unique-edge fraction rose to `0.0283`. The caveat is that dRMSD softened to
`10.2661` and predicted radius stayed under-expanded at `11.1540 / 15.4034`.

E82 answered the fixed-sparse-complex question positively. It continued E79
from step 7000 to 7500 with the sparse caps held at `24` face cells and `48`
tetra cells per anchor and reached a new best `val_lddt_ca=0.3924`,
FoldScore `0.3788`, and `val_ca_drmsd=10.2523`. Selected face/tetra boundary
lDDT improved again to `0.7135` / `0.6987`, boundary length MAE improved to
`1.1560` / `1.2579`, and tetra boundary-edge mean degree dropped to `34.2`.
The caveat remains under-expansion: predicted/true C-alpha radius is
`11.3363 / 15.4034`, still far from solved.

E83 showed that simply holding the fixed sparse caps is not enough. It resumed
E82 from step 7500 to 8000 with the same sparse caps and fell to
`val_lddt_ca=0.3876`, FoldScore `0.3747`, and `val_ca_drmsd=10.3539`.
Selected face/tetra boundary lDDT also softened to `0.7034` / `0.6881`, and
boundary length MAE worsened to `1.2296` / `1.3345`. Reject this continuation
as a primary branch.

E81 answered the degree-penalized sparse-cell question positively. It resumed
the stronger E82 checkpoint from step 7500 to 8000 with the same fixed sparse
caps plus `--simplex-cell-score-degree-penalty 0.75`, and reached a new best
`val_lddt_ca=0.3980`, FoldScore `0.3826`, `val_ca_drmsd=10.0954`, and
predicted/true C-alpha radius `11.4973 / 15.4034`. The selected-complex
diagnostics improved too: face/tetra boundary lDDT rose to `0.7335` /
`0.7178`, boundary length MAE fell to `1.0733` / `1.1727`, contraction
fraction fell to `0.5781` / `0.5791`, and unique boundary-edge fraction rose
to `0.0856` / `0.0304`. This supports the hypothesis that high boundary-edge
reuse is a real topological construction failure mode.

E84 tested whether the E81 degree-penalized sparse selector would keep climbing
to step 8500, but it regressed to `val_lddt_ca=0.3964`, FoldScore `0.3767`,
and `val_ca_drmsd=10.4047`. E85 then resumed the stronger E81 checkpoint from
step 8000 to 8500 and added incidence normalization inside selected
edge-face-tetra cochain transport. Reject E85: it fell to
`val_lddt_ca=0.3858`, FoldScore `0.3767`, and selected face/tetra boundary
lDDT `0.7265` / `0.7090`, without reducing boundary-edge reuse.

E86 returned as the new tiny best at `val_lddt_ca=0.3990`, FoldScore
`0.3858`, and `val_ca_drmsd=10.0281`. It resumed the strongest sparse-complex
checkpoint, E81, and added a deliberately weak directed outer-edge context
route by allocating `simplex_outer_edge_context_scale=0.05` but ramping
runtime contribution only from `0.0` to `0.025` during the 8000-8500 gate.
The next active gate should continue E86 to step 9000 with the weak
outer-edge runtime contribution held at `0.025`. This is still a short
topology-communication gate, not a commitment to a blind 30,000-step run. Do
not launch a blind 30,000-step confirmation until a branch shows a credible
trajectory toward `val_lddt_ca > 0.7`, not merely a small local best below
0.4.

E91 tested that continuation and should be rejected as a primary-lDDT branch.
It reached `val_lddt_ca=0.3897`, below E86's `0.3990`, while improving
`val_ca_drmsd` to `9.9309` and selected face/tetra boundary lDDT to
`0.7414` / `0.7256`. This is useful evidence: weak outer-edge communication
continues to improve global/selected-complex geometry, but it no longer
preserves local C-alpha agreement. The next active gate should therefore pivot
to E87 directed boundary readout from the strongest sparse-complex checkpoint,
so the simplex cochains can write source/target-aware information into
`Z_ij` without adding a new loss.

E87 is now running from the cleaner E81 checkpoint, not the regressed E91
checkpoint, to isolate the source/target boundary-readout mechanism. It ramps
`simplex_boundary_readout_directionality` from `0.0` to `0.5` over steps
8000-8500 while preserving the fixed `24` / `48` sparse caps, degree-penalized
cell scoring, selected-boundary realization losses, half-scale edge-frame
messages, and incidence-normalized cochain transport. Keep it only if primary
`val_lddt_ca` recovers against E81/E86 without selected-boundary collapse.

E87 returned a tiny new primary-lDDT best: `val_lddt_ca=0.3992`, just above
E86's `0.3990`, with selected face/tetra boundary lDDT improving to
`0.7446` / `0.7280` and contraction fractions returning to about `0.576` /
`0.579`. The caveat is that FoldScore softened to `0.3831` and dRMSD worsened
to `10.2428`, so this is not a 30k-step confirmation branch. Run one short
E92 continuation with the directed boundary-readout runtime scale held at
`0.5`. Keep it only if local lDDT continues upward or at least stays at the
E87/E86 level without selected-boundary collapse; otherwise pivot to the
outer-edge-supported cell scorer.

E92 rejected the directed boundary-readout continuation. It resumed E87 to
step 9000 with directionality held at `0.5` and returned
`val_lddt_ca=0.3968`, below both E87's `0.3992` and E86's `0.3990`.
FoldScore stayed roughly flat at `0.3829` and dRMSD improved to `9.9617`, but
the written rule was primary-lDDT preservation. Stop continuing the same
readout mechanism.

E90 also rejected as a primary-lDDT branch. It resumed E81 to step 8500 and
ramped `simplex_cell_score_outer_edge_weight` from `0.0` to `0.25`.
Selected-boundary contraction improved to about `0.546`, but
`val_lddt_ca=0.3920`, FoldScore `0.3783`, and selected-boundary lDDT
`0.7365` / `0.7197` all stayed below the E81/E86/E87 leaders. Outer-edge
availability is therefore not enough as a cell-score bonus.

E88 rejected the runtime-gated latent segment-cell route in this combined
form. It resumed E81, allocated latent contiguous segment cochains, and ramped
their contribution into selected face states, but returned
`val_lddt_ca=0.3891`, below E81/E86/E87. More importantly, the actual
launched module combination had `3,282,002` parameters, which exceeds the
AF2-medium +5% ceiling of `3,261,974`. Treat this as a budget failure as well
as a primary-lDDT failure. Segment cells can be reconsidered only after a
budget-safe topology-module combination is explicitly counted before launch.

E89 rejected the pair-preserving simplex readout route as a primary-lDDT
branch. It stayed within budget and improved FoldScore to `0.3861`, but
returned `val_lddt_ca=0.3947`, below E81/E86/E87. Pair-first cochain readout
therefore helps aggregate geometry a bit but does not recover the local
C-alpha objective.

E93 rejected as a primary-lDDT branch, but it is diagnostically useful. It
tightened the selected sparse complex from `24/48` to `12/24` and returned
`val_lddt_ca=0.3973`, below E81/E86/E87, with FoldScore `0.3819`,
`val_ca_drmsd=10.2949`, and predicted/true C-alpha radius
`11.0952 / 15.4034`. The selected complex itself became much cleaner:
selected face/tetra boundary lDDT rose to `0.7897` / `0.7549`, boundary
length MAE fell to `0.8182` / `0.9820`, and boundary-edge mean degree fell to
`9.29` / `26.09`. The interpretation is that `12/24` is too narrow by itself:
it realizes a cleaner selected complex but loses enough higher-rank context
to under-expand the global structure and soften primary C-alpha lDDT.

E94 also rejected the filtration route. It combined E87's directed
source/target boundary readout with a gentler `24/48 -> 18/36` filtration and
returned `val_lddt_ca=0.3914`, below E81, E86, E87, and E93. It did reduce
selected-boundary contraction to `0.5057` / `0.5157`, but boundary-edge reuse
stayed high and selected-boundary lDDT `0.7600` / `0.7294` did not approach
E93. The interpretation is that further cap tightening, even moderate
tightening, is trading away local C-alpha agreement faster than it improves
the useful selected complex.

E95 also rejected as a primary-lDDT branch. It kept the broader E81 `24/48`
sparse complex and stacked weak directed outer-edge context with directed
boundary readout. The result was `val_lddt_ca=0.3931`, below both E86 and
E87, even though `val_ca_drmsd=9.9984` was the best local dRMSD. The selected
face/tetra boundary lDDT softened to `0.7295` / `0.7140`, so stacking the two
communication routes improves one global geometry metric while interfering
with local C-alpha agreement.

E96 is now the primary-lDDT leader. It treated E87's directed boundary readout
as a curriculum rather than a permanent setting, resuming E87 from step 8500
to 9000 and ramping boundary-readout directionality down from `0.5` to
`0.25`. The result was `val_lddt_ca=0.4043`, FoldScore `0.3852`,
`val_ca_drmsd=10.1973`, and predicted/true C-alpha radius
`11.2733 / 15.4034` with `3,154,242` parameters. This beats E87's
`0.3992` primary lDDT and avoids E92's held-`0.5` regression.

E98 rejected the partial directed-readout continuation. It resumed E96 from
step 9000 to 9500 while holding boundary-readout directionality at `0.25`, but
returned `val_lddt_ca=0.3939`, below E96's `0.4043`, with FoldScore `0.3807`,
`val_ca_drmsd=10.0459`, and predicted/true C-alpha radius
`11.5860 / 15.4034`. The selected complex did not collapse: face/tetra
boundary lDDT stayed at `0.7355` / `0.7193`, and contraction fractions were
`0.5656` / `0.5663`. The interpretation is that partial directed readout is
useful as an annealed curriculum into E96, but holding it for another gate
overdrives local C-alpha agreement.

E97 returned a useful but not decisive stabilization result. It resumed E96,
ramped an outer-edge-supported selected-cell score from `0.0` to `0.25`, and
ramped the partial directed boundary readout from `0.25` to `0.0`. It reached
`val_lddt_ca=0.4036`, just below E96's `0.4043`, while improving FoldScore to
`0.3867`, dRMSD to `9.7492`, and predicted/true C-alpha radius to
`11.7951 / 15.4034`. Selected face/tetra boundary lDDT rose to
`0.7488` / `0.7318`, and outer-edge active fractions were `1.0` / `1.0`.
This supports the topology-construction handoff as a stabilizer, but not yet
as a new primary-lDDT leader.

The current state does not yet justify a 30,000-step confirmation spend as if
the `0.7` target were likely. The strongest branch has moved from roughly
`0.348` at step 1000 to `0.404` by step 9000-9500, and recent topology-native
changes mostly trade within a narrow `0.39-0.404` band. Before spending on a
long run, the next work should either find a branch whose short gates show a
clearer slope in primary C-alpha lDDT or run a deliberately diagnostic longer
continuation with the expectation that it may falsify the current branch.

E99 rejected the simple continuation hypothesis. It continued E97 past the
10,000-step line with final E97 topology settings fixed and reached
`val_lddt_ca=0.3972` at step 10000 and `0.4003` at step 10500, below both E96
and E97. FoldScore stayed near the local best at `0.3857`, but dRMSD worsened
to `10.1507` and predicted/true C-alpha radius returned to
`11.3807 / 15.4034`. The selected complex itself kept improving:
face/tetra boundary lDDT reached `0.7574` / `0.7386`, and contraction
fractions fell to `0.5290` / `0.5271`. This falsifies the idea that the
current E96/E97 lineage only needed to cross 10,000 steps before taking off.
The next experiment should change how selected higher-rank states affect the
residue/pair trunk, rather than continuing the same lineage longer.

The current 2026-05-12 full reread of the saved PDFs reinforces the E79-E81
direction and the E96 interpretation. The right lesson is not to add another
generic coordinate objective; it is to treat directed incidence as a
cochain-routing curriculum and to measure whether that route helps the selected
complex write useful edge information back into `Z_ij` without overdriving the
pair tensor.

Because E98 regressed, the next launchable fallback is outer-edge-supported
cell scoring rather than latent segment cells. The remote parameter audit
shows that an outer-edge scorer on the E87/E96-style sparse/edge-frame setup
stays at `3,154,242` parameters, while latent segment cells plus edge-frame
modules exceed the `3,261,974` cap even with `simplex_c_segment=4`. Segment
cells remain paper-aligned, but only as a separate no-edge-frame branch from a
sparse checkpoint; they are not the immediate continuation of E96/E98.

The 2026-05-12 full reread of the saved PDFs reinforces the E79-E81 direction.
The TDL guide frames construction of the topological domain, intra-rank
aggregation, inter-rank aggregation, and topology-aware diagnostics as core
model choices. Topotein makes the same point in protein terms: directed
incidence, outer-edge neighborhoods, edge-centric scalarization, and
comprehensive rank-wise updates matter more than superficial higher-rank
features. For SimplexFold, that means the next branch should keep changing
which sparse cells exist or how selected cochains communicate through their
incidence/outer-edge structure.

The other prepared alternatives are now directed outer-edge transport and
directed boundary readout. E81 showed that changing the selected-cell score is
a valid topology-construction lever. E85 showed that plain incidence
normalization is not enough. The existing `simplex_outer_edge_context_scale`
path already supplies directed incoming and outgoing outer-edge summaries, so
the paper-aligned E86 version is to combine that path with sparse cells plus
incidence normalization, not to add a second duplicate outer-edge module.
Future runs now report selected face/tetra outer-edge availability alongside
boundary-edge reuse, so E86-style runs can be interpreted in terms of the
actual selected cochain neighborhoods. A second prepared fallback is directed
boundary readout:
`simplex_boundary_readout_directionality` keeps the default symmetric
simplex-to-pair scatter at `0`, but can blend toward source/target directed
boundary-edge writes. The prepared test should ramp the runtime contribution
from `0.0` to `0.5` rather than switching it on abruptly. This is another
zero-parameter cochain-communication test aligned with the directed-incidence
view rather than a new metric loss.

A third prepared fallback is runtime-gated latent segment cells. This borrows
Topotein's secondary-structure-cell idea without using DSSP/SSE labels:
segment cochains are derived only from official sequence/MSA/pair features and
recycled geometry, then passed into selected face states. The new
`simplex_segment_cell_runtime_scale` gate lets a resumed sparse-complex model
allocate that local rank-2 route but ramp its contribution from zero, avoiding
the abrupt static sidecar tested much earlier.

A fourth prepared fallback is a pair-preserving simplex readout gate. It keeps
selected face/tetra cochain evidence flowing into the pair tensor `Z_ij` while
allowing the direct single/residue readout to be damped separately. This is a
zero-parameter routing test for the central README claim that higher-order
states should improve pair/edge reasoning before the structure module.

A fifth prepared fallback is an outer-edge-supported cell scorer. It keeps the
degree-penalized sparse complex, but adds an optional zero-parameter bonus for
candidate faces/tetras whose vertices have selected neighbor edges leaving the
cell. The score can be ramped in at runtime on a resumed checkpoint, treating
outer-edge availability as part of the topological domain construction itself
rather than merely as a diagnostic after training.

Yes. With templates forbidden, the right construction is:

[
\text{MSA} ;\longleftrightarrow; \text{pair/edge tensor } Z_{ij}
;\longleftrightarrow; \text{sparse face tensor } F_{ijk}
;\longleftrightarrow; \text{sparse tetra tensor } U_{ijkl}
;\longrightarrow; \text{structure module}
;\longrightarrow; \text{recycling geometry}.
]

The important trick is that **the 2-/3-simplex topology should not come from templates**. It should come from either:

1. **latent contact/topology predictions from the MSA/pair stack**, especially in the first pass; or
2. **the model’s own recycled coordinates**, after one structure-module pass.

I verified the current public NanoFold rules: the official track allows architecture/loss/curriculum changes and train-from-scratch biological priors, but disallows external sequences, structures, pretrained weights, external MSA/template retrieval, and network downloads; template tensors exist in the schema, but official preprocessing uses (T=0). ([GitHub][1]) So the design below stays in-bounds as long as the higher-order cells are built only from official inputs and model-generated intermediate predictions.

---

## 1. Baseline tensors

Start from an AlphaFold-like trunk.

Let:

[
M \in \mathbb{R}^{B \times N_{\text{msa}} \times L \times C_m}
]

be the MSA representation,

[
Z \in \mathbb{R}^{B \times L \times L \times C_z}
]

be the pair representation, and

[
S \in \mathbb{R}^{B \times L \times C_s}
]

be the single/residue representation, usually the first MSA row or a learned single stream.

In NanoFold, the public feature schema includes `aatype`, `msa`, `deletions`, and template placeholders; labels include Cα and atom14 targets for training, while hidden inference is features-only. ([GitHub][1]) The official budget also uses crop size (L=256), MSA depth (192), effective batch (2), and a 50M parameter cap on the limited/research tracks, so a design should be sparse and not rely on all triples or all quadruples. ([GitHub][1])

AlphaFold2’s Evoformer already treats structure prediction as graph inference over residue-pair edges, with MSA columns as residue positions and pair entries as residue relations. It also has triangle multiplicative/self-attention updates and recycling, where previous predictions are fed back for iterative refinement. ([Nature][2]) The proposal here is to add **explicit sparse states** for selected faces and tetrahedra instead of keeping all higher-order reasoning implicit inside (Z).

---

## 2. Do not instantiate all triples/quads

For (L=256):

[
\binom{256}{3} \approx 2.76\text{M}
]

triangles, and

[
\binom{256}{4} \approx 174.8\text{M}
]

tetrahedra. Dense (F_{ijk}) and (U_{ijkl}) are not viable.

Instead, build a sparse local complex from a top-(K) neighbor graph.

For each residue (i), choose (K) candidate neighbors:

[
\mathcal{N}(i) = \operatorname{TopK}*{j \ne i}; \text{score}*{ij}.
]

Then construct anchored faces and tetrahedra:

[
(i,j,k), \qquad j,k \in \mathcal{N}(i),
]

[
(i,j,k,\ell), \qquad j,k,\ell \in \mathcal{N}(i).
]

With (K=12):

[
L \binom{K}{2} = 256 \cdot 66 = 16{,}896
]

anchored faces per crop, and

[
L \binom{K}{3} = 256 \cdot 220 = 56{,}320
]

anchored tetrahedra per crop. That is totally feasible.

With (K=16), you get about (30{,}720) faces and (143{,}360) tetrahedra per crop, still plausible if (C_f) and (C_u) are small.

---

## 3. Topology construction without templates

Use two regimes.

### Pass 0: latent topology from sequence/MSA/pair features

Before the model has coordinates, define a neighbor score from the pair tensor:

[
\text{score}_{ij}
=================

w^\top Z_{ij}
+
b_{\text{local}}(|i-j|)
+
b_{\text{sep}}(i,j).
]

Here (w^\top Z_{ij}) is a learned contact/topology logit. The local bias forces inclusion of backbone-near residues such as (i\pm1, i\pm2, i\pm4). This prevents the early model from building nonsense higher-order cells before it understands long-range contacts.

A practical scoring rule:

[
\text{score}_{ij}
=================

\ell^{\text{contact}}*{ij}
+
\lambda*{\text{local}}\mathbf{1}[|i-j|\le r]
+
\lambda_{\text{long}}\mathbf{1}[|i-j|>s].
]

Then select top-(K) neighbors per residue.

### Pass (r \ge 1): topology from recycled coordinates

After the structure module predicts atom14/Cα coordinates, recycle the predicted geometry.

Let:

[
X_i \in \mathbb{R}^3
]

be the predicted Cα coordinate for residue (i). Define:

[
d_{ij} = |X_i - X_j|.
]

Then use a hybrid score:

[
\text{score}_{ij}
=================

## \ell^{\text{contact}}_{ij}

\alpha d_{ij}
+
\lambda_{\text{local}}\mathbf{1}[|i-j|\le r].
]

This says: keep residues that the model believes are in contact, but also prefer residues that are close in the currently predicted structure.

Topology selection should usually be **stop-gradient**:

```python
with torch.no_grad():
    nbr_idx = build_neighbors(pair_logits, recycled_ca_coords)
```

The message passing through the selected tensors is differentiable; the discrete top-(K) choice does not need to be.

This mirrors the spirit of recycling: the model’s own previous structure estimate becomes an internal geometric prior. AF2’s paper explicitly describes recycling as feeding outputs recursively into the same modules for iterative refinement. ([Nature][2])

---

## 4. Add explicit 2-simplex and 3-simplex states

Define sparse face states:

[
F_{i,p} \in \mathbb{R}^{C_f},
]

where (p) indexes a pair of neighbors ((j,k)) of anchor (i). Equivalently:

[
F_{ijk} \in \mathbb{R}^{C_f}.
]

Define sparse tetra states:

[
U_{i,q} \in \mathbb{R}^{C_u},
]

where (q) indexes a triple of neighbors ((j,k,\ell)). Equivalently:

[
U_{ijkl} \in \mathbb{R}^{C_u}.
]

Use small channels:

[
C_f \in {16,32}, \qquad C_u \in {8,16}.
]

The point is not to build a second giant Evoformer. It is to add a lightweight geometric higher-order correction to the pair/single streams.

---

## 5. Initialize faces from edge states

For a face ((i,j,k)), gather the three edge states:

[
Z_{ij}, \quad Z_{ik}, \quad Z_{jk}.
]

Then initialize:

[
F_{ijk}
=======

\phi_F
\left(
Z_{ij},
Z_{ik},
Z_{jk},
S_i,
S_j,
S_k,
g^{(2)}_{ijk}
\right).
]

Here (g^{(2)}_{ijk}) is optional geometric input. Before recycling, it may contain only sequence features:

[
|i-j|,\ |i-k|,\ |j-k|.
]

After recycling, add actual geometric scalars:

[
d_{ij}, d_{ik}, d_{jk},
]

triangle area,

[
A_{ijk}
=======

\frac{1}{2}
\left|
(X_j-X_i)\times(X_k-X_i)
\right|,
]

internal angle cosines,

[
\cos\theta_i
============

\frac{(X_j-X_i)\cdot(X_k-X_i)}
{|X_j-X_i||X_k-X_i|},
]

and perhaps a normal vector expressed in residue (i)’s local frame:

[
n^{(i)}_{ijk}
=============

R_i^\top
\frac{(X_j-X_i)\times(X_k-X_i)}
{|(X_j-X_i)\times(X_k-X_i)|+\varepsilon}.
]

Using local frames makes the feature invariant to global rotation/translation.

---

## 6. Initialize tetrahedra from edge/face states

For a tetrahedron ((i,j,k,\ell)), gather the six edge states:

[
Z_{ij}, Z_{ik}, Z_{i\ell}, Z_{jk}, Z_{j\ell}, Z_{k\ell}.
]

Optionally gather its four face states:

[
F_{ijk}, F_{ij\ell}, F_{ik\ell}, F_{jk\ell}.
]

Then initialize:

[
U_{ijkl}
========

\phi_U
\left(
{Z_{ab}}*{a,b\in{i,j,k,\ell}},
{F*{abc}}*{a,b,c\in{i,j,k,\ell}},
g^{(3)}*{ijkl}
\right).
]

After recycling, useful tetrahedral geometric features include:

six edge distances,

[
d_{ij}, d_{ik}, d_{i\ell}, d_{jk}, d_{j\ell}, d_{k\ell},
]

four face areas,

[
A_{ijk}, A_{ij\ell}, A_{ik\ell}, A_{jk\ell},
]

and signed volume,

[
V_{ijkl}
========

\frac{1}{6}
\det
\left[
X_j-X_i,,
X_k-X_i,,
X_\ell-X_i
\right].
]

The absolute volume captures compactness/packing; the sign captures handedness when expressed relative to a consistent local frame.

---

## 7. Message passing via boundary and coboundary maps

The clean mathematical version is simplicial message passing.

A 2-simplex has boundary edges:

[
\partial(i,j,k) = {(i,j),(i,k),(j,k)}.
]

A 3-simplex has boundary faces:

[
\partial(i,j,k,\ell)
====================

{(i,j,k),(i,j,\ell),(i,k,\ell),(j,k,\ell)}.
]

So perform updates:

### Edge (\rightarrow) face

[
F_{ijk}
\leftarrow
F_{ijk}
+
\phi_{2\leftarrow1}
\left(
F_{ijk},
Z_{ij},
Z_{ik},
Z_{jk}
\right).
]

### Face (\rightarrow) edge

[
Z_{ij}
\leftarrow
Z_{ij}
+
\sum_{k:,(i,j,k)\in K_2}
\phi_{1\leftarrow2}
\left(
Z_{ij},
F_{ijk}
\right).
]

### Face (\rightarrow) tetra

[
U_{ijkl}
\leftarrow
U_{ijkl}
+
\phi_{3\leftarrow2}
\left(
U_{ijkl},
F_{ijk},
F_{ij\ell},
F_{ik\ell},
F_{jk\ell}
\right).
]

### Tetra (\rightarrow) face

[
F_{ijk}
\leftarrow
F_{ijk}
+
\sum_{\ell:,(i,j,k,\ell)\in K_3}
\phi_{2\leftarrow3}
\left(
F_{ijk},
U_{ijkl}
\right).
]

### Simplices (\rightarrow) residue/single stream

[
S_i
\leftarrow
S_i
+
\sum_{j} \phi_{0\leftarrow1}(Z_{ij})
+
\sum_{j,k} \phi_{0\leftarrow2}(F_{ijk})
+
\sum_{j,k,\ell} \phi_{0\leftarrow3}(U_{ijkl}).
]

Then the updated (Z) flows back to the MSA through pair-biased MSA attention, exactly as in the AF2-style MSA/pair loop. AF2’s IPA/structure module also uses single, pair, and geometric representations, with pair representation controlling structure generation through attention biases/values. ([Nature][2])

---

## 8. Efficient tensor layout

Use an **anchored neighbor tensor** rather than a generic sparse COO graph for the first implementation.

Let:

```python
nbr_idx: [B, L, K]
```

where `nbr_idx[b, i, :]` are the selected neighbors of residue `i`.

Precompute local combinations:

```python
face_combos = combinations(range(K), 2)  # [M2, 2], M2 = K*(K-1)//2
tet_combos  = combinations(range(K), 3)  # [M3, 3], M3 = K*(K-1)*(K-2)//6
```

Then:

```python
face_j = nbr_idx[:, :, face_combos[:, 0]]  # [B, L, M2]
face_k = nbr_idx[:, :, face_combos[:, 1]]  # [B, L, M2]

tet_j = nbr_idx[:, :, tet_combos[:, 0]]    # [B, L, M3]
tet_k = nbr_idx[:, :, tet_combos[:, 1]]    # [B, L, M3]
tet_l = nbr_idx[:, :, tet_combos[:, 2]]    # [B, L, M3]
```

Face tensor:

```python
F: [B, L, M2, C_f]
```

Tetra tensor:

```python
U: [B, L, M3, C_u]
```

This layout is GPU-friendly because the shapes are dense and static for fixed (K).

---

## 9. PyTorch-style gather for faces

Suppose:

```python
z: [B, L, L, Cz]
s: [B, L, Cs]
nbr_idx: [B, L, K]
```

Then:

```python
B, L, _, Cz = z.shape
device = z.device

i = torch.arange(L, device=device)[None, :, None]      # [1, L, 1]
i_face = i.expand(B, L, M2)                            # [B, L, M2]

j = face_j                                             # [B, L, M2]
k = face_k                                             # [B, L, M2]

b = torch.arange(B, device=device)[:, None, None].expand(B, L, M2)

z_ij = z[b, i_face, j]                                 # [B, L, M2, Cz]
z_ik = z[b, i_face, k]                                 # [B, L, M2, Cz]
z_jk = z[b, j, k]                                      # [B, L, M2, Cz]

s_i = s[b, i_face]                                     # [B, L, M2, Cs]
s_j = s[b, j]                                          # [B, L, M2, Cs]
s_k = s[b, k]                                          # [B, L, M2, Cs]

face_in = torch.cat([z_ij, z_ik, z_jk, s_i, s_j, s_k, face_geom], dim=-1)
F = face_mlp(face_in)                                  # [B, L, M2, Cf]
```

This is the core implementation. No dense (L^3) tensor.

---

## 10. PyTorch-style scatter back to pair representation

A face ((i,j,k)) should update edges ((i,j)), ((i,k)), and ((j,k)).

Let:

```python
edge_update = face_to_edge(F)  # [B, L, M2, 3, Cz]
```

where slot 0 updates ((i,j)), slot 1 updates ((i,k)), slot 2 updates ((j,k)).

Flatten pair indices:

[
\operatorname{edge_id}(a,b) = aL + b.
]

```python
edge_ij = i_face * L + j
edge_ik = i_face * L + k
edge_jk = j * L + k

edge_ids = torch.stack([edge_ij, edge_ik, edge_jk], dim=-1)  # [B, L, M2, 3]
```

Then scatter:

```python
z_flat = z.reshape(B, L * L, Cz)

upd = edge_update.reshape(B, L * M2 * 3, Cz)
ids = edge_ids.reshape(B, L * M2 * 3)

z_delta = torch.zeros_like(z_flat)
z_delta.scatter_add_(
    dim=1,
    index=ids[..., None].expand(-1, -1, Cz),
    src=upd,
)

counts = torch.zeros(B, L * L, 1, device=z.device, dtype=z.dtype)
counts.scatter_add_(
    dim=1,
    index=ids[..., None],
    src=torch.ones(B, L * M2 * 3, 1, device=z.device, dtype=z.dtype),
)

z = z_flat + z_delta / counts.clamp_min(1.0)
z = z.reshape(B, L, L, Cz)
```

For directed pair representations, also update reverse edges ((j,i)), ((k,i)), ((k,j)), either with a separate projection or by sharing the same update.

---

## 11. Tetra tensor construction

For tetrahedra anchored at (i):

```python
j = tet_j  # [B, L, M3]
k = tet_k
l = tet_l
```

Gather six edges:

```python
z_ij = z[b, i_tet, j]
z_ik = z[b, i_tet, k]
z_il = z[b, i_tet, l]
z_jk = z[b, j, k]
z_jl = z[b, j, l]
z_kl = z[b, k, l]
```

Then:

```python
tet_in = torch.cat(
    [z_ij, z_ik, z_il, z_jk, z_jl, z_kl, tet_geom],
    dim=-1,
)

U = tet_mlp(tet_in)  # [B, L, M3, Cu]
```

For a first version, I would not maintain canonical face IDs for every ((j,k,\ell)). I would let tetrahedra update the six pair edges directly and optionally update the three anchored faces ((i,j,k)), ((i,j,\ell)), ((i,k,\ell)). That gives most of the value without implementing full topological closure/deduplication.

A mathematically cleaner version deduplicates faces/tetrahedra into COO lists:

```python
edge_index: [B, E, 2]
face_index: [B, F, 3]
tet_index:  [B, U, 4]

edge_of_face: [B, F, 3]
face_of_tet:  [B, U, 4]
edge_of_tet:  [B, U, 6]
```

Then all message passing is just `gather -> MLP -> scatter_add`. That is cleaner, but more engineering.

---

## 12. MSA to simplex communication

There are three levels of ambition.

### Level 1: MSA communicates through pair only

This is simplest and probably the first implementation.

[
M \rightarrow Z
]

through outer product mean, then:

[
Z \rightarrow F \rightarrow U \rightarrow Z,
]

then:

[
Z \rightarrow M
]

through pair-biased MSA attention.

This already lets MSA information influence higher-order cells.

### Level 2: low-rank MSA-to-face update

Define a third-order MSA moment only over selected faces:

[
\Delta F_{ijk}
==============

W_F
\left[
\frac{1}{N_{\text{msa}}}
\sum_a
(A M_{a i})
\odot
(B M_{a j})
\odot
(C M_{a k})
\right].
]

Use Hadamard products, not full tensor outer products.

Implementation sketch:

```python
rank = 16

m_proj_i = A(msa)  # [B, Nmsa, L, rank]
m_proj_j = B(msa)
m_proj_k = C(msa)

mi = gather_msa_columns(m_proj_i, i_face)  # [B, Nmsa, L, M2, rank]
mj = gather_msa_columns(m_proj_j, j)
mk = gather_msa_columns(m_proj_k, k)

triple_moment = (mi * mj * mk).mean(dim=1) # [B, L, M2, rank]
F = F + msa_to_face(triple_moment)
```

This is expensive if done naively, so chunk over faces/residues and keep `rank` small. With (K=8) or (K=12), it is feasible. I would run this only every few blocks.

### Level 3: MSA-to-tetra fourth-order moment

In principle:

[
\Delta U_{ijkl}
===============

W_U
\left[
\frac{1}{N_{\text{msa}}}
\sum_a
(A M_{a i})
\odot
(B M_{a j})
\odot
(C M_{a k})
\odot
(D M_{a \ell})
\right].
]

I would not start here. It is costly and probably noisy. Let tetrahedra be built from pair/face/geometric states.

---

## 13. A concrete Evoformer block variant

Call it a **Simplicial Evoformer block**.

```text
Input:
    M: MSA tensor                 [B, Nmsa, L, Cm]
    Z: pair / edge tensor          [B, L, L, Cz]
    S: single tensor               [B, L, Cs]
    X_prev: recycled Cα coords      [B, L, 3] or None
    R_prev: recycled frames         [B, L, 3, 3] or None

1. MSA row attention with pair bias:
    M <- RowAttention(M, bias=Linear(Z))

2. MSA column attention:
    M <- ColumnAttention(M)

3. MSA transition:
    M <- M + MLP(LN(M))

4. MSA -> pair:
    Z <- Z + OuterProductMean(M)

5. Pair update:
    Z <- Z + TriangleMul/PairMixer/PairTransition(Z)

6. Build sparse neighbor graph:
    nbr_idx <- TopK(score(Z, X_prev))

7. Build 2-simplex states:
    F <- FaceInit(Z, S, nbr_idx, X_prev, R_prev)

8. Build 3-simplex states:
    U <- TetraInit(Z, F, S, nbr_idx, X_prev, R_prev)

9. Simplicial message passing:
    F <- F + EdgeToFace(Z, F)
    U <- U + FaceToTetra(F, U)
    F <- F + TetraToFace(U, F)
    Z <- Z + FaceToEdge(F)
    Z <- Z + TetraToEdge(U)
    S <- S + SimplexToSingle(F, U)

10. Pair/single transitions:
    Z <- Z + MLP(LN(Z))
    S <- S + MLP(LN(S))

Output:
    M, Z, S
```

Then after several blocks:

```text
StructureModule(S, Z) -> atom14 coordinates, residue frames
Recycle coordinates/frames into next trunk pass
```

---

## 14. Gating is essential

Every simplex update should be gated. For example:

[
\Delta Z_{ij}
=============

\sigma(W_g Z_{ij})
\odot
W_o
\operatorname{Pool}*{k}
F*{ijk}.
]

Similarly:

[
F_{ijk}
\leftarrow
F_{ijk}
+
\sigma(W_g F_{ijk})
\odot
\phi(F_{ijk}, Z_{ij}, Z_{ik}, Z_{jk}).
]

This keeps early training stable. Without gates, the sparse simplex module may inject high-variance updates into (Z) and destabilize the structure module.

Use residual + layer norm everywhere:

[
H \leftarrow H + \operatorname{Dropout}(\Delta H),
]

[
H \leftarrow H + \operatorname{MLP}(\operatorname{LN}(H)).
]

---

## 15. Geometry features after recycling

Once recycled coordinates exist, add scalar geometric features.

For edges:

[
g_{ij}^{(1)} =
\left[
\operatorname{RBF}(d_{ij}),
\operatorname{RBF}(|i-j|),
R_i^\top(X_j-X_i)
\right].
]

For faces:

[
g_{ijk}^{(2)}
=============

\left[
\operatorname{RBF}(d_{ij}),
\operatorname{RBF}(d_{ik}),
\operatorname{RBF}(d_{jk}),
A_{ijk},
\cos\theta_i,
\cos\theta_j,
\cos\theta_k,
R_i^\top n_{ijk}
\right].
]

For tetrahedra:

[
g_{ijkl}^{(3)}
==============

\left[
\operatorname{RBF}(\text{six distances}),
\operatorname{RBF}(\text{four face areas}),
V_{ijkl},
|V_{ijkl}|,
R_g
\right],
]

where (R_g) is the radius of gyration of the four Cα points:

[
R_g^2
=====

\frac{1}{4}
\sum_{a\in{i,j,k,\ell}}
\left|
X_a - \bar X
\right|^2.
]

This makes tetrahedra useful as local packing descriptors.

---

## 16. Losses that make the simplex states learn something useful

The simplest risk is that (F) and (U) become decorative: the network routes around them. Add auxiliary losses.

### Pair distogram loss

Standard pairwise distance-bin supervision:

[
\mathcal{L}_{\text{dist}}
=========================

\operatorname{CE}
\left(
\hat p_{ij}^{\text{dist}},
\operatorname{bin}(d_{ij}^{\text{true}})
\right).
]

### Face geometry loss

For selected faces, predict triangle area or binned area:

[
\hat A_{ijk} = h_F(F_{ijk}),
]

[
\mathcal{L}_{\text{area}}
=========================

\left|
\hat A_{ijk} - A_{ijk}^{\text{true}}
\right|.
]

Or classify binned face shape using the three true distances.

### Tetra geometry loss

For selected tetrahedra, predict binned volume or compactness:

[
\hat V_{ijkl} = h_U(U_{ijkl}),
]

[
\mathcal{L}_{\text{vol}}
========================

\left|
\hat V_{ijkl} - V_{ijkl}^{\text{true}}
\right|.
]

Also useful:

[
\mathcal{L}_{\text{tet-dist}}
=============================

\sum_{a<b}
\operatorname{CE}
\left(
\hat p_{ab}^{\text{dist from }U},
\operatorname{bin}(d_{ab}^{\text{true}})
\right).
]

Use these only as training losses, not as input features. Training labels are available for supervised loss, but hidden inference is features-only under the competition protocol. ([GitHub][1])

---


## 17. Practical hyperparameters

For NanoFold-scale constraints, I would start with:

```yaml
c_m: 64
c_z: 64
c_s: 64
c_face: 32
c_tetra: 16

neighbor_k: 12
simplex_every_n_blocks: 2
num_simplicial_layers_per_call: 1
num_recycles: 2 or 3

use_faces: true
use_tetra: initially false, then true
use_msa_to_face: initially false
detach_recycled_geometry: true
```

For (L=256, K=12):

```text
faces per crop:  16,896
tetras per crop: 56,320
```

Memory is very manageable:

```text
F: 16,896 × 32 ≈ 540k floats
U: 56,320 × 16 ≈ 900k floats
```

per example before activations. The real cost is MLP/gather/scatter overhead, not raw storage.

---

## 18. Main architectural caveat

AF2’s triangle updates are already very strong. An explicit 2-simplex state can easily become redundant unless it adds something AF2 does not already get cheaply.

The added value should come from one or more of:

1. **persistent face/tetra states**, not just transient triangle updates;
2. **geometry from recycled coordinates**, especially areas/volumes/chirality;
3. **low-rank MSA-to-face moments**, capturing third-order evolutionary correlations;
4. **auxiliary face/tetra losses**, forcing the representations to encode real shape;
5. **local packing priors**, especially through tetrahedral volume and compactness.

The highest-probability useful contribution is:

[
\boxed{
\text{sparse recycled 2-simplex face adapter}
}
]

not full tetrahedral message passing on day one.

The full concept is:

```text
official MSA/sequence features
        ↓
MSA representation M
        ↓ outer product mean
pair / 1-simplex tensor Z
        ↓ top-K latent/geometric neighbor graph
2-simplex tensor F
        ↓ optional local tetra construction
3-simplex tensor U
        ↓ scatter/gated residuals
updated Z and S
        ↓
structure module
        ↓
recycled coordinates
        ↺
```

That is the clean modern tensor implementation of your mentor’s idea without using templates.

[1]: https://github.com/ChrisHayduk/nanoFold-Competition/blob/main/docs/COMPETITION.md "nanoFold-Competition/docs/COMPETITION.md at main · ChrisHayduk/nanoFold-Competition · GitHub"
[2]: https://www.nature.com/articles/s41586-021-03819-2 "Highly accurate protein structure prediction with AlphaFold | Nature"

## 2026-05-09 NanoFold AF2-Medium-Matched Iteration Plan

Target: improve `simplexfold_medium_param_matched` on NanoFold public
validation toward `val_lddt_ca > 0.7` while keeping total parameters within
5% of the AF2-medium pair-only baseline.

Budget contract:

- AF2-medium baseline is `configs/medium.toml` with
  `use_simplicial_evoformer = false`: 3,106,642 parameters.
- `configs/simplexfold_medium_param_matched.toml` is 3,106,690 parameters.
- Allowed 5% upper bound is 3,261,974 parameters.
- Prefer zero-parameter or near-zero-parameter changes first: topology
  selection, simplex auxiliary objectives, cell construction, recycling
  policy, and curriculum.
- After the zero-parameter selector/curriculum pass, spend any remaining
  budget only inside the explicit face/tetra/MSA-to-face pathway. Do not use
  the slack to widen the generic AF2 MSA/pair/structure trunk.

Scientific constraint: every iteration must be justified through the
simplicial/topological view. A change that would help a pair-only AF2 trunk
equally well is not a SimplexFold result unless it is specifically mediated by
the sparse face/tetra complex.

Iteration ladder:

1. Establish short-run controls for `no_simplex`, `faces`, and `full` using
   the same AF2-medium-matched profile, same data subset, same seed, and
   same recycle/crop/MSA settings.
2. Improve the learned topology scorer first. The first-pass complex depends
   on pair/contact logits, so contact supervision should be balanced enough
   for true contacts to influence top-k neighbor selection instead of being
   swamped by non-contacts.
3. Keep simplex boundary supervision focused on selected cells: face/tetra
   edge distances, face area, tetra volume/chirality, and boundary
   consistency. Do not add a generic all-pairs C-alpha distance loss.
4. If topology quality improves but structure does not, strengthen the
   simplex-to-pair/single residual path or make `K`/long-contact bias a
   controlled ablation.
5. If selected face/tetra coordinate realization improves collapse but remains
   too weak, tune those selected-cell realization weights directly rather than
   adding a generic all-pairs C-alpha loss.
6. If area/volume realization still under-expands structures, add coordinate
   losses on the boundary edges of the selected face/tetra cells. This stays
   inside the simplicial chain view: the model realizes the metric 1-skeleton
   induced by its own sparse 2-/3-simplex complex, not a dense all-pairs
   distance matrix.
7. If direct long-range bias destabilizes the complex, reserve a small
   sequence-local scaffold of neighbor slots and let the remaining slots be
   learned/global. This keeps local manifold continuity while giving selected
   faces and tetras room to include nonlocal packing edges.
8. If the E15 curriculum reaches a plateau, test a simplex-only capacity
   profile that increases persistent face/tetra channels and the MSA-to-face
   low-rank adapter while preserving the medium AF2 trunk and staying within
   the 5% parameter contract. This asks whether the higher-order state itself
   is currently under-capacitated.
9. If extra capacity helps mid-run geometry but not final lDDT, add a
   tolerance-style metric realization loss only on the selected face/tetra
   boundary edges. This is not a dense all-pairs lDDT objective; it asks the
   learned sparse cell complex to realize its own 1-skeleton accurately under
   the same local-distance tolerances used by C-alpha lDDT.
10. If simplex capacity, auxiliary schedules, and coupling warmups remain
    below the E09/E15 band, change the readout path rather than adding more
    scalar loss pressure. Pool persistent face/tetra states back to their
    boundary residues/edges and gate that summary into the structure input, so
    coordinates are generated from realized 2-/3-cells instead of only from
    residual perturbations to the AF2 trunk. Prefer a zero-parameter readout
    first by reusing the adapter's existing selected-boundary summaries before
    spending any remaining parameter headroom.
11. If structure readouts fail, treat that as evidence that the selected
    simplex states are not yet reliable enough to condition coordinates
    directly. Return to the topology construction path: improve the sparse
    2-skeleton before reusing it downstream, using selected-cell objectives,
    topology curricula, and selector regularization rather than dense
    structure losses that would be independent of the simplicial complex.
12. If selector regularization is revisited, prefer gentle or staged
    curricula over always-on hard contact margins. The sparse complex should
    preserve enough geometric diversity for face/tetra realization; forcing
    the 1-skeleton too aggressively toward binary contacts can damage local
    structure quality even when global radius improves.
13. Revisit the geometric content of selected cells before widening the
    generic trunk. The current selected-face realization covers area and
    boundary distances, but a 2-simplex in the README is also an oriented
    patch. Orientation losses should be expressed in residue-local backbone
    frames so they remain rigid-motion invariant and stay attached to the
    selected sparse face complex.
14. Promote only changes that pass parameter-budget tests and at least a
    NanoFold smoke/short-run comparison.

Current direction after E43: the zero-parameter Hodge residual and auxiliary
anneal did not recover the stronger early-validation band. Treat this as
evidence that downstream face-state smoothing is less important than the
construction and realization of the selected sparse complex. The next branch
should change the selected 2-/3-cell construction or boundary realization
itself, using topological operators or incidence-aware curricula from the
reference PDFs, before spending parameter budget on additional structure
readouts.

Immediate E44 branch: make selected faces and tetras a soft flag complex by
weighting each cell by the plausibility of its full boundary 1-skeleton. This
keeps the intervention squarely in the simplicial/topological view: a filled
2- or 3-simplex should be trusted most when the learned boundary edges are
also trusted, instead of treating every pair or triplet of anchor neighbors
as an equally valid higher-order cell.

E44 result update: fixed-strength flag closure is too suppressive by the end
of the 500-step gate. If closure is revisited, ramp it in after the topology
scorer has learned useful edges, or apply it only to the auxiliary realization
losses rather than to the message-passing masks from step 1.

Immediate E45 branch: run the same flag-complex idea at a much lighter mask
blend (`0.1` instead of `0.5`). This tests whether the failure was the
topological closure prior itself or simply too much early suppression of
selected 2-/3-cells.

E45 result update: lighter fixed closure is still not competitive. The next
architecture branch should leave selected cell masks intact and instead make
the face/tetra states more useful to the structure module, or revisit closure
only as a scheduled auxiliary weighting rather than a message mask.

Immediate E46 branch: increase selected complex coverage without changing
parameters by raising `simplex_neighbor_k` from 12 to 14. This expands the
available faces and tetra cofaces while preserving the same AF2 trunk and
learned simplex channels, testing whether recent failures were caused by
under-covering the sparse packing complex rather than by weak state updates.

E46 result update: fixed K-expansion is not the missing piece. The step-250
checkpoint reached only `val_lddt_ca=0.2517`, and the step-500 checkpoint
fell to `0.2327` with renewed radius collapse. The next branch should stop
changing static cell count alone and instead make the selected complex
adaptive over training: either schedule flag/closure weights only on
auxiliary realization losses, anneal the neighbor complex from local to
learned/global cells, or add an incidence-aware objective that improves cell
quality without suppressing message-passing masks from the first step.

Immediate E47 branch: revisit flag-complex closure as an auxiliary
realization curriculum rather than a message mask. Keep the learned
face/tetra states and their pair/single boundary messages on the ordinary
E09/E15 `full_msa_to_face` path, but ramp a true-boundary closure weight onto
only the selected-cell coordinate realization losses. This tests whether
filled 2-/3-cells should be trusted most when their boundary 1-skeleton is a
plausible local complex, without starving early message passing through
potentially useful open cells.

E47 result update: auxiliary-only flag closure is also not competitive. It
peaked at `val_lddt_ca=0.2466` before the closure ramp and ended at `0.2262`
after the ramp, with predicted C-alpha radius falling to `5.5581 / 15.4034`.
Closure appears harmful whether used as a message mask or as realization-loss
weighting. The next branch should avoid closure heuristics and instead test
an adaptive local-to-global topology curriculum or a longer confirmation run
only for the still-best E15/E12 family.

## 2026-05-10 Reference PDF Update

Local reference copies were saved in `references/papers/`:

- `hands_on_geometric_deep_learning_nodes_to_complexes.pdf`
- `2509.03885v1.pdf`

The PDFs are intentionally ignored by git until redistribution rights are
confirmed. Tracked reading notes live in
`references/papers/READING_NOTES.md`.

The main planning update from reading both references is that closure should
not be the next SimplexFold family. The Topotein paper explicitly motivates
combinatorial complexes because protein hierarchy can benefit from flexible
set-type cells without strict boundary constraints. That matches E44-E47:
flag-complex closure, whether used as a message mask or only as an auxiliary
realization weighting, weakens the validation curve.

Immediate E48 branch: implement an adaptive local-to-global topology
curriculum. Keep the E09/E15 `full_msa_to_face` architecture and selected
coordinate/boundary losses, but schedule selected neighbor construction from
a small sequence-local scaffold toward the ordinary learned/global selector.
This follows the TDL view from the references: the neighborhood operator is
part of the model, and changing it over training is a topological curriculum
rather than a generic lDDT-targeted loss.

E48 implementation update: the curriculum is implemented as a training-only
`simplex_local_neighbor_k` override, scheduled from `4` to `0`. The benchmark
variant is architecturally identical to `full_msa_to_face`; parameter count is
unchanged at `3,106,690`.

E48 result update: reject. The 500-step Runpod gate reached best/final
`val_lddt_ca=0.2274`, FoldScore `0.2191`, `val_ca_drmsd=15.7749`, and
predicted/true C-alpha radius of gyration `5.5326 / 15.4034`. The local
scaffold did not recover the stronger early-validation band and still ended
in coordinate collapse.

Immediate E49 branch: test Topotein-style outer-edge communication among
selected cells. Instead of forcing filled faces/tetras to be closed, let
face/tetra cochains exchange messages through selected boundary edges that
leave one cell and enter another, preserving edge-level geometry and the
combinatorial-complex flexibility.

E49 implementation update: implemented a directed outer-edge context pass
with `simplex_outer_edge_context_scale=0.25`. Each selected face/tetra
cochain gathers outgoing and incoming pair-edge states from its vertices to
selected neighbors outside the cell, then uses that pooled context to update
the higher-rank state before boundary readout. This is distinct from E39's
shared-boundary face averaging and better matches the Topotein outer-edge
neighborhood. Parameter count is `3,183,282`, within the 5% AF2-medium
budget.

E49 result update: reject. The 500-step Runpod gate reached best/final
`val_lddt_ca=0.2695`, FoldScore `0.2429`, `val_ca_drmsd=14.5377`, and
predicted/true C-alpha radius of gyration `6.7858 / 15.4034`. This is an
improvement over E47/E48 but remains below the stronger E22/E25/E30 pilot
band and far below E15. The next branch should probably address global
coordinate expansion directly inside the topological readout, not add another
same-rank or edge-context side pass.

Immediate E50 branch: add a selected-boundary expansion hinge to the
coordinate realization losses. This should remain inside the simplicial view:
for each selected face or tetra, act only on the boundary edges induced by
that selected higher-rank cell, and penalize only contraction of the predicted
edge length below the true selected boundary length. It is not a generic
radius-of-gyration or all-pairs distance objective; it asks whether the
learned 2-/3-cell complex can realize its own boundary 1-skeleton without
collapsing. Keep the architecture identical to `full_msa_to_face`, so the
parameter count remains `3,106,690`.

E50 result update: reject. The hinge produced the intended early expansion
signal at step 250, with predicted/true C-alpha radius of gyration
`10.6522 / 15.4034`, but validation lDDT was only `0.1593`. By step 500,
`val_lddt_ca` recovered to `0.2731` while radius collapsed back to
`6.6087 / 15.4034`, below E49 and well below E15. The next branch should
avoid simply increasing auxiliary expansion weight; it should make realized
selected-boundary geometry influence the topology/readout stream directly.

Immediate E51 branch: combine the selected-boundary expansion hinge with the
existing simplicial structure-readout pathway. Keep the same selected
face/tetra complex and expansion loss from E50, but use
`full_msa_to_face_structure_readout` so the simplex pair/single boundary
readouts are injected into the representation consumed by the structure
module. This tests whether the topological realization signal must be on the
same path that places atoms, rather than only an auxiliary loss attached
after the fact.

E51 result update: reject. Structure readout plus the E50 hinge reached only
`val_lddt_ca=0.2375` at step 250 and ended at `0.2272`, with predicted/true
C-alpha radius `5.7622 / 15.4034`. It neither preserved E50's early expansion
effect nor recovered E49's final lDDT. Avoid broad structure-readout
reinjection for now; the next useful branch should likely return to the
best E15 family and change optimization/curriculum around the existing
selected-boundary losses rather than adding another auxiliary/readout path.

## 2026-05-11 E62 Update

E62 result: reject. The scheduled Hodge face residual from the E55 checkpoint
completed at step 3500 with `val_lddt_ca=0.3468`, FoldScore `0.3450`,
`val_ca_drmsd=10.9016`, and predicted/true C-alpha radius of gyration
`10.7278 / 15.4034`. The Hodge residual is a clean topological incidence
operation and adds no parameters, but its lower/upper face-adjacency mixing
still did not preserve E55's `0.3604` C-alpha lDDT.

Immediate E63 branch: use the selected-boundary diagnostics as the next
topology-native target. E61/E62 leave selected face/tetra boundary lDDT below
`0.5` with high contraction fractions, so test a conservative selected-boundary
lDDT objective only on the boundary 1-skeleton induced by model-selected
faces and tetras. This remains inside the SimplexFold motivation: it asks the
explicit sparse complex to realize its own edges, rather than optimizing a
generic all-pairs or full-structure lDDT surrogate.

E63 result: keep for confirmation. The selected-boundary lDDT objective reached
`val_lddt_ca=0.3611` at step 3500, a small but real improvement over E55's
`0.3604`, with FoldScore `0.3576`, `val_ca_drmsd=10.6815`, and
predicted/true C-alpha radius `11.4310 / 15.4034`. The important topological
signal also moved in the right direction: selected face/tetra boundary lDDT
rose to `0.5208` / `0.5065`, and contraction fractions fell to roughly `0.69`.

Immediate E64 branch: confirm E63 beyond the 30k-example mark by continuing
the selected-boundary lDDT objective from the E63 checkpoint to step 4000
(`32,000` effective examples at batch 8). This is a necessary stability check
because E56 showed that continuing E55 without this topology-mediated loss
regressed C-alpha lDDT despite improving some global geometry metrics.

## 2026-05-14 E123 Ramp Prepared

Current leader E120 improves the selected-complex branch to `val_lddt_ca=0.4248`
with strong selected face/tetra boundary lDDT, but it is still not a 30k-step
candidate. E121 is testing an abrupt pre-triangle simplex cochain update from
the E120 checkpoint on the owned Runpod pod. If that abrupt gate is unstable
or too strong, the next topology-native fallback is a ramped pair-only
pre-triangle route.

The E123 plan keeps parameters unchanged and keeps the change inside the
README motivation: selected face/tetra cochains write back into the pair
1-skeleton before AlphaFold2 triangle multiplication and attention globalize
the pair tensor. The runtime ramp lets a resumed checkpoint receive this
message gradually, so the experiment tests whether explicit higher-rank
boundary cochains can become useful inputs to triangle reasoning rather than
acting as an unrelated output loss.

When E121b returns, the next launch decision should be mechanical rather than
improvised. If E121b underperforms E120 or only shows a tiny low-0.4 gain,
prefer E123: same checkpoint and recipe, but pair-only pre-triangle injection
ramped from `0.0` to `0.25` over steps 7500-8000 with single pre-triangle
updates held at `0.0`. If E121b/E123 suggest the pre-triangle route is useful
but still fails to translate selected-boundary geometry into global C-alpha
assembly, use E124 as the next short gate: add the face boundary-edge-frame
gate at `0.05`, with no tetra gate unless the parameter budget is re-audited.

## 2026-05-14 E124 Edge-Frame Gate Prepared

E121b remains the active run, and E123 remains the no-new-parameter fallback.
The next heavier fallback is now implemented locally but should stay parked
until E121b returns: a face-boundary-edge-frame gate. It scalarizes selected
face geometry in each directed boundary edge frame, concatenates that oriented
geometry with the face cochain and current boundary `Z_ij`, and uses the
result to gate only the selected face-to-edge message before scatter back into
the pair tensor.

This is topology-native rather than an lDDT hack: the change asks whether a
learned 2-simplex state can communicate through its own oriented boundary
1-simplices more cleanly than an ungated cochain scatter. It adds no new loss
and no dense coordinate objective. To stay within the AF2-medium +5% cap, the
gate is intentionally face-boundary-only for now; the audited E120-style
medium config with the new gate has `3,239,522` parameters, below the
`3,261,974` cap.

## 2026-05-15 E132 Runtime Boundary-Readout Ramp Prepared

E130 is still the active Runpod run and E131 remains the first parked
fallback. I prepared one additional no-parameter fallback for the same
boundary-cochain family: runtime schedules for
`simplex_boundary_hodge_readout_scale` and
`simplex_boundary_edge_star_readout_scale`.

Rationale: E130/E131 change the selected boundary 1-cochain readout, not the
output loss. If E130 or E131 regresses, the failure may be an abrupt
resume-time perturbation rather than a negative result for Hodge centering or
edge-star smoothing themselves. The new runtime scales let a resumed checkpoint
receive the same topological operation gradually over the short gate.

This stays within the SimplexFold view: selected faces and tetras still write
through their boundary 1-skeleton into `Z_ij`; the only change is a training
schedule for how strongly the selected boundary cochain is Hodge-centered or
edge-star-smoothed before pair update. It adds no parameters and no generic
C-alpha, radius, all-pairs distance, or coordinate objective.

Do not launch this while E130 is active. If E130/E131 look promising but
unstable, the next clean gate is a ramped E132 variant from the E128
checkpoint with the E128 recipe fixed and either:

```bash
--simplex-boundary-hodge-readout-scale 0.25 \
--simplex-boundary-hodge-readout-runtime-scale 0.0 \
--simplex-boundary-hodge-readout-runtime-scale-final 0.25 \
--simplex-boundary-hodge-readout-runtime-scale-ramp-start-step 8500 \
--simplex-boundary-hodge-readout-runtime-scale-ramp-steps 500
```

or the E131 combination:

```bash
--simplex-boundary-hodge-readout-scale 0.25 \
--simplex-boundary-edge-star-readout-scale 0.5 \
--simplex-boundary-hodge-readout-runtime-scale 0.0 \
--simplex-boundary-hodge-readout-runtime-scale-final 0.25 \
--simplex-boundary-edge-star-readout-runtime-scale 0.0 \
--simplex-boundary-edge-star-readout-runtime-scale-final 0.5 \
--simplex-boundary-hodge-readout-runtime-scale-ramp-start-step 8500 \
--simplex-boundary-edge-star-readout-runtime-scale-ramp-start-step 8500 \
--simplex-boundary-hodge-readout-runtime-scale-ramp-steps 500 \
--simplex-boundary-edge-star-readout-runtime-scale-ramp-steps 500
```

Decision rule is unchanged: a short gate must beat E128/E130/E131 on primary
C-alpha lDDT, keep FoldScore/dRMSD/Rg coherent, and clear `0.45` before any
30k-step spend.

## 2026-05-15 E133 Runtime Triangle-Attention Ramp Prepared

E130 remains active on the owned Runpod pod, so E133 is only a parked local
candidate. It extends the runtime scheduling idea to the sparse
simplex-to-triangle-attention route: keep the projection modules allocated by
the static triangle-attention bias/value config, but let training-time
overrides ramp the actual sparse selected face/tetra bias or value
contribution during a resumed gate.

This stays inside the SimplexFold motivation. The operation is not an
output-side lDDT or coordinate hack; it asks whether explicit higher-order
cochains can enter the same AF2 triangle-reasoning pathway more gently. E128
showed that a damped sparse simplex triangle-attention bias can help global
C-alpha lDDT, while E129 showed that abruptly adding value content improved
selected-boundary diagnostics but worsened global assembly. E133 tests whether
the problem is the simplicial route itself or the sudden activation of that
route in a resumed checkpoint.

Do not launch E133 while E130 is active. If E130 returns weakly and E132-style
readout ramps do not look like the next best use of the pod, the clean E133
gate is a ramped triangle-attention probe from E128/E130-family weights with
the E128 selected-complex recipe fixed. Candidate examples:

```bash
--simplex-triangle-attention-bias-scale 0.0125 \
--simplex-triangle-attention-bias-runtime-scale 0.0 \
--simplex-triangle-attention-bias-runtime-scale-final 0.0125 \
--simplex-triangle-attention-bias-runtime-scale-ramp-start-step 8500 \
--simplex-triangle-attention-bias-runtime-scale-ramp-steps 500
```

or, only if retesting the E129 value route deliberately:

```bash
--simplex-triangle-attention-value-scale 0.025 \
--simplex-triangle-attention-value-runtime-scale 0.0 \
--simplex-triangle-attention-value-runtime-scale-final 0.025 \
--simplex-triangle-attention-value-runtime-scale-ramp-start-step 8500 \
--simplex-triangle-attention-value-runtime-scale-ramp-steps 500
```

Decision rule is still strict: the branch must beat E128/E130/E129 on primary
C-alpha lDDT and keep FoldScore, dRMSD, and C-alpha Rg coherent. It must clear
`0.45` in the short gate before any 30k-step spend.

## 2026-05-15 E134 Edge-Star Residual Boundary Readout Prepared

E130 is still active on the owned Runpod pod, so E134 is only a parked local
candidate. It adds a parameter-neutral high-pass transform on the selected
boundary-edge 1-cochain: after selected face/tetra states scatter to boundary
edges and after any Hodge centering or edge-star smoothing, the readout can be
blended toward its deviation from the lower edge-star mean before it updates
`Z_ij`.

This is a simplicial/topological change, not a new loss. E131's edge-star
readout tests diffusion across residue stars; E134 tests the complementary
coexact-style residual signal, asking whether the pair trunk needs the
non-smooth circulation/local contrast component of the selected boundary
cochain rather than another smoothed average. It adds no parameters and no
generic C-alpha, radius, all-pairs distance, or coordinate objective.

Do not launch E134 while E130 is active. If E130/E131/E132 show that
boundary-edge readout is plausible but smoothing suppresses useful local
orientation/contrast, a short E134 gate from the E128-family checkpoint should
keep the E128 selected-complex recipe fixed and add:

```bash
--simplex-boundary-hodge-readout-scale 0.25 \
--simplex-boundary-edge-star-residual-scale 0.25
```

If testing it specifically against E131, use the same static E131 edge-star
smoothing scale and add only the residual scale:

```bash
--simplex-boundary-edge-star-readout-scale 0.5 \
--simplex-boundary-edge-star-residual-scale 0.25
```

Decision rule is unchanged: reject unless it beats E128 and any returned
E130/E131/E132 result on primary C-alpha lDDT while keeping FoldScore, dRMSD,
and C-alpha Rg coherent. It still must clear `0.45` before any 30k-step spend.

## 2026-05-15 E135 Ramped Edge-Star Residual Readout Prepared

E135 extends E134 with runtime scheduling for
`simplex_boundary_edge_star_residual_scale`. The motivation is the same
resume-safety lesson as E132/E133: a high-pass selected boundary 1-cochain may
be useful, but abruptly activating it in a checkpoint already tuned around
smooth boundary readout could destabilize global assembly.

This remains a topology-native change. The model still computes selected
face/tetra cochains, scatters them through boundary incidence to selected
edges, then transforms that boundary-edge cochain before writing into `Z_ij`.
The runtime schedule changes only the strength of the edge-star residual
operation. It adds no parameters and no output-side loss.

Do not launch E135 while E130 is active. If E134 is the right next gate after
E130 returns, prefer the ramped version first:

```bash
--simplex-boundary-edge-star-residual-scale 0.25 \
--simplex-boundary-edge-star-residual-runtime-scale 0.0 \
--simplex-boundary-edge-star-residual-runtime-scale-final 0.25 \
--simplex-boundary-edge-star-residual-runtime-scale-ramp-start-step 8500 \
--simplex-boundary-edge-star-residual-runtime-scale-ramp-steps 500
```

Decision rule is the same as E134: beat E128 and any returned E130/E131/E132
result on primary C-alpha lDDT, keep FoldScore/dRMSD/Rg coherent, and clear
`0.45` before any 30k-step spend.
