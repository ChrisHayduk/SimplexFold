## Current Plan: E78 Light-Geometry Selector Continuation

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

The active branch is now E80 on the owned H100 pod `o1dy17ouv8w5mz` as
`e80_light_geom0025_from_e78_s7000_c256_m64`. It resumes the E78 checkpoint
from step 6500 to 7000 with the same light-geometry topology-construction
recipe. Route the next run by evidence: if E80 keeps
improving primary lDDT and does not collapse selected-boundary lDDT/length,
continue short gates until the curve bends or a plausible longer confirmation
emerges. If E80 loses primary lDDT while selected-boundary diagnostics remain
strong, start E79 from the strongest E78/E80 checkpoint so the next change
acts upstream on which higher-rank cochains exist. Do not launch a blind
30,000-step confirmation until a branch shows a credible trajectory toward
`val_lddt_ca > 0.7`, not merely a small local best below 0.4.

The other prepared alternatives are E75 and E79. E75 caps active face/tetra
cells per anchor with `--simplex-face-top-k` and `--simplex-tetra-top-k`,
ranking candidate cells by selected boundary-edge logits. E79 adds a runtime
schedule for those caps so the selected higher-rank complex can be sparsified
gradually during a continuation. Both are cell-complex construction changes
rather than generic output-coordinate losses.

The 2026-05-12 full-text recheck of the saved reference PDFs keeps this plan
topology-first. The strongest paper-derived criterion is that a new branch
should change the selected cell complex, incidence/outer-edge communication,
or realization of selected cells. E74/E78 satisfy that by changing the
recycled-geometry prior in topology construction; E75/E79 remain prepared
fallbacks because they sparsify which higher-rank cochains exist and send
messages.

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
