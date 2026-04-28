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
