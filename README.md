# SimplexFold

SimplexFold is a research fork of a minimal AlphaFold2-style model that asks a
specific question:

> What happens if a protein-folding trunk reasons not only over residues and
> residue pairs, but also over learned triangular faces and tetrahedral cells?

AlphaFold2 already has powerful pairwise and triangle-style reasoning inside
the Evoformer. Its pair tensor `Z_ij` acts like an edge representation between
residues, and the triangle updates move information through third residues to
make those pair features more geometrically consistent. But the triangle
operations still write back into an edge tensor. They do not maintain persistent
learned states for the filled triangle `(i, j, k)` or the tetrahedron
`(i, j, k, l)`.

SimplexFold makes those higher-order objects explicit.

```text
MSA representation M
        <-> pair / edge tensor Z_ij
        <-> sparse face tensor F_ijk
        <-> sparse tetra tensor U_ijkl
        -> structure module
        -> recycled geometry
        loops back into the next pass
```

## Intuition

A simplex is the simplest object of a given dimension:

| Object | Simplex | Protein interpretation |
| --- | ---: | --- |
| point | 0-simplex | residue, atom, or residue frame |
| line segment | 1-simplex | bond, contact, pairwise residue relation |
| filled triangle | 2-simplex | three-residue patch with area, angles, and normal direction |
| tetrahedron | 3-simplex | four-residue packing unit with volume, compactness, and chirality |

Most protein neural networks are graph-like: they represent residues as nodes
and residue-residue relationships as edges. In topological language, that graph
is the 1-skeleton of a richer geometric object. The motivating idea behind
SimplexFold is that proteins are not only collections of pairwise contacts.
Folding is full of three-body and four-body constraints: backbone angles,
torsions, sheet geometry, turns, hydrophobic-core packing, side-chain
arrangements, cavities, and local residue-contact motifs.

A pair feature can say:

> Residue `i` is near residue `j`.

A face feature can say:

> Residues `i, j, k` form a local oriented patch with this area and angle
> pattern.

A tetra feature can say:

> Residues `i, j, k, l` form a compact local 3D packing unit with this volume,
> handedness, and steric profile.

The bet is not that pairwise distances are insufficient in principle. A perfect
distance matrix can determine a structure up to rigid motion and reflection.
The bet is that learned pair tensors are noisy, partial, and data-limited, and
that explicit higher-order geometric states may provide a useful inductive bias
for sample-efficient folding.

## Why This Is Not Just AlphaFold2 Triangle Attention

AlphaFold2 is not merely walking along the protein backbone. Its Evoformer uses
pair features, triangle multiplication, triangle self-attention, MSA-to-pair
communication, and recycling. That is already extremely strong geometric
reasoning.

The distinction here is narrower:

- AF2 updates edge states `Z_ij` through triangle-shaped computations.
- SimplexFold adds first-class face states `F_ijk` and tetra states `U_ijkl`.

In other words, AF2 has triangle-aware pair reasoning. SimplexFold experiments
with persistent higher-order cells:

```text
edges <-> faces <-> tetrahedra
  Z        F          U
```

Those states can carry features that are awkward to represent as only a bag of
edges: triangle area, face normals, internal angle systems, signed volume,
radius of gyration, local packing density, and chirality.

## Why No Templates Are Needed

SimplexFold is designed for settings where templates are unavailable. The sparse simplicial complex is constructed only from information available to the model:

1. On the first pass, topology comes from learned pair/contact logits plus a
   local sequence bias.
2. On recycled passes, topology also uses the model's own predicted C-alpha
   coordinates and residue frames.

The discrete top-k neighbor selection is stop-gradient. The selected face and
tetra tensors are ordinary differentiable PyTorch tensors, but the hard choice
of which cells exist is treated like an internal routing decision.

This mirrors the spirit of AF2 recycling: the model's previous structure
estimate becomes an internal geometric prior for the next pass.

## Sparse Construction

Dense triples and quadruples are not viable. For a crop of length `L = 256`:

- all triples: about 2.76M
- all quadruples: about 174.8M

SimplexFold instead builds an anchored top-k local complex.

For each residue `i`, choose `K` neighbors:

```text
N(i) = TopK_j score_ij
```

Then construct:

```text
faces:       (i, j, k)       where j, k    in N(i)
tetrahedra:  (i, j, k, l)    where j,k,l   in N(i)
```

The tensor layout is dense in the local neighbor dimension, which keeps it
GPU-friendly:

```text
nbr_idx: [B, L, K]
F:       [B, L, choose(K, 2), C_face]
U:       [B, L, choose(K, 3), C_tetra]
```

With `L=256` and `K=12`, this gives:

```text
faces per crop:  16,896
tetras per crop: 56,320
```

That is large enough to express local higher-order geometry, but small enough
to avoid `O(L^3)` and `O(L^4)` tensors.

## Architecture

The new block is `SimplicialEvoformer`.

At a high level:

1. Run standard MSA row attention with pair bias.
2. Run MSA column attention and MSA transition.
3. Update pair features with outer product mean and triangle modules.
4. Build a sparse neighbor graph from pair logits and optional recycled
   geometry.
5. Initialize face states from three pair edges, three residue states, and
   face geometry.
6. Initialize tetra states from six pair edges, local face states, residue
   states, and tetra geometry.
7. Run gated message passing:

```text
edge -> face
face -> tetra
tetra -> face
face/tetra -> pair
face/tetra -> single
```

8. Feed the updated pair and single streams to the structure module.
9. Recycle predicted C-alpha coordinates and residue frames into the next pass.

The implementation keeps the old AF2-style `Evoformer` available for ablations,
but the shipped configs enable `SimplicialEvoformer`.

## Geometry Features

Before recycling, simplex cells use sequence-separation features and learned
pair/single representations.

After recycling, faces receive invariant geometric descriptors such as:

- three C-alpha distances
- triangle area
- three internal angle cosines
- face normal expressed in the anchor residue's local frame

Tetrahedra receive:

- six C-alpha distances
- four face areas
- signed and absolute volume
- radius of gyration

The local-frame features are invariant to global rotation and translation.
That matters because the model should reason about protein shape, not the
coordinate system used to express it.

## MSA To Simplex Communication

The default path is:

```text
MSA -> pair -> face -> tetra -> pair/single -> MSA
```

This lets evolutionary information flow into simplex states through the pair
representation and then back into the MSA through pair-biased attention.

There is also an optional low-rank MSA-to-face moment:

```text
mean_a (A M_ai) * (B M_aj) * (C M_ak)
```

computed only over selected faces. This is a cheap way to test whether
third-order evolutionary couplings add value without constructing dense
third-order MSA tensors.

## Auxiliary Losses

A risk with any adapter is that the main network simply routes around it. To
make the simplex states learn useful geometry, SimplexFold adds auxiliary
training losses:

- topology/contact supervision from true C-alpha distances
- face boundary distance prediction for the three triangle edges
- face-area regression for selected triangles
- tetra boundary distance prediction for the six tetrahedron edges
- tetra geometry regression for signed volume, absolute volume, and compactness
- boundary consistency tying face distance distributions to pair distograms and
  tetra distance distributions to their boundary faces

These losses use labels only during training. At inference, the topology and
simplex features come only from model inputs and recycled predictions.

## Why Stop At 3-Simplices?

A 4-simplex would involve five residues. It can represent five-body motif
consistency, but it is not a new nondegenerate geometric primitive in 3D space:
proteins live in `R^3`, and tetrahedra are the highest-dimensional ordinary
simplex with real volume.

Five-residue cells may eventually be useful as a consistency or motif module,
but they are not the first thing to try. The experimental ladder is:

```text
pair-only baseline
baseline + 2-simplex faces
baseline + faces + recycled geometry
baseline + faces + 3-simplex tetrahedra
later: five-point / 4-simplex consistency ablations
```

The practical bet is:

```text
2-simplex gain > 3-simplex gain >> 4-simplex gain
```

So SimplexFold focuses on faces and tetrahedra first.

## What Is Implemented

- Sparse neighbor topology: `nbr_idx [B, L, K]`.
- Face states: `F [B, L, choose(K, 2), C_face]`.
- Tetra states: `U [B, L, choose(K, 3), C_tetra]`.
- Rigid-invariant recycled geometry features for distances, triangle areas,
  angle cosines, local normals, tetra volumes, and radius of gyration.
- Optional low-rank MSA-to-face third-order moment
  (`simplex_use_msa_to_face`).
- Gated edge-face-tetra message passing with scatter-add back to dense pair
  and single representations.
- Auxiliary contact, face-distance, face-area, tetra-distance,
  tetra-geometry, and boundary-consistency losses.
- Tiny/medium/full configs with simplex defaults.
- Benchmark harness: `scripts/benchmark_simplexfold.py`.
- Publication benchmark protocol: `BENCHMARK_PLAN.md`.

## Research Context

This project sits between several older and newer lines of work:

- Delaunay/tetrahedral protein geometry and four-body statistical potentials.
- Alpha-shape and simplicial-complex descriptions of protein surfaces,
  cavities, and contacts.
- Three-body protein-packing and decoy-discrimination potentials.
- Modern topological and geometric deep learning for protein representation,
  docking, interface quality assessment, and binding prediction.
- AlphaFold-style MSA/pair/recycling architectures.

The specific combination explored here is narrower and, to our knowledge,
still underexplored: an AF2-like sequence-to-structure model with recycling
that maintains sparse learned 2-simplex and 3-simplex states inside the trunk.

## Benchmarking

The goal is not just to show that the architecture runs. The useful scientific
claim is whether explicit higher-order cells improve accuracy, sample
efficiency, or calibration at acceptable compute.

See `BENCHMARK_PLAN.md` for the full evaluation plan. The short version:

- Compare against a matched pair-only trunk.
- Ablate faces, recycled geometry, tetrahedra, MSA-to-face moments, and
  auxiliary losses.
- Sweep `K = {8, 12, 16}`.
- Report structure metrics, contact precision, simplex-geometry losses,
  latency, memory, parameter count, and seed-to-seed variance.

## Quick Checks

```bash
pytest -q tests/test_simplex.py
pytest -q
```

## Microbenchmark

```bash
python scripts/benchmark_simplexfold.py \
  --model-config tiny \
  --device cpu \
  --length 128 \
  --msa-depth 32 \
  --extra-msa-depth 0 \
  --n-cycles 2
```
