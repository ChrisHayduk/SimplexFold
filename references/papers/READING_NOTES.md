# Reference Paper Reading Notes

These notes summarize local reference PDFs used to guide SimplexFold
experiment design. The PDFs themselves are kept locally in this directory and
ignored by git until redistribution rights are confirmed.

## From Nodes to Complexes: A Guide to Topological Deep Learning

Main takeaways for SimplexFold:

- Topological deep learning is not just adding a loss on graph outputs. It
  changes the domain and message-passing neighborhoods: cochains live on
  nodes, edges, faces, cells, hyperedges, or combinatorial-complex cells.
- A TDL pipeline has an explicit complex-construction step, topological neural
  layers, and topology-aware evaluation. For SimplexFold, this means neighbor
  selection, incidence, coface structure, and face/tetra realization are first
  class design choices.
- The useful architectural distinction from a GNN is two-level aggregation:
  intra-neighborhood aggregation within a rank and inter-neighborhood
  aggregation across ranks. This supports experiments that improve
  residue-edge-face-tetra communication, not generic dense coordinate losses.
- Incidence matrices and adjacency-through-incidence operators are the natural
  handles for principled simplex updates. This supports Hodge/co-boundary,
  outer-edge, and local-to-global topology curricula.

Implication:

The next experiments should continue to alter how selected cells are built and
how cochains exchange information across ranks. Dense all-pairs metric losses
remain poorly aligned with the SimplexFold claim unless they are restricted to
the selected cell complex.

## Topotein: Topological Deep Learning for Protein Representation Learning

Main takeaways for SimplexFold:

- Topotein argues that residue-level graphs bottleneck communication between
  secondary-structure-scale units. Protein topology benefits from persistent
  higher-rank states and communication through multiple boundary/interior
  edges rather than single superedges.
- The Protein Combinatorial Complex ranks residues, directed interactions,
  secondary-structure cells, and a global protein cell. For NanoFold, DSSP/SSE
  labels cannot be required in official hidden inference, but the hierarchy
  motivates latent segment or selected-cell cochains built from official
  sequence/MSA features and recycled geometry.
- Combinatorial complexes relax strict boundary requirements. This is
  important because E44-E47 showed that forcing or weighting selected
  face/tetra cells by flag-closure assumptions hurts the NanoFold validation
  curve. The paper gives a principled reason to preserve irregular/open cells
  instead of making closure the next main path.
- Topotein's "outer-edge" neighborhoods are a strong design cue: higher-rank
  cells should communicate through edges that leave one cell and enter another,
  retaining edge-level geometry instead of collapsing a cell pair to one
  coarse superedge.
- Edge-centric scalarization is protein-relevant. Higher-rank vector or
  geometric content can be projected onto associated boundary-edge frames,
  preserving SE(3)-equivariance while allowing scalar MLPs to process
  orientation-aware information.
- The paper's results suggest comprehensive topological integration matters.
  Superficial higher-rank features without dedicated update paths can be
  counterproductive.

Experiment implications:

- Avoid more flag-closure heuristics for now; closure is not required for the
  combinatorial-complex view and has failed empirically.
- Prefer an adaptive local-to-global selected-complex curriculum over a fixed
  closure mask or fixed K expansion.
- Test outer-edge style communication among selected face/tetra cells through
  boundary edges that connect different selected cells.
- Revisit oriented face geometry using boundary-edge frames and residue-local
  frames, but keep the loss attached to selected faces/tetras rather than
  applying it densely.
- Consider a small global/readout cochain only if it receives and returns
  information through incidence-aware selected-cell summaries, not as a
  generic pooled feature head.
