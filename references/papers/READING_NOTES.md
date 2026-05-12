# Reference Paper Reading Notes

These notes summarize local reference PDFs used to guide SimplexFold
experiment design. The PDFs themselves are kept locally in this directory and
ignored by git until redistribution rights are confirmed.

Read status: both PDFs were re-read in full on 2026-05-11 from local
`pdftotext -layout` extraction and rechecked on 2026-05-12. The local copies
hash-match the files in `/Users/christopherhayduk/Downloads/`.

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
- For NanoFold, do not depend on DSSP/SSE labels or external structure
  annotations. If we borrow Topotein's secondary-structure hierarchy, it must
  be through latent selected cells derived from official features and recycled
  predictions.
- Treat outer-edge context as an architectural path that likely needs gating
  or scheduling. Fresh high-rank context modules can improve global geometry
  while disrupting local C-alpha agreement, so the next useful test should
  measure whether weak or delayed cochain exchange preserves the E55 lDDT peak.
- Add topology-aware diagnostics where cheap: selected-cell degree, boundary
  edge length error, and per-rank message scale can explain whether a run is
  learning a useful complex or merely increasing global expansion.

Full-read details to carry forward:

- Topotein's PCC ranks are residues, directed interaction edges, secondary
  structure cells, and a protein cell. NanoFold cannot require DSSP/SSE labels
  at official inference time, but the directed-edge and outer-edge machinery
  can be adapted to SimplexFold's learned selected faces/tetras.
- The outer-edge equations use incidence through residues and directed edges
  to collect edges that leave or enter a higher-rank cell while excluding
  redundant self/internal edges. This is the cleanest justification for E58's
  directed outer-edge context from the E55 checkpoint.
- TCPNet's lesson is that topological enhancement must be deeply integrated:
  cells need their own update paths and geometric frames. The negative result
  for a shallow ETNN adaptation matches our weak E39-E52 sidecar/readout
  attempts and argues for resume-compatible architecture changes over more
  output-only losses.
- Edge-centric scalarization is a second architecture branch after E58: vector
  or orientation content from selected cells should be projected onto boundary
  or outer-edge frames before scalar MLP updates, preserving geometric
  sensitivity while staying in the selected complex.

## 2026-05-11 Experiment Rules From Full Reread

I re-extracted both PDFs with `pdftotext -layout` and reread the full texts.
The rules I will carry into the next SimplexFold experiments are:

- Treat topology construction as part of the model, not as a reporting detail.
  The selected neighbor graph, face/tetra cells, incidence relations, and
  outer-edge neighborhoods are experimental variables.
- Prefer interventions that change cochain communication: residue-to-edge,
  edge-to-face, face/tetra-to-boundary-edge, and outer-edge exchange between
  selected cells. A loss is justified only when it supervises realization of
  the selected sparse complex.
- Avoid strict flag-closure as the default prior. Topotein's combinatorial
  complex framing supports irregular/open higher-rank cells, and our E44-E47
  closure family already looked empirically brittle.
- Do not borrow Topotein's DSSP/SSE labels in official NanoFold paths. Any
  secondary-structure-like hierarchy must be latent and derived from official
  sequence/MSA features plus recycled model predictions.
- Add topology-aware diagnostics alongside lDDT/FoldScore when a run changes
  the complex: selected face/tetra coverage, boundary-edge reuse, outer-edge
  degree, and eventually boundary-edge length error.
- Keep the near-term architecture path on scheduled or gated outer-edge
  communication, followed by edge-centric scalarization/readout if the gated
  context preserves local C-alpha lDDT.

## Current Session Recheck

The current full-text recheck did not add a better generic objective. It
reinforces that the most defensible SimplexFold changes are those that alter
the learned cell complex or its incidence/outer-edge communication paths. For
the active E73/E74 branch, the paper-derived rationale is strongest for
runtime-gated boundary-edge messages and for changing the recycled-geometry
prior in neighbor selection, because both choices affect which edge/face/tetra
cochains exist and how they exchange information.

## 2026-05-12 Recheck

Extraction audit: the guide is 28 pages and `2509.03885v1.pdf` is 22 pages;
the extracted text totals about 14.5k words. The saved copies in
`references/papers/` still hash-match the Downloads originals.

Experiment implication after rereading the full texts: keep prioritizing
topology-construction and cochain-communication changes over generic
coordinate losses. E74 is aligned with that rule because it changes the
recycled-geometry prior used to construct the sparse edge/face/tetra complex.
If E74 turns over, E75 is the next paper-aligned branch because top-k caps
change which higher-rank cells exist and therefore which selected-cell losses
and messages are active.

## 2026-05-12 Full Reread for Sparse-Cell Branch

I reread both extracted texts again after E79 became the leading branch.
The papers strengthen the interpretation that E79/E82 are the right kind of
experiment: they modify the selected combinatorial complex itself by changing
which higher-rank face/tetra cochains exist.

Specific ideas to carry forward:

- The TDL guide's distinction between intra-neighborhood and
  inter-neighborhood aggregation maps cleanly to SimplexFold's residue/pair
  updates versus face/tetra-to-boundary-edge exchange. Future changes should
  specify which aggregation route they alter.
- Topotein's directed PCC edges argue against treating boundary transport as
  a fully undirected pooled message. A better SimplexFold route would keep
  source/target orientation when passing selected face/tetra information back
  into pair or single streams.
- Topotein's outer-edge equations are still the strongest literature-backed
  justification for cell-to-cell communication. Earlier dense outer-edge
  experiments were disruptive, but a sparse, delayed, incidence-normalized
  outer-edge route from the E79/E82 complex remains worth testing.
- Topotein's negative comparison for shallow TDL adaptations matches our
  failed sidecar/readout runs. The next attempt should be an actual selected
  cochain update path, not a pooled output correction.
- The high boundary-edge degree and low unique-edge fraction in E79 are
  exactly the kind of topological pathology the papers suggest measuring.
  Degree-penalized cell scoring and incidence-normalized message transport
  are therefore principled topology-construction fixes, not arbitrary metric
  hacks.

## 2026-05-12 E81/E84 Update

I rechecked the extracted full text while E84 was running from the E81
checkpoint. The references still point away from generic lDDT-targeted losses
and toward two concrete SimplexFold levers:

- Complex construction: choose which rank-2/rank-3 cochains exist. E81's
  degree-penalized scorer is directly aligned with this because it changes
  the selected face/tetra complex by discouraging repeated use of the same
  boundary edges.
- Incidence-aware communication: when construction alone stops improving,
  normalize or gate boundary/outer-edge messages by selected edge-cell degree
  while preserving directed source/target incidence.

E81's returned diagnostics make this literature link stronger: the run
improved primary lDDT while also improving selected-boundary lDDT, boundary
length error, contraction fraction, and boundary unique-edge fraction. That is
evidence that the change helped the learned complex itself, not merely the
output metric. E84 is therefore a justified short continuation. If it regresses,
the next paper-aligned branch should be incidence-normalized boundary or
directed outer-edge transport rather than another output-coordinate loss.
