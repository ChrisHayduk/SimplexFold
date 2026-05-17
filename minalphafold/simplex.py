"""Sparse simplicial adapters for SimplexFold.

The module implements the plan in ``PLAN.md`` as an anchored sparse complex:
each residue owns a top-k neighbor list, faces are pairs of neighbors, and
tetrahedra are triples of neighbors.  All tensors stay dense in the local
``[B, L, M, C]`` layout so the adapter can be used inside the Evoformer
without materialising all ``O(L^3)`` faces or ``O(L^4)`` tetrahedra.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import torch

from .initialization import init_gate_linear, init_linear


@dataclass
class SimplexTopology:
    """Anchored sparse complex indices and masks.

    ``face_indices[..., :]`` stores ``(i, j, k)`` for each anchored face and
    ``tetra_indices[..., :]`` stores ``(i, j, k, l)`` for each anchored
    tetrahedron.  Masks are float tensors so they can multiply activations
    directly.
    """

    nbr_idx: torch.Tensor
    face_indices: torch.Tensor
    tetra_indices: torch.Tensor
    face_mask: torch.Tensor
    tetra_mask: torch.Tensor
    face_neighbor_slots: torch.Tensor
    tetra_neighbor_slots: torch.Tensor
    tetra_face_slots: torch.Tensor


def _combination_tensor(k: int, r: int, device: torch.device) -> torch.Tensor:
    if k < r:
        return torch.empty((0, r), device=device, dtype=torch.long)
    values = list(combinations(range(k), r))
    return torch.tensor(values, device=device, dtype=torch.long)


def _tetra_face_slots(
    face_combos: torch.Tensor,
    tetra_combos: torch.Tensor,
) -> torch.Tensor:
    """Map each local tetra triplet to its three anchored face slots."""

    if tetra_combos.numel() == 0:
        return torch.empty((0, 3), device=tetra_combos.device, dtype=torch.long)
    face_lookup = {
        (int(pair[0].item()), int(pair[1].item())): idx
        for idx, pair in enumerate(face_combos)
    }
    slots: list[list[int]] = []
    for tet in tetra_combos:
        a, b, c = (int(tet[0].item()), int(tet[1].item()), int(tet[2].item()))
        slots.append([face_lookup[(a, b)], face_lookup[(a, c)], face_lookup[(b, c)]])
    return torch.tensor(slots, device=tetra_combos.device, dtype=torch.long)


def _topk_cell_mask(cell_mask: torch.Tensor, cell_score: torch.Tensor, cell_top_k: int) -> torch.Tensor:
    limit = int(cell_top_k)
    if limit <= 0 or cell_mask.shape[-1] <= limit:
        return cell_mask
    very_negative = torch.finfo(cell_score.dtype).min / 4
    masked_score = cell_score.masked_fill(cell_mask <= 0, very_negative)
    keep_idx = torch.topk(masked_score, k=limit, dim=-1).indices
    keep = torch.zeros_like(cell_mask)
    keep.scatter_(-1, keep_idx, 1.0)
    return cell_mask * keep


def _cell_boundary_degree_penalty(
    edge_indices: torch.Tensor,
    cell_mask: torch.Tensor,
    *,
    num_residues: int,
) -> torch.Tensor:
    """Measure how much candidate cells reuse the same undirected edges."""

    if edge_indices.numel() == 0 or cell_mask.numel() == 0:
        return cell_mask.new_zeros(cell_mask.shape)
    lo = torch.minimum(edge_indices[..., 0], edge_indices[..., 1])
    hi = torch.maximum(edge_indices[..., 0], edge_indices[..., 1])
    edge_ids = lo * int(num_residues) + hi
    batch = edge_indices.shape[0]
    flat_ids = edge_ids.reshape(batch, -1)
    flat_mask = cell_mask[..., None].expand_as(edge_ids).reshape(batch, -1).to(cell_mask.dtype)
    degree = cell_mask.new_zeros((batch, int(num_residues) * int(num_residues)))
    degree.scatter_add_(1, flat_ids, flat_mask)
    edge_degree = torch.gather(degree, 1, flat_ids).reshape_as(edge_ids).clamp_min(1.0)
    return torch.log1p(edge_degree).mean(dim=-1)


def _cell_outer_edge_support(
    cell_indices: torch.Tensor,
    cell_mask: torch.Tensor,
    nbr_idx: torch.Tensor,
) -> torch.Tensor:
    """Score candidate cells by how many selected neighbor edges leave them."""

    if cell_indices.numel() == 0 or nbr_idx.numel() == 0:
        return cell_mask.new_zeros(cell_mask.shape)
    outer_nbrs = gather_single(nbr_idx, cell_indices)
    is_inner = (outer_nbrs[..., None] == cell_indices[..., None, None, :]).any(dim=-1)
    outer_mask = (~is_inner).to(cell_mask.dtype) * cell_mask[..., None, None].to(cell_mask.dtype)
    support = outer_mask.sum(dim=(-2, -1))
    max_support = max(int(cell_indices.shape[-1]) * int(nbr_idx.shape[-1]), 1)
    denom = torch.log1p(cell_mask.new_tensor(float(max_support))).clamp_min(1.0)
    return torch.log1p(support) / denom


def _cell_segment_support(
    cell_indices: torch.Tensor,
    cell_mask: torch.Tensor,
    *,
    radius: int,
) -> torch.Tensor:
    """Score cells by how much their boundary lives inside local sequence segments."""

    if cell_indices.numel() == 0 or cell_mask.numel() == 0:
        return cell_mask.new_zeros(cell_mask.shape)
    segment_radius = max(int(radius), 1)
    separations = torch.abs(cell_indices[..., :, None] - cell_indices[..., None, :]).to(cell_mask.dtype)
    cell_rank = int(cell_indices.shape[-1])
    boundary_mask = torch.triu(
        torch.ones((cell_rank, cell_rank), device=cell_indices.device, dtype=cell_mask.dtype),
        diagonal=1,
    )
    segment_support = torch.exp(-separations / float(segment_radius)) * boundary_mask
    denom = boundary_mask.sum().clamp_min(1.0)
    return segment_support.sum(dim=(-2, -1)) / denom * cell_mask.to(cell_mask.dtype)


def rbf(values: torch.Tensor, *, n_bins: int, max_value: float) -> torch.Tensor:
    """Radial basis encoding on the last implicit scalar dimension."""

    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    centers = torch.linspace(
        0.0,
        float(max_value),
        n_bins,
        device=values.device,
        dtype=values.dtype,
    )
    if n_bins == 1:
        width = torch.as_tensor(max(float(max_value), 1.0), device=values.device, dtype=values.dtype)
    else:
        width = centers[1] - centers[0]
    return torch.exp(-((values[..., None].clamp_min(0.0) - centers) / width.clamp_min(1e-6)) ** 2)


def face_geometry_dim(rbf_bins: int) -> int:
    # sequence-separation RBFs, coordinate-distance RBFs, area, 3 angles, local normal
    return 6 * rbf_bins + 7


def tetra_geometry_dim(rbf_bins: int) -> int:
    # 6 sequence RBFs + 6 distance RBFs + 4 area RBFs + 1 radius-gyration RBF
    # + signed/absolute log-volume scalars.
    return 17 * rbf_bins + 2


def _batch_arange_like(indices: torch.Tensor) -> torch.Tensor:
    shape = [indices.shape[0]] + [1] * (indices.ndim - 1)
    return torch.arange(indices.shape[0], device=indices.device).reshape(shape).expand_as(indices)


def gather_single(single: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather ``single[b, idx[b, ...]]`` for any index rank after batch."""

    batch = _batch_arange_like(idx)
    return single[batch, idx]


def gather_pair(pair: torch.Tensor, idx_a: torch.Tensor, idx_b: torch.Tensor) -> torch.Tensor:
    """Gather ``pair[b, idx_a[b, ...], idx_b[b, ...]]``."""

    batch = _batch_arange_like(idx_a)
    return pair[batch, idx_a, idx_b]


def gather_msa_columns(msa: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather MSA columns into ``[B, N_msa, ...idx_shape_without_batch, C]``."""

    if msa.ndim != 4:
        raise ValueError(f"msa must be [B, N_msa, L, C], got {tuple(msa.shape)}")
    b, n_msa = msa.shape[:2]
    batch_shape = [b, 1] + [1] * (idx.ndim - 1)
    seq_shape = [1, n_msa] + [1] * (idx.ndim - 1)
    batch = torch.arange(b, device=idx.device).reshape(batch_shape).expand(b, n_msa, *idx.shape[1:])
    seq = torch.arange(n_msa, device=idx.device).reshape(seq_shape).expand(b, n_msa, *idx.shape[1:])
    expanded_idx = idx[:, None, ...].expand(b, n_msa, *idx.shape[1:])
    return msa[batch, seq, expanded_idx]


def _safe_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(torch.sum(x * x, dim=dim).clamp_min(eps))


def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sum(a * b, dim=-1) / (_safe_norm(a) * _safe_norm(b)).clamp_min(1e-8)


def _distance_bin_indices(
    distances: torch.Tensor,
    *,
    n_bins: int,
    d_min: float = 2.0,
    d_max: float = 22.0,
) -> torch.Tensor:
    step = (d_max - d_min) / n_bins
    edges = d_min + step * torch.arange(1, n_bins, device=distances.device, dtype=distances.dtype)
    return torch.bucketize(distances, edges).clamp_(max=n_bins - 1)


def _masked_cross_entropy_from_bins(
    logits: torch.Tensor,
    target_bins: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    ce = -torch.gather(log_probs, -1, target_bins[..., None]).squeeze(-1)
    ce = ce * mask
    reduce_dims = tuple(range(1, ce.ndim))
    return ce.sum(dim=reduce_dims) / mask.sum(dim=reduce_dims).clamp_min(1.0)


def _masked_symmetric_kl(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    log_a = torch.nn.functional.log_softmax(logits_a, dim=-1)
    log_b = torch.nn.functional.log_softmax(logits_b, dim=-1)
    prob_a = log_a.exp()
    prob_b = log_b.exp()
    # Stop-gradient targets avoid a pure agreement collapse while still
    # applying gradients to both prediction heads through the two directions.
    kl_ab = torch.sum(prob_a * (log_a - log_b.detach()), dim=-1)
    kl_ba = torch.sum(prob_b * (log_b - log_a.detach()), dim=-1)
    loss = 0.5 * (kl_ab + kl_ba) * mask
    reduce_dims = tuple(range(1, loss.ndim))
    return loss.sum(dim=reduce_dims) / mask.sum(dim=reduce_dims).clamp_min(1.0)


def boundary_metric_confidence(distance_logits: torch.Tensor) -> torch.Tensor:
    """Estimate per-boundary-edge metric confidence from simplex distance heads."""

    if distance_logits.numel() == 0:
        return distance_logits.new_empty(distance_logits.shape[:-1])
    probs = torch.nn.functional.softmax(distance_logits.float(), dim=-1)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
    max_entropy = torch.log(distance_logits.new_tensor(float(distance_logits.shape[-1])).float()).clamp_min(1e-6)
    confidence = 1.0 - entropy / max_entropy
    return confidence.clamp(0.0, 1.0).to(distance_logits.dtype)


def apply_boundary_metric_gate(
    edge_update: torch.Tensor,
    distance_logits: torch.Tensor,
    edge_mask: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Gate selected boundary-edge transport by the cell's own metric confidence."""

    strength = min(max(float(scale), 0.0), 1.0)
    if strength == 0.0 or edge_update.numel() == 0:
        return edge_update
    confidence = boundary_metric_confidence(distance_logits)
    gate = 1.0 + strength * (2.0 * confidence - 1.0)
    masked_gate = torch.where(edge_mask > 0, gate, torch.ones_like(gate))
    return edge_update * masked_gate[..., None].to(edge_update.dtype)


def _selected_boundary_lddt_loss(
    pred_distances: torch.Tensor,
    true_distances: torch.Tensor,
    mask: torch.Tensor,
    *,
    cutoff: float = 15.0,
) -> torch.Tensor:
    """lDDT-style distance tolerance loss on selected simplex boundary edges."""

    local_mask = mask * (true_distances < float(cutoff)).to(mask.dtype)
    thresholds = torch.tensor(
        [0.5, 1.0, 2.0, 4.0],
        device=pred_distances.device,
        dtype=pred_distances.dtype,
    )
    error = torch.abs(pred_distances - true_distances)[..., None]
    edge_score = torch.clamp(1.0 - error / thresholds, min=0.0, max=1.0).mean(dim=-1)
    edge_loss = 1.0 - edge_score
    reduce_dims = tuple(range(1, edge_loss.ndim))
    return (edge_loss * local_mask).sum(dim=reduce_dims) / local_mask.sum(dim=reduce_dims).clamp_min(1.0)


def _selected_boundary_contraction_loss(
    pred_distances: torch.Tensor,
    true_distances: torch.Tensor,
    mask: torch.Tensor,
    *,
    distance_scale: float,
    tolerance: float,
) -> torch.Tensor:
    """One-sided loss for collapsed selected simplex boundary edges."""

    denom = torch.log1p(torch.as_tensor(distance_scale, device=pred_distances.device, dtype=pred_distances.dtype))
    denom = denom.clamp_min(1.0)
    pred_log = torch.log1p(pred_distances) / denom
    true_log = torch.log1p(true_distances) / denom
    contraction = torch.clamp(true_log - pred_log - float(tolerance), min=0.0)
    loss = torch.nn.functional.smooth_l1_loss(contraction, torch.zeros_like(contraction), reduction="none") * mask
    reduce_dims = tuple(range(1, loss.ndim))
    return loss.sum(dim=reduce_dims) / mask.sum(dim=reduce_dims).clamp_min(1.0)


def _selected_cell_centroid_contraction_loss(
    pred_points: torch.Tensor,
    true_points: torch.Tensor,
    pred_global_center: torch.Tensor,
    true_global_center: torch.Tensor,
    mask: torch.Tensor,
    *,
    distance_scale: float,
    tolerance: float,
) -> torch.Tensor:
    """One-sided loss for selected cells collapsed toward the chain center."""

    pred_centroid = pred_points.mean(dim=-2)
    true_centroid = true_points.mean(dim=-2)
    pred_radius = _safe_norm(pred_centroid - pred_global_center[:, None, None, :])
    true_radius = _safe_norm(true_centroid - true_global_center[:, None, None, :])
    denom = torch.log1p(torch.as_tensor(distance_scale, device=pred_points.device, dtype=pred_points.dtype))
    denom = denom.clamp_min(1.0)
    pred_log = torch.log1p(pred_radius) / denom
    true_log = torch.log1p(true_radius) / denom
    contraction = torch.clamp(true_log - pred_log - float(tolerance), min=0.0)
    loss = torch.nn.functional.smooth_l1_loss(contraction, torch.zeros_like(contraction), reduction="none") * mask
    reduce_dims = tuple(range(1, loss.ndim))
    return loss.sum(dim=reduce_dims) / mask.sum(dim=reduce_dims).clamp_min(1.0)


def _cell_closure_weighted_mask(
    cell_mask: torch.Tensor,
    true_distances: torch.Tensor,
    *,
    closure_weight: float,
    cutoff: float,
    temperature: float,
) -> torch.Tensor:
    """Blend a cell mask with the soft flag-complex closure of its boundary."""

    strength = min(max(float(closure_weight), 0.0), 1.0)
    if strength == 0.0:
        return cell_mask
    edge_temperature = max(float(temperature), 1e-6)
    edge_probs = torch.sigmoid((float(cutoff) - true_distances) / edge_temperature).clamp_min(1e-6)
    closure = edge_probs.log().mean(dim=-1).exp().to(cell_mask.dtype)
    return cell_mask * ((1.0 - strength) + strength * closure)


def _boundary_degree_weights(
    edge_indices: torch.Tensor,
    edge_mask: torch.Tensor,
    *,
    num_residues: int,
) -> torch.Tensor:
    """Weight selected boundary edges by inverse undirected incidence degree."""

    if edge_mask.numel() == 0:
        return edge_mask
    a, c = edge_indices.unbind(dim=-1)
    lo = torch.minimum(a, c)
    hi = torch.maximum(a, c)
    edge_ids = lo * int(num_residues) + hi
    batch = edge_indices.shape[0]
    flat_ids = edge_ids.reshape(batch, -1)
    active = (edge_mask > 0).to(edge_mask.dtype)
    flat_mask = active.reshape(batch, -1)
    degree = edge_mask.new_zeros((batch, int(num_residues) * int(num_residues)))
    degree.scatter_add_(1, flat_ids, flat_mask)
    flat_degree = torch.gather(degree, 1, flat_ids).reshape_as(edge_mask)
    return edge_mask / flat_degree.clamp_min(1.0)


def _selected_simplex_shape_loss(
    pred_points: torch.Tensor,
    true_points: torch.Tensor,
    mask: torch.Tensor,
    *,
    length_scale: float,
) -> torch.Tensor:
    """Rigidly align selected simplex vertices and score their local shape."""

    pred_centered = pred_points - pred_points.mean(dim=-2, keepdim=True)
    true_centered = true_points - true_points.mean(dim=-2, keepdim=True)
    with torch.no_grad():
        covariance = torch.matmul(
            pred_centered.detach().to(torch.float32).transpose(-1, -2),
            true_centered.detach().to(torch.float32),
        )
        u, _, vh = torch.linalg.svd(covariance, full_matrices=False)
        uvh = torch.matmul(u, vh)
        correction = torch.ones(*covariance.shape[:-2], 3, device=pred_points.device, dtype=torch.float32)
        correction[..., -1] = torch.where(
            torch.linalg.det(uvh) < 0.0,
            -1.0,
            1.0,
        )
        rotation = torch.matmul(torch.matmul(u, torch.diag_embed(correction)), vh)
    rotation = rotation.to(pred_points.dtype)
    aligned_pred = torch.matmul(pred_centered, rotation)
    squared_error = torch.sum((aligned_pred - true_centered) ** 2, dim=-1)
    cell_rmsd = torch.sqrt(squared_error.mean(dim=-1).clamp_min(0.0))
    cell_loss = cell_rmsd / max(float(length_scale), 1e-6)
    reduce_dims = tuple(range(1, cell_loss.ndim))
    return (cell_loss * mask).sum(dim=reduce_dims) / mask.sum(dim=reduce_dims).clamp_min(1.0)


def _triangle_area(x_i: torch.Tensor, x_j: torch.Tensor, x_k: torch.Tensor) -> torch.Tensor:
    return 0.5 * _safe_norm(torch.cross(x_j - x_i, x_k - x_i, dim=-1))


def _backbone_frames_from_atom14(atom14_coords: torch.Tensor) -> torch.Tensor:
    n = atom14_coords[..., 0, :]
    ca = atom14_coords[..., 1, :]
    c = atom14_coords[..., 2, :]
    e1 = c - ca
    e1 = e1 / _safe_norm(e1)[..., None].clamp_min(1e-8)
    n_axis = n - ca
    e2 = n_axis - torch.sum(n_axis * e1, dim=-1, keepdim=True) * e1
    e2 = e2 / _safe_norm(e2)[..., None].clamp_min(1e-8)
    e3 = torch.cross(e1, e2, dim=-1)
    e3 = e3 / _safe_norm(e3)[..., None].clamp_min(1e-8)
    return torch.stack([e1, e2, e3], dim=-1)


def _express_in_frame(frames: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...ij,...i->...j", frames, vectors)


def build_simplex_topology(
    score: torch.Tensor,
    *,
    neighbor_k: int,
    seq_mask: Optional[torch.Tensor] = None,
    pair_mask: Optional[torch.Tensor] = None,
    recycled_ca_coords: Optional[torch.Tensor] = None,
    local_neighbor_k: int = 0,
    local_radius: int = 4,
    local_bias: float = 5.0,
    long_min_sep: int = 24,
    long_bias: float = 0.0,
    geometry_distance_weight: float = 0.1,
    boundary_closure_weight: float = 0.0,
    boundary_closure_temperature: float = 1.0,
    face_top_k: int = 0,
    tetra_top_k: int = 0,
    cell_score_degree_penalty: float = 0.0,
    cell_score_outer_edge_weight: float = 0.0,
    cell_score_segment_weight: float = 0.0,
    segment_radius: int = 4,
) -> SimplexTopology:
    """Select top-k neighbors and expand them into anchored faces/tetrahedra.

    The input ``score`` is typically a learned contact/topology logit from the
    pair representation.  Coordinate-derived terms are optional and should be
    passed only for recycled structure cycles.
    """

    if score.ndim != 3:
        raise ValueError(f"score must be [B, L, L], got {tuple(score.shape)}")
    b, l, _ = score.shape
    device = score.device
    dtype = score.dtype
    k = min(int(neighbor_k), max(l - 1, 0))

    residue_ids = torch.arange(l, device=device)
    sep = torch.abs(residue_ids[:, None] - residue_ids[None, :])
    work_score = score.clone()
    if local_radius >= 0 and local_bias != 0.0:
        work_score = work_score + (sep <= int(local_radius)).to(dtype)[None, :, :] * float(local_bias)
    if long_min_sep >= 0 and long_bias != 0.0:
        work_score = work_score + (sep >= int(long_min_sep)).to(dtype)[None, :, :] * float(long_bias)
    if recycled_ca_coords is not None and geometry_distance_weight != 0.0:
        distances = torch.cdist(recycled_ca_coords, recycled_ca_coords)
        work_score = work_score - float(geometry_distance_weight) * distances.to(dtype)

    valid_pair = torch.ones((b, l, l), device=device, dtype=torch.bool)
    if seq_mask is not None:
        valid_res = seq_mask > 0
        valid_pair = valid_pair & valid_res[:, :, None] & valid_res[:, None, :]
    if pair_mask is not None:
        valid_pair = valid_pair & (pair_mask > 0)
    valid_pair = valid_pair & (~torch.eye(l, device=device, dtype=torch.bool)[None, :, :])

    very_negative = torch.finfo(dtype).min / 4
    work_score = work_score.masked_fill(~valid_pair, very_negative)

    self_mask = torch.eye(l, device=device, dtype=torch.bool)[None, :, :]
    work_score = work_score.masked_fill(self_mask, very_negative * 2)

    local_k = min(max(int(local_neighbor_k), 0), k)
    if k == 0:
        nbr_idx = torch.empty((b, l, 0), device=device, dtype=torch.long)
    elif local_k > 0:
        local_score = (-sep).to(dtype)[None, :, :].expand(b, -1, -1).clone()
        local_score = local_score.masked_fill(~valid_pair, very_negative)
        local_score = local_score.masked_fill(self_mask, very_negative * 2)
        local_idx = torch.topk(local_score, k=local_k, dim=-1).indices
        global_k = k - local_k
        if global_k == 0:
            nbr_idx = local_idx
        else:
            global_score = work_score.clone()
            global_score.scatter_(dim=-1, index=local_idx, value=very_negative)
            global_idx = torch.topk(global_score, k=global_k, dim=-1).indices
            nbr_idx = torch.cat([local_idx, global_idx], dim=-1)
    else:
        nbr_idx = torch.topk(work_score, k=k, dim=-1).indices

    face_combos = _combination_tensor(k, 2, device)
    tetra_combos = _combination_tensor(k, 3, device)
    tetra_face_slots = _tetra_face_slots(face_combos, tetra_combos)

    anchor = residue_ids[None, :, None].expand(b, l, -1)
    closure_weight = min(max(float(boundary_closure_weight), 0.0), 1.0)
    closure_temperature = max(float(boundary_closure_temperature), 1e-6)

    if face_combos.numel() == 0:
        face_indices = torch.empty((b, l, 0, 3), device=device, dtype=torch.long)
        face_mask = torch.empty((b, l, 0), device=device, dtype=dtype)
    else:
        face_j = nbr_idx[:, :, face_combos[:, 0]]
        face_k = nbr_idx[:, :, face_combos[:, 1]]
        face_i = anchor.expand(b, l, face_combos.shape[0])
        face_indices = torch.stack([face_i, face_j, face_k], dim=-1)
        ij = gather_pair(valid_pair.to(dtype), face_i, face_j)
        ik = gather_pair(valid_pair.to(dtype), face_i, face_k)
        jk = gather_pair(valid_pair.to(dtype), face_j, face_k)
        face_mask = ij * ik * jk
        face_edge_scores = torch.stack(
            [
                gather_pair(work_score, face_i, face_j),
                gather_pair(work_score, face_i, face_k),
                gather_pair(work_score, face_j, face_k),
            ],
            dim=-1,
        )
        face_cell_score = face_edge_scores.mean(dim=-1)
        if closure_weight > 0.0:
            face_edge_probs = torch.sigmoid(face_edge_scores / closure_temperature).clamp_min(1e-6)
            face_closure = face_edge_probs.log().mean(dim=-1).exp()
            face_mask = face_mask * ((1.0 - closure_weight) + closure_weight * face_closure)
        if cell_score_degree_penalty > 0.0 and face_top_k > 0:
            face_edges = torch.stack(
                [
                    torch.stack([face_i, face_j], dim=-1),
                    torch.stack([face_i, face_k], dim=-1),
                    torch.stack([face_j, face_k], dim=-1),
                ],
                dim=-2,
            )
            face_cell_score = face_cell_score - float(cell_score_degree_penalty) * _cell_boundary_degree_penalty(
                face_edges,
                face_mask,
                num_residues=l,
            )
        if cell_score_outer_edge_weight != 0.0 and face_top_k > 0:
            face_cell_score = face_cell_score + float(cell_score_outer_edge_weight) * _cell_outer_edge_support(
                face_indices,
                face_mask,
                nbr_idx,
            )
        if cell_score_segment_weight != 0.0 and face_top_k > 0:
            face_cell_score = face_cell_score + float(cell_score_segment_weight) * _cell_segment_support(
                face_indices,
                face_mask,
                radius=segment_radius,
            )
        face_mask = _topk_cell_mask(face_mask, face_cell_score, face_top_k)

    if tetra_combos.numel() == 0:
        tetra_indices = torch.empty((b, l, 0, 4), device=device, dtype=torch.long)
        tetra_mask = torch.empty((b, l, 0), device=device, dtype=dtype)
    else:
        tet_j = nbr_idx[:, :, tetra_combos[:, 0]]
        tet_k = nbr_idx[:, :, tetra_combos[:, 1]]
        tet_l = nbr_idx[:, :, tetra_combos[:, 2]]
        tet_i = anchor.expand(b, l, tetra_combos.shape[0])
        tetra_indices = torch.stack([tet_i, tet_j, tet_k, tet_l], dim=-1)
        pairs = ((tet_i, tet_j), (tet_i, tet_k), (tet_i, tet_l), (tet_j, tet_k), (tet_j, tet_l), (tet_k, tet_l))
        tetra_mask = torch.ones_like(tet_i, dtype=dtype)
        for a, c in pairs:
            tetra_mask = tetra_mask * gather_pair(valid_pair.to(dtype), a, c)
        tetra_edge_scores = torch.stack([gather_pair(work_score, a, c) for a, c in pairs], dim=-1)
        tetra_cell_score = tetra_edge_scores.mean(dim=-1)
        if closure_weight > 0.0:
            tetra_edge_probs = torch.sigmoid(tetra_edge_scores / closure_temperature).clamp_min(1e-6)
            tetra_closure = tetra_edge_probs.log().mean(dim=-1).exp()
            tetra_mask = tetra_mask * ((1.0 - closure_weight) + closure_weight * tetra_closure)
        if cell_score_degree_penalty > 0.0 and tetra_top_k > 0:
            tetra_edges = torch.stack([torch.stack([a, c], dim=-1) for a, c in pairs], dim=-2)
            tetra_cell_score = tetra_cell_score - float(cell_score_degree_penalty) * _cell_boundary_degree_penalty(
                tetra_edges,
                tetra_mask,
                num_residues=l,
            )
        if cell_score_outer_edge_weight != 0.0 and tetra_top_k > 0:
            tetra_cell_score = tetra_cell_score + float(cell_score_outer_edge_weight) * _cell_outer_edge_support(
                tetra_indices,
                tetra_mask,
                nbr_idx,
            )
        if cell_score_segment_weight != 0.0 and tetra_top_k > 0:
            tetra_cell_score = tetra_cell_score + float(cell_score_segment_weight) * _cell_segment_support(
                tetra_indices,
                tetra_mask,
                radius=segment_radius,
            )
        tetra_mask = _topk_cell_mask(tetra_mask, tetra_cell_score, tetra_top_k)

    return SimplexTopology(
        nbr_idx=nbr_idx,
        face_indices=face_indices,
        tetra_indices=tetra_indices,
        face_mask=face_mask,
        tetra_mask=tetra_mask,
        face_neighbor_slots=face_combos,
        tetra_neighbor_slots=tetra_combos,
        tetra_face_slots=tetra_face_slots,
    )


def face_geometry_features(
    face_indices: torch.Tensor,
    *,
    recycled_ca_coords: Optional[torch.Tensor],
    recycled_frames: Optional[torch.Tensor],
    rbf_bins: int,
    sequence_max: float,
    distance_max: float,
    area_max: float,
) -> torch.Tensor:
    """Build invariant face features for ``(i, j, k)`` triples."""

    dtype = torch.float32
    if recycled_ca_coords is not None:
        dtype = recycled_ca_coords.dtype
    device = face_indices.device
    i, j, k = face_indices.unbind(dim=-1)
    seq_sep = torch.stack(
        [
            torch.abs(i - j),
            torch.abs(i - k),
            torch.abs(j - k),
        ],
        dim=-1,
    ).to(dtype)
    seq_feat = rbf(seq_sep, n_bins=rbf_bins, max_value=sequence_max).flatten(-2)

    coord_dim = 3 * rbf_bins + 7
    if recycled_ca_coords is None:
        zeros = torch.zeros(*face_indices.shape[:-1], coord_dim, device=device, dtype=dtype)
        return torch.cat([seq_feat, zeros], dim=-1)

    x_i = gather_single(recycled_ca_coords, i)
    x_j = gather_single(recycled_ca_coords, j)
    x_k = gather_single(recycled_ca_coords, k)
    d_ij = _safe_norm(x_j - x_i)
    d_ik = _safe_norm(x_k - x_i)
    d_jk = _safe_norm(x_k - x_j)
    dist_feat = rbf(torch.stack([d_ij, d_ik, d_jk], dim=-1), n_bins=rbf_bins, max_value=distance_max).flatten(-2)

    cross = torch.cross(x_j - x_i, x_k - x_i, dim=-1)
    area = 0.5 * _safe_norm(cross)
    normal = cross / _safe_norm(cross)[..., None].clamp_min(1e-8)
    if recycled_frames is not None:
        r_i = gather_single(recycled_frames, i)
        normal = torch.einsum("...ij,...j->...i", r_i.transpose(-1, -2), normal)

    angles = torch.stack(
        [
            _cosine(x_j - x_i, x_k - x_i),
            _cosine(x_i - x_j, x_k - x_j),
            _cosine(x_i - x_k, x_j - x_k),
        ],
        dim=-1,
    )
    area_scalar = torch.log1p(area)[..., None] / torch.log1p(
        torch.as_tensor(area_max, device=device, dtype=dtype)
    ).clamp_min(1.0)
    return torch.cat([seq_feat, dist_feat, area_scalar, angles, normal], dim=-1)


def tetra_geometry_features(
    tetra_indices: torch.Tensor,
    *,
    recycled_ca_coords: Optional[torch.Tensor],
    rbf_bins: int,
    sequence_max: float,
    distance_max: float,
    area_max: float,
    volume_scale: float,
) -> torch.Tensor:
    """Build invariant tetrahedral features for ``(i, j, k, l)`` quadruples."""

    dtype = torch.float32
    if recycled_ca_coords is not None:
        dtype = recycled_ca_coords.dtype
    device = tetra_indices.device
    i, j, k, l = tetra_indices.unbind(dim=-1)
    seq_pairs = torch.stack(
        [
            torch.abs(i - j),
            torch.abs(i - k),
            torch.abs(i - l),
            torch.abs(j - k),
            torch.abs(j - l),
            torch.abs(k - l),
        ],
        dim=-1,
    ).to(dtype)
    seq_feat = rbf(seq_pairs, n_bins=rbf_bins, max_value=sequence_max).flatten(-2)

    coord_dim = 11 * rbf_bins + 2
    if recycled_ca_coords is None:
        zeros = torch.zeros(*tetra_indices.shape[:-1], coord_dim, device=device, dtype=dtype)
        return torch.cat([seq_feat, zeros], dim=-1)

    x_i = gather_single(recycled_ca_coords, i)
    x_j = gather_single(recycled_ca_coords, j)
    x_k = gather_single(recycled_ca_coords, k)
    x_l = gather_single(recycled_ca_coords, l)
    distances = torch.stack(
        [
            _safe_norm(x_j - x_i),
            _safe_norm(x_k - x_i),
            _safe_norm(x_l - x_i),
            _safe_norm(x_k - x_j),
            _safe_norm(x_l - x_j),
            _safe_norm(x_l - x_k),
        ],
        dim=-1,
    )
    dist_feat = rbf(distances, n_bins=rbf_bins, max_value=distance_max).flatten(-2)

    areas = torch.stack(
        [
            _triangle_area(x_i, x_j, x_k),
            _triangle_area(x_i, x_j, x_l),
            _triangle_area(x_i, x_k, x_l),
            _triangle_area(x_j, x_k, x_l),
        ],
        dim=-1,
    )
    area_feat = rbf(areas, n_bins=rbf_bins, max_value=area_max).flatten(-2)

    signed_volume = torch.sum(torch.cross(x_j - x_i, x_k - x_i, dim=-1) * (x_l - x_i), dim=-1) / 6.0
    points = torch.stack([x_i, x_j, x_k, x_l], dim=-2)
    center = points.mean(dim=-2, keepdim=True)
    rg = torch.sqrt(torch.mean(torch.sum((points - center) ** 2, dim=-1), dim=-1).clamp_min(1e-8))
    rg_feat = rbf(rg, n_bins=rbf_bins, max_value=distance_max)

    volume_denom = torch.log1p(torch.as_tensor(volume_scale, device=device, dtype=dtype)).clamp_min(1.0)
    signed_log_volume = torch.sign(signed_volume) * torch.log1p(torch.abs(signed_volume)) / volume_denom
    abs_log_volume = torch.log1p(torch.abs(signed_volume)) / volume_denom
    volume_feat = torch.stack([signed_log_volume, abs_log_volume], dim=-1)

    return torch.cat([seq_feat, dist_feat, area_feat, rg_feat, volume_feat], dim=-1)


def scatter_to_pair(
    updates: torch.Tensor,
    edge_indices: torch.Tensor,
    *,
    pair_shape: tuple[int, int, int, int],
    edge_mask: Optional[torch.Tensor] = None,
    include_reverse: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter local edge updates into a dense pair tensor."""

    b, l, _, c_z = pair_shape
    if updates.numel() == 0:
        empty_delta = updates.new_zeros((b, l, l, c_z))
        empty_counts = updates.new_zeros((b, l, l, 1))
        return empty_delta, empty_counts

    if edge_mask is not None:
        updates = updates * edge_mask[..., None]
    a = edge_indices[..., 0]
    c = edge_indices[..., 1]
    ids = a * l + c
    if include_reverse:
        reverse_ids = c * l + a
        ids = torch.cat([ids, reverse_ids], dim=-1)
        updates = torch.cat([updates, updates], dim=-2)
        if edge_mask is not None:
            edge_mask = torch.cat([edge_mask, edge_mask], dim=-1)

    flat_updates = updates.reshape(b, -1, c_z)
    flat_ids = ids.reshape(b, -1)
    delta = updates.new_zeros((b, l * l, c_z))
    delta.scatter_add_(1, flat_ids[..., None].expand(-1, -1, c_z), flat_updates)

    if edge_mask is None:
        flat_counts_src = updates.new_ones((b, flat_ids.shape[1], 1))
    else:
        flat_counts_src = edge_mask.reshape(b, -1, 1).to(updates.dtype)
    counts = updates.new_zeros((b, l * l, 1))
    counts.scatter_add_(1, flat_ids[..., None], flat_counts_src)
    return delta.reshape(b, l, l, c_z), counts.reshape(b, l, l, 1)


def _distance_logits_to_recycling_bins(
    logits: torch.Tensor,
    *,
    n_recycle_bins: int,
    dist_min: float = 2.0,
    dist_max: float = 22.0,
    recycle_min: float = 3.375,
    recycle_bin_width: float = 1.25,
) -> torch.Tensor:
    """Map selected-cell distance logits to AF2-style recycling-bin evidence."""

    if logits.numel() == 0:
        return logits.new_empty(*logits.shape[:-1], n_recycle_bins)
    if n_recycle_bins <= 0:
        raise ValueError("n_recycle_bins must be positive.")
    n_dist_bins = int(logits.shape[-1])
    dist_centers = torch.linspace(
        float(dist_min),
        float(dist_max),
        n_dist_bins,
        device=logits.device,
        dtype=logits.dtype,
    )
    probs = torch.softmax(logits, dim=-1)
    recycle_pos = ((dist_centers - float(recycle_min)) / float(recycle_bin_width)).clamp(
        min=0.0,
        max=float(n_recycle_bins - 1),
    )
    lower_idx = torch.floor(recycle_pos).to(torch.long)
    upper_idx = torch.ceil(recycle_pos).to(torch.long)
    upper_weight = recycle_pos - lower_idx.to(logits.dtype)
    lower_weight = 1.0 - upper_weight
    dist_to_recycle = (
        torch.nn.functional.one_hot(lower_idx, n_recycle_bins).to(dtype=logits.dtype) * lower_weight[:, None]
        + torch.nn.functional.one_hot(upper_idx, n_recycle_bins).to(dtype=logits.dtype) * upper_weight[:, None]
    )
    return torch.matmul(probs, dist_to_recycle)


def simplex_boundary_metric_recycling_bins(
    aux: dict[str, torch.Tensor],
    *,
    num_residues: int,
    n_recycle_bins: int = 15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter selected face/tetra boundary metric cochains into recycling bins."""

    face_indices = aux.get("simplex_face_indices")
    if face_indices is None:
        raise KeyError("simplex_face_indices")
    b = int(face_indices.shape[0])
    device = face_indices.device
    face_logits = aux.get("simplex_face_distance_logits")
    tetra_logits = aux.get("simplex_tetra_distance_logits")
    dtype_source = face_logits if face_logits is not None else tetra_logits
    dtype = dtype_source.dtype if dtype_source is not None else torch.float32
    metric_sum = torch.zeros((b, num_residues, num_residues, n_recycle_bins), device=device, dtype=dtype)
    metric_counts = torch.zeros((b, num_residues, num_residues, 1), device=device, dtype=dtype)

    face_mask = aux.get("simplex_face_mask")
    if face_logits is not None and face_mask is not None and face_indices.numel() > 0:
        i, j, k = face_indices.unbind(dim=-1)
        face_edges = torch.stack(
            [
                torch.stack([i, j], dim=-1),
                torch.stack([i, k], dim=-1),
                torch.stack([j, k], dim=-1),
            ],
            dim=-2,
        )
        face_bins = _distance_logits_to_recycling_bins(face_logits, n_recycle_bins=n_recycle_bins)
        face_edge_mask = face_mask[..., None].expand_as(face_edges[..., 0]).to(dtype)
        face_delta, face_counts = scatter_to_pair(
            face_bins,
            face_edges,
            pair_shape=tuple(metric_sum.shape),  # type: ignore[arg-type]
            edge_mask=face_edge_mask,
            include_reverse=True,
        )
        metric_sum = metric_sum + face_delta
        metric_counts = metric_counts + face_counts

    tetra_indices = aux.get("simplex_tetra_indices")
    tetra_mask = aux.get("simplex_tetra_mask")
    if (
        tetra_indices is not None
        and tetra_logits is not None
        and tetra_mask is not None
        and tetra_indices.numel() > 0
    ):
        i, j, k, l = tetra_indices.unbind(dim=-1)
        tetra_edges = torch.stack(
            [
                torch.stack([i, j], dim=-1),
                torch.stack([i, k], dim=-1),
                torch.stack([i, l], dim=-1),
                torch.stack([j, k], dim=-1),
                torch.stack([j, l], dim=-1),
                torch.stack([k, l], dim=-1),
            ],
            dim=-2,
        )
        tetra_bins = _distance_logits_to_recycling_bins(tetra_logits, n_recycle_bins=n_recycle_bins)
        tetra_edge_mask = tetra_mask[..., None].expand_as(tetra_edges[..., 0]).to(dtype)
        tetra_delta, tetra_counts = scatter_to_pair(
            tetra_bins,
            tetra_edges,
            pair_shape=tuple(metric_sum.shape),  # type: ignore[arg-type]
            edge_mask=tetra_edge_mask,
            include_reverse=True,
        )
        metric_sum = metric_sum + tetra_delta
        metric_counts = metric_counts + tetra_counts

    metric_mask = (metric_counts > 0).to(dtype)
    metric_bins = metric_sum / metric_counts.clamp_min(1.0)
    return metric_bins * metric_mask, metric_mask


def simplex_boundary_metric_confidence_map(
    aux: dict[str, torch.Tensor],
    *,
    num_residues: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter selected face/tetra metric confidence onto boundary edges."""

    face_indices = aux.get("simplex_face_indices")
    if face_indices is None:
        raise KeyError("simplex_face_indices")
    batch_size = int(face_indices.shape[0])
    device = face_indices.device
    face_logits = aux.get("simplex_face_distance_logits")
    tetra_logits = aux.get("simplex_tetra_distance_logits")
    dtype_source = face_logits if face_logits is not None else tetra_logits
    dtype = dtype_source.dtype if dtype_source is not None else torch.float32
    confidence_sum = torch.zeros((batch_size, num_residues, num_residues, 1), device=device, dtype=dtype)
    confidence_counts = torch.zeros_like(confidence_sum)

    face_mask = aux.get("simplex_face_mask")
    if face_logits is not None and face_mask is not None and face_indices.numel() > 0:
        i, j, k = face_indices.unbind(dim=-1)
        face_edges = torch.stack(
            [
                torch.stack([i, j], dim=-1),
                torch.stack([i, k], dim=-1),
                torch.stack([j, k], dim=-1),
            ],
            dim=-2,
        )
        face_confidence = boundary_metric_confidence(face_logits)[..., None]
        face_edge_mask = face_mask[..., None].expand_as(face_edges[..., 0]).to(dtype)
        face_delta, face_counts = scatter_to_pair(
            face_confidence,
            face_edges,
            pair_shape=tuple(confidence_sum.shape),  # type: ignore[arg-type]
            edge_mask=face_edge_mask,
            include_reverse=True,
        )
        confidence_sum = confidence_sum + face_delta
        confidence_counts = confidence_counts + face_counts

    tetra_indices = aux.get("simplex_tetra_indices")
    tetra_mask = aux.get("simplex_tetra_mask")
    if (
        tetra_indices is not None
        and tetra_logits is not None
        and tetra_mask is not None
        and tetra_indices.numel() > 0
    ):
        i, j, k, tet_l = tetra_indices.unbind(dim=-1)
        tetra_edges = torch.stack(
            [
                torch.stack([i, j], dim=-1),
                torch.stack([i, k], dim=-1),
                torch.stack([i, tet_l], dim=-1),
                torch.stack([j, k], dim=-1),
                torch.stack([j, tet_l], dim=-1),
                torch.stack([k, tet_l], dim=-1),
            ],
            dim=-2,
        )
        tetra_confidence = boundary_metric_confidence(tetra_logits)[..., None]
        tetra_edge_mask = tetra_mask[..., None].expand_as(tetra_edges[..., 0]).to(dtype)
        tetra_delta, tetra_counts = scatter_to_pair(
            tetra_confidence,
            tetra_edges,
            pair_shape=tuple(confidence_sum.shape),  # type: ignore[arg-type]
            edge_mask=tetra_edge_mask,
            include_reverse=True,
        )
        confidence_sum = confidence_sum + tetra_delta
        confidence_counts = confidence_counts + tetra_counts

    confidence_mask = (confidence_counts > 0).to(dtype)
    confidence = confidence_sum / confidence_counts.clamp_min(1.0)
    return confidence * confidence_mask, confidence_mask


def coface_degree_attenuate_pair_readout(
    pair_readout: torch.Tensor,
    pair_counts: torch.Tensor,
    attenuation: float,
) -> torch.Tensor:
    """Dampen pair updates on boundary edges reused by many selected cells."""

    strength = max(float(attenuation), 0.0)
    if strength == 0.0:
        return pair_readout
    degree = pair_counts.to(pair_readout.dtype).clamp_min(1.0)
    return pair_readout / degree.pow(strength)


def hodge_center_boundary_readout(
    pair_readout: torch.Tensor,
    pair_counts: torch.Tensor,
    *,
    pair_mask: torch.Tensor | None = None,
    scale: float,
) -> torch.Tensor:
    """Double-center a selected boundary 1-cochain over residue vertex stars."""

    scale = min(max(float(scale), 0.0), 1.0)
    if scale <= 0.0:
        return pair_readout
    edge_mask = (pair_counts > 0).to(pair_readout.dtype)
    if pair_mask is not None:
        edge_mask = edge_mask * pair_mask[..., None].to(pair_readout.dtype)

    row_count = edge_mask.sum(dim=2).clamp_min(1.0)
    col_count = edge_mask.sum(dim=1).clamp_min(1.0)
    global_count = edge_mask.sum(dim=(1, 2)).clamp_min(1.0)
    row_mean = (pair_readout * edge_mask).sum(dim=2) / row_count
    col_mean = (pair_readout * edge_mask).sum(dim=1) / col_count
    global_mean = (pair_readout * edge_mask).sum(dim=(1, 2)) / global_count
    centered = pair_readout - row_mean[:, :, None, :] - col_mean[:, None, :, :] + global_mean[:, None, None, :]
    centered = centered * edge_mask
    return pair_readout + pair_readout.new_tensor(scale) * (centered - pair_readout)


def edge_star_smooth_boundary_readout(
    pair_readout: torch.Tensor,
    pair_counts: torch.Tensor,
    *,
    pair_mask: torch.Tensor | None = None,
    scale: float,
) -> torch.Tensor:
    """Diffuse a selected boundary 1-cochain through residue edge-stars."""

    scale = min(max(float(scale), 0.0), 1.0)
    if scale <= 0.0:
        return pair_readout
    edge_mask = (pair_counts > 0).to(pair_readout.dtype)
    if pair_mask is not None:
        edge_mask = edge_mask * pair_mask[..., None].to(pair_readout.dtype)

    source_count = edge_mask.sum(dim=2).clamp_min(1.0)
    target_count = edge_mask.sum(dim=1).clamp_min(1.0)
    source_mean = (pair_readout * edge_mask).sum(dim=2) / source_count
    target_mean = (pair_readout * edge_mask).sum(dim=1) / target_count
    smoothed = 0.5 * (source_mean[:, :, None, :] + target_mean[:, None, :, :])
    edge_masked_delta = edge_mask * (smoothed - pair_readout)
    return pair_readout + pair_readout.new_tensor(scale) * edge_masked_delta


def edge_star_residual_boundary_readout(
    pair_readout: torch.Tensor,
    pair_counts: torch.Tensor,
    *,
    pair_mask: torch.Tensor | None = None,
    scale: float,
) -> torch.Tensor:
    """Project a selected boundary 1-cochain away from edge-star means."""

    scale = min(max(float(scale), 0.0), 1.0)
    if scale <= 0.0:
        return pair_readout
    edge_mask = (pair_counts > 0).to(pair_readout.dtype)
    if pair_mask is not None:
        edge_mask = edge_mask * pair_mask[..., None].to(pair_readout.dtype)

    source_count = edge_mask.sum(dim=2).clamp_min(1.0)
    target_count = edge_mask.sum(dim=1).clamp_min(1.0)
    source_mean = (pair_readout * edge_mask).sum(dim=2) / source_count
    target_mean = (pair_readout * edge_mask).sum(dim=1) / target_count
    lower_star_mean = 0.5 * (source_mean[:, :, None, :] + target_mean[:, None, :, :])
    residual = edge_mask * (pair_readout - lower_star_mean)
    edge_masked_delta = edge_mask * (residual - pair_readout)
    return pair_readout + pair_readout.new_tensor(scale) * edge_masked_delta


def oriented_boundary_cochain_readout(
    pair_readout: torch.Tensor,
    pair_counts: torch.Tensor,
    *,
    pair_mask: torch.Tensor | None = None,
    scale: float,
) -> torch.Tensor:
    """Blend a selected boundary 1-cochain toward its oriented difference."""

    scale = min(max(float(scale), 0.0), 1.0)
    if scale <= 0.0:
        return pair_readout
    edge_mask = (pair_counts > 0).to(pair_readout.dtype)
    if pair_mask is not None:
        edge_mask = edge_mask * pair_mask[..., None].to(pair_readout.dtype)

    reverse_mask = edge_mask.transpose(1, 2)
    reverse_readout = pair_readout.transpose(1, 2)
    oriented = edge_mask * (pair_readout - reverse_mask * reverse_readout)
    edge_masked_delta = edge_mask * (oriented - pair_readout)
    return pair_readout + pair_readout.new_tensor(scale) * edge_masked_delta


def cyclic_face_boundary_edges(face_indices: torch.Tensor) -> torch.Tensor:
    """Return the oriented boundary cycle edges of selected triangular faces."""

    i, j, k = face_indices.unbind(dim=-1)
    return torch.stack(
        [
            torch.stack([i, j], dim=-1),
            torch.stack([j, k], dim=-1),
            torch.stack([k, i], dim=-1),
        ],
        dim=-2,
    )


def cyclic_face_boundary_updates(face_edge_update: torch.Tensor) -> torch.Tensor:
    """Reorder face-edge updates from `(ij, ik, jk)` slots to `(ij, jk, ki)`."""

    return torch.stack(
        [
            face_edge_update[..., 0, :],
            face_edge_update[..., 2, :],
            face_edge_update[..., 1, :],
        ],
        dim=-2,
    )


def signed_cyclic_face_boundary_updates(face_edge_update: torch.Tensor) -> torch.Tensor:
    """Apply the signed oriented boundary of a face to edge updates."""

    return torch.stack(
        [
            face_edge_update[..., 0, :],
            face_edge_update[..., 2, :],
            -face_edge_update[..., 1, :],
        ],
        dim=-2,
    )


def boundary_incidence_weights(
    edge_indices: torch.Tensor,
    edge_mask: torch.Tensor,
    *,
    num_residues: int,
    strength: float,
) -> torch.Tensor:
    """Weight each selected cell-edge incidence by inverse boundary degree."""

    power = max(float(strength), 0.0)
    if power == 0.0 or edge_mask.numel() == 0:
        return torch.ones_like(edge_mask)
    a, c = edge_indices.unbind(dim=-1)
    lo = torch.minimum(a, c)
    hi = torch.maximum(a, c)
    edge_ids = lo * int(num_residues) + hi
    batch = edge_indices.shape[0]
    flat_ids = edge_ids.reshape(batch, -1)
    active = (edge_mask > 0).to(edge_mask.dtype)
    flat_mask = active.reshape(batch, -1)
    degree = edge_mask.new_zeros((batch, int(num_residues) * int(num_residues)))
    degree.scatter_add_(1, flat_ids, flat_mask)
    flat_degree = torch.gather(degree, 1, flat_ids).reshape_as(edge_mask).clamp_min(1.0)
    return active / flat_degree.pow(power)


def _cell_boundary_incidence_weights(
    edge_indices: torch.Tensor,
    edge_mask: torch.Tensor,
    *,
    num_residues: int,
    strength: float,
) -> torch.Tensor:
    edge_weights = boundary_incidence_weights(
        edge_indices,
        edge_mask,
        num_residues=num_residues,
        strength=strength,
    )
    denom = (edge_mask > 0).to(edge_weights.dtype).sum(dim=-1).clamp_min(1.0)
    return edge_weights.sum(dim=-1) / denom


def scatter_to_single(
    updates: torch.Tensor,
    residue_indices: torch.Tensor,
    *,
    single_shape: tuple[int, int, int],
    residue_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter local residue updates into a dense single representation."""

    b, l, c_s = single_shape
    if updates.numel() == 0:
        return updates.new_zeros((b, l, c_s)), updates.new_zeros((b, l, 1))
    if residue_mask is not None:
        updates = updates * residue_mask[..., None]
    flat_updates = updates.reshape(b, -1, c_s)
    flat_ids = residue_indices.reshape(b, -1)
    delta = updates.new_zeros((b, l, c_s))
    delta.scatter_add_(1, flat_ids[..., None].expand(-1, -1, c_s), flat_updates)

    if residue_mask is None:
        count_src = updates.new_ones((b, flat_ids.shape[1], 1))
    else:
        count_src = residue_mask.reshape(b, -1, 1).to(updates.dtype)
    counts = updates.new_zeros((b, l, 1))
    counts.scatter_add_(1, flat_ids[..., None], count_src)
    return delta, counts


def scatter_directed_edges_to_residue(
    updates: torch.Tensor,
    edge_indices: torch.Tensor,
    *,
    num_residues: int,
    edge_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scatter directed boundary-edge messages into source and target residues."""

    c_edge = updates.shape[-1]
    source_delta, source_counts = scatter_to_single(
        updates,
        edge_indices[..., 0],
        single_shape=(updates.shape[0], int(num_residues), c_edge),
        residue_mask=edge_mask,
    )
    target_delta, target_counts = scatter_to_single(
        updates,
        edge_indices[..., 1],
        single_shape=(updates.shape[0], int(num_residues), c_edge),
        residue_mask=edge_mask,
    )
    return source_delta, source_counts, target_delta, target_counts


def face_outer_edge_delta(
    face_state: torch.Tensor,
    face_indices: torch.Tensor,
    face_mask: torch.Tensor,
    *,
    num_residues: int,
) -> torch.Tensor:
    """Average neighboring face states through shared undirected boundary edges."""

    if face_state.numel() == 0:
        return torch.zeros_like(face_state)

    b, _, _, c_face = face_state.shape
    i, j, k = face_indices.unbind(dim=-1)
    face_edges = torch.stack(
        [
            torch.stack([i, j], dim=-1),
            torch.stack([i, k], dim=-1),
            torch.stack([j, k], dim=-1),
        ],
        dim=-2,
    )
    edge_lo = torch.minimum(face_edges[..., 0], face_edges[..., 1])
    edge_hi = torch.maximum(face_edges[..., 0], face_edges[..., 1])
    edge_ids = edge_lo * num_residues + edge_hi

    edge_mask = face_mask[..., None].to(face_state.dtype).expand_as(edge_ids)
    edge_values = face_state[..., None, :].expand(*edge_ids.shape, c_face) * edge_mask[..., None]
    flat_ids = edge_ids.reshape(b, -1)
    flat_values = edge_values.reshape(b, -1, c_face)

    edge_sum = face_state.new_zeros((b, num_residues * num_residues, c_face))
    edge_sum.scatter_add_(1, flat_ids[..., None].expand(-1, -1, c_face), flat_values)
    edge_count = face_state.new_zeros((b, num_residues * num_residues, 1))
    edge_count.scatter_add_(1, flat_ids[..., None], edge_mask.reshape(b, -1, 1))

    gathered_sum = edge_sum.gather(1, flat_ids[..., None].expand(-1, -1, c_face)).reshape(*edge_ids.shape, c_face)
    gathered_count = edge_count.gather(1, flat_ids[..., None]).reshape(*edge_ids.shape, 1)
    self_values = face_state[..., None, :] * edge_mask[..., None]
    outer_sum = gathered_sum - self_values
    outer_count = (gathered_count - edge_mask[..., None]).clamp_min(0.0)
    has_outer = (outer_count > 0).to(face_state.dtype)
    outer_mean_by_edge = outer_sum / outer_count.clamp_min(1.0)
    face_outer_mean = (outer_mean_by_edge * has_outer).sum(dim=-2) / has_outer.sum(dim=-2).clamp_min(1.0)
    has_neighbor = (has_outer.sum(dim=-2) > 0).to(face_state.dtype)
    return (face_outer_mean - face_state) * has_neighbor * face_mask[..., None]


def face_tetra_coboundary_delta(
    face_state: torch.Tensor,
    face_mask: torch.Tensor,
    tetra_face_slots: torch.Tensor,
    tetra_mask: torch.Tensor,
) -> torch.Tensor:
    """Average sibling face states through selected tetra cofaces."""

    if face_state.numel() == 0 or tetra_face_slots.numel() == 0 or tetra_mask.numel() == 0:
        return torch.zeros_like(face_state)

    b, l, _, c_face = face_state.shape
    num_tetra = tetra_face_slots.shape[0]
    face_slot_ids = tetra_face_slots.reshape(1, 1, num_tetra, 3).expand(b, l, -1, -1)
    flat_face_slots = face_slot_ids.reshape(b, l, -1)
    gathered_face = face_state.gather(
        2,
        flat_face_slots[..., None].expand(-1, -1, -1, c_face),
    ).reshape(b, l, num_tetra, 3, c_face)
    gathered_face_mask = face_mask.gather(2, flat_face_slots).reshape(b, l, num_tetra, 3)
    incident_mask = gathered_face_mask.to(face_state.dtype) * tetra_mask[..., None].to(face_state.dtype)

    tetra_face_sum = (gathered_face * incident_mask[..., None]).sum(dim=-2, keepdim=True)
    tetra_face_count = incident_mask.sum(dim=-1, keepdim=True)
    sibling_count = (tetra_face_count - incident_mask).clamp_min(0.0)
    sibling_sum = tetra_face_sum - gathered_face * incident_mask[..., None]
    sibling_mean = sibling_sum / sibling_count[..., None].clamp_min(1.0)
    has_sibling = (sibling_count > 0).to(face_state.dtype) * incident_mask
    sibling_update = (sibling_mean - gathered_face) * has_sibling[..., None]

    flat_updates = sibling_update.reshape(b, l, -1, c_face)
    flat_counts = has_sibling.reshape(b, l, -1, 1)
    delta = torch.zeros_like(face_state)
    counts = face_state.new_zeros(*face_state.shape[:-1], 1)
    delta.scatter_add_(2, flat_face_slots[..., None].expand(-1, -1, -1, c_face), flat_updates)
    counts.scatter_add_(2, flat_face_slots[..., None], flat_counts)
    return (delta / counts.clamp_min(1.0)) * face_mask[..., None]


def signed_face_tetra_coboundary_delta(
    face_state: torch.Tensor,
    face_mask: torch.Tensor,
    tetra_face_slots: torch.Tensor,
    tetra_mask: torch.Tensor,
) -> torch.Tensor:
    """Average sibling face states using oriented tetra boundary signs."""

    if face_state.numel() == 0 or tetra_face_slots.numel() == 0 or tetra_mask.numel() == 0:
        return torch.zeros_like(face_state)

    b, l, _, c_face = face_state.shape
    num_tetra = tetra_face_slots.shape[0]
    face_slot_ids = tetra_face_slots.reshape(1, 1, num_tetra, 3).expand(b, l, -1, -1)
    flat_face_slots = face_slot_ids.reshape(b, l, -1)
    gathered_face = face_state.gather(
        2,
        flat_face_slots[..., None].expand(-1, -1, -1, c_face),
    ).reshape(b, l, num_tetra, 3, c_face)
    gathered_face_mask = face_mask.gather(2, flat_face_slots).reshape(b, l, num_tetra, 3)
    incident_mask = gathered_face_mask.to(face_state.dtype) * tetra_mask[..., None].to(face_state.dtype)

    signs = face_state.new_tensor([-1.0, 1.0, -1.0]).reshape(1, 1, 1, 3)
    signed_face = gathered_face * signs[..., None]
    signed_tetra_sum = (signed_face * incident_mask[..., None]).sum(dim=-2, keepdim=True)
    signed_sibling_sum = signed_tetra_sum - signed_face * incident_mask[..., None]
    sibling_count = (incident_mask.sum(dim=-1, keepdim=True) - incident_mask).clamp_min(0.0)
    aligned_sibling_mean = signs[..., None] * signed_sibling_sum / sibling_count[..., None].clamp_min(1.0)
    has_sibling = (sibling_count > 0).to(face_state.dtype) * incident_mask
    sibling_update = (aligned_sibling_mean - gathered_face) * has_sibling[..., None]

    flat_updates = sibling_update.reshape(b, l, -1, c_face)
    flat_counts = has_sibling.reshape(b, l, -1, 1)
    delta = torch.zeros_like(face_state)
    counts = face_state.new_zeros(*face_state.shape[:-1], 1)
    delta.scatter_add_(2, flat_face_slots[..., None].expand(-1, -1, -1, c_face), flat_updates)
    counts.scatter_add_(2, flat_face_slots[..., None], flat_counts)
    return (delta / counts.clamp_min(1.0)) * face_mask[..., None]


def signed_tetra_face_boundary_updates(
    tetra_face_update: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Blend learned tetra-to-face messages toward oriented boundary signs."""

    blend = min(max(float(scale), 0.0), 1.0)
    if tetra_face_update.numel() == 0 or blend == 0.0:
        return tetra_face_update
    signs = tetra_face_update.new_tensor([-1.0, 1.0, -1.0]).reshape(
        *((1,) * (tetra_face_update.ndim - 2)),
        3,
        1,
    )
    signed_update = tetra_face_update * signs
    return tetra_face_update + tetra_face_update.new_tensor(blend) * (signed_update - tetra_face_update)


def cell_outer_edge_context(
    pair: torch.Tensor,
    cell_indices: torch.Tensor,
    cell_mask: torch.Tensor,
    nbr_idx: torch.Tensor,
) -> torch.Tensor:
    """Pool directed edges that leave or enter selected higher-rank cells."""

    if cell_indices.numel() == 0:
        return pair.new_empty(*cell_indices.shape[:-1], 2 * pair.shape[-1])

    vertices = cell_indices
    outer_nbrs = gather_single(nbr_idx, vertices)
    if outer_nbrs.shape[-1] == 0:
        return pair.new_zeros(*cell_indices.shape[:-1], 2 * pair.shape[-1])

    src = vertices[..., None].expand_as(outer_nbrs)
    is_inner = (outer_nbrs[..., None] == vertices[..., None, None, :]).any(dim=-1)
    outer_mask = cell_mask[..., None, None].to(pair.dtype) * (~is_inner).to(pair.dtype)

    count = outer_mask.sum(dim=(-2, -1)).clamp_min(1.0)
    outgoing = (
        (gather_pair(pair, src, outer_nbrs) * outer_mask[..., None]).sum(dim=(-3, -2)) / count[..., None]
    )
    incoming = (
        (gather_pair(pair, outer_nbrs, src) * outer_mask[..., None]).sum(dim=(-3, -2)) / count[..., None]
    )
    return torch.cat([outgoing, incoming], dim=-1)


def _fold_feature_channels(source: torch.Tensor, channels: int) -> torch.Tensor:
    """Fold arbitrary feature width into a fixed channel count without parameters."""

    width = int(source.shape[-1])
    target = int(channels)
    if target <= 0:
        return source.new_empty(*source.shape[:-1], 0)
    if width <= 0:
        return source.new_zeros(*source.shape[:-1], target)
    if width == target:
        return source
    chunks = (width + target - 1) // target
    padded_width = chunks * target
    if padded_width != width:
        source = torch.cat([source, source.new_zeros(*source.shape[:-1], padded_width - width)], dim=-1)
    folded = source.reshape(*source.shape[:-1], chunks, target).sum(dim=-2)
    offsets = torch.arange(target, device=source.device)
    counts = ((width - 1 - offsets).div(target, rounding_mode="floor") + 1).clamp_min(0)
    counts = counts.to(dtype=source.dtype).reshape(*([1] * (source.ndim - 1)), target)
    return folded / counts.clamp_min(1.0)


def _rms_match_update(update: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    update_rms = torch.sqrt(update.float().pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-6)).to(update.dtype)
    reference_rms = torch.sqrt(
        reference.detach().float().pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-6)
    ).to(update.dtype)
    return update / update_rms * reference_rms


def parameter_free_outer_edge_context_delta(
    pair: torch.Tensor,
    cell_indices: torch.Tensor,
    cell_mask: torch.Tensor,
    nbr_idx: torch.Tensor,
    *,
    channels: int,
) -> torch.Tensor:
    """Project directed external edge neighborhoods into cell cochains without parameters."""

    if cell_indices.numel() == 0:
        return pair.new_empty(*cell_indices.shape[:-1], int(channels))
    context = cell_outer_edge_context(pair, cell_indices, cell_mask, nbr_idx)
    if context.shape[-1] == 0:
        return pair.new_zeros(*cell_indices.shape[:-1], int(channels))
    outgoing, incoming = context.split(pair.shape[-1], dim=-1)
    symmetric = 0.5 * (outgoing + incoming)
    oriented = outgoing - incoming
    folded = _fold_feature_channels(torch.cat([symmetric, oriented], dim=-1), int(channels))
    return folded * cell_mask[..., None].to(folded.dtype)


def masked_cell_mean(state: torch.Tensor, mask: torch.Tensor, channels: int) -> torch.Tensor:
    """Pool selected cell cochains into one protein-level cochain per batch item."""

    if state.numel() == 0:
        return state.new_zeros((state.shape[0], int(channels)))
    weights = mask.to(state.dtype)[..., None]
    total = (state * weights).sum(dim=(1, 2))
    count = weights.sum(dim=(1, 2)).clamp_min(1.0)
    return total / count


def vertex_star_cell_mean(
    state: torch.Tensor,
    cell_indices: torch.Tensor,
    cell_mask: torch.Tensor,
    *,
    num_residues: int,
    channels: int,
) -> torch.Tensor:
    """Pool selected cell cochains into residue vertex-star cochains."""

    if state.numel() == 0 or cell_indices.numel() == 0:
        return state.new_zeros((state.shape[0], int(num_residues), int(channels)))
    batch_size = int(cell_indices.shape[0])
    cell_rank = int(cell_indices.shape[-1])
    cell_weights = cell_mask.to(state.dtype)
    vertex_values = state[..., None, :].expand(*state.shape[:-1], cell_rank, int(channels))
    vertex_values = vertex_values * cell_weights[..., None, None]
    flat_ids = cell_indices.reshape(batch_size, -1)
    flat_values = vertex_values.reshape(batch_size, -1, int(channels))
    flat_weights = cell_weights[..., None].expand_as(cell_indices).reshape(batch_size, -1, 1)

    star_sum = state.new_zeros((batch_size, int(num_residues), int(channels)))
    star_sum.scatter_add_(1, flat_ids[..., None].expand(-1, -1, int(channels)), flat_values)
    star_counts = state.new_zeros((batch_size, int(num_residues), 1))
    star_counts.scatter_add_(1, flat_ids[..., None], flat_weights)
    return star_sum / star_counts.clamp_min(1.0)


def _cell_boundary_edges(cell_indices: torch.Tensor) -> torch.Tensor:
    """Return all unordered boundary edges for each selected cell."""

    cell_rank = int(cell_indices.shape[-1])
    edge_pairs = list(combinations(range(cell_rank), 2))
    return torch.stack(
        [
            torch.stack([cell_indices[..., source], cell_indices[..., target]], dim=-1)
            for source, target in edge_pairs
        ],
        dim=-2,
    )


def edge_star_cell_mean(
    state: torch.Tensor,
    cell_indices: torch.Tensor,
    cell_mask: torch.Tensor,
    *,
    num_residues: int,
    channels: int,
) -> torch.Tensor:
    """Pool selected cell cochains into boundary-edge star cochains."""

    if state.numel() == 0 or cell_indices.numel() == 0:
        return state.new_zeros((state.shape[0], int(num_residues), int(num_residues), int(channels)))
    batch_size = int(cell_indices.shape[0])
    edge_indices = _cell_boundary_edges(cell_indices)
    edge_weights = cell_mask.to(state.dtype)[..., None].expand_as(edge_indices[..., 0])
    edge_values = state[..., None, :].expand(*edge_indices.shape[:-1], int(channels)) * edge_weights[..., None]
    lo = torch.minimum(edge_indices[..., 0], edge_indices[..., 1])
    hi = torch.maximum(edge_indices[..., 0], edge_indices[..., 1])
    forward_ids = (lo * int(num_residues) + hi).reshape(batch_size, -1)
    reverse_ids = (hi * int(num_residues) + lo).reshape(batch_size, -1)
    flat_ids = torch.cat([forward_ids, reverse_ids], dim=1)
    flat_values_src = edge_values.reshape(batch_size, -1, int(channels))
    flat_values = torch.cat([flat_values_src, flat_values_src], dim=1)
    flat_weights_src = edge_weights.reshape(batch_size, -1, 1)
    flat_weights = torch.cat([flat_weights_src, flat_weights_src], dim=1)

    star_sum = state.new_zeros((batch_size, int(num_residues) * int(num_residues), int(channels)))
    star_sum.scatter_add_(1, flat_ids[..., None].expand(-1, -1, int(channels)), flat_values)
    star_counts = state.new_zeros((batch_size, int(num_residues) * int(num_residues), 1))
    star_counts.scatter_add_(1, flat_ids[..., None], flat_weights)
    star = star_sum / star_counts.clamp_min(1.0)
    return star.reshape(batch_size, int(num_residues), int(num_residues), int(channels))


def boundary_edge_star_context(
    state: torch.Tensor,
    cell_indices: torch.Tensor,
    cell_mask: torch.Tensor,
    target_edges: torch.Tensor,
    *,
    num_residues: int,
    channels: int,
) -> torch.Tensor:
    """Gather boundary-edge star means for target cell edges without dense LxL state."""

    target_shape = target_edges.shape[:-2]
    if state.numel() == 0 or cell_indices.numel() == 0 or target_edges.numel() == 0:
        return state.new_zeros(*target_shape, int(channels))

    batch_size = int(cell_indices.shape[0])
    edge_indices = _cell_boundary_edges(cell_indices)
    edge_weights = cell_mask.to(state.dtype)[..., None].expand_as(edge_indices[..., 0])
    edge_values = state[..., None, :].expand(*edge_indices.shape[:-1], int(channels)) * edge_weights[..., None]

    lo = torch.minimum(edge_indices[..., 0], edge_indices[..., 1])
    hi = torch.maximum(edge_indices[..., 0], edge_indices[..., 1])
    source_ids = lo * int(num_residues) + hi
    batch_offsets = torch.arange(batch_size, device=source_ids.device, dtype=source_ids.dtype) * (
        int(num_residues) * int(num_residues)
    )
    flat_source_ids = (source_ids.reshape(batch_size, -1) + batch_offsets[:, None]).reshape(-1)
    flat_values = edge_values.reshape(batch_size, -1, int(channels)).reshape(-1, int(channels))
    flat_weights = edge_weights.reshape(batch_size, -1, 1).reshape(-1, 1)

    unique_ids, inverse = torch.unique(flat_source_ids, sorted=True, return_inverse=True)
    if unique_ids.numel() == 0:
        return state.new_zeros(*target_shape, int(channels))
    star_sum = state.new_zeros((unique_ids.shape[0], int(channels)))
    star_sum.scatter_add_(0, inverse[:, None].expand(-1, int(channels)), flat_values)
    star_counts = state.new_zeros((unique_ids.shape[0], 1))
    star_counts.scatter_add_(0, inverse[:, None], flat_weights)
    star_mean = star_sum / star_counts.clamp_min(1.0)

    target_lo = torch.minimum(target_edges[..., 0], target_edges[..., 1])
    target_hi = torch.maximum(target_edges[..., 0], target_edges[..., 1])
    target_ids = target_lo * int(num_residues) + target_hi
    offset_shape = [batch_size] + [1] * (target_ids.ndim - 1)
    flat_target_ids = (target_ids + batch_offsets.reshape(offset_shape)).reshape(-1)
    positions = torch.searchsorted(unique_ids, flat_target_ids)
    safe_positions = positions.clamp(max=int(unique_ids.shape[0]) - 1)
    matched = unique_ids[safe_positions] == flat_target_ids
    flat_context = torch.where(
        matched[:, None],
        star_mean[safe_positions],
        state.new_zeros((flat_target_ids.shape[0], int(channels))),
    )
    edge_context = flat_context.reshape(*target_edges.shape[:-1], int(channels))
    return edge_context.mean(dim=-2)


FACE_EDGE_FRAME_DIM = 10
TETRA_EDGE_FRAME_DIM = 18
SEGMENT_GEOMETRY_DIM = 4


def _fallback_axis(direction: torch.Tensor) -> torch.Tensor:
    axis = torch.argmin(torch.abs(direction), dim=-1)
    return torch.nn.functional.one_hot(axis, num_classes=3).to(dtype=direction.dtype, device=direction.device)


def _edge_frame_axes(
    edge_indices: torch.Tensor,
    recycled_ca_coords: torch.Tensor,
    recycled_frames: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    a, b = edge_indices.unbind(dim=-1)
    x_a = gather_single(recycled_ca_coords, a)
    x_b = gather_single(recycled_ca_coords, b)
    edge = x_b - x_a
    edge_len = _safe_norm(edge)
    e1 = edge / edge_len[..., None].clamp_min(1e-8)
    if recycled_frames is not None:
        frame_a = gather_single(recycled_frames, a)
        reference = frame_a[..., :, 1]
    else:
        reference = _fallback_axis(e1)
    reference_perp = reference - torch.sum(reference * e1, dim=-1, keepdim=True) * e1
    fallback = _fallback_axis(e1)
    fallback_perp = fallback - torch.sum(fallback * e1, dim=-1, keepdim=True) * e1
    use_fallback = _safe_norm(reference_perp) < 1e-4
    e2_source = torch.where(use_fallback[..., None], fallback_perp, reference_perp)
    e2 = e2_source / _safe_norm(e2_source)[..., None].clamp_min(1e-8)
    e3 = torch.cross(e1, e2, dim=-1)
    return x_a, e1, e2, e3, edge_len


def _project_to_edge_frame(
    vector: torch.Tensor,
    e1: torch.Tensor,
    e2: torch.Tensor,
    e3: torch.Tensor,
) -> torch.Tensor:
    direction = vector / _safe_norm(vector)[..., None].clamp_min(1e-8)
    return torch.stack(
        [
            torch.sum(direction * e1, dim=-1),
            torch.sum(direction * e2, dim=-1),
            torch.sum(direction * e3, dim=-1),
        ],
        dim=-1,
    )


def _other_vertex_edge_features(
    vector: torch.Tensor,
    e1: torch.Tensor,
    e2: torch.Tensor,
    e3: torch.Tensor,
    *,
    distance_max: float,
) -> torch.Tensor:
    scale = max(float(distance_max), 1.0)
    axial = torch.sum(vector * e1, dim=-1)
    radial_vector = vector - axial[..., None] * e1
    return torch.cat(
        [
            (_safe_norm(vector) / scale)[..., None],
            (axial / scale)[..., None],
            (_safe_norm(radial_vector) / scale)[..., None],
            _project_to_edge_frame(vector, e1, e2, e3),
        ],
        dim=-1,
    )


def face_edge_frame_features(
    edge_indices: torch.Tensor,
    opposite_indices: torch.Tensor,
    *,
    recycled_ca_coords: Optional[torch.Tensor],
    recycled_frames: Optional[torch.Tensor],
    distance_max: float,
) -> torch.Tensor:
    """Scalarize selected face geometry in each directed boundary-edge frame."""

    if recycled_ca_coords is None:
        return torch.zeros(*edge_indices.shape[:-1], FACE_EDGE_FRAME_DIM, device=edge_indices.device)
    x_a, e1, e2, e3, edge_len = _edge_frame_axes(edge_indices, recycled_ca_coords, recycled_frames)
    x_b = gather_single(recycled_ca_coords, edge_indices[..., 1])
    x_c = gather_single(recycled_ca_coords, opposite_indices)
    vc = x_c - x_a
    normal = torch.cross(x_b - x_a, vc, dim=-1)
    scale = max(float(distance_max), 1.0)
    return torch.cat(
        [
            (edge_len / scale)[..., None],
            _other_vertex_edge_features(vc, e1, e2, e3, distance_max=distance_max),
            _project_to_edge_frame(normal, e1, e2, e3),
        ],
        dim=-1,
    )


def tetra_edge_frame_features(
    edge_indices: torch.Tensor,
    opposite_indices: torch.Tensor,
    *,
    recycled_ca_coords: Optional[torch.Tensor],
    recycled_frames: Optional[torch.Tensor],
    distance_max: float,
    volume_scale: float,
) -> torch.Tensor:
    """Scalarize selected tetra geometry in each directed boundary-edge frame."""

    if recycled_ca_coords is None:
        return torch.zeros(*edge_indices.shape[:-1], TETRA_EDGE_FRAME_DIM, device=edge_indices.device)
    x_a, e1, e2, e3, edge_len = _edge_frame_axes(edge_indices, recycled_ca_coords, recycled_frames)
    x_b = gather_single(recycled_ca_coords, edge_indices[..., 1])
    x_c = gather_single(recycled_ca_coords, opposite_indices[..., 0])
    x_d = gather_single(recycled_ca_coords, opposite_indices[..., 1])
    vc = x_c - x_a
    vd = x_d - x_a
    plane_normal = torch.cross(vc, vd, dim=-1)
    signed_volume = torch.sum(torch.cross(x_b - x_a, vc, dim=-1) * vd, dim=-1) / 6.0
    volume_denom = torch.log1p(torch.as_tensor(volume_scale, device=edge_indices.device, dtype=edge_len.dtype))
    signed_log_volume = torch.sign(signed_volume) * torch.log1p(torch.abs(signed_volume)) / volume_denom.clamp_min(1.0)
    scale = max(float(distance_max), 1.0)
    return torch.cat(
        [
            (edge_len / scale)[..., None],
            _other_vertex_edge_features(vc, e1, e2, e3, distance_max=distance_max),
            _other_vertex_edge_features(vd, e1, e2, e3, distance_max=distance_max),
            _project_to_edge_frame(plane_normal, e1, e2, e3),
            _cosine(vc, vd)[..., None],
            signed_log_volume[..., None],
        ],
        dim=-1,
    )


def segment_cell_indices(
    *,
    batch_size: int,
    num_residues: int,
    radius: int,
    device: torch.device,
    seq_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return contiguous latent rank-2 segment cells centered on each residue."""

    offsets = torch.arange(-int(radius), int(radius) + 1, device=device)
    centers = torch.arange(num_residues, device=device)
    raw = centers[:, None] + offsets[None, :]
    in_bounds = (raw >= 0) & (raw < num_residues)
    indices = raw.clamp(0, max(num_residues - 1, 0)).to(torch.long)
    indices = indices[None, :, :].expand(batch_size, -1, -1)
    mask = in_bounds[None, :, :].expand(batch_size, -1, -1).to(dtype=torch.float32)
    if seq_mask is not None:
        residue_mask = gather_single(seq_mask[..., None].to(mask.dtype), indices).squeeze(-1)
        anchor_mask = seq_mask.to(mask.dtype)[..., None]
        mask = mask * residue_mask * anchor_mask
    return indices, mask


def segment_geometry_features(
    segment_indices: torch.Tensor,
    segment_mask: torch.Tensor,
    *,
    recycled_ca_coords: Optional[torch.Tensor],
    sequence_max: float,
    distance_max: float,
) -> torch.Tensor:
    """Build invariant features for contiguous latent segment cells."""

    dtype = torch.float32 if recycled_ca_coords is None else recycled_ca_coords.dtype
    device = segment_indices.device
    b, l, _ = segment_indices.shape
    centers = torch.arange(l, device=device).reshape(1, l, 1).expand(b, -1, segment_indices.shape[-1])
    mask = segment_mask.to(dtype)
    count = mask.sum(dim=-1).clamp_min(1.0)
    valid_fraction = mask.mean(dim=-1)
    mean_abs_offset = (torch.abs(segment_indices - centers).to(dtype) * mask).sum(dim=-1) / count
    seq_scale = max(float(sequence_max), 1.0)
    if recycled_ca_coords is None:
        mean_distance = torch.zeros_like(valid_fraction)
        max_distance = torch.zeros_like(valid_fraction)
    else:
        x_center = gather_single(recycled_ca_coords, centers[..., 0])
        x_window = gather_single(recycled_ca_coords, segment_indices)
        distances = _safe_norm(x_window - x_center[..., None, :]) * mask
        mean_distance = distances.sum(dim=-1) / count
        max_distance = distances.masked_fill(mask <= 0, 0.0).amax(dim=-1)
    dist_scale = max(float(distance_max), 1.0)
    return torch.stack(
        [
            valid_fraction,
            mean_abs_offset / seq_scale,
            mean_distance / dist_scale,
            max_distance / dist_scale,
        ],
        dim=-1,
    ).to(dtype)


class SimplexMLP(torch.nn.Module):
    """LayerNorm-ReLU MLP with AF2-style final initialisation."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, *, final_init: str = "default"):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(in_dim)
        self.linear_1 = torch.nn.Linear(in_dim, hidden_dim)
        self.linear_2 = torch.nn.Linear(hidden_dim, out_dim)
        init_linear(self.linear_1, init="relu")
        init_linear(self.linear_2, init=final_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(torch.relu(self.linear_1(self.layer_norm(x))))


class SimplexSingleTransition(torch.nn.Module):
    """Small residual transition for the single stream."""

    def __init__(self, config):
        super().__init__()
        c_s = config.c_s
        hidden = int(getattr(config, "simplex_single_transition_n", 2)) * c_s
        self.transition = SimplexMLP(c_s, hidden, c_s, final_init="final")

    def forward(self, single: torch.Tensor, seq_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        update = self.transition(single)
        if seq_mask is not None:
            update = update * seq_mask[..., None]
        return update


class SimplexAuxiliaryHeads(torch.nn.Module):
    """Geometry heads for forcing face/tetra states to encode real shape."""

    def __init__(self, config):
        super().__init__()
        hidden = int(getattr(config, "simplex_hidden_dim", max(config.c_z, config.c_s)))
        c_face = int(getattr(config, "simplex_c_face", 32))
        c_tetra = int(getattr(config, "simplex_c_tetra", 16))
        self.n_dist_bins = int(getattr(config, "n_dist_bins", 64))
        self.face_area = SimplexMLP(c_face, hidden, 1, final_init="final")
        self.face_distances = SimplexMLP(c_face, hidden, 3 * self.n_dist_bins, final_init="final")
        self.tetra_geometry = SimplexMLP(c_tetra, hidden, 3, final_init="final")
        self.tetra_distances = SimplexMLP(c_tetra, hidden, 6 * self.n_dist_bins, final_init="final")

    def forward(self, face_state: torch.Tensor, tetra_state: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = {
            "simplex_face_area_logits": self.face_area(face_state).squeeze(-1),
            "simplex_face_distance_logits": self.face_distances(face_state).reshape(
                *face_state.shape[:-1], 3, self.n_dist_bins
            ),
        }
        if tetra_state.numel() == 0:
            outputs["simplex_tetra_geometry_logits"] = tetra_state.new_empty(*tetra_state.shape[:-1], 3)
            outputs["simplex_tetra_distance_logits"] = tetra_state.new_empty(
                *tetra_state.shape[:-1], 6, self.n_dist_bins
            )
        else:
            outputs["simplex_tetra_geometry_logits"] = self.tetra_geometry(tetra_state)
            outputs["simplex_tetra_distance_logits"] = self.tetra_distances(tetra_state).reshape(
                *tetra_state.shape[:-1], 6, self.n_dist_bins
            )
        return outputs


class SimplicialAdapter(torch.nn.Module):
    """Sparse 2-/3-simplex message passing adapter.

    The adapter builds non-zero face/tetra states from pair, single, and
    recycled geometry inputs, then passes their boundary messages back into
    pair and single streams.  Auxiliary heads supervise the same explicit
    simplex states that produce those boundary messages.
    """

    def __init__(self, config):
        super().__init__()
        self.c_z = config.c_z
        self.c_s = config.c_s
        self.c_face = int(getattr(config, "simplex_c_face", 32))
        self.c_tetra = int(getattr(config, "simplex_c_tetra", 16))
        self.c_segment = int(getattr(config, "simplex_c_segment", max(8, self.c_tetra)))
        self.hidden_dim = int(getattr(config, "simplex_hidden_dim", max(config.c_z, config.c_s)))
        self.neighbor_k = int(getattr(config, "simplex_neighbor_k", 12))
        self.use_faces = bool(getattr(config, "simplex_use_faces", True))
        self.use_tetra = bool(getattr(config, "simplex_use_tetra", True))
        self.use_msa_to_face = bool(getattr(config, "simplex_use_msa_to_face", False))
        self.msa_to_face_rank = int(getattr(config, "simplex_msa_to_face_rank", 16))
        self.use_recycled_geometry = bool(getattr(config, "simplex_use_recycled_geometry", True))
        self.local_neighbor_k = int(getattr(config, "simplex_local_neighbor_k", 0))
        self.local_radius = int(getattr(config, "simplex_local_radius", 4))
        self.local_bias = float(getattr(config, "simplex_local_bias", 5.0))
        self.long_min_sep = int(getattr(config, "simplex_long_min_sep", 24))
        self.long_bias = float(getattr(config, "simplex_long_bias", 0.0))
        self.geometry_distance_weight = float(getattr(config, "simplex_geometry_distance_weight", 0.1))
        self.boundary_closure_weight = float(getattr(config, "simplex_boundary_closure_weight", 0.0))
        self.boundary_closure_temperature = float(getattr(config, "simplex_boundary_closure_temperature", 1.0))
        self.face_top_k = int(getattr(config, "simplex_face_top_k", 0))
        self.tetra_top_k = int(getattr(config, "simplex_tetra_top_k", 0))
        self.cell_score_degree_penalty = float(getattr(config, "simplex_cell_score_degree_penalty", 0.0))
        self.cell_score_outer_edge_weight = float(getattr(config, "simplex_cell_score_outer_edge_weight", 0.0))
        self.cell_score_segment_weight = float(getattr(config, "simplex_cell_score_segment_weight", 0.0))
        self.rbf_bins = int(getattr(config, "simplex_rbf_bins", 8))
        self.sequence_max = float(getattr(config, "simplex_sequence_max", 64.0))
        self.distance_max = float(getattr(config, "simplex_distance_max", 32.0))
        self.area_max = float(getattr(config, "simplex_area_max", 300.0))
        self.volume_scale = float(getattr(config, "simplex_volume_scale", 1000.0))
        self.dropout = torch.nn.Dropout(float(getattr(config, "simplex_dropout", 0.0)))
        self.cell_dropout = min(max(float(getattr(config, "simplex_cell_dropout", 0.0)), 0.0), 0.95)
        self.pair_update_scale = float(getattr(config, "simplex_pair_update_scale", 1.0))
        self.single_update_scale = float(getattr(config, "simplex_single_update_scale", 1.0))
        self.structure_readout_scale = float(getattr(config, "simplex_structure_readout_scale", 0.0))
        self.structure_pair_readout_scale = float(
            getattr(config, "simplex_structure_pair_readout_scale", 0.0)
        )
        self.msa_feedback_scale = float(getattr(config, "simplex_msa_feedback_scale", 0.0))
        self.boundary_msa_feedback_scale = float(getattr(config, "simplex_boundary_msa_feedback_scale", 0.0))
        self.boundary_pair_feedback_scale = float(getattr(config, "simplex_boundary_pair_feedback_scale", 0.0))
        self.boundary_pair_gate_scale = float(getattr(config, "simplex_boundary_pair_gate_scale", 0.0))
        self.boundary_metric_gate_scale = float(getattr(config, "simplex_boundary_metric_gate_scale", 0.0))
        self.boundary_cochain_recycling_scale = float(
            getattr(config, "simplex_boundary_cochain_recycling_scale", 0.0)
        )
        self.outer_edge_update_scale = float(getattr(config, "simplex_outer_edge_update_scale", 0.0))
        self.outer_edge_context_scale = float(getattr(config, "simplex_outer_edge_context_scale", 0.0))
        self.outer_edge_residual_context_scale = min(
            max(float(getattr(config, "simplex_outer_edge_residual_context_scale", 0.0)), 0.0),
            1.0,
        )
        self.hodge_face_update_scale = float(getattr(config, "simplex_hodge_face_update_scale", 0.0))
        self.signed_tetra_coboundary_scale = float(
            getattr(config, "simplex_signed_tetra_coboundary_scale", 0.0)
        )
        self.signed_tetra_to_face_scale = min(
            max(float(getattr(config, "simplex_signed_tetra_to_face_scale", 0.0)), 0.0),
            1.0,
        )
        self.edge_frame_message_scale = float(getattr(config, "simplex_edge_frame_message_scale", 0.0))
        self.boundary_edge_frame_gate_scale = float(
            getattr(config, "simplex_boundary_edge_frame_gate_scale", 0.0)
        )
        self.boundary_message_degree_attenuation = float(
            getattr(config, "simplex_boundary_message_degree_attenuation", 0.0)
        )
        self.boundary_incidence_normalization = float(
            getattr(config, "simplex_boundary_incidence_normalization", 0.0)
        )
        self.boundary_readout_directionality = min(
            max(float(getattr(config, "simplex_boundary_readout_directionality", 0.0)), 0.0),
            1.0,
        )
        self.boundary_hodge_readout_scale = min(
            max(float(getattr(config, "simplex_boundary_hodge_readout_scale", 0.0)), 0.0),
            1.0,
        )
        self.boundary_edge_star_readout_scale = min(
            max(float(getattr(config, "simplex_boundary_edge_star_readout_scale", 0.0)), 0.0),
            1.0,
        )
        self.boundary_edge_star_residual_scale = min(
            max(float(getattr(config, "simplex_boundary_edge_star_residual_scale", 0.0)), 0.0),
            1.0,
        )
        self.boundary_oriented_cochain_scale = min(
            max(float(getattr(config, "simplex_boundary_oriented_cochain_scale", 0.0)), 0.0),
            1.0,
        )
        self.boundary_face_cyclic_readout_scale = min(
            max(float(getattr(config, "simplex_boundary_face_cyclic_readout_scale", 0.0)), 0.0),
            1.0,
        )
        self.boundary_signed_face_cyclic_readout_scale = min(
            max(float(getattr(config, "simplex_boundary_signed_face_cyclic_readout_scale", 0.0)), 0.0),
            1.0,
        )
        self.triangle_attention_bias_scale = float(
            getattr(config, "simplex_triangle_attention_bias_scale", 0.0)
        )
        self.triangle_attention_value_scale = float(
            getattr(config, "simplex_triangle_attention_value_scale", 0.0)
        )
        self.triangle_num_heads = int(getattr(config, "triangle_num_heads", 1))
        self.global_context_scale = float(getattr(config, "simplex_global_context_scale", 0.0))
        self.vertex_star_context_scale = min(
            max(float(getattr(config, "simplex_vertex_star_context_scale", 0.0)), 0.0),
            1.0,
        )
        self.edge_star_context_scale = min(
            max(float(getattr(config, "simplex_edge_star_context_scale", 0.0)), 0.0),
            1.0,
        )
        self.segment_cell_scale = float(getattr(config, "simplex_segment_cell_scale", 0.0))
        self.segment_radius = int(getattr(config, "simplex_segment_radius", 4))

        self.pair_score_norm = torch.nn.LayerNorm(config.c_z)
        self.topology_score = torch.nn.Linear(config.c_z, 1)
        init_linear(self.topology_score, init="default")

        face_geom = face_geometry_dim(self.rbf_bins)
        self.face_init = SimplexMLP(3 * config.c_z + 3 * config.c_s + face_geom, self.hidden_dim, self.c_face)
        self.edge_to_face = SimplexMLP(self.c_face + 3 * config.c_z, self.hidden_dim, self.c_face)
        self.face_gate = torch.nn.Linear(self.c_face, self.c_face)
        init_gate_linear(self.face_gate)
        self.face_to_edge = SimplexMLP(self.c_face, self.hidden_dim, 3 * config.c_z)
        self.face_to_single = SimplexMLP(self.c_face, self.hidden_dim, 3 * config.c_s)
        self.msa_to_face_a = torch.nn.Linear(config.c_m, self.msa_to_face_rank, bias=False)
        self.msa_to_face_b = torch.nn.Linear(config.c_m, self.msa_to_face_rank, bias=False)
        self.msa_to_face_c = torch.nn.Linear(config.c_m, self.msa_to_face_rank, bias=False)
        self.msa_to_face = SimplexMLP(self.msa_to_face_rank, self.hidden_dim, self.c_face, final_init="final")
        init_linear(self.msa_to_face_a, init="default")
        init_linear(self.msa_to_face_b, init="default")
        init_linear(self.msa_to_face_c, init="default")

        tetra_geom = tetra_geometry_dim(self.rbf_bins)
        tetra_input_dim = 6 * config.c_z + 3 * self.c_face + 4 * config.c_s + tetra_geom
        self.tetra_init = SimplexMLP(tetra_input_dim, self.hidden_dim, self.c_tetra)
        self.face_to_tetra = SimplexMLP(self.c_tetra + 3 * self.c_face, self.hidden_dim, self.c_tetra)
        self.tetra_gate = torch.nn.Linear(self.c_tetra, self.c_tetra)
        init_gate_linear(self.tetra_gate)
        self.tetra_to_face = SimplexMLP(self.c_tetra, self.hidden_dim, 3 * self.c_face)
        self.tetra_to_edge = SimplexMLP(self.c_tetra, self.hidden_dim, 6 * config.c_z)
        self.tetra_to_single = SimplexMLP(self.c_tetra, self.hidden_dim, 4 * config.c_s)

        self.single_gate = torch.nn.Linear(config.c_s, config.c_s)
        init_gate_linear(self.single_gate)
        self.single_norm = torch.nn.LayerNorm(config.c_s)
        if self.msa_feedback_scale > 0.0:
            self.single_to_msa_feedback = SimplexMLP(config.c_s, self.hidden_dim, config.c_m)
        if self.boundary_msa_feedback_scale > 0.0:
            self.boundary_to_msa_feedback = SimplexMLP(2 * config.c_z, self.hidden_dim, config.c_m)
        if self.boundary_pair_feedback_scale > 0.0:
            self.boundary_to_pair_feedback = SimplexMLP(3 * config.c_z, self.hidden_dim, config.c_z)
        if self.boundary_pair_gate_scale > 0.0:
            self.boundary_pair_gate = SimplexMLP(2 * config.c_z, self.hidden_dim, config.c_z)
        if self.triangle_attention_bias_scale > 0.0:
            self.face_to_triangle_attention_bias = torch.nn.Linear(
                self.c_face,
                2 * self.triangle_num_heads,
            )
            init_linear(self.face_to_triangle_attention_bias, init="final")
            if self.use_tetra:
                self.tetra_to_triangle_attention_bias = torch.nn.Linear(
                    self.c_tetra,
                    2 * self.triangle_num_heads,
                )
                init_linear(self.tetra_to_triangle_attention_bias, init="final")
        if self.triangle_attention_value_scale > 0.0:
            self.face_to_triangle_attention_value = torch.nn.Linear(
                self.c_face,
                2 * config.c_z,
            )
            init_linear(self.face_to_triangle_attention_value, init="final")
            if self.use_tetra:
                self.tetra_to_triangle_attention_value = torch.nn.Linear(
                    self.c_tetra,
                    2 * config.c_z,
                )
                init_linear(self.tetra_to_triangle_attention_value, init="final")
        self.auxiliary_heads = SimplexAuxiliaryHeads(config)
        if self.outer_edge_context_scale > 0.0:
            self.face_outer_edge_context = SimplexMLP(
                self.c_face + 2 * config.c_z,
                self.hidden_dim,
                self.c_face,
            )
            if self.use_tetra:
                self.tetra_outer_edge_context = SimplexMLP(
                    self.c_tetra + 2 * config.c_z,
                    self.hidden_dim,
                    self.c_tetra,
                )
        if self.edge_frame_message_scale > 0.0:
            self.face_edge_frame_to_edge = SimplexMLP(
                self.c_face + FACE_EDGE_FRAME_DIM,
                self.hidden_dim,
                self.c_z,
            )
            if self.use_tetra:
                self.tetra_edge_frame_to_edge = SimplexMLP(
                    self.c_tetra + TETRA_EDGE_FRAME_DIM,
                    self.hidden_dim,
                    self.c_z,
                )
        if self.boundary_edge_frame_gate_scale > 0.0:
            self.face_boundary_edge_frame_gate = SimplexMLP(
                self.c_face + config.c_z + FACE_EDGE_FRAME_DIM,
                self.hidden_dim,
                self.c_z,
            )
        if self.global_context_scale > 0.0:
            global_dim = self.c_face + self.c_tetra
            self.global_to_face = SimplexMLP(
                self.c_face + global_dim,
                self.hidden_dim,
                self.c_face,
                final_init="final",
            )
            if self.use_tetra:
                self.global_to_tetra = SimplexMLP(
                    self.c_tetra + global_dim,
                    self.hidden_dim,
                    self.c_tetra,
                    final_init="final",
                )
        if self.segment_cell_scale > 0.0:
            self.segment_init = SimplexMLP(
                2 * config.c_s + config.c_z + SEGMENT_GEOMETRY_DIM,
                self.hidden_dim,
                self.c_segment,
            )
            self.segment_to_face = SimplexMLP(
                self.c_face + self.c_segment,
                self.hidden_dim,
                self.c_face,
            )

    def _empty_outputs(
        self,
        pair: torch.Tensor,
        single: torch.Tensor,
        contact_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        b, l, _, _ = pair.shape
        empty_face_state = pair.new_empty((b, l, 0, self.c_face))
        empty_tetra_state = pair.new_empty((b, l, 0, self.c_tetra))
        aux = self.auxiliary_heads(empty_face_state, empty_tetra_state)
        aux.update(
            {
                "simplex_contact_logits": contact_logits,
                "simplex_face_indices": torch.empty((b, l, 0, 3), device=pair.device, dtype=torch.long),
                "simplex_tetra_indices": torch.empty((b, l, 0, 4), device=pair.device, dtype=torch.long),
                "simplex_face_mask": pair.new_empty((b, l, 0)),
                "simplex_tetra_mask": pair.new_empty((b, l, 0)),
                "simplex_tetra_face_slots": torch.empty((0, 3), device=pair.device, dtype=torch.long),
                "simplex_neighbor_indices": torch.empty((b, l, 0), device=pair.device, dtype=torch.long),
            }
        )
        return aux

    def _apply_cell_dropout(self, topology: SimplexTopology) -> SimplexTopology:
        if (not self.training) or self.cell_dropout <= 0.0:
            return topology
        keep_probability = 1.0 - self.cell_dropout
        face_mask = topology.face_mask
        tetra_mask = topology.tetra_mask
        if face_mask.numel() > 0:
            face_keep = (torch.rand_like(face_mask) < keep_probability).to(face_mask.dtype)
            face_mask = face_mask * face_keep
        if tetra_mask.numel() > 0:
            tetra_keep = (torch.rand_like(tetra_mask) < keep_probability).to(tetra_mask.dtype)
            tetra_mask = tetra_mask * tetra_keep
        return SimplexTopology(
            nbr_idx=topology.nbr_idx,
            face_indices=topology.face_indices,
            tetra_indices=topology.tetra_indices,
            face_mask=face_mask,
            tetra_mask=tetra_mask,
            face_neighbor_slots=topology.face_neighbor_slots,
            tetra_neighbor_slots=topology.tetra_neighbor_slots,
            tetra_face_slots=topology.tetra_face_slots,
        )

    def forward(
        self,
        pair: torch.Tensor,
        single: torch.Tensor,
        *,
        msa_representation: Optional[torch.Tensor] = None,
        msa_mask: Optional[torch.Tensor] = None,
        seq_mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
        recycled_ca_coords: Optional[torch.Tensor] = None,
        recycled_frames: Optional[torch.Tensor] = None,
        simplex_teacher_ca_coords: Optional[torch.Tensor] = None,
        simplex_teacher_ca_mask: Optional[torch.Tensor] = None,
        simplex_teacher_forcing_weight: Optional[torch.Tensor] = None,
        simplex_pair_update_scale_override: Optional[torch.Tensor] = None,
        simplex_single_update_scale_override: Optional[torch.Tensor] = None,
        simplex_outer_edge_context_scale_override: Optional[torch.Tensor] = None,
        simplex_outer_edge_residual_context_scale_override: Optional[torch.Tensor] = None,
        simplex_hodge_face_update_scale_override: Optional[torch.Tensor] = None,
        simplex_signed_tetra_coboundary_scale_override: Optional[torch.Tensor] = None,
        simplex_signed_tetra_to_face_scale_override: Optional[torch.Tensor] = None,
        simplex_edge_frame_message_scale_override: Optional[torch.Tensor] = None,
        simplex_boundary_edge_frame_gate_scale_override: Optional[torch.Tensor] = None,
        simplex_boundary_readout_directionality_override: Optional[torch.Tensor] = None,
        simplex_boundary_hodge_readout_scale_override: Optional[torch.Tensor] = None,
        simplex_boundary_edge_star_readout_scale_override: Optional[torch.Tensor] = None,
        simplex_boundary_edge_star_residual_scale_override: Optional[torch.Tensor] = None,
        simplex_boundary_oriented_cochain_scale_override: Optional[torch.Tensor] = None,
        simplex_boundary_face_cyclic_readout_scale_override: Optional[torch.Tensor] = None,
        simplex_boundary_signed_face_cyclic_readout_scale_override: Optional[torch.Tensor] = None,
        simplex_vertex_star_context_scale_override: Optional[torch.Tensor] = None,
        simplex_edge_star_context_scale_override: Optional[torch.Tensor] = None,
        simplex_segment_cell_scale_override: Optional[torch.Tensor] = None,
        simplex_msa_feedback_scale_override: Optional[torch.Tensor] = None,
        simplex_boundary_pair_feedback_scale_override: Optional[torch.Tensor] = None,
        simplex_boundary_pair_gate_scale_override: Optional[torch.Tensor] = None,
        simplex_boundary_metric_gate_scale_override: Optional[torch.Tensor] = None,
        simplex_triangle_attention_bias_scale_override: Optional[torch.Tensor] = None,
        simplex_triangle_attention_value_scale_override: Optional[torch.Tensor] = None,
        simplex_local_neighbor_k_override: Optional[torch.Tensor] = None,
        simplex_geometry_distance_weight_override: Optional[torch.Tensor] = None,
        simplex_face_top_k_override: Optional[torch.Tensor] = None,
        simplex_tetra_top_k_override: Optional[torch.Tensor] = None,
        simplex_cell_score_outer_edge_weight_override: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if pair.ndim != 4 or single.ndim != 3:
            raise ValueError("pair must be [B, L, L, Cz] and single must be [B, L, Cs]")

        pair_update_scale = self.pair_update_scale
        if simplex_pair_update_scale_override is not None:
            pair_update_scale = max(float(simplex_pair_update_scale_override.detach().float().cpu().item()), 0.0)
        single_update_scale = self.single_update_scale
        if simplex_single_update_scale_override is not None:
            single_update_scale = max(float(simplex_single_update_scale_override.detach().float().cpu().item()), 0.0)
        outer_edge_context_scale = self.outer_edge_context_scale
        if simplex_outer_edge_context_scale_override is not None:
            outer_edge_context_scale = max(
                float(simplex_outer_edge_context_scale_override.detach().float().cpu().item()),
                0.0,
            )
        outer_edge_residual_context_scale = self.outer_edge_residual_context_scale
        if simplex_outer_edge_residual_context_scale_override is not None:
            outer_edge_residual_context_scale = min(
                max(
                    float(simplex_outer_edge_residual_context_scale_override.detach().float().cpu().item()),
                    0.0,
                ),
                1.0,
            )
        hodge_face_update_scale = self.hodge_face_update_scale
        if simplex_hodge_face_update_scale_override is not None:
            hodge_face_update_scale = max(
                float(simplex_hodge_face_update_scale_override.detach().float().cpu().item()),
                0.0,
            )
        signed_tetra_coboundary_scale = self.signed_tetra_coboundary_scale
        if simplex_signed_tetra_coboundary_scale_override is not None:
            signed_tetra_coboundary_scale = max(
                float(simplex_signed_tetra_coboundary_scale_override.detach().float().cpu().item()),
                0.0,
            )
        signed_tetra_to_face_scale = self.signed_tetra_to_face_scale
        if simplex_signed_tetra_to_face_scale_override is not None:
            signed_tetra_to_face_scale = min(
                max(
                    float(simplex_signed_tetra_to_face_scale_override.detach().float().cpu().item()),
                    0.0,
                ),
                1.0,
            )
        edge_frame_message_scale = self.edge_frame_message_scale
        if simplex_edge_frame_message_scale_override is not None:
            edge_frame_message_scale = max(
                float(simplex_edge_frame_message_scale_override.detach().float().cpu().item()),
                0.0,
            )
        boundary_edge_frame_gate_scale = self.boundary_edge_frame_gate_scale
        if simplex_boundary_edge_frame_gate_scale_override is not None:
            boundary_edge_frame_gate_scale = max(
                float(simplex_boundary_edge_frame_gate_scale_override.detach().float().cpu().item()),
                0.0,
            )
        boundary_readout_directionality = self.boundary_readout_directionality
        if simplex_boundary_readout_directionality_override is not None:
            boundary_readout_directionality = min(
                max(float(simplex_boundary_readout_directionality_override.detach().float().cpu().item()), 0.0),
                1.0,
            )
        boundary_hodge_readout_scale = self.boundary_hodge_readout_scale
        if simplex_boundary_hodge_readout_scale_override is not None:
            boundary_hodge_readout_scale = min(
                max(float(simplex_boundary_hodge_readout_scale_override.detach().float().cpu().item()), 0.0),
                1.0,
            )
        boundary_edge_star_readout_scale = self.boundary_edge_star_readout_scale
        if simplex_boundary_edge_star_readout_scale_override is not None:
            boundary_edge_star_readout_scale = min(
                max(float(simplex_boundary_edge_star_readout_scale_override.detach().float().cpu().item()), 0.0),
                1.0,
            )
        boundary_edge_star_residual_scale = self.boundary_edge_star_residual_scale
        if simplex_boundary_edge_star_residual_scale_override is not None:
            boundary_edge_star_residual_scale = min(
                max(
                    float(simplex_boundary_edge_star_residual_scale_override.detach().float().cpu().item()),
                    0.0,
                ),
                1.0,
            )
        boundary_oriented_cochain_scale = self.boundary_oriented_cochain_scale
        if simplex_boundary_oriented_cochain_scale_override is not None:
            boundary_oriented_cochain_scale = min(
                max(
                    float(simplex_boundary_oriented_cochain_scale_override.detach().float().cpu().item()),
                    0.0,
                ),
                1.0,
            )
        boundary_face_cyclic_readout_scale = self.boundary_face_cyclic_readout_scale
        if simplex_boundary_face_cyclic_readout_scale_override is not None:
            boundary_face_cyclic_readout_scale = min(
                max(
                    float(simplex_boundary_face_cyclic_readout_scale_override.detach().float().cpu().item()),
                    0.0,
                ),
                1.0,
            )
        boundary_signed_face_cyclic_readout_scale = self.boundary_signed_face_cyclic_readout_scale
        if simplex_boundary_signed_face_cyclic_readout_scale_override is not None:
            boundary_signed_face_cyclic_readout_scale = min(
                max(
                    float(
                        simplex_boundary_signed_face_cyclic_readout_scale_override.detach().float().cpu().item()
                    ),
                    0.0,
                ),
                1.0,
            )
        vertex_star_context_scale = self.vertex_star_context_scale
        if simplex_vertex_star_context_scale_override is not None:
            vertex_star_context_scale = min(
                max(float(simplex_vertex_star_context_scale_override.detach().float().cpu().item()), 0.0),
                1.0,
            )
        edge_star_context_scale = self.edge_star_context_scale
        if simplex_edge_star_context_scale_override is not None:
            edge_star_context_scale = min(
                max(float(simplex_edge_star_context_scale_override.detach().float().cpu().item()), 0.0),
                1.0,
            )
        segment_cell_scale = self.segment_cell_scale
        if simplex_segment_cell_scale_override is not None:
            segment_cell_scale = max(
                float(simplex_segment_cell_scale_override.detach().float().cpu().item()),
                0.0,
            )
        msa_feedback_scale = self.msa_feedback_scale
        boundary_msa_feedback_scale = self.boundary_msa_feedback_scale
        if simplex_msa_feedback_scale_override is not None:
            feedback_runtime_scale = max(
                float(simplex_msa_feedback_scale_override.detach().float().cpu().item()),
                0.0,
            )
            if self.msa_feedback_scale > 0.0:
                msa_feedback_scale = feedback_runtime_scale
            if self.boundary_msa_feedback_scale > 0.0:
                boundary_msa_feedback_scale = feedback_runtime_scale
        boundary_pair_feedback_scale = self.boundary_pair_feedback_scale
        if simplex_boundary_pair_feedback_scale_override is not None:
            boundary_pair_feedback_scale = max(
                float(simplex_boundary_pair_feedback_scale_override.detach().float().cpu().item()),
                0.0,
            )
        boundary_pair_gate_scale = self.boundary_pair_gate_scale
        if simplex_boundary_pair_gate_scale_override is not None:
            boundary_pair_gate_scale = max(
                float(simplex_boundary_pair_gate_scale_override.detach().float().cpu().item()),
                0.0,
            )
        boundary_metric_gate_scale = self.boundary_metric_gate_scale
        if simplex_boundary_metric_gate_scale_override is not None:
            boundary_metric_gate_scale = max(
                float(simplex_boundary_metric_gate_scale_override.detach().float().cpu().item()),
                0.0,
            )
        triangle_attention_bias_scale = self.triangle_attention_bias_scale
        if simplex_triangle_attention_bias_scale_override is not None:
            triangle_attention_bias_scale = max(
                float(simplex_triangle_attention_bias_scale_override.detach().float().cpu().item()),
                0.0,
            )
        triangle_attention_value_scale = self.triangle_attention_value_scale
        if simplex_triangle_attention_value_scale_override is not None:
            triangle_attention_value_scale = max(
                float(simplex_triangle_attention_value_scale_override.detach().float().cpu().item()),
                0.0,
            )
        local_neighbor_k = self.local_neighbor_k
        if simplex_local_neighbor_k_override is not None:
            local_neighbor_k = int(
                round(max(float(simplex_local_neighbor_k_override.detach().float().cpu().item()), 0.0))
            )
        geometry_distance_weight = self.geometry_distance_weight
        if simplex_geometry_distance_weight_override is not None:
            geometry_distance_weight = max(
                float(simplex_geometry_distance_weight_override.detach().float().cpu().item()),
                0.0,
            )
        face_top_k = self.face_top_k
        if simplex_face_top_k_override is not None:
            face_top_k = int(round(max(float(simplex_face_top_k_override.detach().float().cpu().item()), 0.0)))
        tetra_top_k = self.tetra_top_k
        if simplex_tetra_top_k_override is not None:
            tetra_top_k = int(round(max(float(simplex_tetra_top_k_override.detach().float().cpu().item()), 0.0)))
        cell_score_outer_edge_weight = self.cell_score_outer_edge_weight
        if simplex_cell_score_outer_edge_weight_override is not None:
            cell_score_outer_edge_weight = max(
                float(simplex_cell_score_outer_edge_weight_override.detach().float().cpu().item()),
                0.0,
            )

        score_raw = self.topology_score(self.pair_score_norm(pair)).squeeze(-1)
        contact_logits = 0.5 * (score_raw + score_raw.transpose(1, 2))
        if not self.use_faces:
            return pair, single, self._empty_outputs(pair, single, contact_logits)

        coords_for_topology = recycled_ca_coords if self.use_recycled_geometry else None
        frames_for_geometry = recycled_frames if self.use_recycled_geometry else None
        coords_for_geometry = recycled_ca_coords if self.use_recycled_geometry else None
        topology_score = contact_logits.detach()
        topology_pair_mask = pair_mask
        teacher_weight = 0.0
        if simplex_teacher_forcing_weight is not None:
            teacher_weight = float(simplex_teacher_forcing_weight.detach().float().cpu().item())
        if simplex_teacher_ca_coords is not None and teacher_weight > 0.0:
            teacher_weight = min(max(teacher_weight, 0.0), 1.0)
            teacher_distances = torch.cdist(
                simplex_teacher_ca_coords.detach().to(dtype=pair.dtype),
                simplex_teacher_ca_coords.detach().to(dtype=pair.dtype),
            )
            teacher_score = -teacher_distances.to(dtype=pair.dtype)
            topology_score = (1.0 - teacher_weight) * topology_score + teacher_weight * teacher_score
            if simplex_teacher_ca_mask is not None:
                teacher_valid = simplex_teacher_ca_mask > 0
                teacher_pair_mask = teacher_valid[:, :, None] & teacher_valid[:, None, :]
                if topology_pair_mask is None:
                    topology_pair_mask = teacher_pair_mask.to(dtype=pair.dtype)
                else:
                    topology_pair_mask = topology_pair_mask * teacher_pair_mask.to(dtype=topology_pair_mask.dtype)
        with torch.no_grad():
            topology = build_simplex_topology(
                topology_score.detach(),
                neighbor_k=self.neighbor_k,
                seq_mask=seq_mask,
                pair_mask=topology_pair_mask,
                recycled_ca_coords=coords_for_topology,
                local_neighbor_k=local_neighbor_k,
                local_radius=self.local_radius,
                local_bias=self.local_bias,
                long_min_sep=self.long_min_sep,
                long_bias=self.long_bias,
                geometry_distance_weight=geometry_distance_weight,
                boundary_closure_weight=self.boundary_closure_weight,
                boundary_closure_temperature=self.boundary_closure_temperature,
                face_top_k=face_top_k,
                tetra_top_k=tetra_top_k,
                cell_score_degree_penalty=self.cell_score_degree_penalty,
                cell_score_outer_edge_weight=cell_score_outer_edge_weight,
                cell_score_segment_weight=self.cell_score_segment_weight,
                segment_radius=self.segment_radius,
            )
        topology = self._apply_cell_dropout(topology)

        face_indices = topology.face_indices
        if face_indices.shape[2] == 0:
            return pair, single, self._empty_outputs(pair, single, contact_logits)

        i, j, k = face_indices.unbind(dim=-1)
        z_ij = gather_pair(pair, i, j)
        z_ik = gather_pair(pair, i, k)
        z_jk = gather_pair(pair, j, k)
        s_i = gather_single(single, i)
        s_j = gather_single(single, j)
        s_k = gather_single(single, k)
        face_geom = face_geometry_features(
            face_indices,
            recycled_ca_coords=coords_for_geometry,
            recycled_frames=frames_for_geometry,
            rbf_bins=self.rbf_bins,
            sequence_max=self.sequence_max,
            distance_max=self.distance_max,
            area_max=self.area_max,
        ).to(pair.dtype)
        face_state = self.face_init(torch.cat([z_ij, z_ik, z_jk, s_i, s_j, s_k, face_geom], dim=-1))
        face_state = face_state * topology.face_mask[..., None]
        segment_state = pair.new_empty((pair.shape[0], pair.shape[1], 0))
        if self.segment_cell_scale > 0.0 and segment_cell_scale > 0.0:
            segment_state = self._segment_pass(
                pair,
                single,
                seq_mask=seq_mask,
                recycled_ca_coords=coords_for_geometry,
            )
        if self.use_msa_to_face and msa_representation is not None:
            face_state = face_state + self._msa_to_face_update(
                msa_representation,
                face_indices,
                topology.face_mask,
                msa_mask=msa_mask,
            )
            face_state = face_state * topology.face_mask[..., None]
        if self.segment_cell_scale > 0.0 and segment_cell_scale > 0.0:
            seg_i = gather_single(segment_state, i)
            seg_j = gather_single(segment_state, j)
            seg_k = gather_single(segment_state, k)
            segment_context = (seg_i + seg_j + seg_k) / 3.0
            segment_msg = self.segment_to_face(torch.cat([face_state, segment_context], dim=-1))
            face_state = face_state + self.dropout(
                segment_cell_scale
                * torch.sigmoid(self.face_gate(face_state))
                * segment_msg
                * topology.face_mask[..., None]
            )
            face_state = face_state * topology.face_mask[..., None]

        edge_msg = self.edge_to_face(torch.cat([face_state, z_ij, z_ik, z_jk], dim=-1))
        face_edge_indices = torch.stack(
            [
                torch.stack([i, j], dim=-1),
                torch.stack([i, k], dim=-1),
                torch.stack([j, k], dim=-1),
            ],
            dim=-2,
        )
        face_edge_mask = topology.face_mask[..., None].expand(-1, -1, -1, 3)
        if self.boundary_incidence_normalization > 0.0:
            face_cell_weight = _cell_boundary_incidence_weights(
                face_edge_indices,
                face_edge_mask,
                num_residues=pair.shape[1],
                strength=self.boundary_incidence_normalization,
            )
            edge_msg = edge_msg * face_cell_weight[..., None]
        face_state = face_state + self.dropout(
            torch.sigmoid(self.face_gate(face_state)) * edge_msg * topology.face_mask[..., None]
        )
        if outer_edge_context_scale > 0.0:
            outer_context = cell_outer_edge_context(pair, face_indices, topology.face_mask, topology.nbr_idx)
            outer_msg = self.face_outer_edge_context(torch.cat([face_state, outer_context], dim=-1))
            face_state = face_state + self.dropout(
                outer_edge_context_scale
                * torch.sigmoid(self.face_gate(face_state))
                * outer_msg
                * topology.face_mask[..., None]
            )
            face_state = face_state * topology.face_mask[..., None]
        if outer_edge_residual_context_scale > 0.0:
            outer_msg = parameter_free_outer_edge_context_delta(
                pair,
                face_indices,
                topology.face_mask,
                topology.nbr_idx,
                channels=self.c_face,
            )
            outer_msg = _rms_match_update(outer_msg, face_state)
            face_state = face_state + self.dropout(
                outer_edge_residual_context_scale
                * torch.sigmoid(self.face_gate(face_state))
                * outer_msg
                * topology.face_mask[..., None]
            )
            face_state = face_state * topology.face_mask[..., None]
        if self.outer_edge_update_scale > 0.0:
            outer_msg = face_outer_edge_delta(
                face_state,
                face_indices,
                topology.face_mask,
                num_residues=pair.shape[1],
            )
            face_state = face_state + self.dropout(
                self.outer_edge_update_scale * torch.sigmoid(self.face_gate(face_state)) * outer_msg
            )
            face_state = face_state * topology.face_mask[..., None]
        if hodge_face_update_scale > 0.0:
            hodge_parts = [
                face_outer_edge_delta(
                    face_state,
                    face_indices,
                    topology.face_mask,
                    num_residues=pair.shape[1],
                )
            ]
            if self.use_tetra and topology.tetra_indices.shape[2] > 0:
                hodge_parts.append(
                    face_tetra_coboundary_delta(
                        face_state,
                        topology.face_mask,
                        topology.tetra_face_slots,
                        topology.tetra_mask,
                    )
                )
            hodge_msg = sum(hodge_parts) / float(len(hodge_parts))
            face_state = face_state + self.dropout(
                hodge_face_update_scale
                * torch.sigmoid(self.face_gate(face_state))
                * hodge_msg
                * topology.face_mask[..., None]
            )
            face_state = face_state * topology.face_mask[..., None]
        if (
            signed_tetra_coboundary_scale > 0.0
            and self.use_tetra
            and topology.tetra_indices.shape[2] > 0
        ):
            signed_coboundary_msg = signed_face_tetra_coboundary_delta(
                face_state,
                topology.face_mask,
                topology.tetra_face_slots,
                topology.tetra_mask,
            )
            face_state = face_state + self.dropout(
                signed_tetra_coboundary_scale
                * torch.sigmoid(self.face_gate(face_state))
                * signed_coboundary_msg
                * topology.face_mask[..., None]
            )
            face_state = face_state * topology.face_mask[..., None]

        tetra_state = pair.new_empty((pair.shape[0], pair.shape[1], 0, self.c_tetra))
        if self.use_tetra and topology.tetra_indices.shape[2] > 0:
            tetra_state = self._tetra_pass(
                pair,
                single,
                face_state,
                topology,
                coords_for_geometry,
                outer_edge_context_scale=outer_edge_context_scale,
                outer_edge_residual_context_scale=outer_edge_residual_context_scale,
                boundary_incidence_normalization=self.boundary_incidence_normalization,
            )

            face_delta = self.tetra_to_face(tetra_state).reshape(*tetra_state.shape[:-1], 3, self.c_face)
            if signed_tetra_to_face_scale > 0.0:
                face_delta = signed_tetra_face_boundary_updates(
                    face_delta,
                    scale=signed_tetra_to_face_scale,
                )
            face_delta = face_delta * topology.tetra_mask[..., None, None]
            face_delta_scattered = torch.zeros_like(face_state)
            face_counts = torch.zeros(*face_state.shape[:-1], 1, device=face_state.device, dtype=face_state.dtype)
            face_slot_ids = topology.tetra_face_slots[None, None, :, :].expand(
                face_state.shape[0], face_state.shape[1], -1, -1
            )
            face_delta_scattered.scatter_add_(
                2,
                face_slot_ids[..., None].expand(-1, -1, -1, -1, self.c_face).reshape(
                    face_state.shape[0], face_state.shape[1], -1, self.c_face
                ),
                face_delta.reshape(face_state.shape[0], face_state.shape[1], -1, self.c_face),
            )
            face_counts.scatter_add_(
                2,
                face_slot_ids[..., None].reshape(face_state.shape[0], face_state.shape[1], -1, 1),
                topology.tetra_mask[..., None, None].expand(-1, -1, -1, 3, -1).reshape(
                    face_state.shape[0], face_state.shape[1], -1, 1
                ),
            )
            face_state = face_state + self.dropout(face_delta_scattered / face_counts.clamp_min(1.0))
            face_state = face_state * topology.face_mask[..., None]

        if self.global_context_scale > 0.0:
            face_state, tetra_state = self._apply_global_context(
                face_state,
                face_indices,
                topology.face_mask,
                tetra_state,
                topology.tetra_indices,
                topology.tetra_mask,
                vertex_star_context_scale=vertex_star_context_scale,
                edge_star_context_scale=edge_star_context_scale,
            )

        triangle_attention_aux: dict[str, torch.Tensor] | None = None
        if (
            (self.triangle_attention_bias_scale > 0.0 and triangle_attention_bias_scale > 0.0)
            or (self.triangle_attention_value_scale > 0.0 and triangle_attention_value_scale > 0.0)
        ):
            triangle_attention_aux = self._triangle_attention_aux(
                face_state,
                face_indices,
                topology.face_mask,
                tetra_state,
                topology.tetra_indices,
                topology.tetra_mask,
                bias_scale=triangle_attention_bias_scale,
                value_scale=triangle_attention_value_scale,
            )

        face_edge_update = self.face_to_edge(face_state).reshape(*face_state.shape[:-1], 3, self.c_z)
        needs_face_edge_frame = (
            edge_frame_message_scale > 0.0 and self.edge_frame_message_scale > 0.0
        ) or self.boundary_edge_frame_gate_scale > 0.0
        if needs_face_edge_frame:
            face_opposite_indices = torch.stack([k, j, i], dim=-1)
            face_frame_features = face_edge_frame_features(
                face_edge_indices,
                face_opposite_indices,
                recycled_ca_coords=coords_for_geometry,
                recycled_frames=frames_for_geometry,
                distance_max=self.distance_max,
            ).to(pair.dtype)
            face_edge_state = face_state[..., None, :].expand(*face_edge_indices.shape[:-1], self.c_face)
        if (
            self.boundary_edge_frame_gate_scale > 0.0
            or (self.boundary_pair_gate_scale > 0.0 and boundary_pair_gate_scale > 0.0)
        ):
            face_edge_pair_state = torch.stack([z_ij, z_ik, z_jk], dim=-2)
        if edge_frame_message_scale > 0.0 and self.edge_frame_message_scale > 0.0:
            face_edge_update = face_edge_update + edge_frame_message_scale * self.face_edge_frame_to_edge(
                torch.cat([face_edge_state, face_frame_features], dim=-1)
            )
        if self.boundary_edge_frame_gate_scale > 0.0 and boundary_edge_frame_gate_scale > 0.0:
            face_edge_update = face_edge_update + self.dropout(
                boundary_edge_frame_gate_scale
                * torch.tanh(
                    self.face_boundary_edge_frame_gate(
                        torch.cat([face_edge_state, face_edge_pair_state, face_frame_features], dim=-1)
                    )
                )
                * face_edge_update
                * face_edge_mask[..., None]
            )
        if self.boundary_pair_gate_scale > 0.0 and boundary_pair_gate_scale > 0.0:
            face_edge_update = face_edge_update + self.dropout(
                boundary_pair_gate_scale
                * torch.tanh(self.boundary_pair_gate(torch.cat([face_edge_update, face_edge_pair_state], dim=-1)))
                * face_edge_update
                * face_edge_mask[..., None]
            )
        if boundary_metric_gate_scale > 0.0:
            face_distance_logits = self.auxiliary_heads.face_distances(face_state).reshape(
                *face_state.shape[:-1],
                3,
                self.auxiliary_heads.n_dist_bins,
            )
            face_edge_update = apply_boundary_metric_gate(
                face_edge_update,
                face_distance_logits,
                face_edge_mask,
                scale=boundary_metric_gate_scale,
            )
        if self.boundary_incidence_normalization > 0.0:
            face_edge_update = face_edge_update * boundary_incidence_weights(
                face_edge_indices,
                face_edge_mask,
                num_residues=pair.shape[1],
                strength=self.boundary_incidence_normalization,
            )[..., None]
        boundary_source_delta = pair.new_zeros(pair.shape[0], pair.shape[1], self.c_z)
        boundary_source_counts = pair.new_zeros(pair.shape[0], pair.shape[1], 1)
        boundary_target_delta = pair.new_zeros(pair.shape[0], pair.shape[1], self.c_z)
        boundary_target_counts = pair.new_zeros(pair.shape[0], pair.shape[1], 1)
        if self.boundary_msa_feedback_scale > 0.0 or self.boundary_pair_feedback_scale > 0.0:
            (
                boundary_source_delta,
                boundary_source_counts,
                boundary_target_delta,
                boundary_target_counts,
            ) = scatter_directed_edges_to_residue(
                face_edge_update,
                face_edge_indices,
                num_residues=pair.shape[1],
                edge_mask=face_edge_mask,
            )
        pair_delta, pair_counts = scatter_to_pair(
            face_edge_update,
            face_edge_indices,
            pair_shape=tuple(pair.shape),  # type: ignore[arg-type]
            edge_mask=face_edge_mask,
            include_reverse=True,
        )
        directed_pair_delta = pair_delta
        directed_pair_counts = pair_counts
        if boundary_readout_directionality > 0.0:
            directed_pair_delta, directed_pair_counts = scatter_to_pair(
                face_edge_update,
                face_edge_indices,
                pair_shape=tuple(pair.shape),  # type: ignore[arg-type]
                edge_mask=face_edge_mask,
                include_reverse=False,
            )
            if boundary_face_cyclic_readout_scale > 0.0:
                cyclic_face_pair_delta, cyclic_face_pair_counts = scatter_to_pair(
                    cyclic_face_boundary_updates(face_edge_update),
                    cyclic_face_boundary_edges(topology.face_indices),
                    pair_shape=tuple(pair.shape),  # type: ignore[arg-type]
                    edge_mask=face_edge_mask,
                    include_reverse=False,
                )
                cyclic_scale = pair.new_tensor(boundary_face_cyclic_readout_scale)
                directed_pair_delta = (
                    (1.0 - cyclic_scale) * directed_pair_delta
                    + cyclic_scale * cyclic_face_pair_delta
                )
                directed_pair_counts = (
                    (1.0 - cyclic_scale) * directed_pair_counts
                    + cyclic_scale * cyclic_face_pair_counts
                )
            if boundary_signed_face_cyclic_readout_scale > 0.0:
                signed_face_pair_delta, signed_face_pair_counts = scatter_to_pair(
                    signed_cyclic_face_boundary_updates(face_edge_update),
                    cyclic_face_boundary_edges(topology.face_indices),
                    pair_shape=tuple(pair.shape),  # type: ignore[arg-type]
                    edge_mask=face_edge_mask,
                    include_reverse=False,
                )
                signed_scale = pair.new_tensor(boundary_signed_face_cyclic_readout_scale)
                directed_pair_delta = (
                    (1.0 - signed_scale) * directed_pair_delta
                    + signed_scale * signed_face_pair_delta
                )
                directed_pair_counts = (
                    (1.0 - signed_scale) * directed_pair_counts
                    + signed_scale * signed_face_pair_counts
                )

        if self.use_tetra and tetra_state.numel() > 0:
            tet_i, tet_j, tet_k, tet_l = topology.tetra_indices.unbind(dim=-1)
            tet_edge_indices = torch.stack(
                [
                    torch.stack([tet_i, tet_j], dim=-1),
                    torch.stack([tet_i, tet_k], dim=-1),
                    torch.stack([tet_i, tet_l], dim=-1),
                    torch.stack([tet_j, tet_k], dim=-1),
                    torch.stack([tet_j, tet_l], dim=-1),
                    torch.stack([tet_k, tet_l], dim=-1),
                ],
                dim=-2,
            )
            tet_edge_update = self.tetra_to_edge(tetra_state).reshape(*tetra_state.shape[:-1], 6, self.c_z)
            tet_edge_mask = topology.tetra_mask[..., None].expand(-1, -1, -1, 6)
            needs_tet_edge_frame = edge_frame_message_scale > 0.0 and self.edge_frame_message_scale > 0.0
            if needs_tet_edge_frame:
                tet_opposite_indices = torch.stack(
                    [
                        torch.stack([tet_k, tet_l], dim=-1),
                        torch.stack([tet_j, tet_l], dim=-1),
                        torch.stack([tet_j, tet_k], dim=-1),
                        torch.stack([tet_i, tet_l], dim=-1),
                        torch.stack([tet_i, tet_k], dim=-1),
                        torch.stack([tet_i, tet_j], dim=-1),
                    ],
                    dim=-2,
                )
                tet_frame_features = tetra_edge_frame_features(
                    tet_edge_indices,
                    tet_opposite_indices,
                    recycled_ca_coords=coords_for_geometry,
                    recycled_frames=frames_for_geometry,
                    distance_max=self.distance_max,
                    volume_scale=self.volume_scale,
                ).to(pair.dtype)
                tetra_edge_state = tetra_state[..., None, :].expand(*tet_edge_indices.shape[:-1], self.c_tetra)
            if (
                self.boundary_pair_gate_scale > 0.0 and boundary_pair_gate_scale > 0.0
            ):
                tet_edge_pair_state = torch.stack(
                    [
                        gather_pair(pair, tet_i, tet_j),
                        gather_pair(pair, tet_i, tet_k),
                        gather_pair(pair, tet_i, tet_l),
                        gather_pair(pair, tet_j, tet_k),
                        gather_pair(pair, tet_j, tet_l),
                        gather_pair(pair, tet_k, tet_l),
                    ],
                    dim=-2,
                )
            if edge_frame_message_scale > 0.0 and self.edge_frame_message_scale > 0.0:
                tet_edge_update = tet_edge_update + edge_frame_message_scale * self.tetra_edge_frame_to_edge(
                    torch.cat([tetra_edge_state, tet_frame_features], dim=-1)
                )
            if self.boundary_pair_gate_scale > 0.0 and boundary_pair_gate_scale > 0.0:
                tet_edge_update = tet_edge_update + self.dropout(
                    boundary_pair_gate_scale
                    * torch.tanh(self.boundary_pair_gate(torch.cat([tet_edge_update, tet_edge_pair_state], dim=-1)))
                    * tet_edge_update
                    * tet_edge_mask[..., None]
                )
            if boundary_metric_gate_scale > 0.0:
                tet_distance_logits = self.auxiliary_heads.tetra_distances(tetra_state).reshape(
                    *tetra_state.shape[:-1],
                    6,
                    self.auxiliary_heads.n_dist_bins,
                )
                tet_edge_update = apply_boundary_metric_gate(
                    tet_edge_update,
                    tet_distance_logits,
                    tet_edge_mask,
                    scale=boundary_metric_gate_scale,
                )
            if self.boundary_incidence_normalization > 0.0:
                tet_edge_update = tet_edge_update * boundary_incidence_weights(
                    tet_edge_indices,
                    tet_edge_mask,
                    num_residues=pair.shape[1],
                    strength=self.boundary_incidence_normalization,
                )[..., None]
            tet_pair_delta, tet_pair_counts = scatter_to_pair(
                tet_edge_update,
                tet_edge_indices,
                pair_shape=tuple(pair.shape),  # type: ignore[arg-type]
                edge_mask=tet_edge_mask,
                include_reverse=True,
            )
            pair_delta = pair_delta + tet_pair_delta
            pair_counts = pair_counts + tet_pair_counts
            if self.boundary_msa_feedback_scale > 0.0 or self.boundary_pair_feedback_scale > 0.0:
                (
                    tet_boundary_source_delta,
                    tet_boundary_source_counts,
                    tet_boundary_target_delta,
                    tet_boundary_target_counts,
                ) = scatter_directed_edges_to_residue(
                    tet_edge_update,
                    tet_edge_indices,
                    num_residues=pair.shape[1],
                    edge_mask=tet_edge_mask,
                )
                boundary_source_delta = boundary_source_delta + tet_boundary_source_delta
                boundary_source_counts = boundary_source_counts + tet_boundary_source_counts
                boundary_target_delta = boundary_target_delta + tet_boundary_target_delta
                boundary_target_counts = boundary_target_counts + tet_boundary_target_counts
            if boundary_readout_directionality > 0.0:
                directed_tet_pair_delta, directed_tet_pair_counts = scatter_to_pair(
                    tet_edge_update,
                    tet_edge_indices,
                    pair_shape=tuple(pair.shape),  # type: ignore[arg-type]
                    edge_mask=tet_edge_mask,
                    include_reverse=False,
                )
                directed_pair_delta = directed_pair_delta + directed_tet_pair_delta
                directed_pair_counts = directed_pair_counts + directed_tet_pair_counts

        pair_readout = pair_delta / pair_counts.clamp_min(1.0)
        if boundary_readout_directionality > 0.0:
            directed_pair_readout = directed_pair_delta / directed_pair_counts.clamp_min(1.0)
            directionality = pair_readout.new_tensor(boundary_readout_directionality)
            pair_readout = (1.0 - directionality) * pair_readout + directionality * directed_pair_readout
            pair_counts = (1.0 - directionality) * pair_counts + directionality * directed_pair_counts
        pair_readout = oriented_boundary_cochain_readout(
            pair_readout,
            pair_counts,
            pair_mask=pair_mask,
            scale=boundary_oriented_cochain_scale,
        )
        pair_readout = hodge_center_boundary_readout(
            pair_readout,
            pair_counts,
            pair_mask=pair_mask,
            scale=boundary_hodge_readout_scale,
        )
        pair_readout = edge_star_smooth_boundary_readout(
            pair_readout,
            pair_counts,
            pair_mask=pair_mask,
            scale=boundary_edge_star_readout_scale,
        )
        pair_readout = edge_star_residual_boundary_readout(
            pair_readout,
            pair_counts,
            pair_mask=pair_mask,
            scale=boundary_edge_star_residual_scale,
        )
        pair_readout = coface_degree_attenuate_pair_readout(
            pair_readout,
            pair_counts,
            self.boundary_message_degree_attenuation,
        )
        if pair_mask is not None:
            pair_readout = pair_readout * pair_mask[..., None]
        pair = pair + self.dropout(pair_update_scale * pair_readout)
        if pair_mask is not None:
            pair = pair * pair_mask[..., None]
        boundary_pair_feedback: torch.Tensor | None = None
        if self.boundary_pair_feedback_scale > 0.0 and boundary_pair_feedback_scale > 0.0:
            boundary_source_readout = boundary_source_delta / boundary_source_counts.clamp_min(1.0)
            boundary_target_readout = boundary_target_delta / boundary_target_counts.clamp_min(1.0)
            source_pair = boundary_source_readout[:, :, None, :].expand(-1, -1, pair.shape[2], -1)
            target_pair = boundary_target_readout[:, None, :, :].expand(-1, pair.shape[1], -1, -1)
            boundary_pair_input = torch.cat([pair, source_pair, target_pair], dim=-1)
            boundary_pair_feedback = self.dropout(
                boundary_pair_feedback_scale * self.boundary_to_pair_feedback(boundary_pair_input)
            )
            if pair_mask is not None:
                boundary_pair_feedback = boundary_pair_feedback * pair_mask[..., None]
            pair = pair + boundary_pair_feedback
            if pair_mask is not None:
                pair = pair * pair_mask[..., None]

        face_single_update = self.face_to_single(face_state).reshape(*face_state.shape[:-1], 3, self.c_s)
        face_residue_indices = torch.stack([i, j, k], dim=-1)
        face_residue_mask = topology.face_mask[..., None].expand(-1, -1, -1, 3)
        single_delta, single_counts = scatter_to_single(
            face_single_update,
            face_residue_indices,
            single_shape=tuple(single.shape),  # type: ignore[arg-type]
            residue_mask=face_residue_mask,
        )

        if self.use_tetra and tetra_state.numel() > 0:
            tet_i, tet_j, tet_k, tet_l = topology.tetra_indices.unbind(dim=-1)
            tet_single_update = self.tetra_to_single(tetra_state).reshape(*tetra_state.shape[:-1], 4, self.c_s)
            tet_residue_indices = torch.stack([tet_i, tet_j, tet_k, tet_l], dim=-1)
            tet_residue_mask = topology.tetra_mask[..., None].expand(-1, -1, -1, 4)
            tet_single_delta, tet_single_counts = scatter_to_single(
                tet_single_update,
                tet_residue_indices,
                single_shape=tuple(single.shape),  # type: ignore[arg-type]
                residue_mask=tet_residue_mask,
            )
            single_delta = single_delta + tet_single_delta
            single_counts = single_counts + tet_single_counts

        single_gate = torch.sigmoid(self.single_gate(self.single_norm(single)))
        single_readout = single_gate * (single_delta / single_counts.clamp_min(1.0))
        if seq_mask is not None:
            single_readout = single_readout * seq_mask[..., None]
        msa_feedback: torch.Tensor | None = None
        if self.msa_feedback_scale > 0.0 and msa_feedback_scale > 0.0:
            msa_feedback = self.dropout(msa_feedback_scale * self.single_to_msa_feedback(single_readout))
            if seq_mask is not None:
                msa_feedback = msa_feedback * seq_mask[..., None]
        if self.boundary_msa_feedback_scale > 0.0 and boundary_msa_feedback_scale > 0.0:
            boundary_source_readout = boundary_source_delta / boundary_source_counts.clamp_min(1.0)
            boundary_target_readout = boundary_target_delta / boundary_target_counts.clamp_min(1.0)
            boundary_readout = torch.cat([boundary_source_readout, boundary_target_readout], dim=-1)
            boundary_feedback = self.dropout(
                boundary_msa_feedback_scale * self.boundary_to_msa_feedback(boundary_readout)
            )
            if seq_mask is not None:
                boundary_feedback = boundary_feedback * seq_mask[..., None]
            msa_feedback = boundary_feedback if msa_feedback is None else msa_feedback + boundary_feedback
        single = single + self.dropout(single_update_scale * single_readout)
        if seq_mask is not None:
            single = single * seq_mask[..., None]

        aux = self.auxiliary_heads(face_state, tetra_state)
        aux.update(
            {
                "simplex_contact_logits": contact_logits,
                "simplex_face_indices": topology.face_indices,
                "simplex_tetra_indices": topology.tetra_indices,
                "simplex_face_mask": topology.face_mask,
                "simplex_tetra_mask": topology.tetra_mask,
                "simplex_tetra_face_slots": topology.tetra_face_slots,
                "simplex_neighbor_indices": topology.nbr_idx,
            }
        )
        if (
            self.structure_readout_scale > 0.0
            or self.structure_pair_readout_scale > 0.0
            or self.boundary_cochain_recycling_scale > 0.0
        ):
            aux["simplex_structure_pair_readout"] = pair_readout
        if self.structure_readout_scale > 0.0:
            aux["simplex_structure_single_readout"] = single_readout
        if msa_feedback is not None:
            aux["simplex_msa_feedback"] = msa_feedback
        if boundary_pair_feedback is not None:
            aux["simplex_boundary_pair_feedback"] = boundary_pair_feedback
        if triangle_attention_aux is not None:
            aux.update(triangle_attention_aux)
        return pair, single, aux

    def _triangle_attention_aux(
        self,
        face_state: torch.Tensor,
        face_indices: torch.Tensor,
        face_mask: torch.Tensor,
        tetra_state: torch.Tensor,
        tetra_indices: torch.Tensor,
        tetra_mask: torch.Tensor,
        *,
        bias_scale: float,
        value_scale: float,
    ) -> dict[str, torch.Tensor]:
        """Emit sparse triangle-attention cochains from selected face/tetra states."""

        batch_size = face_state.shape[0]
        indices = [face_indices.reshape(batch_size, -1, 3)]
        masks = [face_mask.reshape(batch_size, -1)]
        start_biases: list[torch.Tensor] = []
        end_biases: list[torch.Tensor] = []
        start_values: list[torch.Tensor] = []
        end_values: list[torch.Tensor] = []
        if self.triangle_attention_bias_scale > 0.0 and bias_scale > 0.0:
            face_bias = torch.tanh(self.face_to_triangle_attention_bias(face_state))
            face_bias = face_bias.reshape(*face_state.shape[:-1], 2, self.triangle_num_heads)
            start_biases.append(face_bias[..., 0, :].reshape(batch_size, -1, self.triangle_num_heads))
            end_biases.append(face_bias[..., 1, :].reshape(batch_size, -1, self.triangle_num_heads))
        if self.triangle_attention_value_scale > 0.0 and value_scale > 0.0:
            face_value = torch.tanh(self.face_to_triangle_attention_value(face_state))
            face_value = face_value.reshape(*face_state.shape[:-1], 2, self.c_z)
            start_values.append(face_value[..., 0, :].reshape(batch_size, -1, self.c_z))
            end_values.append(face_value[..., 1, :].reshape(batch_size, -1, self.c_z))

        if self.use_tetra and tetra_state.numel() > 0 and tetra_indices.shape[2] > 0:
            tet_i, tet_j, tet_k, tet_l = tetra_indices.unbind(dim=-1)
            tetra_face_indices = torch.stack(
                [
                    torch.stack([tet_i, tet_j, tet_k], dim=-1),
                    torch.stack([tet_i, tet_j, tet_l], dim=-1),
                    torch.stack([tet_i, tet_k, tet_l], dim=-1),
                    torch.stack([tet_j, tet_k, tet_l], dim=-1),
                ],
                dim=-2,
            )
            tetra_face_mask = tetra_mask[..., None].expand(*tetra_face_indices.shape[:-1])
            indices.append(tetra_face_indices.reshape(batch_size, -1, 3))
            masks.append(tetra_face_mask.reshape(batch_size, -1))
            if self.triangle_attention_bias_scale > 0.0 and bias_scale > 0.0:
                tetra_bias = torch.tanh(self.tetra_to_triangle_attention_bias(tetra_state))
                tetra_bias = tetra_bias.reshape(*tetra_state.shape[:-1], 1, 2, self.triangle_num_heads)
                tetra_bias = tetra_bias.expand(*tetra_face_indices.shape[:-1], 2, self.triangle_num_heads)
                start_biases.append(tetra_bias[..., 0, :].reshape(batch_size, -1, self.triangle_num_heads))
                end_biases.append(tetra_bias[..., 1, :].reshape(batch_size, -1, self.triangle_num_heads))
            if self.triangle_attention_value_scale > 0.0 and value_scale > 0.0:
                tetra_value = torch.tanh(self.tetra_to_triangle_attention_value(tetra_state))
                tetra_value = tetra_value.reshape(*tetra_state.shape[:-1], 1, 2, self.c_z)
                tetra_value = tetra_value.expand(*tetra_face_indices.shape[:-1], 2, self.c_z)
                start_values.append(tetra_value[..., 0, :].reshape(batch_size, -1, self.c_z))
                end_values.append(tetra_value[..., 1, :].reshape(batch_size, -1, self.c_z))

        aux = {
            "simplex_triangle_attention_indices": torch.cat(indices, dim=1),
            "simplex_triangle_attention_mask": torch.cat(masks, dim=1),
        }
        if start_biases:
            bias_scale_tensor = face_state.new_tensor(float(bias_scale))
            aux["simplex_triangle_attention_start_bias"] = bias_scale_tensor * torch.cat(start_biases, dim=1)
            aux["simplex_triangle_attention_end_bias"] = bias_scale_tensor * torch.cat(end_biases, dim=1)
        if start_values:
            value_scale_tensor = face_state.new_tensor(float(value_scale))
            aux["simplex_triangle_attention_start_value"] = value_scale_tensor * torch.cat(start_values, dim=1)
            aux["simplex_triangle_attention_end_value"] = value_scale_tensor * torch.cat(end_values, dim=1)
        return aux

    def _apply_global_context(
        self,
        face_state: torch.Tensor,
        face_indices: torch.Tensor,
        face_mask: torch.Tensor,
        tetra_state: torch.Tensor,
        tetra_indices: torch.Tensor,
        tetra_mask: torch.Tensor,
        *,
        vertex_star_context_scale: float,
        edge_star_context_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route selected-complex summary cochains back into active cells."""

        face_mean = masked_cell_mean(face_state, face_mask, self.c_face)
        tetra_mean = masked_cell_mean(tetra_state, tetra_mask, self.c_tetra)
        global_context = torch.cat([face_mean, tetra_mean], dim=-1)
        face_context = global_context[:, None, None, :].expand(*face_state.shape[:-1], -1)
        if vertex_star_context_scale > 0.0:
            face_star = vertex_star_cell_mean(
                face_state,
                face_indices,
                face_mask,
                num_residues=face_state.shape[1],
                channels=self.c_face,
            )
            tetra_star = vertex_star_cell_mean(
                tetra_state,
                tetra_indices,
                tetra_mask,
                num_residues=face_state.shape[1],
                channels=self.c_tetra,
            )
            face_vertex_context = torch.cat(
                [
                    gather_single(face_star, face_indices).mean(dim=-2),
                    gather_single(tetra_star, face_indices).mean(dim=-2),
                ],
                dim=-1,
            )
            face_context = face_context + vertex_star_context_scale * (face_vertex_context - face_context)
        if edge_star_context_scale > 0.0:
            face_edges = _cell_boundary_edges(face_indices)
            face_edge_context = torch.cat(
                [
                    boundary_edge_star_context(
                        face_state,
                        face_indices,
                        face_mask,
                        face_edges,
                        num_residues=face_state.shape[1],
                        channels=self.c_face,
                    ),
                    boundary_edge_star_context(
                        tetra_state,
                        tetra_indices,
                        tetra_mask,
                        face_edges,
                        num_residues=face_state.shape[1],
                        channels=self.c_tetra,
                    ),
                ],
                dim=-1,
            )
            face_context = face_context + edge_star_context_scale * (face_edge_context - face_context)
        face_msg = self.global_to_face(torch.cat([face_state, face_context], dim=-1))
        face_state = face_state + self.dropout(
            self.global_context_scale
            * torch.sigmoid(self.face_gate(face_state))
            * face_msg
            * face_mask[..., None]
        )
        face_state = face_state * face_mask[..., None]

        if tetra_state.numel() > 0 and self.use_tetra:
            tetra_context = global_context[:, None, None, :].expand(*tetra_state.shape[:-1], -1)
            if vertex_star_context_scale > 0.0:
                tetra_vertex_context = torch.cat(
                    [
                        gather_single(face_star, tetra_indices).mean(dim=-2),
                        gather_single(tetra_star, tetra_indices).mean(dim=-2),
                    ],
                    dim=-1,
                )
                tetra_context = tetra_context + vertex_star_context_scale * (tetra_vertex_context - tetra_context)
            if edge_star_context_scale > 0.0:
                tetra_edges = _cell_boundary_edges(tetra_indices)
                tetra_edge_context = torch.cat(
                    [
                        boundary_edge_star_context(
                            face_state,
                            face_indices,
                            face_mask,
                            tetra_edges,
                            num_residues=face_state.shape[1],
                            channels=self.c_face,
                        ),
                        boundary_edge_star_context(
                            tetra_state,
                            tetra_indices,
                            tetra_mask,
                            tetra_edges,
                            num_residues=face_state.shape[1],
                            channels=self.c_tetra,
                        ),
                    ],
                    dim=-1,
                )
                tetra_context = tetra_context + edge_star_context_scale * (tetra_edge_context - tetra_context)
            tetra_msg = self.global_to_tetra(torch.cat([tetra_state, tetra_context], dim=-1))
            tetra_state = tetra_state + self.dropout(
                self.global_context_scale
                * torch.sigmoid(self.tetra_gate(tetra_state))
                * tetra_msg
                * tetra_mask[..., None]
            )
            tetra_state = tetra_state * tetra_mask[..., None]
        return face_state, tetra_state

    def _segment_pass(
        self,
        pair: torch.Tensor,
        single: torch.Tensor,
        *,
        seq_mask: Optional[torch.Tensor],
        recycled_ca_coords: Optional[torch.Tensor],
    ) -> torch.Tensor:
        b, l, _ = single.shape
        segment_indices, segment_mask = segment_cell_indices(
            batch_size=b,
            num_residues=l,
            radius=self.segment_radius,
            device=single.device,
            seq_mask=seq_mask,
        )
        mask = segment_mask.to(single.dtype)
        count = mask.sum(dim=-1).clamp_min(1.0)
        segment_single = gather_single(single, segment_indices)
        pooled_single = (segment_single * mask[..., None]).sum(dim=-2) / count[..., None]
        anchors = torch.arange(l, device=single.device).reshape(1, l, 1).expand(b, -1, segment_indices.shape[-1])
        segment_pair = gather_pair(pair, anchors, segment_indices)
        pooled_pair = (segment_pair * mask[..., None]).sum(dim=-2) / count[..., None]
        geom = segment_geometry_features(
            segment_indices,
            segment_mask,
            recycled_ca_coords=recycled_ca_coords,
            sequence_max=self.sequence_max,
            distance_max=self.distance_max,
        ).to(single.dtype)
        segment_state = self.segment_init(torch.cat([single, pooled_single, pooled_pair, geom], dim=-1))
        anchor_mask = (count > 0).to(single.dtype)
        return segment_state * anchor_mask[..., None]

    def _msa_to_face_update(
        self,
        msa_representation: torch.Tensor,
        face_indices: torch.Tensor,
        face_mask: torch.Tensor,
        *,
        msa_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        i, j, k = face_indices.unbind(dim=-1)
        proj_i = self.msa_to_face_a(msa_representation)
        proj_j = self.msa_to_face_b(msa_representation)
        proj_k = self.msa_to_face_c(msa_representation)
        mi = gather_msa_columns(proj_i, i)
        mj = gather_msa_columns(proj_j, j)
        mk = gather_msa_columns(proj_k, k)
        moment = mi * mj * mk
        if msa_mask is not None:
            mask_i = gather_msa_columns(msa_mask[..., None].to(moment.dtype), i).squeeze(-1)
            mask_j = gather_msa_columns(msa_mask[..., None].to(moment.dtype), j).squeeze(-1)
            mask_k = gather_msa_columns(msa_mask[..., None].to(moment.dtype), k).squeeze(-1)
            moment_mask = mask_i * mask_j * mask_k
            moment = (moment * moment_mask[..., None]).sum(dim=1) / moment_mask.sum(dim=1).clamp_min(1.0)[..., None]
        else:
            moment = moment.mean(dim=1)
        return self.msa_to_face(moment) * face_mask[..., None]

    def _tetra_pass(
        self,
        pair: torch.Tensor,
        single: torch.Tensor,
        face_state: torch.Tensor,
        topology: SimplexTopology,
        recycled_ca_coords: Optional[torch.Tensor],
        *,
        outer_edge_context_scale: float,
        outer_edge_residual_context_scale: float,
        boundary_incidence_normalization: float,
    ) -> torch.Tensor:
        tet_i, tet_j, tet_k, tet_l = topology.tetra_indices.unbind(dim=-1)
        z_ij = gather_pair(pair, tet_i, tet_j)
        z_ik = gather_pair(pair, tet_i, tet_k)
        z_il = gather_pair(pair, tet_i, tet_l)
        z_jk = gather_pair(pair, tet_j, tet_k)
        z_jl = gather_pair(pair, tet_j, tet_l)
        z_kl = gather_pair(pair, tet_k, tet_l)
        s_i = gather_single(single, tet_i)
        s_j = gather_single(single, tet_j)
        s_k = gather_single(single, tet_k)
        s_l = gather_single(single, tet_l)

        face_slots = topology.tetra_face_slots
        f_ijk = face_state[:, :, face_slots[:, 0], :]
        f_ijl = face_state[:, :, face_slots[:, 1], :]
        f_ikl = face_state[:, :, face_slots[:, 2], :]
        tet_geom = tetra_geometry_features(
            topology.tetra_indices,
            recycled_ca_coords=recycled_ca_coords,
            rbf_bins=self.rbf_bins,
            sequence_max=self.sequence_max,
            distance_max=self.distance_max,
            area_max=self.area_max,
            volume_scale=self.volume_scale,
        ).to(pair.dtype)
        tetra_state = self.tetra_init(
            torch.cat(
                [
                    z_ij,
                    z_ik,
                    z_il,
                    z_jk,
                    z_jl,
                    z_kl,
                    f_ijk,
                    f_ijl,
                    f_ikl,
                    s_i,
                    s_j,
                    s_k,
                    s_l,
                    tet_geom,
                ],
                dim=-1,
            )
        )
        tetra_state = tetra_state * topology.tetra_mask[..., None]
        tetra_msg = self.face_to_tetra(torch.cat([tetra_state, f_ijk, f_ijl, f_ikl], dim=-1))
        if boundary_incidence_normalization > 0.0:
            tet_edge_indices = torch.stack(
                [
                    torch.stack([tet_i, tet_j], dim=-1),
                    torch.stack([tet_i, tet_k], dim=-1),
                    torch.stack([tet_i, tet_l], dim=-1),
                    torch.stack([tet_j, tet_k], dim=-1),
                    torch.stack([tet_j, tet_l], dim=-1),
                    torch.stack([tet_k, tet_l], dim=-1),
                ],
                dim=-2,
            )
            tet_edge_mask = topology.tetra_mask[..., None].expand(-1, -1, -1, 6)
            tetra_msg = tetra_msg * _cell_boundary_incidence_weights(
                tet_edge_indices,
                tet_edge_mask,
                num_residues=pair.shape[1],
                strength=boundary_incidence_normalization,
            )[..., None]
        tetra_state = tetra_state + self.dropout(
            torch.sigmoid(self.tetra_gate(tetra_state)) * tetra_msg * topology.tetra_mask[..., None]
        )
        if outer_edge_context_scale > 0.0:
            outer_context = cell_outer_edge_context(pair, topology.tetra_indices, topology.tetra_mask, topology.nbr_idx)
            outer_msg = self.tetra_outer_edge_context(torch.cat([tetra_state, outer_context], dim=-1))
            tetra_state = tetra_state + self.dropout(
                outer_edge_context_scale
                * torch.sigmoid(self.tetra_gate(tetra_state))
                * outer_msg
                * topology.tetra_mask[..., None]
            )
            tetra_state = tetra_state * topology.tetra_mask[..., None]
        if outer_edge_residual_context_scale > 0.0:
            outer_msg = parameter_free_outer_edge_context_delta(
                pair,
                topology.tetra_indices,
                topology.tetra_mask,
                topology.nbr_idx,
                channels=self.c_tetra,
            )
            outer_msg = _rms_match_update(outer_msg, tetra_state)
            tetra_state = tetra_state + self.dropout(
                outer_edge_residual_context_scale
                * torch.sigmoid(self.tetra_gate(tetra_state))
                * outer_msg
                * topology.tetra_mask[..., None]
            )
            tetra_state = tetra_state * topology.tetra_mask[..., None]
        return tetra_state


class SimplexGeometryLoss(torch.nn.Module):
    """Auxiliary losses for topology contact logits and simplex geometry heads."""

    def __init__(
        self,
        *,
        contact_distance_threshold: float = 8.0,
        contact_weight: float = 0.05,
        topology_neighborhood_weight: float = 0.05,
        topology_margin_weight: float = 0.0,
        topology_margin: float = 1.0,
        topology_margin_hard_negatives: int = 8,
        face_area_weight: float = 0.05,
        face_coordinate_weight: float = 0.1,
        face_coordinate_distance_weight: float = 0.0,
        face_coordinate_expansion_weight: float = 0.0,
        face_centroid_expansion_weight: float = 0.0,
        face_shape_weight: float = 0.0,
        face_normal_weight: float = 0.0,
        face_boundary_lddt_weight: float = 0.0,
        face_distance_weight: float = 0.05,
        tetra_geometry_weight: float = 0.02,
        tetra_coordinate_weight: float = 0.1,
        tetra_coordinate_distance_weight: float = 0.0,
        tetra_coordinate_expansion_weight: float = 0.0,
        tetra_centroid_expansion_weight: float = 0.0,
        tetra_shape_weight: float = 0.0,
        tetra_boundary_lddt_weight: float = 0.0,
        tetra_distance_weight: float = 0.05,
        pair_face_consistency_weight: float = 0.02,
        face_tetra_consistency_weight: float = 0.02,
        boundary_degree_normalize: bool = False,
        cell_closure_weight: float = 0.0,
        cell_closure_cutoff: float = 15.0,
        cell_closure_temperature: float = 2.0,
        coordinate_expansion_tolerance: float = 0.0,
        distance_scale: float = 32.0,
        volume_scale: float = 1000.0,
        area_scale: float = 300.0,
    ):
        super().__init__()
        self.contact_distance_threshold = contact_distance_threshold
        self.contact_weight = contact_weight
        self.topology_neighborhood_weight = topology_neighborhood_weight
        self.topology_margin_weight = topology_margin_weight
        self.topology_margin = topology_margin
        self.topology_margin_hard_negatives = topology_margin_hard_negatives
        self.face_area_weight = face_area_weight
        self.face_coordinate_weight = face_coordinate_weight
        self.face_coordinate_distance_weight = face_coordinate_distance_weight
        self.face_coordinate_expansion_weight = face_coordinate_expansion_weight
        self.face_centroid_expansion_weight = face_centroid_expansion_weight
        self.face_shape_weight = face_shape_weight
        self.face_normal_weight = face_normal_weight
        self.face_boundary_lddt_weight = face_boundary_lddt_weight
        self.face_distance_weight = face_distance_weight
        self.tetra_geometry_weight = tetra_geometry_weight
        self.tetra_coordinate_weight = tetra_coordinate_weight
        self.tetra_coordinate_distance_weight = tetra_coordinate_distance_weight
        self.tetra_coordinate_expansion_weight = tetra_coordinate_expansion_weight
        self.tetra_centroid_expansion_weight = tetra_centroid_expansion_weight
        self.tetra_shape_weight = tetra_shape_weight
        self.tetra_boundary_lddt_weight = tetra_boundary_lddt_weight
        self.tetra_distance_weight = tetra_distance_weight
        self.pair_face_consistency_weight = pair_face_consistency_weight
        self.face_tetra_consistency_weight = face_tetra_consistency_weight
        self.boundary_degree_normalize = boundary_degree_normalize
        self.cell_closure_weight = cell_closure_weight
        self.cell_closure_cutoff = cell_closure_cutoff
        self.cell_closure_temperature = cell_closure_temperature
        self.coordinate_expansion_tolerance = coordinate_expansion_tolerance
        self.distance_scale = distance_scale
        self.volume_scale = volume_scale
        self.area_scale = area_scale

    def forward(
        self,
        prediction: dict[str, torch.Tensor],
        true_ca: torch.Tensor,
        ca_mask: torch.Tensor,
        *,
        seq_mask: Optional[torch.Tensor] = None,
        true_atom_positions: Optional[torch.Tensor] = None,
        true_atom_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        b, l, _ = true_ca.shape
        device = true_ca.device
        dtype = true_ca.dtype
        valid_res = ca_mask
        if seq_mask is not None:
            valid_res = valid_res * seq_mask.to(dtype)
        pair_valid = valid_res[:, :, None] * valid_res[:, None, :]
        eye = torch.eye(l, device=device, dtype=dtype)[None, :, :]
        pair_valid = pair_valid * (1.0 - eye)
        true_global_center = (true_ca * valid_res[..., None]).sum(dim=1) / valid_res.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0)

        loss_terms: dict[str, torch.Tensor] = {}
        total = true_ca.new_zeros((b,))

        if "simplex_contact_logits" in prediction:
            contact_logits = prediction["simplex_contact_logits"]
            with torch.no_grad():
                distances = torch.cdist(true_ca, true_ca)
                contact_true = (distances < self.contact_distance_threshold).to(dtype)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                contact_logits,
                contact_true,
                reduction="none",
            )
            positive_mask = pair_valid * contact_true
            negative_mask = pair_valid * (1.0 - contact_true)
            positive_count_by_row = positive_mask.sum(dim=-1)
            negative_count_by_row = negative_mask.sum(dim=-1)
            positive_count = positive_count_by_row.sum(dim=1)
            negative_count = negative_mask.sum(dim=(1, 2))
            positive_loss = (bce * positive_mask).sum(dim=(1, 2)) / positive_count.clamp_min(1.0)
            negative_loss = (bce * negative_mask).sum(dim=(1, 2)) / negative_count.clamp_min(1.0)
            has_positive = (positive_count > 0).to(dtype)
            has_negative = (negative_count > 0).to(dtype)
            contact_loss = (
                positive_loss * has_positive + negative_loss * has_negative
            ) / (has_positive + has_negative).clamp_min(1.0)
            loss_terms["simplex_contact_loss"] = contact_loss
            weighted = self.contact_weight * contact_loss
            loss_terms["weighted_simplex_contact_loss"] = weighted
            total = total + weighted

            very_negative = torch.finfo(contact_logits.dtype).min / 4
            masked_logits = contact_logits.masked_fill(pair_valid <= 0, very_negative)
            log_probs = torch.nn.functional.log_softmax(masked_logits, dim=-1)
            positive_distribution = positive_mask / positive_count_by_row[..., None].clamp_min(1.0)
            row_loss = -(positive_distribution * log_probs).sum(dim=-1)
            row_has_positive = (positive_count_by_row > 0).to(dtype)
            topology_neighborhood_loss = (row_loss * row_has_positive).sum(dim=1) / row_has_positive.sum(
                dim=1
            ).clamp_min(1.0)
            loss_terms["simplex_topology_neighborhood_loss"] = topology_neighborhood_loss
            weighted = self.topology_neighborhood_weight * topology_neighborhood_loss
            loss_terms["weighted_simplex_topology_neighborhood_loss"] = weighted
            total = total + weighted

            hard_negative_count = min(max(int(self.topology_margin_hard_negatives), 1), max(l - 1, 1))
            positive_logits = contact_logits.masked_fill(positive_mask <= 0, very_negative)
            positive_energy = torch.logsumexp(positive_logits, dim=-1) - torch.log(
                positive_count_by_row.clamp_min(1.0)
            )
            negative_logits = contact_logits.masked_fill(negative_mask <= 0, very_negative)
            hard_negative_logits = torch.topk(negative_logits, k=hard_negative_count, dim=-1).values
            hard_negative_slots = torch.arange(hard_negative_count, device=device)
            hard_negative_valid = hard_negative_slots[None, None, :] < negative_count_by_row[:, :, None]
            margin_losses = torch.nn.functional.softplus(
                float(self.topology_margin) + hard_negative_logits - positive_energy[..., None]
            )
            margin_losses = margin_losses * hard_negative_valid.to(dtype)
            row_margin_loss = margin_losses.sum(dim=-1) / hard_negative_valid.to(dtype).sum(dim=-1).clamp_min(1.0)
            row_has_margin = ((positive_count_by_row > 0) & (negative_count_by_row > 0)).to(dtype)
            topology_margin_loss = (row_margin_loss * row_has_margin).sum(dim=1) / row_has_margin.sum(
                dim=1
            ).clamp_min(1.0)
            loss_terms["simplex_topology_margin_loss"] = topology_margin_loss
            weighted = self.topology_margin_weight * topology_margin_loss
            loss_terms["weighted_simplex_topology_margin_loss"] = weighted
            total = total + weighted

        if (
            "simplex_face_area_logits" in prediction
            and "simplex_face_indices" in prediction
            and prediction["simplex_face_indices"].shape[2] > 0
        ):
            face_indices = prediction["simplex_face_indices"]
            face_mask = prediction["simplex_face_mask"].to(dtype)
            i, j, k = face_indices.unbind(dim=-1)
            x_i = gather_single(true_ca, i)
            x_j = gather_single(true_ca, j)
            x_k = gather_single(true_ca, k)
            true_area = _triangle_area(x_i, x_j, x_k)
            true_face_distances = torch.stack(
                [
                    _safe_norm(x_j - x_i),
                    _safe_norm(x_k - x_i),
                    _safe_norm(x_k - x_j),
                ],
                dim=-1,
            )
            true_face_points = torch.stack([x_i, x_j, x_k], dim=-2)
            denom = torch.log1p(torch.as_tensor(self.area_scale, device=device, dtype=dtype)).clamp_min(1.0)
            target_area = torch.log1p(true_area) / denom
            pred_area = prediction["simplex_face_area_logits"].to(dtype)
            valid = face_mask * gather_single(valid_res, i) * gather_single(valid_res, j) * gather_single(valid_res, k)
            coordinate_valid = _cell_closure_weighted_mask(
                valid,
                true_face_distances,
                closure_weight=self.cell_closure_weight,
                cutoff=self.cell_closure_cutoff,
                temperature=self.cell_closure_temperature,
            )
            face_loss = (torch.nn.functional.smooth_l1_loss(pred_area, target_area, reduction="none") * valid).sum(
                dim=(1, 2)
            ) / valid.sum(dim=(1, 2)).clamp_min(1.0)
            loss_terms["simplex_face_area_loss"] = face_loss
            weighted = self.face_area_weight * face_loss
            loss_terms["weighted_simplex_face_area_loss"] = weighted
            total = total + weighted

            if "atom14_coords" in prediction:
                pred_ca = prediction["atom14_coords"][:, :, 1, :].to(dtype)
                pred_i = gather_single(pred_ca, i)
                pred_j = gather_single(pred_ca, j)
                pred_k = gather_single(pred_ca, k)
                pred_global_center = (pred_ca * valid_res[..., None]).sum(dim=1) / valid_res.sum(
                    dim=1,
                    keepdim=True,
                ).clamp_min(1.0)
                pred_face_points = torch.stack([pred_i, pred_j, pred_k], dim=-2)
                pred_area_from_coords = torch.log1p(_triangle_area(pred_i, pred_j, pred_k)) / denom
                distance_denom = torch.log1p(
                    torch.as_tensor(self.distance_scale, device=device, dtype=dtype)
                ).clamp_min(1.0)
                pred_face_distances_raw = torch.stack(
                    [
                        _safe_norm(pred_j - pred_i),
                        _safe_norm(pred_k - pred_i),
                        _safe_norm(pred_k - pred_j),
                    ],
                    dim=-1,
                )
                target_face_distances = torch.log1p(true_face_distances) / distance_denom
                pred_face_distances = torch.log1p(pred_face_distances_raw) / distance_denom
                face_distance_valid = coordinate_valid[..., None].expand_as(true_face_distances)
                face_edge_indices = torch.stack(
                    [
                        torch.stack([i, j], dim=-1),
                        torch.stack([i, k], dim=-1),
                        torch.stack([j, k], dim=-1),
                    ],
                    dim=-2,
                )
                if self.boundary_degree_normalize:
                    face_distance_valid = _boundary_degree_weights(
                        face_edge_indices,
                        face_distance_valid,
                        num_residues=l,
                    )
                face_coordinate_loss = (
                    torch.nn.functional.smooth_l1_loss(pred_area_from_coords, target_area, reduction="none")
                    * coordinate_valid
                ).sum(dim=(1, 2)) / coordinate_valid.sum(dim=(1, 2)).clamp_min(1.0)
                loss_terms["simplex_face_coordinate_area_loss"] = face_coordinate_loss
                weighted = self.face_coordinate_weight * face_coordinate_loss
                loss_terms["weighted_simplex_face_coordinate_area_loss"] = weighted
                total = total + weighted

                face_coordinate_distance_loss = (
                    torch.nn.functional.smooth_l1_loss(
                        pred_face_distances,
                        target_face_distances,
                        reduction="none",
                    )
                    * face_distance_valid
                ).sum(dim=(1, 2, 3)) / face_distance_valid.sum(dim=(1, 2, 3)).clamp_min(1.0)
                loss_terms["simplex_face_coordinate_distance_loss"] = face_coordinate_distance_loss
                weighted = self.face_coordinate_distance_weight * face_coordinate_distance_loss
                loss_terms["weighted_simplex_face_coordinate_distance_loss"] = weighted
                total = total + weighted

                face_coordinate_expansion_loss = _selected_boundary_contraction_loss(
                    pred_face_distances_raw,
                    true_face_distances,
                    face_distance_valid,
                    distance_scale=self.distance_scale,
                    tolerance=self.coordinate_expansion_tolerance,
                )
                loss_terms["simplex_face_coordinate_expansion_loss"] = face_coordinate_expansion_loss
                weighted = self.face_coordinate_expansion_weight * face_coordinate_expansion_loss
                loss_terms["weighted_simplex_face_coordinate_expansion_loss"] = weighted
                total = total + weighted

                face_centroid_expansion_loss = _selected_cell_centroid_contraction_loss(
                    pred_face_points,
                    true_face_points,
                    pred_global_center,
                    true_global_center,
                    coordinate_valid,
                    distance_scale=self.distance_scale,
                    tolerance=self.coordinate_expansion_tolerance,
                )
                loss_terms["simplex_face_centroid_expansion_loss"] = face_centroid_expansion_loss
                weighted = self.face_centroid_expansion_weight * face_centroid_expansion_loss
                loss_terms["weighted_simplex_face_centroid_expansion_loss"] = weighted
                total = total + weighted

                if self.face_shape_weight > 0.0:
                    face_shape_loss = _selected_simplex_shape_loss(
                        pred_face_points,
                        true_face_points,
                        coordinate_valid,
                        length_scale=self.distance_scale,
                    )
                    loss_terms["simplex_face_shape_loss"] = face_shape_loss
                    weighted = self.face_shape_weight * face_shape_loss
                    loss_terms["weighted_simplex_face_shape_loss"] = weighted
                    total = total + weighted

                if (
                    true_atom_positions is not None
                    and true_atom_mask is not None
                    and "atom14_mask" in prediction
                ):
                    pred_atom14 = prediction["atom14_coords"].to(dtype)
                    pred_atom14_mask = prediction["atom14_mask"].to(dtype)
                    true_atom14 = true_atom_positions.to(dtype)
                    true_atom14_mask = true_atom_mask.to(dtype)
                    true_frames = _backbone_frames_from_atom14(true_atom14)
                    pred_frames = _backbone_frames_from_atom14(pred_atom14)
                    true_normal = torch.cross(x_j - x_i, x_k - x_i, dim=-1)
                    true_normal = true_normal / _safe_norm(true_normal)[..., None].clamp_min(1e-8)
                    pred_normal = torch.cross(pred_j - pred_i, pred_k - pred_i, dim=-1)
                    pred_normal = pred_normal / _safe_norm(pred_normal)[..., None].clamp_min(1e-8)
                    true_frame_i = gather_single(true_frames, i)
                    true_frame_j = gather_single(true_frames, j)
                    true_frame_k = gather_single(true_frames, k)
                    pred_frame_i = gather_single(pred_frames, i)
                    pred_frame_j = gather_single(pred_frames, j)
                    pred_frame_k = gather_single(pred_frames, k)
                    true_local_normal = torch.stack(
                        [
                            _express_in_frame(true_frame_i, true_normal),
                            _express_in_frame(true_frame_j, true_normal),
                            _express_in_frame(true_frame_k, true_normal),
                        ],
                        dim=-2,
                    )
                    pred_local_normal = torch.stack(
                        [
                            _express_in_frame(pred_frame_i, pred_normal),
                            _express_in_frame(pred_frame_j, pred_normal),
                            _express_in_frame(pred_frame_k, pred_normal),
                        ],
                        dim=-2,
                    )
                    normal_dot = torch.sum(pred_local_normal * true_local_normal, dim=-1).clamp(-1.0, 1.0)
                    normal_loss_per_frame = 0.5 * (1.0 - normal_dot)
                    true_backbone_mask = (
                        true_atom14_mask[..., 0] * true_atom14_mask[..., 1] * true_atom14_mask[..., 2]
                    )
                    pred_backbone_mask = (
                        pred_atom14_mask[..., 0] * pred_atom14_mask[..., 1] * pred_atom14_mask[..., 2]
                    )
                    normal_valid = coordinate_valid * (true_area > 1e-4).to(dtype)
                    normal_frame_valid = torch.stack(
                        [
                            normal_valid * gather_single(true_backbone_mask, i) * gather_single(pred_backbone_mask, i),
                            normal_valid * gather_single(true_backbone_mask, j) * gather_single(pred_backbone_mask, j),
                            normal_valid * gather_single(true_backbone_mask, k) * gather_single(pred_backbone_mask, k),
                        ],
                        dim=-1,
                    )
                    face_normal_loss = (normal_loss_per_frame * normal_frame_valid).sum(dim=(1, 2, 3)) / (
                        normal_frame_valid.sum(dim=(1, 2, 3)).clamp_min(1.0)
                    )
                    loss_terms["simplex_face_normal_loss"] = face_normal_loss
                    weighted = self.face_normal_weight * face_normal_loss
                    loss_terms["weighted_simplex_face_normal_loss"] = weighted
                    total = total + weighted

                face_boundary_lddt_loss = _selected_boundary_lddt_loss(
                    pred_face_distances_raw,
                    true_face_distances,
                    face_distance_valid,
                )
                loss_terms["simplex_face_boundary_lddt_loss"] = face_boundary_lddt_loss
                weighted = self.face_boundary_lddt_weight * face_boundary_lddt_loss
                loss_terms["weighted_simplex_face_boundary_lddt_loss"] = weighted
                total = total + weighted

            if "simplex_face_distance_logits" in prediction:
                face_distance_logits = prediction["simplex_face_distance_logits"]
                target_bins = _distance_bin_indices(true_face_distances, n_bins=face_distance_logits.shape[-1])
                distance_mask = valid[..., None].expand_as(true_face_distances)
                if self.boundary_degree_normalize:
                    face_edge_indices = torch.stack(
                        [
                            torch.stack([i, j], dim=-1),
                            torch.stack([i, k], dim=-1),
                            torch.stack([j, k], dim=-1),
                        ],
                        dim=-2,
                    )
                    distance_mask = _boundary_degree_weights(face_edge_indices, distance_mask, num_residues=l)
                face_distance_loss = _masked_cross_entropy_from_bins(
                    face_distance_logits,
                    target_bins,
                    distance_mask,
                )
                loss_terms["simplex_face_distance_loss"] = face_distance_loss
                weighted = self.face_distance_weight * face_distance_loss
                loss_terms["weighted_simplex_face_distance_loss"] = weighted
                total = total + weighted

                if "distogram_logits" in prediction and prediction["distogram_logits"].shape[-1] == face_distance_logits.shape[-1]:
                    pair_logits = prediction["distogram_logits"]
                    edge_pair_logits = torch.stack(
                        [
                            gather_pair(pair_logits, i, j),
                            gather_pair(pair_logits, i, k),
                            gather_pair(pair_logits, j, k),
                        ],
                        dim=-2,
                    )
                    pair_face_consistency_loss = _masked_symmetric_kl(
                        face_distance_logits,
                        edge_pair_logits,
                        distance_mask,
                    )
                    loss_terms["simplex_pair_face_consistency_loss"] = pair_face_consistency_loss
                    weighted = self.pair_face_consistency_weight * pair_face_consistency_loss
                    loss_terms["weighted_simplex_pair_face_consistency_loss"] = weighted
                    total = total + weighted

        if (
            "simplex_tetra_geometry_logits" in prediction
            and "simplex_tetra_indices" in prediction
            and prediction["simplex_tetra_indices"].shape[2] > 0
            and prediction["simplex_tetra_geometry_logits"].shape[2]
            == prediction["simplex_tetra_indices"].shape[2]
        ):
            tetra_indices = prediction["simplex_tetra_indices"]
            tetra_mask = prediction["simplex_tetra_mask"].to(dtype)
            i, j, k, l_idx = tetra_indices.unbind(dim=-1)
            x_i = gather_single(true_ca, i)
            x_j = gather_single(true_ca, j)
            x_k = gather_single(true_ca, k)
            x_l = gather_single(true_ca, l_idx)
            true_tetra_distances = torch.stack(
                [
                    _safe_norm(x_j - x_i),
                    _safe_norm(x_k - x_i),
                    _safe_norm(x_l - x_i),
                    _safe_norm(x_k - x_j),
                    _safe_norm(x_l - x_j),
                    _safe_norm(x_l - x_k),
                ],
                dim=-1,
            )
            signed_volume = torch.sum(torch.cross(x_j - x_i, x_k - x_i, dim=-1) * (x_l - x_i), dim=-1) / 6.0
            points = torch.stack([x_i, x_j, x_k, x_l], dim=-2)
            center = points.mean(dim=-2, keepdim=True)
            rg = torch.sqrt(torch.mean(torch.sum((points - center) ** 2, dim=-1), dim=-1).clamp_min(1e-8))
            volume_denom = torch.log1p(torch.as_tensor(self.volume_scale, device=device, dtype=dtype)).clamp_min(1.0)
            target = torch.stack(
                [
                    torch.sign(signed_volume) * torch.log1p(torch.abs(signed_volume)) / volume_denom,
                    torch.log1p(torch.abs(signed_volume)) / volume_denom,
                    torch.log1p(rg) / torch.log1p(torch.as_tensor(32.0, device=device, dtype=dtype)).clamp_min(1.0),
                ],
                dim=-1,
            )
            pred = prediction["simplex_tetra_geometry_logits"].to(dtype)
            valid = (
                tetra_mask
                * gather_single(valid_res, i)
                * gather_single(valid_res, j)
                * gather_single(valid_res, k)
                * gather_single(valid_res, l_idx)
            )
            coordinate_valid = _cell_closure_weighted_mask(
                valid,
                true_tetra_distances,
                closure_weight=self.cell_closure_weight,
                cutoff=self.cell_closure_cutoff,
                temperature=self.cell_closure_temperature,
            )
            tet_loss_per = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none").mean(dim=-1)
            tetra_loss = (tet_loss_per * valid).sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp_min(1.0)
            loss_terms["simplex_tetra_geometry_loss"] = tetra_loss
            weighted = self.tetra_geometry_weight * tetra_loss
            loss_terms["weighted_simplex_tetra_geometry_loss"] = weighted
            total = total + weighted

            if "atom14_coords" in prediction:
                pred_ca = prediction["atom14_coords"][:, :, 1, :].to(dtype)
                pred_i = gather_single(pred_ca, i)
                pred_j = gather_single(pred_ca, j)
                pred_k = gather_single(pred_ca, k)
                pred_l = gather_single(pred_ca, l_idx)
                pred_signed_volume = (
                    torch.sum(torch.cross(pred_j - pred_i, pred_k - pred_i, dim=-1) * (pred_l - pred_i), dim=-1)
                    / 6.0
                )
                pred_tetra_distances_raw = torch.stack(
                    [
                        _safe_norm(pred_j - pred_i),
                        _safe_norm(pred_k - pred_i),
                        _safe_norm(pred_l - pred_i),
                        _safe_norm(pred_k - pred_j),
                        _safe_norm(pred_l - pred_j),
                        _safe_norm(pred_l - pred_k),
                    ],
                    dim=-1,
                )
                pred_points = torch.stack([pred_i, pred_j, pred_k, pred_l], dim=-2)
                pred_global_center = (pred_ca * valid_res[..., None]).sum(dim=1) / valid_res.sum(
                    dim=1,
                    keepdim=True,
                ).clamp_min(1.0)
                pred_center = pred_points.mean(dim=-2, keepdim=True)
                pred_rg = torch.sqrt(
                    torch.mean(torch.sum((pred_points - pred_center) ** 2, dim=-1), dim=-1).clamp_min(1e-8)
                )
                pred_geometry = torch.stack(
                    [
                        torch.sign(pred_signed_volume)
                        * torch.log1p(torch.abs(pred_signed_volume))
                        / volume_denom,
                        torch.log1p(torch.abs(pred_signed_volume)) / volume_denom,
                        torch.log1p(pred_rg)
                        / torch.log1p(torch.as_tensor(32.0, device=device, dtype=dtype)).clamp_min(1.0),
                    ],
                    dim=-1,
                )
                tetra_coordinate_per = torch.nn.functional.smooth_l1_loss(
                    pred_geometry,
                    target,
                    reduction="none",
                ).mean(dim=-1)
                tetra_coordinate_loss = (tetra_coordinate_per * coordinate_valid).sum(
                    dim=(1, 2)
                ) / coordinate_valid.sum(dim=(1, 2)).clamp_min(1.0)
                loss_terms["simplex_tetra_coordinate_geometry_loss"] = tetra_coordinate_loss
                weighted = self.tetra_coordinate_weight * tetra_coordinate_loss
                loss_terms["weighted_simplex_tetra_coordinate_geometry_loss"] = weighted
                total = total + weighted

                distance_denom = torch.log1p(
                    torch.as_tensor(self.distance_scale, device=device, dtype=dtype)
                ).clamp_min(1.0)
                target_tetra_distances = torch.log1p(true_tetra_distances) / distance_denom
                pred_tetra_distances = torch.log1p(pred_tetra_distances_raw) / distance_denom
                tetra_distance_valid = coordinate_valid[..., None].expand_as(true_tetra_distances)
                tetra_edge_indices = torch.stack(
                    [
                        torch.stack([i, j], dim=-1),
                        torch.stack([i, k], dim=-1),
                        torch.stack([i, l_idx], dim=-1),
                        torch.stack([j, k], dim=-1),
                        torch.stack([j, l_idx], dim=-1),
                        torch.stack([k, l_idx], dim=-1),
                    ],
                    dim=-2,
                )
                if self.boundary_degree_normalize:
                    tetra_distance_valid = _boundary_degree_weights(
                        tetra_edge_indices,
                        tetra_distance_valid,
                        num_residues=l,
                    )
                tetra_coordinate_distance_loss = (
                    torch.nn.functional.smooth_l1_loss(
                        pred_tetra_distances,
                        target_tetra_distances,
                        reduction="none",
                    )
                    * tetra_distance_valid
                ).sum(dim=(1, 2, 3)) / tetra_distance_valid.sum(dim=(1, 2, 3)).clamp_min(1.0)
                loss_terms["simplex_tetra_coordinate_distance_loss"] = tetra_coordinate_distance_loss
                weighted = self.tetra_coordinate_distance_weight * tetra_coordinate_distance_loss
                loss_terms["weighted_simplex_tetra_coordinate_distance_loss"] = weighted
                total = total + weighted

                tetra_coordinate_expansion_loss = _selected_boundary_contraction_loss(
                    pred_tetra_distances_raw,
                    true_tetra_distances,
                    tetra_distance_valid,
                    distance_scale=self.distance_scale,
                    tolerance=self.coordinate_expansion_tolerance,
                )
                loss_terms["simplex_tetra_coordinate_expansion_loss"] = tetra_coordinate_expansion_loss
                weighted = self.tetra_coordinate_expansion_weight * tetra_coordinate_expansion_loss
                loss_terms["weighted_simplex_tetra_coordinate_expansion_loss"] = weighted
                total = total + weighted

                tetra_centroid_expansion_loss = _selected_cell_centroid_contraction_loss(
                    pred_points,
                    points,
                    pred_global_center,
                    true_global_center,
                    coordinate_valid,
                    distance_scale=self.distance_scale,
                    tolerance=self.coordinate_expansion_tolerance,
                )
                loss_terms["simplex_tetra_centroid_expansion_loss"] = tetra_centroid_expansion_loss
                weighted = self.tetra_centroid_expansion_weight * tetra_centroid_expansion_loss
                loss_terms["weighted_simplex_tetra_centroid_expansion_loss"] = weighted
                total = total + weighted

                if self.tetra_shape_weight > 0.0:
                    tetra_shape_loss = _selected_simplex_shape_loss(
                        pred_points,
                        points,
                        coordinate_valid,
                        length_scale=self.distance_scale,
                    )
                    loss_terms["simplex_tetra_shape_loss"] = tetra_shape_loss
                    weighted = self.tetra_shape_weight * tetra_shape_loss
                    loss_terms["weighted_simplex_tetra_shape_loss"] = weighted
                    total = total + weighted

                tetra_boundary_lddt_loss = _selected_boundary_lddt_loss(
                    pred_tetra_distances_raw,
                    true_tetra_distances,
                    tetra_distance_valid,
                )
                loss_terms["simplex_tetra_boundary_lddt_loss"] = tetra_boundary_lddt_loss
                weighted = self.tetra_boundary_lddt_weight * tetra_boundary_lddt_loss
                loss_terms["weighted_simplex_tetra_boundary_lddt_loss"] = weighted
                total = total + weighted

            if "simplex_tetra_distance_logits" in prediction:
                tetra_distance_logits = prediction["simplex_tetra_distance_logits"]
                target_bins = _distance_bin_indices(true_tetra_distances, n_bins=tetra_distance_logits.shape[-1])
                distance_mask = valid[..., None].expand_as(true_tetra_distances)
                if self.boundary_degree_normalize:
                    tetra_edge_indices = torch.stack(
                        [
                            torch.stack([i, j], dim=-1),
                            torch.stack([i, k], dim=-1),
                            torch.stack([i, l_idx], dim=-1),
                            torch.stack([j, k], dim=-1),
                            torch.stack([j, l_idx], dim=-1),
                            torch.stack([k, l_idx], dim=-1),
                        ],
                        dim=-2,
                    )
                    distance_mask = _boundary_degree_weights(tetra_edge_indices, distance_mask, num_residues=l)
                tetra_distance_loss = _masked_cross_entropy_from_bins(
                    tetra_distance_logits,
                    target_bins,
                    distance_mask,
                )
                loss_terms["simplex_tetra_distance_loss"] = tetra_distance_loss
                weighted = self.tetra_distance_weight * tetra_distance_loss
                loss_terms["weighted_simplex_tetra_distance_loss"] = weighted
                total = total + weighted

                if (
                    "simplex_face_distance_logits" in prediction
                    and "simplex_tetra_face_slots" in prediction
                    and tetra_distance_logits.shape[-1] == prediction["simplex_face_distance_logits"].shape[-1]
                ):
                    face_logits = prediction["simplex_face_distance_logits"]
                    face_slots = prediction["simplex_tetra_face_slots"].to(device=device)
                    if face_slots.shape[0] == tetra_distance_logits.shape[2]:
                        # The anchored tetra uses the three anchored boundary
                        # faces: (i,j,k), (i,j,l), and (i,k,l). Their edge slots
                        # map to tetra edge slots [ij,ik,il,jk,jl,kl].
                        f_ijk = face_logits[:, :, face_slots[:, 0], :, :]
                        f_ijl = face_logits[:, :, face_slots[:, 1], :, :]
                        f_ikl = face_logits[:, :, face_slots[:, 2], :, :]
                        face_boundary_logits = torch.stack(
                            [
                                f_ijk[..., 0, :],
                                f_ijk[..., 1, :],
                                f_ijl[..., 1, :],
                                f_ijk[..., 2, :],
                                f_ijl[..., 2, :],
                                f_ikl[..., 2, :],
                            ],
                            dim=-2,
                        )
                        face_tetra_consistency_loss = _masked_symmetric_kl(
                            tetra_distance_logits,
                            face_boundary_logits,
                            distance_mask,
                        )
                        loss_terms["simplex_face_tetra_consistency_loss"] = face_tetra_consistency_loss
                        weighted = self.face_tetra_consistency_weight * face_tetra_consistency_loss
                        loss_terms["weighted_simplex_face_tetra_consistency_loss"] = weighted
                        total = total + weighted

        loss_terms["simplex_aux_loss"] = total
        return loss_terms
