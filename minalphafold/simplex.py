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

from .initialization import init_gate_linear, init_linear, zero_linear


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


def _triangle_area(x_i: torch.Tensor, x_j: torch.Tensor, x_k: torch.Tensor) -> torch.Tensor:
    return 0.5 * _safe_norm(torch.cross(x_j - x_i, x_k - x_i, dim=-1))


def build_simplex_topology(
    score: torch.Tensor,
    *,
    neighbor_k: int,
    seq_mask: Optional[torch.Tensor] = None,
    pair_mask: Optional[torch.Tensor] = None,
    recycled_ca_coords: Optional[torch.Tensor] = None,
    local_radius: int = 4,
    local_bias: float = 5.0,
    long_min_sep: int = 24,
    long_bias: float = 0.0,
    geometry_distance_weight: float = 0.1,
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

    if k == 0:
        nbr_idx = torch.empty((b, l, 0), device=device, dtype=torch.long)
    else:
        nbr_idx = torch.topk(work_score, k=k, dim=-1).indices

    face_combos = _combination_tensor(k, 2, device)
    tetra_combos = _combination_tensor(k, 3, device)
    tetra_face_slots = _tetra_face_slots(face_combos, tetra_combos)

    anchor = residue_ids[None, :, None].expand(b, l, -1)
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

    The adapter starts as an identity on the pair/single streams because all
    residual projection heads are zero-initialised.  Auxiliary heads still
    receive non-zero face/tetra states, so their losses can immediately train
    the simplex representations and the topology contact scorer.
    """

    def __init__(self, config):
        super().__init__()
        self.c_z = config.c_z
        self.c_s = config.c_s
        self.c_face = int(getattr(config, "simplex_c_face", 32))
        self.c_tetra = int(getattr(config, "simplex_c_tetra", 16))
        self.hidden_dim = int(getattr(config, "simplex_hidden_dim", max(config.c_z, config.c_s)))
        self.neighbor_k = int(getattr(config, "simplex_neighbor_k", 12))
        self.use_faces = bool(getattr(config, "simplex_use_faces", True))
        self.use_tetra = bool(getattr(config, "simplex_use_tetra", True))
        self.use_msa_to_face = bool(getattr(config, "simplex_use_msa_to_face", False))
        self.msa_to_face_rank = int(getattr(config, "simplex_msa_to_face_rank", 16))
        self.use_recycled_geometry = bool(getattr(config, "simplex_use_recycled_geometry", True))
        self.local_radius = int(getattr(config, "simplex_local_radius", 4))
        self.local_bias = float(getattr(config, "simplex_local_bias", 5.0))
        self.long_min_sep = int(getattr(config, "simplex_long_min_sep", 24))
        self.long_bias = float(getattr(config, "simplex_long_bias", 0.0))
        self.geometry_distance_weight = float(getattr(config, "simplex_geometry_distance_weight", 0.1))
        self.rbf_bins = int(getattr(config, "simplex_rbf_bins", 8))
        self.sequence_max = float(getattr(config, "simplex_sequence_max", 64.0))
        self.distance_max = float(getattr(config, "simplex_distance_max", 32.0))
        self.area_max = float(getattr(config, "simplex_area_max", 300.0))
        self.volume_scale = float(getattr(config, "simplex_volume_scale", 1000.0))
        self.dropout = torch.nn.Dropout(float(getattr(config, "simplex_dropout", 0.0)))

        self.pair_score_norm = torch.nn.LayerNorm(config.c_z)
        self.topology_score = torch.nn.Linear(config.c_z, 1)
        init_linear(self.topology_score, init="default")

        face_geom = face_geometry_dim(self.rbf_bins)
        self.face_init = SimplexMLP(3 * config.c_z + 3 * config.c_s + face_geom, self.hidden_dim, self.c_face)
        self.edge_to_face = SimplexMLP(self.c_face + 3 * config.c_z, self.hidden_dim, self.c_face, final_init="final")
        self.face_gate = torch.nn.Linear(self.c_face, self.c_face)
        init_gate_linear(self.face_gate)
        self.face_to_edge = SimplexMLP(self.c_face, self.hidden_dim, 3 * config.c_z, final_init="final")
        self.face_to_single = SimplexMLP(self.c_face, self.hidden_dim, 3 * config.c_s, final_init="final")
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
        self.face_to_tetra = SimplexMLP(self.c_tetra + 3 * self.c_face, self.hidden_dim, self.c_tetra, final_init="final")
        self.tetra_gate = torch.nn.Linear(self.c_tetra, self.c_tetra)
        init_gate_linear(self.tetra_gate)
        self.tetra_to_face = SimplexMLP(self.c_tetra, self.hidden_dim, 3 * self.c_face, final_init="final")
        self.tetra_to_edge = SimplexMLP(self.c_tetra, self.hidden_dim, 6 * config.c_z, final_init="final")
        self.tetra_to_single = SimplexMLP(self.c_tetra, self.hidden_dim, 4 * config.c_s, final_init="final")

        self.single_gate = torch.nn.Linear(config.c_s, config.c_s)
        init_gate_linear(self.single_gate)
        self.single_norm = torch.nn.LayerNorm(config.c_s)
        self.auxiliary_heads = SimplexAuxiliaryHeads(config)

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
            }
        )
        return aux

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
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if pair.ndim != 4 or single.ndim != 3:
            raise ValueError("pair must be [B, L, L, Cz] and single must be [B, L, Cs]")

        score_raw = self.topology_score(self.pair_score_norm(pair)).squeeze(-1)
        contact_logits = 0.5 * (score_raw + score_raw.transpose(1, 2))
        if not self.use_faces:
            return pair, single, self._empty_outputs(pair, single, contact_logits)

        coords_for_topology = recycled_ca_coords if self.use_recycled_geometry else None
        frames_for_geometry = recycled_frames if self.use_recycled_geometry else None
        coords_for_geometry = recycled_ca_coords if self.use_recycled_geometry else None
        with torch.no_grad():
            topology = build_simplex_topology(
                contact_logits.detach(),
                neighbor_k=self.neighbor_k,
                seq_mask=seq_mask,
                pair_mask=pair_mask,
                recycled_ca_coords=coords_for_topology,
                local_radius=self.local_radius,
                local_bias=self.local_bias,
                long_min_sep=self.long_min_sep,
                long_bias=self.long_bias,
                geometry_distance_weight=self.geometry_distance_weight,
            )

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
        if self.use_msa_to_face and msa_representation is not None:
            face_state = face_state + self._msa_to_face_update(
                msa_representation,
                face_indices,
                topology.face_mask,
                msa_mask=msa_mask,
            )
            face_state = face_state * topology.face_mask[..., None]

        edge_msg = self.edge_to_face(torch.cat([face_state, z_ij, z_ik, z_jk], dim=-1))
        face_state = face_state + self.dropout(
            torch.sigmoid(self.face_gate(face_state)) * edge_msg * topology.face_mask[..., None]
        )

        tetra_state = pair.new_empty((pair.shape[0], pair.shape[1], 0, self.c_tetra))
        if self.use_tetra and topology.tetra_indices.shape[2] > 0:
            tetra_state = self._tetra_pass(
                pair,
                single,
                face_state,
                topology,
                coords_for_geometry,
            )

            face_delta = self.tetra_to_face(tetra_state).reshape(*tetra_state.shape[:-1], 3, self.c_face)
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

        face_edge_update = self.face_to_edge(face_state).reshape(*face_state.shape[:-1], 3, self.c_z)
        face_edge_indices = torch.stack(
            [
                torch.stack([i, j], dim=-1),
                torch.stack([i, k], dim=-1),
                torch.stack([j, k], dim=-1),
            ],
            dim=-2,
        )
        face_edge_mask = topology.face_mask[..., None].expand(-1, -1, -1, 3)
        pair_delta, pair_counts = scatter_to_pair(
            face_edge_update,
            face_edge_indices,
            pair_shape=tuple(pair.shape),  # type: ignore[arg-type]
            edge_mask=face_edge_mask,
            include_reverse=True,
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
            tet_pair_delta, tet_pair_counts = scatter_to_pair(
                tet_edge_update,
                tet_edge_indices,
                pair_shape=tuple(pair.shape),  # type: ignore[arg-type]
                edge_mask=tet_edge_mask,
                include_reverse=True,
            )
            pair_delta = pair_delta + tet_pair_delta
            pair_counts = pair_counts + tet_pair_counts

        pair = pair + self.dropout(pair_delta / pair_counts.clamp_min(1.0))
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

        single_update = single_delta / single_counts.clamp_min(1.0)
        single = single + self.dropout(torch.sigmoid(self.single_gate(self.single_norm(single))) * single_update)
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
            }
        )
        return pair, single, aux

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
        tetra_state = tetra_state + self.dropout(
            torch.sigmoid(self.tetra_gate(tetra_state)) * tetra_msg * topology.tetra_mask[..., None]
        )
        return tetra_state


class SimplexGeometryLoss(torch.nn.Module):
    """Auxiliary losses for topology contact logits and simplex geometry heads."""

    def __init__(
        self,
        *,
        contact_distance_threshold: float = 8.0,
        contact_weight: float = 0.05,
        face_area_weight: float = 0.05,
        face_distance_weight: float = 0.05,
        tetra_geometry_weight: float = 0.02,
        tetra_distance_weight: float = 0.05,
        pair_face_consistency_weight: float = 0.02,
        face_tetra_consistency_weight: float = 0.02,
        volume_scale: float = 1000.0,
        area_scale: float = 300.0,
    ):
        super().__init__()
        self.contact_distance_threshold = contact_distance_threshold
        self.contact_weight = contact_weight
        self.face_area_weight = face_area_weight
        self.face_distance_weight = face_distance_weight
        self.tetra_geometry_weight = tetra_geometry_weight
        self.tetra_distance_weight = tetra_distance_weight
        self.pair_face_consistency_weight = pair_face_consistency_weight
        self.face_tetra_consistency_weight = face_tetra_consistency_weight
        self.volume_scale = volume_scale
        self.area_scale = area_scale

    def forward(
        self,
        prediction: dict[str, torch.Tensor],
        true_ca: torch.Tensor,
        ca_mask: torch.Tensor,
        *,
        seq_mask: Optional[torch.Tensor] = None,
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
            contact_loss = (bce * pair_valid).sum(dim=(1, 2)) / pair_valid.sum(dim=(1, 2)).clamp_min(1.0)
            loss_terms["simplex_contact_loss"] = contact_loss
            weighted = self.contact_weight * contact_loss
            loss_terms["weighted_simplex_contact_loss"] = weighted
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
            target_area = _triangle_area(x_i, x_j, x_k)
            denom = torch.log1p(torch.as_tensor(self.area_scale, device=device, dtype=dtype)).clamp_min(1.0)
            target_area = torch.log1p(target_area) / denom
            pred_area = prediction["simplex_face_area_logits"].to(dtype)
            valid = face_mask * gather_single(valid_res, i) * gather_single(valid_res, j) * gather_single(valid_res, k)
            face_loss = (torch.nn.functional.smooth_l1_loss(pred_area, target_area, reduction="none") * valid).sum(
                dim=(1, 2)
            ) / valid.sum(dim=(1, 2)).clamp_min(1.0)
            loss_terms["simplex_face_area_loss"] = face_loss
            weighted = self.face_area_weight * face_loss
            loss_terms["weighted_simplex_face_area_loss"] = weighted
            total = total + weighted

            if "simplex_face_distance_logits" in prediction:
                face_distance_logits = prediction["simplex_face_distance_logits"]
                face_distances = torch.stack(
                    [
                        _safe_norm(x_j - x_i),
                        _safe_norm(x_k - x_i),
                        _safe_norm(x_k - x_j),
                    ],
                    dim=-1,
                )
                target_bins = _distance_bin_indices(face_distances, n_bins=face_distance_logits.shape[-1])
                distance_mask = valid[..., None].expand_as(face_distances)
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
        ):
            tetra_indices = prediction["simplex_tetra_indices"]
            tetra_mask = prediction["simplex_tetra_mask"].to(dtype)
            i, j, k, l_idx = tetra_indices.unbind(dim=-1)
            x_i = gather_single(true_ca, i)
            x_j = gather_single(true_ca, j)
            x_k = gather_single(true_ca, k)
            x_l = gather_single(true_ca, l_idx)
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
            tet_loss_per = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none").mean(dim=-1)
            tetra_loss = (tet_loss_per * valid).sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp_min(1.0)
            loss_terms["simplex_tetra_geometry_loss"] = tetra_loss
            weighted = self.tetra_geometry_weight * tetra_loss
            loss_terms["weighted_simplex_tetra_geometry_loss"] = weighted
            total = total + weighted

            if "simplex_tetra_distance_logits" in prediction:
                tetra_distance_logits = prediction["simplex_tetra_distance_logits"]
                tetra_distances = torch.stack(
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
                target_bins = _distance_bin_indices(tetra_distances, n_bins=tetra_distance_logits.shape[-1])
                distance_mask = valid[..., None].expand_as(tetra_distances)
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
