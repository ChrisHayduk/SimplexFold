import torch

from minalphafold.evoformer import SimplicialEvoformer
from minalphafold.simplex import (
    SimplexGeometryLoss,
    SimplicialAdapter,
    _boundary_degree_weights,
    build_simplex_topology,
    face_edge_frame_features,
    face_outer_edge_delta,
    face_tetra_coboundary_delta,
    face_geometry_features,
    segment_cell_indices,
    segment_geometry_features,
    tetra_edge_frame_features,
    tetra_geometry_features,
)


class SimplexConfig:
    c_m = 24
    c_s = 20
    c_z = 16
    c_t = 12
    c_e = 12

    dim = 4
    num_heads = 4
    msa_transition_n = 2
    outer_product_dim = 8
    triangle_mult_c = 12
    triangle_dim = 4
    triangle_num_heads = 2
    pair_transition_n = 2

    evoformer_msa_dropout = 0.0
    evoformer_pair_dropout = 0.0

    simplex_every_n_blocks = 1
    simplex_neighbor_k = 4
    simplex_c_face = 12
    simplex_c_tetra = 8
    simplex_hidden_dim = 32
    simplex_use_faces = True
    simplex_use_tetra = True
    simplex_use_msa_to_face = False
    simplex_msa_to_face_rank = 8
    simplex_use_recycled_geometry = True
    simplex_local_neighbor_k = 0
    simplex_local_radius = 2
    simplex_local_bias = 4.0
    simplex_long_min_sep = 8
    simplex_long_bias = 0.0
    simplex_geometry_distance_weight = 0.1
    simplex_boundary_closure_weight = 0.0
    simplex_boundary_closure_temperature = 1.0
    simplex_rbf_bins = 4
    simplex_sequence_max = 16.0
    simplex_distance_max = 32.0
    simplex_area_max = 300.0
    simplex_volume_scale = 1000.0
    simplex_dropout = 0.0
    simplex_single_transition_n = 2
    simplex_structure_readout_scale = 0.0
    simplex_outer_edge_update_scale = 0.0
    simplex_hodge_face_update_scale = 0.0
    simplex_edge_frame_message_scale = 0.0
    simplex_segment_cell_scale = 0.0
    simplex_segment_radius = 2
    simplex_c_segment = 8
    n_dist_bins = 16


def test_build_simplex_topology_excludes_self_and_respects_masks():
    score = torch.zeros((1, 5, 5), dtype=torch.float32)
    seq_mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0]])

    topology = build_simplex_topology(
        score,
        neighbor_k=3,
        seq_mask=seq_mask,
        local_radius=1,
        local_bias=10.0,
        long_min_sep=10,
    )

    anchors = torch.arange(5)[None, :, None].expand_as(topology.nbr_idx)
    assert not torch.any(topology.nbr_idx == anchors)
    assert not torch.any(topology.nbr_idx[:, :4] == 4)
    assert torch.all(topology.face_mask[:, 4] == 0)
    assert topology.face_indices.shape == (1, 5, 3, 3)
    assert topology.tetra_indices.shape == (1, 5, 1, 4)


def test_build_simplex_topology_reserves_local_neighbor_slots():
    score = torch.zeros((1, 6, 6), dtype=torch.float32)
    score[:, :, 5] = 100.0

    topology = build_simplex_topology(
        score,
        neighbor_k=4,
        local_neighbor_k=2,
        local_radius=-1,
        local_bias=0.0,
        long_min_sep=-1,
    )

    assert set(topology.nbr_idx[0, 2, :2].tolist()) == {1, 3}
    assert 5 in topology.nbr_idx[0, 2, 2:].tolist()
    assert topology.face_indices.shape == (1, 6, 6, 3)
    assert topology.tetra_indices.shape == (1, 6, 4, 4)


def test_build_simplex_topology_flag_closure_downweights_open_cells():
    score = torch.full((1, 4, 4), -8.0, dtype=torch.float32)
    score[:, 0, 1:] = 8.0
    score[:, 1:, 0] = 8.0
    score[:, 1, 2] = 8.0
    score[:, 2, 1] = 8.0
    score[:, 1, 3] = 8.0
    score[:, 3, 1] = 8.0
    score[:, 2, 3] = -8.0
    score[:, 3, 2] = -8.0

    topology = build_simplex_topology(
        score,
        neighbor_k=3,
        local_radius=-1,
        local_bias=0.0,
        long_min_sep=-1,
        boundary_closure_weight=1.0,
        boundary_closure_temperature=1.0,
    )

    anchor_face_masks = topology.face_mask[0, 0]
    assert anchor_face_masks.max() > 0.99
    assert anchor_face_masks.min() < 0.08
    assert topology.tetra_mask[0, 0, 0] < 0.3


def test_simplicial_adapter_teacher_forcing_selects_true_nearest_neighbors():
    class TeacherConfig(SimplexConfig):
        simplex_neighbor_k = 2
        simplex_local_radius = -1
        simplex_local_bias = 0.0
        simplex_long_min_sep = -1
        simplex_geometry_distance_weight = 0.0

    adapter = SimplicialAdapter(TeacherConfig())
    pair = torch.randn(1, 5, 5, TeacherConfig.c_z)
    single = torch.randn(1, 5, TeacherConfig.c_s)
    teacher_ca = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [9.0, 0.0, 0.0], [12.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    _, _, aux = adapter(
        pair,
        single,
        seq_mask=torch.ones(1, 5),
        simplex_teacher_ca_coords=teacher_ca,
        simplex_teacher_ca_mask=torch.ones(1, 5),
        simplex_teacher_forcing_weight=torch.tensor(1.0),
    )

    assert set(aux["simplex_face_indices"][0, 0, 0, 1:].tolist()) == {1, 2}


def test_simplicial_adapter_local_neighbor_override_reserves_scaffold_slots():
    class LocalOverrideConfig(SimplexConfig):
        simplex_neighbor_k = 4
        simplex_use_recycled_geometry = False
        simplex_local_radius = -1
        simplex_local_bias = 0.0
        simplex_long_min_sep = -1

    adapter = SimplicialAdapter(LocalOverrideConfig())
    pair = torch.zeros(1, 6, 6, LocalOverrideConfig.c_z)
    single = torch.randn(1, 6, LocalOverrideConfig.c_s)

    _, _, aux = adapter(
        pair,
        single,
        seq_mask=torch.ones(1, 6),
        simplex_local_neighbor_k_override=torch.tensor(2.0),
    )

    assert set(aux["simplex_face_indices"][0, 2, 0, 1:].tolist()) == {1, 3}


def test_simplicial_adapter_update_scale_override_gates_boundary_residuals():
    class FaceOnlyConfig(SimplexConfig):
        simplex_neighbor_k = 3
        simplex_use_tetra = False
        simplex_use_recycled_geometry = False
        simplex_local_radius = -1
        simplex_local_bias = 0.0
        simplex_long_min_sep = -1

    torch.manual_seed(1)
    adapter = SimplicialAdapter(FaceOnlyConfig())
    pair = torch.randn(1, 5, 5, FaceOnlyConfig.c_z)
    pair = 0.5 * (pair + pair.transpose(1, 2))
    single = torch.randn(1, 5, FaceOnlyConfig.c_s)
    seq_mask = torch.ones(1, 5)
    pair_mask = torch.ones(1, 5, 5)

    pair_zero, single_zero, _ = adapter(
        pair,
        single,
        seq_mask=seq_mask,
        pair_mask=pair_mask,
        simplex_pair_update_scale_override=torch.tensor(0.0),
        simplex_single_update_scale_override=torch.tensor(0.0),
    )
    pair_full, single_full, _ = adapter(
        pair,
        single,
        seq_mask=seq_mask,
        pair_mask=pair_mask,
        simplex_pair_update_scale_override=torch.tensor(1.0),
        simplex_single_update_scale_override=torch.tensor(1.0),
    )

    assert torch.allclose(pair_zero, pair)
    assert torch.allclose(single_zero, single)
    assert not torch.allclose(pair_full, pair_zero)
    assert not torch.allclose(single_full, single_zero)


def test_face_outer_edge_delta_uses_shared_boundary_edges_only():
    face_state = torch.tensor(
        [
            [
                [[1.0, 0.0], [3.0, 0.0], [7.0, 0.0]],
                [[11.0, 0.0], [13.0, 0.0], [17.0, 0.0]],
                [[19.0, 0.0], [23.0, 0.0], [29.0, 0.0]],
                [[31.0, 0.0], [37.0, 0.0], [41.0, 0.0]],
            ]
        ]
    )
    face_indices = torch.tensor(
        [
            [
                [[0, 1, 2], [0, 1, 3], [0, 2, 3]],
                [[1, 0, 2], [1, 0, 3], [1, 2, 3]],
                [[2, 0, 1], [2, 0, 3], [2, 1, 3]],
                [[3, 0, 1], [3, 0, 2], [3, 1, 2]],
            ]
        ],
        dtype=torch.long,
    )
    face_mask = torch.ones(1, 4, 3)

    delta = face_outer_edge_delta(face_state, face_indices, face_mask, num_residues=4)

    expected_face_012 = torch.tensor([19.4, 0.0])
    assert torch.allclose(delta[0, 0, 0], expected_face_012 - face_state[0, 0, 0])
    assert torch.isfinite(delta).all()
    isolated_delta = face_outer_edge_delta(
        face_state[:, :1, :1],
        face_indices[:, :1, :1],
        face_mask[:, :1, :1],
        num_residues=4,
    )
    assert torch.allclose(isolated_delta, torch.zeros_like(isolated_delta))


def test_outer_edge_adapter_scale_changes_outputs_without_new_parameters():
    class OuterEdgeConfig(SimplexConfig):
        simplex_neighbor_k = 3
        simplex_use_tetra = False
        simplex_use_recycled_geometry = False
        simplex_local_radius = -1
        simplex_local_bias = 0.0
        simplex_long_min_sep = -1

    class OuterEdgeEnabledConfig(OuterEdgeConfig):
        simplex_outer_edge_update_scale = 0.25

    torch.manual_seed(4)
    off_adapter = SimplicialAdapter(OuterEdgeConfig())
    torch.manual_seed(4)
    on_adapter = SimplicialAdapter(OuterEdgeEnabledConfig())
    pair = torch.randn(1, 5, 5, OuterEdgeConfig.c_z)
    pair = 0.5 * (pair + pair.transpose(1, 2))
    single = torch.randn(1, 5, OuterEdgeConfig.c_s)

    off_params = sum(p.numel() for p in off_adapter.parameters())
    on_params = sum(p.numel() for p in on_adapter.parameters())
    off_pair, off_single, _ = off_adapter(pair, single)
    on_pair, on_single, _ = on_adapter(pair, single)

    assert on_params == off_params
    assert not torch.allclose(on_pair, off_pair)
    assert not torch.allclose(on_single, off_single)


def test_face_tetra_coboundary_delta_uses_sibling_faces_in_selected_tetras():
    face_state = torch.tensor([[[[1.0, 0.0], [3.0, 0.0], [7.0, 0.0], [11.0, 0.0]]]])
    face_mask = torch.ones(1, 1, 4)
    tetra_face_slots = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    tetra_mask = torch.ones(1, 1, 2)

    delta = face_tetra_coboundary_delta(face_state, face_mask, tetra_face_slots, tetra_mask)

    expected = torch.tensor([[[[6.0, 0.0], [1.0, 0.0], [-3.0, 0.0], [-7.0, 0.0]]]])
    assert torch.allclose(delta, expected)
    assert torch.isfinite(delta).all()
    assert torch.allclose(
        face_tetra_coboundary_delta(face_state, face_mask, tetra_face_slots, torch.zeros_like(tetra_mask)),
        torch.zeros_like(face_state),
    )


def test_hodge_face_adapter_scale_changes_outputs_without_new_parameters():
    class HodgeConfig(SimplexConfig):
        simplex_neighbor_k = 3
        simplex_use_tetra = True
        simplex_use_recycled_geometry = False
        simplex_local_radius = -1
        simplex_local_bias = 0.0
        simplex_long_min_sep = -1

    class HodgeEnabledConfig(HodgeConfig):
        simplex_hodge_face_update_scale = 0.25

    torch.manual_seed(7)
    off_adapter = SimplicialAdapter(HodgeConfig())
    torch.manual_seed(7)
    on_adapter = SimplicialAdapter(HodgeEnabledConfig())
    pair = torch.randn(1, 5, 5, HodgeConfig.c_z)
    pair = 0.5 * (pair + pair.transpose(1, 2))
    single = torch.randn(1, 5, HodgeConfig.c_s)

    off_params = sum(p.numel() for p in off_adapter.parameters())
    on_params = sum(p.numel() for p in on_adapter.parameters())
    off_pair, off_single, _ = off_adapter(pair, single)
    on_pair, on_single, _ = on_adapter(pair, single)

    assert on_params == off_params
    assert not torch.allclose(on_pair, off_pair)
    assert not torch.allclose(on_single, off_single)


def test_edge_frame_features_are_rigid_transform_invariant():
    face_edge_indices = torch.tensor([[[[[0, 1], [0, 2], [1, 2]]]]], dtype=torch.long)
    face_opposite_indices = torch.tensor([[[[2, 1, 0]]]], dtype=torch.long)
    tet_edge_indices = torch.tensor(
        [[[[[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]]]],
        dtype=torch.long,
    )
    tet_opposite_indices = torch.tensor(
        [[[[[2, 3], [1, 3], [1, 2], [0, 3], [0, 2], [0, 1]]]]],
        dtype=torch.long,
    )
    coords = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [3.0, 0.5, 0.0],
                [0.5, 2.0, 0.75],
                [0.25, 0.75, 2.5],
            ]
        ],
        dtype=torch.float32,
    )
    frames = torch.eye(3).reshape(1, 1, 3, 3).expand(1, 4, 3, 3).clone()
    theta = torch.tensor(0.7)
    rotation = torch.tensor(
        [
            [torch.cos(theta), -torch.sin(theta), 0.0],
            [torch.sin(theta), torch.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated_coords = torch.einsum("ij,bnj->bni", rotation, coords) + torch.tensor([2.0, -1.0, 0.5])
    rotated_frames = torch.einsum("ij,bnjk->bnik", rotation, frames)

    face_features = face_edge_frame_features(
        face_edge_indices,
        face_opposite_indices,
        recycled_ca_coords=coords,
        recycled_frames=frames,
        distance_max=32.0,
    )
    rotated_face_features = face_edge_frame_features(
        face_edge_indices,
        face_opposite_indices,
        recycled_ca_coords=rotated_coords,
        recycled_frames=rotated_frames,
        distance_max=32.0,
    )
    tetra_features = tetra_edge_frame_features(
        tet_edge_indices,
        tet_opposite_indices,
        recycled_ca_coords=coords,
        recycled_frames=frames,
        distance_max=32.0,
        volume_scale=1000.0,
    )
    rotated_tetra_features = tetra_edge_frame_features(
        tet_edge_indices,
        tet_opposite_indices,
        recycled_ca_coords=rotated_coords,
        recycled_frames=rotated_frames,
        distance_max=32.0,
        volume_scale=1000.0,
    )

    assert face_features.shape[-1] == 10
    assert tetra_features.shape[-1] == 18
    assert torch.allclose(face_features, rotated_face_features, atol=1e-5)
    assert torch.allclose(tetra_features, rotated_tetra_features, atol=1e-5)


def test_edge_frame_message_scale_changes_pair_readout_within_adapter():
    class EdgeFrameConfig(SimplexConfig):
        simplex_neighbor_k = 3
        simplex_use_tetra = False
        simplex_local_radius = -1
        simplex_local_bias = 0.0
        simplex_long_min_sep = -1

    class EdgeFrameEnabledConfig(EdgeFrameConfig):
        simplex_edge_frame_message_scale = 0.25

    torch.manual_seed(5)
    off_adapter = SimplicialAdapter(EdgeFrameConfig())
    torch.manual_seed(5)
    on_adapter = SimplicialAdapter(EdgeFrameEnabledConfig())
    pair = torch.randn(1, 5, 5, EdgeFrameConfig.c_z)
    pair = 0.5 * (pair + pair.transpose(1, 2))
    single = torch.randn(1, 5, EdgeFrameConfig.c_s)
    coords = torch.randn(1, 5, 3)
    frames = torch.eye(3).reshape(1, 1, 3, 3).expand(1, 5, 3, 3).clone()

    off_pair, off_single, _ = off_adapter(pair, single, recycled_ca_coords=coords, recycled_frames=frames)
    on_pair, on_single, _ = on_adapter(pair, single, recycled_ca_coords=coords, recycled_frames=frames)

    assert sum(p.numel() for p in on_adapter.parameters()) > sum(p.numel() for p in off_adapter.parameters())
    assert not torch.allclose(on_pair, off_pair)
    assert torch.allclose(on_single, off_single)


def test_segment_cell_indices_are_contiguous_and_respect_sequence_mask():
    seq_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 1.0]])

    indices, mask = segment_cell_indices(
        batch_size=1,
        num_residues=5,
        radius=1,
        device=seq_mask.device,
        seq_mask=seq_mask,
    )

    assert indices.shape == (1, 5, 3)
    assert mask.shape == (1, 5, 3)
    assert indices[0, 2].tolist() == [1, 2, 3]
    assert torch.allclose(mask[0, 2], torch.tensor([1.0, 1.0, 0.0]))
    assert torch.all(mask[0, 3] == 0)


def test_segment_geometry_features_are_rigid_transform_invariant():
    indices = torch.tensor([[[0, 1, 2], [1, 2, 3]]], dtype=torch.long)
    mask = torch.ones(1, 2, 3)
    coords = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.5, 1.5, 0.0],
                [3.0, 1.5, 2.0],
            ]
        ],
        dtype=torch.float32,
    )
    theta = torch.tensor(0.4)
    rotation = torch.tensor(
        [
            [torch.cos(theta), -torch.sin(theta), 0.0],
            [torch.sin(theta), torch.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated_coords = torch.einsum("ij,bnj->bni", rotation, coords) + torch.tensor([-3.0, 1.0, 0.25])

    features = segment_geometry_features(
        indices,
        mask,
        recycled_ca_coords=coords,
        sequence_max=16.0,
        distance_max=32.0,
    )
    rotated_features = segment_geometry_features(
        indices,
        mask,
        recycled_ca_coords=rotated_coords,
        sequence_max=16.0,
        distance_max=32.0,
    )

    assert features.shape[-1] == 4
    assert torch.allclose(features, rotated_features, atol=1e-5)


def test_segment_cells_change_face_mediated_outputs_within_adapter():
    class SegmentConfig(SimplexConfig):
        simplex_neighbor_k = 3
        simplex_use_tetra = False
        simplex_local_radius = -1
        simplex_local_bias = 0.0
        simplex_long_min_sep = -1

    class SegmentEnabledConfig(SegmentConfig):
        simplex_segment_cell_scale = 0.25
        simplex_segment_radius = 1
        simplex_c_segment = 8

    torch.manual_seed(6)
    off_adapter = SimplicialAdapter(SegmentConfig())
    torch.manual_seed(6)
    on_adapter = SimplicialAdapter(SegmentEnabledConfig())
    pair = torch.randn(1, 5, 5, SegmentConfig.c_z)
    pair = 0.5 * (pair + pair.transpose(1, 2))
    single = torch.randn(1, 5, SegmentConfig.c_s)
    coords = torch.randn(1, 5, 3)

    off_pair, off_single, _ = off_adapter(pair, single, recycled_ca_coords=coords)
    on_pair, on_single, _ = on_adapter(pair, single, recycled_ca_coords=coords)

    assert sum(p.numel() for p in on_adapter.parameters()) > sum(p.numel() for p in off_adapter.parameters())
    assert not torch.allclose(on_pair, off_pair)
    assert not torch.allclose(on_single, off_single)


def test_simplex_geometry_features_are_rigid_transform_invariant():
    face_indices = torch.tensor([[[[0, 1, 2], [1, 2, 3]]]], dtype=torch.long)
    tetra_indices = torch.tensor([[[[0, 1, 2, 3]]]], dtype=torch.long)
    coords = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ],
        dtype=torch.float32,
    )
    frames = torch.eye(3).reshape(1, 1, 3, 3).expand(1, 4, 3, 3).clone()
    rotation = torch.tensor(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    translation = torch.tensor([2.0, -3.0, 0.5], dtype=torch.float32)
    rotated_coords = torch.einsum("ij,bnj->bni", rotation, coords) + translation
    rotated_frames = torch.einsum("ij,bnjk->bnik", rotation, frames)

    face = face_geometry_features(
        face_indices,
        recycled_ca_coords=coords,
        recycled_frames=frames,
        rbf_bins=4,
        sequence_max=16.0,
        distance_max=32.0,
        area_max=300.0,
    )
    face_rotated = face_geometry_features(
        face_indices,
        recycled_ca_coords=rotated_coords,
        recycled_frames=rotated_frames,
        rbf_bins=4,
        sequence_max=16.0,
        distance_max=32.0,
        area_max=300.0,
    )
    tetra = tetra_geometry_features(
        tetra_indices,
        recycled_ca_coords=coords,
        rbf_bins=4,
        sequence_max=16.0,
        distance_max=32.0,
        area_max=300.0,
        volume_scale=1000.0,
    )
    tetra_rotated = tetra_geometry_features(
        tetra_indices,
        recycled_ca_coords=rotated_coords,
        rbf_bins=4,
        sequence_max=16.0,
        distance_max=32.0,
        area_max=300.0,
        volume_scale=1000.0,
    )

    assert torch.allclose(face, face_rotated, atol=1e-5)
    assert torch.allclose(tetra, tetra_rotated, atol=1e-5)


def test_simplicial_adapter_shapes_masks_and_gradients():
    torch.manual_seed(0)
    cfg = SimplexConfig()
    adapter = SimplicialAdapter(cfg)
    pair = torch.randn(2, 6, 6, cfg.c_z, requires_grad=True)
    single = torch.randn(2, 6, cfg.c_s, requires_grad=True)
    seq_mask = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        ]
    )
    pair_mask = seq_mask[:, :, None] * seq_mask[:, None, :]
    coords = torch.randn(2, 6, 3)
    frames = torch.eye(3).reshape(1, 1, 3, 3).expand(2, 6, 3, 3).clone()

    out_pair, out_single, aux = adapter(
        pair,
        single,
        seq_mask=seq_mask,
        pair_mask=pair_mask,
        recycled_ca_coords=coords,
        recycled_frames=frames,
    )
    assert out_pair.shape == pair.shape
    assert out_single.shape == single.shape
    assert aux["simplex_face_indices"].shape[-1] == 3
    assert aux["simplex_tetra_indices"].shape[-1] == 4
    assert aux["simplex_face_distance_logits"].shape[-2:] == (3, cfg.n_dist_bins)
    assert aux["simplex_tetra_distance_logits"].shape[-2:] == (6, cfg.n_dist_bins)
    assert not torch.allclose(out_pair[0], pair.detach()[0])
    assert not torch.allclose(out_single[0], single.detach()[0])
    assert torch.all(out_single[1, 4:] == 0)
    assert torch.all(out_pair[1, 4:, :, :] == 0)
    assert torch.all(out_pair[1, :, 4:, :] == 0)

    loss = out_pair.sum() + out_single.sum() + aux["simplex_contact_logits"].sum()
    loss.backward()
    assert pair.grad is not None
    assert single.grad is not None
    assert adapter.topology_score.weight.grad is not None


def test_simplicial_adapter_can_emit_structure_readout_from_selected_cells():
    class ReadoutConfig(SimplexConfig):
        simplex_structure_readout_scale = 0.25

    torch.manual_seed(3)
    cfg = ReadoutConfig()
    adapter = SimplicialAdapter(cfg)
    pair = torch.randn(2, 6, 6, cfg.c_z)
    single = torch.randn(2, 6, cfg.c_s)
    seq_mask = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        ]
    )
    pair_mask = seq_mask[:, :, None] * seq_mask[:, None, :]

    _, _, aux = adapter(pair, single, seq_mask=seq_mask, pair_mask=pair_mask)

    assert aux["simplex_structure_pair_readout"].shape == pair.shape
    assert aux["simplex_structure_single_readout"].shape == single.shape
    assert not torch.allclose(aux["simplex_structure_pair_readout"][0], torch.zeros_like(pair[0]))
    assert not torch.allclose(aux["simplex_structure_single_readout"][0], torch.zeros_like(single[0]))
    assert torch.all(aux["simplex_structure_single_readout"][1, 4:] == 0)
    assert torch.all(aux["simplex_structure_pair_readout"][1, 4:, :, :] == 0)
    assert torch.all(aux["simplex_structure_pair_readout"][1, :, 4:, :] == 0)


def test_optional_low_rank_msa_to_face_path_runs():
    torch.manual_seed(2)
    cfg = SimplexConfig()
    cfg.simplex_use_msa_to_face = True
    adapter = SimplicialAdapter(cfg)
    pair = torch.randn(1, 6, 6, cfg.c_z)
    single = torch.randn(1, 6, cfg.c_s)
    msa = torch.randn(1, 3, 6, cfg.c_m)
    msa_mask = torch.ones(1, 3, 6)

    _, _, aux = adapter(pair, single, msa_representation=msa, msa_mask=msa_mask)

    assert aux["simplex_face_area_logits"].shape[:2] == (1, 6)
    assert adapter.msa_to_face_a.weight.shape == (cfg.simplex_msa_to_face_rank, cfg.c_m)


def test_simplicial_evoformer_returns_auxiliary_simplex_outputs():
    torch.manual_seed(1)
    cfg = SimplexConfig()
    block = SimplicialEvoformer(cfg)
    msa = torch.randn(1, 3, 6, cfg.c_m)
    pair = torch.randn(1, 6, 6, cfg.c_z)
    single = torch.randn(1, 6, cfg.c_s)

    out_msa, out_pair, out_single, aux = block(msa, pair, single)

    assert out_msa.shape == msa.shape
    assert out_pair.shape == pair.shape
    assert out_single.shape == single.shape
    assert aux["simplex_contact_logits"].shape == (1, 6, 6)
    assert aux["simplex_face_area_logits"].shape[:2] == (1, 6)
    assert aux["simplex_face_distance_logits"].shape[-2:] == (3, cfg.n_dist_bins)
    assert aux["simplex_tetra_geometry_logits"].shape[:2] == (1, 6)
    assert aux["simplex_tetra_distance_logits"].shape[-2:] == (6, cfg.n_dist_bins)


def test_simplex_geometry_loss_is_finite_with_and_without_tetrahedra():
    true_ca = torch.randn(1, 5, 3)
    ca_mask = torch.ones(1, 5)
    prediction = {
        "simplex_contact_logits": torch.zeros(1, 5, 5),
        "simplex_face_indices": torch.tensor([[[[0, 1, 2]], [[1, 2, 3]], [[2, 3, 4]], [[3, 4, 0]], [[4, 0, 1]]]]),
        "simplex_face_mask": torch.ones(1, 5, 1),
        "simplex_face_area_logits": torch.zeros(1, 5, 1),
        "simplex_tetra_indices": torch.empty(1, 5, 0, 4, dtype=torch.long),
        "simplex_tetra_mask": torch.empty(1, 5, 0),
        "simplex_tetra_geometry_logits": torch.empty(1, 5, 0, 3),
    }

    terms = SimplexGeometryLoss()(prediction, true_ca, ca_mask)

    assert terms["simplex_aux_loss"].shape == (1,)
    assert torch.isfinite(terms["simplex_aux_loss"]).all()
    assert "simplex_face_area_loss" in terms
    assert "simplex_tetra_geometry_loss" not in terms


def test_simplex_contact_loss_balances_contacts_and_non_contacts():
    true_ca = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
                [40.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    ca_mask = torch.ones(1, 4)
    contact_logits = torch.ones(1, 4, 4)
    contact_logits[0, 0, 1] = -2.0
    contact_logits[0, 1, 0] = -2.0
    prediction = {"simplex_contact_logits": contact_logits}

    terms = SimplexGeometryLoss(contact_weight=1.0)(prediction, true_ca, ca_mask)

    distances = torch.cdist(true_ca, true_ca)
    contact_true = (distances < 8.0).float()
    valid = 1.0 - torch.eye(4).reshape(1, 4, 4)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        contact_logits,
        contact_true,
        reduction="none",
    )
    positive = valid * contact_true
    negative = valid * (1.0 - contact_true)
    expected = 0.5 * (
        (bce * positive).sum() / positive.sum()
        + (bce * negative).sum() / negative.sum()
    )

    assert torch.allclose(terms["simplex_contact_loss"], expected.reshape(1))
    assert torch.allclose(terms["weighted_simplex_contact_loss"], expected.reshape(1))


def test_simplex_topology_neighborhood_loss_targets_anchor_neighbors():
    true_ca = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
                [40.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    ca_mask = torch.ones(1, 4)
    contact_logits = torch.tensor(
        [
            [
                [0.0, -2.0, 3.0, 1.0],
                [-2.0, 0.0, 3.0, 1.0],
                [1.0, 2.0, 0.0, 3.0],
                [1.0, 2.0, 3.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    prediction = {"simplex_contact_logits": contact_logits}

    terms = SimplexGeometryLoss(contact_weight=0.0, topology_neighborhood_weight=1.0)(
        prediction,
        true_ca,
        ca_mask,
    )

    distances = torch.cdist(true_ca, true_ca)
    contact_true = (distances < 8.0).float()
    valid = 1.0 - torch.eye(4).reshape(1, 4, 4)
    positive = valid * contact_true
    positive_count_by_row = positive.sum(dim=-1)
    masked_logits = contact_logits.masked_fill(valid <= 0, torch.finfo(contact_logits.dtype).min / 4)
    log_probs = torch.nn.functional.log_softmax(masked_logits, dim=-1)
    target_distribution = positive / positive_count_by_row[..., None].clamp_min(1.0)
    row_loss = -(target_distribution * log_probs).sum(dim=-1)
    has_positive = (positive_count_by_row > 0).float()
    expected = (row_loss * has_positive).sum(dim=1) / has_positive.sum(dim=1).clamp_min(1.0)

    assert torch.allclose(terms["simplex_topology_neighborhood_loss"], expected)
    assert torch.allclose(terms["weighted_simplex_topology_neighborhood_loss"], expected)
    assert torch.allclose(terms["simplex_aux_loss"], expected)


def test_simplex_topology_margin_loss_ranks_contacts_above_hard_non_contacts():
    true_ca = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
                [40.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    ca_mask = torch.ones(1, 4)
    contact_logits = torch.tensor(
        [
            [
                [0.0, -2.0, 3.0, 1.0],
                [-2.0, 0.0, 3.0, 1.0],
                [1.0, 2.0, 0.0, 3.0],
                [1.0, 2.0, 3.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    prediction = {"simplex_contact_logits": contact_logits}

    terms = SimplexGeometryLoss(
        contact_weight=0.0,
        topology_neighborhood_weight=0.0,
        topology_margin_weight=1.0,
        topology_margin=1.0,
        topology_margin_hard_negatives=1,
    )(prediction, true_ca, ca_mask)

    expected = torch.nn.functional.softplus(torch.tensor([6.0]))

    assert torch.allclose(terms["simplex_topology_margin_loss"], expected)
    assert torch.allclose(terms["weighted_simplex_topology_margin_loss"], expected)
    assert torch.allclose(terms["simplex_aux_loss"], expected)


def test_simplex_geometry_loss_adds_distance_and_consistency_terms():
    cfg = SimplexConfig()
    true_ca = torch.randn(1, 5, 3)
    ca_mask = torch.ones(1, 5)
    prediction = {
        "distogram_logits": torch.zeros(1, 5, 5, cfg.n_dist_bins),
        "simplex_contact_logits": torch.zeros(1, 5, 5),
        "simplex_face_indices": torch.tensor([[[[0, 1, 2]], [[1, 2, 3]], [[2, 3, 4]], [[3, 4, 0]], [[4, 0, 1]]]]),
        "simplex_face_mask": torch.ones(1, 5, 1),
        "simplex_face_area_logits": torch.zeros(1, 5, 1),
        "simplex_face_distance_logits": torch.zeros(1, 5, 1, 3, cfg.n_dist_bins),
        "simplex_tetra_indices": torch.tensor([[[[0, 1, 2, 3]], [[1, 2, 3, 4]], [[2, 3, 4, 0]], [[3, 4, 0, 1]], [[4, 0, 1, 2]]]]),
        "simplex_tetra_mask": torch.ones(1, 5, 1),
        "simplex_tetra_geometry_logits": torch.zeros(1, 5, 1, 3),
        "simplex_tetra_distance_logits": torch.zeros(1, 5, 1, 6, cfg.n_dist_bins),
        "simplex_tetra_face_slots": torch.zeros(1, 3, dtype=torch.long),
    }

    terms = SimplexGeometryLoss()(prediction, true_ca, ca_mask)

    assert "simplex_face_distance_loss" in terms
    assert "simplex_tetra_distance_loss" in terms
    assert "simplex_pair_face_consistency_loss" in terms
    assert "simplex_face_tetra_consistency_loss" in terms
    assert torch.isfinite(terms["simplex_aux_loss"]).all()


def test_simplex_face_normal_loss_is_global_rotation_invariant():
    true_atom14 = torch.zeros(1, 3, 14, 3)
    ca = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    true_atom14[0, :, 1, :] = ca
    true_atom14[0, :, 0, :] = ca + torch.tensor([0.0, 1.0, 0.0])
    true_atom14[0, :, 2, :] = ca + torch.tensor([1.0, 0.0, 0.0])
    atom14_mask = torch.zeros(1, 3, 14)
    atom14_mask[:, :, :3] = 1.0
    theta = torch.tensor(0.7)
    rotation = torch.tensor(
        [
            [torch.cos(theta), -torch.sin(theta), 0.0],
            [torch.sin(theta), torch.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    pred_atom14 = torch.einsum("...i,ji->...j", true_atom14, rotation) + torch.tensor([3.0, -2.0, 5.0])
    prediction = {
        "atom14_coords": pred_atom14,
        "atom14_mask": atom14_mask,
        "simplex_face_indices": torch.tensor([[[[0, 1, 2]]]]),
        "simplex_face_mask": torch.ones(1, 1, 1),
        "simplex_face_area_logits": torch.zeros(1, 1, 1),
    }

    terms = SimplexGeometryLoss(
        contact_weight=0.0,
        topology_neighborhood_weight=0.0,
        face_area_weight=0.0,
        face_coordinate_weight=0.0,
        face_coordinate_distance_weight=0.0,
        face_normal_weight=1.0,
        face_distance_weight=0.0,
    )(
        prediction,
        true_atom14[:, :, 1, :],
        atom14_mask[:, :, 1],
        true_atom_positions=true_atom14,
        true_atom_mask=atom14_mask,
    )

    assert terms["simplex_face_normal_loss"].item() < 1e-6
    assert terms["weighted_simplex_face_normal_loss"].item() < 1e-6


def test_simplex_shape_loss_is_local_rigid_motion_invariant():
    true_ca = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [0.0, 5.0, 0.0],
                [0.0, 0.0, 6.0],
            ]
        ],
        dtype=torch.float32,
    )
    theta = torch.tensor(0.7)
    rotation = torch.tensor(
        [
            [torch.cos(theta), -torch.sin(theta), 0.0],
            [torch.sin(theta), torch.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated_ca = torch.einsum("...i,ji->...j", true_ca, rotation) + torch.tensor([3.0, -2.0, 5.0])
    reflected_ca = true_ca.clone()
    reflected_ca[:, 3, 2] = -reflected_ca[:, 3, 2]
    collapsed_ca = torch.zeros_like(true_ca)
    ca_mask = torch.ones(1, 4)
    matching_atom14 = torch.zeros(1, 4, 14, 3)
    matching_atom14[:, :, 1, :] = rotated_ca
    reflected_atom14 = torch.zeros(1, 4, 14, 3)
    reflected_atom14[:, :, 1, :] = reflected_ca
    collapsed_atom14 = torch.zeros(1, 4, 14, 3)
    collapsed_atom14[:, :, 1, :] = collapsed_ca
    base_prediction = {
        "simplex_face_indices": torch.tensor([[[[0, 1, 2]]]]),
        "simplex_face_mask": torch.ones(1, 1, 1),
        "simplex_face_area_logits": torch.zeros(1, 1, 1),
        "simplex_tetra_indices": torch.tensor([[[[0, 1, 2, 3]]]]),
        "simplex_tetra_mask": torch.ones(1, 1, 1),
        "simplex_tetra_geometry_logits": torch.zeros(1, 1, 1, 3),
    }
    loss_fn = SimplexGeometryLoss(
        contact_weight=0.0,
        topology_neighborhood_weight=0.0,
        face_area_weight=0.0,
        face_coordinate_weight=0.0,
        face_coordinate_distance_weight=0.0,
        face_shape_weight=1.0,
        face_distance_weight=0.0,
        tetra_geometry_weight=0.0,
        tetra_coordinate_weight=0.0,
        tetra_coordinate_distance_weight=0.0,
        tetra_shape_weight=1.0,
        tetra_distance_weight=0.0,
        pair_face_consistency_weight=0.0,
        face_tetra_consistency_weight=0.0,
    )

    matching_terms = loss_fn({**base_prediction, "atom14_coords": matching_atom14}, true_ca, ca_mask)
    reflected_terms = loss_fn({**base_prediction, "atom14_coords": reflected_atom14}, true_ca, ca_mask)
    collapsed_terms = loss_fn({**base_prediction, "atom14_coords": collapsed_atom14}, true_ca, ca_mask)
    learned_atom14 = collapsed_atom14.clone().requires_grad_(True)
    learned_terms = loss_fn({**base_prediction, "atom14_coords": learned_atom14}, true_ca, ca_mask)
    learned_loss = (
        learned_terms["weighted_simplex_face_shape_loss"] + learned_terms["weighted_simplex_tetra_shape_loss"]
    ).sum()
    learned_loss.backward()

    assert matching_terms["simplex_face_shape_loss"].item() < 1e-6
    assert matching_terms["simplex_tetra_shape_loss"].item() < 1e-6
    assert reflected_terms["simplex_tetra_shape_loss"] > matching_terms["simplex_tetra_shape_loss"]
    assert collapsed_terms["simplex_face_shape_loss"] > matching_terms["simplex_face_shape_loss"]
    assert collapsed_terms["simplex_tetra_shape_loss"] > matching_terms["simplex_tetra_shape_loss"]
    assert learned_atom14.grad is not None
    assert torch.isfinite(learned_atom14.grad).all()


def test_simplex_coordinate_realization_loss_penalizes_collapsed_cells():
    true_ca = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [0.0, 5.0, 0.0],
                [0.0, 0.0, 6.0],
                [8.0, 8.0, 8.0],
            ]
        ],
        dtype=torch.float32,
    )
    ca_mask = torch.ones(1, 5)
    matching_atom14 = torch.zeros(1, 5, 14, 3)
    matching_atom14[:, :, 1, :] = true_ca
    collapsed_atom14 = matching_atom14.clone()
    collapsed_atom14[:, :, 1, :] = 0.0
    base_prediction = {
        "simplex_contact_logits": torch.zeros(1, 5, 5),
        "simplex_face_indices": torch.tensor([[[[0, 1, 2]], [[1, 2, 3]], [[2, 3, 4]], [[3, 4, 0]], [[4, 0, 1]]]]),
        "simplex_face_mask": torch.ones(1, 5, 1),
        "simplex_face_area_logits": torch.zeros(1, 5, 1),
        "simplex_tetra_indices": torch.tensor([[[[0, 1, 2, 3]], [[1, 2, 3, 4]], [[2, 3, 4, 0]], [[3, 4, 0, 1]], [[4, 0, 1, 2]]]]),
        "simplex_tetra_mask": torch.ones(1, 5, 1),
        "simplex_tetra_geometry_logits": torch.zeros(1, 5, 1, 3),
    }
    loss_fn = SimplexGeometryLoss(
        contact_weight=0.0,
        topology_neighborhood_weight=0.0,
        face_area_weight=0.0,
        face_coordinate_weight=1.0,
        face_coordinate_distance_weight=1.0,
        face_boundary_lddt_weight=1.0,
        face_distance_weight=0.0,
        tetra_geometry_weight=0.0,
        tetra_coordinate_weight=1.0,
        tetra_coordinate_distance_weight=1.0,
        tetra_boundary_lddt_weight=1.0,
        tetra_distance_weight=0.0,
        pair_face_consistency_weight=0.0,
        face_tetra_consistency_weight=0.0,
    )

    matching_terms = loss_fn({**base_prediction, "atom14_coords": matching_atom14}, true_ca, ca_mask)
    collapsed_terms = loss_fn({**base_prediction, "atom14_coords": collapsed_atom14}, true_ca, ca_mask)

    assert matching_terms["simplex_face_coordinate_area_loss"].item() < 1e-6
    assert matching_terms["simplex_face_coordinate_distance_loss"].item() < 1e-6
    assert matching_terms["simplex_face_boundary_lddt_loss"].item() < 1e-6
    assert matching_terms["simplex_tetra_coordinate_geometry_loss"].item() < 1e-6
    assert matching_terms["simplex_tetra_coordinate_distance_loss"].item() < 1e-6
    assert matching_terms["simplex_tetra_boundary_lddt_loss"].item() < 1e-6
    assert collapsed_terms["simplex_aux_loss"] > matching_terms["simplex_aux_loss"]
    assert collapsed_terms["simplex_face_coordinate_area_loss"] > matching_terms["simplex_face_coordinate_area_loss"]
    assert collapsed_terms["simplex_face_coordinate_distance_loss"] > matching_terms[
        "simplex_face_coordinate_distance_loss"
    ]
    assert collapsed_terms["simplex_tetra_coordinate_geometry_loss"] > matching_terms[
        "simplex_tetra_coordinate_geometry_loss"
    ]
    assert collapsed_terms["simplex_tetra_coordinate_distance_loss"] > matching_terms[
        "simplex_tetra_coordinate_distance_loss"
    ]
    assert collapsed_terms["simplex_face_boundary_lddt_loss"] > matching_terms["simplex_face_boundary_lddt_loss"]
    assert collapsed_terms["simplex_tetra_boundary_lddt_loss"] > matching_terms["simplex_tetra_boundary_lddt_loss"]


def test_cell_closure_weight_downweights_open_coordinate_realization_cells():
    true_ca = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
                [40.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    ca_mask = torch.ones(1, 4)
    collapsed_atom14 = torch.zeros(1, 4, 14, 3)
    prediction = {
        "simplex_face_indices": torch.tensor([[[[0, 1, 2], [0, 1, 3]]]]),
        "simplex_face_mask": torch.ones(1, 1, 2),
        "simplex_face_area_logits": torch.zeros(1, 1, 2),
        "atom14_coords": collapsed_atom14,
    }
    base_loss = SimplexGeometryLoss(
        contact_weight=0.0,
        topology_neighborhood_weight=0.0,
        face_area_weight=0.0,
        face_coordinate_weight=1.0,
        face_coordinate_distance_weight=1.0,
        face_distance_weight=0.0,
        tetra_geometry_weight=0.0,
        tetra_distance_weight=0.0,
        pair_face_consistency_weight=0.0,
        face_tetra_consistency_weight=0.0,
        cell_closure_weight=0.0,
        cell_closure_cutoff=8.0,
        cell_closure_temperature=0.5,
    )
    closure_loss = SimplexGeometryLoss(
        contact_weight=0.0,
        topology_neighborhood_weight=0.0,
        face_area_weight=0.0,
        face_coordinate_weight=1.0,
        face_coordinate_distance_weight=1.0,
        face_distance_weight=0.0,
        tetra_geometry_weight=0.0,
        tetra_distance_weight=0.0,
        pair_face_consistency_weight=0.0,
        face_tetra_consistency_weight=0.0,
        cell_closure_weight=1.0,
        cell_closure_cutoff=8.0,
        cell_closure_temperature=0.5,
    )

    base_terms = base_loss(prediction, true_ca, ca_mask)
    closure_terms = closure_loss(prediction, true_ca, ca_mask)

    assert closure_terms["simplex_face_coordinate_distance_loss"] < base_terms[
        "simplex_face_coordinate_distance_loss"
    ]
    assert torch.isfinite(closure_terms["simplex_aux_loss"]).all()


def test_boundary_degree_weights_normalize_repeated_edges():
    edge_indices = torch.tensor(
        [[[[[[0, 1], [0, 2], [1, 2]], [[0, 1], [0, 3], [1, 3]]]]]],
        dtype=torch.long,
    )
    edge_mask = torch.ones(1, 1, 1, 2, 3)

    weights = _boundary_degree_weights(edge_indices, edge_mask, num_residues=4)

    assert weights.shape == edge_mask.shape
    assert torch.allclose(weights[0, 0, 0, :, 0], torch.full((2,), 0.5))
    assert torch.allclose(weights[0, 0, 0, :, 1:], torch.ones(2, 2))
    assert torch.allclose(weights.sum(), torch.tensor(5.0))


def test_simplex_geometry_loss_skips_disabled_tetra_heads():
    true_ca = torch.randn(1, 5, 3)
    ca_mask = torch.ones(1, 5)
    prediction = {
        "simplex_contact_logits": torch.zeros(1, 5, 5),
        "simplex_tetra_indices": torch.tensor([[[[0, 1, 2, 3]], [[1, 2, 3, 4]], [[2, 3, 4, 0]], [[3, 4, 0, 1]], [[4, 0, 1, 2]]]]),
        "simplex_tetra_mask": torch.ones(1, 5, 1),
        "simplex_tetra_geometry_logits": torch.zeros(1, 5, 0, 3),
        "simplex_tetra_distance_logits": torch.zeros(1, 5, 0, 6, 64),
    }

    terms = SimplexGeometryLoss()(prediction, true_ca, ca_mask)

    assert "simplex_tetra_geometry_loss" not in terms
    assert "simplex_tetra_distance_loss" not in terms
    assert torch.isfinite(terms["simplex_aux_loss"]).all()


def test_tiny_alphafold2_profile_emits_simplex_training_tensors():
    from minalphafold.model import AlphaFold2
    from minalphafold.trainer import load_model_config

    torch.manual_seed(3)
    cfg = load_model_config("tiny")
    model = AlphaFold2(cfg).eval()
    batch, length, n_seq = 1, 6, 4

    with torch.no_grad():
        outputs = model(
            torch.randn(batch, length, 22),
            torch.arange(length).unsqueeze(0),
            torch.randn(batch, n_seq, length, 49),
            torch.empty(batch, 0, length, 25),
            torch.empty(batch, 0, length, length, 88),
            torch.randint(0, 20, (batch, length)),
            template_angle_feat=torch.empty(batch, 0, length, 57),
            template_mask=torch.empty(batch, 0),
            n_cycles=1,
            n_ensemble=1,
        )

    assert outputs["simplex_contact_logits"].shape == (batch, length, length)
    assert outputs["simplex_face_indices"].shape[-1] == 3
    assert outputs["simplex_tetra_indices"].shape[-1] == 4
    assert outputs["simplex_face_area_logits"].shape[:2] == (batch, length)
    assert outputs["simplex_face_distance_logits"].shape[-2:] == (3, cfg.n_dist_bins)
    assert outputs["simplex_tetra_geometry_logits"].shape[:2] == (batch, length)
    assert outputs["simplex_tetra_distance_logits"].shape[-2:] == (6, cfg.n_dist_bins)
