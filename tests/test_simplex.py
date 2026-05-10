import torch

from minalphafold.evoformer import SimplicialEvoformer
from minalphafold.simplex import (
    SimplexGeometryLoss,
    SimplicialAdapter,
    build_simplex_topology,
    face_geometry_features,
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
    simplex_local_radius = 2
    simplex_local_bias = 4.0
    simplex_long_min_sep = 8
    simplex_long_bias = 0.0
    simplex_geometry_distance_weight = 0.1
    simplex_rbf_bins = 4
    simplex_sequence_max = 16.0
    simplex_distance_max = 32.0
    simplex_area_max = 300.0
    simplex_volume_scale = 1000.0
    simplex_dropout = 0.0
    simplex_single_transition_n = 2
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
    assert adapter.msa_to_face.linear_2.weight.abs().sum() > 0


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
        face_distance_weight=0.0,
        tetra_geometry_weight=0.0,
        tetra_coordinate_weight=1.0,
        tetra_coordinate_distance_weight=1.0,
        tetra_distance_weight=0.0,
        pair_face_consistency_weight=0.0,
        face_tetra_consistency_weight=0.0,
    )

    matching_terms = loss_fn({**base_prediction, "atom14_coords": matching_atom14}, true_ca, ca_mask)
    collapsed_terms = loss_fn({**base_prediction, "atom14_coords": collapsed_atom14}, true_ca, ca_mask)

    assert matching_terms["simplex_face_coordinate_area_loss"].item() < 1e-6
    assert matching_terms["simplex_face_coordinate_distance_loss"].item() < 1e-6
    assert matching_terms["simplex_tetra_coordinate_geometry_loss"].item() < 1e-6
    assert matching_terms["simplex_tetra_coordinate_distance_loss"].item() < 1e-6
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
