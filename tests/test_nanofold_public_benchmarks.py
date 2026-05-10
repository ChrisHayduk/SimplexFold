from minalphafold.trainer import TrainingConfig, load_model_config
from scripts.run_nanofold_public_benchmarks import _build_loss_fn, _variant_config, parse_args


def test_full_msa_to_face_variant_keeps_tetra_and_enables_msa_faces():
    cfg = _variant_config(load_model_config("simplexfold_medium_param_matched"), "full_msa_to_face")

    assert cfg.use_simplicial_evoformer is True
    assert cfg.simplex_use_faces is True
    assert cfg.simplex_use_tetra is True
    assert cfg.simplex_use_msa_to_face is True


def test_full_msa_to_face_long_variant_adds_long_range_topology_bias():
    cfg = _variant_config(load_model_config("simplexfold_medium_param_matched"), "full_msa_to_face_long")

    assert cfg.use_simplicial_evoformer is True
    assert cfg.simplex_use_faces is True
    assert cfg.simplex_use_tetra is True
    assert cfg.simplex_use_msa_to_face is True
    assert cfg.simplex_long_min_sep == 16
    assert cfg.simplex_long_bias == 2.0


def test_full_msa_to_face_mixed_variant_reserves_local_scaffold_slots():
    cfg = _variant_config(load_model_config("simplexfold_medium_param_matched"), "full_msa_to_face_mixed")

    assert cfg.use_simplicial_evoformer is True
    assert cfg.simplex_use_faces is True
    assert cfg.simplex_use_tetra is True
    assert cfg.simplex_use_msa_to_face is True
    assert cfg.simplex_local_neighbor_k == 4
    assert cfg.simplex_local_bias == 0.0


def test_full_msa_to_face_mixed_soft_variant_keeps_reduced_local_bias():
    cfg = _variant_config(load_model_config("simplexfold_medium_param_matched"), "full_msa_to_face_mixed_soft")

    assert cfg.use_simplicial_evoformer is True
    assert cfg.simplex_use_faces is True
    assert cfg.simplex_use_tetra is True
    assert cfg.simplex_use_msa_to_face is True
    assert cfg.simplex_local_neighbor_k == 4
    assert cfg.simplex_local_bias == 2.0


def test_full_msa_to_face_strong_messages_scales_simplex_residuals():
    cfg = _variant_config(load_model_config("simplexfold_medium_param_matched"), "full_msa_to_face_strong_messages")

    assert cfg.use_simplicial_evoformer is True
    assert cfg.simplex_use_faces is True
    assert cfg.simplex_use_tetra is True
    assert cfg.simplex_use_msa_to_face is True
    assert cfg.simplex_pair_update_scale == 1.5
    assert cfg.simplex_single_update_scale == 1.5


def test_full_msa_to_face_damped_messages_scales_simplex_residuals_down():
    cfg = _variant_config(load_model_config("simplexfold_medium_param_matched"), "full_msa_to_face_damped_messages")

    assert cfg.use_simplicial_evoformer is True
    assert cfg.simplex_use_faces is True
    assert cfg.simplex_use_tetra is True
    assert cfg.simplex_use_msa_to_face is True
    assert cfg.simplex_pair_update_scale == 0.5
    assert cfg.simplex_single_update_scale == 0.5


def test_full_msa_to_face_edge_messages_biases_updates_to_pair_stream():
    cfg = _variant_config(load_model_config("simplexfold_medium_param_matched"), "full_msa_to_face_edge_messages")

    assert cfg.use_simplicial_evoformer is True
    assert cfg.simplex_use_faces is True
    assert cfg.simplex_use_tetra is True
    assert cfg.simplex_use_msa_to_face is True
    assert cfg.simplex_pair_update_scale == 1.5
    assert cfg.simplex_single_update_scale == 0.5


def test_full_msa_to_face_no_recycled_topology_keeps_latent_selector():
    cfg = _variant_config(
        load_model_config("simplexfold_medium_param_matched"),
        "full_msa_to_face_no_recycled_topology",
    )

    assert cfg.use_simplicial_evoformer is True
    assert cfg.simplex_use_faces is True
    assert cfg.simplex_use_tetra is True
    assert cfg.simplex_use_msa_to_face is True
    assert cfg.simplex_use_recycled_geometry is False


def test_full_msa_to_face_structure_readout_adds_simplicial_structure_path():
    cfg = _variant_config(
        load_model_config("simplexfold_medium_param_matched"),
        "full_msa_to_face_structure_readout",
    )

    assert cfg.use_simplicial_evoformer is True
    assert cfg.simplex_use_faces is True
    assert cfg.simplex_use_tetra is True
    assert cfg.simplex_use_msa_to_face is True
    assert cfg.simplex_structure_readout_scale == 0.25


def test_structure_readout_variant_is_accepted_by_cli_parser():
    args = parse_args(["--variants", "full_msa_to_face_structure_readout"])

    assert args.variants == ["full_msa_to_face_structure_readout"]


def test_full_msa_to_face_structure_readout_only_disables_trunk_residuals():
    cfg = _variant_config(
        load_model_config("simplexfold_medium_param_matched"),
        "full_msa_to_face_structure_readout_only",
    )

    assert cfg.use_simplicial_evoformer is True
    assert cfg.simplex_use_faces is True
    assert cfg.simplex_use_tetra is True
    assert cfg.simplex_use_msa_to_face is True
    assert cfg.simplex_pair_update_scale == 0.0
    assert cfg.simplex_single_update_scale == 0.0
    assert cfg.simplex_structure_readout_scale == 0.5


def test_full_msa_to_face_outer_edge_enables_cell_neighborhood_update():
    cfg = _variant_config(
        load_model_config("simplexfold_medium_param_matched"),
        "full_msa_to_face_outer_edge",
    )

    assert cfg.use_simplicial_evoformer is True
    assert cfg.simplex_use_faces is True
    assert cfg.simplex_use_tetra is True
    assert cfg.simplex_use_msa_to_face is True
    assert cfg.simplex_outer_edge_update_scale == 0.25


def test_full_msa_to_face_outer_edge_variant_is_accepted_by_cli_parser():
    args = parse_args(["--variants", "full_msa_to_face_outer_edge"])

    assert args.variants == ["full_msa_to_face_outer_edge"]


def test_full_msa_to_face_edge_frame_messages_adds_local_frame_readout():
    cfg = _variant_config(
        load_model_config("simplexfold_medium_param_matched"),
        "full_msa_to_face_edge_frame_messages",
    )

    assert cfg.use_simplicial_evoformer is True
    assert cfg.simplex_use_faces is True
    assert cfg.simplex_use_tetra is True
    assert cfg.simplex_use_msa_to_face is True
    assert cfg.simplex_edge_frame_message_scale == 0.25


def test_full_msa_to_face_edge_frame_messages_variant_is_accepted_by_cli_parser():
    args = parse_args(["--variants", "full_msa_to_face_edge_frame_messages"])

    assert args.variants == ["full_msa_to_face_edge_frame_messages"]


def test_structure_readout_only_variant_is_accepted_by_cli_parser():
    args = parse_args(["--variants", "full_msa_to_face_structure_readout_only"])

    assert args.variants == ["full_msa_to_face_structure_readout_only"]


def test_face_structure_readout_only_uses_2_skeleton_sidecar():
    cfg = _variant_config(
        load_model_config("simplexfold_medium_param_matched"),
        "face_structure_readout_only",
    )

    assert cfg.use_simplicial_evoformer is True
    assert cfg.simplex_use_faces is True
    assert cfg.simplex_use_tetra is False
    assert cfg.simplex_use_msa_to_face is True
    assert cfg.simplex_pair_update_scale == 0.0
    assert cfg.simplex_single_update_scale == 0.0
    assert cfg.simplex_structure_readout_scale == 0.5


def test_face_structure_readout_only_variant_is_accepted_by_cli_parser():
    args = parse_args(["--variants", "face_structure_readout_only"])

    assert args.variants == ["face_structure_readout_only"]


def test_topology_margin_args_are_accepted_by_cli_parser():
    args = parse_args(
        [
            "--simplex-face-shape-weight",
            "0.2",
            "--simplex-face-normal-weight",
            "0.1",
            "--simplex-tetra-shape-weight",
            "0.3",
            "--simplex-topology-margin-weight",
            "0.05",
            "--simplex-topology-margin",
            "1.25",
            "--simplex-topology-margin-hard-negatives",
            "4",
        ]
    )

    assert args.simplex_face_shape_weight == 0.2
    assert args.simplex_face_normal_weight == 0.1
    assert args.simplex_tetra_shape_weight == 0.3
    assert args.simplex_topology_margin_weight == 0.05
    assert args.simplex_topology_margin == 1.25
    assert args.simplex_topology_margin_hard_negatives == 4


def test_benchmark_loss_builder_applies_topology_margin_config():
    loss_fn = _build_loss_fn(
        TrainingConfig(
            simplex_face_shape_weight=0.2,
            simplex_face_normal_weight=0.1,
            simplex_tetra_shape_weight=0.3,
            simplex_topology_margin_weight=0.05,
            simplex_topology_margin=1.25,
            simplex_topology_margin_hard_negatives=4,
        )
    )

    assert loss_fn.simplex_geometry_loss.face_shape_weight == 0.2
    assert loss_fn.simplex_geometry_loss.face_normal_weight == 0.1
    assert loss_fn.simplex_geometry_loss.tetra_shape_weight == 0.3
    assert loss_fn.simplex_geometry_loss.topology_margin_weight == 0.05
    assert loss_fn.simplex_geometry_loss.topology_margin == 1.25
    assert loss_fn.simplex_geometry_loss.topology_margin_hard_negatives == 4
