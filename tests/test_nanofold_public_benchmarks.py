import torch

from minalphafold.trainer import (
    TrainingConfig,
    load_model_config,
    model_inputs_from_batch,
    simplex_edge_frame_message_runtime_scale_at_step,
    simplex_outer_edge_context_runtime_scale_at_step,
)
from scripts.run_nanofold_public_benchmarks import (
    _apply_model_config_overrides,
    _build_loss_fn,
    _simplex_topology_metrics,
    _variant_config,
    parse_args,
)


def test_full_msa_to_face_variant_keeps_tetra_and_enables_msa_faces():
    cfg = _variant_config(load_model_config("simplexfold_medium_param_matched"), "full_msa_to_face")

    assert cfg.use_simplicial_evoformer is True
    assert cfg.simplex_use_faces is True
    assert cfg.simplex_use_tetra is True
    assert cfg.simplex_use_msa_to_face is True


def test_full_msa_to_face_expansion_hinge_variant_keeps_base_topology():
    cfg = _variant_config(
        load_model_config("simplexfold_medium_param_matched"),
        "full_msa_to_face_expansion_hinge",
    )

    assert cfg.use_simplicial_evoformer is True
    assert cfg.simplex_use_faces is True
    assert cfg.simplex_use_tetra is True
    assert cfg.simplex_use_msa_to_face is True


def test_full_msa_to_face_expansion_hinge_variant_is_accepted_by_cli_parser():
    args = parse_args(["--variants", "full_msa_to_face_expansion_hinge"])

    assert args.variants == ["full_msa_to_face_expansion_hinge"]


def test_full_msa_to_face_aux_closure_variant_keeps_message_masks_unchanged():
    cfg = _variant_config(load_model_config("simplexfold_medium_param_matched"), "full_msa_to_face_aux_closure")

    assert cfg.use_simplicial_evoformer is True
    assert cfg.simplex_use_faces is True
    assert cfg.simplex_use_tetra is True
    assert cfg.simplex_use_msa_to_face is True
    assert cfg.simplex_boundary_closure_weight == 0.0


def test_full_msa_to_face_aux_closure_variant_is_accepted_by_cli_parser():
    args = parse_args(["--variants", "full_msa_to_face_aux_closure"])

    assert args.variants == ["full_msa_to_face_aux_closure"]


def test_full_msa_to_face_topology_curriculum_variant_keeps_base_topology():
    cfg = _variant_config(
        load_model_config("simplexfold_medium_param_matched"),
        "full_msa_to_face_topology_curriculum",
    )

    assert cfg.use_simplicial_evoformer is True
    assert cfg.simplex_use_faces is True
    assert cfg.simplex_use_tetra is True
    assert cfg.simplex_use_msa_to_face is True
    assert cfg.simplex_local_neighbor_k == 0


def test_topology_curriculum_flags_are_accepted_by_cli_parser():
    args = parse_args(
        [
            "--variants",
            "full_msa_to_face_topology_curriculum",
            "--simplex-local-neighbor-k",
            "4",
            "--simplex-local-neighbor-k-final",
            "0",
            "--simplex-local-neighbor-k-ramp-start-step",
            "250",
            "--simplex-local-neighbor-k-ramp-steps",
            "250",
        ]
    )

    assert args.variants == ["full_msa_to_face_topology_curriculum"]
    assert args.simplex_local_neighbor_k == 4.0
    assert args.simplex_local_neighbor_k_final == 0.0
    assert args.simplex_local_neighbor_k_ramp_start_step == 250
    assert args.simplex_local_neighbor_k_ramp_steps == 250


def test_full_msa_to_face_expanded_complex_increases_cell_coverage():
    cfg = _variant_config(
        load_model_config("simplexfold_medium_param_matched"),
        "full_msa_to_face_expanded_complex",
    )

    assert cfg.use_simplicial_evoformer is True
    assert cfg.simplex_use_faces is True
    assert cfg.simplex_use_tetra is True
    assert cfg.simplex_use_msa_to_face is True
    assert cfg.simplex_neighbor_k == 14


def test_full_msa_to_face_expanded_complex_variant_is_accepted_by_cli_parser():
    args = parse_args(["--variants", "full_msa_to_face_expanded_complex"])

    assert args.variants == ["full_msa_to_face_expanded_complex"]


def test_full_msa_to_face_cell_dropout_uses_training_subcomplex_dropout():
    cfg = _variant_config(
        load_model_config("simplexfold_medium_param_matched"),
        "full_msa_to_face_cell_dropout",
    )

    assert cfg.use_simplicial_evoformer is True
    assert cfg.simplex_use_faces is True
    assert cfg.simplex_use_tetra is True
    assert cfg.simplex_use_msa_to_face is True
    assert cfg.simplex_cell_dropout == 0.15


def test_full_msa_to_face_cell_dropout_variant_is_accepted_by_cli_parser():
    args = parse_args(["--variants", "full_msa_to_face_cell_dropout"])

    assert args.variants == ["full_msa_to_face_cell_dropout"]


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


def test_full_msa_to_face_outer_edge_context_adds_directed_edge_context():
    cfg = _variant_config(
        load_model_config("simplexfold_medium_param_matched"),
        "full_msa_to_face_outer_edge_context",
    )

    assert cfg.use_simplicial_evoformer is True
    assert cfg.simplex_use_faces is True
    assert cfg.simplex_use_tetra is True
    assert cfg.simplex_use_msa_to_face is True
    assert cfg.simplex_outer_edge_context_scale == 0.25


def test_full_msa_to_face_outer_edge_context_variant_is_accepted_by_cli_parser():
    args = parse_args(["--variants", "full_msa_to_face_outer_edge_context"])

    assert args.variants == ["full_msa_to_face_outer_edge_context"]


def test_model_config_override_flags_are_accepted_by_cli_parser():
    args = parse_args(
        [
            "--variants",
            "full_msa_to_face",
            "--simplex-outer-edge-context-scale",
            "0.25",
            "--simplex-outer-edge-context-runtime-scale",
            "0.0",
            "--simplex-outer-edge-context-runtime-scale-final",
            "0.05",
            "--simplex-outer-edge-context-runtime-scale-ramp-start-step",
            "3000",
            "--simplex-outer-edge-context-runtime-scale-ramp-steps",
            "500",
            "--simplex-edge-frame-message-runtime-scale",
            "0.0",
            "--simplex-edge-frame-message-runtime-scale-final",
            "0.05",
            "--simplex-edge-frame-message-runtime-scale-ramp-start-step",
            "3000",
            "--simplex-edge-frame-message-runtime-scale-ramp-steps",
            "500",
            "--simplex-segment-radius",
            "5",
            "--resume-model-weights-only",
        ]
    )

    assert args.variants == ["full_msa_to_face"]
    assert args.simplex_outer_edge_context_scale == 0.25
    assert args.simplex_outer_edge_context_runtime_scale == 0.0
    assert args.simplex_outer_edge_context_runtime_scale_final == 0.05
    assert args.simplex_outer_edge_context_runtime_scale_ramp_start_step == 3000
    assert args.simplex_outer_edge_context_runtime_scale_ramp_steps == 500
    assert args.simplex_edge_frame_message_runtime_scale == 0.0
    assert args.simplex_edge_frame_message_runtime_scale_final == 0.05
    assert args.simplex_edge_frame_message_runtime_scale_ramp_start_step == 3000
    assert args.simplex_edge_frame_message_runtime_scale_ramp_steps == 500
    assert args.simplex_segment_radius == 5
    assert args.resume_model_weights_only is True


def test_runtime_simplex_message_scales_ramp_and_enter_model_inputs():
    cfg = TrainingConfig(
        simplex_outer_edge_context_runtime_scale=0.0,
        simplex_outer_edge_context_runtime_scale_final=0.05,
        simplex_outer_edge_context_runtime_scale_ramp_start_step=3000,
        simplex_outer_edge_context_runtime_scale_ramp_steps=500,
        simplex_edge_frame_message_runtime_scale=0.0,
        simplex_edge_frame_message_runtime_scale_final=0.05,
        simplex_edge_frame_message_runtime_scale_ramp_start_step=3000,
        simplex_edge_frame_message_runtime_scale_ramp_steps=500,
    )
    batch = {
        "target_feat": torch.zeros(1, 4, 22),
        "residue_index": torch.arange(4).unsqueeze(0),
        "msa_feat": torch.zeros(1, 2, 4, 49),
        "extra_msa_feat": torch.zeros(1, 0, 4, 25),
        "template_pair_feat": torch.zeros(1, 0, 4, 4, 88),
        "aatype": torch.zeros(1, 4, dtype=torch.long),
        "template_angle_feat": torch.zeros(1, 0, 4, 57),
        "template_mask": torch.zeros(1, 0),
        "template_residue_mask": torch.zeros(1, 0, 4),
        "seq_mask": torch.ones(1, 4),
        "msa_mask": torch.ones(1, 2, 4),
        "extra_msa_mask": torch.ones(1, 0, 4),
    }

    assert simplex_outer_edge_context_runtime_scale_at_step(cfg, 3000) == 0.0
    assert simplex_outer_edge_context_runtime_scale_at_step(cfg, 3250) == 0.025
    assert simplex_outer_edge_context_runtime_scale_at_step(cfg, 3500) == 0.05
    assert simplex_edge_frame_message_runtime_scale_at_step(cfg, 3000) == 0.0
    assert simplex_edge_frame_message_runtime_scale_at_step(cfg, 3250) == 0.025
    assert simplex_edge_frame_message_runtime_scale_at_step(cfg, 3500) == 0.05
    inputs = model_inputs_from_batch(
        batch,
        cfg,
        use_simplex_outer_edge_context_runtime_scale=True,
        use_simplex_edge_frame_message_runtime_scale=True,
        step=3250,
    )

    assert torch.isclose(inputs["simplex_outer_edge_context_scale_override"], torch.tensor(0.025))
    assert torch.isclose(inputs["simplex_edge_frame_message_scale_override"], torch.tensor(0.025))


def test_simplex_topology_metrics_report_boundary_reuse():
    outputs = {
        "simplex_face_indices": torch.tensor([[[[0, 1, 2], [0, 1, 3]]]]),
        "simplex_face_mask": torch.ones(1, 1, 2),
        "simplex_tetra_indices": torch.tensor([[[[0, 1, 2, 3]]]]),
        "simplex_tetra_mask": torch.ones(1, 1, 1),
    }

    metrics = _simplex_topology_metrics(outputs)

    assert metrics["simplex_face_active_cells"] == [2.0]
    assert metrics["simplex_face_active_fraction"] == [1.0]
    assert abs(metrics["simplex_face_boundary_edge_mean_degree"][0] - 1.2) < 1e-6
    assert metrics["simplex_face_boundary_edge_max_degree"] == [2.0]
    assert metrics["simplex_face_boundary_unique_edge_fraction"] == [5.0 / 6.0]
    assert metrics["simplex_tetra_active_cells"] == [1.0]
    assert metrics["simplex_tetra_boundary_edge_mean_degree"] == [1.0]


def test_model_config_overrides_preserve_resume_compatible_variant_name():
    args = parse_args(
        [
            "--variants",
            "full_msa_to_face",
            "--simplex-outer-edge-context-scale",
            "0.25",
            "--simplex-segment-radius",
            "5",
        ]
    )
    cfg = _apply_model_config_overrides(
        _variant_config(load_model_config("simplexfold_medium_param_matched"), "full_msa_to_face"),
        args,
    )

    assert cfg.use_simplicial_evoformer is True
    assert cfg.simplex_use_faces is True
    assert cfg.simplex_use_tetra is True
    assert cfg.simplex_use_msa_to_face is True
    assert cfg.simplex_outer_edge_context_scale == 0.25
    assert cfg.simplex_segment_radius == 5


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


def test_full_msa_to_face_segment_cells_adds_latent_rank2_cells():
    cfg = _variant_config(
        load_model_config("simplexfold_medium_param_matched"),
        "full_msa_to_face_segment_cells",
    )

    assert cfg.use_simplicial_evoformer is True
    assert cfg.simplex_use_faces is True
    assert cfg.simplex_use_tetra is True
    assert cfg.simplex_use_msa_to_face is True
    assert cfg.simplex_segment_cell_scale == 0.25
    assert cfg.simplex_segment_radius == 4
    assert cfg.simplex_c_segment == 12


def test_full_msa_to_face_segment_cells_variant_is_accepted_by_cli_parser():
    args = parse_args(["--variants", "full_msa_to_face_segment_cells"])

    assert args.variants == ["full_msa_to_face_segment_cells"]


def test_full_msa_to_face_hodge_residual_adds_zero_parameter_face_laplacian():
    cfg = _variant_config(
        load_model_config("simplexfold_medium_param_matched"),
        "full_msa_to_face_hodge_residual",
    )

    assert cfg.use_simplicial_evoformer is True
    assert cfg.simplex_use_faces is True
    assert cfg.simplex_use_tetra is True
    assert cfg.simplex_use_msa_to_face is True
    assert cfg.simplex_hodge_face_update_scale == 0.25


def test_full_msa_to_face_hodge_residual_variant_is_accepted_by_cli_parser():
    args = parse_args(["--variants", "full_msa_to_face_hodge_residual"])

    assert args.variants == ["full_msa_to_face_hodge_residual"]


def test_full_msa_to_face_flag_closure_weights_selected_cells_by_boundary_edges():
    cfg = _variant_config(
        load_model_config("simplexfold_medium_param_matched"),
        "full_msa_to_face_flag_closure",
    )

    assert cfg.use_simplicial_evoformer is True
    assert cfg.simplex_use_faces is True
    assert cfg.simplex_use_tetra is True
    assert cfg.simplex_use_msa_to_face is True
    assert cfg.simplex_boundary_closure_weight == 0.5
    assert cfg.simplex_boundary_closure_temperature == 1.0


def test_full_msa_to_face_flag_closure_variant_is_accepted_by_cli_parser():
    args = parse_args(["--variants", "full_msa_to_face_flag_closure"])

    assert args.variants == ["full_msa_to_face_flag_closure"]


def test_full_msa_to_face_flag_closure_soft_uses_light_boundary_gate():
    cfg = _variant_config(
        load_model_config("simplexfold_medium_param_matched"),
        "full_msa_to_face_flag_closure_soft",
    )

    assert cfg.use_simplicial_evoformer is True
    assert cfg.simplex_use_faces is True
    assert cfg.simplex_use_tetra is True
    assert cfg.simplex_use_msa_to_face is True
    assert cfg.simplex_boundary_closure_weight == 0.1
    assert cfg.simplex_boundary_closure_temperature == 1.0


def test_full_msa_to_face_flag_closure_soft_variant_is_accepted_by_cli_parser():
    args = parse_args(["--variants", "full_msa_to_face_flag_closure_soft"])

    assert args.variants == ["full_msa_to_face_flag_closure_soft"]


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
            "--simplex-cell-closure-weight",
            "0.1",
            "--simplex-cell-closure-weight-final",
            "0.5",
            "--simplex-cell-closure-ramp-start-step",
            "250",
            "--simplex-cell-closure-ramp-steps",
            "250",
            "--simplex-cell-closure-cutoff",
            "12.0",
            "--simplex-cell-closure-temperature",
            "1.5",
            "--simplex-face-coordinate-expansion-weight",
            "0.4",
            "--simplex-tetra-coordinate-expansion-weight",
            "0.5",
            "--simplex-coordinate-expansion-tolerance",
            "0.05",
        ]
    )

    assert args.simplex_face_shape_weight == 0.2
    assert args.simplex_face_normal_weight == 0.1
    assert args.simplex_tetra_shape_weight == 0.3
    assert args.simplex_topology_margin_weight == 0.05
    assert args.simplex_topology_margin == 1.25
    assert args.simplex_topology_margin_hard_negatives == 4
    assert args.simplex_cell_closure_weight == 0.1
    assert args.simplex_cell_closure_weight_final == 0.5
    assert args.simplex_cell_closure_ramp_start_step == 250
    assert args.simplex_cell_closure_ramp_steps == 250
    assert args.simplex_cell_closure_cutoff == 12.0
    assert args.simplex_cell_closure_temperature == 1.5
    assert args.simplex_face_coordinate_expansion_weight == 0.4
    assert args.simplex_tetra_coordinate_expansion_weight == 0.5
    assert args.simplex_coordinate_expansion_tolerance == 0.05


def test_benchmark_loss_builder_applies_topology_margin_config():
    loss_fn = _build_loss_fn(
        TrainingConfig(
            simplex_face_shape_weight=0.2,
            simplex_face_normal_weight=0.1,
            simplex_tetra_shape_weight=0.3,
            simplex_topology_margin_weight=0.05,
            simplex_topology_margin=1.25,
            simplex_topology_margin_hard_negatives=4,
            simplex_cell_closure_weight=0.25,
            simplex_cell_closure_cutoff=12.0,
            simplex_cell_closure_temperature=1.5,
            simplex_face_coordinate_expansion_weight=0.4,
            simplex_tetra_coordinate_expansion_weight=0.5,
            simplex_coordinate_expansion_tolerance=0.05,
        )
    )

    assert loss_fn.simplex_geometry_loss.face_shape_weight == 0.2
    assert loss_fn.simplex_geometry_loss.face_normal_weight == 0.1
    assert loss_fn.simplex_geometry_loss.tetra_shape_weight == 0.3
    assert loss_fn.simplex_geometry_loss.topology_margin_weight == 0.05
    assert loss_fn.simplex_geometry_loss.topology_margin == 1.25
    assert loss_fn.simplex_geometry_loss.topology_margin_hard_negatives == 4
    assert loss_fn.simplex_geometry_loss.cell_closure_weight == 0.25
    assert loss_fn.simplex_geometry_loss.cell_closure_cutoff == 12.0
    assert loss_fn.simplex_geometry_loss.cell_closure_temperature == 1.5
    assert loss_fn.simplex_geometry_loss.face_coordinate_expansion_weight == 0.4
    assert loss_fn.simplex_geometry_loss.tetra_coordinate_expansion_weight == 0.5
    assert loss_fn.simplex_geometry_loss.coordinate_expansion_tolerance == 0.05
