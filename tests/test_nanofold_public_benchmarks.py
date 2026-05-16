from pathlib import Path

import pytest
import torch

from minalphafold.trainer import (
    TrainingConfig,
    load_model_config,
    model_inputs_from_batch,
    simplex_boundary_cochain_recycling_runtime_scale_at_step,
    simplex_boundary_edge_frame_gate_runtime_scale_at_step,
    simplex_boundary_metric_gate_runtime_scale_at_step,
    simplex_boundary_metric_recycling_runtime_scale_at_step,
    simplex_boundary_pair_feedback_runtime_scale_at_step,
    simplex_boundary_pair_gate_runtime_scale_at_step,
    simplex_boundary_edge_star_residual_runtime_scale_at_step,
    simplex_boundary_edge_star_readout_runtime_scale_at_step,
    simplex_boundary_face_cyclic_readout_runtime_scale_at_step,
    simplex_boundary_hodge_readout_runtime_scale_at_step,
    simplex_boundary_oriented_cochain_runtime_scale_at_step,
    simplex_boundary_readout_directionality_runtime_scale_at_step,
    simplex_boundary_signed_face_cyclic_readout_runtime_scale_at_step,
    simplex_cell_score_outer_edge_weight_at_step,
    simplex_edge_star_context_runtime_scale_at_step,
    simplex_edge_frame_message_runtime_scale_at_step,
    simplex_face_top_k_at_step,
    simplex_geometry_distance_weight_at_step,
    simplex_hodge_face_runtime_scale_at_step,
    simplex_msa_feedback_runtime_scale_at_step,
    simplex_outer_edge_context_runtime_scale_at_step,
    simplex_outer_edge_residual_context_runtime_scale_at_step,
    simplex_pair_update_runtime_scale_at_step,
    simplex_pre_triangle_single_update_runtime_scale_at_step,
    simplex_pre_triangle_update_runtime_scale_at_step,
    simplex_segment_cell_runtime_scale_at_step,
    simplex_signed_tetra_coboundary_runtime_scale_at_step,
    simplex_signed_tetra_to_face_runtime_scale_at_step,
    simplex_single_update_runtime_scale_at_step,
    simplex_tetra_top_k_at_step,
    simplex_triangle_attention_bias_runtime_scale_at_step,
    simplex_triangle_attention_value_runtime_scale_at_step,
    simplex_vertex_star_context_runtime_scale_at_step,
)
from scripts.run_nanofold_public_benchmarks import (
    _apply_model_config_overrides,
    _build_loss_fn,
    _enforce_parameter_budget,
    _evaluate,
    _run_status_payload,
    _should_force_microbatch_status,
    _write_run_status_file,
    _simplex_boundary_geometry_metrics,
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


def test_max_parameters_guard_is_accepted_and_enforced():
    args = parse_args(["--max-parameters", "3261974"])

    assert args.max_parameters == 3_261_974
    _enforce_parameter_budget(
        variant="full_msa_to_face",
        parameter_count=3_261_974,
        max_parameters=args.max_parameters,
    )
    with pytest.raises(ValueError, match="exceeding --max-parameters"):
        _enforce_parameter_budget(
            variant="full_msa_to_face",
            parameter_count=3_261_975,
            max_parameters=args.max_parameters,
        )


def test_num_workers_guardrail_is_accepted_by_cli_parser():
    args = parse_args(["--num-workers", "4"])

    assert args.num_workers == 4


def test_run_status_payload_tracks_live_progress(tmp_path):
    payload = _run_status_payload(
        variant="full_msa_to_face",
        phase="training",
        completed_step=125,
        target_steps=9000,
        start_step=101,
        total_examples=1000,
        effective_batch_size=8,
        num_workers=4,
        elapsed_seconds_total=12.5,
        elapsed_seconds_run=7.5,
        history=[{"step": 100}, {"step": 125}],
        train_losses=[3.2, 3.0],
        latest_checkpoint_path=tmp_path / "full_msa_to_face_latest.pt",
        stopped_early=False,
        active_step=126,
        active_microbatch=3,
        active_microbatches=8,
        active_eval_batch=17,
        active_eval_batches=1000,
        active_eval_examples=16,
    )

    assert payload["variant"] == "full_msa_to_face"
    assert payload["phase"] == "training"
    assert payload["completed_step"] == 125
    assert payload["target_steps"] == 9000
    assert payload["effective_batch_size"] == 8
    assert payload["num_workers"] == 4
    assert payload["elapsed_seconds_run"] == 7.5
    assert payload["last_history_step"] == 125
    assert payload["last_train_loss"] == 3.0
    assert payload["latest_checkpoint"].endswith("full_msa_to_face_latest.pt")
    assert payload["stopped_early"] is False
    assert payload["active_step"] == 126
    assert payload["active_microbatch"] == 3
    assert payload["active_microbatches"] == 8
    assert payload["active_eval_batch"] == 17
    assert payload["active_eval_batches"] == 1000
    assert payload["active_eval_examples"] == 16


def test_write_run_status_file_is_best_effort_and_preserves_prior_status(tmp_path, monkeypatch, capsys):
    status_path = tmp_path / "status_full_msa_to_face.json"
    status_path.write_text('{"phase":"old"}', encoding="utf-8")
    real_write_text = Path.write_text

    def fail_tmp_write(self, data, *args, **kwargs):
        if self.name == ".status_full_msa_to_face.json.tmp":
            raise OSError("simulated status I/O")
        return real_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_tmp_write)

    assert _write_run_status_file(status_path, {"phase": "new"}) is False
    assert status_path.read_text(encoding="utf-8") == '{"phase":"old"}'
    assert "warning: could not write run status" in capsys.readouterr().err


def test_final_step_forces_every_microbatch_status_phase():
    assert _should_force_microbatch_status(microbatch_index=0, is_final_step=False) is True
    assert _should_force_microbatch_status(microbatch_index=4, is_final_step=False) is False
    assert _should_force_microbatch_status(microbatch_index=4, is_final_step=True) is True


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
            "--simplex-pair-update-runtime-scale",
            "0.5",
            "--simplex-pair-update-runtime-scale-final",
            "1.0",
            "--simplex-pair-update-runtime-scale-ramp-start-step",
            "3000",
            "--simplex-pair-update-runtime-scale-ramp-steps",
            "500",
            "--simplex-single-update-runtime-scale",
            "1.0",
            "--simplex-single-update-runtime-scale-final",
            "0.25",
            "--simplex-single-update-runtime-scale-ramp-start-step",
            "3000",
            "--simplex-single-update-runtime-scale-ramp-steps",
            "500",
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
            "--simplex-outer-edge-residual-context-runtime-scale",
            "0.0",
            "--simplex-outer-edge-residual-context-runtime-scale-final",
            "0.25",
            "--simplex-outer-edge-residual-context-runtime-scale-ramp-start-step",
            "3000",
            "--simplex-outer-edge-residual-context-runtime-scale-ramp-steps",
            "500",
            "--simplex-edge-frame-message-runtime-scale",
            "0.0",
            "--simplex-edge-frame-message-runtime-scale-final",
            "0.05",
            "--simplex-edge-frame-message-runtime-scale-ramp-start-step",
            "3000",
            "--simplex-edge-frame-message-runtime-scale-ramp-steps",
            "500",
            "--simplex-boundary-readout-directionality-runtime-scale",
            "0.0",
            "--simplex-boundary-readout-directionality-runtime-scale-final",
            "0.5",
            "--simplex-boundary-readout-directionality-runtime-scale-ramp-start-step",
            "3000",
            "--simplex-boundary-readout-directionality-runtime-scale-ramp-steps",
            "500",
            "--simplex-hodge-face-runtime-scale",
            "0.0",
            "--simplex-hodge-face-runtime-scale-final",
            "0.05",
            "--simplex-hodge-face-runtime-scale-ramp-start-step",
            "3000",
            "--simplex-hodge-face-runtime-scale-ramp-steps",
            "500",
            "--simplex-segment-cell-runtime-scale",
            "0.0",
            "--simplex-segment-cell-runtime-scale-final",
            "0.05",
            "--simplex-segment-cell-runtime-scale-ramp-start-step",
            "3000",
            "--simplex-segment-cell-runtime-scale-ramp-steps",
            "500",
            "--simplex-structure-pair-readout-scale",
            "0.125",
            "--simplex-msa-feedback-scale",
            "0.25",
            "--simplex-boundary-msa-feedback-scale",
            "0.125",
            "--simplex-boundary-pair-feedback-scale",
            "0.0625",
            "--simplex-boundary-pair-gate-scale",
            "0.03125",
            "--simplex-boundary-metric-gate-scale",
            "0.5",
            "--simplex-boundary-metric-recycling-scale",
            "0.125",
            "--simplex-boundary-cochain-recycling-scale",
            "0.0625",
            "--simplex-boundary-cochain-recycling-metric-gate-scale",
            "0.5",
            "--simplex-msa-feedback-runtime-scale",
            "0.0",
            "--simplex-msa-feedback-runtime-scale-final",
            "0.05",
            "--simplex-msa-feedback-runtime-scale-ramp-start-step",
            "3000",
            "--simplex-msa-feedback-runtime-scale-ramp-steps",
            "500",
            "--simplex-boundary-pair-feedback-runtime-scale",
            "0.0",
            "--simplex-boundary-pair-feedback-runtime-scale-final",
            "0.025",
            "--simplex-boundary-pair-feedback-runtime-scale-ramp-start-step",
            "3000",
            "--simplex-boundary-pair-feedback-runtime-scale-ramp-steps",
            "500",
            "--simplex-boundary-pair-gate-runtime-scale",
            "0.0",
            "--simplex-boundary-pair-gate-runtime-scale-final",
            "0.025",
            "--simplex-boundary-pair-gate-runtime-scale-ramp-start-step",
            "3000",
            "--simplex-boundary-pair-gate-runtime-scale-ramp-steps",
            "500",
            "--simplex-boundary-metric-gate-runtime-scale",
            "0.0",
            "--simplex-boundary-metric-gate-runtime-scale-final",
            "0.25",
            "--simplex-boundary-metric-gate-runtime-scale-ramp-start-step",
            "3000",
            "--simplex-boundary-metric-gate-runtime-scale-ramp-steps",
            "500",
            "--simplex-boundary-metric-recycling-runtime-scale",
            "0.0",
            "--simplex-boundary-metric-recycling-runtime-scale-final",
            "0.125",
            "--simplex-boundary-metric-recycling-runtime-scale-ramp-start-step",
            "3000",
            "--simplex-boundary-metric-recycling-runtime-scale-ramp-steps",
            "500",
            "--simplex-boundary-cochain-recycling-runtime-scale",
            "0.0",
            "--simplex-boundary-cochain-recycling-runtime-scale-final",
            "0.0625",
            "--simplex-boundary-cochain-recycling-runtime-scale-ramp-start-step",
            "3000",
            "--simplex-boundary-cochain-recycling-runtime-scale-ramp-steps",
            "500",
            "--simplex-geometry-distance-weight",
            "0.1",
            "--simplex-geometry-distance-weight-final",
            "0.025",
            "--simplex-geometry-distance-weight-ramp-start-step",
            "3000",
            "--simplex-geometry-distance-weight-ramp-steps",
            "500",
            "--simplex-boundary-message-degree-attenuation",
            "0.5",
            "--simplex-boundary-incidence-normalization",
            "1.0",
            "--simplex-boundary-readout-directionality",
            "0.5",
            "--simplex-boundary-hodge-readout-scale",
            "0.25",
            "--simplex-boundary-edge-star-readout-scale",
            "0.5",
            "--simplex-boundary-edge-star-residual-scale",
            "0.25",
            "--simplex-boundary-oriented-cochain-scale",
            "0.25",
            "--simplex-boundary-face-cyclic-readout-scale",
            "0.5",
            "--simplex-boundary-signed-face-cyclic-readout-scale",
            "0.25",
            "--simplex-signed-tetra-coboundary-scale",
            "0.125",
            "--simplex-signed-tetra-coboundary-runtime-scale",
            "0.0",
            "--simplex-signed-tetra-coboundary-runtime-scale-final",
            "0.125",
            "--simplex-signed-tetra-coboundary-runtime-scale-ramp-start-step",
            "3000",
            "--simplex-signed-tetra-coboundary-runtime-scale-ramp-steps",
            "500",
            "--simplex-signed-tetra-to-face-scale",
            "0.25",
            "--simplex-signed-tetra-to-face-runtime-scale",
            "0.0",
            "--simplex-signed-tetra-to-face-runtime-scale-final",
            "0.25",
            "--simplex-signed-tetra-to-face-runtime-scale-ramp-start-step",
            "3000",
            "--simplex-signed-tetra-to-face-runtime-scale-ramp-steps",
            "500",
            "--simplex-boundary-edge-star-residual-runtime-scale",
            "0.0",
            "--simplex-boundary-edge-star-residual-runtime-scale-final",
            "0.25",
            "--simplex-boundary-edge-star-residual-runtime-scale-ramp-start-step",
            "3000",
            "--simplex-boundary-edge-star-residual-runtime-scale-ramp-steps",
            "500",
            "--simplex-boundary-oriented-cochain-runtime-scale",
            "0.0",
            "--simplex-boundary-oriented-cochain-runtime-scale-final",
            "0.25",
            "--simplex-boundary-oriented-cochain-runtime-scale-ramp-start-step",
            "3000",
            "--simplex-boundary-oriented-cochain-runtime-scale-ramp-steps",
            "500",
            "--simplex-boundary-face-cyclic-readout-runtime-scale",
            "0.0",
            "--simplex-boundary-face-cyclic-readout-runtime-scale-final",
            "0.5",
            "--simplex-boundary-face-cyclic-readout-runtime-scale-ramp-start-step",
            "3000",
            "--simplex-boundary-face-cyclic-readout-runtime-scale-ramp-steps",
            "500",
            "--simplex-boundary-signed-face-cyclic-readout-runtime-scale",
            "0.0",
            "--simplex-boundary-signed-face-cyclic-readout-runtime-scale-final",
            "0.25",
            "--simplex-boundary-signed-face-cyclic-readout-runtime-scale-ramp-start-step",
            "3000",
            "--simplex-boundary-signed-face-cyclic-readout-runtime-scale-ramp-steps",
            "500",
            "--simplex-boundary-edge-frame-gate-scale",
            "0.05",
            "--simplex-boundary-edge-frame-gate-runtime-scale",
            "0.0",
            "--simplex-boundary-edge-frame-gate-runtime-scale-final",
            "0.05",
            "--simplex-boundary-edge-frame-gate-runtime-scale-ramp-start-step",
            "3000",
            "--simplex-boundary-edge-frame-gate-runtime-scale-ramp-steps",
            "500",
            "--simplex-boundary-hodge-readout-runtime-scale",
            "0.0",
            "--simplex-boundary-hodge-readout-runtime-scale-final",
            "0.25",
            "--simplex-boundary-hodge-readout-runtime-scale-ramp-start-step",
            "3000",
            "--simplex-boundary-hodge-readout-runtime-scale-ramp-steps",
            "500",
            "--simplex-boundary-edge-star-readout-runtime-scale",
            "0.0",
            "--simplex-boundary-edge-star-readout-runtime-scale-final",
            "0.5",
            "--simplex-boundary-edge-star-readout-runtime-scale-ramp-start-step",
            "3000",
            "--simplex-boundary-edge-star-readout-runtime-scale-ramp-steps",
            "500",
            "--simplex-global-context-scale",
            "0.125",
            "--simplex-vertex-star-context-scale",
            "0.75",
            "--simplex-edge-star-context-scale",
            "0.5",
            "--simplex-pre-triangle-update-scale",
            "0.25",
            "--simplex-pre-triangle-single-update-scale",
            "0.0",
            "--simplex-pre-triangle-update-runtime-scale",
            "0.0",
            "--simplex-pre-triangle-update-runtime-scale-final",
            "0.25",
            "--simplex-pre-triangle-update-runtime-scale-ramp-start-step",
            "3000",
            "--simplex-pre-triangle-update-runtime-scale-ramp-steps",
            "500",
            "--simplex-pre-triangle-single-update-runtime-scale",
            "0.0",
            "--simplex-pre-triangle-single-update-runtime-scale-final",
            "0.0",
            "--simplex-pre-triangle-single-update-runtime-scale-ramp-start-step",
            "3000",
            "--simplex-pre-triangle-single-update-runtime-scale-ramp-steps",
            "500",
            "--simplex-triangle-attention-bias-scale",
            "0.05",
            "--simplex-triangle-attention-value-scale",
            "0.025",
            "--simplex-triangle-attention-bias-runtime-scale",
            "0.0",
            "--simplex-triangle-attention-bias-runtime-scale-final",
            "0.0125",
            "--simplex-triangle-attention-bias-runtime-scale-ramp-start-step",
            "3000",
            "--simplex-triangle-attention-bias-runtime-scale-ramp-steps",
            "500",
            "--simplex-triangle-attention-value-runtime-scale",
            "0.025",
            "--simplex-triangle-attention-value-runtime-scale-final",
            "0.0",
            "--simplex-triangle-attention-value-runtime-scale-ramp-start-step",
            "3000",
            "--simplex-triangle-attention-value-runtime-scale-ramp-steps",
            "500",
            "--simplex-vertex-star-context-runtime-scale",
            "0.0",
            "--simplex-vertex-star-context-runtime-scale-final",
            "1.0",
            "--simplex-vertex-star-context-runtime-scale-ramp-start-step",
            "3000",
            "--simplex-vertex-star-context-runtime-scale-ramp-steps",
            "500",
            "--simplex-edge-star-context-runtime-scale",
            "1.0",
            "--simplex-edge-star-context-runtime-scale-final",
            "0.0",
            "--simplex-edge-star-context-runtime-scale-ramp-start-step",
            "3000",
            "--simplex-edge-star-context-runtime-scale-ramp-steps",
            "500",
            "--simplex-face-boundary-lddt-weight-final",
            "0.025",
            "--simplex-tetra-boundary-lddt-weight-final",
            "0.025",
            "--simplex-boundary-lddt-ramp-start-step",
            "3500",
            "--simplex-boundary-lddt-ramp-steps",
            "500",
            "--simplex-segment-radius",
            "5",
            "--simplex-face-top-k",
            "24",
            "--simplex-face-top-k-final",
            "12",
            "--simplex-face-top-k-ramp-start-step",
            "3500",
            "--simplex-face-top-k-ramp-steps",
            "500",
            "--simplex-tetra-top-k",
            "48",
            "--simplex-tetra-top-k-final",
            "24",
            "--simplex-tetra-top-k-ramp-start-step",
            "3500",
            "--simplex-tetra-top-k-ramp-steps",
            "500",
            "--simplex-cell-score-degree-penalty",
            "0.75",
            "--simplex-cell-score-outer-edge-weight",
            "0.25",
            "--simplex-cell-score-segment-weight",
            "0.125",
            "--simplex-cell-score-outer-edge-weight-final",
            "0.5",
            "--simplex-cell-score-outer-edge-weight-ramp-start-step",
            "3500",
            "--simplex-cell-score-outer-edge-weight-ramp-steps",
            "500",
            "--resume-model-weights-only",
        ]
    )

    assert args.variants == ["full_msa_to_face"]
    assert args.simplex_pair_update_runtime_scale == 0.5
    assert args.simplex_pair_update_runtime_scale_final == 1.0
    assert args.simplex_pair_update_runtime_scale_ramp_start_step == 3000
    assert args.simplex_pair_update_runtime_scale_ramp_steps == 500
    assert args.simplex_single_update_runtime_scale == 1.0
    assert args.simplex_single_update_runtime_scale_final == 0.25
    assert args.simplex_single_update_runtime_scale_ramp_start_step == 3000
    assert args.simplex_single_update_runtime_scale_ramp_steps == 500
    assert args.simplex_outer_edge_context_scale == 0.25
    assert args.simplex_outer_edge_context_runtime_scale == 0.0
    assert args.simplex_outer_edge_context_runtime_scale_final == 0.05
    assert args.simplex_outer_edge_context_runtime_scale_ramp_start_step == 3000
    assert args.simplex_outer_edge_context_runtime_scale_ramp_steps == 500
    assert args.simplex_edge_frame_message_runtime_scale == 0.0
    assert args.simplex_edge_frame_message_runtime_scale_final == 0.05
    assert args.simplex_edge_frame_message_runtime_scale_ramp_start_step == 3000
    assert args.simplex_edge_frame_message_runtime_scale_ramp_steps == 500
    assert args.simplex_boundary_readout_directionality_runtime_scale == 0.0
    assert args.simplex_boundary_readout_directionality_runtime_scale_final == 0.5
    assert args.simplex_boundary_readout_directionality_runtime_scale_ramp_start_step == 3000
    assert args.simplex_boundary_readout_directionality_runtime_scale_ramp_steps == 500
    assert args.simplex_boundary_hodge_readout_runtime_scale == 0.0
    assert args.simplex_boundary_hodge_readout_runtime_scale_final == 0.25
    assert args.simplex_boundary_hodge_readout_runtime_scale_ramp_start_step == 3000
    assert args.simplex_boundary_hodge_readout_runtime_scale_ramp_steps == 500
    assert args.simplex_boundary_edge_star_readout_runtime_scale == 0.0
    assert args.simplex_boundary_edge_star_readout_runtime_scale_final == 0.5
    assert args.simplex_boundary_edge_star_readout_runtime_scale_ramp_start_step == 3000
    assert args.simplex_boundary_edge_star_readout_runtime_scale_ramp_steps == 500
    assert args.simplex_hodge_face_runtime_scale == 0.0
    assert args.simplex_hodge_face_runtime_scale_final == 0.05
    assert args.simplex_hodge_face_runtime_scale_ramp_start_step == 3000
    assert args.simplex_hodge_face_runtime_scale_ramp_steps == 500
    assert args.simplex_segment_cell_runtime_scale == 0.0
    assert args.simplex_segment_cell_runtime_scale_final == 0.05
    assert args.simplex_segment_cell_runtime_scale_ramp_start_step == 3000
    assert args.simplex_segment_cell_runtime_scale_ramp_steps == 500
    assert args.simplex_structure_pair_readout_scale == 0.125
    assert args.simplex_msa_feedback_scale == 0.25
    assert args.simplex_boundary_msa_feedback_scale == 0.125
    assert args.simplex_boundary_pair_feedback_scale == 0.0625
    assert args.simplex_boundary_pair_gate_scale == 0.03125
    assert args.simplex_boundary_metric_gate_scale == 0.5
    assert args.simplex_boundary_metric_recycling_scale == 0.125
    assert args.simplex_boundary_cochain_recycling_scale == 0.0625
    assert args.simplex_boundary_cochain_recycling_metric_gate_scale == 0.5
    assert args.simplex_msa_feedback_runtime_scale == 0.0
    assert args.simplex_msa_feedback_runtime_scale_final == 0.05
    assert args.simplex_msa_feedback_runtime_scale_ramp_start_step == 3000
    assert args.simplex_msa_feedback_runtime_scale_ramp_steps == 500
    assert args.simplex_boundary_pair_feedback_runtime_scale == 0.0
    assert args.simplex_boundary_pair_feedback_runtime_scale_final == 0.025
    assert args.simplex_boundary_pair_feedback_runtime_scale_ramp_start_step == 3000
    assert args.simplex_boundary_pair_feedback_runtime_scale_ramp_steps == 500
    assert args.simplex_boundary_pair_gate_runtime_scale == 0.0
    assert args.simplex_boundary_pair_gate_runtime_scale_final == 0.025
    assert args.simplex_boundary_pair_gate_runtime_scale_ramp_start_step == 3000
    assert args.simplex_boundary_pair_gate_runtime_scale_ramp_steps == 500
    assert args.simplex_boundary_metric_gate_runtime_scale == 0.0
    assert args.simplex_boundary_metric_gate_runtime_scale_final == 0.25
    assert args.simplex_boundary_metric_gate_runtime_scale_ramp_start_step == 3000
    assert args.simplex_boundary_metric_gate_runtime_scale_ramp_steps == 500
    assert args.simplex_boundary_metric_recycling_runtime_scale == 0.0
    assert args.simplex_boundary_metric_recycling_runtime_scale_final == 0.125
    assert args.simplex_boundary_metric_recycling_runtime_scale_ramp_start_step == 3000
    assert args.simplex_boundary_metric_recycling_runtime_scale_ramp_steps == 500
    assert args.simplex_boundary_cochain_recycling_runtime_scale == 0.0
    assert args.simplex_boundary_cochain_recycling_runtime_scale_final == 0.0625
    assert args.simplex_boundary_cochain_recycling_runtime_scale_ramp_start_step == 3000
    assert args.simplex_boundary_cochain_recycling_runtime_scale_ramp_steps == 500
    assert args.simplex_geometry_distance_weight == 0.1
    assert args.simplex_geometry_distance_weight_final == 0.025
    assert args.simplex_geometry_distance_weight_ramp_start_step == 3000
    assert args.simplex_geometry_distance_weight_ramp_steps == 500
    assert args.simplex_boundary_message_degree_attenuation == 0.5
    assert args.simplex_boundary_incidence_normalization == 1.0
    assert args.simplex_boundary_readout_directionality == 0.5
    assert args.simplex_boundary_hodge_readout_scale == 0.25
    assert args.simplex_boundary_edge_star_readout_scale == 0.5
    assert args.simplex_boundary_edge_star_residual_scale == 0.25
    assert args.simplex_boundary_oriented_cochain_scale == 0.25
    assert args.simplex_boundary_face_cyclic_readout_scale == 0.5
    assert args.simplex_boundary_signed_face_cyclic_readout_scale == 0.25
    assert args.simplex_signed_tetra_coboundary_scale == 0.125
    assert args.simplex_signed_tetra_coboundary_runtime_scale == 0.0
    assert args.simplex_signed_tetra_coboundary_runtime_scale_final == 0.125
    assert args.simplex_signed_tetra_coboundary_runtime_scale_ramp_start_step == 3000
    assert args.simplex_signed_tetra_coboundary_runtime_scale_ramp_steps == 500
    assert args.simplex_signed_tetra_to_face_scale == 0.25
    assert args.simplex_signed_tetra_to_face_runtime_scale == 0.0
    assert args.simplex_signed_tetra_to_face_runtime_scale_final == 0.25
    assert args.simplex_signed_tetra_to_face_runtime_scale_ramp_start_step == 3000
    assert args.simplex_signed_tetra_to_face_runtime_scale_ramp_steps == 500
    assert args.simplex_boundary_edge_star_residual_runtime_scale == 0.0
    assert args.simplex_boundary_edge_star_residual_runtime_scale_final == 0.25
    assert args.simplex_boundary_edge_star_residual_runtime_scale_ramp_start_step == 3000
    assert args.simplex_boundary_edge_star_residual_runtime_scale_ramp_steps == 500
    assert args.simplex_boundary_oriented_cochain_runtime_scale == 0.0
    assert args.simplex_boundary_oriented_cochain_runtime_scale_final == 0.25
    assert args.simplex_boundary_oriented_cochain_runtime_scale_ramp_start_step == 3000
    assert args.simplex_boundary_oriented_cochain_runtime_scale_ramp_steps == 500
    assert args.simplex_boundary_face_cyclic_readout_runtime_scale == 0.0
    assert args.simplex_boundary_face_cyclic_readout_runtime_scale_final == 0.5
    assert args.simplex_boundary_face_cyclic_readout_runtime_scale_ramp_start_step == 3000
    assert args.simplex_boundary_face_cyclic_readout_runtime_scale_ramp_steps == 500
    assert args.simplex_boundary_signed_face_cyclic_readout_runtime_scale == 0.0
    assert args.simplex_boundary_signed_face_cyclic_readout_runtime_scale_final == 0.25
    assert args.simplex_boundary_signed_face_cyclic_readout_runtime_scale_ramp_start_step == 3000
    assert args.simplex_boundary_signed_face_cyclic_readout_runtime_scale_ramp_steps == 500
    assert args.simplex_boundary_edge_frame_gate_scale == 0.05
    assert args.simplex_boundary_edge_frame_gate_runtime_scale == 0.0
    assert args.simplex_boundary_edge_frame_gate_runtime_scale_final == 0.05
    assert args.simplex_boundary_edge_frame_gate_runtime_scale_ramp_start_step == 3000
    assert args.simplex_boundary_edge_frame_gate_runtime_scale_ramp_steps == 500
    assert args.simplex_global_context_scale == 0.125
    assert args.simplex_vertex_star_context_scale == 0.75
    assert args.simplex_edge_star_context_scale == 0.5
    assert args.simplex_pre_triangle_update_scale == 0.25
    assert args.simplex_pre_triangle_single_update_scale == 0.0
    assert args.simplex_pre_triangle_update_runtime_scale == 0.0
    assert args.simplex_pre_triangle_update_runtime_scale_final == 0.25
    assert args.simplex_pre_triangle_update_runtime_scale_ramp_start_step == 3000
    assert args.simplex_pre_triangle_update_runtime_scale_ramp_steps == 500
    assert args.simplex_pre_triangle_single_update_runtime_scale == 0.0
    assert args.simplex_pre_triangle_single_update_runtime_scale_final == 0.0
    assert args.simplex_pre_triangle_single_update_runtime_scale_ramp_start_step == 3000
    assert args.simplex_pre_triangle_single_update_runtime_scale_ramp_steps == 500
    assert args.simplex_triangle_attention_bias_scale == 0.05
    assert args.simplex_triangle_attention_value_scale == 0.025
    assert args.simplex_triangle_attention_bias_runtime_scale == 0.0
    assert args.simplex_triangle_attention_bias_runtime_scale_final == 0.0125
    assert args.simplex_triangle_attention_bias_runtime_scale_ramp_start_step == 3000
    assert args.simplex_triangle_attention_bias_runtime_scale_ramp_steps == 500
    assert args.simplex_triangle_attention_value_runtime_scale == 0.025
    assert args.simplex_triangle_attention_value_runtime_scale_final == 0.0
    assert args.simplex_triangle_attention_value_runtime_scale_ramp_start_step == 3000
    assert args.simplex_triangle_attention_value_runtime_scale_ramp_steps == 500
    assert args.simplex_vertex_star_context_runtime_scale == 0.0
    assert args.simplex_vertex_star_context_runtime_scale_final == 1.0
    assert args.simplex_vertex_star_context_runtime_scale_ramp_start_step == 3000
    assert args.simplex_vertex_star_context_runtime_scale_ramp_steps == 500
    assert args.simplex_edge_star_context_runtime_scale == 1.0
    assert args.simplex_edge_star_context_runtime_scale_final == 0.0
    assert args.simplex_edge_star_context_runtime_scale_ramp_start_step == 3000
    assert args.simplex_edge_star_context_runtime_scale_ramp_steps == 500
    assert args.simplex_face_boundary_lddt_weight_final == 0.025
    assert args.simplex_tetra_boundary_lddt_weight_final == 0.025
    assert args.simplex_boundary_lddt_ramp_start_step == 3500
    assert args.simplex_boundary_lddt_ramp_steps == 500
    assert args.simplex_segment_radius == 5
    assert args.simplex_face_top_k == 24
    assert args.simplex_face_top_k_final == 12
    assert args.simplex_face_top_k_ramp_start_step == 3500
    assert args.simplex_face_top_k_ramp_steps == 500
    assert args.simplex_tetra_top_k == 48
    assert args.simplex_tetra_top_k_final == 24
    assert args.simplex_tetra_top_k_ramp_start_step == 3500
    assert args.simplex_tetra_top_k_ramp_steps == 500
    assert args.simplex_cell_score_degree_penalty == 0.75
    assert args.simplex_cell_score_outer_edge_weight == 0.25
    assert args.simplex_cell_score_segment_weight == 0.125
    assert args.simplex_cell_score_outer_edge_weight_final == 0.5
    assert args.simplex_cell_score_outer_edge_weight_ramp_start_step == 3500
    assert args.simplex_cell_score_outer_edge_weight_ramp_steps == 500
    assert args.simplex_outer_edge_residual_context_runtime_scale == 0.0
    assert args.simplex_outer_edge_residual_context_runtime_scale_final == 0.25
    assert args.simplex_outer_edge_residual_context_runtime_scale_ramp_start_step == 3000
    assert args.simplex_outer_edge_residual_context_runtime_scale_ramp_steps == 500
    assert args.resume_model_weights_only is True

    cfg = _apply_model_config_overrides(load_model_config("simplexfold_medium_param_matched"), args)
    assert cfg.simplex_geometry_distance_weight == 0.1
    assert cfg.simplex_boundary_message_degree_attenuation == 0.5
    assert cfg.simplex_boundary_incidence_normalization == 1.0
    assert cfg.simplex_boundary_readout_directionality == 0.5
    assert cfg.simplex_boundary_hodge_readout_scale == 0.25
    assert cfg.simplex_boundary_edge_star_readout_scale == 0.5
    assert cfg.simplex_boundary_edge_star_residual_scale == 0.25
    assert cfg.simplex_boundary_oriented_cochain_scale == 0.25
    assert cfg.simplex_boundary_face_cyclic_readout_scale == 0.5
    assert cfg.simplex_boundary_signed_face_cyclic_readout_scale == 0.25
    assert cfg.simplex_signed_tetra_coboundary_scale == 0.125
    assert cfg.simplex_signed_tetra_to_face_scale == 0.25
    assert cfg.simplex_global_context_scale == 0.125
    assert cfg.simplex_vertex_star_context_scale == 0.75
    assert cfg.simplex_edge_star_context_scale == 0.5
    assert cfg.simplex_pre_triangle_update_scale == 0.25
    assert cfg.simplex_pre_triangle_single_update_scale == 0.0
    assert cfg.simplex_triangle_attention_bias_scale == 0.05
    assert cfg.simplex_triangle_attention_value_scale == 0.025
    assert cfg.simplex_structure_pair_readout_scale == 0.125
    assert cfg.simplex_msa_feedback_scale == 0.25
    assert cfg.simplex_boundary_msa_feedback_scale == 0.125
    assert cfg.simplex_boundary_pair_feedback_scale == 0.0625
    assert cfg.simplex_boundary_pair_gate_scale == 0.03125
    assert cfg.simplex_boundary_metric_gate_scale == 0.5
    assert cfg.simplex_boundary_metric_recycling_scale == 0.125
    assert cfg.simplex_boundary_cochain_recycling_scale == 0.0625
    assert cfg.simplex_boundary_cochain_recycling_metric_gate_scale == 0.5
    assert cfg.simplex_face_top_k == 24
    assert cfg.simplex_tetra_top_k == 48
    assert cfg.simplex_cell_score_degree_penalty == 0.75
    assert cfg.simplex_cell_score_outer_edge_weight == 0.25
    assert cfg.simplex_cell_score_segment_weight == 0.125


def test_runtime_simplex_message_scales_ramp_and_enter_model_inputs():
    cfg = TrainingConfig(
        simplex_pair_update_runtime_scale=0.5,
        simplex_pair_update_runtime_scale_final=1.0,
        simplex_pair_update_runtime_scale_ramp_start_step=3000,
        simplex_pair_update_runtime_scale_ramp_steps=500,
        simplex_single_update_runtime_scale=1.0,
        simplex_single_update_runtime_scale_final=0.25,
        simplex_single_update_runtime_scale_ramp_start_step=3000,
        simplex_single_update_runtime_scale_ramp_steps=500,
        simplex_outer_edge_context_runtime_scale=0.0,
        simplex_outer_edge_context_runtime_scale_final=0.05,
        simplex_outer_edge_context_runtime_scale_ramp_start_step=3000,
        simplex_outer_edge_context_runtime_scale_ramp_steps=500,
        simplex_outer_edge_residual_context_runtime_scale=0.0,
        simplex_outer_edge_residual_context_runtime_scale_final=0.25,
        simplex_outer_edge_residual_context_runtime_scale_ramp_start_step=3000,
        simplex_outer_edge_residual_context_runtime_scale_ramp_steps=500,
        simplex_edge_frame_message_runtime_scale=0.0,
        simplex_edge_frame_message_runtime_scale_final=0.05,
        simplex_edge_frame_message_runtime_scale_ramp_start_step=3000,
        simplex_edge_frame_message_runtime_scale_ramp_steps=500,
        simplex_boundary_edge_frame_gate_runtime_scale=0.0,
        simplex_boundary_edge_frame_gate_runtime_scale_final=0.05,
        simplex_boundary_edge_frame_gate_runtime_scale_ramp_start_step=3000,
        simplex_boundary_edge_frame_gate_runtime_scale_ramp_steps=500,
        simplex_boundary_readout_directionality_runtime_scale=0.0,
        simplex_boundary_readout_directionality_runtime_scale_final=0.5,
        simplex_boundary_readout_directionality_runtime_scale_ramp_start_step=3000,
        simplex_boundary_readout_directionality_runtime_scale_ramp_steps=500,
        simplex_boundary_hodge_readout_runtime_scale=0.0,
        simplex_boundary_hodge_readout_runtime_scale_final=0.25,
        simplex_boundary_hodge_readout_runtime_scale_ramp_start_step=3000,
        simplex_boundary_hodge_readout_runtime_scale_ramp_steps=500,
        simplex_boundary_edge_star_readout_runtime_scale=0.0,
        simplex_boundary_edge_star_readout_runtime_scale_final=0.5,
        simplex_boundary_edge_star_readout_runtime_scale_ramp_start_step=3000,
        simplex_boundary_edge_star_readout_runtime_scale_ramp_steps=500,
        simplex_boundary_edge_star_residual_runtime_scale=0.0,
        simplex_boundary_edge_star_residual_runtime_scale_final=0.25,
        simplex_boundary_edge_star_residual_runtime_scale_ramp_start_step=3000,
        simplex_boundary_edge_star_residual_runtime_scale_ramp_steps=500,
        simplex_boundary_oriented_cochain_runtime_scale=0.0,
        simplex_boundary_oriented_cochain_runtime_scale_final=0.25,
        simplex_boundary_oriented_cochain_runtime_scale_ramp_start_step=3000,
        simplex_boundary_oriented_cochain_runtime_scale_ramp_steps=500,
        simplex_boundary_face_cyclic_readout_runtime_scale=0.0,
        simplex_boundary_face_cyclic_readout_runtime_scale_final=0.5,
        simplex_boundary_face_cyclic_readout_runtime_scale_ramp_start_step=3000,
        simplex_boundary_face_cyclic_readout_runtime_scale_ramp_steps=500,
        simplex_boundary_signed_face_cyclic_readout_runtime_scale=0.0,
        simplex_boundary_signed_face_cyclic_readout_runtime_scale_final=0.25,
        simplex_boundary_signed_face_cyclic_readout_runtime_scale_ramp_start_step=3000,
        simplex_boundary_signed_face_cyclic_readout_runtime_scale_ramp_steps=500,
        simplex_vertex_star_context_runtime_scale=0.0,
        simplex_vertex_star_context_runtime_scale_final=1.0,
        simplex_vertex_star_context_runtime_scale_ramp_start_step=3000,
        simplex_vertex_star_context_runtime_scale_ramp_steps=500,
        simplex_edge_star_context_runtime_scale=1.0,
        simplex_edge_star_context_runtime_scale_final=0.0,
        simplex_edge_star_context_runtime_scale_ramp_start_step=3000,
        simplex_edge_star_context_runtime_scale_ramp_steps=500,
        simplex_pre_triangle_update_runtime_scale=0.0,
        simplex_pre_triangle_update_runtime_scale_final=0.25,
        simplex_pre_triangle_update_runtime_scale_ramp_start_step=3000,
        simplex_pre_triangle_update_runtime_scale_ramp_steps=500,
        simplex_pre_triangle_single_update_runtime_scale=0.0,
        simplex_pre_triangle_single_update_runtime_scale_final=0.0,
        simplex_pre_triangle_single_update_runtime_scale_ramp_start_step=3000,
        simplex_pre_triangle_single_update_runtime_scale_ramp_steps=500,
        simplex_triangle_attention_bias_runtime_scale=0.0,
        simplex_triangle_attention_bias_runtime_scale_final=0.0125,
        simplex_triangle_attention_bias_runtime_scale_ramp_start_step=3000,
        simplex_triangle_attention_bias_runtime_scale_ramp_steps=500,
        simplex_triangle_attention_value_runtime_scale=0.025,
        simplex_triangle_attention_value_runtime_scale_final=0.0,
        simplex_triangle_attention_value_runtime_scale_ramp_start_step=3000,
        simplex_triangle_attention_value_runtime_scale_ramp_steps=500,
        simplex_hodge_face_runtime_scale=0.0,
        simplex_hodge_face_runtime_scale_final=0.05,
        simplex_hodge_face_runtime_scale_ramp_start_step=3000,
        simplex_hodge_face_runtime_scale_ramp_steps=500,
        simplex_signed_tetra_coboundary_runtime_scale=0.0,
        simplex_signed_tetra_coboundary_runtime_scale_final=0.125,
        simplex_signed_tetra_coboundary_runtime_scale_ramp_start_step=3000,
        simplex_signed_tetra_coboundary_runtime_scale_ramp_steps=500,
        simplex_signed_tetra_to_face_runtime_scale=0.0,
        simplex_signed_tetra_to_face_runtime_scale_final=0.25,
        simplex_signed_tetra_to_face_runtime_scale_ramp_start_step=3000,
        simplex_signed_tetra_to_face_runtime_scale_ramp_steps=500,
        simplex_segment_cell_runtime_scale=0.0,
        simplex_segment_cell_runtime_scale_final=0.05,
        simplex_segment_cell_runtime_scale_ramp_start_step=3000,
        simplex_segment_cell_runtime_scale_ramp_steps=500,
        simplex_msa_feedback_runtime_scale=0.0,
        simplex_msa_feedback_runtime_scale_final=0.05,
        simplex_msa_feedback_runtime_scale_ramp_start_step=3000,
        simplex_msa_feedback_runtime_scale_ramp_steps=500,
        simplex_boundary_pair_feedback_runtime_scale=0.0,
        simplex_boundary_pair_feedback_runtime_scale_final=0.025,
        simplex_boundary_pair_feedback_runtime_scale_ramp_start_step=3000,
        simplex_boundary_pair_feedback_runtime_scale_ramp_steps=500,
        simplex_boundary_pair_gate_runtime_scale=0.0,
        simplex_boundary_pair_gate_runtime_scale_final=0.025,
        simplex_boundary_pair_gate_runtime_scale_ramp_start_step=3000,
        simplex_boundary_pair_gate_runtime_scale_ramp_steps=500,
        simplex_boundary_metric_gate_runtime_scale=0.0,
        simplex_boundary_metric_gate_runtime_scale_final=0.25,
        simplex_boundary_metric_gate_runtime_scale_ramp_start_step=3000,
        simplex_boundary_metric_gate_runtime_scale_ramp_steps=500,
        simplex_boundary_metric_recycling_runtime_scale=0.0,
        simplex_boundary_metric_recycling_runtime_scale_final=0.125,
        simplex_boundary_metric_recycling_runtime_scale_ramp_start_step=3000,
        simplex_boundary_metric_recycling_runtime_scale_ramp_steps=500,
        simplex_boundary_cochain_recycling_runtime_scale=0.0,
        simplex_boundary_cochain_recycling_runtime_scale_final=0.0625,
        simplex_boundary_cochain_recycling_runtime_scale_ramp_start_step=3000,
        simplex_boundary_cochain_recycling_runtime_scale_ramp_steps=500,
        simplex_geometry_distance_weight=0.1,
        simplex_geometry_distance_weight_final=0.025,
        simplex_geometry_distance_weight_ramp_start_step=3000,
        simplex_geometry_distance_weight_ramp_steps=500,
        simplex_face_top_k=0,
        simplex_face_top_k_final=24,
        simplex_face_top_k_ramp_start_step=3000,
        simplex_face_top_k_ramp_steps=500,
        simplex_tetra_top_k=0,
        simplex_tetra_top_k_final=48,
        simplex_tetra_top_k_ramp_start_step=3000,
        simplex_tetra_top_k_ramp_steps=500,
        simplex_cell_score_outer_edge_weight=0.0,
        simplex_cell_score_outer_edge_weight_final=0.25,
        simplex_cell_score_outer_edge_weight_ramp_start_step=3000,
        simplex_cell_score_outer_edge_weight_ramp_steps=500,
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

    assert simplex_pair_update_runtime_scale_at_step(cfg, 3000) == 0.5
    assert simplex_pair_update_runtime_scale_at_step(cfg, 3250) == 0.75
    assert simplex_pair_update_runtime_scale_at_step(cfg, 3500) == 1.0
    assert simplex_single_update_runtime_scale_at_step(cfg, 3000) == 1.0
    assert simplex_single_update_runtime_scale_at_step(cfg, 3250) == 0.625
    assert simplex_single_update_runtime_scale_at_step(cfg, 3500) == 0.25
    assert simplex_outer_edge_context_runtime_scale_at_step(cfg, 3000) == 0.0
    assert simplex_outer_edge_context_runtime_scale_at_step(cfg, 3250) == 0.025
    assert simplex_outer_edge_context_runtime_scale_at_step(cfg, 3500) == 0.05
    assert simplex_outer_edge_residual_context_runtime_scale_at_step(cfg, 3000) == 0.0
    assert simplex_outer_edge_residual_context_runtime_scale_at_step(cfg, 3250) == 0.125
    assert simplex_outer_edge_residual_context_runtime_scale_at_step(cfg, 3500) == 0.25
    assert simplex_edge_frame_message_runtime_scale_at_step(cfg, 3000) == 0.0
    assert simplex_edge_frame_message_runtime_scale_at_step(cfg, 3250) == 0.025
    assert simplex_edge_frame_message_runtime_scale_at_step(cfg, 3500) == 0.05
    assert simplex_boundary_edge_frame_gate_runtime_scale_at_step(cfg, 3000) == 0.0
    assert simplex_boundary_edge_frame_gate_runtime_scale_at_step(cfg, 3250) == 0.025
    assert simplex_boundary_edge_frame_gate_runtime_scale_at_step(cfg, 3500) == 0.05
    assert simplex_boundary_readout_directionality_runtime_scale_at_step(cfg, 3000) == 0.0
    assert simplex_boundary_readout_directionality_runtime_scale_at_step(cfg, 3250) == 0.25
    assert simplex_boundary_readout_directionality_runtime_scale_at_step(cfg, 3500) == 0.5
    assert simplex_boundary_hodge_readout_runtime_scale_at_step(cfg, 3000) == 0.0
    assert simplex_boundary_hodge_readout_runtime_scale_at_step(cfg, 3250) == 0.125
    assert simplex_boundary_hodge_readout_runtime_scale_at_step(cfg, 3500) == 0.25
    assert simplex_boundary_edge_star_readout_runtime_scale_at_step(cfg, 3000) == 0.0
    assert simplex_boundary_edge_star_readout_runtime_scale_at_step(cfg, 3250) == 0.25
    assert simplex_boundary_edge_star_readout_runtime_scale_at_step(cfg, 3500) == 0.5
    assert simplex_boundary_edge_star_residual_runtime_scale_at_step(cfg, 3000) == 0.0
    assert simplex_boundary_edge_star_residual_runtime_scale_at_step(cfg, 3250) == 0.125
    assert simplex_boundary_edge_star_residual_runtime_scale_at_step(cfg, 3500) == 0.25
    assert simplex_boundary_oriented_cochain_runtime_scale_at_step(cfg, 3000) == 0.0
    assert simplex_boundary_oriented_cochain_runtime_scale_at_step(cfg, 3250) == 0.125
    assert simplex_boundary_oriented_cochain_runtime_scale_at_step(cfg, 3500) == 0.25
    assert simplex_boundary_face_cyclic_readout_runtime_scale_at_step(cfg, 3000) == 0.0
    assert simplex_boundary_face_cyclic_readout_runtime_scale_at_step(cfg, 3250) == 0.25
    assert simplex_boundary_face_cyclic_readout_runtime_scale_at_step(cfg, 3500) == 0.5
    assert simplex_boundary_signed_face_cyclic_readout_runtime_scale_at_step(cfg, 3000) == 0.0
    assert simplex_boundary_signed_face_cyclic_readout_runtime_scale_at_step(cfg, 3250) == 0.125
    assert simplex_boundary_signed_face_cyclic_readout_runtime_scale_at_step(cfg, 3500) == 0.25
    assert simplex_vertex_star_context_runtime_scale_at_step(cfg, 3000) == 0.0
    assert simplex_vertex_star_context_runtime_scale_at_step(cfg, 3250) == 0.5
    assert simplex_vertex_star_context_runtime_scale_at_step(cfg, 3500) == 1.0
    assert simplex_edge_star_context_runtime_scale_at_step(cfg, 3000) == 1.0
    assert simplex_edge_star_context_runtime_scale_at_step(cfg, 3250) == 0.5
    assert simplex_edge_star_context_runtime_scale_at_step(cfg, 3500) == 0.0
    assert simplex_pre_triangle_update_runtime_scale_at_step(cfg, 3000) == 0.0
    assert simplex_pre_triangle_update_runtime_scale_at_step(cfg, 3250) == 0.125
    assert simplex_pre_triangle_update_runtime_scale_at_step(cfg, 3500) == 0.25
    assert simplex_pre_triangle_single_update_runtime_scale_at_step(cfg, 3000) == 0.0
    assert simplex_pre_triangle_single_update_runtime_scale_at_step(cfg, 3250) == 0.0
    assert simplex_pre_triangle_single_update_runtime_scale_at_step(cfg, 3500) == 0.0
    assert simplex_triangle_attention_bias_runtime_scale_at_step(cfg, 3000) == 0.0
    assert simplex_triangle_attention_bias_runtime_scale_at_step(cfg, 3250) == 0.00625
    assert simplex_triangle_attention_bias_runtime_scale_at_step(cfg, 3500) == 0.0125
    assert simplex_triangle_attention_value_runtime_scale_at_step(cfg, 3000) == 0.025
    assert simplex_triangle_attention_value_runtime_scale_at_step(cfg, 3250) == 0.0125
    assert simplex_triangle_attention_value_runtime_scale_at_step(cfg, 3500) == 0.0
    assert simplex_hodge_face_runtime_scale_at_step(cfg, 3000) == 0.0
    assert simplex_hodge_face_runtime_scale_at_step(cfg, 3250) == 0.025
    assert simplex_hodge_face_runtime_scale_at_step(cfg, 3500) == 0.05
    assert simplex_signed_tetra_coboundary_runtime_scale_at_step(cfg, 3000) == 0.0
    assert simplex_signed_tetra_coboundary_runtime_scale_at_step(cfg, 3250) == 0.0625
    assert simplex_signed_tetra_coboundary_runtime_scale_at_step(cfg, 3500) == 0.125
    assert simplex_signed_tetra_to_face_runtime_scale_at_step(cfg, 3000) == 0.0
    assert simplex_signed_tetra_to_face_runtime_scale_at_step(cfg, 3250) == 0.125
    assert simplex_signed_tetra_to_face_runtime_scale_at_step(cfg, 3500) == 0.25
    assert simplex_segment_cell_runtime_scale_at_step(cfg, 3000) == 0.0
    assert simplex_segment_cell_runtime_scale_at_step(cfg, 3250) == 0.025
    assert simplex_segment_cell_runtime_scale_at_step(cfg, 3500) == 0.05
    assert simplex_msa_feedback_runtime_scale_at_step(cfg, 3000) == 0.0
    assert simplex_msa_feedback_runtime_scale_at_step(cfg, 3250) == 0.025
    assert simplex_msa_feedback_runtime_scale_at_step(cfg, 3500) == 0.05
    assert simplex_boundary_pair_feedback_runtime_scale_at_step(cfg, 3000) == 0.0
    assert simplex_boundary_pair_feedback_runtime_scale_at_step(cfg, 3250) == 0.0125
    assert simplex_boundary_pair_feedback_runtime_scale_at_step(cfg, 3500) == 0.025
    assert simplex_boundary_pair_gate_runtime_scale_at_step(cfg, 3000) == 0.0
    assert simplex_boundary_pair_gate_runtime_scale_at_step(cfg, 3250) == 0.0125
    assert simplex_boundary_pair_gate_runtime_scale_at_step(cfg, 3500) == 0.025
    assert simplex_boundary_metric_gate_runtime_scale_at_step(cfg, 3000) == 0.0
    assert simplex_boundary_metric_gate_runtime_scale_at_step(cfg, 3250) == 0.125
    assert simplex_boundary_metric_gate_runtime_scale_at_step(cfg, 3500) == 0.25
    assert simplex_boundary_metric_recycling_runtime_scale_at_step(cfg, 3000) == 0.0
    assert simplex_boundary_metric_recycling_runtime_scale_at_step(cfg, 3250) == 0.0625
    assert simplex_boundary_metric_recycling_runtime_scale_at_step(cfg, 3500) == 0.125
    assert simplex_boundary_cochain_recycling_runtime_scale_at_step(cfg, 3000) == 0.0
    assert simplex_boundary_cochain_recycling_runtime_scale_at_step(cfg, 3250) == 0.03125
    assert simplex_boundary_cochain_recycling_runtime_scale_at_step(cfg, 3500) == 0.0625
    assert simplex_geometry_distance_weight_at_step(cfg, 3000) == 0.1
    assert simplex_geometry_distance_weight_at_step(cfg, 3250) == 0.0625
    assert abs(simplex_geometry_distance_weight_at_step(cfg, 3500) - 0.025) < 1e-9
    assert simplex_face_top_k_at_step(cfg, 3000) == 0
    assert simplex_face_top_k_at_step(cfg, 3250) == 12
    assert simplex_face_top_k_at_step(cfg, 3500) == 24
    assert simplex_tetra_top_k_at_step(cfg, 3000) == 0
    assert simplex_tetra_top_k_at_step(cfg, 3250) == 24
    assert simplex_tetra_top_k_at_step(cfg, 3500) == 48
    assert simplex_cell_score_outer_edge_weight_at_step(cfg, 3000) == 0.0
    assert simplex_cell_score_outer_edge_weight_at_step(cfg, 3250) == 0.125
    assert simplex_cell_score_outer_edge_weight_at_step(cfg, 3500) == 0.25
    inputs = model_inputs_from_batch(
        batch,
        cfg,
        use_simplex_update_scale=True,
        use_simplex_outer_edge_context_runtime_scale=True,
        use_simplex_outer_edge_residual_context_runtime_scale=True,
        use_simplex_edge_frame_message_runtime_scale=True,
        use_simplex_boundary_edge_frame_gate_runtime_scale=True,
        use_simplex_boundary_readout_directionality_runtime_scale=True,
        use_simplex_boundary_hodge_readout_runtime_scale=True,
        use_simplex_boundary_edge_star_readout_runtime_scale=True,
        use_simplex_boundary_edge_star_residual_runtime_scale=True,
        use_simplex_boundary_oriented_cochain_runtime_scale=True,
        use_simplex_boundary_face_cyclic_readout_runtime_scale=True,
        use_simplex_boundary_signed_face_cyclic_readout_runtime_scale=True,
        use_simplex_vertex_star_context_runtime_scale=True,
        use_simplex_edge_star_context_runtime_scale=True,
        use_simplex_pre_triangle_runtime_scale=True,
        use_simplex_triangle_attention_runtime_scale=True,
        use_simplex_hodge_face_runtime_scale=True,
        use_simplex_signed_tetra_coboundary_runtime_scale=True,
        use_simplex_signed_tetra_to_face_runtime_scale=True,
        use_simplex_segment_cell_runtime_scale=True,
        use_simplex_msa_feedback_runtime_scale=True,
        use_simplex_boundary_pair_feedback_runtime_scale=True,
        use_simplex_boundary_pair_gate_runtime_scale=True,
        use_simplex_boundary_metric_gate_runtime_scale=True,
        use_simplex_boundary_metric_recycling_runtime_scale=True,
        use_simplex_boundary_cochain_recycling_runtime_scale=True,
        use_simplex_geometry_distance_weight=True,
        use_simplex_cell_top_k=True,
        step=3250,
    )

    assert torch.isclose(inputs["simplex_pair_update_scale_override"], torch.tensor(0.75))
    assert torch.isclose(inputs["simplex_single_update_scale_override"], torch.tensor(0.625))
    assert torch.isclose(inputs["simplex_outer_edge_context_scale_override"], torch.tensor(0.025))
    assert torch.isclose(inputs["simplex_outer_edge_residual_context_scale_override"], torch.tensor(0.125))
    assert torch.isclose(inputs["simplex_edge_frame_message_scale_override"], torch.tensor(0.025))
    assert torch.isclose(inputs["simplex_boundary_edge_frame_gate_scale_override"], torch.tensor(0.025))
    assert torch.isclose(inputs["simplex_boundary_readout_directionality_override"], torch.tensor(0.25))
    assert torch.isclose(inputs["simplex_boundary_hodge_readout_scale_override"], torch.tensor(0.125))
    assert torch.isclose(inputs["simplex_boundary_edge_star_readout_scale_override"], torch.tensor(0.25))
    assert torch.isclose(inputs["simplex_boundary_edge_star_residual_scale_override"], torch.tensor(0.125))
    assert torch.isclose(inputs["simplex_boundary_oriented_cochain_scale_override"], torch.tensor(0.125))
    assert torch.isclose(inputs["simplex_boundary_face_cyclic_readout_scale_override"], torch.tensor(0.25))
    assert torch.isclose(
        inputs["simplex_boundary_signed_face_cyclic_readout_scale_override"],
        torch.tensor(0.125),
    )
    assert torch.isclose(inputs["simplex_vertex_star_context_scale_override"], torch.tensor(0.5))
    assert torch.isclose(inputs["simplex_edge_star_context_scale_override"], torch.tensor(0.5))
    assert torch.isclose(inputs["simplex_pre_triangle_update_scale_override"], torch.tensor(0.125))
    assert torch.isclose(inputs["simplex_pre_triangle_single_update_scale_override"], torch.tensor(0.0))
    assert torch.isclose(inputs["simplex_triangle_attention_bias_scale_override"], torch.tensor(0.00625))
    assert torch.isclose(inputs["simplex_triangle_attention_value_scale_override"], torch.tensor(0.0125))
    assert torch.isclose(inputs["simplex_hodge_face_update_scale_override"], torch.tensor(0.025))
    assert torch.isclose(inputs["simplex_signed_tetra_coboundary_scale_override"], torch.tensor(0.0625))
    assert torch.isclose(inputs["simplex_signed_tetra_to_face_scale_override"], torch.tensor(0.125))
    assert torch.isclose(inputs["simplex_segment_cell_scale_override"], torch.tensor(0.025))
    assert torch.isclose(inputs["simplex_msa_feedback_scale_override"], torch.tensor(0.025))
    assert torch.isclose(inputs["simplex_boundary_pair_feedback_scale_override"], torch.tensor(0.0125))
    assert torch.isclose(inputs["simplex_boundary_pair_gate_scale_override"], torch.tensor(0.0125))
    assert torch.isclose(inputs["simplex_boundary_metric_gate_scale_override"], torch.tensor(0.125))
    assert torch.isclose(inputs["simplex_boundary_metric_recycling_scale_override"], torch.tensor(0.0625))
    assert torch.isclose(inputs["simplex_boundary_cochain_recycling_scale_override"], torch.tensor(0.03125))
    assert torch.isclose(inputs["simplex_geometry_distance_weight_override"], torch.tensor(0.0625))
    assert torch.isclose(inputs["simplex_face_top_k_override"], torch.tensor(12.0))
    assert torch.isclose(inputs["simplex_tetra_top_k_override"], torch.tensor(24.0))
    assert torch.isclose(inputs["simplex_cell_score_outer_edge_weight_override"], torch.tensor(0.125))


def test_evaluate_uses_runtime_simplex_overrides_for_validation(monkeypatch):
    class CaptureModel:
        def __init__(self):
            self.kwargs = None

        def eval(self):
            return self

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return {"atom14_coords": torch.zeros(1, 4, 14, 3)}

    cfg = TrainingConfig(
        simplex_pair_update_runtime_scale=0.5,
        simplex_pair_update_runtime_scale_final=1.0,
        simplex_pair_update_runtime_scale_ramp_start_step=3000,
        simplex_pair_update_runtime_scale_ramp_steps=500,
        simplex_single_update_runtime_scale=1.0,
        simplex_single_update_runtime_scale_final=0.25,
        simplex_single_update_runtime_scale_ramp_start_step=3000,
        simplex_single_update_runtime_scale_ramp_steps=500,
        simplex_outer_edge_residual_context_runtime_scale=0.0,
        simplex_outer_edge_residual_context_runtime_scale_final=0.25,
        simplex_outer_edge_residual_context_runtime_scale_ramp_start_step=3000,
        simplex_outer_edge_residual_context_runtime_scale_ramp_steps=500,
        simplex_edge_frame_message_runtime_scale=0.0,
        simplex_edge_frame_message_runtime_scale_final=0.05,
        simplex_edge_frame_message_runtime_scale_ramp_start_step=3000,
        simplex_edge_frame_message_runtime_scale_ramp_steps=500,
        simplex_boundary_edge_frame_gate_runtime_scale=0.0,
        simplex_boundary_edge_frame_gate_runtime_scale_final=0.05,
        simplex_boundary_edge_frame_gate_runtime_scale_ramp_start_step=3000,
        simplex_boundary_edge_frame_gate_runtime_scale_ramp_steps=500,
        simplex_boundary_readout_directionality_runtime_scale=0.0,
        simplex_boundary_readout_directionality_runtime_scale_final=0.5,
        simplex_boundary_readout_directionality_runtime_scale_ramp_start_step=3000,
        simplex_boundary_readout_directionality_runtime_scale_ramp_steps=500,
        simplex_boundary_hodge_readout_runtime_scale=0.0,
        simplex_boundary_hodge_readout_runtime_scale_final=0.25,
        simplex_boundary_hodge_readout_runtime_scale_ramp_start_step=3000,
        simplex_boundary_hodge_readout_runtime_scale_ramp_steps=500,
        simplex_boundary_edge_star_readout_runtime_scale=0.0,
        simplex_boundary_edge_star_readout_runtime_scale_final=0.5,
        simplex_boundary_edge_star_readout_runtime_scale_ramp_start_step=3000,
        simplex_boundary_edge_star_readout_runtime_scale_ramp_steps=500,
        simplex_boundary_edge_star_residual_runtime_scale=0.0,
        simplex_boundary_edge_star_residual_runtime_scale_final=0.25,
        simplex_boundary_edge_star_residual_runtime_scale_ramp_start_step=3000,
        simplex_boundary_edge_star_residual_runtime_scale_ramp_steps=500,
        simplex_boundary_oriented_cochain_runtime_scale=0.0,
        simplex_boundary_oriented_cochain_runtime_scale_final=0.25,
        simplex_boundary_oriented_cochain_runtime_scale_ramp_start_step=3000,
        simplex_boundary_oriented_cochain_runtime_scale_ramp_steps=500,
        simplex_boundary_face_cyclic_readout_runtime_scale=0.0,
        simplex_boundary_face_cyclic_readout_runtime_scale_final=0.5,
        simplex_boundary_face_cyclic_readout_runtime_scale_ramp_start_step=3000,
        simplex_boundary_face_cyclic_readout_runtime_scale_ramp_steps=500,
        simplex_boundary_signed_face_cyclic_readout_runtime_scale=0.0,
        simplex_boundary_signed_face_cyclic_readout_runtime_scale_final=0.25,
        simplex_boundary_signed_face_cyclic_readout_runtime_scale_ramp_start_step=3000,
        simplex_boundary_signed_face_cyclic_readout_runtime_scale_ramp_steps=500,
        simplex_vertex_star_context_runtime_scale=0.0,
        simplex_vertex_star_context_runtime_scale_final=1.0,
        simplex_vertex_star_context_runtime_scale_ramp_start_step=3000,
        simplex_vertex_star_context_runtime_scale_ramp_steps=500,
        simplex_edge_star_context_runtime_scale=1.0,
        simplex_edge_star_context_runtime_scale_final=0.0,
        simplex_edge_star_context_runtime_scale_ramp_start_step=3000,
        simplex_edge_star_context_runtime_scale_ramp_steps=500,
        simplex_pre_triangle_update_runtime_scale=0.0,
        simplex_pre_triangle_update_runtime_scale_final=0.25,
        simplex_pre_triangle_update_runtime_scale_ramp_start_step=3000,
        simplex_pre_triangle_update_runtime_scale_ramp_steps=500,
        simplex_pre_triangle_single_update_runtime_scale=0.0,
        simplex_pre_triangle_single_update_runtime_scale_final=0.0,
        simplex_pre_triangle_single_update_runtime_scale_ramp_start_step=3000,
        simplex_pre_triangle_single_update_runtime_scale_ramp_steps=500,
        simplex_triangle_attention_bias_runtime_scale=0.0,
        simplex_triangle_attention_bias_runtime_scale_final=0.0125,
        simplex_triangle_attention_bias_runtime_scale_ramp_start_step=3000,
        simplex_triangle_attention_bias_runtime_scale_ramp_steps=500,
        simplex_triangle_attention_value_runtime_scale=0.025,
        simplex_triangle_attention_value_runtime_scale_final=0.0,
        simplex_triangle_attention_value_runtime_scale_ramp_start_step=3000,
        simplex_triangle_attention_value_runtime_scale_ramp_steps=500,
        simplex_signed_tetra_coboundary_runtime_scale=0.0,
        simplex_signed_tetra_coboundary_runtime_scale_final=0.125,
        simplex_signed_tetra_coboundary_runtime_scale_ramp_start_step=3000,
        simplex_signed_tetra_coboundary_runtime_scale_ramp_steps=500,
        simplex_signed_tetra_to_face_runtime_scale=0.0,
        simplex_signed_tetra_to_face_runtime_scale_final=0.25,
        simplex_signed_tetra_to_face_runtime_scale_ramp_start_step=3000,
        simplex_signed_tetra_to_face_runtime_scale_ramp_steps=500,
        simplex_segment_cell_runtime_scale=0.0,
        simplex_segment_cell_runtime_scale_final=0.05,
        simplex_segment_cell_runtime_scale_ramp_start_step=3000,
        simplex_segment_cell_runtime_scale_ramp_steps=500,
        simplex_msa_feedback_runtime_scale=0.0,
        simplex_msa_feedback_runtime_scale_final=0.05,
        simplex_msa_feedback_runtime_scale_ramp_start_step=3000,
        simplex_msa_feedback_runtime_scale_ramp_steps=500,
        simplex_boundary_pair_feedback_runtime_scale=0.0,
        simplex_boundary_pair_feedback_runtime_scale_final=0.025,
        simplex_boundary_pair_feedback_runtime_scale_ramp_start_step=3000,
        simplex_boundary_pair_feedback_runtime_scale_ramp_steps=500,
        simplex_boundary_pair_gate_runtime_scale=0.0,
        simplex_boundary_pair_gate_runtime_scale_final=0.025,
        simplex_boundary_pair_gate_runtime_scale_ramp_start_step=3000,
        simplex_boundary_pair_gate_runtime_scale_ramp_steps=500,
        simplex_boundary_metric_gate_runtime_scale=0.0,
        simplex_boundary_metric_gate_runtime_scale_final=0.25,
        simplex_boundary_metric_gate_runtime_scale_ramp_start_step=3000,
        simplex_boundary_metric_gate_runtime_scale_ramp_steps=500,
        simplex_boundary_metric_recycling_runtime_scale=0.0,
        simplex_boundary_metric_recycling_runtime_scale_final=0.125,
        simplex_boundary_metric_recycling_runtime_scale_ramp_start_step=3000,
        simplex_boundary_metric_recycling_runtime_scale_ramp_steps=500,
        simplex_boundary_cochain_recycling_runtime_scale=0.0,
        simplex_boundary_cochain_recycling_runtime_scale_final=0.0625,
        simplex_boundary_cochain_recycling_runtime_scale_ramp_start_step=3000,
        simplex_boundary_cochain_recycling_runtime_scale_ramp_steps=500,
        simplex_geometry_distance_weight=0.1,
        simplex_geometry_distance_weight_final=0.025,
        simplex_geometry_distance_weight_ramp_start_step=3000,
        simplex_geometry_distance_weight_ramp_steps=500,
        simplex_face_top_k=0,
        simplex_face_top_k_final=24,
        simplex_face_top_k_ramp_start_step=3000,
        simplex_face_top_k_ramp_steps=500,
        simplex_tetra_top_k=0,
        simplex_tetra_top_k_final=48,
        simplex_tetra_top_k_ramp_start_step=3000,
        simplex_tetra_top_k_ramp_steps=500,
        simplex_cell_score_outer_edge_weight=0.0,
        simplex_cell_score_outer_edge_weight_final=0.25,
        simplex_cell_score_outer_edge_weight_ramp_start_step=3000,
        simplex_cell_score_outer_edge_weight_ramp_steps=500,
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
    model = CaptureModel()
    progress_events = []

    monkeypatch.setattr(
        "scripts.run_nanofold_public_benchmarks._loss_with_terms",
        lambda loss_fn, batch, outputs: (torch.ones(1), {}),
    )
    monkeypatch.setattr(
        "scripts.run_nanofold_public_benchmarks._structure_metrics",
        lambda outputs, batch, foldscore_components_fn: {"lddt_ca": [0.1], "ca_rmsd": [1.0]},
    )
    monkeypatch.setattr("scripts.run_nanofold_public_benchmarks._simplex_topology_metrics", lambda outputs: {})
    monkeypatch.setattr(
        "scripts.run_nanofold_public_benchmarks._simplex_boundary_geometry_metrics",
        lambda outputs, batch: {},
    )

    result = _evaluate(
        model,
        object(),
        [batch],
        cfg,
        torch.device("cpu"),
        max_batches=None,
        foldscore_components_fn=None,
        mixed_precision="off",
        step=3250,
        progress_callback=lambda batch_index, total_batches, examples: progress_events.append(
            (batch_index, total_batches, examples)
        ),
    )

    assert result["val_lddt_ca"] == 0.1
    assert progress_events == [(0, 1, 0), (1, 1, 1)]
    assert torch.isclose(model.kwargs["simplex_pair_update_scale_override"], torch.tensor(0.75))
    assert torch.isclose(model.kwargs["simplex_single_update_scale_override"], torch.tensor(0.625))
    assert torch.isclose(model.kwargs["simplex_outer_edge_residual_context_scale_override"], torch.tensor(0.125))
    assert torch.isclose(model.kwargs["simplex_edge_frame_message_scale_override"], torch.tensor(0.025))
    assert torch.isclose(model.kwargs["simplex_boundary_edge_frame_gate_scale_override"], torch.tensor(0.025))
    assert torch.isclose(model.kwargs["simplex_boundary_readout_directionality_override"], torch.tensor(0.25))
    assert torch.isclose(model.kwargs["simplex_boundary_hodge_readout_scale_override"], torch.tensor(0.125))
    assert torch.isclose(model.kwargs["simplex_boundary_edge_star_readout_scale_override"], torch.tensor(0.25))
    assert torch.isclose(model.kwargs["simplex_boundary_edge_star_residual_scale_override"], torch.tensor(0.125))
    assert torch.isclose(model.kwargs["simplex_boundary_oriented_cochain_scale_override"], torch.tensor(0.125))
    assert torch.isclose(
        model.kwargs["simplex_boundary_face_cyclic_readout_scale_override"],
        torch.tensor(0.25),
    )
    assert torch.isclose(
        model.kwargs["simplex_boundary_signed_face_cyclic_readout_scale_override"],
        torch.tensor(0.125),
    )
    assert torch.isclose(
        model.kwargs["simplex_signed_tetra_coboundary_scale_override"],
        torch.tensor(0.0625),
    )
    assert torch.isclose(
        model.kwargs["simplex_signed_tetra_to_face_scale_override"],
        torch.tensor(0.125),
    )
    assert torch.isclose(model.kwargs["simplex_vertex_star_context_scale_override"], torch.tensor(0.5))
    assert torch.isclose(model.kwargs["simplex_edge_star_context_scale_override"], torch.tensor(0.5))
    assert torch.isclose(model.kwargs["simplex_pre_triangle_update_scale_override"], torch.tensor(0.125))
    assert torch.isclose(model.kwargs["simplex_pre_triangle_single_update_scale_override"], torch.tensor(0.0))
    assert torch.isclose(model.kwargs["simplex_triangle_attention_bias_scale_override"], torch.tensor(0.00625))
    assert torch.isclose(model.kwargs["simplex_triangle_attention_value_scale_override"], torch.tensor(0.0125))
    assert torch.isclose(model.kwargs["simplex_segment_cell_scale_override"], torch.tensor(0.025))
    assert torch.isclose(model.kwargs["simplex_msa_feedback_scale_override"], torch.tensor(0.025))
    assert torch.isclose(model.kwargs["simplex_boundary_pair_feedback_scale_override"], torch.tensor(0.0125))
    assert torch.isclose(model.kwargs["simplex_boundary_pair_gate_scale_override"], torch.tensor(0.0125))
    assert torch.isclose(model.kwargs["simplex_boundary_metric_gate_scale_override"], torch.tensor(0.125))
    assert torch.isclose(model.kwargs["simplex_boundary_metric_recycling_scale_override"], torch.tensor(0.0625))
    assert torch.isclose(model.kwargs["simplex_boundary_cochain_recycling_scale_override"], torch.tensor(0.03125))
    assert torch.isclose(model.kwargs["simplex_geometry_distance_weight_override"], torch.tensor(0.0625))
    assert torch.isclose(model.kwargs["simplex_face_top_k_override"], torch.tensor(12.0))
    assert torch.isclose(model.kwargs["simplex_tetra_top_k_override"], torch.tensor(24.0))
    assert torch.isclose(model.kwargs["simplex_cell_score_outer_edge_weight_override"], torch.tensor(0.125))


def test_simplex_topology_metrics_report_boundary_reuse():
    outputs = {
        "simplex_face_indices": torch.tensor([[[[0, 1, 2], [0, 1, 3]]]]),
        "simplex_face_mask": torch.ones(1, 1, 2),
        "simplex_tetra_indices": torch.tensor([[[[0, 1, 2, 3]]]]),
        "simplex_tetra_mask": torch.ones(1, 1, 1),
        "simplex_neighbor_indices": torch.tensor(
            [
                [
                    [1, 2, 3, 4],
                    [0, 2, 3, 4],
                    [0, 1, 3, 4],
                    [0, 1, 2, 4],
                    [0, 1, 2, 3],
                ]
            ]
        ),
    }

    metrics = _simplex_topology_metrics(outputs)

    assert metrics["simplex_face_active_cells"] == [2.0]
    assert metrics["simplex_face_active_fraction"] == [1.0]
    assert abs(metrics["simplex_face_boundary_edge_mean_degree"][0] - 1.2) < 1e-6
    assert metrics["simplex_face_boundary_edge_max_degree"] == [2.0]
    assert metrics["simplex_face_boundary_unique_edge_fraction"] == [5.0 / 6.0]
    assert metrics["simplex_face_outer_edge_mean_degree"] == [6.0]
    assert metrics["simplex_face_outer_edge_max_degree"] == [6.0]
    assert metrics["simplex_face_outer_edge_active_fraction"] == [1.0]
    assert metrics["simplex_tetra_active_cells"] == [1.0]
    assert metrics["simplex_tetra_boundary_edge_mean_degree"] == [1.0]
    assert metrics["simplex_tetra_outer_edge_mean_degree"] == [4.0]
    assert metrics["simplex_tetra_outer_edge_active_fraction"] == [1.0]


def test_simplex_boundary_geometry_metrics_report_selected_edge_errors():
    true_ca = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ]
    )
    pred_ca = true_ca.clone()
    pred_ca[0, 1] = torch.tensor([2.0, 0.0, 0.0])
    atom14 = torch.zeros(1, 4, 14, 3)
    atom14[:, :, 1, :] = pred_ca
    true_atom14 = torch.zeros(1, 4, 14, 3)
    true_atom14[:, :, 1, :] = true_ca
    batch = {
        "true_atom_positions": true_atom14,
        "true_atom_mask": torch.ones(1, 4, 14),
        "seq_mask": torch.ones(1, 4),
    }
    outputs = {
        "atom14_coords": atom14,
        "simplex_face_indices": torch.tensor([[[[0, 1, 2]]]]),
        "simplex_face_mask": torch.ones(1, 1, 1),
        "simplex_tetra_indices": torch.tensor([[[[0, 1, 2, 3]]]]),
        "simplex_tetra_mask": torch.ones(1, 1, 1),
    }

    metrics = _simplex_boundary_geometry_metrics(outputs, batch)

    assert metrics["simplex_face_boundary_length_mae"][0] > 0.0
    assert metrics["simplex_face_boundary_length_rmse"][0] >= metrics["simplex_face_boundary_length_mae"][0]
    assert metrics["simplex_face_boundary_contraction_fraction"] == [0.0]
    assert 0.0 <= metrics["simplex_tetra_boundary_lddt"][0] <= 1.0


def test_model_config_overrides_preserve_resume_compatible_variant_name():
    args = parse_args(
        [
            "--variants",
            "full_msa_to_face",
            "--simplex-outer-edge-context-scale",
            "0.25",
            "--simplex-outer-edge-residual-context-scale",
            "0.125",
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
    assert cfg.simplex_outer_edge_residual_context_scale == 0.125
    assert cfg.simplex_segment_radius == 5


def test_e140_selected_boundary_expansion_recipe_matches_running_gate():
    args = parse_args(
        [
            "--variants",
            "full_msa_to_face",
            "--run-name",
            "e140_selected_boundary_expansion_from_e128_s9000_c256_m64",
            "--steps",
            "9000",
            "--batch-size",
            "1",
            "--grad-accum-steps",
            "8",
            "--crop-size",
            "256",
            "--msa-depth",
            "64",
            "--extra-msa-depth",
            "0",
            "--max-templates",
            "0",
            "--max-parameters",
            "3261974",
            "--resume-model-weights-only",
            "--simplex-boundary-edge-frame-gate-scale",
            "0.05",
            "--simplex-triangle-attention-bias-scale",
            "0.0125",
            "--simplex-face-coordinate-expansion-weight",
            "0.05",
            "--simplex-tetra-coordinate-expansion-weight",
            "0.05",
            "--simplex-coordinate-expansion-tolerance",
            "0.05",
        ]
    )

    assert args.run_name == "e140_selected_boundary_expansion_from_e128_s9000_c256_m64"
    assert args.steps == 9000
    assert args.batch_size * args.grad_accum_steps == 8
    assert args.crop_size == 256
    assert args.msa_depth == 64
    assert args.extra_msa_depth == 0
    assert args.max_templates == 0
    assert args.max_parameters == 3_261_974
    assert args.num_workers == 0
    assert args.resume_model_weights_only is True
    assert args.simplex_boundary_edge_frame_gate_scale == 0.05
    assert args.simplex_triangle_attention_bias_scale == 0.0125
    assert args.simplex_face_coordinate_expansion_weight == 0.05
    assert args.simplex_tetra_coordinate_expansion_weight == 0.05
    assert args.simplex_coordinate_expansion_tolerance == 0.05
    assert args.simplex_boundary_signed_face_cyclic_readout_scale is None
    _enforce_parameter_budget(
        variant="full_msa_to_face",
        parameter_count=3_240_738,
        max_parameters=args.max_parameters,
    )


def test_e141_signed_face_cyclic_recipe_matches_running_gate():
    args = parse_args(
        [
            "--variants",
            "full_msa_to_face",
            "--run-name",
            "e141_signed_face_cyclic_boundary_from_e128_s9000_c256_m64",
            "--steps",
            "9000",
            "--batch-size",
            "1",
            "--grad-accum-steps",
            "8",
            "--crop-size",
            "256",
            "--msa-depth",
            "64",
            "--extra-msa-depth",
            "0",
            "--max-templates",
            "0",
            "--max-parameters",
            "3261974",
            "--resume-model-weights-only",
            "--simplex-boundary-edge-frame-gate-scale",
            "0.05",
            "--simplex-triangle-attention-bias-scale",
            "0.0125",
            "--simplex-boundary-readout-directionality",
            "0.25",
            "--simplex-boundary-signed-face-cyclic-readout-scale",
            "0.25",
            "--simplex-boundary-signed-face-cyclic-readout-runtime-scale",
            "0.0",
            "--simplex-boundary-signed-face-cyclic-readout-runtime-scale-final",
            "0.25",
            "--simplex-boundary-signed-face-cyclic-readout-runtime-scale-ramp-start-step",
            "8500",
            "--simplex-boundary-signed-face-cyclic-readout-runtime-scale-ramp-steps",
            "500",
        ]
    )

    assert args.run_name == "e141_signed_face_cyclic_boundary_from_e128_s9000_c256_m64"
    assert args.steps == 9000
    assert args.batch_size * args.grad_accum_steps == 8
    assert args.crop_size == 256
    assert args.msa_depth == 64
    assert args.extra_msa_depth == 0
    assert args.max_templates == 0
    assert args.max_parameters == 3_261_974
    assert args.num_workers == 0
    assert args.resume_model_weights_only is True
    assert args.simplex_boundary_edge_frame_gate_scale == 0.05
    assert args.simplex_triangle_attention_bias_scale == 0.0125
    assert args.simplex_boundary_readout_directionality == 0.25
    assert args.simplex_face_coordinate_expansion_weight is None
    assert args.simplex_tetra_coordinate_expansion_weight is None

    cfg = _apply_model_config_overrides(
        _variant_config(load_model_config("simplexfold_medium_param_matched"), "full_msa_to_face"),
        args,
    )
    assert cfg.simplex_boundary_signed_face_cyclic_readout_scale == 0.25
    _enforce_parameter_budget(
        variant="full_msa_to_face",
        parameter_count=3_240_738,
        max_parameters=args.max_parameters,
    )

    training_config = TrainingConfig(
        simplex_boundary_signed_face_cyclic_readout_runtime_scale=(
            args.simplex_boundary_signed_face_cyclic_readout_runtime_scale
        ),
        simplex_boundary_signed_face_cyclic_readout_runtime_scale_final=(
            args.simplex_boundary_signed_face_cyclic_readout_runtime_scale_final
        ),
        simplex_boundary_signed_face_cyclic_readout_runtime_scale_ramp_start_step=(
            args.simplex_boundary_signed_face_cyclic_readout_runtime_scale_ramp_start_step
        ),
        simplex_boundary_signed_face_cyclic_readout_runtime_scale_ramp_steps=(
            args.simplex_boundary_signed_face_cyclic_readout_runtime_scale_ramp_steps
        ),
    )
    assert simplex_boundary_signed_face_cyclic_readout_runtime_scale_at_step(training_config, 8500) == 0.0
    assert simplex_boundary_signed_face_cyclic_readout_runtime_scale_at_step(training_config, 8750) == 0.125
    assert simplex_boundary_signed_face_cyclic_readout_runtime_scale_at_step(training_config, 9000) == 0.25


def test_e142_signed_tetra_coboundary_recipe_matches_documented_gate():
    args = parse_args(
        [
            "--variants",
            "full_msa_to_face",
            "--run-name",
            "e142_signed_tetra_coboundary_from_e128_s9000_c256_m64",
            "--steps",
            "9000",
            "--batch-size",
            "1",
            "--grad-accum-steps",
            "8",
            "--crop-size",
            "256",
            "--msa-depth",
            "64",
            "--extra-msa-depth",
            "0",
            "--max-templates",
            "0",
            "--max-parameters",
            "3261974",
            "--resume-model-weights-only",
            "--simplex-boundary-edge-frame-gate-scale",
            "0.05",
            "--simplex-triangle-attention-bias-scale",
            "0.0125",
            "--simplex-signed-tetra-coboundary-scale",
            "0.25",
            "--simplex-signed-tetra-coboundary-runtime-scale",
            "0.0",
            "--simplex-signed-tetra-coboundary-runtime-scale-final",
            "0.25",
            "--simplex-signed-tetra-coboundary-runtime-scale-ramp-start-step",
            "8500",
            "--simplex-signed-tetra-coboundary-runtime-scale-ramp-steps",
            "500",
        ]
    )

    assert args.run_name == "e142_signed_tetra_coboundary_from_e128_s9000_c256_m64"
    assert args.steps == 9000
    assert args.batch_size * args.grad_accum_steps == 8
    assert args.crop_size == 256
    assert args.msa_depth == 64
    assert args.extra_msa_depth == 0
    assert args.max_templates == 0
    assert args.max_parameters == 3_261_974
    assert args.num_workers == 0
    assert args.resume_model_weights_only is True
    assert args.simplex_boundary_edge_frame_gate_scale == 0.05
    assert args.simplex_triangle_attention_bias_scale == 0.0125

    cfg = _apply_model_config_overrides(
        _variant_config(load_model_config("simplexfold_medium_param_matched"), "full_msa_to_face"),
        args,
    )
    assert cfg.simplex_signed_tetra_coboundary_scale == 0.25
    _enforce_parameter_budget(
        variant="full_msa_to_face",
        parameter_count=3_240_738,
        max_parameters=args.max_parameters,
    )

    training_config = TrainingConfig(
        simplex_signed_tetra_coboundary_runtime_scale=args.simplex_signed_tetra_coboundary_runtime_scale,
        simplex_signed_tetra_coboundary_runtime_scale_final=(
            args.simplex_signed_tetra_coboundary_runtime_scale_final
        ),
        simplex_signed_tetra_coboundary_runtime_scale_ramp_start_step=(
            args.simplex_signed_tetra_coboundary_runtime_scale_ramp_start_step
        ),
        simplex_signed_tetra_coboundary_runtime_scale_ramp_steps=(
            args.simplex_signed_tetra_coboundary_runtime_scale_ramp_steps
        ),
    )
    assert simplex_signed_tetra_coboundary_runtime_scale_at_step(training_config, 8500) == 0.0
    assert simplex_signed_tetra_coboundary_runtime_scale_at_step(training_config, 8750) == 0.125
    assert simplex_signed_tetra_coboundary_runtime_scale_at_step(training_config, 9000) == 0.25


def test_e143_signed_tetra_to_face_recipe_matches_documented_gate():
    args = parse_args(
        [
            "--variants",
            "full_msa_to_face",
            "--run-name",
            "e143_signed_tetra_to_face_from_e128_s9000_c256_m64",
            "--steps",
            "9000",
            "--batch-size",
            "1",
            "--grad-accum-steps",
            "8",
            "--crop-size",
            "256",
            "--msa-depth",
            "64",
            "--extra-msa-depth",
            "0",
            "--max-templates",
            "0",
            "--max-parameters",
            "3261974",
            "--resume-model-weights-only",
            "--simplex-boundary-edge-frame-gate-scale",
            "0.05",
            "--simplex-triangle-attention-bias-scale",
            "0.0125",
            "--simplex-signed-tetra-to-face-scale",
            "0.25",
            "--simplex-signed-tetra-to-face-runtime-scale",
            "0.0",
            "--simplex-signed-tetra-to-face-runtime-scale-final",
            "0.25",
            "--simplex-signed-tetra-to-face-runtime-scale-ramp-start-step",
            "8500",
            "--simplex-signed-tetra-to-face-runtime-scale-ramp-steps",
            "500",
        ]
    )

    assert args.run_name == "e143_signed_tetra_to_face_from_e128_s9000_c256_m64"
    assert args.steps == 9000
    assert args.batch_size * args.grad_accum_steps == 8
    assert args.crop_size == 256
    assert args.msa_depth == 64
    assert args.extra_msa_depth == 0
    assert args.max_templates == 0
    assert args.max_parameters == 3_261_974
    assert args.num_workers == 0
    assert args.resume_model_weights_only is True
    assert args.simplex_boundary_edge_frame_gate_scale == 0.05
    assert args.simplex_triangle_attention_bias_scale == 0.0125

    cfg = _apply_model_config_overrides(
        _variant_config(load_model_config("simplexfold_medium_param_matched"), "full_msa_to_face"),
        args,
    )
    assert cfg.simplex_signed_tetra_to_face_scale == 0.25
    _enforce_parameter_budget(
        variant="full_msa_to_face",
        parameter_count=3_240_738,
        max_parameters=args.max_parameters,
    )

    training_config = TrainingConfig(
        simplex_signed_tetra_to_face_runtime_scale=args.simplex_signed_tetra_to_face_runtime_scale,
        simplex_signed_tetra_to_face_runtime_scale_final=args.simplex_signed_tetra_to_face_runtime_scale_final,
        simplex_signed_tetra_to_face_runtime_scale_ramp_start_step=(
            args.simplex_signed_tetra_to_face_runtime_scale_ramp_start_step
        ),
        simplex_signed_tetra_to_face_runtime_scale_ramp_steps=(
            args.simplex_signed_tetra_to_face_runtime_scale_ramp_steps
        ),
    )
    assert simplex_signed_tetra_to_face_runtime_scale_at_step(training_config, 8500) == 0.0
    assert simplex_signed_tetra_to_face_runtime_scale_at_step(training_config, 8750) == 0.125
    assert simplex_signed_tetra_to_face_runtime_scale_at_step(training_config, 9000) == 0.25


def test_e144_edge_star_residual_recipe_matches_documented_gate():
    args = parse_args(
        [
            "--variants",
            "full_msa_to_face",
            "--run-name",
            "e144_no_hodge_edge_star_residual_from_e128_s9000_c256_m64",
            "--steps",
            "9000",
            "--batch-size",
            "1",
            "--grad-accum-steps",
            "8",
            "--crop-size",
            "256",
            "--msa-depth",
            "64",
            "--extra-msa-depth",
            "0",
            "--max-templates",
            "0",
            "--max-parameters",
            "3261974",
            "--resume-model-weights-only",
            "--simplex-boundary-edge-frame-gate-scale",
            "0.05",
            "--simplex-triangle-attention-bias-scale",
            "0.0125",
            "--simplex-boundary-edge-star-residual-scale",
            "0.25",
            "--simplex-boundary-edge-star-residual-runtime-scale",
            "0.0",
            "--simplex-boundary-edge-star-residual-runtime-scale-final",
            "0.25",
            "--simplex-boundary-edge-star-residual-runtime-scale-ramp-start-step",
            "8500",
            "--simplex-boundary-edge-star-residual-runtime-scale-ramp-steps",
            "500",
        ]
    )

    assert args.run_name == "e144_no_hodge_edge_star_residual_from_e128_s9000_c256_m64"
    assert args.steps == 9000
    assert args.batch_size * args.grad_accum_steps == 8
    assert args.crop_size == 256
    assert args.msa_depth == 64
    assert args.extra_msa_depth == 0
    assert args.max_templates == 0
    assert args.max_parameters == 3_261_974
    assert args.num_workers == 0
    assert args.resume_model_weights_only is True
    assert args.simplex_boundary_edge_frame_gate_scale == 0.05
    assert args.simplex_triangle_attention_bias_scale == 0.0125

    cfg = _apply_model_config_overrides(
        _variant_config(load_model_config("simplexfold_medium_param_matched"), "full_msa_to_face"),
        args,
    )
    assert cfg.simplex_boundary_edge_star_residual_scale == 0.25
    _enforce_parameter_budget(
        variant="full_msa_to_face",
        parameter_count=3_240_738,
        max_parameters=args.max_parameters,
    )

    training_config = TrainingConfig(
        simplex_boundary_edge_star_residual_runtime_scale=args.simplex_boundary_edge_star_residual_runtime_scale,
        simplex_boundary_edge_star_residual_runtime_scale_final=(
            args.simplex_boundary_edge_star_residual_runtime_scale_final
        ),
        simplex_boundary_edge_star_residual_runtime_scale_ramp_start_step=(
            args.simplex_boundary_edge_star_residual_runtime_scale_ramp_start_step
        ),
        simplex_boundary_edge_star_residual_runtime_scale_ramp_steps=(
            args.simplex_boundary_edge_star_residual_runtime_scale_ramp_steps
        ),
    )
    assert simplex_boundary_edge_star_residual_runtime_scale_at_step(training_config, 8500) == 0.0
    assert simplex_boundary_edge_star_residual_runtime_scale_at_step(training_config, 8750) == 0.125
    assert simplex_boundary_edge_star_residual_runtime_scale_at_step(training_config, 9000) == 0.25


def test_e145_outer_residual_context_recipe_matches_documented_gate():
    args = parse_args(
        [
            "--variants",
            "full_msa_to_face",
            "--run-name",
            "e145_outer_residual_context_from_e128_s9000_c256_m64",
            "--steps",
            "9000",
            "--batch-size",
            "1",
            "--grad-accum-steps",
            "8",
            "--crop-size",
            "256",
            "--msa-depth",
            "64",
            "--extra-msa-depth",
            "0",
            "--max-templates",
            "0",
            "--max-parameters",
            "3261974",
            "--num-workers",
            "4",
            "--resume-model-weights-only",
            "--simplex-outer-edge-residual-context-scale",
            "0.25",
            "--simplex-outer-edge-residual-context-runtime-scale",
            "0.0",
            "--simplex-outer-edge-residual-context-runtime-scale-final",
            "0.25",
            "--simplex-outer-edge-residual-context-runtime-scale-ramp-start-step",
            "8500",
            "--simplex-outer-edge-residual-context-runtime-scale-ramp-steps",
            "500",
        ]
    )

    assert args.run_name == "e145_outer_residual_context_from_e128_s9000_c256_m64"
    assert args.steps == 9000
    assert args.batch_size * args.grad_accum_steps == 8
    assert args.crop_size == 256
    assert args.msa_depth == 64
    assert args.extra_msa_depth == 0
    assert args.max_templates == 0
    assert args.max_parameters == 3_261_974
    assert args.num_workers == 4
    assert args.resume_model_weights_only is True

    cfg = _apply_model_config_overrides(
        _variant_config(load_model_config("simplexfold_medium_param_matched"), "full_msa_to_face"),
        args,
    )
    assert cfg.simplex_outer_edge_residual_context_scale == 0.25
    _enforce_parameter_budget(
        variant="full_msa_to_face",
        parameter_count=3_240_738,
        max_parameters=args.max_parameters,
    )

    training_config = TrainingConfig(
        simplex_outer_edge_residual_context_runtime_scale=args.simplex_outer_edge_residual_context_runtime_scale,
        simplex_outer_edge_residual_context_runtime_scale_final=(
            args.simplex_outer_edge_residual_context_runtime_scale_final
        ),
        simplex_outer_edge_residual_context_runtime_scale_ramp_start_step=(
            args.simplex_outer_edge_residual_context_runtime_scale_ramp_start_step
        ),
        simplex_outer_edge_residual_context_runtime_scale_ramp_steps=(
            args.simplex_outer_edge_residual_context_runtime_scale_ramp_steps
        ),
    )
    assert simplex_outer_edge_residual_context_runtime_scale_at_step(training_config, 8500) == 0.0
    assert simplex_outer_edge_residual_context_runtime_scale_at_step(training_config, 8750) == 0.125
    assert simplex_outer_edge_residual_context_runtime_scale_at_step(training_config, 9000) == 0.25


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
            "--simplex-boundary-degree-normalize",
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
    assert args.simplex_boundary_degree_normalize is True


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
            simplex_boundary_degree_normalize=True,
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
    assert loss_fn.simplex_geometry_loss.boundary_degree_normalize is True
