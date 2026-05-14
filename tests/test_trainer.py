import os
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from minalphafold.losses import AlphaFoldLoss
from minalphafold.model import AlphaFold2
from tests.test_data_pipeline import write_processed_cache


def _linear(module: object) -> torch.nn.Linear:
    """Narrow ``nn.Module.__getattr__`` → ``Tensor | Module`` down to ``nn.Linear``.

    PyTorch's ``Module.__getattr__`` is typed as ``Tensor | Module`` so
    chained attribute access like ``model.evoformer_blocks[0].msa_row_att
    .linear_output`` comes out as a bare ``Module`` with no ``.weight`` /
    ``.bias`` visible to the type checker. This helper asserts at runtime
    that the target is an ``nn.Linear`` and returns the narrowed type —
    stricter than a bare ``cast`` and no less compact.
    """
    assert isinstance(module, torch.nn.Linear), f"expected nn.Linear, got {type(module).__name__}"
    return module


from minalphafold.trainer import (
    CONFIGS_DIR,
    OptimizerConfig,
    StageConfig,
    TrainingProtocol,
    build_ema_model,
    build_optimizer,
    DataConfig,
    TrainingConfig,
    apply_model_config_cli_overrides,
    apply_loss_weight_schedule,
    build_dataloader,
    evaluate,
    fit,
    learning_rate_at_step,
    learning_rate_for_samples,
    learning_rate_for_step,
    list_available_profiles,
    list_available_training_protocols,
    load_checkpoint_for_resume,
    load_model_config,
    load_training_protocol,
    main,
    model_inputs_from_batch,
    parse_args,
    simplex_boundary_cochain_recycling_runtime_scale_at_step,
    simplex_boundary_metric_gate_runtime_scale_at_step,
    simplex_boundary_metric_recycling_runtime_scale_at_step,
    simplex_boundary_pair_feedback_runtime_scale_at_step,
    simplex_boundary_pair_gate_runtime_scale_at_step,
    simplex_boundary_readout_directionality_runtime_scale_at_step,
    simplex_cell_score_outer_edge_weight_at_step,
    simplex_edge_star_context_runtime_scale_at_step,
    save_checkpoint,
    simplex_hodge_face_runtime_scale_at_step,
    simplex_local_neighbor_k_at_step,
    simplex_msa_feedback_runtime_scale_at_step,
    simplex_pair_update_runtime_scale_at_step,
    simplex_pre_triangle_single_update_runtime_scale_at_step,
    simplex_pre_triangle_update_runtime_scale_at_step,
    simplex_segment_cell_runtime_scale_at_step,
    simplex_single_update_runtime_scale_at_step,
    simplex_update_scale_at_step,
    simplex_topology_teacher_forcing_weight_at_step,
    simplex_vertex_star_context_runtime_scale_at_step,
    train_step,
    use_finetune_loss,
    zero_dropout_model_config,
)


def make_processed_cache_dirs(tmp_path: Path) -> tuple[Path, Path]:
    feature_dir = tmp_path / "processed_features"
    label_dir = tmp_path / "processed_labels"
    feature_dir.mkdir()
    label_dir.mkdir()
    write_processed_cache(feature_dir, label_dir, "1abc_A", "AGAGA", include_templates=True)
    write_processed_cache(feature_dir, label_dir, "2xyz_A", "AGGAA", include_templates=False)
    return feature_dir, label_dir


def test_simplicial_runtime_overrides_reach_model_path():
    import inspect

    from minalphafold.evoformer import SimplicialEvoformer
    from minalphafold.simplex import SimplicialAdapter

    for parameter in (
        "simplex_cell_score_outer_edge_weight_override",
        "simplex_msa_feedback_scale_override",
        "simplex_boundary_pair_feedback_scale_override",
        "simplex_boundary_pair_gate_scale_override",
        "simplex_boundary_metric_gate_scale_override",
    ):
        assert parameter in inspect.signature(AlphaFold2.forward).parameters
        assert parameter in inspect.signature(SimplicialEvoformer.forward).parameters
        assert parameter in inspect.signature(SimplicialAdapter.forward).parameters
    assert "simplex_boundary_metric_recycling_scale_override" in inspect.signature(AlphaFold2.forward).parameters
    assert "simplex_boundary_cochain_recycling_scale_override" in inspect.signature(AlphaFold2.forward).parameters
    for parameter in (
        "simplex_pre_triangle_update_scale_override",
        "simplex_pre_triangle_single_update_scale_override",
    ):
        assert parameter in inspect.signature(AlphaFold2.forward).parameters
        assert parameter in inspect.signature(SimplicialEvoformer.forward).parameters


def test_train_step_updates_model_parameters(tmp_path):
    feature_dir, label_dir = make_processed_cache_dirs(tmp_path)
    data_config = DataConfig(
        processed_features_dir=feature_dir,
        processed_labels_dir=label_dir,
        val_fraction=0.0,
        crop_size=8,
        msa_depth=3,
        extra_msa_depth=2,
        max_templates=1,
    )
    training_config = TrainingConfig(
        epochs=1,
        batch_size=1,
        device="cpu",
        seed=0,
        n_cycles=1,
        n_ensemble=1,
        simplex_topology_teacher_forcing_weight=1.0,
    )

    dataloader = build_dataloader(
        "all",
        data_config,
        training=True,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
        device=training_config.device,
        seed=training_config.seed,
    )
    batch = next(iter(dataloader))

    model = AlphaFold2(load_model_config("tiny"))
    loss_fn = AlphaFoldLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    metrics = train_step(model, loss_fn, optimizer, batch, training_config)

    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert any(
        not torch.allclose(previous, current.detach())
        for previous, current in zip(before, model.parameters())
    )


def test_model_inputs_add_training_only_simplex_curricula():
    batch = {
        "target_feat": torch.zeros(1, 4, 22),
        "residue_index": torch.arange(4).reshape(1, 4),
        "msa_feat": torch.zeros(1, 2, 4, 49),
        "extra_msa_feat": torch.zeros(1, 1, 4, 25),
        "template_pair_feat": torch.zeros(1, 0, 4, 4, 88),
        "aatype": torch.zeros(1, 4, dtype=torch.long),
        "template_angle_feat": torch.zeros(1, 0, 4, 57),
        "template_mask": torch.zeros(1, 0),
        "template_residue_mask": torch.zeros(1, 0, 4),
        "seq_mask": torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
        "msa_mask": torch.ones(1, 2, 4),
        "extra_msa_mask": torch.ones(1, 1, 4),
        "true_atom_positions": torch.zeros(1, 4, 14, 3),
        "true_atom_mask": torch.ones(1, 4, 14),
    }
    training_config = TrainingConfig(
        simplex_topology_teacher_forcing_weight=1.0,
        simplex_topology_teacher_forcing_weight_final=0.0,
        simplex_topology_teacher_forcing_ramp_start_step=10,
        simplex_topology_teacher_forcing_ramp_steps=10,
        simplex_update_scale=0.25,
        simplex_update_scale_final=1.0,
        simplex_update_scale_ramp_start_step=10,
        simplex_update_scale_ramp_steps=10,
        simplex_pair_update_runtime_scale=0.5,
        simplex_pair_update_runtime_scale_final=1.0,
        simplex_pair_update_runtime_scale_ramp_start_step=10,
        simplex_pair_update_runtime_scale_ramp_steps=10,
        simplex_single_update_runtime_scale=1.0,
        simplex_single_update_runtime_scale_final=0.0,
        simplex_single_update_runtime_scale_ramp_start_step=10,
        simplex_single_update_runtime_scale_ramp_steps=10,
        simplex_hodge_face_runtime_scale=0.0,
        simplex_hodge_face_runtime_scale_final=0.1,
        simplex_hodge_face_runtime_scale_ramp_start_step=10,
        simplex_hodge_face_runtime_scale_ramp_steps=10,
        simplex_boundary_readout_directionality_runtime_scale=0.0,
        simplex_boundary_readout_directionality_runtime_scale_final=0.5,
        simplex_boundary_readout_directionality_runtime_scale_ramp_start_step=10,
        simplex_boundary_readout_directionality_runtime_scale_ramp_steps=10,
        simplex_vertex_star_context_runtime_scale=0.0,
        simplex_vertex_star_context_runtime_scale_final=1.0,
        simplex_vertex_star_context_runtime_scale_ramp_start_step=10,
        simplex_vertex_star_context_runtime_scale_ramp_steps=10,
        simplex_edge_star_context_runtime_scale=1.0,
        simplex_edge_star_context_runtime_scale_final=0.0,
        simplex_edge_star_context_runtime_scale_ramp_start_step=10,
        simplex_edge_star_context_runtime_scale_ramp_steps=10,
        simplex_pre_triangle_update_runtime_scale=0.0,
        simplex_pre_triangle_update_runtime_scale_final=0.25,
        simplex_pre_triangle_update_runtime_scale_ramp_start_step=10,
        simplex_pre_triangle_update_runtime_scale_ramp_steps=10,
        simplex_pre_triangle_single_update_runtime_scale=0.0,
        simplex_pre_triangle_single_update_runtime_scale_final=0.0,
        simplex_pre_triangle_single_update_runtime_scale_ramp_start_step=10,
        simplex_pre_triangle_single_update_runtime_scale_ramp_steps=10,
        simplex_segment_cell_runtime_scale=0.0,
        simplex_segment_cell_runtime_scale_final=0.1,
        simplex_segment_cell_runtime_scale_ramp_start_step=10,
        simplex_segment_cell_runtime_scale_ramp_steps=10,
        simplex_msa_feedback_runtime_scale=0.0,
        simplex_msa_feedback_runtime_scale_final=0.1,
        simplex_msa_feedback_runtime_scale_ramp_start_step=10,
        simplex_msa_feedback_runtime_scale_ramp_steps=10,
        simplex_boundary_pair_feedback_runtime_scale=0.0,
        simplex_boundary_pair_feedback_runtime_scale_final=0.1,
        simplex_boundary_pair_feedback_runtime_scale_ramp_start_step=10,
        simplex_boundary_pair_feedback_runtime_scale_ramp_steps=10,
        simplex_boundary_pair_gate_runtime_scale=0.0,
        simplex_boundary_pair_gate_runtime_scale_final=0.1,
        simplex_boundary_pair_gate_runtime_scale_ramp_start_step=10,
        simplex_boundary_pair_gate_runtime_scale_ramp_steps=10,
        simplex_boundary_metric_gate_runtime_scale=0.0,
        simplex_boundary_metric_gate_runtime_scale_final=0.1,
        simplex_boundary_metric_gate_runtime_scale_ramp_start_step=10,
        simplex_boundary_metric_gate_runtime_scale_ramp_steps=10,
        simplex_boundary_metric_recycling_runtime_scale=0.0,
        simplex_boundary_metric_recycling_runtime_scale_final=0.1,
        simplex_boundary_metric_recycling_runtime_scale_ramp_start_step=10,
        simplex_boundary_metric_recycling_runtime_scale_ramp_steps=10,
        simplex_boundary_cochain_recycling_runtime_scale=0.0,
        simplex_boundary_cochain_recycling_runtime_scale_final=0.1,
        simplex_boundary_cochain_recycling_runtime_scale_ramp_start_step=10,
        simplex_boundary_cochain_recycling_runtime_scale_ramp_steps=10,
        simplex_local_neighbor_k=4.0,
        simplex_local_neighbor_k_final=0.0,
        simplex_local_neighbor_k_ramp_start_step=10,
        simplex_local_neighbor_k_ramp_steps=10,
        simplex_cell_score_outer_edge_weight=0.0,
        simplex_cell_score_outer_edge_weight_final=0.2,
        simplex_cell_score_outer_edge_weight_ramp_start_step=10,
        simplex_cell_score_outer_edge_weight_ramp_steps=10,
    )

    assert simplex_topology_teacher_forcing_weight_at_step(training_config, 15) == 0.5
    assert simplex_update_scale_at_step(training_config, 15) == 0.625
    assert simplex_pair_update_runtime_scale_at_step(training_config, 15) == 0.75
    assert simplex_single_update_runtime_scale_at_step(training_config, 15) == 0.5
    assert simplex_hodge_face_runtime_scale_at_step(training_config, 15) == 0.05
    assert simplex_boundary_readout_directionality_runtime_scale_at_step(training_config, 15) == 0.25
    assert simplex_vertex_star_context_runtime_scale_at_step(training_config, 15) == 0.5
    assert simplex_edge_star_context_runtime_scale_at_step(training_config, 15) == 0.5
    assert simplex_pre_triangle_update_runtime_scale_at_step(training_config, 15) == 0.125
    assert simplex_pre_triangle_single_update_runtime_scale_at_step(training_config, 15) == 0.0
    assert simplex_segment_cell_runtime_scale_at_step(training_config, 15) == 0.05
    assert simplex_msa_feedback_runtime_scale_at_step(training_config, 15) == 0.05
    assert simplex_boundary_pair_feedback_runtime_scale_at_step(training_config, 15) == 0.05
    assert simplex_boundary_pair_gate_runtime_scale_at_step(training_config, 15) == 0.05
    assert simplex_boundary_metric_gate_runtime_scale_at_step(training_config, 15) == 0.05
    assert simplex_boundary_metric_recycling_runtime_scale_at_step(training_config, 15) == 0.05
    assert simplex_boundary_cochain_recycling_runtime_scale_at_step(training_config, 15) == 0.05
    assert simplex_local_neighbor_k_at_step(training_config, 15) == 2.0
    assert simplex_cell_score_outer_edge_weight_at_step(training_config, 15) == 0.1

    eval_inputs = model_inputs_from_batch(batch, training_config)
    assert "simplex_teacher_ca_coords" not in eval_inputs
    assert "simplex_pair_update_scale_override" not in eval_inputs
    assert "simplex_hodge_face_update_scale_override" not in eval_inputs
    assert "simplex_boundary_readout_directionality_override" not in eval_inputs
    assert "simplex_vertex_star_context_scale_override" not in eval_inputs
    assert "simplex_edge_star_context_scale_override" not in eval_inputs
    assert "simplex_pre_triangle_update_scale_override" not in eval_inputs
    assert "simplex_pre_triangle_single_update_scale_override" not in eval_inputs
    assert "simplex_segment_cell_scale_override" not in eval_inputs
    assert "simplex_msa_feedback_scale_override" not in eval_inputs
    assert "simplex_boundary_pair_feedback_scale_override" not in eval_inputs
    assert "simplex_boundary_pair_gate_scale_override" not in eval_inputs
    assert "simplex_boundary_metric_gate_scale_override" not in eval_inputs
    assert "simplex_boundary_metric_recycling_scale_override" not in eval_inputs
    assert "simplex_boundary_cochain_recycling_scale_override" not in eval_inputs
    assert "simplex_local_neighbor_k_override" not in eval_inputs
    assert "simplex_cell_score_outer_edge_weight_override" not in eval_inputs

    train_inputs = model_inputs_from_batch(
        batch,
        training_config,
        use_simplex_teacher_forcing=True,
        use_simplex_update_scale=True,
        use_simplex_hodge_face_runtime_scale=True,
        use_simplex_boundary_readout_directionality_runtime_scale=True,
        use_simplex_vertex_star_context_runtime_scale=True,
        use_simplex_edge_star_context_runtime_scale=True,
        use_simplex_pre_triangle_runtime_scale=True,
        use_simplex_segment_cell_runtime_scale=True,
        use_simplex_msa_feedback_runtime_scale=True,
        use_simplex_boundary_pair_feedback_runtime_scale=True,
        use_simplex_boundary_pair_gate_runtime_scale=True,
        use_simplex_boundary_metric_gate_runtime_scale=True,
        use_simplex_boundary_metric_recycling_runtime_scale=True,
        use_simplex_boundary_cochain_recycling_runtime_scale=True,
        use_simplex_local_neighbor_k=True,
        use_simplex_cell_top_k=True,
        step=15,
    )
    assert torch.allclose(train_inputs["simplex_teacher_ca_coords"], batch["true_atom_positions"][:, :, 1, :])
    assert torch.allclose(train_inputs["simplex_teacher_ca_mask"], batch["seq_mask"])
    assert torch.allclose(train_inputs["simplex_teacher_forcing_weight"], torch.tensor(0.5))
    assert torch.allclose(train_inputs["simplex_pair_update_scale_override"], torch.tensor(0.75))
    assert torch.allclose(train_inputs["simplex_single_update_scale_override"], torch.tensor(0.5))
    assert torch.allclose(train_inputs["simplex_hodge_face_update_scale_override"], torch.tensor(0.05))
    assert torch.allclose(train_inputs["simplex_boundary_readout_directionality_override"], torch.tensor(0.25))
    assert torch.allclose(train_inputs["simplex_vertex_star_context_scale_override"], torch.tensor(0.5))
    assert torch.allclose(train_inputs["simplex_edge_star_context_scale_override"], torch.tensor(0.5))
    assert torch.allclose(train_inputs["simplex_pre_triangle_update_scale_override"], torch.tensor(0.125))
    assert torch.allclose(train_inputs["simplex_pre_triangle_single_update_scale_override"], torch.tensor(0.0))
    assert torch.allclose(train_inputs["simplex_segment_cell_scale_override"], torch.tensor(0.05))
    assert torch.allclose(train_inputs["simplex_msa_feedback_scale_override"], torch.tensor(0.05))
    assert torch.allclose(train_inputs["simplex_boundary_pair_feedback_scale_override"], torch.tensor(0.05))
    assert torch.allclose(train_inputs["simplex_boundary_pair_gate_scale_override"], torch.tensor(0.05))
    assert torch.allclose(train_inputs["simplex_boundary_metric_gate_scale_override"], torch.tensor(0.05))
    assert torch.allclose(train_inputs["simplex_boundary_metric_recycling_scale_override"], torch.tensor(0.05))
    assert torch.allclose(train_inputs["simplex_boundary_cochain_recycling_scale_override"], torch.tensor(0.05))
    assert torch.allclose(train_inputs["simplex_local_neighbor_k_override"], torch.tensor(2.0))
    assert torch.allclose(train_inputs["simplex_cell_score_outer_edge_weight_override"], torch.tensor(0.1))


def test_alphafold2_uses_canonical_constructor_initialization():
    # ``get_submodule`` returns ``nn.Module`` directly — bypasses the
    # ``Tensor | Module`` ambiguity of chained ``__getattr__`` on a
    # ``ModuleList`` — so the helper ``_linear`` narrows the leaf cleanly.
    model = AlphaFold2(load_model_config("tiny"))

    row_output = _linear(model.get_submodule("evoformer_blocks.0.msa_row_att.linear_output"))
    assert torch.allclose(row_output.weight, torch.zeros_like(row_output.weight))

    row_gate = _linear(model.get_submodule("evoformer_blocks.0.msa_row_att.linear_gate"))
    assert torch.allclose(row_gate.bias, torch.ones_like(row_gate.bias))

    tmo_out = _linear(model.get_submodule("evoformer_blocks.0.triangle_mult_out.out_linear"))
    assert torch.allclose(tmo_out.weight, torch.zeros_like(tmo_out.weight))

    tmo_gate = _linear(model.get_submodule("evoformer_blocks.0.triangle_mult_out.gate"))
    assert torch.allclose(tmo_gate.bias, torch.ones_like(tmo_gate.bias))

    input_msa = _linear(model.get_submodule("input_embedder.linear_msa"))
    assert torch.allclose(input_msa.bias, torch.zeros_like(input_msa.bias))
    assert not torch.allclose(input_msa.weight, torch.zeros_like(input_msa.weight))

    tm_head = _linear(model.get_submodule("tm_score_head.linear"))
    assert torch.allclose(tm_head.weight, torch.zeros_like(tm_head.weight))


def test_evaluate_returns_finite_mean_loss_without_gradients(tmp_path):
    feature_dir, label_dir = make_processed_cache_dirs(tmp_path)
    data_config = DataConfig(
        processed_features_dir=feature_dir,
        processed_labels_dir=label_dir,
        val_fraction=0.0,
        crop_size=8,
        msa_depth=3,
        extra_msa_depth=2,
        max_templates=1,
    )
    training_config = TrainingConfig(
        epochs=1,
        batch_size=1,
        device="cpu",
        seed=0,
        n_cycles=1,
        n_ensemble=1,
    )
    dataloader = build_dataloader(
        "all",
        data_config,
        training=False,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
        device=training_config.device,
        seed=training_config.seed,
    )

    model = AlphaFold2(load_model_config("tiny"))
    loss_fn = AlphaFoldLoss()
    metrics = evaluate(model, loss_fn, dataloader, training_config)

    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert all(parameter.grad is None for parameter in model.parameters())


def test_fit_runs_and_writes_checkpoints(tmp_path):
    feature_dir, label_dir = make_processed_cache_dirs(tmp_path)
    latest_path = tmp_path / "latest.pt"
    best_path = tmp_path / "best.pt"

    model, history = fit(
        model_config=load_model_config("tiny"),
        data_config=DataConfig(
            processed_features_dir=feature_dir,
            processed_labels_dir=label_dir,
            val_fraction=0.5,
            crop_size=8,
            msa_depth=3,
            extra_msa_depth=2,
            max_templates=1,
        ),
        training_config=TrainingConfig(
            epochs=2,
            batch_size=1,
            device="cpu",
            seed=0,
            n_cycles=1,
            n_ensemble=1,
            latest_checkpoint_path=latest_path,
            best_checkpoint_path=best_path,
        ),
    )

    assert isinstance(model, AlphaFold2)
    assert len(history) == 2
    assert history[0]["epoch"] == 1
    assert "train_loss" in history[0]
    assert "val_loss" in history[0]
    assert latest_path.exists()
    assert best_path.exists()


def test_main_runs_one_epoch_from_cli_args(tmp_path):
    feature_dir, label_dir = make_processed_cache_dirs(tmp_path)
    latest_path = tmp_path / "cli_latest.pt"

    model, history = main(
        [
            "--processed-features-dir",
            str(feature_dir),
            "--processed-labels-dir",
            str(label_dir),
            "--val-fraction",
            "0.5",
            "--crop-size",
            "8",
            "--msa-depth",
            "3",
            "--extra-msa-depth",
            "2",
            "--max-templates",
            "1",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--device",
            "cpu",
            "--n-cycles",
            "1",
            "--n-ensemble",
            "1",
            "--latest-checkpoint-path",
            str(latest_path),
        ]
    )

    assert isinstance(model, AlphaFold2)
    assert len(history) == 1
    assert history[0]["epoch"] == 1
    assert latest_path.exists()


def test_load_model_config_selects_requested_profile():
    assert load_model_config("tiny").model_profile == "tiny"
    assert load_model_config("medium").model_profile == "medium"
    assert load_model_config("alphafold2").model_profile == "alphafold2"
    assert load_model_config("simplexfold_param_matched").model_profile == "simplexfold_param_matched"
    assert load_model_config("simplexfold_medium_topology_plus").model_profile == "simplexfold_medium_topology_plus"
    assert load_model_config("simplexfold_medium_width_matched").model_profile == "simplexfold_medium_width_matched"


def test_load_model_config_accepts_an_explicit_toml_path():
    path = CONFIGS_DIR / "tiny.toml"
    assert load_model_config(path).model_profile == "tiny"
    assert load_model_config(str(path)).model_profile == "tiny"


def test_trainer_cli_accepts_simplex_star_context_overrides():
    args = parse_args(
        [
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
            "--simplex-boundary-edge-frame-gate-scale",
            "0.05",
            "--simplex-vertex-star-context-runtime-scale",
            "0.0",
            "--simplex-vertex-star-context-runtime-scale-final",
            "1.0",
            "--simplex-vertex-star-context-runtime-scale-ramp-start-step",
            "6000",
            "--simplex-vertex-star-context-runtime-scale-ramp-steps",
            "500",
            "--simplex-edge-star-context-runtime-scale",
            "1.0",
            "--simplex-edge-star-context-runtime-scale-final",
            "0.0",
            "--simplex-edge-star-context-runtime-scale-ramp-start-step",
            "6000",
            "--simplex-edge-star-context-runtime-scale-ramp-steps",
            "500",
            "--simplex-pre-triangle-update-runtime-scale",
            "0.0",
            "--simplex-pre-triangle-update-runtime-scale-final",
            "0.25",
            "--simplex-pre-triangle-update-runtime-scale-ramp-start-step",
            "6000",
            "--simplex-pre-triangle-update-runtime-scale-ramp-steps",
            "500",
            "--simplex-pre-triangle-single-update-runtime-scale",
            "0.0",
            "--simplex-pre-triangle-single-update-runtime-scale-final",
            "0.0",
            "--simplex-pre-triangle-single-update-runtime-scale-ramp-start-step",
            "6000",
            "--simplex-pre-triangle-single-update-runtime-scale-ramp-steps",
            "500",
        ]
    )

    cfg = apply_model_config_cli_overrides(load_model_config("simplexfold_medium_param_matched"), args)

    assert cfg.simplex_global_context_scale == 0.125
    assert cfg.simplex_vertex_star_context_scale == 0.75
    assert cfg.simplex_edge_star_context_scale == 0.5
    assert cfg.simplex_pre_triangle_update_scale == 0.25
    assert cfg.simplex_pre_triangle_single_update_scale == 0.0
    assert cfg.simplex_boundary_edge_frame_gate_scale == 0.05
    assert args.simplex_vertex_star_context_runtime_scale == 0.0
    assert args.simplex_vertex_star_context_runtime_scale_final == 1.0
    assert args.simplex_vertex_star_context_runtime_scale_ramp_start_step == 6000
    assert args.simplex_vertex_star_context_runtime_scale_ramp_steps == 500
    assert args.simplex_edge_star_context_runtime_scale == 1.0
    assert args.simplex_edge_star_context_runtime_scale_final == 0.0
    assert args.simplex_edge_star_context_runtime_scale_ramp_start_step == 6000
    assert args.simplex_edge_star_context_runtime_scale_ramp_steps == 500
    assert args.simplex_pre_triangle_update_runtime_scale == 0.0
    assert args.simplex_pre_triangle_update_runtime_scale_final == 0.25
    assert args.simplex_pre_triangle_update_runtime_scale_ramp_start_step == 6000
    assert args.simplex_pre_triangle_update_runtime_scale_ramp_steps == 500
    assert args.simplex_pre_triangle_single_update_runtime_scale == 0.0
    assert args.simplex_pre_triangle_single_update_runtime_scale_final == 0.0
    assert args.simplex_pre_triangle_single_update_runtime_scale_ramp_start_step == 6000
    assert args.simplex_pre_triangle_single_update_runtime_scale_ramp_steps == 500


def test_load_model_config_raises_for_missing_profile():
    with pytest.raises(FileNotFoundError):
        load_model_config("does_not_exist")


def test_list_available_profiles_includes_shipped_json_configs():
    profiles = list_available_profiles()
    assert {
        "tiny",
        "medium",
        "alphafold2",
        "simplexfold_param_matched",
        "simplexfold_medium_param_matched",
        "simplexfold_medium_topology_plus",
        "simplexfold_medium_width_matched",
    }.issubset(set(profiles))


def test_shipped_profiles_have_expected_scales():
    tiny = load_model_config("tiny")
    medium = load_model_config("medium")
    alphafold2 = load_model_config("alphafold2")

    assert medium.model_profile == "medium"
    assert alphafold2.model_profile == "alphafold2"
    assert medium.c_m > tiny.c_m
    assert medium.c_z > tiny.c_z
    assert medium.num_evoformer > tiny.num_evoformer
    # alphafold2 profile keeps the AF2-sized base trunk (supplement 1.5 / 1.6 / Algorithm 22).
    assert alphafold2.c_m == 256
    assert alphafold2.c_s == 384
    assert alphafold2.c_z == 128
    assert alphafold2.num_evoformer == 48
    assert alphafold2.structure_module_layers == 8
    assert alphafold2.ipa_num_heads == 12
    assert alphafold2.ipa_c == 16
    # Supplement 1.7.1 / Algorithm 16: TemplatePair overrides.
    assert alphafold2.template_triangle_mult_c == 64
    assert alphafold2.template_triangle_attn_c == 64
    assert alphafold2.template_pair_transition_n == 2

    param_matched = load_model_config("simplexfold_param_matched")
    assert param_matched.use_simplicial_evoformer
    assert param_matched.num_evoformer == alphafold2.num_evoformer
    assert param_matched.c_m < alphafold2.c_m
    assert param_matched.c_s < alphafold2.c_s
    assert param_matched.c_z < alphafold2.c_z
    assert param_matched.simplex_c_face < alphafold2.simplex_c_face
    assert param_matched.simplex_c_tetra < alphafold2.simplex_c_tetra

    medium_param_matched = load_model_config("simplexfold_medium_param_matched")
    assert medium_param_matched.use_simplicial_evoformer
    assert medium_param_matched.num_evoformer == medium.num_evoformer
    assert medium_param_matched.structure_module_layers == medium.structure_module_layers
    assert medium_param_matched.template_pair_num_blocks == medium.template_pair_num_blocks
    assert medium_param_matched.num_extra_msa == medium.num_extra_msa
    assert medium_param_matched.recommended_n_cycles == 4
    assert medium_param_matched.c_m < medium.c_m
    assert medium_param_matched.c_s < medium.c_s
    assert medium_param_matched.c_z < medium.c_z

    medium_width_matched = load_model_config("simplexfold_medium_width_matched")
    assert medium_width_matched.use_simplicial_evoformer
    assert medium_width_matched.c_m == medium.c_m
    assert medium_width_matched.c_s == medium.c_s
    assert medium_width_matched.c_z == medium.c_z
    assert medium_width_matched.structure_module_layers == medium.structure_module_layers
    assert medium_width_matched.template_pair_num_blocks == 0
    assert medium_width_matched.num_extra_msa == 0
    assert medium_width_matched.num_evoformer == 2
    assert medium_width_matched.recommended_n_cycles == 4


def test_simplexfold_medium_param_matched_matches_af2_medium_budget():
    medium = load_model_config("medium")
    af2_medium = replace(medium, use_simplicial_evoformer=False)
    simplex_medium = load_model_config("simplexfold_medium_param_matched")

    af2_params = sum(parameter.numel() for parameter in AlphaFold2(af2_medium).parameters())
    simplex_params = sum(parameter.numel() for parameter in AlphaFold2(simplex_medium).parameters())

    assert af2_params == 3_106_642
    assert simplex_params == 3_106_690
    assert abs(simplex_params - af2_params) <= 64


def test_simplexfold_medium_topology_plus_uses_only_allowed_simplex_headroom():
    medium = load_model_config("medium")
    af2_medium = replace(medium, use_simplicial_evoformer=False)
    param_matched = load_model_config("simplexfold_medium_param_matched")
    topology_plus = load_model_config("simplexfold_medium_topology_plus")

    af2_params = sum(parameter.numel() for parameter in AlphaFold2(af2_medium).parameters())
    topology_plus_params = sum(parameter.numel() for parameter in AlphaFold2(topology_plus).parameters())

    assert af2_params == 3_106_642
    assert topology_plus_params == 3_256_126
    assert topology_plus_params <= int(af2_params * 1.05)
    assert topology_plus.c_m == param_matched.c_m
    assert topology_plus.c_s == param_matched.c_s
    assert topology_plus.c_z == param_matched.c_z
    assert topology_plus.num_evoformer == param_matched.num_evoformer
    assert topology_plus.structure_module_layers == param_matched.structure_module_layers
    assert topology_plus.simplex_c_face > param_matched.simplex_c_face
    assert topology_plus.simplex_c_tetra > param_matched.simplex_c_tetra
    assert topology_plus.simplex_hidden_dim > param_matched.simplex_hidden_dim
    assert topology_plus.simplex_msa_to_face_rank > param_matched.simplex_msa_to_face_rank


def test_simplicial_structure_readout_adds_no_parameters():
    simplex_medium = load_model_config("simplexfold_medium_param_matched")
    readout_medium = replace(simplex_medium, simplex_structure_readout_scale=0.25)

    simplex_params = sum(parameter.numel() for parameter in AlphaFold2(simplex_medium).parameters())
    readout_params = sum(parameter.numel() for parameter in AlphaFold2(readout_medium).parameters())

    assert simplex_params == 3_106_690
    assert readout_params == simplex_params


def test_simplicial_structure_pair_readout_adds_no_parameters():
    simplex_medium = load_model_config("simplexfold_medium_param_matched")
    readout_medium = replace(simplex_medium, simplex_structure_pair_readout_scale=0.25)

    simplex_params = sum(parameter.numel() for parameter in AlphaFold2(simplex_medium).parameters())
    readout_params = sum(parameter.numel() for parameter in AlphaFold2(readout_medium).parameters())

    assert simplex_params == 3_106_690
    assert readout_params == simplex_params


def test_simplicial_boundary_metric_recycling_adds_no_parameters():
    simplex_medium = load_model_config("simplexfold_medium_param_matched")
    recycling_medium = replace(simplex_medium, simplex_boundary_metric_recycling_scale=0.25)

    simplex_params = sum(parameter.numel() for parameter in AlphaFold2(simplex_medium).parameters())
    recycling_params = sum(parameter.numel() for parameter in AlphaFold2(recycling_medium).parameters())

    assert simplex_params == 3_106_690
    assert recycling_params == simplex_params


def test_simplicial_boundary_cochain_recycling_adds_no_parameters():
    simplex_medium = load_model_config("simplexfold_medium_param_matched")
    recycling_medium = replace(simplex_medium, simplex_boundary_cochain_recycling_scale=0.25)

    simplex_params = sum(parameter.numel() for parameter in AlphaFold2(simplex_medium).parameters())
    recycling_params = sum(parameter.numel() for parameter in AlphaFold2(recycling_medium).parameters())

    assert simplex_params == 3_106_690
    assert recycling_params == simplex_params


def test_simplicial_metric_gated_boundary_cochain_recycling_adds_no_parameters():
    simplex_medium = load_model_config("simplexfold_medium_param_matched")
    recycling_medium = replace(
        simplex_medium,
        simplex_boundary_cochain_recycling_scale=0.25,
        simplex_boundary_cochain_recycling_metric_gate_scale=1.0,
    )

    simplex_params = sum(parameter.numel() for parameter in AlphaFold2(simplex_medium).parameters())
    recycling_params = sum(parameter.numel() for parameter in AlphaFold2(recycling_medium).parameters())

    assert simplex_params == 3_106_690
    assert recycling_params == simplex_params


def test_simplicial_expansion_hinge_adds_no_parameters():
    simplex_medium = load_model_config("simplexfold_medium_param_matched")
    expansion_medium = replace(simplex_medium, simplex_use_msa_to_face=True)

    simplex_params = sum(parameter.numel() for parameter in AlphaFold2(simplex_medium).parameters())
    expansion_params = sum(parameter.numel() for parameter in AlphaFold2(expansion_medium).parameters())

    assert simplex_params == 3_106_690
    assert expansion_params == simplex_params


def test_simplicial_outer_edge_update_adds_no_parameters():
    simplex_medium = load_model_config("simplexfold_medium_param_matched")
    outer_edge_medium = replace(simplex_medium, simplex_outer_edge_update_scale=0.25)

    simplex_params = sum(parameter.numel() for parameter in AlphaFold2(simplex_medium).parameters())
    outer_edge_params = sum(parameter.numel() for parameter in AlphaFold2(outer_edge_medium).parameters())

    assert simplex_params == 3_106_690
    assert outer_edge_params == simplex_params


def test_simplicial_outer_edge_context_stays_within_medium_budget():
    af2_medium = replace(load_model_config("medium"), use_simplicial_evoformer=False)
    simplex_medium = load_model_config("simplexfold_medium_param_matched")
    outer_edge_context_medium = replace(
        simplex_medium,
        simplex_use_msa_to_face=True,
        simplex_outer_edge_context_scale=0.25,
    )

    af2_params = sum(parameter.numel() for parameter in AlphaFold2(af2_medium).parameters())
    simplex_params = sum(parameter.numel() for parameter in AlphaFold2(simplex_medium).parameters())
    outer_edge_context_params = sum(
        parameter.numel() for parameter in AlphaFold2(outer_edge_context_medium).parameters()
    )

    assert simplex_params == 3_106_690
    assert outer_edge_context_params > simplex_params
    assert outer_edge_context_params <= int(af2_params * 1.05)


def test_simplicial_msa_feedback_stays_within_medium_budget():
    af2_medium = replace(load_model_config("medium"), use_simplicial_evoformer=False)
    simplex_medium = load_model_config("simplexfold_medium_param_matched")
    feedback_medium = replace(
        simplex_medium,
        simplex_use_msa_to_face=True,
        simplex_msa_feedback_scale=0.05,
    )

    af2_params = sum(parameter.numel() for parameter in AlphaFold2(af2_medium).parameters())
    simplex_params = sum(parameter.numel() for parameter in AlphaFold2(simplex_medium).parameters())
    feedback_params = sum(parameter.numel() for parameter in AlphaFold2(feedback_medium).parameters())

    assert simplex_params == 3_106_690
    assert feedback_params > simplex_params
    assert feedback_params <= int(af2_params * 1.05)


def test_simplicial_boundary_msa_feedback_stays_within_medium_budget():
    af2_medium = replace(load_model_config("medium"), use_simplicial_evoformer=False)
    simplex_medium = load_model_config("simplexfold_medium_param_matched")
    feedback_medium = replace(
        simplex_medium,
        simplex_use_msa_to_face=True,
        simplex_boundary_msa_feedback_scale=0.05,
    )

    af2_params = sum(parameter.numel() for parameter in AlphaFold2(af2_medium).parameters())
    simplex_params = sum(parameter.numel() for parameter in AlphaFold2(simplex_medium).parameters())
    feedback_params = sum(parameter.numel() for parameter in AlphaFold2(feedback_medium).parameters())

    assert simplex_params == 3_106_690
    assert feedback_params > simplex_params
    assert feedback_params <= int(af2_params * 1.05)


def test_simplicial_boundary_pair_feedback_stays_within_medium_budget():
    af2_medium = replace(load_model_config("medium"), use_simplicial_evoformer=False)
    simplex_medium = load_model_config("simplexfold_medium_param_matched")
    feedback_medium = replace(
        simplex_medium,
        simplex_use_msa_to_face=True,
        simplex_boundary_pair_feedback_scale=0.05,
    )

    af2_params = sum(parameter.numel() for parameter in AlphaFold2(af2_medium).parameters())
    simplex_params = sum(parameter.numel() for parameter in AlphaFold2(simplex_medium).parameters())
    feedback_params = sum(parameter.numel() for parameter in AlphaFold2(feedback_medium).parameters())

    assert simplex_params == 3_106_690
    assert feedback_params > simplex_params
    assert feedback_params <= int(af2_params * 1.05)


def test_simplicial_boundary_pair_gate_stays_within_medium_budget():
    af2_medium = replace(load_model_config("medium"), use_simplicial_evoformer=False)
    simplex_medium = load_model_config("simplexfold_medium_param_matched")
    gate_medium = replace(
        simplex_medium,
        simplex_use_msa_to_face=True,
        simplex_boundary_pair_gate_scale=0.05,
    )

    af2_params = sum(parameter.numel() for parameter in AlphaFold2(af2_medium).parameters())
    simplex_params = sum(parameter.numel() for parameter in AlphaFold2(simplex_medium).parameters())
    gate_params = sum(parameter.numel() for parameter in AlphaFold2(gate_medium).parameters())

    assert simplex_params == 3_106_690
    assert gate_params > simplex_params
    assert gate_params <= int(af2_params * 1.05)


def test_simplicial_boundary_metric_gate_adds_no_parameters():
    simplex_medium = load_model_config("simplexfold_medium_param_matched")
    metric_gate_medium = replace(
        simplex_medium,
        simplex_boundary_metric_gate_scale=0.25,
    )

    simplex_params = sum(parameter.numel() for parameter in AlphaFold2(simplex_medium).parameters())
    metric_gate_params = sum(parameter.numel() for parameter in AlphaFold2(metric_gate_medium).parameters())

    assert metric_gate_params == simplex_params


def test_simplicial_hodge_face_update_adds_no_parameters():
    simplex_medium = load_model_config("simplexfold_medium_param_matched")
    hodge_medium = replace(
        simplex_medium,
        simplex_use_msa_to_face=True,
        simplex_hodge_face_update_scale=0.25,
    )

    simplex_params = sum(parameter.numel() for parameter in AlphaFold2(simplex_medium).parameters())
    hodge_params = sum(parameter.numel() for parameter in AlphaFold2(hodge_medium).parameters())

    assert simplex_params == 3_106_690
    assert hodge_params == simplex_params


def test_simplicial_flag_closure_adds_no_parameters():
    simplex_medium = load_model_config("simplexfold_medium_param_matched")
    closure_medium = replace(
        simplex_medium,
        simplex_use_msa_to_face=True,
        simplex_boundary_closure_weight=0.5,
        simplex_boundary_closure_temperature=1.0,
    )

    simplex_params = sum(parameter.numel() for parameter in AlphaFold2(simplex_medium).parameters())
    closure_params = sum(parameter.numel() for parameter in AlphaFold2(closure_medium).parameters())

    assert simplex_params == 3_106_690
    assert closure_params == simplex_params


def test_simplicial_expanded_complex_adds_no_parameters():
    simplex_medium = load_model_config("simplexfold_medium_param_matched")
    expanded_medium = replace(
        simplex_medium,
        simplex_use_msa_to_face=True,
        simplex_neighbor_k=14,
    )

    simplex_params = sum(parameter.numel() for parameter in AlphaFold2(simplex_medium).parameters())
    expanded_params = sum(parameter.numel() for parameter in AlphaFold2(expanded_medium).parameters())

    assert simplex_params == 3_106_690
    assert expanded_params == simplex_params


def test_simplicial_geometry_selector_weight_adds_no_parameters():
    simplex_medium = load_model_config("simplexfold_medium_param_matched")
    light_geometry_medium = replace(
        simplex_medium,
        simplex_use_msa_to_face=True,
        simplex_geometry_distance_weight=0.025,
    )

    simplex_params = sum(parameter.numel() for parameter in AlphaFold2(simplex_medium).parameters())
    light_geometry_params = sum(parameter.numel() for parameter in AlphaFold2(light_geometry_medium).parameters())

    assert simplex_params == 3_106_690
    assert light_geometry_params == simplex_params


def test_simplicial_cell_topk_selector_adds_no_parameters():
    simplex_medium = load_model_config("simplexfold_medium_param_matched")
    sparse_cell_medium = replace(
        simplex_medium,
        simplex_use_msa_to_face=True,
        simplex_face_top_k=24,
        simplex_tetra_top_k=48,
    )

    simplex_params = sum(parameter.numel() for parameter in AlphaFold2(simplex_medium).parameters())
    sparse_cell_params = sum(parameter.numel() for parameter in AlphaFold2(sparse_cell_medium).parameters())

    assert simplex_params == 3_106_690
    assert sparse_cell_params == simplex_params


def test_simplicial_cell_degree_penalty_adds_no_parameters():
    simplex_medium = load_model_config("simplexfold_medium_param_matched")
    degree_penalty_medium = replace(
        simplex_medium,
        simplex_use_msa_to_face=True,
        simplex_face_top_k=24,
        simplex_tetra_top_k=48,
        simplex_cell_score_degree_penalty=0.75,
    )

    simplex_params = sum(parameter.numel() for parameter in AlphaFold2(simplex_medium).parameters())
    degree_penalty_params = sum(parameter.numel() for parameter in AlphaFold2(degree_penalty_medium).parameters())

    assert simplex_params == 3_106_690
    assert degree_penalty_params == simplex_params


def test_simplicial_cell_outer_edge_score_adds_no_parameters():
    simplex_medium = load_model_config("simplexfold_medium_param_matched")
    outer_edge_score_medium = replace(
        simplex_medium,
        simplex_use_msa_to_face=True,
        simplex_face_top_k=24,
        simplex_tetra_top_k=48,
        simplex_cell_score_outer_edge_weight=0.25,
    )

    simplex_params = sum(parameter.numel() for parameter in AlphaFold2(simplex_medium).parameters())
    outer_edge_score_params = sum(
        parameter.numel() for parameter in AlphaFold2(outer_edge_score_medium).parameters()
    )

    assert simplex_params == 3_106_690
    assert outer_edge_score_params == simplex_params


def test_simplicial_cell_segment_score_adds_no_parameters():
    simplex_medium = load_model_config("simplexfold_medium_param_matched")
    segment_score_medium = replace(
        simplex_medium,
        simplex_use_msa_to_face=True,
        simplex_face_top_k=24,
        simplex_tetra_top_k=48,
        simplex_cell_score_segment_weight=0.25,
        simplex_segment_radius=4,
    )

    simplex_params = sum(parameter.numel() for parameter in AlphaFold2(simplex_medium).parameters())
    segment_score_params = sum(
        parameter.numel() for parameter in AlphaFold2(segment_score_medium).parameters()
    )

    assert simplex_params == 3_106_690
    assert segment_score_params == simplex_params


def test_simplicial_boundary_message_degree_attenuation_adds_no_parameters():
    simplex_medium = load_model_config("simplexfold_medium_param_matched")
    attenuated_medium = replace(
        simplex_medium,
        simplex_use_msa_to_face=True,
        simplex_edge_frame_message_scale=0.25,
        simplex_boundary_message_degree_attenuation=1.0,
    )
    edge_frame_medium = replace(
        simplex_medium,
        simplex_use_msa_to_face=True,
        simplex_edge_frame_message_scale=0.25,
    )

    attenuated_params = sum(parameter.numel() for parameter in AlphaFold2(attenuated_medium).parameters())
    edge_frame_params = sum(parameter.numel() for parameter in AlphaFold2(edge_frame_medium).parameters())

    assert attenuated_params == edge_frame_params


def test_simplicial_boundary_incidence_normalization_adds_no_parameters():
    simplex_medium = load_model_config("simplexfold_medium_param_matched")
    incidence_medium = replace(
        simplex_medium,
        simplex_use_msa_to_face=True,
        simplex_face_top_k=24,
        simplex_tetra_top_k=48,
        simplex_boundary_incidence_normalization=1.0,
    )

    simplex_params = sum(parameter.numel() for parameter in AlphaFold2(simplex_medium).parameters())
    incidence_params = sum(parameter.numel() for parameter in AlphaFold2(incidence_medium).parameters())

    assert simplex_params == 3_106_690
    assert incidence_params == simplex_params


def test_simplicial_boundary_readout_directionality_adds_no_parameters():
    simplex_medium = load_model_config("simplexfold_medium_param_matched")
    directed_medium = replace(
        simplex_medium,
        simplex_use_msa_to_face=True,
        simplex_face_top_k=24,
        simplex_tetra_top_k=48,
        simplex_boundary_readout_directionality=1.0,
    )

    simplex_params = sum(parameter.numel() for parameter in AlphaFold2(simplex_medium).parameters())
    directed_params = sum(parameter.numel() for parameter in AlphaFold2(directed_medium).parameters())

    assert simplex_params == 3_106_690
    assert directed_params == simplex_params


def test_simplicial_global_context_stays_inside_af2_medium_budget():
    af2_medium = replace(load_model_config("medium"), use_simplicial_evoformer=False)
    simplex_medium = load_model_config("simplexfold_medium_param_matched")
    global_context_medium = replace(
        simplex_medium,
        simplex_use_msa_to_face=True,
        simplex_face_top_k=24,
        simplex_tetra_top_k=48,
        simplex_cell_score_degree_penalty=0.75,
        simplex_cell_score_outer_edge_weight=0.25,
        simplex_edge_frame_message_scale=0.025,
        simplex_boundary_readout_directionality=0.25,
        simplex_boundary_incidence_normalization=1.0,
        simplex_boundary_cochain_recycling_scale=0.10,
        simplex_global_context_scale=0.10,
    )

    af2_params = sum(parameter.numel() for parameter in AlphaFold2(af2_medium).parameters())
    simplex_params = sum(parameter.numel() for parameter in AlphaFold2(simplex_medium).parameters())
    global_context_params = sum(parameter.numel() for parameter in AlphaFold2(global_context_medium).parameters())

    assert af2_params == 3_106_642
    assert simplex_params == 3_106_690
    assert global_context_params == 3_201_970
    assert global_context_params <= int(af2_params * 1.05)


def test_simplicial_vertex_star_context_adds_no_parameters():
    simplex_medium = load_model_config("simplexfold_medium_param_matched")
    global_context_medium = replace(
        simplex_medium,
        simplex_use_msa_to_face=True,
        simplex_face_top_k=24,
        simplex_tetra_top_k=48,
        simplex_cell_score_degree_penalty=0.75,
        simplex_cell_score_outer_edge_weight=0.25,
        simplex_edge_frame_message_scale=0.025,
        simplex_boundary_readout_directionality=0.25,
        simplex_boundary_incidence_normalization=1.0,
        simplex_global_context_scale=0.10,
    )
    vertex_star_medium = replace(
        global_context_medium,
        simplex_vertex_star_context_scale=1.0,
    )

    global_context_params = sum(parameter.numel() for parameter in AlphaFold2(global_context_medium).parameters())
    vertex_star_params = sum(parameter.numel() for parameter in AlphaFold2(vertex_star_medium).parameters())

    assert global_context_params == 3_201_970
    assert vertex_star_params == global_context_params


def test_simplicial_edge_star_context_adds_no_parameters():
    simplex_medium = load_model_config("simplexfold_medium_param_matched")
    global_context_medium = replace(
        simplex_medium,
        simplex_use_msa_to_face=True,
        simplex_face_top_k=24,
        simplex_tetra_top_k=48,
        simplex_cell_score_degree_penalty=0.75,
        simplex_cell_score_outer_edge_weight=0.25,
        simplex_edge_frame_message_scale=0.025,
        simplex_boundary_readout_directionality=0.25,
        simplex_boundary_incidence_normalization=1.0,
        simplex_global_context_scale=0.10,
    )
    edge_star_medium = replace(
        global_context_medium,
        simplex_edge_star_context_scale=1.0,
    )

    global_context_params = sum(parameter.numel() for parameter in AlphaFold2(global_context_medium).parameters())
    edge_star_params = sum(parameter.numel() for parameter in AlphaFold2(edge_star_medium).parameters())

    assert global_context_params == 3_201_970
    assert edge_star_params == global_context_params


def test_simplicial_pre_triangle_update_adds_no_parameters():
    simplex_medium = load_model_config("simplexfold_medium_param_matched")
    global_context_medium = replace(
        simplex_medium,
        simplex_use_msa_to_face=True,
        simplex_face_top_k=24,
        simplex_tetra_top_k=48,
        simplex_cell_score_degree_penalty=0.75,
        simplex_cell_score_outer_edge_weight=0.25,
        simplex_edge_frame_message_scale=0.025,
        simplex_boundary_readout_directionality=0.25,
        simplex_boundary_incidence_normalization=1.0,
        simplex_global_context_scale=0.10,
        simplex_vertex_star_context_scale=1.0,
        simplex_edge_star_context_scale=1.0,
    )
    pre_triangle_medium = replace(
        global_context_medium,
        simplex_pre_triangle_update_scale=0.25,
    )
    pair_only_pre_triangle_medium = replace(
        pre_triangle_medium,
        simplex_pre_triangle_single_update_scale=0.0,
    )

    global_context_params = sum(parameter.numel() for parameter in AlphaFold2(global_context_medium).parameters())
    pre_triangle_params = sum(parameter.numel() for parameter in AlphaFold2(pre_triangle_medium).parameters())
    pair_only_pre_triangle_params = sum(
        parameter.numel() for parameter in AlphaFold2(pair_only_pre_triangle_medium).parameters()
    )

    assert global_context_params == 3_201_970
    assert pre_triangle_params == global_context_params
    assert pair_only_pre_triangle_params == global_context_params


def test_simplicial_boundary_edge_frame_gate_stays_inside_medium_budget():
    af2_medium = replace(load_model_config("medium"), use_simplicial_evoformer=False)
    simplex_medium = load_model_config("simplexfold_medium_param_matched")
    edge_frame_gate_medium = replace(
        simplex_medium,
        simplex_use_msa_to_face=True,
        simplex_face_top_k=24,
        simplex_tetra_top_k=48,
        simplex_cell_score_degree_penalty=0.75,
        simplex_cell_score_outer_edge_weight=0.25,
        simplex_edge_frame_message_scale=0.025,
        simplex_boundary_edge_frame_gate_scale=0.05,
        simplex_boundary_readout_directionality=0.25,
        simplex_boundary_incidence_normalization=1.0,
        simplex_global_context_scale=0.10,
        simplex_vertex_star_context_scale=1.0,
        simplex_edge_star_context_scale=1.0,
    )

    af2_params = sum(parameter.numel() for parameter in AlphaFold2(af2_medium).parameters())
    simplex_params = sum(parameter.numel() for parameter in AlphaFold2(simplex_medium).parameters())
    edge_frame_gate_params = sum(parameter.numel() for parameter in AlphaFold2(edge_frame_gate_medium).parameters())

    assert af2_params == 3_106_642
    assert simplex_params == 3_106_690
    assert edge_frame_gate_params == 3_239_522
    assert edge_frame_gate_params > 3_201_970
    assert edge_frame_gate_params <= int(af2_params * 1.05)


def test_pre_triangle_simplex_update_runs_evoformer_block_eagerly(monkeypatch):
    import minalphafold.model as model_module
    from minalphafold.evoformer import SimplicialEvoformer

    model_config = replace(load_model_config("tiny"), simplex_pre_triangle_update_scale=0.25)
    model = AlphaFold2(model_config)
    model.train()

    original_checkpoint = model_module.torch_checkpoint.checkpoint

    def checkpoint_spy(function, *args, **kwargs):
        if isinstance(function, SimplicialEvoformer):
            raise AssertionError("pre-triangle simplex blocks must bypass activation checkpointing")
        return original_checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(model_module.torch_checkpoint, "checkpoint", checkpoint_spy)

    target_feat = torch.zeros(1, 4, 22)
    residue_index = torch.arange(4).reshape(1, 4)
    msa_feat = torch.zeros(1, 2, 4, 49)
    extra_msa_feat = torch.zeros(1, 0, 4, 25)
    template_pair_feat = torch.zeros(1, 0, 4, 4, 88)
    aatype = torch.zeros(1, 4, dtype=torch.long)

    outputs = model(
        target_feat,
        residue_index,
        msa_feat,
        extra_msa_feat,
        template_pair_feat,
        aatype,
        n_cycles=1,
    )

    assert outputs["atom14_coords"].shape == (1, 4, 14, 3)


def test_simplicial_cell_dropout_adds_no_parameters():
    simplex_medium = load_model_config("simplexfold_medium_param_matched")
    dropout_medium = replace(
        simplex_medium,
        simplex_use_msa_to_face=True,
        simplex_cell_dropout=0.15,
    )

    simplex_params = sum(parameter.numel() for parameter in AlphaFold2(simplex_medium).parameters())
    dropout_params = sum(parameter.numel() for parameter in AlphaFold2(dropout_medium).parameters())

    assert simplex_params == 3_106_690
    assert dropout_params == simplex_params


def test_simplicial_edge_frame_messages_stay_within_medium_budget():
    medium = load_model_config("medium")
    af2_medium = replace(medium, use_simplicial_evoformer=False)
    simplex_medium = load_model_config("simplexfold_medium_param_matched")
    edge_frame_medium = replace(
        simplex_medium,
        simplex_use_msa_to_face=True,
        simplex_edge_frame_message_scale=0.25,
    )

    af2_params = sum(parameter.numel() for parameter in AlphaFold2(af2_medium).parameters())
    simplex_params = sum(parameter.numel() for parameter in AlphaFold2(simplex_medium).parameters())
    edge_frame_params = sum(parameter.numel() for parameter in AlphaFold2(edge_frame_medium).parameters())

    assert af2_params == 3_106_642
    assert simplex_params == 3_106_690
    assert edge_frame_params > simplex_params
    assert edge_frame_params <= int(af2_params * 1.05)


def test_simplicial_segment_cells_stay_within_medium_budget():
    medium = load_model_config("medium")
    af2_medium = replace(medium, use_simplicial_evoformer=False)
    simplex_medium = load_model_config("simplexfold_medium_param_matched")
    segment_medium = replace(
        simplex_medium,
        simplex_use_msa_to_face=True,
        simplex_segment_cell_scale=0.25,
        simplex_segment_radius=4,
        simplex_c_segment=12,
    )

    af2_params = sum(parameter.numel() for parameter in AlphaFold2(af2_medium).parameters())
    simplex_params = sum(parameter.numel() for parameter in AlphaFold2(simplex_medium).parameters())
    segment_params = sum(parameter.numel() for parameter in AlphaFold2(segment_medium).parameters())

    assert af2_params == 3_106_642
    assert simplex_params == 3_106_690
    assert segment_params > simplex_params
    assert segment_params <= int(af2_params * 1.05)


def test_simplicial_segment_cells_with_edge_frame_messages_exceed_medium_budget():
    medium = load_model_config("medium")
    af2_medium = replace(medium, use_simplicial_evoformer=False)
    simplex_medium = load_model_config("simplexfold_medium_param_matched")
    e88_medium = replace(
        simplex_medium,
        simplex_use_msa_to_face=True,
        simplex_face_top_k=24,
        simplex_tetra_top_k=48,
        simplex_cell_score_degree_penalty=0.75,
        simplex_edge_frame_message_scale=0.025,
        simplex_segment_cell_scale=0.05,
        simplex_segment_radius=4,
        simplex_c_segment=12,
    )

    af2_params = sum(parameter.numel() for parameter in AlphaFold2(af2_medium).parameters())
    e88_params = sum(parameter.numel() for parameter in AlphaFold2(e88_medium).parameters())

    assert af2_params == 3_106_642
    assert e88_params == 3_282_002
    assert e88_params > int(af2_params * 1.05)


def test_simplicial_structure_readout_forward_keeps_internal_tensors_private():
    model_config = replace(load_model_config("tiny"), simplex_structure_readout_scale=0.25)
    model = AlphaFold2(model_config)
    model.eval()
    target_feat = torch.zeros(1, 4, 22)
    residue_index = torch.arange(4).reshape(1, 4)
    msa_feat = torch.zeros(1, 2, 4, 49)
    extra_msa_feat = torch.zeros(1, 0, 4, 25)
    template_pair_feat = torch.zeros(1, 0, 4, 4, 88)
    aatype = torch.zeros(1, 4, dtype=torch.long)

    with torch.no_grad():
        outputs = model(
            target_feat,
            residue_index,
            msa_feat,
            extra_msa_feat,
            template_pair_feat,
            aatype,
            n_cycles=1,
        )

    assert outputs["atom14_coords"].shape == (1, 4, 14, 3)
    assert "simplex_structure_single_readout" not in outputs
    assert "simplex_structure_pair_readout" not in outputs


def test_simplicial_structure_pair_readout_forward_uses_private_pair_cochain():
    torch.manual_seed(23)
    base_config = replace(load_model_config("tiny"), simplex_structure_pair_readout_scale=0.0)
    readout_config = replace(load_model_config("tiny"), simplex_structure_pair_readout_scale=0.25)
    base_model = AlphaFold2(base_config)
    readout_model = AlphaFold2(readout_config)
    readout_model.load_state_dict(base_model.state_dict())
    base_model.eval()
    readout_model.eval()

    target_feat = torch.randn(1, 5, 22)
    residue_index = torch.arange(5).reshape(1, 5)
    msa_feat = torch.randn(1, 2, 5, 49)
    extra_msa_feat = torch.zeros(1, 0, 5, 25)
    template_pair_feat = torch.zeros(1, 0, 5, 5, 88)
    aatype = torch.zeros(1, 5, dtype=torch.long)

    with torch.no_grad():
        base_outputs = base_model(
            target_feat,
            residue_index,
            msa_feat,
            extra_msa_feat,
            template_pair_feat,
            aatype,
            n_cycles=1,
        )
        readout_outputs = readout_model(
            target_feat,
            residue_index,
            msa_feat,
            extra_msa_feat,
            template_pair_feat,
            aatype,
            n_cycles=1,
        )

    assert "simplex_structure_pair_readout" not in readout_outputs
    assert "simplex_structure_single_readout" not in readout_outputs
    assert not torch.allclose(
        readout_outputs["pair_representation"],
        base_outputs["pair_representation"],
    )


def test_simplicial_boundary_metric_recycling_changes_only_recycled_cycles():
    torch.manual_seed(9)
    base_config = replace(load_model_config("tiny"), simplex_boundary_metric_recycling_scale=0.0)
    recycling_config = replace(load_model_config("tiny"), simplex_boundary_metric_recycling_scale=0.5)
    base_model = AlphaFold2(base_config)
    recycling_model = AlphaFold2(recycling_config)
    recycling_model.load_state_dict(base_model.state_dict())
    base_model.eval()
    recycling_model.eval()

    target_feat = torch.zeros(1, 5, 22)
    residue_index = torch.arange(5).reshape(1, 5)
    msa_feat = torch.zeros(1, 2, 5, 49)
    extra_msa_feat = torch.zeros(1, 0, 5, 25)
    template_pair_feat = torch.zeros(1, 0, 5, 5, 88)
    aatype = torch.zeros(1, 5, dtype=torch.long)

    with torch.no_grad():
        base_one_cycle = base_model(
            target_feat,
            residue_index,
            msa_feat,
            extra_msa_feat,
            template_pair_feat,
            aatype,
            n_cycles=1,
        )
        recycling_one_cycle = recycling_model(
            target_feat,
            residue_index,
            msa_feat,
            extra_msa_feat,
            template_pair_feat,
            aatype,
            n_cycles=1,
        )
        base_two_cycles = base_model(
            target_feat,
            residue_index,
            msa_feat,
            extra_msa_feat,
            template_pair_feat,
            aatype,
            n_cycles=2,
        )
        recycling_two_cycles = recycling_model(
            target_feat,
            residue_index,
            msa_feat,
            extra_msa_feat,
            template_pair_feat,
            aatype,
            n_cycles=2,
        )

    assert torch.allclose(base_one_cycle["pair_representation"], recycling_one_cycle["pair_representation"])
    assert not torch.allclose(base_two_cycles["pair_representation"], recycling_two_cycles["pair_representation"])


def test_simplicial_boundary_cochain_recycling_changes_only_recycled_cycles():
    torch.manual_seed(10)
    base_config = replace(load_model_config("tiny"), simplex_boundary_cochain_recycling_scale=0.0)
    recycling_config = replace(load_model_config("tiny"), simplex_boundary_cochain_recycling_scale=0.5)
    base_model = AlphaFold2(base_config)
    recycling_model = AlphaFold2(recycling_config)
    recycling_model.load_state_dict(base_model.state_dict())
    base_model.eval()
    recycling_model.eval()

    target_feat = torch.zeros(1, 5, 22)
    residue_index = torch.arange(5).reshape(1, 5)
    msa_feat = torch.zeros(1, 2, 5, 49)
    extra_msa_feat = torch.zeros(1, 0, 5, 25)
    template_pair_feat = torch.zeros(1, 0, 5, 5, 88)
    aatype = torch.zeros(1, 5, dtype=torch.long)

    with torch.no_grad():
        base_one_cycle = base_model(
            target_feat,
            residue_index,
            msa_feat,
            extra_msa_feat,
            template_pair_feat,
            aatype,
            n_cycles=1,
        )
        recycling_one_cycle = recycling_model(
            target_feat,
            residue_index,
            msa_feat,
            extra_msa_feat,
            template_pair_feat,
            aatype,
            n_cycles=1,
        )
        base_two_cycles = base_model(
            target_feat,
            residue_index,
            msa_feat,
            extra_msa_feat,
            template_pair_feat,
            aatype,
            n_cycles=2,
        )
        recycling_two_cycles = recycling_model(
            target_feat,
            residue_index,
            msa_feat,
            extra_msa_feat,
            template_pair_feat,
            aatype,
            n_cycles=2,
        )

    assert "simplex_structure_pair_readout" not in recycling_one_cycle
    assert torch.allclose(base_one_cycle["pair_representation"], recycling_one_cycle["pair_representation"])
    assert not torch.allclose(base_two_cycles["pair_representation"], recycling_two_cycles["pair_representation"])


def test_metric_gated_boundary_cochain_recycling_suppresses_uncertain_recycled_cochains():
    torch.manual_seed(11)
    base_config = replace(load_model_config("tiny"), simplex_boundary_cochain_recycling_scale=0.0)
    recycling_config = replace(load_model_config("tiny"), simplex_boundary_cochain_recycling_scale=0.5)
    gated_config = replace(
        load_model_config("tiny"),
        simplex_boundary_cochain_recycling_scale=0.5,
        simplex_boundary_cochain_recycling_metric_gate_scale=1.0,
    )
    base_model = AlphaFold2(base_config)
    recycling_model = AlphaFold2(recycling_config)
    gated_model = AlphaFold2(gated_config)
    recycling_model.load_state_dict(base_model.state_dict())
    gated_model.load_state_dict(base_model.state_dict())
    base_model.eval()
    recycling_model.eval()
    gated_model.eval()

    target_feat = torch.zeros(1, 5, 22)
    residue_index = torch.arange(5).reshape(1, 5)
    msa_feat = torch.zeros(1, 2, 5, 49)
    extra_msa_feat = torch.zeros(1, 0, 5, 25)
    template_pair_feat = torch.zeros(1, 0, 5, 5, 88)
    aatype = torch.zeros(1, 5, dtype=torch.long)

    with torch.no_grad():
        base_two_cycles = base_model(
            target_feat,
            residue_index,
            msa_feat,
            extra_msa_feat,
            template_pair_feat,
            aatype,
            n_cycles=2,
        )
        recycling_two_cycles = recycling_model(
            target_feat,
            residue_index,
            msa_feat,
            extra_msa_feat,
            template_pair_feat,
            aatype,
            n_cycles=2,
        )
        gated_two_cycles = gated_model(
            target_feat,
            residue_index,
            msa_feat,
            extra_msa_feat,
            template_pair_feat,
            aatype,
            n_cycles=2,
        )

    assert not torch.allclose(base_two_cycles["pair_representation"], recycling_two_cycles["pair_representation"])
    assert torch.allclose(base_two_cycles["pair_representation"], gated_two_cycles["pair_representation"])


def test_simplexfold_medium_width_matched_preserves_widths_near_af2_medium_budget():
    medium = load_model_config("medium")
    af2_medium = replace(medium, use_simplicial_evoformer=False)
    simplex_medium = load_model_config("simplexfold_medium_width_matched")

    af2_params = sum(parameter.numel() for parameter in AlphaFold2(af2_medium).parameters())
    simplex_params = sum(parameter.numel() for parameter in AlphaFold2(simplex_medium).parameters())

    assert af2_params == 3_106_642
    assert simplex_params == 3_348_544
    assert simplex_medium.c_m == medium.c_m
    assert simplex_medium.c_s == medium.c_s
    assert simplex_medium.c_z == medium.c_z
    assert (simplex_params - af2_params) / af2_params < 0.08


def test_zero_dropout_model_config_preserves_dimensions_and_clears_dropout():
    config = load_model_config("alphafold2")
    overfit_config = zero_dropout_model_config(config)

    assert overfit_config.model_profile == "alphafold2_no_dropout"
    assert overfit_config.c_m == config.c_m
    assert overfit_config.c_s == config.c_s
    assert overfit_config.c_z == config.c_z
    assert overfit_config.num_evoformer == config.num_evoformer
    assert overfit_config.template_pair_dropout == 0.0
    assert overfit_config.extra_msa_dropout == 0.0
    assert overfit_config.extra_pair_dropout == 0.0
    assert overfit_config.evoformer_msa_dropout == 0.0
    assert overfit_config.evoformer_pair_dropout == 0.0
    assert overfit_config.simplex_dropout == 0.0
    assert overfit_config.structure_module_dropout_ipa == 0.0
    assert overfit_config.structure_module_dropout_transition == 0.0


def test_learning_rate_for_step_supports_warmup_cosine():
    training_config = TrainingConfig(
        learning_rate=1e-3,
        min_learning_rate=1e-4,
        lr_schedule="warmup_cosine",
        warmup_steps=10,
    )

    warmup_lr = learning_rate_for_step(training_config, step=4, total_steps=100)
    after_warmup_lr = learning_rate_for_step(training_config, step=10, total_steps=100)
    late_lr = learning_rate_for_step(training_config, step=99, total_steps=100)

    assert abs(warmup_lr - 5e-4) < 1e-8
    assert after_warmup_lr < training_config.learning_rate
    assert late_lr >= training_config.min_learning_rate
    assert late_lr < after_warmup_lr


def test_build_optimizer_uses_configured_adam_hyperparameters():
    model = AlphaFold2(load_model_config("tiny"))
    training_config = TrainingConfig(
        learning_rate=2e-4,
        adam_beta1=0.8,
        adam_beta2=0.95,
        adam_eps=1e-6,
        weight_decay=1e-3,
    )

    optimizer = build_optimizer(model, training_config)
    group = optimizer.param_groups[0]

    assert abs(group["lr"] - 2e-4) < 1e-12
    assert group["betas"] == (0.8, 0.95)
    assert abs(group["eps"] - 1e-6) < 1e-12
    assert abs(group["weight_decay"] - 1e-3) < 1e-12


def test_learning_rate_at_step_applies_finetune_scale_without_warmup():
    """Supplement 1.11.3: fine-tuning has no warmup, half base LR."""
    training_config = TrainingConfig(
        learning_rate=1e-3,
        lr_schedule="warmup_cosine",
        warmup_steps=10,
        finetune_lr_scale=0.5,
    )

    # Step 0 during pre-training is inside the linear warmup → tiny LR.
    pretrain_lr = learning_rate_at_step(training_config, step=0, total_steps=100, is_finetune=False)
    assert pretrain_lr < training_config.learning_rate

    # Same step during fine-tuning ignores warmup and returns lr * 0.5.
    finetune_lr = learning_rate_at_step(training_config, step=0, total_steps=100, is_finetune=True)
    assert abs(finetune_lr - 5e-4) < 1e-12


def test_training_config_defaults_match_supplement_1_11_3():
    """Supplement 1.11.3 fixes base lr=1e-3, ε=1e-6, clip=0.1, halving at fine-tune."""
    cfg = TrainingConfig()
    assert cfg.learning_rate == 1e-3
    assert cfg.adam_beta1 == 0.9
    assert cfg.adam_beta2 == 0.999
    assert cfg.adam_eps == 1e-6
    assert cfg.grad_clip_norm == 0.1
    assert cfg.finetune_lr_scale == 0.5


def test_use_finetune_loss_supports_two_phase_schedule():
    always_pretrain = TrainingConfig(finetune=False, finetune_start_step=None)
    always_finetune = TrainingConfig(finetune=True, finetune_start_step=None)
    scheduled = TrainingConfig(finetune=False, finetune_start_step=5)

    assert use_finetune_loss(always_pretrain, global_step=0) is False
    assert use_finetune_loss(always_finetune, global_step=0) is True
    assert use_finetune_loss(scheduled, global_step=4) is False
    assert use_finetune_loss(scheduled, global_step=5) is True


def test_apply_loss_weight_schedule_ramps_research_weights():
    cfg = TrainingConfig(
        msa_loss_weight=2.0,
        distogram_loss_weight=0.3,
        simplex_aux_weight=1.0,
        simplex_cell_closure_weight=0.0,
        simplex_cell_closure_weight_final=0.5,
        simplex_cell_closure_ramp_start_step=10,
        simplex_cell_closure_ramp_steps=10,
        simplex_face_boundary_lddt_weight=0.05,
        simplex_face_boundary_lddt_weight_final=0.025,
        simplex_tetra_boundary_lddt_weight=0.04,
        simplex_tetra_boundary_lddt_weight_final=0.02,
        simplex_boundary_lddt_ramp_start_step=10,
        simplex_boundary_lddt_ramp_steps=10,
        backbone_loss_weight=1.0,
        sidechain_fape_loss_weight=1.0,
        torsion_loss_weight=1.0,
        loss_weight_ramp_start_step=10,
        loss_weight_ramp_steps=10,
        msa_loss_weight_final=0.5,
        distogram_loss_weight_final=0.1,
        simplex_aux_weight_final=0.0,
        backbone_loss_weight_final=6.0,
        sidechain_fape_loss_weight_final=2.0,
        torsion_loss_weight_final=0.25,
    )
    loss_fn = AlphaFoldLoss()

    apply_loss_weight_schedule(loss_fn, cfg, step=5)
    assert loss_fn.msa_weight == pytest.approx(2.0)
    assert loss_fn.backbone_loss_weight == pytest.approx(1.0)
    assert loss_fn.simplex_geometry_loss.cell_closure_weight == pytest.approx(0.0)
    assert loss_fn.simplex_geometry_loss.face_boundary_lddt_weight == pytest.approx(0.05)
    assert loss_fn.simplex_geometry_loss.tetra_boundary_lddt_weight == pytest.approx(0.04)

    apply_loss_weight_schedule(loss_fn, cfg, step=15)
    assert loss_fn.msa_weight == pytest.approx(1.25)
    assert loss_fn.distogram_weight == pytest.approx(0.2)
    assert loss_fn.simplex_aux_weight == pytest.approx(0.5)
    assert loss_fn.simplex_geometry_loss.cell_closure_weight == pytest.approx(0.25)
    assert loss_fn.simplex_geometry_loss.face_boundary_lddt_weight == pytest.approx(0.0375)
    assert loss_fn.simplex_geometry_loss.tetra_boundary_lddt_weight == pytest.approx(0.03)
    assert loss_fn.backbone_loss_weight == pytest.approx(3.5)
    assert loss_fn.sidechain_fape_loss_weight == pytest.approx(1.5)
    assert loss_fn.torsion_loss_weight == pytest.approx(0.625)

    apply_loss_weight_schedule(loss_fn, cfg, step=20)
    assert loss_fn.msa_weight == pytest.approx(0.5)
    assert loss_fn.simplex_geometry_loss.cell_closure_weight == pytest.approx(0.5)
    assert loss_fn.simplex_geometry_loss.face_boundary_lddt_weight == pytest.approx(0.025)
    assert loss_fn.simplex_geometry_loss.tetra_boundary_lddt_weight == pytest.approx(0.02)
    assert loss_fn.backbone_loss_weight == pytest.approx(6.0)


def test_alphafold_loss_overrides_simplex_coordinate_weights():
    loss_fn = AlphaFoldLoss(
        simplex_face_coordinate_weight=0.4,
        simplex_face_coordinate_distance_weight=0.45,
        simplex_face_coordinate_expansion_weight=0.46,
        simplex_face_shape_weight=0.48,
        simplex_face_normal_weight=0.5,
        simplex_face_boundary_lddt_weight=0.55,
        simplex_tetra_coordinate_weight=0.6,
        simplex_tetra_coordinate_distance_weight=0.65,
        simplex_tetra_coordinate_expansion_weight=0.66,
        simplex_tetra_shape_weight=0.68,
        simplex_tetra_boundary_lddt_weight=0.7,
        simplex_topology_margin_weight=0.8,
        simplex_topology_margin=1.5,
        simplex_topology_margin_hard_negatives=3,
        simplex_boundary_degree_normalize=True,
        simplex_cell_closure_weight=0.25,
        simplex_cell_closure_cutoff=12.0,
        simplex_cell_closure_temperature=1.5,
        simplex_coordinate_expansion_tolerance=0.05,
    )

    assert loss_fn.simplex_geometry_loss.face_coordinate_weight == pytest.approx(0.4)
    assert loss_fn.simplex_geometry_loss.face_coordinate_distance_weight == pytest.approx(0.45)
    assert loss_fn.simplex_geometry_loss.face_coordinate_expansion_weight == pytest.approx(0.46)
    assert loss_fn.simplex_geometry_loss.face_shape_weight == pytest.approx(0.48)
    assert loss_fn.simplex_geometry_loss.face_normal_weight == pytest.approx(0.5)
    assert loss_fn.simplex_geometry_loss.face_boundary_lddt_weight == pytest.approx(0.55)
    assert loss_fn.simplex_geometry_loss.tetra_coordinate_weight == pytest.approx(0.6)
    assert loss_fn.simplex_geometry_loss.tetra_coordinate_distance_weight == pytest.approx(0.65)
    assert loss_fn.simplex_geometry_loss.tetra_coordinate_expansion_weight == pytest.approx(0.66)
    assert loss_fn.simplex_geometry_loss.tetra_shape_weight == pytest.approx(0.68)
    assert loss_fn.simplex_geometry_loss.tetra_boundary_lddt_weight == pytest.approx(0.7)
    assert loss_fn.simplex_geometry_loss.topology_margin_weight == pytest.approx(0.8)
    assert loss_fn.simplex_geometry_loss.topology_margin == pytest.approx(1.5)
    assert loss_fn.simplex_geometry_loss.topology_margin_hard_negatives == 3
    assert loss_fn.simplex_geometry_loss.boundary_degree_normalize is True
    assert loss_fn.simplex_geometry_loss.cell_closure_weight == pytest.approx(0.25)
    assert loss_fn.simplex_geometry_loss.cell_closure_cutoff == pytest.approx(12.0)
    assert loss_fn.simplex_geometry_loss.cell_closure_temperature == pytest.approx(1.5)
    assert loss_fn.simplex_geometry_loss.coordinate_expansion_tolerance == pytest.approx(0.05)


def test_build_dataloader_can_fix_training_features(tmp_path):
    feature_dir, label_dir = make_processed_cache_dirs(tmp_path)
    data_config = DataConfig(
        processed_features_dir=feature_dir,
        processed_labels_dir=label_dir,
        val_fraction=0.0,
        crop_size=8,
        msa_depth=3,
        extra_msa_depth=2,
        max_templates=1,
        block_delete_training_msa=False,
        fixed_feature_seed=11,
    )
    loader = build_dataloader(
        "all",
        data_config,
        training=True,
        batch_size=1,
        num_workers=0,
        device="cpu",
        seed=0,
    )

    first = next(iter(loader))
    second = next(iter(loader))

    assert torch.equal(first["msa_feat"], second["msa_feat"])
    assert torch.equal(first["masked_msa_mask"], second["masked_msa_mask"])


def test_build_dataloader_can_emit_recycling_feature_samples(tmp_path):
    feature_dir, label_dir = make_processed_cache_dirs(tmp_path)
    data_config = DataConfig(
        processed_features_dir=feature_dir,
        processed_labels_dir=label_dir,
        val_fraction=0.0,
        crop_size=8,
        msa_depth=3,
        extra_msa_depth=2,
        max_templates=1,
        fixed_feature_seed=13,
    )
    loader = build_dataloader(
        "all",
        data_config,
        training=True,
        batch_size=1,
        num_workers=0,
        device="cpu",
        seed=0,
        n_cycles=2,
        n_ensemble=1,
    )

    batch = next(iter(loader))

    assert batch["msa_feat"].shape[:3] == (2, 1, 1)
    assert batch["extra_msa_feat"].shape[:3] == (2, 1, 1)


# ---------------------------------------------------------------------
# Two-stage training protocol — supplement Table 4 + §1.11.
# ---------------------------------------------------------------------


def test_load_training_protocol_alphafold2_matches_table4():
    """The shipped TOML must reproduce Supplementary Table 4 verbatim."""
    protocol = load_training_protocol("alphafold2")

    assert protocol.protocol == "alphafold2"

    # Shared optimizer (§1.11.3 / §1.11.7).
    assert protocol.optimizer.adam_beta1 == 0.9
    assert protocol.optimizer.adam_beta2 == 0.999
    assert protocol.optimizer.adam_eps == 1e-6
    assert protocol.optimizer.grad_clip_norm == 0.1
    assert protocol.optimizer.ema_decay == 0.999
    assert protocol.optimizer.mini_batch_size == 128
    assert protocol.optimizer.lr_decay_samples == 6_400_000
    assert protocol.optimizer.lr_decay_factor == 0.95

    # Initial stage (Table 4, "Initial training" column).
    assert protocol.initial.crop_size == 256
    assert protocol.initial.msa_depth == 128
    assert protocol.initial.extra_msa_depth == 1024
    assert protocol.initial.max_templates == 4
    assert protocol.initial.learning_rate == 1e-3
    assert protocol.initial.warmup_samples == 128_000
    assert protocol.initial.violation_loss_weight == 0.0
    assert protocol.initial.total_samples == 10_000_000

    # Fine-tuning stage (Table 4, "Fine-tuning" column).
    assert protocol.finetune.crop_size == 384
    assert protocol.finetune.msa_depth == 512
    assert protocol.finetune.extra_msa_depth == 5120
    assert protocol.finetune.max_templates == 4
    assert protocol.finetune.learning_rate == 5e-4
    assert protocol.finetune.warmup_samples == 0
    assert protocol.finetune.violation_loss_weight == 1.0
    assert protocol.finetune.total_samples == 1_500_000


def test_load_training_protocol_unknown_name_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_training_protocol("definitely_not_a_real_protocol")


def test_load_training_protocol_rejects_unknown_stage_key(tmp_path):
    """Typos in a TOML field must surface at load time, not during training."""
    path = tmp_path / "training_bad.toml"
    path.write_text(
        'protocol = "bad"\n'
        "[optimizer]\n"
        "adam_beta1 = 0.9\nadam_beta2 = 0.999\nadam_eps = 1e-6\n"
        "grad_clip_norm = 0.1\nema_decay = 0.999\nmini_batch_size = 128\n"
        "lr_decay_samples = 6400000\nlr_decay_factor = 0.95\n"
        "[initial]\n"
        "crop_size = 256\nmsa_depth = 128\nextra_msa_depth = 1024\n"
        "max_templates = 4\nlearning_rate = 1e-3\nwarmup_samples = 0\n"
        "violation_loss_weight = 0.0\ntotal_samples = 1000\n"
        'not_a_real_key = "oops"\n'
        "[finetune]\n"
        "crop_size = 384\nmsa_depth = 512\nextra_msa_depth = 5120\n"
        "max_templates = 4\nlearning_rate = 5e-4\nwarmup_samples = 0\n"
        "violation_loss_weight = 1.0\ntotal_samples = 1000\n"
    )
    with pytest.raises(TypeError):
        load_training_protocol(path)


def test_training_protocol_stage_returns_matching_stage():
    protocol = load_training_protocol("alphafold2")
    assert protocol.stage("initial") is protocol.initial
    assert protocol.stage("finetune") is protocol.finetune


def test_training_protocol_stage_rejects_unknown_name():
    protocol = load_training_protocol("alphafold2")
    with pytest.raises(ValueError):
        protocol.stage("pretrain")


def test_list_available_training_protocols_excludes_model_profiles():
    protocols = list_available_training_protocols()
    assert "alphafold2" in protocols
    # Model profiles (tiny/medium/alphafold2 without the ``training_``
    # prefix) must not leak into the training-protocol list.
    assert all(
        not protocol.startswith("training_") for protocol in protocols
    )


def test_list_available_profiles_excludes_training_protocols():
    profiles = list_available_profiles()
    assert all(not p.startswith("training_") for p in profiles)
    # Sanity: the canonical model profiles should still show up.
    assert "alphafold2" in profiles


# ---------------------------------------------------------------------
# Paper samples-based LR schedule — supplement §1.11.3.
# ---------------------------------------------------------------------


def test_learning_rate_for_samples_linear_warmup_range():
    """Linear warmup: LR(0) = 0, LR(warmup/2) = base/2, LR(warmup) = base."""
    base = 1e-3
    warmup = 1000
    assert learning_rate_for_samples(base, 0, warmup, None, 1.0) == 0.0
    assert learning_rate_for_samples(base, warmup // 2, warmup, None, 1.0) == base / 2
    # At exactly warmup_samples we exit the warmup branch — the constant
    # phase returns the full base LR.
    assert learning_rate_for_samples(base, warmup, warmup, None, 1.0) == base


def test_learning_rate_for_samples_constant_phase():
    """Between warmup end and decay trigger, LR stays at base."""
    base = 1e-3
    assert learning_rate_for_samples(base, 2_000_000, 128_000, 6_400_000, 0.95) == base
    # Without any decay configured, same story indefinitely.
    assert learning_rate_for_samples(base, 10_000_000, 128_000, None, 1.0) == base


def test_learning_rate_for_samples_one_shot_decay_at_threshold():
    """At and past lr_decay_samples the LR drops by lr_decay_factor once."""
    base = 1e-3
    pre = learning_rate_for_samples(base, 6_400_000 - 1, 128_000, 6_400_000, 0.95)
    post = learning_rate_for_samples(base, 6_400_000, 128_000, 6_400_000, 0.95)
    later = learning_rate_for_samples(base, 10_000_000, 128_000, 6_400_000, 0.95)
    assert pre == base
    assert post == pytest.approx(base * 0.95)
    assert later == pytest.approx(base * 0.95)


def test_learning_rate_at_step_uses_samples_when_configured():
    """Setting warmup_samples > 0 switches to the samples-based path."""
    cfg = TrainingConfig(learning_rate=1e-3, warmup_samples=1000)
    assert learning_rate_at_step(cfg, step=0, total_steps=100, is_finetune=False, samples_seen=0) == 0.0
    assert learning_rate_at_step(cfg, step=0, total_steps=100, is_finetune=False, samples_seen=500) == 5e-4


def test_learning_rate_at_step_falls_back_to_step_schedule_by_default():
    """With no samples knobs set we keep the old step-based ``lr_schedule``."""
    cfg = TrainingConfig(learning_rate=1e-3)  # default: constant, no warmup
    lr = learning_rate_at_step(
        cfg, step=5, total_steps=10, is_finetune=False, samples_seen=999_999,
    )
    assert lr == 1e-3


def test_learning_rate_at_step_finetune_applies_decay_factor_past_threshold():
    """Fine-tuning LR also respects the one-shot ×0.95 drop."""
    cfg = TrainingConfig(
        learning_rate=1e-3,
        finetune_lr_scale=0.5,
        lr_decay_samples=6_400_000,
        lr_decay_factor=0.95,
    )
    pre = learning_rate_at_step(
        cfg, step=0, total_steps=1, is_finetune=True, samples_seen=1_000_000,
    )
    post = learning_rate_at_step(
        cfg, step=0, total_steps=1, is_finetune=True, samples_seen=7_000_000,
    )
    assert pre == pytest.approx(1e-3 * 0.5)
    assert post == pytest.approx(1e-3 * 0.5 * 0.95)


# ---------------------------------------------------------------------
# EMA — supplement §1.11.7.
# ---------------------------------------------------------------------


def test_build_ema_model_first_update_copies_current_params():
    """At num_averaged=0 the EMA should mirror the current model exactly."""
    linear = torch.nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.ones(2, 3))
    ema = build_ema_model(linear, ema_decay=0.9)
    # Training "step": change the live params before updating the EMA.
    with torch.no_grad():
        linear.weight.copy_(torch.zeros(2, 3))
    ema.update_parameters(linear)
    ema_weight = _linear(ema.module).weight
    assert torch.allclose(ema_weight, torch.zeros(2, 3))


def test_build_ema_model_subsequent_updates_apply_decay():
    """After the first sample the EMA blends decay·avg + (1−decay)·current."""
    linear = torch.nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.ones(2, 3))
    ema = build_ema_model(linear, ema_decay=0.9)
    ema.update_parameters(linear)  # first call → EMA = ones
    with torch.no_grad():
        linear.weight.copy_(torch.zeros(2, 3))
    ema.update_parameters(linear)  # second call → 0.9*ones + 0.1*zeros = 0.9
    ema_weight = _linear(ema.module).weight
    assert torch.allclose(ema_weight, torch.full((2, 3), 0.9))


# ---------------------------------------------------------------------
# Gradient accumulation + EMA + resume inside fit().
# ---------------------------------------------------------------------


def test_fit_with_grad_accumulation_performs_fewer_optimizer_steps(tmp_path):
    """With grad_accum_steps=2 and 2 training examples, fit should do 1 step.

    Two micro-batches accumulate into one optimizer.step, so global_step
    ends at 1. Without accumulation the same run would do 2 steps.
    """
    feature_dir, label_dir = make_processed_cache_dirs(tmp_path)
    _, history = fit(
        model_config=zero_dropout_model_config(load_model_config("tiny")),
        data_config=DataConfig(
            processed_features_dir=feature_dir,
            processed_labels_dir=label_dir,
            val_fraction=0.0,
            crop_size=5,
            msa_depth=2,
            extra_msa_depth=2,
            max_templates=1,
        ),
        training_config=TrainingConfig(
            epochs=1,
            batch_size=1,
            grad_accum_steps=2,
            device="cpu",
            seed=0,
            n_cycles=1,
            n_ensemble=1,
        ),
    )
    assert len(history) == 1
    # Two micro-batches → one optimizer.step.
    assert history[0]["global_step"] == 1
    assert history[0]["global_samples"] == 2


def test_fit_with_ema_writes_ema_state_in_checkpoint(tmp_path):
    """Enabling ema_decay should persist an ``ema_state_dict`` on disk."""
    feature_dir, label_dir = make_processed_cache_dirs(tmp_path)
    latest_path = tmp_path / "ema.pt"
    fit(
        model_config=zero_dropout_model_config(load_model_config("tiny")),
        data_config=DataConfig(
            processed_features_dir=feature_dir,
            processed_labels_dir=label_dir,
            val_fraction=0.0,
            crop_size=5,
            msa_depth=2,
            extra_msa_depth=2,
            max_templates=1,
        ),
        training_config=TrainingConfig(
            epochs=1,
            batch_size=1,
            ema_decay=0.9,
            device="cpu",
            seed=0,
            n_cycles=1,
            n_ensemble=1,
            latest_checkpoint_path=latest_path,
        ),
    )
    payload = torch.load(latest_path, weights_only=False, map_location="cpu")
    assert "ema_state_dict" in payload
    assert payload["global_step"] >= 1
    assert payload["global_samples"] >= 1


def test_fit_resumes_global_step_and_samples(tmp_path):
    """A resumed run should continue the counters from where it left off."""
    feature_dir, label_dir = make_processed_cache_dirs(tmp_path)
    checkpoint_path = tmp_path / "ck.pt"

    data_config = DataConfig(
        processed_features_dir=feature_dir,
        processed_labels_dir=label_dir,
        val_fraction=0.0,
        crop_size=5,
        msa_depth=2,
        extra_msa_depth=2,
        max_templates=1,
    )
    common_training = dict(
        batch_size=1,
        grad_accum_steps=1,
        device="cpu",
        seed=0,
        n_cycles=1,
        n_ensemble=1,
    )

    _, history_first = fit(
        model_config=zero_dropout_model_config(load_model_config("tiny")),
        data_config=data_config,
        training_config=TrainingConfig(
            epochs=1,
            latest_checkpoint_path=checkpoint_path,
            **common_training,
        ),
    )
    first_samples = int(history_first[0]["global_samples"])
    first_step = int(history_first[0]["global_step"])
    assert first_samples > 0

    _, history_resumed = fit(
        model_config=zero_dropout_model_config(load_model_config("tiny")),
        data_config=data_config,
        training_config=TrainingConfig(
            epochs=2,
            resume_from_checkpoint=checkpoint_path,
            **common_training,
        ),
    )
    # ``fit`` preserves the checkpoint's history list and appends new
    # epochs onto the end, so the resumed run's final entry is epoch 2.
    assert len(history_resumed) == 2
    assert history_resumed[-1]["epoch"] == 2
    assert int(history_resumed[-1]["global_samples"]) > first_samples
    assert int(history_resumed[-1]["global_step"]) > first_step


from train_af2 import (  # noqa: E402 — sys.path fix happens in conftest
    _epochs_for_target_samples,
    data_config_for_stage,
    training_config_for_stage,
)


# ---------------------------------------------------------------------
# scripts/train_af2.py — protocol → (DataConfig, TrainingConfig) plumbing.
# ---------------------------------------------------------------------


def test_epochs_for_target_samples_rounds_up():
    """Enough epochs to cover the sample budget, always rounded up."""
    # Exactly divisible → exact number of epochs.
    assert _epochs_for_target_samples(1000, 100) == 10
    # Not divisible → ceiling.
    assert _epochs_for_target_samples(1001, 100) == 11
    # Smaller target than one epoch → one epoch (never zero).
    assert _epochs_for_target_samples(50, 100) == 1
    # Empty / missing dataset → one epoch (avoid div-by-zero).
    assert _epochs_for_target_samples(1000, 0) == 1


def test_data_config_for_stage_mirrors_table4_columns():
    """Initial + fine-tune stages produce DataConfigs with Table 4 numbers."""
    protocol = load_training_protocol("alphafold2")
    initial_dc = data_config_for_stage(
        protocol.initial,
        processed_features_dir=Path("/f"),
        processed_labels_dir=Path("/l"),
        val_fraction=0.0,
    )
    assert initial_dc.crop_size == 256
    assert initial_dc.msa_depth == 128
    assert initial_dc.extra_msa_depth == 1024
    assert initial_dc.max_templates == 4

    finetune_dc = data_config_for_stage(
        protocol.finetune,
        processed_features_dir=Path("/f"),
        processed_labels_dir=Path("/l"),
        val_fraction=0.0,
    )
    assert finetune_dc.crop_size == 384
    assert finetune_dc.msa_depth == 512
    assert finetune_dc.extra_msa_depth == 5120
    assert finetune_dc.max_templates == 4


def test_training_config_for_stage_uses_stage_lr_without_halving():
    """Fine-tune stage LR comes straight from Table 4 (5e-4), not halved again.

    The trainer's ``finetune_lr_scale`` would normally halve the base LR
    during fine-tuning. ``training_config_for_stage`` sets that scale to
    1.0 because the stage LR field is already the post-halving 5e-4;
    applying 0.5 again would produce 2.5e-4, which isn't the paper value.
    """
    protocol = load_training_protocol("alphafold2")
    cfg = training_config_for_stage(
        protocol, protocol.finetune,
        device="cpu",
        seed=0,
        batch_size=1,
        grad_accum_steps=1,
        num_workers=0,
        n_cycles=1,
        n_ensemble=1,
        epochs=1,
        is_finetune=True,
        latest_checkpoint_path=Path("/tmp/latest.pt"),
        best_checkpoint_path=None,
        resume_from_checkpoint=None,
        init_weights_from_checkpoint=None,
    )
    assert cfg.learning_rate == 5e-4
    assert cfg.finetune_lr_scale == 1.0
    assert cfg.finetune is True


def test_training_config_for_stage_forwards_optimizer_and_ema():
    """Shared optimizer settings (§1.11.3, §1.11.7) flow through unchanged."""
    protocol = load_training_protocol("alphafold2")
    cfg = training_config_for_stage(
        protocol, protocol.initial,
        device="cpu",
        seed=42,
        batch_size=1,
        grad_accum_steps=128,
        num_workers=0,
        n_cycles=4,
        n_ensemble=1,
        epochs=10,
        is_finetune=False,
        latest_checkpoint_path=Path("/tmp/latest.pt"),
        best_checkpoint_path=None,
        resume_from_checkpoint=None,
        init_weights_from_checkpoint=None,
    )
    assert cfg.adam_beta1 == 0.9
    assert cfg.adam_beta2 == 0.999
    assert cfg.adam_eps == 1e-6
    assert cfg.grad_clip_norm == 0.1
    assert cfg.ema_decay == 0.999
    assert cfg.warmup_samples == 128_000
    assert cfg.lr_decay_samples == 6_400_000
    assert cfg.lr_decay_factor == 0.95
    assert cfg.grad_accum_steps == 128


def test_load_checkpoint_for_resume_restores_counters(tmp_path):
    """Direct-call sanity: the loader returns the saved step/sample state."""
    feature_dir, label_dir = make_processed_cache_dirs(tmp_path)
    model_config = zero_dropout_model_config(load_model_config("tiny"))
    model = AlphaFold2(model_config)
    optimizer = build_optimizer(
        model,
        TrainingConfig(batch_size=1, device="cpu"),
    )

    path = tmp_path / "direct.pt"
    save_checkpoint(
        path,
        epoch=3,
        global_step=42,
        global_samples=42 * 128,
        model=model,
        optimizer=optimizer,
        best_val_loss=None,
        history=[{"epoch": 3, "global_step": 42}],
        data_config=DataConfig(
            processed_features_dir=feature_dir,
            processed_labels_dir=label_dir,
        ),
        training_config=TrainingConfig(batch_size=1, device="cpu"),
        model_config=model_config,
    )
    fresh_model = AlphaFold2(model_config)
    fresh_optimizer = build_optimizer(
        fresh_model,
        TrainingConfig(batch_size=1, device="cpu"),
    )
    restored = load_checkpoint_for_resume(path, fresh_model, fresh_optimizer, None)
    assert restored["epoch"] == 4
    assert restored["global_step"] == 42
    assert restored["global_samples"] == 42 * 128
