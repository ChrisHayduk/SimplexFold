from minalphafold.trainer import load_model_config
from scripts.run_nanofold_public_benchmarks import _variant_config


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
