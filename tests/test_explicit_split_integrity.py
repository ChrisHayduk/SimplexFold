import hashlib
import json

import pytest

from minalphafold.data import ProcessedOpenProteinSetDataset, validate_split_manifests
from tests.test_data_pipeline import write_processed_cache


def test_explicit_roles_preserve_selection_order_and_reject_missing_pairs(tmp_path):
    features = tmp_path / "features"
    labels = tmp_path / "labels"
    features.mkdir()
    labels.mkdir()
    for chain in ("a_A", "b_A", "c_A"):
        write_processed_cache(features, labels, chain, "AGAGA", include_templates=False)
    role = tmp_path / "train.txt"
    role.write_text("c_A\na_A\n")
    dataset = ProcessedOpenProteinSetDataset(
        features, labels, split="all", split_manifest=role
    )
    assert dataset.chain_ids == ["c_A", "a_A"]
    assert dataset[0]["chain_id"] == "c_A"
    with pytest.raises(ValueError, match="do not split twice"):
        ProcessedOpenProteinSetDataset(features, labels, split_manifest=role)
    role.write_text("c_A\nmissing_A\n")
    with pytest.raises(ValueError, match="lacks its feature/label pair"):
        ProcessedOpenProteinSetDataset(
            features, labels, split="all", split_manifest=role
        )


def test_split_validation_rejects_shared_groups_duplicates_and_partial_annotations(
    tmp_path,
):
    train = tmp_path / "train.json"
    val = tmp_path / "val.json"
    train.write_text(
        json.dumps({"chains": [{"chain_id": "a_A", "full_group_id": "g1"}]})
    )
    val.write_text(json.dumps({"chains": [{"chain_id": "b_A", "full_group_id": "g2"}]}))
    validate_split_manifests(train, val)
    val.write_text(json.dumps({"chains": [{"chain_id": "b_A", "full_group_id": "g1"}]}))
    with pytest.raises(ValueError, match="share full_group_id"):
        validate_split_manifests(train, val)
    val.write_text("b_A\n")
    with pytest.raises(ValueError, match="must have full_group_id"):
        validate_split_manifests(train, val)
    val.write_text("b_A\nb_A\n")
    with pytest.raises(ValueError, match="duplicate"):
        validate_split_manifests(train, val)


def test_filter_receipt_hash_detects_mutated_cache(tmp_path):
    features = tmp_path / "features"
    labels = tmp_path / "labels"
    features.mkdir()
    labels.mkdir()
    write_processed_cache(features, labels, "a_A", "AGAGA", include_templates=False)
    entry = {"chain_id": "a_A", "accepted": True}
    for field, directory in (("features_sha256", features), ("labels_sha256", labels)):
        entry[field] = hashlib.sha256((directory / "a_A.npz").read_bytes()).hexdigest()
    manifest = tmp_path / "filter.json"
    manifest.write_text(json.dumps({"chains": [entry]}))
    dataset = ProcessedOpenProteinSetDataset(
        features, labels, split="all", chains_manifest=manifest
    )
    dataset[0]
    with (features / "a_A.npz").open("ab") as handle:
        handle.write(b"different generation")
    with pytest.raises(ValueError, match="Stale manifest features_sha256"):
        dataset[0]


def test_role_cannot_be_silently_shrunk_by_filter(tmp_path):
    features, labels = tmp_path / "features", tmp_path / "labels"
    features.mkdir()
    labels.mkdir()
    for chain in ("a_A", "b_A"):
        write_processed_cache(features, labels, chain, "AGAGA", include_templates=False)
    role, filtered = tmp_path / "role.txt", tmp_path / "filter.txt"
    role.write_text("a_A\nb_A\n")
    filtered.write_text("a_A\n")
    with pytest.raises(ValueError, match="absent from the accepted"):
        ProcessedOpenProteinSetDataset(
            features, labels, split="all", manifest_path=role, chains_manifest=filtered
        )


def test_cache_identity_rejects_label_index_disagreement_and_duplicate_names(tmp_path):
    import numpy as np

    from minalphafold.data import chain_npz_path

    features, labels = tmp_path / "features", tmp_path / "labels"
    features.mkdir()
    labels.mkdir()
    write_processed_cache(features, labels, "a_A", "AGAGA", include_templates=False)
    path = labels / "a_A.npz"
    with np.load(path) as payload:
        arrays = dict(payload)
    arrays["residue_index"] = np.arange(5) + 10
    np.savez(path, **arrays)
    dataset = ProcessedOpenProteinSetDataset(features, labels, split="all")
    with pytest.raises(ValueError, match="residue indices disagree"):
        dataset[0]
    chain_npz_path(features, "a_A").write_bytes((features / "a_A.npz").read_bytes())
    with pytest.raises(ValueError, match="Ambiguous"):
        ProcessedOpenProteinSetDataset(features, labels, split="all")


def test_explicit_role_loader_cannot_widen_to_all(tmp_path):
    from minalphafold.trainer import DataConfig, build_dataloader
    config = DataConfig(train_manifest=tmp_path / "train.txt", val_manifest=tmp_path / "val.txt", val_fraction=0)
    with pytest.raises(ValueError, match="widen the cohort"):
        build_dataloader("all", config, training=False)
