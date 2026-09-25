import pytest
from minalphafold.trainer import DataConfig, build_dataloader
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler

from tests.test_trainer import make_processed_cache_dirs


def _config(tmp_path):
    feature_dir, label_dir = make_processed_cache_dirs(tmp_path)
    return DataConfig(
        processed_features_dir=feature_dir,
        processed_labels_dir=label_dir,
        val_fraction=0.0,
    )


def test_distributed_validation_partitions_without_padding(tmp_path):
    config = _config(tmp_path)
    datasets = [
        build_dataloader(
            "all", config, training=False, distributed_rank=rank, distributed_world_size=3
        ).dataset
        for rank in range(3)
    ]
    subsets = []
    for dataset in datasets:
        assert isinstance(dataset, Subset)
        subsets.append(dataset)
    indices = [list(dataset.indices) for dataset in subsets]
    flattened = [index for shard in indices for index in shard]
    assert sorted(flattened) == list(range(len(subsets[0].dataset)))
    assert len(flattened) == len(set(flattened))
    assert any(not shard for shard in indices)


def test_distributed_training_uses_seeded_sampler(tmp_path):
    loader = build_dataloader(
        "all", _config(tmp_path), training=True, seed=17,
        distributed_rank=1, distributed_world_size=2,
    )
    assert isinstance(loader.sampler, DistributedSampler)
    assert loader.sampler.rank == 1
    assert loader.sampler.num_replicas == 2
    assert loader.sampler.seed == 17


@pytest.mark.parametrize("rank,world_size", [(None, 0), (None, 2), (-1, 2), (2, 2), (1, 1)])
def test_distributed_dataloader_rejects_invalid_topology(tmp_path, rank, world_size):
    with pytest.raises(ValueError, match="distributed_"):
        build_dataloader(
            "all", _config(tmp_path), training=False,
            distributed_rank=rank, distributed_world_size=world_size,
        )
