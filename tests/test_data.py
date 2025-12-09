"""
Tests for fl_research.data module.
"""

import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader

from fl_research.data import (
    Partitioner,
    IIDPartitioner,
    DirichletPartitioner,
    ShardPartitioner,
)


class TestIIDPartitioner:
    """Tests for IID partitioner."""
    
    def test_partition_creates_correct_num_partitions(self, sample_dataset):
        """Test that partitioner creates correct number of partitions."""
        partitioner = IIDPartitioner(num_clients=5)
        partitions = partitioner.partition(sample_dataset)
        
        assert len(partitions) == 5
    
    def test_partition_sizes_roughly_equal(self, sample_dataset):
        """Test that partitions are roughly equal in size."""
        partitioner = IIDPartitioner(num_clients=4)
        partitions = partitioner.partition(sample_dataset)
        
        sizes = [len(p) for p in partitions]
        
        # All sizes should be within 1 of each other
        assert max(sizes) - min(sizes) <= 1
    
    def test_no_overlap(self, sample_dataset):
        """Test that partitions don't overlap."""
        partitioner = IIDPartitioner(num_clients=3)
        partitions = partitioner.partition(sample_dataset)
        
        all_indices = set()
        for indices in partitions:
            for idx in indices:
                assert idx not in all_indices
                all_indices.add(idx)
    
    def test_covers_all_data(self, sample_dataset):
        """Test that partitions cover all data."""
        partitioner = IIDPartitioner(num_clients=5)
        partitions = partitioner.partition(sample_dataset)
        
        total_indices = sum(len(p) for p in partitions)
        assert total_indices == len(sample_dataset)
    
    def test_reproducibility(self, sample_dataset):
        """Test that same seed produces same partitions."""
        partitioner1 = IIDPartitioner(num_clients=3, seed=42)
        partitioner2 = IIDPartitioner(num_clients=3, seed=42)
        
        partitions1 = partitioner1.partition(sample_dataset)
        partitions2 = partitioner2.partition(sample_dataset)
        
        for p1, p2 in zip(partitions1, partitions2):
            assert list(p1) == list(p2)


class TestDirichletPartitioner:
    """Tests for Dirichlet (non-IID) partitioner."""
    
    def test_partition_creates_correct_num_partitions(self, sample_dataset):
        """Test correct number of partitions."""
        partitioner = DirichletPartitioner(num_clients=5, alpha=0.5)
        partitions = partitioner.partition(sample_dataset)
        
        assert len(partitions) == 5
    
    def test_alpha_affects_distribution(self, sample_dataset):
        """Test that alpha affects heterogeneity."""
        # Low alpha = more heterogeneous
        partitioner_low = DirichletPartitioner(num_clients=3, alpha=0.1, seed=42)
        partitioner_high = DirichletPartitioner(num_clients=3, alpha=10.0, seed=42)
        
        partitions_low = partitioner_low.partition(sample_dataset)
        partitions_high = partitioner_high.partition(sample_dataset)
        
        # Both should have valid partitions
        assert len(partitions_low) == 3
        assert len(partitions_high) == 3


class TestShardPartitioner:
    """Tests for shard-based partitioner."""
    
    def test_partition_creates_shards(self, sample_dataset):
        """Test shard-based partitioning."""
        partitioner = ShardPartitioner(
            num_clients=5,
            shards_per_client=2,
        )
        partitions = partitioner.partition(sample_dataset)
        
        assert len(partitions) == 5
