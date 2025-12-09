"""xgboost_quickstart: A Flower / XGBoost app.

Modified to use sklearn synthetic data instead of Higgs dataset for faster execution.
"""

import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split as sklearn_split

# Global cache for partitioned data
_cached_data = None


def load_data(partition_id, num_clients):
    """Load partition of synthetic classification data.
    
    Uses sklearn's make_classification to generate data locally,
    avoiding slow HuggingFace downloads on HPC clusters.
    """
    global _cached_data
    
    if _cached_data is None:
        # Generate a synthetic binary classification dataset
        # Similar structure to Higgs: 28 features, binary classification
        X, y = make_classification(
            n_samples=10000 * num_clients,  # 10k samples per client
            n_features=28,
            n_informative=20,
            n_redundant=5,
            n_clusters_per_class=3,
            random_state=42,
        )
        _cached_data = (X, y)
    
    X, y = _cached_data
    
    # Partition data across clients
    total_size = len(X)
    partition_size = total_size // num_clients
    start_idx = partition_id * partition_size
    end_idx = start_idx + partition_size
    
    X_partition = X[start_idx:end_idx]
    y_partition = y[start_idx:end_idx]
    
    # Train/test split (80/20)
    X_train, X_valid, y_train, y_valid = sklearn_split(
        X_partition, y_partition, test_size=0.2, random_state=42
    )
    
    num_train = len(X_train)
    num_val = len(X_valid)
    
    # Convert to DMatrix for XGBoost
    train_dmatrix = xgb.DMatrix(X_train, label=y_train)
    valid_dmatrix = xgb.DMatrix(X_valid, label=y_valid)
    
    return train_dmatrix, valid_dmatrix, num_train, num_val


def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict
