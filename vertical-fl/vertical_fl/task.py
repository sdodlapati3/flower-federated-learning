"""Vertical FL with Flower - Modified to use local Titanic dataset.

Vertical FL (Split Learning) - each client has different features for the same samples.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURE_COLUMNS = [
    "Age",
    "Sex",
    "Fare",
    "Siblings/Spouses Aboard",
    "Name",
    "Parents/Children Aboard",
    "Pclass",
]

# Cache the titanic dataset
_titanic_data = None


def get_titanic_data():
    """Load Titanic dataset from local file or create synthetic version."""
    global _titanic_data
    
    if _titanic_data is not None:
        return _titanic_data
    
    data_dir = Path(__file__).parent.parent / "data"
    data_file = data_dir / "titanic.csv"
    
    if data_file.exists():
        _titanic_data = pd.read_csv(data_file)
    else:
        # Create a synthetic Titanic-like dataset
        np.random.seed(42)
        n_samples = 891  # Same as original Titanic
        
        _titanic_data = pd.DataFrame({
            "Age": np.random.uniform(1, 80, n_samples),
            "Sex": np.random.choice(["male", "female"], n_samples),
            "Fare": np.random.uniform(5, 500, n_samples),
            "Siblings/Spouses Aboard": np.random.randint(0, 5, n_samples),
            "Name": [f"Person_{i}, Mr/Mrs. Test" for i in range(n_samples)],
            "Parents/Children Aboard": np.random.randint(0, 3, n_samples),
            "Pclass": np.random.choice([1, 2, 3], n_samples),
            "Survived": np.random.randint(0, 2, n_samples),
        })
        
        # Save for future use
        data_dir.mkdir(parents=True, exist_ok=True)
        _titanic_data.to_csv(data_file, index=False)
    
    return _titanic_data


def load_and_preprocess(
    dataframe: pd.DataFrame,
):
    """Preprocess a subset of the titanic-survival dataset columns into a purely
    numerical numpy array suitable for model training."""

    # Make a copy to avoid modifying the original
    X_df = dataframe.copy()

    # Identify which columns are present
    available_cols = set(X_df.columns)

    # ----------------------------------------------------------------------
    # FEATURE ENGINEERING ON NAME (if present)
    # ----------------------------------------------------------------------
    if "Name" in available_cols:
        X_df["Title"] = X_df["Name"].str.extract(r"([A-Za-z]+)\.", expand=False)
        X_df["NameLength"] = X_df["Name"].str.len()
        X_df = X_df.drop(columns=["Name"])

    # ----------------------------------------------------------------------
    # IDENTIFY NUMERIC + CATEGORICAL COLUMNS
    # ----------------------------------------------------------------------
    categorical_cols = []
    if "Sex" in X_df.columns:
        categorical_cols.append("Sex")
    if "Title" in X_df.columns:
        categorical_cols.append("Title")
    if "Pclass" in X_df.columns:
        categorical_cols.append("Pclass")

    numeric_cols = [c for c in X_df.columns if c not in categorical_cols]

    # ----------------------------------------------------------------------
    # HANDLE MISSING VALUES
    # ----------------------------------------------------------------------
    if numeric_cols:
        X_df[numeric_cols] = X_df[numeric_cols].fillna(X_df[numeric_cols].median())

    # ----------------------------------------------------------------------
    # PREPROCESSOR (TRANSFORM TO PURE NUMERIC)
    # ----------------------------------------------------------------------
    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    # ----------------------------------------------------------------------
    # FIT TRANSFORMER & CONVERT TO NUMPY
    # ----------------------------------------------------------------------
    X_full = preprocessor.fit_transform(X_df)

    # Ensure output is always a dense numpy array
    if hasattr(X_full, "toarray"):
        X_full = X_full.toarray()

    return X_full.astype(np.float32)


def load_data(partition_id: int, feature_splits: list[int]):
    """Load vertically partitioned data for a specific client.
    
    In vertical FL, each client has a different subset of features
    for the same set of samples.
    """
    df = get_titanic_data()
    
    # Calculate which features this partition gets
    start_idx = sum(feature_splits[:partition_id])
    end_idx = start_idx + feature_splits[partition_id]
    
    # Get the feature columns for this partition
    partition_columns = FEATURE_COLUMNS[start_idx:end_idx]
    
    # For the last partition (active party), also include labels
    if partition_id == len(feature_splits) - 1:
        partition_df = df[partition_columns + ["Survived"]].copy()
    else:
        partition_df = df[partition_columns].copy()
    
    # Process partition
    return load_and_preprocess(dataframe=partition_df)


class ClientModel(nn.Module):
    def __init__(self, input_size, out_feat_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, out_feat_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        return self.fc2(x)


class ServerModel(nn.Module):
    def __init__(self, input_size):
        super(ServerModel, self).__init__()
        self.hidden = nn.Linear(input_size, 96)
        self.fc = nn.Linear(96, 1)
        self.bn = nn.BatchNorm1d(96)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden(x)
        x = nn.functional.relu(x)
        x = self.bn(x)
        x = self.fc(x)
        return self.sigmoid(x)


def evaluate_head_model(
    head: ServerModel, embeddings: torch.Tensor, labels: torch.Tensor
) -> float:
    """Compute accuracy of head."""
    head.eval()
    with torch.no_grad():
        correct = 0
        # Re-compute embeddings for accuracy (detached from grad)
        embeddings_eval = embeddings.detach()
        output = head(embeddings_eval)
        predicted = (output > 0.5).float()
        correct += (predicted == labels).sum().item()
        accuracy = correct / len(labels) * 100

    return accuracy
