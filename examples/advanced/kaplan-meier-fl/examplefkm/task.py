"""examplefkm: A Flower / Lifelines app for Federated Kaplan-Meier Survival Analysis.

Modified to avoid HuggingFace dependencies - uses lifelines directly.
"""

import numpy as np
from lifelines.datasets import load_waltons

# Load the Waltons dataset (comes with lifelines)
X = load_waltons()


def load_partition(partition_id: int):
    """Load partition based on group for federated survival analysis.
    
    The Waltons dataset has groups: 'miR-137' and 'control'
    """
    # Get unique groups
    groups = X["group"].unique()
    
    # Select the group for this partition
    if partition_id < len(groups):
        selected_group = groups[partition_id]
        partition = X[X["group"] == selected_group]
    else:
        # If more partitions than groups, use random split
        np.random.seed(partition_id)
        mask = np.random.choice([True, False], size=len(X), p=[0.5, 0.5])
        partition = X[mask] if partition_id % 2 == 0 else X[~mask]
    
    times = partition["T"].values
    events = partition["E"].values
    return times, events
