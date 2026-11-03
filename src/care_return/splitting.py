"""Patient-disjoint train, validation and test partitioning."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def patient_partitions(features: pd.DataFrame, target: pd.Series, groups: pd.Series, seed: int = 23) -> dict[str, np.ndarray]:
    """Split by patient identifier so no person's encounters cross partitions."""
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_validation, test = next(outer.split(features, target, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1)
    train_rel, validation_rel = next(inner.split(features.iloc[train_validation], target.iloc[train_validation], groups.iloc[train_validation]))
    return {"train": train_validation[train_rel], "validation": train_validation[validation_rel], "test": test}


def assert_patient_isolation(groups: pd.Series, partitions: dict[str, np.ndarray]) -> None:
    patient_sets = {name: set(groups.iloc[index].dropna().tolist()) for name, index in partitions.items()}
    names = list(patient_sets)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = patient_sets[left].intersection(patient_sets[right])
            if overlap:
                raise AssertionError(f"Patient overlap between {left} and {right}: {len(overlap)}")
