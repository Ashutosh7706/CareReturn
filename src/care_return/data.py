"""Dataset validation and feature preparation for CareReturn."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

TARGET = "readmitted"
GROUP = "patient_nbr"
DROP_FROM_MODEL = {"encounter_id", GROUP, TARGET, "discharge_disposition_id"}

@dataclass
class PreparedData:
    features: pd.DataFrame
    target: pd.Series
    patient_groups: pd.Series
    audit: dict


def read_encounters(path: str | Path) -> pd.DataFrame:
    """Load the UCI encounter CSV and validate the minimum schema."""
    frame = pd.read_csv(path, low_memory=False)
    required = {TARGET, GROUP}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("The input dataset contains no encounter rows.")
    return frame


def make_readmission_target(labels: pd.Series) -> pd.Series:
    """Map <30 to 1; map NO and >30 to 0; reject unknown labels."""
    clean = labels.astype("string").str.strip()
    allowed = {"<30", ">30", "NO"}
    unexpected = sorted(set(clean.dropna().unique()) - allowed)
    if unexpected or clean.isna().any():
        raise ValueError(f"Unexpected readmission labels: {unexpected}; null labels: {int(clean.isna().sum())}")
    return clean.eq("<30").astype("int8").rename("readmitted_within_30_days")


def prepare_encounters(frame: pd.DataFrame) -> PreparedData:
    """Separate outcome/group keys and create model-ready raw predictors."""
    if TARGET not in frame or GROUP not in frame:
        raise ValueError(f"Input must contain '{TARGET}' and '{GROUP}'.")
    target = make_readmission_target(frame[TARGET])
    groups = frame[GROUP].copy()
    features = frame.drop(columns=[c for c in DROP_FROM_MODEL if c in frame], errors="ignore").copy()
    # UCI uses '?' for unavailable values. Treat it as missing, not a category.
    features = features.replace({"?": pd.NA, "Unknown/Invalid": pd.NA})
    # Numeric-looking columns are converted conservatively; categorical values remain intact.
    for column in features.columns:
        if features[column].dtype == "object":
            converted = pd.to_numeric(features[column], errors="coerce")
            non_missing = features[column].notna().sum()
            if non_missing and converted.notna().sum() / non_missing > 0.95:
                features[column] = converted
    audit = {
        "rows": int(len(frame)),
        "columns_received": int(frame.shape[1]),
        "predictor_count": int(features.shape[1]),
        "positive_rate": float(target.mean()),
        "unique_patients": int(groups.nunique()),
        "missing_cells": int(features.isna().sum().sum()),
        "excluded_columns": sorted(DROP_FROM_MODEL.intersection(frame.columns)),
    }
    return PreparedData(features, target, groups, audit)
