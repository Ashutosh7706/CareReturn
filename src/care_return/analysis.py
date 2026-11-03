"""Exploratory summaries and subgroup performance auditing."""
import pandas as pd
from sklearn.metrics import average_precision_score, recall_score, precision_score


def cohort_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return compact cohort counts by the original readmission label."""
    if "readmitted" not in frame:
        raise ValueError("Expected a 'readmitted' outcome column.")
    return (frame.groupby("readmitted", dropna=False).size().rename("encounters").to_frame()
            .assign(share=lambda x: x["encounters"] / x["encounters"].sum()).reset_index())


def subgroup_audit(frame: pd.DataFrame, actual, probability, threshold: float, columns=("race", "gender")) -> dict:
    """Calculate descriptive subgroup metrics; small groups are flagged."""
    result = {}
    for column in columns:
        if column not in frame:
            continue
        entries = {}
        for value, indices in frame.groupby(column, dropna=False).groups.items():
            idx = list(indices)
            y = actual.loc[idx] if hasattr(actual, "loc") else actual[idx]
            p = probability.loc[idx] if hasattr(probability, "loc") else probability[idx]
            entries[str(value)] = {
                "n": len(idx),
                "average_precision": float(average_precision_score(y, p)) if len(set(y)) > 1 else None,
                "recall": float(recall_score(y, p >= threshold, zero_division=0)),
                "precision": float(precision_score(y, p >= threshold, zero_division=0)),
                "small_group_warning": len(idx) < 100,
            }
        result[column] = entries
    return result
