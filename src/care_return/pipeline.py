"""Model construction and validation-led training workflow."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score, fbeta_score, brier_score_loss, confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .data import prepare_encounters
from .splitting import patient_partitions, assert_patient_isolation


def _preprocessor(data: pd.DataFrame) -> ColumnTransformer:
    numeric = data.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [name for name in data.columns if name not in numeric]
    numeric_steps = Pipeline([("fill", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())])
    category_steps = Pipeline([("fill", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=8))])
    return ColumnTransformer([("numeric", numeric_steps, numeric), ("category", category_steps, categorical)])


def _candidates(data: pd.DataFrame, seed: int) -> dict[str, Pipeline]:
    return {
        "logistic": Pipeline([("prepare", _preprocessor(data)), ("classifier", LogisticRegression(max_iter=600, class_weight="balanced", random_state=seed))]),
        "random_forest": Pipeline([("prepare", _preprocessor(data)), ("classifier", RandomForestClassifier(n_estimators=220, min_samples_leaf=4, class_weight="balanced_subsample", n_jobs=-1, random_state=seed))]),
        "extra_trees": Pipeline([("prepare", _preprocessor(data)), ("classifier", ExtraTreesClassifier(n_estimators=220, min_samples_leaf=4, class_weight="balanced", n_jobs=-1, random_state=seed))]),
    }


def _threshold_for_f2(actual: np.ndarray, probability: np.ndarray) -> float:
    candidates = np.linspace(0.10, 0.90, 81)
    scores = [fbeta_score(actual, probability >= t, beta=2, zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def score_binary(actual: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    predicted = probability >= threshold
    return {
        "threshold": float(threshold),
        "average_precision": float(average_precision_score(actual, probability)),
        "roc_auc": float(roc_auc_score(actual, probability)) if len(np.unique(actual)) == 2 else None,
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "f2": float(fbeta_score(actual, predicted, beta=2, zero_division=0)),
        "brier_score": float(brier_score_loss(actual, probability)),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1]).tolist(),
    }


def train_project(frame: pd.DataFrame, output_dir: str | Path = "artifacts", report_dir: str | Path = "reports", seed: int = 23) -> dict:
    prepared = prepare_encounters(frame)
    parts = patient_partitions(prepared.features, prepared.target, prepared.patient_groups, seed)
    assert_patient_isolation(prepared.patient_groups, parts)
    validation_scores = {}
    fitted = {}
    for name, estimator in _candidates(prepared.features, seed).items():
        estimator.fit(prepared.features.iloc[parts["train"]], prepared.target.iloc[parts["train"]])
        probability = estimator.predict_proba(prepared.features.iloc[parts["validation"]])[:, 1]
        threshold = _threshold_for_f2(prepared.target.iloc[parts["validation"]].to_numpy(), probability)
        validation_scores[name] = score_binary(prepared.target.iloc[parts["validation"]].to_numpy(), probability, threshold)
        fitted[name] = estimator
    winner = max(validation_scores, key=lambda name: (validation_scores[name]["average_precision"], validation_scores[name]["f2"]))
    threshold = validation_scores[winner]["threshold"]
    train_and_validation = np.concatenate([parts["train"], parts["validation"]])
    final_model = _candidates(prepared.features, seed)[winner]
    final_model.fit(prepared.features.iloc[train_and_validation], prepared.target.iloc[train_and_validation])
    test_probability = final_model.predict_proba(prepared.features.iloc[parts["test"]])[:, 1]
    test_score = score_binary(prepared.target.iloc[parts["test"]].to_numpy(), test_probability, threshold)
    output_dir, report_dir = Path(output_dir), Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True); report_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "care_return_pipeline.joblib"
    joblib.dump({"pipeline": final_model, "threshold": threshold, "model_name": winner, "feature_names": list(prepared.features.columns), "seed": seed}, model_path)
    report = {
        "project": "CareReturn",
        "data_audit": prepared.audit,
        "partition_rows": {k: int(len(v)) for k, v in parts.items()},
        "partition_patients": {k: int(prepared.patient_groups.iloc[v].nunique()) for k, v in parts.items()},
        "validation_comparison": validation_scores,
        "selected_model": winner,
        "held_out_test": test_score,
        "model_artifact": str(model_path),
        "interpretation": "Metrics describe this retrospective dataset only; they are not evidence of clinical utility.",
    }
    (report_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
