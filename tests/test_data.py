import pandas as pd
import pytest
from care_return.data import make_readmission_target, prepare_encounters


def test_target_mapping():
    labels = pd.Series(["<30", ">30", "NO"])
    assert make_readmission_target(labels).tolist() == [1, 0, 0]


def test_unknown_target_is_rejected():
    with pytest.raises(ValueError):
        make_readmission_target(pd.Series(["<30", "maybe"]))


def test_identifiers_and_outcome_are_not_predictors():
    frame = pd.DataFrame({"patient_nbr": [1, 2], "encounter_id": [10, 20], "readmitted": ["<30", "NO"], "age": ["[50-60)", "[60-70)"]})
    prepared = prepare_encounters(frame)
    assert "patient_nbr" not in prepared.features
    assert "readmitted" not in prepared.features
    assert prepared.target.tolist() == [1, 0]
