import pandas as pd
from care_return.splitting import patient_partitions, assert_patient_isolation


def test_patient_groups_never_cross_partitions():
    groups = pd.Series([f"p{i}" for i in range(80) for _ in range(2)])
    features = pd.DataFrame({"x": range(len(groups))})
    target = pd.Series([i % 2 for i in range(len(groups))])
    parts = patient_partitions(features, target, groups)
    assert_patient_isolation(groups, parts)
