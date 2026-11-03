import pandas as pd
from care_return.analysis import cohort_summary


def test_cohort_shares_sum_to_one():
    result = cohort_summary(pd.DataFrame({"readmitted": ["NO", "NO", "<30"]}))
    assert abs(result["share"].sum() - 1.0) < 1e-9
