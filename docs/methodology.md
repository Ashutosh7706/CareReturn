# Methodology

## Outcome definition

The source label has three values: `<30`, `>30`, and `NO`. CareReturn defines a positive outcome as `<30`; the other two labels are negative.

## Data preparation

The patient identifier is retained only as a grouping variable. Encounter identifier, patient identifier, target and `discharge_disposition_id` are excluded from predictors. Question-mark and invalid-value placeholders are treated as missing. Numeric-like object columns are converted only when at least 95% of their nonmissing values parse numerically.

## Validation design

The data is split by patient group into training, validation and test partitions. Candidate models are fitted on training data. Validation average precision is the primary selection criterion; F2 is the tie-breaker. The decision threshold is selected on validation data by maximizing F2 over a fixed threshold grid. The selected model is refitted on training plus validation data and evaluated on the untouched test partition.

## Metrics

Average precision, ROC-AUC, precision, recall, F1, F2, Brier score and confusion matrix are reported. These metrics describe retrospective predictive performance only.

## Known limitations

The split is not temporal, the cohort is historical and geographically limited, and there is no external validation. Group splitting does not address distribution shift, label bias, or all possible forms of leakage. This is not a clinical validation study.
