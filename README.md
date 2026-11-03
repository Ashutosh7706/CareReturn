# CareReturn | Hospital Readmission Analytics

CareReturn is an independently implemented educational machine-learning study that explores whether a hospital encounter is followed by a readmission within 30 days. It combines cohort exploration, patient-aware data partitioning, baseline model comparison and held-out evaluation.

> **Research/learning use only.** This repository is not a medical device, clinical decision-support system or source of medical advice. It must not be used to triage, treat, discharge or otherwise make decisions about real patients.

## What this project does

- Defines the binary outcome as `1` for the UCI label `<30` and `0` for `NO` or `>30`.
- Converts the source dataset's question marks and invalid placeholders into missing values.
- Excludes encounter identifiers, patient identifiers, the outcome and a discharge-time field from predictors.
- Keeps all encounters belonging to one patient in a single partition.
- Compares logistic regression, random forest and extremely randomized trees.
- Selects a candidate using validation average precision, then tunes a recall-oriented F2 threshold on validation data.
- Evaluates the selected model once on the held-out test partition.
- Produces a JSON report and a reusable local model artifact.

## Dataset

The optional downloader retrieves **Diabetes 130-US Hospitals for Years 1999–2008** (UCI dataset 296). Dataset attribution:

Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). *Diabetes 130-US Hospitals for Years 1999–2008*. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J

The UCI dataset is listed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**. That dataset license is separate from the original code project's MIT license. This repository does not include the dataset; retrieve it from UCI and comply with its attribution and terms. Do not commit patient-level data.

## Setup

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Run the analysis and interactive dashboards

First download the dataset and train/evaluate the models:

```bash
python scripts/fetch_dataset.py
python scripts/run_analysis.py
pytest -q
```

### Easiest option on Windows

Double-click **`START_DASHBOARD.bat`** in the project folder. It creates `.venv` if needed, installs all dependencies (including Streamlit), downloads the UCI dataset if it is not already present, trains the models if the evaluation report is missing, and launches the dashboard.

### Manual option

From PowerShell in the project folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/fetch_dataset.py
python scripts/run_analysis.py
python -m streamlit run app.py
```

Use `python -m streamlit` rather than `streamlit` to avoid Windows PATH issues. The dashboard normally opens at `http://localhost:8501`.

The app contains four sections: **Hospital Overview**, **Exploratory Analysis**, **Model Performance**, and **Risk & Feature Analysis**. The final section includes an optional permutation-importance calculation. The model-performance page reads `reports/evaluation.json`; the risk page reads the trained pipeline in `artifacts/`. The model training command must complete before the model-specific dashboard sections can display results.

If the PowerShell activation command is blocked by execution policy, skip activation and run the commands using `.venv\Scripts\python.exe -m pip ...` and `.venv\Scripts\python.exe -m streamlit run app.py`.

To use a CSV already downloaded from the cited UCI source:

```bash
python scripts/run_analysis.py --input path/to/diabetic_data.csv
```

Outputs are written locally to `artifacts/` and `reports/`; these folders are ignored by Git.

## Repository map

```text
care-return/
├── data/raw/                 # local dataset only; not committed
├── docs/                     # methodology and data governance
├── scripts/                  # download and run entry points
├── app.py                    # four-page Streamlit analytics dashboard
├── src/care_return/          # project implementation
├── tests/                    # unit tests
├── artifacts/                # local model files; ignored
├── reports/                  # local evaluation outputs; ignored
├── pyproject.toml
├── RIGHTS.md
└── README.md
```

## Methodological notes

A random row split is inappropriate when a patient can have multiple encounters: it can leak patient-specific patterns across training and evaluation. CareReturn therefore groups partitions by `patient_nbr` and never supplies that identifier to the estimator. Preprocessing is fitted within each model pipeline using training data only. The validation set is used for candidate selection and threshold selection; the test set is reserved for final evaluation.

Average precision is reported because the positive class may be uncommon. Precision, recall, F1, F2, ROC-AUC, Brier score and a confusion matrix are also reported. No single metric establishes clinical usefulness.

## Limitations and responsible use

- The source cohort is historical (1999–2008), US-based and limited to diabetic inpatient encounters.
- Results can be affected by historical practice, missingness, coding patterns and selection bias.
- Group-aware splitting reduces patient overlap but does not provide temporal or external validation.
- Subgroup performance should be examined before drawing conclusions; small groups produce unstable estimates.
- Feature associations are not causal explanations.
- Do not upload raw records, model artifacts containing memorized information, or other sensitive data to a public repository.

## Ownership and third-party material

See [`RIGHTS.md`](RIGHTS.md). The source code in this repository was newly written for CareReturn. The upstream repository's source files and MIT license are not included. The UCI dataset remains subject to its own license and attribution requirements.
