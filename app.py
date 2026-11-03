"""CareReturn interactive analytics dashboard (research/learning use only)."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.inspection import permutation_importance
from care_return.data import read_encounters, prepare_encounters

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data/raw/diabetes_130_us_hospitals.csv"
REPORT_PATH = ROOT / "reports/evaluation.json"
MODEL_PATH = ROOT / "artifacts/care_return_pipeline.joblib"

st.set_page_config(page_title="CareReturn | Readmission Analytics", page_icon="🏥", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.6rem;padding-bottom:2rem}
[data-testid="stMetric"]{background:#f4f7fb;padding:15px;border-radius:12px;border:1px solid #e5eaf1}
[data-testid="stMetricLabel"]{color:#53657d}
</style>""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data(path: str, modified: float) -> pd.DataFrame:
    return read_encounters(path)

@st.cache_resource(show_spinner=False)
def load_model(path: str, modified: float):
    return joblib.load(path)

@st.cache_data(show_spinner=False)
def load_report(path: str, modified: float) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def chart_note():
    st.caption("Descriptive analysis of the historical UCI cohort (1999–2008); not a clinical decision tool.")

def safe_rate(series: pd.Series) -> float:
    return float(series.mean() * 100) if len(series) else 0.0

st.title("🏥 CareReturn")
st.markdown("**Hospital readmission analytics** · Cohort exploration · Model evaluation · Risk patterns")
st.warning("Research and educational use only. Predictions and associations must not be used for patient care, triage, discharge, or treatment decisions.")

if not DATA_PATH.exists():
    with st.spinner("Downloading hospital readmission dataset from UCI..."):
        try:
            from ucimlrepo import fetch_ucirepo

            dataset = fetch_ucirepo(id=296)
            downloaded_data = dataset.data.original

            if downloaded_data is None or downloaded_data.empty:
                raise ValueError("UCI returned an empty dataset.")

            DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            downloaded_data.to_csv(DATA_PATH, index=False)

            st.success("Hospital dataset downloaded successfully!")

        except Exception as e:
            st.error(f"Dataset download failed: {e}")
            st.info("Please check the deployment logs and try again.")
            st.stop()
frame = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)
prepared = prepare_encounters(frame)
frame = frame.copy()
frame["readmission_30d"] = prepared.target.values

page = st.sidebar.radio("DASHBOARDS", ["Hospital Overview", "Exploratory Analysis", "Model Performance", "Risk & Feature Analysis"], index=0)
st.sidebar.divider()
st.sidebar.caption("CareReturn · Independent educational project")
st.sidebar.caption(f"Loaded {len(frame):,} encounter rows")

if page == "Hospital Overview":
    st.subheader("Hospital cohort at a glance")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Encounters", f"{len(frame):,}")
    c2.metric("Unique patients", f"{frame.patient_nbr.nunique():,}")
    c3.metric("30-day readmission", f"{safe_rate(frame.readmission_30d):.1f}%")
    c4.metric("Predictor fields", f"{prepared.features.shape[1]:,}")
    left,right = st.columns(2)
    with left:
        labels = frame.readmitted.value_counts().reindex(["NO",">30","<30"]).dropna().reset_index()
        labels.columns = ["Outcome", "Encounters"]
        st.plotly_chart(px.pie(labels, names="Outcome", values="Encounters", hole=.58, title="Encounter outcome distribution"), use_container_width=True)
    with right:
        age = frame.groupby("age", dropna=False).readmission_30d.agg(["size","mean"]).reset_index()
        age.columns = ["Age band","Encounters","30-day rate"]
        age["30-day rate"] *= 100
        st.plotly_chart(px.bar(age, x="Age band", y="Encounters", color="30-day rate", title="Encounters by age band", color_continuous_scale="Blues"), use_container_width=True)
    st.markdown("#### Data quality snapshot")
    quality = pd.DataFrame({"Measure":["Missing predictor cells","Missingness (% of predictor cells)","Distinct patients","Duplicate encounter IDs"],
      "Value":[f"{prepared.audit['missing_cells']:,}", f"{prepared.audit['missing_cells']/(max(1,len(frame)*prepared.features.shape[1]))*100:.2f}%", f"{prepared.audit['unique_patients']:,}", f"{frame.encounter_id.duplicated().sum():,}" if "encounter_id" in frame else "Not available"]})
    st.dataframe(quality, hide_index=True, use_container_width=True)
    chart_note()

elif page == "Exploratory Analysis":
    st.subheader("Exploratory cohort analysis")
    col1,col2 = st.columns(2)
    with col1:
        field = st.selectbox("Compare readmission by", ["age","gender","race","admission_type_id","discharge_disposition_id","admission_source_id","medical_specialty"])
    with col2:
        top_n = st.slider("Maximum categories", 5, 25, 12)
    grouped = frame.groupby(field, dropna=False).readmission_30d.agg(encounters="size", readmission_rate="mean").reset_index()
    grouped["readmission_rate"] *= 100
    grouped[field] = grouped[field].fillna("Missing").astype(str)
    grouped = grouped.sort_values("encounters", ascending=False).head(top_n)
    st.plotly_chart(px.bar(grouped, x=field, y="readmission_rate", color="encounters", hover_data=["encounters"], title=f"30-day readmission rate by {field}", color_continuous_scale="Teal"), use_container_width=True)
    col1,col2 = st.columns(2)
    with col1:
        if "time_in_hospital" in frame:
            stay = frame.groupby("time_in_hospital").readmission_30d.agg(encounters="size", rate="mean").reset_index()
            stay["rate"] *= 100
            st.plotly_chart(px.line(stay, x="time_in_hospital", y="rate", markers=True, title="Readmission rate by length of stay"), use_container_width=True)
    with col2:
        if "num_medications" in frame:
            med = frame.assign(medication_band=pd.cut(frame.num_medications, bins=[-1,5,10,15,20,30,100], labels=["0–5","6–10","11–15","16–20","21–30","31+"])).groupby("medication_band", observed=False).readmission_30d.mean().mul(100).rename("rate").reset_index()
            st.plotly_chart(px.bar(med, x="medication_band", y="rate", title="Readmission rate by medication-count band"), use_container_width=True)
    st.markdown("#### Group summary")
    st.dataframe(grouped.rename(columns={field:"Group", "encounters":"Encounters", "readmission_rate":"30-day readmission (%)"}), hide_index=True, use_container_width=True)
    chart_note()

elif page == "Model Performance":
    st.subheader("Model evaluation")
    if not REPORT_PATH.exists():
        st.info("No evaluation report found. Train the models first: `python scripts/run_analysis.py`.")
        st.stop()
    report = load_report(str(REPORT_PATH), REPORT_PATH.stat().st_mtime)
    scores = report.get("validation_comparison", {})
    if scores:
        rows = [{"Model":name,"Average precision":v.get("average_precision"),"ROC-AUC":v.get("roc_auc"),"Precision":v.get("precision"),"Recall":v.get("recall"),"F1":v.get("f1"),"F2":v.get("f2")} for name,v in scores.items()]
        metrics = pd.DataFrame(rows)
        st.plotly_chart(px.bar(metrics.melt(id_vars="Model", value_vars=["Average precision","ROC-AUC","Precision","Recall","F1","F2"], var_name="Metric", value_name="Score"), x="Model", y="Score", color="Metric", barmode="group", title="Validation model comparison", range_y=[0,1]), use_container_width=True)
        st.dataframe(metrics.round(3), hide_index=True, use_container_width=True)
    test = report.get("held_out_test", {})
    a,b,c,d = st.columns(4)
    a.metric("Selected model", report.get("selected_model","—"))
    b.metric("Test average precision", f"{test.get('average_precision',0):.3f}")
    c.metric("Test recall", f"{test.get('recall',0):.3f}")
    d.metric("Decision threshold", f"{test.get('threshold',0):.2f}")
    cm = np.array(test.get("confusion_matrix", [[0,0],[0,0]]))
    fig = px.imshow(cm, text_auto=True, x=["Predicted: No","Predicted: Yes"], y=["Actual: No","Actual: Yes"], color_continuous_scale="Blues", title="Held-out test confusion matrix")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Validation metrics compare candidate models. Held-out test metrics are the final evaluation; do not tune the model against the test set.")
    chart_note()

else:
    st.subheader("Risk patterns & feature analysis")
    if not MODEL_PATH.exists():
        st.info("Trained model not found. Run `python scripts/run_analysis.py` first.")
        st.stop()
    bundle = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    estimator = bundle["pipeline"]
    st.markdown(f"Selected model: **{bundle.get('model_name','unknown')}** · Stored threshold: **{bundle.get('threshold',0):.2f}**")
    st.caption("The cohort score distribution below is generated by applying the trained model to the full historical dataset. It is an exploratory in-sample view, not an unbiased estimate of future performance.")
    with st.spinner("Scoring cohort for exploratory analysis…"):
        sample_n = min(12000, len(prepared.features))
        sample = prepared.features.sample(sample_n, random_state=23) if len(prepared.features)>sample_n else prepared.features
        probs = estimator.predict_proba(sample)[:,1]
    st.plotly_chart(px.histogram(pd.DataFrame({"Predicted probability":probs}), x="Predicted probability", nbins=30, title="Exploratory model-score distribution"), use_container_width=True)
    st.markdown("#### Permutation feature importance")
    st.caption("Importance is measured by the decrease in average precision after shuffling one input column. It describes model reliance, not causation.")
    if st.button("Calculate feature importance (may take a minute)"):
        with st.spinner("Calculating permutation importance on a sample…"):
            n = min(1500, len(prepared.features))
            x = prepared.features.sample(n, random_state=23)
            y = prepared.target.loc[x.index]
            result = permutation_importance(estimator, x, y, scoring="average_precision", n_repeats=3, random_state=23, n_jobs=-1)
            importance = pd.DataFrame({"Feature":x.columns,"Importance":result.importances_mean}).sort_values("Importance", ascending=False).head(15)
            st.session_state["care_return_importance"] = importance
    if "care_return_importance" in st.session_state:
        imp = st.session_state["care_return_importance"]
        st.plotly_chart(px.bar(imp.sort_values("Importance"), x="Importance", y="Feature", orientation="h", title="Top 15 permutation importances"), use_container_width=True)
        st.dataframe(imp, hide_index=True, use_container_width=True)
    chart_note()
