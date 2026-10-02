"""
Predictive Maintenance Demo - Streamlit web UI.

Run:  streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Make "src" importable when Streamlit runs this file directly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import FEATURE_COLUMNS, LABEL, build_features  # noqa: E402
from src.generate_data import DATA_DIR, EQUIPMENT_TYPES, SENSORS, generate, load_raw  # noqa: E402
from src.predict import (latest_risk, load_model, predict_proba,  # noqa: E402
                         recommend_action, risk_band, score_all)
from src.train import MODEL_DIR, train  # noqa: E402

DISCLAIMER = "Synthetic data for educational purposes only."
BAND_COLORS = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}
SENSOR_LABELS = {
    "temperature_c": "Temperature (°C)",
    "vibration_mm_s": "Vibration (mm/s)",
    "current_amp": "Current (A)",
    "pressure_bar": "Pressure (bar)",
    "rpm": "Speed (rpm)",
}

st.set_page_config(page_title="Predictive Maintenance Demo", page_icon="🔧", layout="wide")


# ---------------------------------------------------------------- loading ---
@st.cache_resource(show_spinner="Preparing data and model (first run only)...")
def ensure_artifacts():
    """Generate data / train the model automatically if anything is missing."""
    if not (DATA_DIR / "sensor_readings.csv").exists():
        generate()
    if not (MODEL_DIR / "model.joblib").exists():
        train(verbose=False)
    return True


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_data():
    readings, failures, maintenance = load_raw()
    feats = build_features(readings, failures, maintenance)
    return readings, failures, maintenance, feats


@st.cache_data
def get_scored(_bundle, feats):
    return score_all(_bundle, feats)


@st.cache_data
def get_insights():
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
    importance = pd.read_csv(MODEL_DIR / "feature_importance.csv")
    return metrics, importance


ensure_artifacts()
bundle = get_model()
readings, failures, maintenance, feats = get_data()
scored = get_scored(bundle, feats)

# ---------------------------------------------------------------- sidebar ---
st.sidebar.title("🔧 Predictive Maintenance")
page = st.sidebar.radio(
    "Navigate", ["Fleet Overview", "Equipment Detail", "What-If Simulator", "Model Insights"])
st.sidebar.markdown("---")
st.sidebar.caption(DISCLAIMER)
st.sidebar.caption("Pipeline: sensor data → features → Random Forest → failure risk")

st.title("Predictive Maintenance Demo")
st.caption(f"⚠️ {DISCLAIMER} Probability = chance of failure within the next 7 days.")


# ---------------------------------------------------------- page: fleet -----
def page_fleet():
    latest = latest_risk(scored)
    st.subheader(f"Fleet Overview — as of {latest['date'].max():%Y-%m-%d}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total units", len(latest))
    c2.metric("High-risk units", int((latest["risk_band"] == "High").sum()))
    c3.metric("Average risk", f"{latest['failure_probability'].mean():.0%}")

    table = latest[["equipment_id", "equipment_type", "failure_probability", "risk_band",
                    "vibration_mm_s_mean_24h", "temperature_c_mean_24h",
                    "days_since_maintenance"]].rename(columns={
        "equipment_id": "Unit", "equipment_type": "Type",
        "failure_probability": "Failure probability", "risk_band": "Risk band",
        "vibration_mm_s_mean_24h": "Vibration 24h avg", "temperature_c_mean_24h": "Temp 24h avg",
        "days_since_maintenance": "Days since maint.",
    })

    def color_band(v):
        return f"background-color: {BAND_COLORS[v]}; color: white; font-weight: bold"

    styled = (table.style
              .map(color_band, subset=["Risk band"])
              .format({"Failure probability": "{:.1%}", "Vibration 24h avg": "{:.2f}",
                       "Temp 24h avg": "{:.1f}", "Days since maint.": "{:.0f}"}))
    st.dataframe(styled, use_container_width=True, hide_index=True, height=740)
    st.caption("Risk bands: Low < 0.3 · Medium 0.3–0.7 · High > 0.7")


# --------------------------------------------------------- page: detail -----
def page_detail():
    st.subheader("Equipment Detail")
    units = sorted(readings["equipment_id"].unique())
    c1, c2 = st.columns([1, 3])
    unit = c1.selectbox("Unit", units)
    chosen = c2.multiselect("Sensors", SENSORS, default=["vibration_mm_s", "temperature_c", "current_amp"],
                            format_func=SENSOR_LABELS.get)
    if not chosen:
        st.info("Select at least one sensor.")
        return

    r = readings[readings["equipment_id"] == unit]
    f = failures[failures["equipment_id"] == unit]
    m = maintenance[maintenance["equipment_id"] == unit]

    fig = make_subplots(rows=len(chosen), cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        subplot_titles=[SENSOR_LABELS[s] for s in chosen])
    for i, s in enumerate(chosen, start=1):
        fig.add_trace(go.Scatter(x=r["timestamp"], y=r[s], mode="lines", name=SENSOR_LABELS[s],
                                 line=dict(width=1)), row=i, col=1)
    for t in f["failure_time"]:
        fig.add_vline(x=t, line=dict(color="red", dash="dash", width=2), row="all", col=1)
    for t in m["maintenance_date"]:
        fig.add_vline(x=t, line=dict(color="green", dash="dot", width=1.5), row="all", col=1)
    fig.update_layout(height=230 * len(chosen) + 60, showlegend=False, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🔴 dashed = failure event · 🟢 dotted = preventive maintenance")

    # Risk score trend
    s = scored[scored["equipment_id"] == unit]
    risk = go.Figure()
    risk.add_hrect(y0=0, y1=0.3, fillcolor=BAND_COLORS["Low"], opacity=0.12, line_width=0)
    risk.add_hrect(y0=0.3, y1=0.7, fillcolor=BAND_COLORS["Medium"], opacity=0.12, line_width=0)
    risk.add_hrect(y0=0.7, y1=1, fillcolor=BAND_COLORS["High"], opacity=0.12, line_width=0)
    risk.add_trace(go.Scatter(x=s["date"], y=s["failure_probability"], mode="lines+markers",
                              name="Failure probability", line=dict(color="#34495e")))
    for t in f["failure_time"]:
        risk.add_vline(x=t, line=dict(color="red", dash="dash"))
    risk.update_layout(title="Daily failure-risk score (next 7 days)", height=320,
                       yaxis=dict(range=[0, 1], tickformat=".0%"), margin=dict(t=50, b=20))
    st.plotly_chart(risk, use_container_width=True)

    if len(f):
        st.markdown("**Failure history**")
        st.dataframe(f[["failure_time", "failure_mode"]], hide_index=True, use_container_width=True)


# -------------------------------------------------------- page: what-if -----
@st.cache_data
def healthy_reference(eq_type):
    """A typical healthy day for this type: median of the lower-risk half of
    days that were NOT followed by a failure."""
    healthy = scored[(scored["equipment_type"] == eq_type) & (scored[LABEL] == 0)]
    calm = healthy[healthy["failure_probability"] <= healthy["failure_probability"].median()]
    return calm[FEATURE_COLUMNS].median()


def build_whatif_row(eq_type, values, days_since_maint, days_since_fail):
    """Turn slider values into a model feature row.

    Start from a typical *healthy* day for this equipment type, then pretend the
    last 24 hours averaged the slider values. The 7-day window is assumed to be
    halfway between healthy and current, and the slope is the change vs. healthy.
    """
    row = healthy_reference(eq_type).copy()
    for s, v in values.items():
        std24 = row[f"{s}_std_24h"]
        row[f"{s}_mean_24h"] = v
        row[f"{s}_max_24h"] = v + 2 * std24
        new_7d = (row[f"{s}_mean_7d"] + v) / 2
        row[f"{s}_max_7d"] = max(row[f"{s}_max_7d"], row[f"{s}_max_24h"])
        # A week that drifted from old level to v has a wider spread
        row[f"{s}_std_7d"] = max(row[f"{s}_std_7d"], abs(v - row[f"{s}_mean_7d"]) / 2)
        if s == "vibration_mm_s":
            row["vibration_slope_3d"] = (v - row[f"{s}_mean_7d"]) / 3
        if s == "temperature_c":
            row["temperature_slope_3d"] = (v - row[f"{s}_mean_7d"]) / 3
        row[f"{s}_mean_7d"] = new_7d
    row["days_since_maintenance"] = days_since_maint
    row["days_since_failure"] = days_since_fail
    return pd.DataFrame([row])


def page_whatif():
    st.subheader("What-If Simulator")
    st.write("Move the sliders to describe the **last 24 hours** of a unit and watch the risk change.")
    eq_type = st.selectbox("Equipment type", list(EQUIPMENT_TYPES))
    base = EQUIPMENT_TYPES[eq_type]
    ref = healthy_reference(eq_type)

    left, right = st.columns([1, 1])
    values = {}
    with left:
        for s in SENSORS:
            mean = base[s][0]
            lo, hi = (0.8, 2.5) if s == "vibration_mm_s" else (0.7, 1.4)
            # Default = the 24h average on a typical healthy day
            default = float(min(max(ref[f"{s}_mean_24h"], mean * lo), mean * hi))
            values[s] = st.slider(SENSOR_LABELS[s], float(round(mean * lo, 2)),
                                  float(round(mean * hi, 2)), round(default, 2), key=f"{eq_type}_{s}")
        dsm = st.slider("Days since last maintenance", 0, 40, int(ref["days_since_maintenance"]))
        dsf = st.slider("Days since last failure", 0, 180, int(ref["days_since_failure"]))
        # Erratic current is an early warning sign, so it gets its own slider
        cur_std = st.slider("Current fluctuation (std over 24h, A)", 0.0,
                            float(round(base["current_amp"][1] * 6, 1)),
                            float(round(ref["current_amp_std_24h"], 1)))

    row = build_whatif_row(eq_type, values, dsm, dsf)
    row["current_amp_std_24h"] = cur_std
    row["current_amp_max_24h"] = values["current_amp"] + 2 * cur_std
    p = float(predict_proba(bundle, row)[0])
    band = risk_band(p)
    with right:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=p * 100, number={"suffix": "%"},
            title={"text": "Failure probability (next 7 days)"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#34495e"},
                   "steps": [{"range": [0, 30], "color": BAND_COLORS["Low"]},
                             {"range": [30, 70], "color": BAND_COLORS["Medium"]},
                             {"range": [70, 100], "color": BAND_COLORS["High"]}]}))
        gauge.update_layout(height=340, margin=dict(t=60, b=10))
        st.plotly_chart(gauge, use_container_width=True)
        msg = f"**{band} risk** — {recommend_action(p)}"
        {"High": st.error, "Medium": st.warning, "Low": st.success}[band](msg)
        st.caption(f"Baseline for a healthy {eq_type}: "
                   + ", ".join(f"{SENSOR_LABELS[s]} ≈ {base[s][0]:g}" for s in SENSORS))


# -------------------------------------------------------- page: insights ----
def page_insights():
    metrics, importance = get_insights()
    st.subheader("Model Insights")
    st.caption(f"Random Forest trained on {metrics['train_period'][0]} → {metrics['train_period'][1]}, "
               f"tested on {metrics['test_period'][0]} → {metrics['test_period'][1]} "
               f"(time-based split, threshold {metrics['threshold']}).")

    cols = st.columns(5)
    for col, key, name in zip(cols, ["recall", "precision", "f1", "roc_auc", "accuracy"],
                              ["Recall", "Precision", "F1", "ROC-AUC", "Accuracy"]):
        col.metric(name, f"{metrics[key]:.2f}")

    left, right = st.columns(2)
    with left:
        cm = metrics["confusion_matrix"]
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        x=["Predicted: OK", "Predicted: Failure"], y=["Actual: OK", "Actual: Failure"],
                        title="Confusion matrix (test set)")
        fig.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        top = importance.head(15).iloc[::-1]
        fig = px.bar(top, x="importance", y="feature", orientation="h",
                     title="Top 15 feature importances")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    (tn, fp), (fn, tp) = cm
    st.markdown(f"""
#### What do these numbers mean?
- **Recall ({metrics['recall']:.2f})** — of all real "failure is coming" days, how many did we flag?
  We caught **{tp}** and missed **{fn}**. *This is the most important metric in maintenance:*
  a missed failure means unplanned downtime.
- **Precision ({metrics['precision']:.2f})** — when we raise an alarm, how often is it real?
  **{fp}** alarms were false — each costs an inspection, which is much cheaper than a breakdown.
- **F1 ({metrics['f1']:.2f})** — a single score balancing precision and recall.
- **ROC-AUC ({metrics['roc_auc']:.2f})** — how well the model *ranks* risky days above safe ones,
  across all thresholds (0.5 = coin flip, 1.0 = perfect).
- **Accuracy ({metrics['accuracy']:.2f})** — share of all predictions that were correct.
  Misleading here: only {metrics['test_positive_rate']:.0%} of test days are positive, so
  "always predict OK" would already look good on accuracy.
- **Feature importance** — which inputs the forest relied on most. Expect vibration and
  temperature trends near the top, because that's how we built the synthetic degradation.
""")


{"Fleet Overview": page_fleet, "Equipment Detail": page_detail,
 "What-If Simulator": page_whatif, "Model Insights": page_insights}[page]()
