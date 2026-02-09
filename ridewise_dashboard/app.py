import io
import json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from typing import Dict, Any


# -----------------------------
# App Config (WCAG-friendly)
# -----------------------------
st.set_page_config(
    page_title="RideWise Churn Dashboard",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal, accessible CSS (avoid low-contrast; improve focus)
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 1.2rem; }
      .stTabs [data-baseweb="tab"] { font-size: 1rem; }
      .small-note { color: rgba(255,255,255,0.75); font-size: 0.9rem; }
      .wcag-card {
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 14px;
        padding: 14px 16px;
        background: rgba(255,255,255,0.03);
      }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Helpers
# -----------------------------
def safe_float(x, default=None):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default


def api_post_json(url: str, payload: Dict[str, Any], timeout=30) -> Dict[str, Any]:
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def api_post_file(url: str, file_bytes: bytes, filename: str, timeout=60) -> Dict[str, Any]:
    files = {"file": (filename, file_bytes, "text/csv")}
    r = requests.post(url, files=files, timeout=timeout)
    r.raise_for_status()
    return r.json()


def ping_health(base_url: str) -> Dict[str, Any]:
    r = requests.get(f"{base_url}/health", timeout=10)
    r.raise_for_status()
    return r.json()


def plot_risk_distribution(df: pd.DataFrame, col: str = "risk_bucket"):
    # WCAG note: also show counts in a table (not color-only)
    counts = df[col].value_counts().reindex(["Low Risk", "Medium Risk", "High Risk"]).fillna(0).astype(int)
    fig = plt.figure(figsize=(6, 4))
    plt.bar(counts.index, counts.values)
    plt.title("Risk Bucket Distribution (Batch Results)")
    plt.xlabel("Risk Bucket")
    plt.ylabel("Count")
    st.pyplot(fig)
    st.dataframe(counts.rename("count").reset_index().rename(columns={"index": "risk_bucket"}), use_container_width=True)


# -----------------------------
# Sidebar: API Config + Status
# -----------------------------
st.sidebar.title("⚙️ Settings")

default_api = "http://127.0.0.1:8000"
API_BASE = st.sidebar.text_input(
    "FastAPI Base URL",
    value=default_api,
    help="Where your FastAPI service is running. Example: http://127.0.0.1:8000"
).rstrip("/")

with st.sidebar:
    st.markdown("### Service Status")
    try:
        health = ping_health(API_BASE)
        st.success(f"Connected ✅  | Threshold: {health.get('business_threshold', 'N/A')}")
        st.caption(f"Features expected: {health.get('n_features_expected', 'N/A')}")
    except Exception as e:
        st.error("Not connected ❌")
        st.caption("Start FastAPI first: python -m uvicorn app.main:app --reload")
        st.caption(f"Error: {e}")

    st.markdown("---")
    st.markdown(
        "<div class='small-note'>Tip: Keep FastAPI running while using this dashboard.</div>",
        unsafe_allow_html=True
    )


# -----------------------------
# Header
# -----------------------------
st.title("🚕 RideWise Churn Prediction Dashboard")
st.write(
    "This dashboard predicts **churn probability** and generate an **action recommendation** "
    "based on **RFMS segment × churn risk**. Supports **single rider prediction** and **batch CSV scoring**."
)

st.markdown(
    "<div class='wcag-card'>"
    "<b>Accessibility:</b> Labels, help text, and readable layouts are used throughout. "
    "Insights are presented with text + tables (not color-only)."
    "</div>",
    unsafe_allow_html=True
)

st.write("")


# -----------------------------
# Top Navigation (Main Bar Tabs)
# -----------------------------
tab_single, tab_batch, tab_about = st.tabs(["🔎 Single Prediction", "📄 Batch (CSV) Scoring", "ℹ️ About & Tips"])


# ==========================================================
# TAB 1: Single Prediction
# ==========================================================
with tab_single:
    st.subheader("Single Rider Prediction")
    st.write("Fill the form, then click **Predict churn**. Missing fields can be left blank (the API imputes them).")

    with st.form("single_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            rfms_segment = st.selectbox(
                "RFMS Segment",
                ["At Risk", "Occasional Riders", "Core Loyal Riders", "High-Value Surge-Tolerant"],
                help="Customer segment from RFMS analysis."
            )
            city = st.text_input("City", value="Lagos", help="City name (must match training categories where possible).")
            loyalty_status = st.selectbox("Loyalty Status", ["Bronze", "Silver", "Gold"], index=0)

        with col2:
            age = st.number_input("Age", min_value=0, max_value=120, value=32, step=1)
            was_referred = st.selectbox("Was Referred", [0, 1], help="1 = referred, 0 = not referred.")
            avg_rating_given = st.number_input("Avg Rating Given", min_value=0.0, max_value=5.0, value=4.7, step=0.1)

        with col3:
            session_after_last_trip_days = st.number_input(
                "Session After Last Trip (days)",
                min_value=0, max_value=365, value=45, step=1,
                help="Behavioral signal: sessions occurring long after last trip may indicate churn risk."
            )
            total_trips = st.number_input("Total Trips", min_value=0, max_value=5000, value=9, step=1)
            total_spent = st.number_input("Total Spent", min_value=0.0, value=120.5, step=1.0)

        st.markdown("#### Optional RFMS & Pricing Signals")
        c4, c5, c6, c7 = st.columns(4)
        with c4:
            rfms_weighted_score = st.number_input("RFMS Weighted Score", min_value=0.0, value=2.4, step=0.1)
        with c5:
            avg_surge = st.number_input("Avg Surge", min_value=1.0, value=1.1, step=0.05)
        with c6:
            avg_fare = st.number_input("Avg Fare (optional)", min_value=0.0, value=0.0, step=0.5)
        with c7:
            conversion_rate = st.number_input("Conversion Rate (optional)", min_value=0.0, max_value=1.0, value=0.2, step=0.01)

        submit = st.form_submit_button("🚀 Predict churn")

    if submit:
        if not API_BASE:
            st.error("API base URL is empty. Set it in the sidebar.")
        else:
            payload = {
                "data": {
                    "RFMS_segment": rfms_segment,
                    "RFMS_weighted_score": rfms_weighted_score,
                    "session_after_last_trip_days": session_after_last_trip_days,
                    "total_spent": total_spent,
                    "total_trips": int(total_trips),
                    "avg_surge": avg_surge,
                    "city": city,
                    "loyalty_status": loyalty_status,
                    "age": int(age),
                    "was_referred": int(was_referred),
                    "avg_rating_given": avg_rating_given,
                    "avg_fare": avg_fare if avg_fare > 0 else None,
                    "conversion_rate": conversion_rate
                }
            }

            try:
                with st.spinner("Calling FastAPI /predict ..."):
                    result = api_post_json(f"{API_BASE}/predict", payload)

                proba = safe_float(result.get("churn_probability"), 0.0)
                pred = int(result.get("churn_prediction", 0))
                risk_bucket = result.get("risk_bucket", "Unknown")
                action = result.get("recommended_action", "No action returned")

                st.success("Prediction completed ✅")

                m1, m2, m3 = st.columns(3)
                m1.metric("Churn Probability", f"{proba:.3f}")
                m2.metric("Churn Prediction", "Churned (1)" if pred == 1 else "Active (0)")
                m3.metric("Risk Bucket", risk_bucket)

                st.markdown("### Recommended Action")
                st.info(action)

                st.markdown("### Raw API Response (for debugging / audit)")
                st.code(json.dumps(result, indent=2), language="json")

            except Exception as e:
                st.error("Prediction failed ❌")
                st.caption(str(e))


# ==========================================================
# TAB 2: Batch CSV
# ==========================================================
with tab_batch:
    st.subheader("Batch Scoring via CSV Upload")
    st.write(
        "Upload a CSV file and score all riders using the FastAPI **/predict_batch** endpoint. "
        "You’ll get probabilities, churn label, risk bucket, RFMS segment, and recommended action."
    )

    # Provide a downloadable sample template hint
    st.markdown("**Tip:** Your CSV may include only some columns; missing ones will be imputed by the model pipeline.")
    st.caption("Recommended columns: RFMS_segment, RFMS_weighted_score, session_after_last_trip_days, total_spent, total_trips, avg_surge, city, loyalty_status, age, was_referred")

    uploaded = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)

    if uploaded is not None:
        try:
            df_preview = pd.read_csv(uploaded)
            st.markdown("### Preview (first 20 rows)")
            st.dataframe(df_preview.head(20), use_container_width=True)

            colA, colB, colC = st.columns([1, 1, 2])
            with colA:
                do_score = st.button("📌 Score this CSV", type="primary")
            with colB:
                st.write("")
            with colC:
                st.caption("Large files may take longer. Start with small batches during testing.")

            if do_score:
                if not API_BASE:
                    st.error("API base URL is empty. Set it in the sidebar.")
                else:
                    uploaded.seek(0)
                    file_bytes = uploaded.read()

                    try:
                        with st.spinner("Uploading to FastAPI /predict_batch ..."):
                            result = api_post_file(f"{API_BASE}/predict_batch", file_bytes, uploaded.name)

                        if result.get("errors"):
                            st.warning("Batch completed with warnings.")
                            st.write(result["errors"])

                        preds = result.get("predictions", [])
                        df_out = pd.DataFrame(preds)

                        st.success(f"Scored ✅  Rows received: {result.get('n_rows_received')} | Rows scored: {result.get('n_rows_scored')}")

                        # Summary metrics
                        if "risk_bucket" in df_out.columns and "churn_prediction" in df_out.columns:
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Total Rows", len(df_out))
                            c2.metric("Predicted Churned", int(df_out["churn_prediction"].sum()))
                            c3.metric("High Risk", int((df_out["risk_bucket"] == "High Risk").sum()))
                            c4.metric("Low Risk", int((df_out["risk_bucket"] == "Low Risk").sum()))

                        st.markdown("### Scored Results")
                        st.dataframe(df_out, use_container_width=True)

                        # Filters (interactive)
                        st.markdown("### Filters")
                        f1, f2 = st.columns(2)
                        with f1:
                            risk_filter = st.multiselect(
                                "Filter by Risk Bucket",
                                options=["Low Risk", "Medium Risk", "High Risk"],
                                default=["Low Risk", "Medium Risk", "High Risk"]
                            )
                        with f2:
                            seg_filter = st.multiselect(
                                "Filter by RFMS Segment",
                                options=sorted(df_out["rfms_segment"].dropna().unique().tolist()) if "rfms_segment" in df_out.columns else [],
                                default=sorted(df_out["rfms_segment"].dropna().unique().tolist()) if "rfms_segment" in df_out.columns else []
                            )

                        df_filtered = df_out.copy()
                        if "risk_bucket" in df_filtered.columns:
                            df_filtered = df_filtered[df_filtered["risk_bucket"].isin(risk_filter)]
                        if "rfms_segment" in df_filtered.columns and seg_filter:
                            df_filtered = df_filtered[df_filtered["rfms_segment"].isin(seg_filter)]

                        st.markdown("### Filtered Results")
                        st.dataframe(df_filtered, use_container_width=True)

                        st.markdown("### Risk Distribution")
                        if "risk_bucket" in df_out.columns:
                            plot_risk_distribution(df_out, "risk_bucket")

                        # Download
                        st.markdown("### Download")
                        csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="⬇️ Download scored CSV",
                            data=csv_bytes,
                            file_name="scored_predictions.csv",
                            mime="text/csv"
                        )

                    except Exception as e:
                        st.error("Batch scoring failed ❌")
                        st.caption(str(e))

        except Exception as e:
            st.error("Could not read the uploaded CSV.")
            st.caption(str(e))


# ==========================================================
# TAB 3: About & Tips
# ==========================================================
with tab_about:
    st.subheader("About this Dashboard")
    st.write(
        "- **Single Prediction**: sends one rider’s features to `/predict`.\n"
        "- **Batch Scoring**: uploads a CSV to `/predict_batch` and returns scored rows.\n"
        "- Outputs include: probability, churn label, risk bucket, RFMS segment, and recommended action."
    )

    st.markdown("### Best Practices for Production")
    st.write(
        "1) Keep FastAPI and Streamlit in separate services for clean deployment.\n"
        "2) Use environment variables for API URL and secrets.\n"
        "3) Add authentication if exposed publicly.\n"
        "4) Log requests (without sensitive data) for monitoring.\n"
        "5) Version your model artifacts and include a model version in `/health`."
    )

    st.markdown("### WCAG Notes")
    st.write(
        "- Avoid relying on color alone to convey risk.\n"
        "- Provide text labels + tables for all charts.\n"
        "- Keep strong contrast and readable spacing.\n"
        "- Ensure keyboard-friendly controls (Streamlit defaults help here)."
    )
