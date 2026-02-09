from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import pandas as pd
import numpy as np

# Path to model folder (relative to this file)
MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
PIPELINE_PATH = MODEL_DIR / "churn_model_pipeline.joblib"
METADATA_PATH = MODEL_DIR / "churn_model_metadata.joblib"

def load_artifacts():
    """
    Load trained model pipeline + metadata (threshold + feature columns).
    """
    model = joblib.load(PIPELINE_PATH)
    meta = joblib.load(METADATA_PATH)

    threshold = meta.get("business_threshold", 0.35)
    feature_cols = meta.get("feature_columns", [])

    if not feature_cols:
        raise ValueError("Metadata missing 'feature_columns'. Re-save metadata from notebook.")

    return model, float(threshold), feature_cols

def align_payload_to_features(payload: Dict[str, Any], feature_cols: List[str]) -> pd.DataFrame:
    """
    Build a one-row DataFrame with exactly the same columns used in training.
    - Missing columns -> NaN
    - Extra columns -> ignored
    - Column order -> matches training
    """
    row = {col: payload.get(col, np.nan) for col in feature_cols}
    return pd.DataFrame([row], columns=feature_cols)

def get_risk_bucket(prob: float, thr_low: float, thr_mid: float = 0.65) -> str:
    """
    Convert churn probability into Low/Medium/High risk buckets.
    """
    if prob < thr_low:
        return "Low Risk"
    elif prob < thr_mid:
        return "Medium Risk"
    else:
        return "High Risk"

def recommend_action(rfms_segment: str, churn_risk: str) -> str:
    """
    Business rule mapping: RFMS segment × churn risk → recommended action.
    This is NOT machine learning; it's decision logic based on your project strategy.
    """
    # Normalize unknowns
    if rfms_segment is None:
        rfms_segment = "Unknown"

    if churn_risk == "High Risk":
        if rfms_segment == "At Risk":
            return "Highest priority: churn-prevention package (credits + surge relief + service recovery)"
        if rfms_segment == "Core Loyal Riders":
            return "VIP win-back: targeted credit + service recovery + feedback request"
        if rfms_segment == "Occasional Riders":
            return "Reactivation: limited-time discount + convenience messaging"
        if rfms_segment == "High-Value Surge-Tolerant":
            return "White-glove retention: personalized outreach + priority support"

    if churn_risk == "Medium Risk":
        if rfms_segment == "At Risk":
            return "Targeted off-peak discount + education on saving/avoiding surge"
        if rfms_segment == "Core Loyal Riders":
            return "Reinforce loyalty: bonus points + gentle reminder"
        if rfms_segment == "Occasional Riders":
            return "Activation: time-limited offer for next ride"
        if rfms_segment == "High-Value Surge-Tolerant":
            return "Recognition: perks (no discounts) + premium experience"

    # Low Risk
    if rfms_segment == "High-Value Surge-Tolerant":
        return "Reward/recognition (no discounts): perks, priority support, surprise upgrades"
    if rfms_segment == "Core Loyal Riders":
        return "Maintain loyalty: points boosts, referrals, cross-sell bundles"
    if rfms_segment == "Occasional Riders":
        return "Engagement nudges: seasonal campaigns, feature prompts"
    if rfms_segment == "At Risk":
        return "Monitor: low-cost reminders + reduce friction (payments/app UX)"

    return "No action rule defined"

def score_dataframe(df_in: pd.DataFrame, model, feature_cols: list, business_threshold: float) -> pd.DataFrame:
    """
    Score a dataframe of riders.
    - Align to feature_cols
    - Predict probability + label
    - Add risk bucket + recommended action
    """
    # Ensure all required cols exist; missing -> NaN
    for col in feature_cols:
        if col not in df_in.columns:
            df_in[col] = np.nan

    # Keep only the expected columns and in correct order
    X_batch = df_in[feature_cols].copy()

    # Predict
    proba = model.predict_proba(X_batch)[:, 1]
    pred = (proba >= business_threshold).astype(int)

    # Risk bucket
    risk = [get_risk_bucket(float(p), thr_low=business_threshold, thr_mid=0.65) for p in proba]

    # Segment
    rfms_seg = df_in.get("RFMS_segment", pd.Series(["Unknown"] * len(df_in)))

    # Recommended action
    actions = [recommend_action(str(seg), r) for seg, r in zip(rfms_seg, risk)]

    # Attach outputs
    out = df_in.copy()
    out["churn_probability"] = proba.astype(float)
    out["churn_prediction"] = pred.astype(int)
    out["business_threshold"] = business_threshold
    out["risk_bucket"] = risk
    out["rfms_segment"] = rfms_seg.astype(str)
    out["recommended_action"] = actions

    return out
