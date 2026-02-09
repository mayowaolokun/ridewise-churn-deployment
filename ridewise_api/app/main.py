from fastapi import FastAPI, UploadFile, File
import pandas as pd
import io

from app.schemas import PredictRequest, PredictResponse, BatchPredictResponse
from app.utils import (
    load_artifacts,
    align_payload_to_features,
    get_risk_bucket,
    recommend_action,
    score_dataframe
)

app = FastAPI(title="RideWise Churn Prediction API", version="1.0.0")

# Load model once when the API starts
model, BUSINESS_THRESHOLD, FEATURE_COLS = load_artifacts()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "n_features_expected": len(FEATURE_COLS),
        "business_threshold": BUSINESS_THRESHOLD
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # 1) Align request payload to training schema
    X = align_payload_to_features(req.data, FEATURE_COLS)

    # 2) Predict churn probability
    proba = float(model.predict_proba(X)[:, 1][0])

    # 3) Apply business threshold
    pred = int(proba >= BUSINESS_THRESHOLD)

    # 4) Risk bucket
    risk = get_risk_bucket(prob=proba, thr_low=BUSINESS_THRESHOLD, thr_mid=0.65)

    # 5) Segment and recommended action (from incoming request)
    rfms_segment = req.data.get("RFMS_segment", "Unknown")
    action = recommend_action(rfms_segment, risk)

    return PredictResponse(
        churn_probability=proba,
        churn_prediction=pred,
        business_threshold=BUSINESS_THRESHOLD,
        risk_bucket=risk,
        rfms_segment=rfms_segment,
        recommended_action=action
    )


@app.post("/predict_batch", response_model=BatchPredictResponse)
async def predict_batch(file: UploadFile = File(...)):
    """
    Upload a CSV file containing multiple riders.
    Returns churn probability + label + risk + recommended action for each row.
    """
    # 1) Read uploaded file into a pandas dataframe
    try:
        contents = await file.read()
        df_in = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        return BatchPredictResponse(
            n_rows_received=0,
            n_rows_scored=0,
            errors=[f"Could not read CSV: {str(e)}"],
            predictions=[]
        )

    n_received = len(df_in)

    # 2) Score dataframe using the trained pipeline
    try:
        df_out = score_dataframe(df_in, model, FEATURE_COLS, BUSINESS_THRESHOLD)
    except Exception as e:
        return BatchPredictResponse(
            n_rows_received=n_received,
            n_rows_scored=0,
            errors=[f"Scoring failed: {str(e)}"],
            predictions=[]
        )

    # 3) Convert to JSON-friendly output
    preds = df_out.to_dict(orient="records")

    return BatchPredictResponse(
        n_rows_received=n_received,
        n_rows_scored=len(df_out),
        errors=None,
        predictions=preds
    )
