from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class PredictRequest(BaseModel):
    data: Dict[str, Any] = Field(..., description="Feature dictionary for a single rider")


class PredictResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    business_threshold: float
    risk_bucket: str
    rfms_segment: str
    recommended_action: str


class BatchPredictResponse(BaseModel):
    n_rows_received: int
    n_rows_scored: int
    errors: Optional[List[str]] = None
    predictions: List[Dict[str, Any]]
