# RideWise – End-to-End Churn Prediction & Retention Intelligence System

## Overview

RideWise is a full end-to-end data science and machine learning project designed to predict rider churn for a ride-hailing platform and translate predictions into concrete, business-ready retention actions.

The project goes beyond traditional modeling to deliver a production-grade churn intelligence system that combines predictive analytics, customer segmentation, explainability, and deployment.

It demonstrates how data science can move from raw data to real-world impact through:
- Robust data auditing and feature engineering
- Business-grounded churn definition
- RFMS (Recency, Frequency, Monetary, Surge) segmentation
- Interpretable churn modeling
- Threshold tuning aligned with business costs
- Model explainability with SHAP
- API-based inference using FastAPI
- Interactive decision dashboards using Streamlit
- Cloud deployment on Render

---

## Business Problem

Customer churn directly affects revenue, customer lifetime value, and operational efficiency in ride-hailing platforms.

The core business objectives of RideWise are to:
1. Identify riders likely to churn before they disengage
2. Understand the behavioral drivers behind churn risk
3. Segment riders by value and engagement patterns
4. Recommend targeted, cost-effective retention actions
5. Operationalize predictions via scalable APIs and dashboards

---

## Data Sources

The project integrates multiple rider-level data sources, including:
- Rider demographic and profile data
- Trip-level transaction history
- App session and engagement behavior
- Pricing dynamics and surge exposure
- Loyalty and referral information

Each dataset was explored independently, validated for integrity, aggregated to the rider level, and joined into a unified modeling table.

---

## Churn Definition

Churn is defined using a business-aligned rule:

A rider is classified as churned if they have not completed a trip in more than 30 days.

This definition was validated through exploratory analysis comparing inactivity distributions between active and churned riders.

To prevent data leakage:
- `days_since_last_trip` is used strictly to define the churn label
- It is explicitly excluded from modeling features

---

## Feature Engineering

Features were engineered to capture rider behavior, value, and engagement patterns.

### Behavioral Features
- Total number of trips
- Weekend and peak-hour usage ratios
- Session frequency and depth
- Time between last session and last trip

### Monetary Features
- Total spend
- Average fare
- Tip behavior
- Surge exposure and tolerance

### Temporal Features
- Rider tenure
- Trip activity span
- Session span and recency metrics

### Categorical Features
- City
- Loyalty status
- RFMS segment

All preprocessing steps are encapsulated in a scikit-learn pipeline to ensure training–inference consistency.

---

## RFMS Customer Segmentation

Traditional RFM segmentation was extended by incorporating surge sensitivity, a critical factor in ride-hailing economics.

### RFMS Dimensions
- Recency: how recently the rider engaged
- Frequency: number of completed trips
- Monetary: total rider spend
- Surge: tolerance to surge pricing

A weighted RFMS score is computed as:

RFMS_weighted_score =
0.30 × Recency  
0.25 × Frequency  
0.25 × Monetary  
0.20 × Surge  

Riders are segmented into four business-relevant groups:
- At Risk
- Occasional Riders
- Core Loyal Riders
- High-Value Surge-Tolerant Riders

This segmentation later informs promotion and retention strategies.

---

## Modeling Approach

### Model Selection

Logistic Regression was selected as the primary model due to:
- Strong predictive performance
- High interpretability
- Stability in production environments
- Compatibility with SHAP explanations
- Ease of threshold tuning for business objectives

### Model Performance

Key evaluation results:
- ROC-AUC approximately 0.996
- High recall for churned riders
- Strong precision–recall trade-off after threshold tuning

The model reliably distinguishes churned from active riders while remaining interpretable.

---

## Threshold Tuning

Instead of relying on the default 0.5 classification threshold, decision thresholds were tuned to reflect business trade-offs.

The selected business threshold:
- 0.35

This threshold prioritizes minimizing missed churners (false negatives) while controlling unnecessary incentive costs (false positives).

---

## Explainability and Trust

### Global Explainability
- Logistic regression coefficients highlight key churn drivers
- SHAP summary plots confirm feature importance and directionality

Top churn drivers include:
- Time between last session and last trip
- Total rider spend
- RFMS weighted score
- Surge exposure
- Engagement depth

### Local Explainability
- SHAP force plots explain individual rider predictions
- Enables transparency and trust for operational stakeholders

---

## Promotion Strategy Mapping

Churn risk predictions are combined with RFMS segments to drive targeted actions.

Examples include:
- High-risk, At-Risk riders: prioritized churn-prevention packages
- Core Loyal riders with medium risk: loyalty reinforcement and service recovery
- High-value, low-risk riders: recognition without discounts
- Occasional, low-risk riders: engagement nudges and feature prompts

This ensures retention spending is both efficient and value-aligned.

---

## Deployment Architecture

The system is fully deployed and production-ready.

### FastAPI Backend
- Endpoints: /health, /predict, /predict_batch
- Loads trained model and preprocessing pipeline
- Returns churn probability, binary prediction, risk bucket, RFMS segment, and recommended action

FastAPI Deployment:
https://ridewise-churn-api.onrender.com

### Streamlit Dashboard
- Interactive single-rider prediction
- Batch CSV upload for bulk scoring
- Risk visualization and action summaries
- Accessible, responsive interface suitable for non-technical users

Streamlit Deployment:
https://ridewise-churn-dashboard.onrender.com

---

## Project Structure

ridewise-churn-deployment/
├── notebooks/
│   ├── 01_data_audit_and_relationships.ipynb
│   ├── 02_churn_definition_and_eda.ipynb
│   └── 03_feature_engineering_and_modeling.ipynb
│
├── ridewise_api/
│   ├── app/
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── utils.py
│   │
│   ├── model/
│   │   ├── churn_model_pipeline.joblib
│   │   └── churn_model_metadata.joblib
│   │
│   ├── requirements.txt
│   ├── Dockerfile
│   └── sample_batch.csv
│
├── ridewise_dashboard/
│   ├── app.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── data/
│       └── riders_trips_session_churned.csv
│
├── .gitignore
└── README.md


---

## Running the Project Locally

This project is organized as two independent services:
- A FastAPI backend for churn prediction and business logic
- A Streamlit dashboard for interactive exploration and decision support

Both can be run locally for development and testing.

---

### Running the FastAPI Service
Navigate to the API directory:
cd ridewise_api
pip install -r requirements.txt
uvicorn app.main:app --reload

### Running the Streamlit Dashboard
cd ridewise_dashboard
pip install -r requirements.txt
streamlit run app.py


---

## Key Outcomes

- Built a complete churn prediction pipeline from raw data to deployment
- Achieved strong predictive performance with interpretable models
- Integrated segmentation with churn risk for actionable decision-making
- Deployed scalable APIs and interactive dashboards
- Delivered a real-world, business-ready data science solution

---

## Future Improvements

- Model monitoring and drift detection
- Automated retraining pipelines
- Real-time streaming feature ingestion
- Lifetime value prediction
- Experiment tracking and observability

---

## Author

Developed by Mayowa Olokun  
Data Scientist | Data Analyst | Machine Learning | Analytics Engineering  

This project demonstrates end-to-end applied data science, bridging analytics, modeling, explainability, and production deployment.

