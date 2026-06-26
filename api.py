from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import joblib
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(
    title="Seller Performance Scoring API",
    description="ML API for scoring sellers based on behavioral metrics",
    version="1.0.0"
)

# ── load data ─────────────────────────────────────────────────
df = pd.read_csv('data/seller_scores_final.csv')

FEATURES = ['total_orders', 'total_revenue', 'avg_review_score',
            'avg_delivery_days', 'cancellation_rate', 'on_time_rate']

GRADE_MAP   = {3: 'A', 2: 'B', 1: 'C', 0: 'D'}
SEGMENT_MAP = {'A': 'Premium', 'B': 'Stable', 'C': 'Reliable', 'D': 'At Risk'}

# ── fit scaler + surrogate RF прямо здесь ─────────────────────
X = df[FEATURES].copy()
for col in ['total_orders', 'total_revenue']:
    X[col] = np.log1p(X[col])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, df['cluster'])

explainer = shap.TreeExplainer(rf)

# ── endpoints ─────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "Seller Performance Scoring API",
        "version": "1.0.0",
        "endpoints": ["/score/{seller_id}", "/sellers", "/stats"]
    }

@app.get("/sellers")
def get_sellers():
    return {
        "total": len(df),
        "sellers": df[['seller_id', 'seller_city', 'seller_state',
                        'grade', 'segment', 'total_revenue']
                      ].to_dict(orient='records')
    }

@app.get("/stats")
def get_stats():
    stats = df.groupby(['grade', 'segment']).agg(
        count       =('seller_id', 'count'),
        avg_revenue =('total_revenue', 'mean'),
        avg_review  =('avg_review_score', 'mean'),
        avg_delivery=('avg_delivery_days', 'mean'),
        avg_ontime  =('on_time_rate', 'mean')
    ).round(2).reset_index()
    return {"grades": stats.to_dict(orient='records')}

@app.get("/score/{seller_id}")
def get_seller_score(seller_id: str):
    seller = df[df['seller_id'] == seller_id]
    if seller.empty:
        raise HTTPException(status_code=404, detail=f"Seller {seller_id} not found")

    idx = seller.index[0]
    positional_idx = df.index.get_loc(idx)
    seller = seller.iloc[0]
    cluster_id = int(seller['cluster'])

    shap_vals = explainer.shap_values(X_scaled[positional_idx].reshape(1, -1))
    shap_for_cluster = shap_vals[0][:, cluster_id].tolist()

    feature_importance = sorted(
        [{"feature": f, "shap_value": round(s, 4)}
         for f, s in zip(FEATURES, shap_for_cluster)],
        key=lambda x: abs(x['shap_value']),
        reverse=True
    )

    return {
        "seller_id":    seller_id,
        "seller_city":  seller['seller_city'],
        "seller_state": seller['seller_state'],
        "grade":        seller['grade'],
        "segment":      seller['segment'],
        "metrics": {
            "total_orders":      int(seller['total_orders']),
            "total_revenue":     float(seller['total_revenue']),
            "avg_review_score":  float(seller['avg_review_score']),
            "avg_delivery_days": float(seller['avg_delivery_days']),
            "cancellation_rate": float(seller['cancellation_rate']),
            "on_time_rate":      float(seller['on_time_rate'])
        },
        "explanation": {
            "top_factors": feature_importance[:3],
            "all_factors": feature_importance
        }
    }