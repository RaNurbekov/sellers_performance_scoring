Seller Performance Scoring — Brazilian E-Commerce (Olist)
ML system that scores and segments 1,271 sellers into performance tiers using unsupervised learning — inspired by credit risk underwriting.

## Live Demo
https://sellersperformancescoring-25qhxrwczrek74nuaugv7x.streamlit.app/

## Problem
E-commerce platforms need to identify high-risk sellers early — 
those with poor reviews, late deliveries, and high cancellation rates 
damage customer trust and platform revenue.

Solution
Segmented sellers into 4 performance grades using KMeans clustering on behavioral metrics extracted from PostgreSQL via feature engineering.

Results
Grade	Segment	Count	Avg Revenue	Avg Review	On-time %
A	Premium	349	R$ 27,695	4.08 ⭐	91.4%
B	Stable	363	R$ 9,064	4.03 ⭐	89.2%
C	Reliable	456	R$ 2,214	4.26 ⭐	93.5%
D	At Risk	103	R$ 4,887	3.36 ⭐	70.9%
Tech Stack
PostgreSQL — feature engineering via SQL
pandas + NumPy — data processing
scikit-learn — KMeans + StandardScaler
SHAP — explainability via surrogate RandomForest
Streamlit + Plotly — interactive dashboard
ML Pipeline
PostgreSQL → pandas → log transform → StandardScaler → KMeans(k=4) → SHAP → Streamlit

Feature Engineering (SQL → pandas)
Feature	Description
total_orders	Number of delivered orders
total_revenue	Sum of price + freight
avg_review_score	Mean customer rating
avg_delivery_days	Average days from purchase to delivery
cancellation_rate	% of cancelled orders
on_time_rate	% of orders delivered before estimated date
Key Insights
Grade C (Reliable) sellers have the best review score (4.26) and fastest delivery (9.6 days) despite low revenue — high growth potential
Grade D (At Risk) sellers show 16.7 avg delivery days and only 70.9% on-time rate — immediate intervention needed
SHAP analysis reveals cancellation_rate and on_time_rate are the strongest predictors of At Risk classification
Project Structure
seller-performance-scoring/

├── data/

│ └── seller_scores_final.csv

├── models/

│ ├── kmeans_seller_scoring.pkl

│ └── scaler_seller_scoring.pkl

├── notebooks/

│ └── seller_scoring.ipynb

├── app.py

├── requirements.txt

└── README.md

Run Locally
git clone https://github.com/твой_username/seller-performance-scoring
cd seller-performance-scoring
pip install -r requirements.txt
streamlit run app.py
```

## Dataset
[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — 
100k orders from 2016 to 2018 across multiple marketplaces in Brazil.
