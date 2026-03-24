# Merchant Churn Prediction Model

A machine learning pipeline that identifies merchants at risk of churning, explains the drivers behind each prediction using SHAP, and ranks active merchants by churn probability for proactive intervention.

---

## Overview

The model pulls transaction history and churn labels from a PostgreSQL database, engineers 26 behavioral features per merchant, and trains two classifiers (Random Forest and XGBoost) with automated hyperparameter tuning. The better-performing model is used to score all merchants and generate explainability outputs at the individual merchant level.

---

## Features

**26 features across four RFM groups:**

| Group | Features |
|---|---|
| Recency | Days since last transaction, recency ratio (normalised by merchant's own rhythm), max gap, avg inter-transaction days |
| Frequency | Transaction counts for 7d / 30d / 90d / 180d windows, count trend (30d vs prior 30d), tx frequency, active days |
| Monetary | Windowed volumes (7d / 30d / 90d / 180d), volume trend, activity ratios (30d/90d, 90d/180d), amount coefficient of variation, amount per active day, rolling volatility |
| Tenure | Tenure days, total transaction amount, average transaction amount |

---

## Pipeline

```
PostgreSQL (RDS)
    │
    ├── merchant_churn_status      ← churn labels + merchant metadata
    └── consolidated_transactions  ← raw transaction history
            │
            ▼
    Feature Engineering (26 features)
            │
            ▼
    Hyperparameter Search
    ├── Random Forest  (RandomizedSearchCV, 20 iter, 5-fold CV)
    └── XGBoost        (RandomizedSearchCV, 20 iter, 5-fold CV)
            │
            ▼
    Model Evaluation
    ├── CV ROC AUC (5-fold)
    ├── Test ROC AUC
    ├── Classification report
    ├── Confusion matrix
    └── ROC curve
            │
            ▼
    Risk Scoring (best model)
    ├── 🔴 High    35–100% churn probability
    ├── 🟡 Medium  15–35%
    └── 🟢 Low      0–15%
            │
            ▼
    SHAP Analysis
    ├── Global beeswarm plot
    ├── Mean |SHAP| bar chart
    ├── top_risk_drivers column per merchant
    └── Waterfall plots — top 5 at-risk merchants
```

---

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/Carlynn16/churn-model.git
cd churn-model
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install numpy pandas matplotlib seaborn sqlalchemy psycopg2-binary scikit-learn xgboost shap python-dotenv
```

**4. Configure credentials**

```bash
cp .env.example .env
```

Edit `.env` with your database connection details:

```env
DB_USERNAME=your_username
DB_PASSWORD=your_password
DB_HOST=your_rds_host
```

**5. Run**

```bash
python churn_model.py
```

---

## Output

| Output | Description |
|---|---|
| Console | CV AUC, test AUC, classification report, best hyperparameters |
| Confusion matrix | Actual vs predicted for each model |
| ROC curve | With AUC for each model |
| Feature importance | Top 10 features per model |
| SHAP beeswarm | Feature impact distribution across all merchants |
| SHAP bar chart | Mean \|SHAP\| ranking |
| Waterfall plots | Individual explanations for the top 5 at-risk merchants |
| Risk table | All merchants ranked by churn probability with `top_risk_drivers` |

---

## Tech Stack

- **ML:** scikit-learn, XGBoost
- **Explainability:** SHAP
- **Data:** pandas, NumPy, SQLAlchemy, psycopg2
- **Visualisation:** matplotlib, seaborn
- **Database:** PostgreSQL (AWS RDS)
