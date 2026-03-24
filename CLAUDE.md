# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup & Running

```bash
# Install dependencies (none are pinned — install latest compatible versions)
pip install numpy pandas matplotlib seaborn sqlalchemy psycopg2-binary scikit-learn xgboost shap python-dotenv

# Copy the env template and fill in credentials
cp .env.example .env   # then edit .env

# Run the model
python churn_model.py
```

Credentials are loaded from a `.env` file via `python-dotenv`. Required keys: `DB_USERNAME`, `DB_PASSWORD`, `DB_HOST`. The `.env` file is git-ignored.

There is no build step, test suite, or linter configured. The entire project is a single script.

## Architecture

`churn_model.py` is a self-contained merchant churn prediction pipeline with these stages:

1. **Data ingestion** — Connects to an AWS RDS PostgreSQL instance, pulls from `merchant_churn_status` (merchant profiles + churn labels) and `consolidated_transactions` (transaction history), and merges on merchant location ID.

2. **Feature engineering** — Constructs 26 features per merchant across four groups:
   - *Recency*: days since last tx, `recency_ratio` (silence normalised by the merchant's own inter-tx rhythm), `max_gap_days`, `avg_inter_tx_days`
   - *Frequency*: tx counts for 7d / 30d / 90d / 180d windows, count trend (30d vs prior 30d), `tx_frequency`, `active_days`
   - *Monetary*: windowed volumes (7d / 30d / 90d / 180d), volume trend, activity ratios (30d/90d, 90d/180d), `amount_cv` (coefficient of variation), `amount_per_active_day`, `rolling_volatility`
   - *Tenure*: `tenure_days`, `total_tx_amount`, `avg_tx_amount`

3. **Hyperparameter search** — `RandomizedSearchCV` (20 iterations, 5-fold stratified CV, scoring=roc_auc) is run independently for each model before evaluation:
   - Random Forest: searches `n_estimators`, `max_depth`, `min_samples_leaf`, `min_samples_split`, `max_features`
   - XGBoost: searches `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_alpha`, `gamma`

4. **Evaluation** — Prints CV ROC AUC, test ROC AUC, classification reports, confusion matrices, and ROC curves; shows top-10 feature importances for each model.

5. **Risk scoring** — Applies the better-performing model to score all merchants into risk tiers (Low: 0–15%, Medium: 15–35%, High: 35–100%) and surfaces the top 20 at-risk active merchants.

6. **SHAP analysis** — Explains model predictions using `shap.TreeExplainer`:
   - Global beeswarm plot (feature impact distribution across all merchants)
   - Global bar chart (mean |SHAP| ranking)
   - `top_risk_drivers` column in the risk table — top 3 SHAP contributors per merchant with direction and magnitude
   - Waterfall plot for each of the top 5 at-risk active merchants

## Key Customization Points

| What to change | Where |
|---|---|
| Feature set | `feature_cols` list (~line 100) |
| Hyperparameter search space / iterations | `rf_search` / `xgb_search` `RandomizedSearchCV` calls (~lines 155–185) |
| Risk tier thresholds | `pd.cut()` calls in sections 7 & 8 |
| Number of SHAP waterfall plots | `active_at_risk.head(5)` in section 9 |
| Database connection | `.env` file (`DB_USERNAME`, `DB_PASSWORD`, `DB_HOST`) |

## Security Warning

Database credentials are hardcoded in plaintext at the top of `churn_model.py`. These should be moved to environment variables before committing or sharing this file.
