import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings('ignore')
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

USERNAME = os.getenv("DB_USERNAME")
PASSWORD = os.getenv("DB_PASSWORD")
HOST     = os.getenv("DB_HOST")

engine = create_engine(
    f"postgresql+psycopg2://{USERNAME}:{PASSWORD}@{HOST}:5432/postgres"
)
df1 = pd.read_sql("select * from merchant_churn_status", engine)
df2 = pd.read_sql("select * from consolidated_transactions", engine)
df = df2.merge(
    df1,
    left_on="LCTN_ID",
    right_on="merchant_id",
    how="left"
)
df = df[
    (df["churned_date"].isna()) |
    (df["TRAN_DT"] <= df["churned_date"])
]

df = df[df["churn_status"].notna()]
reference_date = df["TRAN_DT"].max()

# ── 1. Global features ────────────────────────────────────────────
merchant_features = df.groupby("LCTN_ID").agg(
    total_tx_amount=("TRAN_AM", "sum"),
    avg_tx_amount=("TRAN_AM", "mean"),
    tx_count=("TRAN_AM", "count"),
    first_tx=("TRAN_DT", "min"),
    last_tx=("TRAN_DT", "max")
).reset_index()

merchant_features["days_since_last_tx"] = (reference_date - merchant_features["last_tx"]).dt.days
merchant_features["tenure_days"] = (reference_date - merchant_features["first_tx"]).dt.days

# ── 2. Time windows ───────────────────────────────────────────────
df["days_before_ref"] = (reference_date - df["TRAN_DT"]).dt.days

last_7   = df[df["days_before_ref"] <= 7]
last_30  = df[df["days_before_ref"] <= 30]
last_90  = df[df["days_before_ref"] <= 90]
last_180 = df[df["days_before_ref"] <= 180]
prev_30  = df[(df["days_before_ref"] > 30) & (df["days_before_ref"] <= 60)]

vol7       = last_7.groupby("LCTN_ID")["TRAN_AM"].sum().rename("vol_last_7d")
vol30      = last_30.groupby("LCTN_ID")["TRAN_AM"].sum().rename("vol_last_30d")
vol90      = last_90.groupby("LCTN_ID")["TRAN_AM"].sum().rename("vol_last_90d")
vol180     = last_180.groupby("LCTN_ID")["TRAN_AM"].sum().rename("vol_last_180d")
cnt7       = last_7.groupby("LCTN_ID")["TRAN_AM"].count().rename("tx_count_7d")
cnt30      = last_30.groupby("LCTN_ID")["TRAN_AM"].count().rename("tx_count_30d")
cnt90      = last_90.groupby("LCTN_ID")["TRAN_AM"].count().rename("tx_count_90d")
cnt180     = last_180.groupby("LCTN_ID")["TRAN_AM"].count().rename("tx_count_180d")
vol_prev30 = prev_30.groupby("LCTN_ID")["TRAN_AM"].sum().rename("vol_prev_30d")
cnt_prev30 = prev_30.groupby("LCTN_ID")["TRAN_AM"].count().rename("tx_count_prev_30d")

# ── 3. Merge + fillna 0 ───────────────────────────────────────────
for serie in [vol7, vol30, vol90, vol180, cnt7, cnt30, cnt90, cnt180, vol_prev30, cnt_prev30]:
    merchant_features = merchant_features.merge(serie, on="LCTN_ID", how="left")

fill_cols = [
    "vol_last_7d", "vol_last_30d", "vol_last_90d", "vol_last_180d",
    "tx_count_7d", "tx_count_30d", "tx_count_90d", "tx_count_180d",
    "vol_prev_30d", "tx_count_prev_30d",
]
merchant_features[fill_cols] = merchant_features[fill_cols].fillna(0)

# ── 4. Inter-transaction gap features ────────────────────────────
def _max_gap(dates):
    s = dates.sort_values()
    return s.diff().dt.days.max() if len(s) >= 2 else np.nan

def _avg_gap(dates):
    s = dates.sort_values()
    return s.diff().dt.days.mean() if len(s) >= 2 else np.nan

tx_gaps = df.groupby("LCTN_ID")["TRAN_DT"].agg(
    max_gap_days=_max_gap,
    avg_inter_tx_days=_avg_gap,
)
merchant_features = merchant_features.merge(tx_gaps, on="LCTN_ID", how="left")

# ── 5. Frequency & volatility ─────────────────────────────────────
freq       = df.groupby("LCTN_ID")["TRAN_DT"].nunique().rename("active_days")
volatility = df.groupby("LCTN_ID")["TRAN_AM"].std().rename("rolling_volatility")

merchant_features = merchant_features.merge(freq, on="LCTN_ID", how="left")
merchant_features = merchant_features.merge(volatility, on="LCTN_ID", how="left")

# ── 6. Derived features ───────────────────────────────────────────
# Volume / count trends (30d vs previous 30d)
merchant_features["volume_trend_30d"]   = merchant_features["vol_last_30d"]   - merchant_features["vol_prev_30d"]
merchant_features["tx_count_trend_30d"] = merchant_features["tx_count_30d"]   - merchant_features["tx_count_prev_30d"]

# Activity ratios across windows
merchant_features["ratio_30d_90d"]  = merchant_features["vol_last_30d"]  / (merchant_features["vol_last_90d"]  + 1)
merchant_features["ratio_90d_180d"] = merchant_features["vol_last_90d"]  / (merchant_features["vol_last_180d"] + 1)

# Normalised frequency (transactions per day of tenure)
merchant_features["tx_frequency"] = merchant_features["tx_count"] / (merchant_features["tenure_days"] + 1)

# Recency in units of the merchant's own inter-transaction rhythm
# High value = merchant has gone silent for many multiples of their usual gap
merchant_features["recency_ratio"] = (
    merchant_features["days_since_last_tx"] / (merchant_features["avg_inter_tx_days"] + 1)
)

# Coefficient of variation of transaction amounts (spending stability)
merchant_features["amount_cv"] = (
    merchant_features["rolling_volatility"] / (merchant_features["avg_tx_amount"].abs() + 1)
)

# Revenue intensity per active day
merchant_features["amount_per_active_day"] = (
    merchant_features["total_tx_amount"] / (merchant_features["active_days"] + 1)
)

# ── 7. Churn label ────────────────────────────────────────────────
merchant_features = merchant_features.merge(
    df1[["merchant_id", "churn_status"]],
    left_on="LCTN_ID",
    right_on="merchant_id",
    how="left"
)
merchant_features["label"] = (merchant_features["churn_status"] == "churn").astype(int)

from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, train_test_split, RandomizedSearchCV
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from xgboost import XGBClassifier

# ══════════════════════════════════════════════════════════════════
# 1. FEATURE PREPARATION
# ══════════════════════════════════════════════════════════════════

feature_cols = [
    # Recency
    "days_since_last_tx",
    "recency_ratio",
    "max_gap_days",
    "avg_inter_tx_days",
    # Frequency / activity
    "tx_count",
    "tx_frequency",
    "active_days",
    "tx_count_7d",
    "tx_count_30d",
    "tx_count_90d",
    "tx_count_180d",
    "tx_count_trend_30d",
    # Monetary / volume
    "total_tx_amount",
    "avg_tx_amount",
    "amount_cv",
    "amount_per_active_day",
    "vol_last_7d",
    "vol_last_30d",
    "vol_last_90d",
    "vol_last_180d",
    "vol_prev_30d",
    "volume_trend_30d",
    "ratio_30d_90d",
    "ratio_90d_180d",
    "rolling_volatility",
    # Tenure
    "tenure_days",
]

merchant_features["target"] = (merchant_features["churn_status"] == "churn").astype(int)

X = merchant_features[feature_cols].fillna(0)
y = merchant_features["target"]

print("Dataset shape:", X.shape)
print("Churn distribution:\n", y.value_counts())
print(f"Churn rate: {y.mean():.1%}")

# ══════════════════════════════════════════════════════════════════
# 2. TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ══════════════════════════════════════════════════════════════════
# 3. HYPERPARAMETER SEARCH  (RandomizedSearchCV, 5-fold stratified)
# ══════════════════════════════════════════════════════════════════

churn_ratio = (y == 0).sum() / (y == 1).sum()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nRunning hyperparameter search — this may take a few minutes...")

rf_search = RandomizedSearchCV(
    RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
    param_distributions={
        "n_estimators":      [200, 300, 500],
        "max_depth":         [4, 6, 8, 10, None],
        "min_samples_leaf":  [2, 5, 10],
        "min_samples_split": [5, 10, 20],
        "max_features":      ["sqrt", 0.4, 0.6],
    },
    n_iter=20, scoring="roc_auc", cv=cv,
    random_state=42, n_jobs=-1, verbose=1,
)
rf_search.fit(X_train, y_train)
print(f"Best RF params:  {rf_search.best_params_}")
print(f"Best RF CV AUC:  {rf_search.best_score_:.4f}")

xgb_search = RandomizedSearchCV(
    XGBClassifier(
        scale_pos_weight=churn_ratio,
        random_state=42,
        eval_metric="auc",
        verbosity=0,
    ),
    param_distributions={
        "n_estimators":     [200, 300, 500],
        "max_depth":        [3, 4, 5, 6],
        "learning_rate":    [0.01, 0.03, 0.05, 0.1],
        "subsample":        [0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.7, 0.8],
        "min_child_weight": [3, 5, 10],
        "reg_alpha":        [0, 0.1, 0.5],
        "gamma":            [0, 0.1, 0.3],
    },
    n_iter=20, scoring="roc_auc", cv=cv,
    random_state=42, n_jobs=-1, verbose=1,
)
xgb_search.fit(X_train, y_train)
print(f"Best XGB params: {xgb_search.best_params_}")
print(f"Best XGB CV AUC: {xgb_search.best_score_:.4f}")

models_trained = {
    "Random Forest": rf_search.best_estimator_,
    "XGBoost":       xgb_search.best_estimator_,
}

# ══════════════════════════════════════════════════════════════════
# 4. EVALUATION — CROSS VALIDATION + TEST SET
# ══════════════════════════════════════════════════════════════════

results = {}

for name, model in models_trained.items():
    print(f"\n{'═'*55}")
    print(f"  {name}")
    print(f"{'═'*55}")

    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"\n📊 Cross-Validation ROC AUC (5-fold):")
    print(f"   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"   Folds: {[round(s,4) for s in cv_scores]}")

    pred  = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    print(f"\n📋 Classification Report (test set):")
    print(classification_report(y_test, pred, target_names=["Active", "Churn"]))
    print(f"🎯 ROC AUC (test set): {roc_auc_score(y_test, proba):.4f}")

    results[name] = {
        "model":   model,
        "proba":   proba,
        "pred":    pred,
        "cv_mean": cv_scores.mean(),
        "cv_std":  cv_scores.std(),
        "auc":     roc_auc_score(y_test, proba),
    }

# ══════════════════════════════════════════════════════════════════
# 5. VISUALISATIONS
# ══════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Churn Model Comparison", fontsize=16, fontweight="bold", y=1.01)

colors = {"Random Forest": "#2196F3", "XGBoost": "#FF5722"}

for idx, (name, res) in enumerate(results.items()):

    ax = axes[idx][0]
    cm = confusion_matrix(y_test, res["pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Active", "Churn"],
                yticklabels=["Active", "Churn"])
    ax.set_title(f"{name}\nConfusion Matrix")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")

    ax = axes[idx][1]
    fpr, tpr, _ = roc_curve(y_test, res["proba"])
    ax.plot(fpr, tpr, color=colors[name], lw=2, label=f'AUC = {res["auc"]:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.fill_between(fpr, tpr, alpha=0.1, color=colors[name])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{name}\nROC Curve")
    ax.legend(loc="lower right")

    ax = axes[idx][2]
    importance = pd.Series(res["model"].feature_importances_, index=feature_cols)
    importance = importance.sort_values(ascending=True).tail(10)
    importance.plot(kind="barh", ax=ax, color=colors[name], alpha=0.8)
    ax.set_title(f"{name}\nTop 10 Feature Importance")
    ax.set_xlabel("Importance")

plt.tight_layout()
plt.show()

# ══════════════════════════════════════════════════════════════════
# 6. SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════

print("\n" + "═"*55)
print("  FINAL SUMMARY")
print("═"*55)
summary = pd.DataFrame({
    name: {
        "CV AUC (mean)": f"{res['cv_mean']:.4f}",
        "CV AUC (±std)": f"{res['cv_std']:.4f}",
        "Test AUC":      f"{res['auc']:.4f}",
    }
    for name, res in results.items()
})
print(summary.to_string())

# ══════════════════════════════════════════════════════════════════
# 7. CHURN RISK TABLE
# ══════════════════════════════════════════════════════════════════

best_model_name = max(results, key=lambda x: results[x]["auc"])
best_model      = results[best_model_name]["model"]
best_proba      = best_model.predict_proba(X)[:, 1]

risk_table = merchant_features[["LCTN_ID", "merchant_id", "churn_status"]].copy()
risk_table["churn_probability"] = best_proba
risk_table["risk_tier"] = pd.cut(
    best_proba,
    bins=[0, 0.15, 0.35, 1.0],
    labels=["🟢 Low", "🟡 Medium", "🔴 High"]
)

active_at_risk = (
    risk_table[risk_table["churn_status"] == "active"]
    .merge(df1[["merchant_id", "merchant_name"]], on="merchant_id", how="left")
    .sort_values("churn_probability", ascending=False)
    .reset_index(drop=True)
)[["merchant_name", "LCTN_ID", "churn_probability", "risk_tier"]]

print(f"\n🏆 Best model: {best_model_name} (AUC: {results[best_model_name]['auc']:.4f})")
print(f"\n🔴 Active merchants at HIGH risk:   {(active_at_risk['risk_tier'] == '🔴 High').sum()}")
print(f"🟡 Active merchants at MEDIUM risk: {(active_at_risk['risk_tier'] == '🟡 Medium').sum()}")
print(f"🟢 Active merchants at LOW risk:    {(active_at_risk['risk_tier'] == '🟢 Low').sum()}")

# ══════════════════════════════════════════════════════════════════
# 8. ALL MERCHANTS RISK TABLE
# ══════════════════════════════════════════════════════════════════

all_merchants_risk = (
    merchant_features[["LCTN_ID", "merchant_id", "churn_status"]].copy()
    .assign(churn_probability=best_proba)
)
all_merchants_risk["risk_tier"] = pd.cut(
    best_proba,
    bins=[0, 0.15, 0.35, 1.0],
    labels=["🟢 Low", "🟡 Medium", "🔴 High"]
)
all_merchants_risk = (
    all_merchants_risk
    .merge(df1[["merchant_id", "merchant_name"]], on="merchant_id", how="left")
    [["merchant_name", "LCTN_ID", "churn_status", "churn_probability", "risk_tier"]]
    .sort_values("churn_probability", ascending=False)
    .reset_index(drop=True)
)

# ══════════════════════════════════════════════════════════════════
# 9. SHAP ANALYSIS
# ══════════════════════════════════════════════════════════════════

print("\nComputing SHAP values...")
explainer   = shap.TreeExplainer(best_model)
shap_raw    = explainer.shap_values(X)

# RF returns [class0_array, class1_array]; XGBoost returns a 2-D array directly
if isinstance(shap_raw, list):
    sv = shap_raw[1]
elif isinstance(shap_raw, np.ndarray) and shap_raw.ndim == 3:
    sv = shap_raw[:, :, 1]
else:
    sv = shap_raw

# ── Global beeswarm (feature impact overview) ─────────────────────
plt.figure()
shap.summary_plot(sv, X, feature_names=feature_cols, show=False)
plt.title(f"SHAP Feature Impact — {best_model_name}", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# ── Global bar chart (mean |SHAP| ranking) ────────────────────────
plt.figure(figsize=(9, 6))
shap.summary_plot(sv, X, feature_names=feature_cols, plot_type="bar", show=False)
plt.title(f"Mean |SHAP| per Feature — {best_model_name}", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# ── Per-merchant top SHAP drivers ────────────────────────────────
shap_df = pd.DataFrame(sv, columns=feature_cols, index=X.index)
shap_df["LCTN_ID"] = merchant_features["LCTN_ID"].values

def _top_drivers(row, n=3):
    vals = row[feature_cols]
    top  = vals.abs().sort_values(ascending=False).head(n)
    return " | ".join(
        f"{'↑' if row[f] > 0 else '↓'}{f} ({row[f]:+.3f})" for f in top.index
    )

shap_df["top_risk_drivers"] = shap_df.apply(_top_drivers, axis=1)

active_at_risk = active_at_risk.merge(
    shap_df[["LCTN_ID", "top_risk_drivers"]], on="LCTN_ID", how="left"
)

print("\nTop 20 active merchants at risk with SHAP explanations:")
pd.set_option("display.max_colwidth", 120)
print(active_at_risk.head(20).to_string(index=False))
print(all_merchants_risk.to_string())

# ── Waterfall plots for the top 5 at-risk active merchants ───────
ev = explainer.expected_value
base_val = float(ev[1]) if isinstance(ev, (list, np.ndarray)) else float(ev)

shap_explanation = shap.Explanation(
    values=sv,
    base_values=base_val,
    data=X.values,
    feature_names=feature_cols,
)

top5_lctns = active_at_risk.head(5)["LCTN_ID"].tolist()
for lctn_id in top5_lctns:
    mask    = merchant_features["LCTN_ID"] == lctn_id
    row_idx = merchant_features.index[mask][0]
    pos     = X.index.get_loc(row_idx)
    mname   = active_at_risk.loc[active_at_risk["LCTN_ID"] == lctn_id, "merchant_name"].iloc[0]
    prob    = active_at_risk.loc[active_at_risk["LCTN_ID"] == lctn_id, "churn_probability"].iloc[0]

    plt.figure()
    shap.plots.waterfall(shap_explanation[pos], show=False)
    plt.title(f"{mname}  |  Churn probability: {prob:.1%}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()
