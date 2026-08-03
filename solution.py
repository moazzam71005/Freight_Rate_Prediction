"""
Freight Rate Prediction — Solution
Trains on train-test.csv (Jan–Oct 2025), predicts validation.csv (Nov–Dec 2025),
and fills december-chart-inputs.csv.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent
TRAIN_PATH = BASE / "train-test.csv"
VAL_PATH = BASE / "validation.csv"
TEMPLATE_PATH = BASE / "validation-predictions-template.csv"
DECEMBER_PATH = BASE / "december-chart-inputs.csv"
PREDICTIONS_OUT = BASE / "validation_predictions.csv"
EDA_DIR = BASE / "eda_plots"
EDA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("=" * 60)
print("Loading data ...")
train_df = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
val_df   = pd.read_csv(VAL_PATH,   parse_dates=["date"])
dec_df   = pd.read_csv(DECEMBER_PATH, parse_dates=["date"])

print(f"  Train shape : {train_df.shape}")
print(f"  Val shape   : {val_df.shape}")
print(f"  Dec shape   : {dec_df.shape}")

# ---------------------------------------------------------------------------
# 2. EDA
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("EDA ...")

# 2a. Basic stats
print("\n--- posted_rate stats ---")
print(train_df["posted_rate"].describe())

# 2b. Rate distribution (raw + log)
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].hist(train_df["posted_rate"], bins=80, color="#064A56", edgecolor="white", linewidth=0.3)
axes[0].set_title("posted_rate distribution (raw)")
axes[0].set_xlabel("Rate ($)")
axes[1].hist(np.log1p(train_df["posted_rate"]), bins=80, color="#1A7D8E", edgecolor="white", linewidth=0.3)
axes[1].set_title("log1p(posted_rate) distribution")
axes[1].set_xlabel("log1p(Rate)")
fig.tight_layout()
fig.savefig(EDA_DIR / "rate_distribution.png", dpi=150)
plt.close(fig)

# 2c. Rate by equipment
fig, ax = plt.subplots(figsize=(8, 4))
train_df.groupby("equipment")["posted_rate"].median().sort_values().plot.barh(ax=ax, color="#064A56")
ax.set_title("Median posted_rate by equipment")
ax.set_xlabel("Median Rate ($)")
fig.tight_layout()
fig.savefig(EDA_DIR / "rate_by_equipment.png", dpi=150)
plt.close(fig)

# 2d. Rate by month
train_df["month"] = train_df["date"].dt.month
monthly = train_df.groupby("month")["posted_rate"].median()
fig, ax = plt.subplots(figsize=(10, 4))
monthly.plot(ax=ax, marker="o", color="#064A56")
ax.set_title("Median posted_rate by Month (train)")
ax.set_xlabel("Month")
ax.set_ylabel("Rate ($)")
fig.tight_layout()
fig.savefig(EDA_DIR / "rate_by_month.png", dpi=150)
plt.close(fig)

# 2e. Correlation heatmap for numeric columns
numeric_cols = ["distance", "weight", "market_index", "quote_signal", "posted_rate"]
corr = train_df[numeric_cols].corr()
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax, linewidths=0.5)
ax.set_title("Numeric feature correlations")
fig.tight_layout()
fig.savefig(EDA_DIR / "correlation_heatmap.png", dpi=150)
plt.close(fig)

# 2f. market_index vs rate scatter
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
sample = train_df.sample(min(5000, len(train_df)), random_state=42)
axes[0].scatter(sample["market_index"], sample["posted_rate"], alpha=0.15, s=8, color="#064A56")
axes[0].set_title("market_index vs posted_rate")
axes[0].set_xlabel("market_index")
axes[0].set_ylabel("Rate ($)")
axes[1].scatter(sample["quote_signal"], sample["posted_rate"], alpha=0.15, s=8, color="#1A7D8E")
axes[1].set_title("quote_signal vs posted_rate")
axes[1].set_xlabel("quote_signal")
axes[1].set_ylabel("Rate ($)")
fig.tight_layout()
fig.savefig(EDA_DIR / "signal_vs_rate.png", dpi=150)
plt.close(fig)

print("  EDA plots saved to eda_plots/")

# Check for missing values
print("\n--- Missing values ---")
print("Train:", train_df.isnull().sum()[train_df.isnull().sum() > 0].to_dict())
print("Val  :", val_df.isnull().sum()[val_df.isnull().sum() > 0].to_dict())

# ---------------------------------------------------------------------------
# 3. Feature Engineering
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Feature engineering ...")


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"]  = df["date"].dt.dayofweek        # 0=Mon ... 6=Sun
    df["day_of_month"] = df["date"].dt.day
    df["month"]        = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["quarter"]      = df["date"].dt.quarter
    return df


def add_load_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["weight_per_mile"] = df["weight"] / (df["distance"] + 1e-9)
    df["rate_per_mile_proxy"] = df["distance"] * 0.001  # rough normaliser only
    # Great-circle Haversine distance — only when lat/lon columns exist
    if all(c in df.columns for c in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]):
        lat1 = np.radians(df["pickup_lat"])
        lon1 = np.radians(df["pickup_lon"])
        lat2 = np.radians(df["delivery_lat"])
        lon2 = np.radians(df["delivery_lon"])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        df["haversine_dist"] = 2 * 6371 * np.arcsin(np.sqrt(a)) * 0.621371  # miles
        df["dist_diff"] = df["distance"] - df["haversine_dist"]
    else:
        # For the December fixed-lane inputs, use the distance column directly
        df["haversine_dist"] = df["distance"]
        df["dist_diff"] = 0.0
    return df


# Apply date + load features to train
train_df = add_date_features(train_df)
train_df = add_load_features(train_df)

# Fill missing values in training data
for col in ["weight", "market_index", "quote_signal"]:
    if train_df[col].isnull().any():
        median_val = train_df[col].median()
        train_df[col] = train_df[col].fillna(median_val)
        print(f"  Filled {train_df[col].isnull().sum()} missing {col} with median {median_val:.4f}")

# Equipment label encode
equip_map = {"Dry Van": 0, "Reefer": 1, "Flatbed": 2}
train_df["equipment_enc"] = train_df["equipment"].map(equip_map).fillna(-1).astype(int)

# Target-mean encoding for pickup and delivery cities
# Compute smoothed mean on FULL training set (leak-free at inference time since we use training means)
SMOOTHING = 50  # Bayesian smoothing count

global_mean = train_df["posted_rate"].mean()

def smooth_mean_encode(train: pd.DataFrame, col: str, target: str = "posted_rate", smoothing: int = SMOOTHING):
    stats = train.groupby(col)[target].agg(["mean", "count"])
    stats["smooth"] = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
    return stats["smooth"].to_dict()

pickup_enc   = smooth_mean_encode(train_df, "pickup")
delivery_enc = smooth_mean_encode(train_df, "delivery")

train_df["pickup_mean_rate"]   = train_df["pickup"].map(pickup_enc).fillna(global_mean)
train_df["delivery_mean_rate"] = train_df["delivery"].map(delivery_enc).fillna(global_mean)

# Lane-level mean rate (pickup + delivery pair)
lane_mean = (
    train_df.groupby(["pickup", "delivery"])["posted_rate"]
    .mean()
    .reset_index()
    .rename(columns={"posted_rate": "lane_mean_rate"})
)
train_df = train_df.merge(lane_mean, on=["pickup", "delivery"], how="left")
train_df["lane_mean_rate"].fillna(global_mean, inplace=True)


# Precompute training medians for use when columns are absent (e.g. December inputs)
TRAIN_MEDIANS = {
    col: train_df[col].median()
    for col in ["weight", "market_index", "quote_signal"]
}


def apply_features(df: pd.DataFrame, fill_cols: dict | None = None) -> pd.DataFrame:
    df = df.copy()
    # Inject any columns that are absent (lat/lon, market signals, etc.)
    if fill_cols:
        for col, val in fill_cols.items():
            if col not in df.columns:
                df[col] = val
    # Also fill market signals that may be absent entirely (December chart case)
    for col in ["market_index", "quote_signal"]:
        if col not in df.columns:
            df[col] = TRAIN_MEDIANS[col]
    df = add_date_features(df)
    df = add_load_features(df)
    # Fill missing numeric features
    for col in ["weight", "market_index", "quote_signal"]:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(TRAIN_MEDIANS[col])
    df["equipment_enc"]       = df["equipment"].map(equip_map).fillna(-1).astype(int)
    df["pickup_mean_rate"]    = df["pickup"].map(pickup_enc).fillna(global_mean)
    df["delivery_mean_rate"]  = df["delivery"].map(delivery_enc).fillna(global_mean)
    df = df.merge(lane_mean, on=["pickup", "delivery"], how="left")
    df["lane_mean_rate"] = df["lane_mean_rate"].fillna(global_mean)
    return df


# Inject lat/lon for Lexington->Fort Wayne fixed lane
LEX_FW_LATLON = {
    "pickup_lat": 36.99152, "pickup_lon": -84.99876,
    "delivery_lat": 41.31561, "delivery_lon": -85.36206,
}

val_df = apply_features(val_df)
dec_df = apply_features(dec_df, fill_cols=LEX_FW_LATLON)

print("  Features added.")

# ---------------------------------------------------------------------------
# 4. Time-based split: Train = Jan–Sep, Hold-out = Oct
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Creating time-based split ...")

FEATURE_COLS = [
    "distance", "weight", "market_index", "quote_signal",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "equipment_enc",
    "day_of_week", "day_of_month", "month", "week_of_year", "is_weekend", "quarter",
    "weight_per_mile", "haversine_dist", "dist_diff",
    "pickup_mean_rate", "delivery_mean_rate", "lane_mean_rate",
]

train_mask = train_df["date"].dt.month <= 9
X_train = train_df.loc[train_mask, FEATURE_COLS]
y_train = train_df.loc[train_mask, "posted_rate"]
X_hold  = train_df.loc[~train_mask, FEATURE_COLS]
y_hold  = train_df.loc[~train_mask, "posted_rate"]
y_train_log = np.log1p(y_train)
y_hold_log  = np.log1p(y_hold)

print(f"  Train (Jan-Sep): {len(X_train):,} rows")
print(f"  Hold-out (Oct) : {len(X_hold):,} rows")

# ---------------------------------------------------------------------------
# 5. Train LightGBM
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Training LightGBM ...")

lgb_train = lgb.Dataset(X_train, label=y_train_log)
lgb_hold  = lgb.Dataset(X_hold,  label=y_hold_log, reference=lgb_train)

lgb_params = {
    "objective":       "regression",
    "metric":          "rmse",
    "num_leaves":      127,
    "learning_rate":   0.05,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq":    5,
    "min_child_samples": 20,
    "reg_alpha":       0.1,
    "reg_lambda":      0.1,
    "verbose":         -1,
    "seed":            42,
}

callbacks = [
    lgb.early_stopping(stopping_rounds=50, verbose=True),
    lgb.log_evaluation(period=100),
]

lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=2000,
    valid_sets=[lgb_hold],
    callbacks=callbacks,
)

lgb_pred_hold = np.expm1(lgb_model.predict(X_hold))
lgb_rmse = mean_squared_error(y_hold, lgb_pred_hold) ** 0.5
lgb_mape = mean_absolute_percentage_error(y_hold, lgb_pred_hold) * 100
print(f"\n  LightGBM Hold-out RMSE : ${lgb_rmse:,.2f}")
print(f"  LightGBM Hold-out MAPE : {lgb_mape:.2f}%")

# ---------------------------------------------------------------------------
# 6. Train XGBoost
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Training XGBoost ...")

xgb_model = xgb.XGBRegressor(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=0.1,
    tree_method="hist",
    early_stopping_rounds=50,
    eval_metric="rmse",
    random_state=42,
    verbosity=0,
)

xgb_model.fit(
    X_train, y_train_log,
    eval_set=[(X_hold, y_hold_log)],
    verbose=100,
)

xgb_pred_hold = np.expm1(xgb_model.predict(X_hold))
xgb_rmse = mean_squared_error(y_hold, xgb_pred_hold) ** 0.5
xgb_mape = mean_absolute_percentage_error(y_hold, xgb_pred_hold) * 100
print(f"\n  XGBoost Hold-out RMSE : ${xgb_rmse:,.2f}")
print(f"  XGBoost Hold-out MAPE : {xgb_mape:.2f}%")

# ---------------------------------------------------------------------------
# 7. Ensemble and evaluate
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Ensemble ...")

# Weight models by inverse RMSE
w_lgb = 1 / lgb_rmse
w_xgb = 1 / xgb_rmse
w_total = w_lgb + w_xgb
ensemble_hold = (w_lgb * lgb_pred_hold + w_xgb * xgb_pred_hold) / w_total

ens_rmse = mean_squared_error(y_hold, ensemble_hold) ** 0.5
ens_mape = mean_absolute_percentage_error(y_hold, ensemble_hold) * 100
print(f"  Ensemble Hold-out RMSE : ${ens_rmse:,.2f}")
print(f"  Ensemble Hold-out MAPE : {ens_mape:.2f}%")

# Summary table
summary = pd.DataFrame({
    "Model":    ["LightGBM", "XGBoost", "Ensemble"],
    "RMSE ($)": [lgb_rmse, xgb_rmse, ens_rmse],
    "MAPE (%)": [lgb_mape, xgb_mape, ens_mape],
})
print("\n" + summary.to_string(index=False))

# Save metrics
summary.to_csv(BASE / "model_metrics.csv", index=False)

# Feature importance plot (LightGBM)
imp = pd.DataFrame({
    "feature":    FEATURE_COLS,
    "importance": lgb_model.feature_importance(importance_type="gain"),
}).sort_values("importance", ascending=False)

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(imp["feature"][::-1], imp["importance"][::-1], color="#064A56")
ax.set_title("LightGBM Feature Importance (gain)")
ax.set_xlabel("Gain")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(EDA_DIR / "feature_importance.png", dpi=150)
plt.close(fig)
print("\n  Feature importance saved to eda_plots/feature_importance.png")

# Hold-out actual vs predicted scatter
fig, ax = plt.subplots(figsize=(7, 6))
sample_idx = np.random.default_rng(42).choice(len(y_hold), min(3000, len(y_hold)), replace=False)
ax.scatter(
    np.array(y_hold)[sample_idx],
    ensemble_hold[sample_idx],
    alpha=0.15, s=6, color="#064A56"
)
lims = [min(y_hold.min(), ensemble_hold.min()), max(y_hold.max(), ensemble_hold.max())]
ax.plot(lims, lims, "r--", linewidth=1)
ax.set_title("Hold-out: Actual vs Predicted (Ensemble)")
ax.set_xlabel("Actual Rate ($)")
ax.set_ylabel("Predicted Rate ($)")
fig.tight_layout()
fig.savefig(EDA_DIR / "actual_vs_predicted.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 8. Retrain on full training data (Jan–Oct) before final predictions
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Retraining on full data (Jan–Oct) for final predictions ...")

X_full = train_df[FEATURE_COLS]
y_full_log = np.log1p(train_df["posted_rate"])

best_lgb_rounds = lgb_model.best_iteration
best_xgb_rounds = xgb_model.best_iteration

lgb_full_data = lgb.Dataset(X_full, label=y_full_log)
lgb_final = lgb.train(
    {**lgb_params, "verbose": -1},
    lgb_full_data,
    num_boost_round=best_lgb_rounds,
)

xgb_final = xgb.XGBRegressor(
    n_estimators=best_xgb_rounds,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=0.1,
    tree_method="hist",
    random_state=42,
    verbosity=0,
)
xgb_final.fit(X_full, y_full_log)

print(f"  LightGBM final trees : {best_lgb_rounds}")
print(f"  XGBoost  final trees : {best_xgb_rounds}")

# ---------------------------------------------------------------------------
# 9. Predict validation.csv
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Predicting validation set ...")

X_val_final = val_df[FEATURE_COLS]

lgb_val_pred = np.expm1(lgb_final.predict(X_val_final))
xgb_val_pred = np.expm1(xgb_final.predict(X_val_final))
ensemble_val = (w_lgb * lgb_val_pred + w_xgb * xgb_val_pred) / w_total

# Ensure positive
ensemble_val = np.clip(ensemble_val, 1e-3, None)

predictions_df = pd.DataFrame({
    "load_id":        val_df["load_id"].values,
    "predicted_rate": ensemble_val,
})
predictions_df.to_csv(PREDICTIONS_OUT, index=False)
print(f"  Saved {len(predictions_df):,} predictions -> {PREDICTIONS_OUT.name}")
print(f"  Rate range: ${ensemble_val.min():,.2f} – ${ensemble_val.max():,.2f}")
print(f"  Rate mean : ${ensemble_val.mean():,.2f}")

# ---------------------------------------------------------------------------
# 10. Predict December chart inputs
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Predicting December chart ...")

X_dec = dec_df[FEATURE_COLS]

lgb_dec_pred = np.expm1(lgb_final.predict(X_dec))
xgb_dec_pred = np.expm1(xgb_final.predict(X_dec))
ensemble_dec = (w_lgb * lgb_dec_pred + w_xgb * xgb_dec_pred) / w_total
ensemble_dec = np.clip(ensemble_dec, 1e-3, None)

dec_df["predicted_rate"] = ensemble_dec
dec_df["date"] = dec_df["date"].dt.strftime("%Y-%m-%d")
dec_out = dec_df[["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]]
dec_out.to_csv(DECEMBER_PATH, index=False)
print(f"  December predictions saved -> {DECEMBER_PATH.name}")
print(f"  Rate range: ${ensemble_dec.min():,.2f} – ${ensemble_dec.max():,.2f}")

print("\n" + "=" * 60)
print("Done. Next: run score.py to validate and generate the chart.")
print(
    "  python score.py "
    "--predictions validation_predictions.csv "
    "--december-predictions december-chart-inputs.csv"
)
