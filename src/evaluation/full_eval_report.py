import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import json

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, accuracy_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from xgboost import XGBClassifier

import matplotlib.pyplot as plt

load_dotenv(".env")

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

QUERY = """
SELECT
  id,
  market_type,
  city_name,
  category,
  model_probability,
  implied_probability,
  edge,
  ev_yes,
  status,
  raw_response
FROM market_data
WHERE status = 'finalized'
"""

df = pd.read_sql(text(QUERY), engine)

df["result"] = df["raw_response"].apply(
    lambda r: r.get("result") if isinstance(r, dict)
    else (json.loads(r).get("result") if isinstance(r, str) else None)
)

df = df.dropna(subset=["result"])
df["label"] = (df["result"] == "yes").astype(int)

drop_cols = ["id", "result", "label", "raw_response"]
feature_candidates = [c for c in df.columns if c not in drop_cols]

X = df[feature_candidates].copy()
y = df["label"].astype(int)

X = pd.get_dummies(X, drop_first=True)
X = X.replace([np.inf, -np.inf], np.nan)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "logistic_regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LogisticRegression(max_iter=2000))
    ]),
    "random_forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        ))
    ]),
    "hist_gradient_boosting": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.05,
            max_iter=300,
            random_state=42
        ))
    ]),
    "xgboost": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        ))
    ])
}

benchmark_results = []
plt.figure(figsize=(8, 6))

for name, model in models.items():
    model.fit(X_train, y_train)
    raw_probs = model.predict_proba(X_test)[:, 1]
    raw_preds = (raw_probs >= 0.5).astype(int)

    benchmark_results.append({
        "model": name,
        "version": "raw",
        "roc_auc": roc_auc_score(y_test, raw_probs),
        "log_loss": log_loss(y_test, raw_probs),
        "brier_score": brier_score_loss(y_test, raw_probs),
        "accuracy": accuracy_score(y_test, raw_preds),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X.shape[1]
    })

    calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=3)
    calibrated.fit(X_train, y_train)
    cal_probs = calibrated.predict_proba(X_test)[:, 1]
    cal_preds = (cal_probs >= 0.5).astype(int)

    benchmark_results.append({
        "model": name,
        "version": "calibrated",
        "roc_auc": roc_auc_score(y_test, cal_probs),
        "log_loss": log_loss(y_test, cal_probs),
        "brier_score": brier_score_loss(y_test, cal_probs),
        "accuracy": accuracy_score(y_test, cal_preds),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X.shape[1]
    })

    frac_pos, mean_pred = calibration_curve(y_test, cal_probs, n_bins=10, strategy="quantile")
    plt.plot(mean_pred, frac_pos, marker="o", label=name)

plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect_calibration")
plt.xlabel("Mean predicted probability")
plt.ylabel("Observed frequency")
plt.title("Calibration Curves")
plt.legend()
plt.tight_layout()

benchmark_df = pd.DataFrame(benchmark_results).sort_values(["model", "version"])

edge_eval = pd.DataFrame([
    {"metric": "edge_correlation", "value": float(df["edge"].corr(df["label"]))},
    {"metric": "ev_correlation", "value": float(df["ev_yes"].corr(df["label"]))},
    {"metric": "mean_edge_yes", "value": float(df[df["label"]==1]["edge"].mean())},
    {"metric": "mean_edge_no", "value": float(df[df["label"]==0]["edge"].mean())},
    {"metric": "mean_ev_yes", "value": float(df[df["label"]==1]["ev_yes"].mean())},
    {"metric": "mean_ev_no", "value": float(df[df["label"]==0]["ev_yes"].mean())}
])

os.makedirs("output", exist_ok=True)

benchmark_df.to_csv("output/model_benchmark_results.csv", index=False)
plt.savefig("output/calibration_curves.png", dpi=200, bbox_inches="tight")
edge_eval.to_csv("output/edge_evaluations.csv", index=False)

print("=== MODEL BENCHMARK ===")
print(benchmark_df)
print("\n=== EDGE & EV EVALUATION ===")
print(edge_eval)
print("\n=== FILES SAVED ===")
print("output/model_benchmark_results.csv")
print("output/calibration_curves.png")
print("output/edge_evaluations.csv")