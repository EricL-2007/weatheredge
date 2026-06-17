import pandas as pd
from sqlalchemy import create_engine
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception as e:
    print("XGBoost import error:", e)
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except Exception as e:
    print("LightGBM import error:", e)
    LIGHTGBM_AVAILABLE = False

DB_URL = "postgresql+psycopg2://ericliu:Ericmliu1234@127.0.0.1:5432/weatheredge"

def load_features(engine):
    query = """
        SELECT *
        FROM feature_table
        ORDER BY city, date
    """
    return pd.read_sql(query, engine)

def create_target(df):
    df = df.copy()
    df["rain_tomorrow"] = (
        df.groupby("city")["precipitation"]
        .shift(-24)
        .fillna(0)
        .gt(0)
        .astype(int)
    )
    return df

def preprocess(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["city", "date"]).reset_index(drop=True)
    df = create_target(df)

    df["season"] = df["season"].astype("category").cat.codes
    df["city"] = df["city"].astype("category").cat.codes

    df = df.drop(columns=["id"])

    feature_cols = [
        "city",
        "temperature",
        "humidity",
        "pressure",
        "wind_speed",
        "precipitation",
        "cloud_cover",
        "month",
        "week",
        "quarter",
        "day_of_week",
        "season",
        "temperature_1d",
        "precipitation_1d",
        "temp_change_1d",
        "rain_trend_1d",
        "wind_trend_1d",
    ]

    required_cols = feature_cols + ["rain_tomorrow", "date"]
    df = df[required_cols].dropna().reset_index(drop=True)

    print("Processed shape:", df.shape)
    print("Target distribution:")
    print(df["rain_tomorrow"].value_counts(dropna=False).sort_index())

    X = df[feature_cols]
    y = df["rain_tomorrow"]
    return X, y, df

def time_split(X, y, df, test_size=0.2):
    split_index = int(len(df) * (1 - test_size))

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("y_train distribution:")
    print(y_train.value_counts(dropna=False).sort_index())
    print("y_test distribution:")
    print(y_test.value_counts(dropna=False).sort_index())

    return X_train, X_test, y_train, y_test

def evaluate_model(name, model, X_train, X_test, y_train, y_test, threshold=0.4):
    model.fit(X_train, y_train)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.predict(X_test)

    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob) if len(y_test.unique()) > 1 else None,
        "avg_precision": average_precision_score(y_test, y_prob),
        "mean_pred_prob": float(y_prob.mean()),
    }

    print(f"\n{name} mean predicted probability: {y_prob.mean():.4f}")
    print(f"{name} positive predictions at threshold {threshold}: {(y_pred == 1).sum()}")

    return metrics

def main():
    print("Loading features...")
    engine = create_engine(DB_URL)
    df = load_features(engine)

    print("Preprocessing...")
    X, y, df_processed = preprocess(df)

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = time_split(X, y, df_processed)

    if len(y_train) == 0 or len(y_test) == 0:
        print("Error: train or test split is empty.")
        return

    positive_count = y_train.sum()
    negative_count = len(y_train) - positive_count
    scale_pos_weight = negative_count / positive_count if positive_count > 0 else 1.0

    models = {
        "baseline_dummy": DummyClassifier(strategy="prior"),
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced_subsample",
            min_samples_leaf=5,
            max_depth=6
        ),
    }

    if XGBOOST_AVAILABLE:
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            min_child_weight=1,
            max_depth=6,
            eval_metric="logloss",
            verbosity=0
        )

    if LIGHTGBM_AVAILABLE:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=300,
            random_state=42,
            is_unbalance=True,
            min_child_samples=5,
            max_depth=6,
            verbose=-1
        )

    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        metrics = evaluate_model(name, model, X_train, X_test, y_train, y_test, threshold=0.4)
        results.append(metrics)

    results_df = pd.DataFrame(results)
    print("\nClassification Results:")
    print(results_df.sort_values("f1", ascending=False))

if __name__ == "__main__":
    main()