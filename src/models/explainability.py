import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor
import shap

DB_URL = "postgresql+psycopg2://ericliu:Ericmliu1234@127.0.0.1:5432/weatheredge"

def load_features(engine):
    query = """
        SELECT *
        FROM feature_table
        ORDER BY city, date
    """
    return pd.read_sql(query, engine)

def create_targets(df):
    df = df.copy()

    df["rain_tomorrow"] = (
        df.groupby("city")["precipitation"]
        .shift(-24)
        .fillna(0)
        .gt(0)
        .astype(int)
    )

    df["tomorrow_max_temp"] = (
        df.groupby("city")["temperature"]
        .shift(-24)
    )

    return df

def preprocess(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["city", "date"]).reset_index(drop=True)

    df = create_targets(df)

    df["season"] = df["season"].astype("category").cat.codes
    df["city"] = df["city"].astype("category").cat.codes

    if "id" in df.columns:
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

    required_cols = feature_cols + ["rain_tomorrow", "tomorrow_max_temp", "date"]
    df = df[required_cols].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    X = df[feature_cols].astype(float)
    y_rain = df["rain_tomorrow"].astype(int)
    y_temp = df["tomorrow_max_temp"].astype(float)

    lower = X.quantile(0.01)
    upper = X.quantile(0.99)
    X = X.clip(lower=lower, upper=upper, axis=1)

    print("Processed shape:", df.shape)
    print("\nFeature range check:")
    range_df = pd.DataFrame({
        "min": X.min(),
        "max": X.max(),
        "mean": X.mean(),
        "std": X.std()
    }).sort_values("std", ascending=False)
    print(range_df)

    return X, y_rain, y_temp, df

def split_data(X, y_rain, y_temp, df):
    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()

    y_rain_train = y_rain.iloc[:split_index].copy()
    y_rain_test = y_rain.iloc[split_index:].copy()

    y_temp_train = y_temp.iloc[:split_index].copy()
    y_temp_test = y_temp.iloc[split_index:].copy()

    df_test = df.iloc[split_index:].copy()

    print("\nTrain shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    return X_train, X_test, y_rain_train, y_rain_test, y_temp_train, y_temp_test, df_test

def train_models(X_train, y_rain_train, y_temp_train):
    clf_model = Pipeline([
        ("scaler", RobustScaler()),
        ("model", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="lbfgs",
            C=0.5,
            random_state=42
        ))
    ])
    clf_model.fit(X_train, y_rain_train)

    xgb_clf = XGBClassifier(
        n_estimators=300,
        random_state=42,
        min_child_weight=1,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        verbosity=0
    )
    xgb_clf.fit(X_train, y_rain_train)

    xgb_reg = XGBRegressor(
        n_estimators=300,
        random_state=42,
        min_child_weight=1,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="rmse",
        verbosity=0
    )
    xgb_reg.fit(X_train, y_temp_train)

    return clf_model, xgb_clf, xgb_reg

def global_importance_xgb(model, feature_names):
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    return importance_df

def local_importance_shap(model, X, feature_names, row_idx=0):
    X_background = X.iloc[:min(100, len(X))].copy()
    X_sample = X.iloc[[row_idx]].copy()

    predict_fn = lambda data: model.predict_proba(
        pd.DataFrame(data, columns=feature_names)
    )[:, 1]

    explainer = shap.Explainer(predict_fn, X_background)
    explanation = explainer(X_sample)

    values = explanation.values

    if len(values.shape) == 2:
        local_values = values[0]
    elif len(values.shape) == 1:
        local_values = values
    else:
        raise ValueError(f"Unexpected SHAP shape: {values.shape}")

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": local_values
    })

    shap_df["abs_shap"] = shap_df["shap_value"].abs()
    shap_df = shap_df.sort_values("abs_shap", ascending=False).drop(columns=["abs_shap"])

    return shap_df

def evaluate_models(clf_model, xgb_clf, xgb_reg, X_test, y_rain_test, y_temp_test):
    log_probs = clf_model.predict_proba(X_test)[:, 1]
    log_preds = (log_probs >= 0.5).astype(int)

    xgb_probs = xgb_clf.predict_proba(X_test)[:, 1]
    xgb_preds = (xgb_probs >= 0.5).astype(int)

    temp_preds = xgb_reg.predict(X_test)

    print("\nClassification metrics (Logistic Regression):")
    print(f"Accuracy: {accuracy_score(y_rain_test, log_preds):.4f}")
    print(f"F1 score: {f1_score(y_rain_test, log_preds):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_rain_test, log_probs):.4f}")

    print("\nClassification metrics (XGBoost):")
    print(f"Accuracy: {accuracy_score(y_rain_test, xgb_preds):.4f}")
    print(f"F1 score: {f1_score(y_rain_test, xgb_preds):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_rain_test, xgb_probs):.4f}")

    print("\nRegression metrics (XGBoost temperature):")
    print(f"MAE: {mean_absolute_error(y_temp_test, temp_preds):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_temp_test, temp_preds)):.4f}")
    print(f"R^2: {r2_score(y_temp_test, temp_preds):.4f}")

def main():
    print("Loading features...")
    engine = create_engine(DB_URL)
    df = load_features(engine)

    print("Preprocessing...")
    X, y_rain, y_temp, df_processed = preprocess(df)
    feature_names = X.columns.tolist()

    print("\nSplitting data...")
    X_train, X_test, y_rain_train, y_rain_test, y_temp_train, y_temp_test, df_test = split_data(
        X, y_rain, y_temp, df_processed
    )

    print("\nTraining models...")
    clf_model, xgb_clf, xgb_reg = train_models(X_train, y_rain_train, y_temp_train)

    evaluate_models(clf_model, xgb_clf, xgb_reg, X_test, y_rain_test, y_temp_test)

    print("\nComputing global importance for XGB rain classifier...")
    rain_importance = global_importance_xgb(xgb_clf, feature_names)
    print("\nRain classification global importance (XGBoost):")
    print(rain_importance.head(10))

    print("\nComputing global importance for XGB temp regressor...")
    temp_importance = global_importance_xgb(xgb_reg, feature_names)
    print("\nTemperature regression global importance (XGBoost):")
    print(temp_importance.head(10))

    print("\nComputing local importance for first test sample (XGB rain classifier)...")
    local_shap = local_importance_shap(xgb_clf, X_test, feature_names, row_idx=0)
    print("\nLocal SHAP values for first test sample (rain prediction):")
    print(local_shap)

    print("\nSample prediction:")
    print(df_test.iloc[0][["date", "temperature", "precipitation"]])

    pred_prob = xgb_clf.predict_proba(X_test.iloc[[0]])[0, 1]
    print(f"Predicted rain probability: {pred_prob:.3f}")
    print(f"Actual rain tomorrow: {y_rain_test.iloc[0]}")

if __name__ == "__main__":
    main()