import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor
from scipy.stats import norm

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
    df = df[required_cols].dropna().reset_index(drop=True)

    print("Processed shape:", df.shape)

    X = df[feature_cols]
    y_rain = df["rain_tomorrow"]
    y_temp = df["tomorrow_max_temp"]

    return X, y_rain, y_temp, df

def split_data(X, y_rain, y_temp, df, test_size=0.2):
    split_index = int(len(df) * (1 - test_size))

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_rain_train, y_rain_test = y_rain.iloc[:split_index], y_rain.iloc[split_index:]
    y_temp_train, y_temp_test = y_temp.iloc[:split_index], y_temp.iloc[split_index:]
    df_test = df.iloc[split_index:].copy()

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    return X_train, X_test, y_rain_train, y_rain_test, y_temp_train, y_temp_test, df_test

def train_classification_model(X_train, y_train):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])
    model.fit(X_train, y_train)
    return model

def train_regression_model(X_train, y_train):
    model = XGBRegressor(
        n_estimators=300,
        random_state=42,
        min_child_weight=1,
        max_depth=6,
        eval_metric="rmse",
        verbosity=0
    )
    model.fit(X_train, y_train)
    return model

def compute_probabilities(X_test, y_temp_test, df_test, clf_model, reg_model, temperature_thresholds=None):
    if temperature_thresholds is None:
        temperature_thresholds = [10, 12, 14]

    rain_probs = clf_model.predict_proba(X_test)[:, 1]

    temp_preds = reg_model.predict(X_test)
    residuals = y_temp_test - temp_preds
    temp_std = float(np.std(residuals))

    if temp_std == 0:
        temp_std = 1e-6

    results_df = df_test.copy()
    results_df["rain_probability"] = rain_probs
    results_df["predicted_temp"] = temp_preds
    results_df["temp_residual"] = residuals

    for threshold in temperature_thresholds:
        results_df[f"prob_temp_gt_{threshold}"] = 1 - norm.cdf(
            threshold,
            loc=results_df["predicted_temp"],
            scale=temp_std
        )

    return results_df, temp_std

def main():
    print("Loading features...")
    engine = create_engine(DB_URL)
    df = load_features(engine)

    print("Preprocessing...")
    X, y_rain, y_temp, df_processed = preprocess(df)

    print("Splitting train/test...")
    X_train, X_test, y_rain_train, y_rain_test, y_temp_train, y_temp_test, df_test = split_data(
        X, y_rain, y_temp, df_processed
    )

    print("Training classification model...")
    clf_model = train_classification_model(X_train, y_rain_train)

    print("Training regression model...")
    reg_model = train_regression_model(X_train, y_temp_train)

    print("Computing probabilities...")
    results_df, temp_std = compute_probabilities(
        X_test,
        y_temp_test,
        df_test,
        clf_model,
        reg_model,
        temperature_thresholds=[10, 12, 14]
    )

    print("\nProbability Forecasting Summary:")
    print(f"Mean rain probability: {results_df['rain_probability'].mean():.3f}")
    print(f"Mean predicted temperature: {results_df['predicted_temp'].mean():.3f}")
    print(f"Residual std used for event probabilities: {temp_std:.3f}")

    print("\nThreshold event probabilities:")
    for threshold in [10, 12, 14]:
        col = f"prob_temp_gt_{threshold}"
        print(f"P(temp > {threshold}) mean: {results_df[col].mean():.3f}")

    print("\nSample predictions:")
    preview_cols = [
        "date",
        "rain_probability",
        "predicted_temp",
        "prob_temp_gt_10",
        "prob_temp_gt_12",
        "prob_temp_gt_14",
    ]
    print(results_df[preview_cols].head(10))

if __name__ == "__main__":
    main()