import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception as e:
    print("XGBoost import error:", e)
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
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
    df["tomorrow_max_temp"] = (
        df.groupby("city")["temperature"]
        .shift(-24)
        .fillna(df["temperature"])
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

    required_cols = feature_cols + ["tomorrow_max_temp", "date"]
    df = df[required_cols].dropna().reset_index(drop=True)

    print("Processed shape:", df.shape)
    print("Target range:")
    print("min:", df["tomorrow_max_temp"].min())
    print("max:", df["tomorrow_max_temp"].max())

    X = df[feature_cols]
    y = df["tomorrow_max_temp"]
    return X, y, df

def time_split(X, y, df, test_size=0.2):
    split_index = int(len(df) * (1 - test_size))

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    return X_train, X_test, y_train, y_test

import math

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "model": name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }

    print(f"\n{name} MAE: {mae:.3f}")
    print(f"{name} RMSE: {rmse:.3f}")
    print(f"{name} R²: {r2:.3f}")

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

    models = {
        "linear_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            min_samples_leaf=5,
            max_depth=6
        ),
    }

    if XGBOOST_AVAILABLE:
        models["xgboost"] = XGBRegressor(
            n_estimators=300,
            random_state=42,
            min_child_weight=1,
            max_depth=6,
            eval_metric="rmse",
            verbosity=0
        )

    if LIGHTGBM_AVAILABLE:
        models["lightgbm"] = LGBMRegressor(
            n_estimators=300,
            random_state=42,
            min_child_samples=5,
            max_depth=6,
            verbose=-1
        )

    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        metrics = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results.append(metrics)

    results_df = pd.DataFrame(results)
    print("\nRegression Results:")
    print(results_df.sort_values("r2", ascending=False))

if __name__ == "__main__":
    main()