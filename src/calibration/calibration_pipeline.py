import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

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

    X = df[feature_cols]
    y = df["rain_tomorrow"]
    return X, y, df

def split_data(X, y, df):
    n = len(df)
    train_end = int(n * 0.6)
    calib_end = int(n * 0.8)

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_calib = X.iloc[train_end:calib_end]
    y_calib = y.iloc[train_end:calib_end]

    X_test = X.iloc[calib_end:]
    y_test = y.iloc[calib_end:]

    return X_train, y_train, X_calib, y_calib, X_test, y_test

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1

    ece = 0.0
    for i in range(n_bins):
        mask = bin_ids == i
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += np.abs(bin_acc - bin_conf) * np.sum(mask) / len(y_true)
    return ece

def main():
    print("Loading features...")
    engine = create_engine(DB_URL)
    df = load_features(engine)

    print("Preprocessing...")
    X, y, df_processed = preprocess(df)

    print("Splitting data into train / calibration / test...")
    X_train, y_train, X_calib, y_calib, X_test, y_test = split_data(X, y, df_processed)

    print("Training base classifier...")
    base_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])
    base_model.fit(X_train, y_train)

    calib_probs = base_model.predict_proba(X_calib)[:, 1]
    test_probs_uncal = base_model.predict_proba(X_test)[:, 1]

    print("Applying Platt scaling...")
    platt_model = LogisticRegression()
    platt_model.fit(calib_probs.reshape(-1, 1), y_calib)
    test_probs_platt = platt_model.predict_proba(test_probs_uncal.reshape(-1, 1))[:, 1]

    print("Applying isotonic regression...")
    iso_model = IsotonicRegression(out_of_bounds="clip")
    iso_model.fit(calib_probs, y_calib)
    test_probs_iso = iso_model.predict(test_probs_uncal)

    print("\nCalibration Metrics:")

    methods = {
        "uncalibrated": test_probs_uncal,
        "platt_scaled": test_probs_platt,
        "isotonic": test_probs_iso,
    }

    for name, probs in methods.items():
        brier = brier_score_loss(y_test, probs)
        ece = expected_calibration_error(y_test.to_numpy(), probs, n_bins=10)
        ll = log_loss(y_test, probs)

        print(f"\n{name}")
        print(f"  Brier score: {brier:.4f}")
        print(f"  ECE: {ece:.4f}")
        print(f"  Log loss: {ll:.4f}")

        frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)
        print("  Reliability curve points:")
        for mp, fp in zip(mean_pred, frac_pos):
            print(f"    predicted={mp:.3f}, actual={fp:.3f}")

if __name__ == "__main__":
    main()