import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier

DB_URL = "postgresql+psycopg2://ericliu:Ericmliu1234@127.0.0.1:5432/weatheredge"

WEATHER_KEYWORDS = [
    "rain",
    "snow",
    "temperature",
    "temp",
    "wind",
    "hurricane",
    "storm",
    "weather",
    "precipitation"
]

def get_engine():
    return create_engine(DB_URL)

def load_feature_data(engine):
    query = """
        SELECT *
        FROM feature_table
        ORDER BY city, date
    """
    return pd.read_sql(query, engine)

def load_market_data(engine):
    query = """
        SELECT *
        FROM market_data
        ORDER BY fetched_at DESC
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

def preprocess_features(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["city", "date"]).reset_index(drop=True)
    df = create_target(df)

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

    required_cols = feature_cols + ["rain_tomorrow", "date"]
    df = df[required_cols].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    X = df[feature_cols].astype(float)
    y = df["rain_tomorrow"].astype(int)

    lower = X.quantile(0.01)
    upper = X.quantile(0.99)
    X = X.clip(lower=lower, upper=upper, axis=1)

    return X, y, df, feature_cols

def split_train_calib_latest(X, y, df):
    n = len(df)
    train_end = int(n * 0.6)
    calib_end = int(n * 0.8)

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_calib = X.iloc[train_end:calib_end]
    y_calib = y.iloc[train_end:calib_end]

    X_latest = X.iloc[[-1]].copy()
    latest_row = df.iloc[[-1]].copy()

    return X_train, y_train, X_calib, y_calib, X_latest, latest_row

def train_calibrated_rain_model(X_train, y_train, X_calib, y_calib):
    base_model = XGBClassifier(
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
    base_model.fit(X_train, y_train)

    calib_probs = base_model.predict_proba(X_calib)[:, 1]

    iso_model = IsotonicRegression(out_of_bounds="clip")
    iso_model.fit(calib_probs, y_calib)

    return base_model, iso_model

def get_latest_model_probability(base_model, iso_model, X_latest):
    raw_prob = base_model.predict_proba(X_latest)[:, 1][0]
    calibrated_prob = iso_model.predict([raw_prob])[0]
    return raw_prob, calibrated_prob

def is_weather_market(question):
    if not isinstance(question, str):
        return False
    q = question.lower()
    return any(keyword in q for keyword in WEATHER_KEYWORDS)

def extract_city_from_question(question):
    if not isinstance(question, str):
        return None

    known_cities = [
        "houston", "austin", "dallas", "denver",
        "chicago", "new york", "los angeles", "seattle"
    ]

    q = question.lower()
    for city in known_cities:
        if city in q:
            return city.title()

    return None

def classify_market_type(question):
    if not isinstance(question, str):
        return "other"

    q = question.lower()

    if "rain" in q or "precipitation" in q:
        return "rain"

    if "temp" in q or "temperature" in q:
        return "temperature"

    if "wind" in q:
        return "wind"

    return "other"

def filter_weather_markets(df_markets):
    df = df_markets.copy()
    df = df[df["question"].apply(is_weather_market)].copy()

    if df.empty:
        return df

    df["market_type"] = df["question"].apply(classify_market_type)
    df["city_name"] = df["question"].apply(extract_city_from_question)

    return df

def build_rankings(df_markets, calibrated_rain_prob):
    if df_markets.empty:
        return pd.DataFrame()

    df = df_markets.copy()
    df = df[df["market_type"] == "rain"].copy()

    if df.empty:
        return pd.DataFrame()

    df["model_probability"] = calibrated_rain_prob
    df["market_probability"] = df["implied_probability"].astype(float)
    df = df[df["market_probability"].notna()].copy()

    df["edge"] = df["model_probability"] - df["market_probability"]
    df["abs_edge"] = df["edge"].abs()

    df["confidence_score"] = 1 - (df["model_probability"] * (1 - df["model_probability"])) * 4
    df["ranking_score"] = df["abs_edge"] * 0.7 + df["confidence_score"] * 0.3

    ranked = df.sort_values("ranking_score", ascending=False).reset_index(drop=True)

    keep_cols = [
        "market_id",
        "question",
        "market_type",
        "city_name",
        "model_probability",
        "market_probability",
        "edge",
        "confidence_score",
        "ranking_score",
        "close_date",
        "status"
    ]
    return ranked[keep_cols]

def save_rankings(engine, df_ranked):
    create_sql = """
    CREATE TABLE IF NOT EXISTS ranked_opportunities (
        id SERIAL PRIMARY KEY,
        market_id VARCHAR(255) UNIQUE,
        question TEXT,
        market_type VARCHAR(50),
        city_name VARCHAR(100),
        model_probability FLOAT,
        market_probability FLOAT,
        edge FLOAT,
        confidence_score FLOAT,
        ranking_score FLOAT,
        close_date TIMESTAMP,
        status VARCHAR(50),
        created_at TIMESTAMP DEFAULT NOW()
    );
    """

    with engine.begin() as conn:
        conn.execute(text(create_sql))

        if df_ranked.empty:
            return

        for _, row in df_ranked.iterrows():
            upsert_sql = text("""
                INSERT INTO ranked_opportunities (
                    market_id,
                    question,
                    market_type,
                    city_name,
                    model_probability,
                    market_probability,
                    edge,
                    confidence_score,
                    ranking_score,
                    close_date,
                    status
                )
                VALUES (
                    :market_id,
                    :question,
                    :market_type,
                    :city_name,
                    :model_probability,
                    :market_probability,
                    :edge,
                    :confidence_score,
                    :ranking_score,
                    :close_date,
                    :status
                )
                ON CONFLICT (market_id)
                DO UPDATE SET
                    question = EXCLUDED.question,
                    market_type = EXCLUDED.market_type,
                    city_name = EXCLUDED.city_name,
                    model_probability = EXCLUDED.model_probability,
                    market_probability = EXCLUDED.market_probability,
                    edge = EXCLUDED.edge,
                    confidence_score = EXCLUDED.confidence_score,
                    ranking_score = EXCLUDED.ranking_score,
                    close_date = EXCLUDED.close_date,
                    status = EXCLUDED.status
            """)

            payload = row.to_dict()
            payload["close_date"] = pd.to_datetime(payload["close_date"]) if pd.notnull(payload["close_date"]) else None
            conn.execute(upsert_sql, payload)

def main():
    print("Connecting to database...")
    engine = get_engine()

    print("Loading weather features...")
    df_features = load_feature_data(engine)

    print("Preprocessing features...")
    X, y, df_processed, feature_cols = preprocess_features(df_features)

    print("Creating train/calibration/latest split...")
    X_train, y_train, X_calib, y_calib, X_latest, latest_row = split_train_calib_latest(X, y, df_processed)

    print("Training calibrated rain model...")
    base_model, iso_model = train_calibrated_rain_model(X_train, y_train, X_calib, y_calib)

    raw_prob, calibrated_prob = get_latest_model_probability(base_model, iso_model, X_latest)
    print(f"Raw latest rain probability: {raw_prob:.4f}")
    print(f"Calibrated latest rain probability: {calibrated_prob:.4f}")

    print("Loading market data...")
    df_markets = load_market_data(engine)
    print(f"Loaded {len(df_markets)} markets from database.")

    print("Filtering weather-related markets...")
    df_weather_markets = filter_weather_markets(df_markets)
    print(f"Found {len(df_weather_markets)} weather-related markets.")

    print("Building ranked opportunities...")
    df_ranked = build_rankings(df_weather_markets, calibrated_prob)

    if df_ranked.empty:
        print("No rankable rain markets found yet.")
    else:
        print(f"Built {len(df_ranked)} ranked opportunities.")
        print("\nTop ranked opportunities:")
        print(df_ranked.head(20))

    print("Saving rankings to database...")
    save_rankings(engine, df_ranked)

    print("Done.")

if __name__ == "__main__":
    main()