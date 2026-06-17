from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text, inspect

DB_URL = "postgresql+psycopg2://ericliu:Ericmliu1234@127.0.0.1:5432/weatheredge"

def load_weather_data(engine):
    query = """
        SELECT
            id,
            date,
            city,
            temperature,
            humidity,
            pressure,
            wind_speed,
            precipitation,
            cloud_cover
        FROM weather_data
        ORDER BY city, date
    """
    return pd.read_sql(query, engine)

def add_calendar_features(df):
    df = df.copy()

    df["month"] = df["date"].dt.month
    df["week"] = df["date"].dt.isocalendar().week
    df["quarter"] = df["date"].dt.quarter
    df["day_of_week"] = df["date"].dt.dayofweek

    def season_from_month(m):
        if m in [3, 4, 5]:
            return "spring"
        elif m in [6, 7, 8]:
            return "summer"
        elif m in [9, 10, 11]:
            return "fall"
        else:
            return "winter"

    df["season"] = df["month"].apply(season_from_month)

    return df

def add_lag_features(df):
    df = df.copy()

    df = df.sort_values(["city", "date"]).reset_index(drop=True)

    for col in ["temperature", "precipitation"]:
        for lag_days in [1, 3, 7]:
            lag_hours = lag_days * 24
            df[f"{col}_{lag_days}d"] = df.groupby("city")[col].shift(lag_hours)

    return df

def add_rolling_features(df):
    df = df.copy()

    df = df.sort_values(["city", "date"])

    for col in ["temperature", "precipitation"]:
        for window_hours in [7*24, 14*24, 30*24]:
            df[f"{col}_{window_hours//24}d_rolling_avg"] = (
                df.groupby("city")[col]
                .transform(lambda x: x.rolling(window=window_hours, min_periods=1).mean())
            )

    return df

def add_trend_features(df):
    df = df.copy()

    df = df.sort_values(["city", "date"])

    df["temp_change_1d"] = df.groupby("city")["temperature"].diff(24)
    df["rain_trend_1d"] = df.groupby("city")["precipitation"].diff(24)
    df["wind_trend_1d"] = df.groupby("city")["wind_speed"].diff(24)

    return df

def build_feature_table(engine):
    df = load_weather_data(engine)
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_trend_features(df)

    return df

def save_features(engine, df):
    inspector = inspect(engine)

    if inspector.has_table("feature_table"):
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE feature_table"))

    df.to_sql("feature_table", engine, if_exists="append", index=False, method="multi")

def main():
    print("Loading weather data...")
    engine = create_engine(DB_URL)

    df = build_feature_table(engine)

    print("Data shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head())

    print("Saving feature table...")
    save_features(engine, df)

    print("Feature table saved to PostgreSQL.")

if __name__ == "__main__":
    main()