import requests
import pandas as pd
from sqlalchemy import create_engine, text

DB_URL = "postgresql+psycopg2://ericliu:Ericmliu1234@127.0.0.1:5432/weatheredge"

CITIES = [
    {"city": "Houston", "latitude": 29.7604, "longitude": -95.3698},
    {"city": "Austin", "latitude": 30.2672, "longitude": -97.7431},
    {"city": "Dallas", "latitude": 32.7767, "longitude": -96.7970},
    {"city": "Denver", "latitude": 39.7392, "longitude": -104.9903},
    {"city": "Chicago", "latitude": 41.8781, "longitude": -87.6298},
    {"city": "New York", "latitude": 40.7128, "longitude": -74.0060},
    {"city": "Los Angeles", "latitude": 34.0522, "longitude": -118.2437},
    {"city": "Seattle", "latitude": 47.6062, "longitude": -122.3321},
]

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

def get_engine():
    return create_engine(DB_URL)

def create_weather_table_if_needed(engine):
    create_sql = """
    CREATE TABLE IF NOT EXISTS weather_data (
        id SERIAL PRIMARY KEY,
        date TIMESTAMP,
        city VARCHAR(100),
        temperature FLOAT,
        humidity FLOAT,
        pressure FLOAT,
        wind_speed FLOAT,
        precipitation FLOAT,
        cloud_cover FLOAT,
        UNIQUE(date, city)
    );
    """
    with engine.begin() as conn:
        conn.execute(text(create_sql))

def fetch_recent_weather(city_info, start_date="2025-01-01", end_date="2026-06-16"):
    params = {
        "latitude": city_info["latitude"],
        "longitude": city_info["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "wind_speed_10m",
            "precipitation",
            "cloud_cover"
        ]),
        "timezone": "auto"
    }

    response = requests.get(ARCHIVE_URL, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    hourly = data["hourly"]

    df = pd.DataFrame({
        "date": pd.to_datetime(hourly["time"]),
        "city": city_info["city"],
        "temperature": hourly["temperature_2m"],
        "humidity": hourly["relative_humidity_2m"],
        "pressure": hourly["surface_pressure"],
        "wind_speed": hourly["wind_speed_10m"],
        "precipitation": hourly["precipitation"],
        "cloud_cover": hourly["cloud_cover"],
    })

    return df

def upsert_weather_data(engine, df):
    if df.empty:
        print("No rows to insert.")
        return

    upsert_sql = text("""
        INSERT INTO weather_data (
            date,
            city,
            temperature,
            humidity,
            pressure,
            wind_speed,
            precipitation,
            cloud_cover
        )
        VALUES (
            :date,
            :city,
            :temperature,
            :humidity,
            :pressure,
            :wind_speed,
            :precipitation,
            :cloud_cover
        )
        ON CONFLICT (date, city)
        DO UPDATE SET
            temperature = EXCLUDED.temperature,
            humidity = EXCLUDED.humidity,
            pressure = EXCLUDED.pressure,
            wind_speed = EXCLUDED.wind_speed,
            precipitation = EXCLUDED.precipitation,
            cloud_cover = EXCLUDED.cloud_cover
    """)

    with engine.begin() as conn:
        for _, row in df.iterrows():
            payload = row.to_dict()
            conn.execute(upsert_sql, payload)

def main():
    print("Connecting to database...")
    engine = get_engine()

    print("Ensuring weather_data table exists...")
    create_weather_table_if_needed(engine)

    all_frames = []

    for city in CITIES:
        print(f"Fetching recent weather for {city['city']}...")
        df_city = fetch_recent_weather(city)
        print(f"Fetched {len(df_city)} rows for {city['city']}.")
        all_frames.append(df_city)

    final_df = pd.concat(all_frames, ignore_index=True)

    print(f"Total rows fetched: {len(final_df)}")
    print("Upserting rows into PostgreSQL...")
    upsert_weather_data(engine, final_df)

    print("Done updating recent weather data.")

if __name__ == "__main__":
    main()