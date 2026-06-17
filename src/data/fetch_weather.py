from pathlib import Path
import json
import requests
import pandas as pd
from sqlalchemy import create_engine, text

RAW_DIR = Path("data/raw/open_meteo")
RAW_DIR.mkdir(parents=True, exist_ok=True)

DB_URL = "postgresql+psycopg2://ericliu:Ericmliu1234@127.0.0.1:5432/weatheredge"

CITIES = [
    {"city": "Houston", "state": "TX", "latitude": 29.7604, "longitude": -95.3698, "elevation": 13},
    {"city": "Austin", "state": "TX", "latitude": 30.2672, "longitude": -97.7431, "elevation": 149},
    {"city": "Dallas", "state": "TX", "latitude": 32.7767, "longitude": -96.7970, "elevation": 131},
    {"city": "Denver", "state": "CO", "latitude": 39.7392, "longitude": -104.9903, "elevation": 1609},
    {"city": "Chicago", "state": "IL", "latitude": 41.8781, "longitude": -87.6298, "elevation": 181},
    {"city": "New York", "state": "NY", "latitude": 40.7128, "longitude": -74.0060, "elevation": 10},
    {"city": "Los Angeles", "state": "CA", "latitude": 34.0522, "longitude": -118.2437, "elevation": 71},
    {"city": "Seattle", "state": "WA", "latitude": 47.6062, "longitude": -122.3321, "elevation": 53},
]

def fetch_weather(city_info, start_date="2024-01-01", end_date="2024-01-07"):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": city_info["latitude"],
        "longitude": city_info["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "wind_speed_10m",
            "precipitation",
            "cloud_cover"
        ],
        "timezone": "auto"
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()

def save_raw_json(city_name, payload):
    safe_name = city_name.lower().replace(" ", "_")
    out_file = RAW_DIR / f"{safe_name}_weather.json"
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)

def to_dataframe(city_info, payload):
    hourly = payload["hourly"]
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

def ensure_city_metadata(engine, city_info):
    with engine.begin() as conn:
        result = conn.execute(
            text("SELECT 1 FROM city_metadata WHERE city = :city"),
            {"city": city_info["city"]}
        ).fetchone()

        if result is None:
            meta_df = pd.DataFrame([city_info])
            meta_df.to_sql("city_metadata", conn, if_exists="append", index=False)

def delete_existing_weather_rows(engine, city_name, start_date, end_date):
    with engine.begin() as conn:
        conn.execute(
            text("""
                DELETE FROM weather_data
                WHERE city = :city
                  AND date >= :start_date
                  AND date <= :end_date
            """),
            {
                "city": city_name,
                "start_date": f"{start_date} 00:00:00",
                "end_date": f"{end_date} 23:59:59",
            }
        )

def insert_weather(engine, df):
    df.to_sql("weather_data", engine, if_exists="append", index=False, method="multi")

def main():
    start_date = "2024-01-01"
    end_date = "2024-01-07"

    engine = create_engine(DB_URL)

    all_counts = []

    for city in CITIES:
        print(f"\n--- Processing {city['city']} ---")

        payload = fetch_weather(city, start_date, end_date)
        save_raw_json(city["city"], payload)
        df = to_dataframe(city, payload)

        ensure_city_metadata(engine, city)
        delete_existing_weather_rows(engine, city["city"], start_date, end_date)
        insert_weather(engine, df)

        print(df.head())
        print(f"Inserted {len(df)} rows for {city['city']}")
        all_counts.append((city["city"], len(df)))

    print("\n=== Summary ===")
    for city_name, count in all_counts:
        print(f"{city_name}: {count} rows")

if __name__ == "__main__":
    main()