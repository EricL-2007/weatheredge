import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from psycopg2.extras import Json
from src.markets.market_classifier import classify_market

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")
BASE_URL = "https://external-api.kalshi.com/trade-api/v2"

CLIMATE_SERIES_KEYWORDS = [
    "temperature",
    "rain",
    "snow",
    "precipitation",
    "weather",
    "climate",
    "hurricane",
    "storm",
    "wildfire",
    "natural disaster",
    "lake mead",
]

EXCLUDE_SERIES_KEYWORDS = [
    "brainard",
    "rotten tomatoes",
    "dragon",
]

def get_engine():
    if not DB_URL:
        raise ValueError("DATABASE_URL is not set in your .env file")
    return create_engine(DB_URL)

def ensure_market_table(engine):
    create_sql = """
    CREATE TABLE IF NOT EXISTS market_data (
        id SERIAL PRIMARY KEY,
        source VARCHAR(50),
        market_id VARCHAR(255) UNIQUE,
        event_ticker VARCHAR(255),
        series_ticker VARCHAR(255),
        question TEXT,
        subtitle TEXT,
        yes_ask_price FLOAT,
        yes_bid_price FLOAT,
        last_price FLOAT,
        implied_probability FLOAT,
        volume FLOAT,
        open_interest FLOAT,
        status VARCHAR(50),
        category VARCHAR(100),
        close_date TIMESTAMP,
        market_type VARCHAR(50),
        city_name VARCHAR(100),
        raw_response JSONB,
        fetched_at TIMESTAMP
    );
    """

    alter_statements = [
        "ALTER TABLE market_data ADD COLUMN IF NOT EXISTS series_ticker VARCHAR(255);",
        "ALTER TABLE market_data ADD COLUMN IF NOT EXISTS category VARCHAR(100);",
        "ALTER TABLE market_data ADD COLUMN IF NOT EXISTS market_type VARCHAR(50);",
        "ALTER TABLE market_data ADD COLUMN IF NOT EXISTS city_name VARCHAR(100);",
        "ALTER TABLE market_data ADD COLUMN IF NOT EXISTS raw_response JSONB;"
    ]

    with engine.begin() as conn:
        conn.execute(text(create_sql))
        for stmt in alter_statements:
            conn.execute(text(stmt))

def get_json(url, params=None, retries=3):
    headers = {"Accept": "application/json", "User-Agent": "weatheredge/1.0"}
    last_err = None

    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=60)
            if response.status_code == 429:
                wait = 2 * (attempt + 1)
                print(f"Rate limited on {url} with params={params}. Sleeping {wait}s...")
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            last_err = e
            wait = 2 * (attempt + 1)
            print(f"Request failed ({e}). Retrying in {wait}s...")
            time.sleep(wait)

    raise last_err

def extract_list(payload, keys):
    if payload is None:
        return []

    for key in keys:
        if isinstance(payload.get(key), list):
            return payload[key]

    data_value = payload.get("data")
    if isinstance(data_value, dict):
        for key in keys:
            if isinstance(data_value.get(key), list):
                return data_value[key]
    elif isinstance(data_value, list):
        return data_value

    return []

def fetch_all_series(limit=20000):
    url = f"{BASE_URL}/series"
    payload = get_json(url, params={"limit": limit})
    return extract_list(payload, ["series"])

def fetch_markets_for_series(series_ticker, limit=500):
    url = f"{BASE_URL}/markets"
    payload = get_json(url, params={"series_ticker": series_ticker, "limit": limit})
    return extract_list(payload, ["markets"])

def is_climate_series(series_obj):
    title = (series_obj.get("title") or "").lower()
    subtitle = (series_obj.get("subtitle") or "").lower()
    ticker = (series_obj.get("ticker") or "").lower()
    category = (series_obj.get("category") or "").lower()
    combined = f"{title} {subtitle} {ticker} {category}"

    include = any(keyword in combined for keyword in CLIMATE_SERIES_KEYWORDS)
    exclude = any(keyword in combined for keyword in EXCLUDE_SERIES_KEYWORDS)

    return include and not exclude

def to_probability(price):
    if price is None:
        return None
    try:
        price = float(price)
        return price / 100.0 if price > 1 else price
    except (TypeError, ValueError):
        return None

def pick_price(m):
    for key in [
        "last_price",
        "last_price_dollars",
        "yes_ask",
        "yes_ask_dollars",
        "yes_bid",
        "yes_bid_dollars",
    ]:
        if m.get(key) is not None:
            return m.get(key)
    return None

def normalize_market(m, forced_series_ticker=None):
    question = m.get("title") or m.get("question") or ""
    subtitle = m.get("subtitle") or m.get("yes_sub_title") or m.get("no_sub_title") or ""

    last_price = pick_price(m)
    implied_probability = to_probability(last_price)

    meta = classify_market(question, subtitle)

    return {
        "source": "kalshi",
        "market_id": m.get("ticker"),
        "event_ticker": m.get("event_ticker"),
        "series_ticker": m.get("series_ticker") or forced_series_ticker,
        "question": question,
        "subtitle": subtitle,
        "yes_ask_price": m.get("yes_ask") or m.get("yes_ask_dollars"),
        "yes_bid_price": m.get("yes_bid") or m.get("yes_bid_dollars"),
        "last_price": last_price,
        "implied_probability": implied_probability,
        "volume": m.get("volume") or m.get("volume_dollars"),
        "open_interest": m.get("open_interest") or m.get("open_interest_fp"),
        "status": m.get("status") or "active",
        "category": "climate",
        "close_date": m.get("close_time"),
        "market_type": meta["market_type"],
        "city_name": meta["city_name"],
        "raw_response": Json(m),
        "fetched_at": datetime.now(timezone.utc)
    }

def upsert_markets(engine, rows):
    sql = text("""
        INSERT INTO market_data (
            source, market_id, event_ticker, series_ticker, question, subtitle,
            yes_ask_price, yes_bid_price, last_price, implied_probability,
            volume, open_interest, status, category, close_date,
            market_type, city_name, raw_response, fetched_at
        )
        VALUES (
            :source, :market_id, :event_ticker, :series_ticker, :question, :subtitle,
            :yes_ask_price, :yes_bid_price, :last_price, :implied_probability,
            :volume, :open_interest, :status, :category, :close_date,
            :market_type, :city_name, :raw_response, :fetched_at
        )
        ON CONFLICT (market_id)
        DO UPDATE SET
            event_ticker = EXCLUDED.event_ticker,
            series_ticker = EXCLUDED.series_ticker,
            question = EXCLUDED.question,
            subtitle = EXCLUDED.subtitle,
            yes_ask_price = EXCLUDED.yes_ask_price,
            yes_bid_price = EXCLUDED.yes_bid_price,
            last_price = EXCLUDED.last_price,
            implied_probability = EXCLUDED.implied_probability,
            volume = EXCLUDED.volume,
            open_interest = EXCLUDED.open_interest,
            status = EXCLUDED.status,
            category = EXCLUDED.category,
            close_date = EXCLUDED.close_date,
            market_type = EXCLUDED.market_type,
            city_name = EXCLUDED.city_name,
            raw_response = EXCLUDED.raw_response,
            fetched_at = EXCLUDED.fetched_at
    """)

    with engine.begin() as conn:
        for row in rows:
            conn.execute(sql, row)

def main():
    print("Connecting to database...")
    engine = get_engine()

    print("Ensuring market_data table and columns exist...")
    ensure_market_table(engine)

    print("Fetching all Kalshi series...")
    all_series = fetch_all_series()
    print(f"Fetched {len(all_series)} total series.")

    climate_series = [s for s in all_series if is_climate_series(s)]
    print(f"Matched {len(climate_series)} climate-related series.")

    if not climate_series:
        print("No climate-related series found.")
        return

    climate_series_df = pd.DataFrame(climate_series)
    preview_cols = [c for c in ["ticker", "title", "category"] if c in climate_series_df.columns]
    print(climate_series_df[preview_cols].head(30))

    all_rows = []

    for idx, series in enumerate(climate_series, start=1):
        series_ticker = series.get("ticker")
        if not series_ticker:
            continue

        print(f"[{idx}/{len(climate_series)}] Fetching markets for series: {series_ticker}")
        markets = fetch_markets_for_series(series_ticker)

        print(f"  Found {len(markets)} markets.")
        for market in markets:
            all_rows.append(normalize_market(market, forced_series_ticker=series_ticker))

        time.sleep(0.25)

    if not all_rows:
        print("No markets found from matched climate series.")
        return

    df = pd.DataFrame(all_rows)
    preview_cols = [c for c in ["series_ticker", "event_ticker", "question", "market_type", "city_name", "implied_probability"] if c in df.columns]
    print(df[preview_cols].head(30))

    print("Upserting into PostgreSQL...")
    upsert_markets(engine, all_rows)

    print("Done.")
    print(f"Inserted/updated {len(all_rows)} climate market rows into market_data.")

if __name__ == "__main__":
    main()
