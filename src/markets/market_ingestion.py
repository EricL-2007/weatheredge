import json
import requests
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timezone

DB_URL = "postgresql+psycopg2://ericliu:Ericmliu1234@127.0.0.1:5432/weatheredge"
KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

def get_engine():
    return create_engine(DB_URL)

def create_market_table(engine):
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS market_data (
        id SERIAL PRIMARY KEY,
        source VARCHAR(50),
        market_id VARCHAR(255) UNIQUE,
        event_ticker VARCHAR(255),
        question TEXT,
        subtitle TEXT,
        yes_ask_price FLOAT,
        yes_bid_price FLOAT,
        last_price FLOAT,
        implied_probability FLOAT,
        volume FLOAT,
        open_interest FLOAT,
        status VARCHAR(50),
        close_date TIMESTAMP,
        raw_response JSONB,
        fetched_at TIMESTAMP
    );
    """
    with engine.begin() as conn:
        conn.execute(text(create_table_sql))

def safe_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None

def normalize_probability(price_value):
    if price_value is None:
        return None
    if price_value > 1:
        return price_value / 100.0
    return price_value

def fetch_kalshi_markets(limit=100, cursor=None, status="open"):
    params = {
        "limit": limit,
        "status": status
    }
    if cursor:
        params["cursor"] = cursor

    url = f"{KALSHI_BASE_URL}/markets"
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()

def parse_kalshi_markets(payload):
    markets = payload.get("markets", [])
    rows = []

    for market in markets:
        market_id = market.get("ticker") or market.get("market_ticker")
        event_ticker = market.get("event_ticker")
        question = market.get("title") or market.get("question")
        subtitle = market.get("subtitle")

        yes_ask_price = safe_float(
            market.get("yes_ask")
            or market.get("yes_ask_price")
            or market.get("ask")
        )
        yes_bid_price = safe_float(
            market.get("yes_bid")
            or market.get("yes_bid_price")
            or market.get("bid")
        )
        last_price = safe_float(
            market.get("last_price")
            or market.get("yes_price")
            or market.get("price")
        )

        implied_probability = None
        for candidate in [last_price, yes_ask_price, yes_bid_price]:
            if candidate is not None:
                implied_probability = normalize_probability(candidate)
                break

        volume = safe_float(market.get("volume"))
        open_interest = safe_float(market.get("open_interest"))
        status = market.get("status")

        close_date = (
            market.get("close_time")
            or market.get("expiration_time")
            or market.get("settlement_time")
        )

        if close_date:
            try:
                close_date = pd.to_datetime(close_date, utc=True)
            except Exception:
                close_date = None

        rows.append({
            "source": "kalshi",
            "market_id": market_id,
            "event_ticker": event_ticker,
            "question": question,
            "subtitle": subtitle,
            "yes_ask_price": yes_ask_price,
            "yes_bid_price": yes_bid_price,
            "last_price": last_price,
            "implied_probability": implied_probability,
            "volume": volume,
            "open_interest": open_interest,
            "status": status,
            "close_date": close_date,
            "raw_response": market,
            "fetched_at": datetime.now(timezone.utc)
        })

    return pd.DataFrame(rows)

def upsert_market_data(engine, df):
    if df.empty:
        print("No market rows to insert.")
        return

    with engine.begin() as conn:
        for _, row in df.iterrows():
            upsert_sql = text("""
                INSERT INTO market_data (
                    source,
                    market_id,
                    event_ticker,
                    question,
                    subtitle,
                    yes_ask_price,
                    yes_bid_price,
                    last_price,
                    implied_probability,
                    volume,
                    open_interest,
                    status,
                    close_date,
                    raw_response,
                    fetched_at
                )
                VALUES (
                    :source,
                    :market_id,
                    :event_ticker,
                    :question,
                    :subtitle,
                    :yes_ask_price,
                    :yes_bid_price,
                    :last_price,
                    :implied_probability,
                    :volume,
                    :open_interest,
                    :status,
                    :close_date,
                    CAST(:raw_response AS JSONB),
                    :fetched_at
                )
                ON CONFLICT (market_id)
                DO UPDATE SET
                    source = EXCLUDED.source,
                    event_ticker = EXCLUDED.event_ticker,
                    question = EXCLUDED.question,
                    subtitle = EXCLUDED.subtitle,
                    yes_ask_price = EXCLUDED.yes_ask_price,
                    yes_bid_price = EXCLUDED.yes_bid_price,
                    last_price = EXCLUDED.last_price,
                    implied_probability = EXCLUDED.implied_probability,
                    volume = EXCLUDED.volume,
                    open_interest = EXCLUDED.open_interest,
                    status = EXCLUDED.status,
                    close_date = EXCLUDED.close_date,
                    raw_response = EXCLUDED.raw_response,
                    fetched_at = EXCLUDED.fetched_at
            """)

            payload = row.to_dict()
            payload["raw_response"] = json.dumps(payload["raw_response"], default=str)
            payload["close_date"] = row["close_date"].to_pydatetime() if pd.notnull(row["close_date"]) else None
            payload["fetched_at"] = row["fetched_at"]

            conn.execute(upsert_sql, payload)

def main():
    print("Connecting to database...")
    engine = get_engine()

    print("Creating market_data table if needed...")
    create_market_table(engine)

    print("Fetching Kalshi markets...")
    payload = fetch_kalshi_markets(limit=100, status="open")

    print("Parsing market payload...")
    df_markets = parse_kalshi_markets(payload)

    print(f"Fetched {len(df_markets)} markets.")

    if not df_markets.empty:
        preview_cols = [
            "market_id",
            "event_ticker",
            "question",
            "last_price",
            "implied_probability",
            "status",
            "close_date"
        ]
        print("\nSample markets:")
        print(df_markets[preview_cols].head(10))

    print("Upserting into PostgreSQL...")
    upsert_market_data(engine, df_markets)

    print("Done.")
    print(f"Inserted/updated {len(df_markets)} market rows into market_data.")

if __name__ == "__main__":
    main()