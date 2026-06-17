import os
from datetime import datetime, timezone
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")

def get_engine():
    if not DB_URL:
        raise ValueError("DATABASE_URL is not set")
    return create_engine(DB_URL)

def normalize_price(x):
    if x is None:
        return None
    try:
        x = float(x)
        return x / 100.0 if x > 1 else x
    except:
        return None

def pick_price(row):
    for col in ["yes_ask_price", "last_price", "yes_bid_price"]:
        val = normalize_price(row.get(col))
        if val is not None:
            return val
    return None

def compute_model_probability(row):
    market_prob = normalize_price(row.get("implied_probability"))
    if market_prob is None:
        return None
    return market_prob

def main():
    engine = get_engine()

    query = """
    SELECT market_id, implied_probability, yes_ask_price, yes_bid_price, last_price
    FROM market_data
    WHERE category = 'climate'
    """
    df = pd.read_sql(query, engine)

    if df.empty:
        print("No rows found.")
        return

    df["price_used"] = df.apply(pick_price, axis=1)
    df["model_probability"] = df.apply(compute_model_probability, axis=1)
    df["edge"] = df["model_probability"] - df["implied_probability"].apply(normalize_price)
    df["ev_yes"] = df["model_probability"] - df["price_used"]
    now = datetime.now(timezone.utc)

    update_sql = text("""
        UPDATE market_data
        SET model_probability = :model_probability,
            price_used = :price_used,
            edge = :edge,
            ev_yes = :ev_yes,
            ev_updated_at = :ev_updated_at
        WHERE market_id = :market_id
    """)

    with engine.begin() as conn:
        for row in df.to_dict(orient="records"):
            conn.execute(update_sql, {
                "market_id": row["market_id"],
                "model_probability": row["model_probability"],
                "price_used": row["price_used"],
                "edge": row["edge"],
                "ev_yes": row["ev_yes"],
                "ev_updated_at": now,
            })

    print(f"Updated EV fields for {len(df)} markets.")

if __name__ == "__main__":
    main()