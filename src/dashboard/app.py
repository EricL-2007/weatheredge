import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import re
import math
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split


try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

from src.models.weather_calibration import (
    run_calibration_on_dashboard_df,
    compute_sane_recommended_bet,
)

load_dotenv(find_dotenv(), override=True)

st.set_page_config(page_title="WeatherEdge Dashboard", layout="wide")

DB_URL = st.secrets.get("DATABASE_URL", os.getenv("DATABASE_URL"))
if not DB_URL:
    st.error("DATABASE_URL is missing from Streamlit secrets or .env")
    st.stop()

engine = create_engine(DB_URL, pool_pre_ping=True)

KNOWN_CITIES = {
    "Austin", "Chicago", "Dallas", "Denver", "Houston",
    "Los Angeles", "New York", "Seattle", "Philadelphia", "Miami"
}

CITY_BASELINES = {
    "Austin": {"summer_high": 96, "summer_low": 74},
    "Chicago": {"summer_high": 83, "summer_low": 65},
    "Dallas": {"summer_high": 94, "summer_low": 75},
    "Denver": {"summer_high": 85, "summer_low": 57},
    "Houston": {"summer_high": 92, "summer_low": 76},
    "Los Angeles": {"summer_high": 79, "summer_low": 61},
    "New York": {"summer_high": 83, "summer_low": 68},
    "Seattle": {"summer_high": 73, "summer_low": 56},
    "Philadelphia": {"summer_high": 84, "summer_low": 66},
    "Miami": {"summer_high": 89, "summer_low": 78},
}


@st.cache_data(ttl=300)
def load_data():
    query = """
    SELECT
        question,
        city_name,
        market_type,
        implied_probability,
        model_probability,
        edge,
        ev_yes,
        price_used,
        fetched_at,
        ev_updated_at
    FROM market_data
    WHERE category = 'climate'
      AND implied_probability IS NOT NULL
      AND model_probability IS NOT NULL
    """
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)


def extract_market_numbers(question: str):
    try:
        if not isinstance(question, str):
            return pd.Series([np.nan, np.nan, None])

        q = question.replace("°", "")

        range_match = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", q)
        if range_match is not None:
            return pd.Series([
                float(range_match.group(1)),
                float(range_match.group(2)),
                "range"
            ])

        gt_match = re.search(r">\s*(\d+(?:\.\d+)?)", q)
        if gt_match is not None:
            return pd.Series([float(gt_match.group(1)), np.nan, "greater_than"])

        lt_match = re.search(r"<\s*(\d+(?:\.\d+)?)", q)
        if lt_match is not None:
            return pd.Series([float(lt_match.group(1)), np.nan, "less_than"])

        above_match = re.search(r"above\s*(\d+(?:\.\d+)?)", q, re.IGNORECASE)
        if above_match is not None:
            return pd.Series([float(above_match.group(1)), np.nan, "greater_than"])

        below_match = re.search(r"below\s*(\d+(?:\.\d+)?)", q, re.IGNORECASE)
        if below_match is not None:
            return pd.Series([float(below_match.group(1)), np.nan, "less_than"])

        return pd.Series([np.nan, np.nan, None])
    except Exception:
        return pd.Series([np.nan, np.nan, None])


def extract_datetime_info(question: str):
    try:
        if not isinstance(question, str):
            return pd.Series([None, None, None])

        month_match = re.search(
            r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2}),\s+(\d{4})",
            question,
            re.IGNORECASE
        )
        hour_match = re.search(r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", question, re.IGNORECASE)

        month_num = None
        day_num = None
        hour_24 = None

        if month_match is not None:
            month_lookup = {
                "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
            }
            month_num = month_lookup[month_match.group(1).lower()[:3]]
            day_num = int(month_match.group(2))

        if hour_match is not None:
            hour = int(hour_match.group(1))
            meridiem = hour_match.group(3).lower()
            if meridiem == "pm" and hour != 12:
                hour += 12
            if meridiem == "am" and hour == 12:
                hour = 0
            hour_24 = hour

        return pd.Series([month_num, day_num, hour_24])
    except Exception:
        return pd.Series([None, None, None])


def infer_city(question, fallback_city):
    if isinstance(fallback_city, str) and fallback_city.strip() and fallback_city.strip() in KNOWN_CITIES:
        return fallback_city.strip()

    if not isinstance(question, str):
        return None

    normalized = question.lower().replace("new york city", "new york").replace("nyc", "new york")
    for city in sorted(KNOWN_CITIES, key=len, reverse=True):
        if city.lower() in normalized:
            return city

    return None


def classify_market_family(question: str, market_type: str, city_name: str, target_type: str):
    q = question.lower() if isinstance(question, str) else ""
    mt = market_type.lower() if isinstance(market_type, str) else ""

    weather_keywords = [
        "temperature", "temp", "high temp", "low temp",
        "maximum temperature", "minimum temperature"
    ]
    if any(k in q for k in weather_keywords) or city_name in KNOWN_CITIES:
        if target_type in {"greater_than", "less_than", "range"}:
            return "temperature"

    if "precip" in mt or "rain" in q or "snow" in q:
        return "precipitation"
    if "hurricane" in mt or "hurricane" in q:
        return "hurricane"
    if "lake mead" in q or "water elevation" in q or "reservoir" in q:
        return "water_level"

    return "unknown"


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def estimate_hourly_temp(city, month_num, hour_24):
    baseline = CITY_BASELINES.get(city)
    if baseline is None or month_num is None:
        return np.nan

    high = baseline["summer_high"]
    low = baseline["summer_low"]

    if month_num in [12, 1, 2]:
        high -= 22
        low -= 20
    elif month_num in [3, 4, 11]:
        high -= 10
        low -= 9
    elif month_num in [5, 10]:
        high -= 4
        low -= 3
    elif month_num in [6, 7, 8]:
        pass
    elif month_num == 9:
        high -= 2
        low -= 2

    if hour_24 is None:
        return (high + low) / 2

    hour_curve = {
        0: low, 1: low - 1, 2: low - 1, 3: low - 2, 4: low - 2, 5: low - 2,
        6: low - 1, 7: low, 8: low + 2, 9: low + 5, 10: low + 8, 11: low + 11,
        12: low + 14, 13: low + 16, 14: low + 18, 15: high, 16: high - 1,
        17: high - 2, 18: high - 4, 19: high - 7, 20: high - 10, 21: low + 4,
        22: low + 2, 23: low + 1
    }
    return hour_curve.get(hour_24, (high + low) / 2)


def weather_probability_from_threshold(row, temp_sd=4.5):
    try:
        if row.get("market_family") != "temperature":
            return np.nan

        city = row.get("resolved_city")
        month_num = row.get("month_num")
        hour_24 = row.get("hour_24")
        target_type = row.get("target_type")
        target_low = row.get("target_low")
        target_high = row.get("target_high")

        mean_temp = estimate_hourly_temp(city, month_num, hour_24)
        if pd.isna(mean_temp):
            return np.nan

        if target_type == "greater_than" and pd.notna(target_low):
            z = (target_low - mean_temp) / temp_sd
            return float(1 - norm_cdf(z))

        if target_type == "less_than" and pd.notna(target_low):
            z = (target_low - mean_temp) / temp_sd
            return float(norm_cdf(z))

        if target_type == "range" and pd.notna(target_low) and pd.notna(target_high):
            z_low = (target_low - mean_temp) / temp_sd
            z_high = (target_high - mean_temp) / temp_sd
            return float(max(0.0, norm_cdf(z_high) - norm_cdf(z_low)))

        return np.nan
    except Exception:
        return np.nan


def safe_corr(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    mask = a.notna() & b.notna()
    if mask.sum() < 3:
        return np.nan
    if a[mask].nunique() <= 1 or b[mask].nunique() <= 1:
        return np.nan
    return float(a[mask].corr(b[mask]))


def classify_model_health(row):
    prob = row.get("calibrated_model_probability", row.get("enhanced_model_probability"))
    market = row.get("implied_probability")
    ev = row.get("best_ev")
    if pd.isna(prob) or pd.isna(market):
        return "missing"
    gap = abs(prob - market)
    if gap >= 0.15 and pd.notna(ev) and ev > 0:
        return "strong_disagreement"
    if gap >= 0.08:
        return "moderate_disagreement"
    return "aligned"


df = load_data().copy()

for c in ["implied_probability", "model_probability", "edge", "ev_yes", "price_used"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["implied_probability", "model_probability"]).copy()
df = df[(df["implied_probability"] > 0) & (df["implied_probability"] < 1)].copy()

df[["target_low", "target_high", "target_type"]] = df["question"].apply(extract_market_numbers)
df[["month_num", "day_num", "hour_24"]] = df["question"].apply(extract_datetime_info)

df["resolved_city"] = df.apply(lambda row: infer_city(row.get("question"), row.get("city_name")), axis=1)

df["market_family"] = df.apply(
    lambda row: classify_market_family(
        row.get("question"),
        row.get("market_type"),
        row.get("resolved_city"),
        row.get("target_type")
    ),
    axis=1
)

df["model_supported"] = (
    (df["market_family"] == "temperature") &
    (df["resolved_city"].notna()) &
    (df["target_type"].notna()) &
    (df["month_num"].notna())
)

df["historical_probability"] = df.apply(weather_probability_from_threshold, axis=1)
df["historical_probability"] = pd.to_numeric(df["historical_probability"], errors="coerce").clip(lower=0.001, upper=0.999)

historical_weight = 0.65
market_weight = 0.10
existing_model_weight = 0.25

df["enhanced_model_probability"] = np.where(
    df["model_supported"] & df["historical_probability"].notna(),
    (
        historical_weight * df["historical_probability"] +
        market_weight * df["implied_probability"] +
        existing_model_weight * df["model_probability"]
    ),
    df["model_probability"]
)

df["no_market_probability"] = 1 - df["implied_probability"]
df["no_model_probability"] = 1 - df["enhanced_model_probability"]

df["yes_edge"] = df["enhanced_model_probability"] - df["implied_probability"]
df["no_edge"] = df["no_model_probability"] - df["no_market_probability"]

if "bankroll" not in st.session_state:
    st.session_state.bankroll = 1000.0
if "min_bet_floor" not in st.session_state:
    st.session_state.min_bet_floor = 5.0
if "min_edge_to_bet" not in st.session_state:
    st.session_state.min_edge_to_bet = 0.01
if "watchlist_stake" not in st.session_state:
    st.session_state.watchlist_stake = 2.0
if "watchlist_confidence" not in st.session_state:
    st.session_state.watchlist_confidence = 0.60
if "min_win_probability" not in st.session_state:
    st.session_state.min_win_probability = 0.60

st.sidebar.header("Filters")

cities = sorted([c for c in df["resolved_city"].dropna().unique().tolist() if c])
selected_cities = st.sidebar.multiselect("City", cities, default=[])

market_types = sorted([m for m in df["market_type"].dropna().unique().tolist() if m])
selected_market_types = st.sidebar.multiselect("Market type", market_types, default=[])

families = sorted([m for m in df["market_family"].dropna().unique().tolist() if m])
selected_families = st.sidebar.multiselect("Model family", families, default=[])

st.sidebar.header("Bet rules")
min_edge_to_bet = st.sidebar.number_input(
    "Minimum edge to bet",
    min_value=0.0,
    max_value=1.0,
    value=float(st.session_state.min_edge_to_bet),
    step=0.005,
    format="%.3f",
    key="min_edge_to_bet",
)

min_win_probability = st.sidebar.number_input(
    "Minimum win probability",
    min_value=0.0,
    max_value=1.0,
    value=float(st.session_state.min_win_probability),
    step=0.01,
    format="%.2f",
    key="min_win_probability",
)

mode = st.sidebar.selectbox(
    "Mode",
    ["Real bets only", "Real bets + watchlist"],
    index=1
)

watchlist_confidence = st.sidebar.number_input(
    "Watchlist min confidence",
    min_value=0.50,
    max_value=1.00,
    value=float(st.session_state.watchlist_confidence),
    step=0.01,
    format="%.2f",
    key="watchlist_confidence",
)

top_n = st.sidebar.selectbox("Rows to show", [25, 50, 100, 250, 500], index=2)
supported_only = st.sidebar.checkbox("Supported model rows only", value=False)

st.sidebar.header("Bet sizing")
bankroll = st.sidebar.number_input(
    "Bankroll ($)",
    min_value=0.0,
    value=float(st.session_state.bankroll),
    key="bankroll"
)
kelly_fraction = st.sidebar.selectbox("Kelly fraction", [1.0, 0.5, 0.25, 0.1], index=2)
min_bet_floor = st.sidebar.number_input(
    "Minimum real-bet floor ($)",
    min_value=0.0,
    value=float(st.session_state.min_bet_floor),
    key="min_bet_floor"
)
watchlist_stake = st.sidebar.number_input(
    "Watchlist manual stake ($)",
    min_value=0.0,
    value=float(st.session_state.watchlist_stake),
    key="watchlist_stake"
)

df["bet_side"] = "PASS"

yes_mask = (
    df["model_supported"] &
    df["yes_edge"].notna() &
    (df["yes_edge"] >= min_edge_to_bet) &
    (df["enhanced_model_probability"] >= min_win_probability)
)

no_mask = (
    df["model_supported"] &
    df["no_edge"].notna() &
    (df["no_edge"] >= min_edge_to_bet) &
    ((1 - df["enhanced_model_probability"]) >= min_win_probability)
)

df.loc[yes_mask, "bet_side"] = "YES"
df.loc[no_mask, "bet_side"] = "NO"

df["best_edge"] = np.select(
    [df["bet_side"] == "YES", df["bet_side"] == "NO"],
    [df["yes_edge"], df["no_edge"]],
    default=np.nan
)

df["best_ev"] = np.select(
    [df["bet_side"] == "YES", df["bet_side"] == "NO"],
    [df["yes_edge"], df["no_edge"]],
    default=np.nan
)

df["market_price_for_bet"] = np.select(
    [df["bet_side"] == "YES", df["bet_side"] == "NO"],
    [df["implied_probability"], df["no_market_probability"]],
    default=np.nan
)

df["model_probability_for_bet"] = np.select(
    [df["bet_side"] == "YES", df["bet_side"] == "NO"],
    [df["enhanced_model_probability"], 1 - df["enhanced_model_probability"]],
    default=np.nan
)

df["decimal_odds_for_bet"] = np.where(
    df["market_price_for_bet"] > 0,
    1 / df["market_price_for_bet"],
    np.nan
)

df["net_odds_for_bet"] = df["decimal_odds_for_bet"] - 1

df["kelly_fraction_full"] = np.where(
    (df["bet_side"].isin(["YES", "NO"])) & (df["net_odds_for_bet"] > 0),
    (
        (df["net_odds_for_bet"] * df["model_probability_for_bet"])
        - (1 - df["model_probability_for_bet"])
    ) / df["net_odds_for_bet"],
    0.0
)

df["kelly_fraction_full"] = df["kelly_fraction_full"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
df["kelly_fraction_full"] = df["kelly_fraction_full"].clip(lower=0, upper=0.25)

df["confidence"] = np.where(
    df["enhanced_model_probability"].notna(),
    np.maximum(df["enhanced_model_probability"], 1 - df["enhanced_model_probability"]),
    df["model_probability"].fillna(0.5)
)

df["watchlist_side"] = np.where(
    df["enhanced_model_probability"].fillna(df["model_probability"]).fillna(0.5) >= 0.5,
    "YES",
    "NO"
)

df["display_side"] = np.where(
    df["bet_side"].isin(["YES", "NO"]),
    df["bet_side"],
    df["watchlist_side"]
)

df["row_type"] = np.where(
    df["bet_side"].isin(["YES", "NO"]),
    "REAL BET",
    "WATCHLIST"
)

df["reason"] = np.select(
    [
        ~df["model_supported"],
        df["bet_side"].isin(["YES", "NO"]),
    ],
    [
        "Unsupported model family or missing city/date/threshold",
        "Passed model support and bet rules",
    ],
    default="Supported but did not pass edge/probability thresholds"
)

filtered = df.copy()

if selected_cities:
    filtered = filtered[filtered["resolved_city"].isin(selected_cities)]

if selected_market_types:
    filtered = filtered[filtered["market_type"].isin(selected_market_types)]

if selected_families:
    filtered = filtered[filtered["market_family"].isin(selected_families)]

if supported_only:
    filtered = filtered[filtered["model_supported"]]

if mode == "Real bets only":
    filtered = filtered[filtered["row_type"] == "REAL BET"]
else:
    filtered = filtered[
        (filtered["row_type"] == "REAL BET") |
        (
            (filtered["row_type"] == "WATCHLIST") &
            (filtered["confidence"] >= watchlist_confidence)
        )
    ]

ev_series = filtered["best_ev"].dropna()
min_ev = float(ev_series.min()) if not ev_series.empty else 0.0
max_ev = float(ev_series.max()) if not ev_series.empty else 0.0

if min_ev >= max_ev:
    st.sidebar.caption(f"EV range: fixed at {min_ev:.4f}")
else:
    ev_range = st.sidebar.slider(
        "EV range",
        min_value=min_ev,
        max_value=max_ev,
        value=(min_ev, max_ev),
    )
    filtered = filtered[
        filtered["best_ev"].fillna(min_ev).between(ev_range[0], ev_range[1])
        | filtered["best_ev"].isna()
    ]

calibration_error = None
try:
    filtered, trained_model, calibration_metrics = run_calibration_on_dashboard_df(filtered)
except Exception as e:
    calibration_error = e
    trained_model = None
    calibration_metrics = None

if calibration_error is not None:
    st.error("Calibration failed")
    st.exception(calibration_error)

if calibration_metrics is not None:
    st.sidebar.caption(
        f"Calibration: log_loss={calibration_metrics['log_loss']:.4f}, "
        f"brier={calibration_metrics['brier_score']:.4f}"
    )
else:
    st.sidebar.caption("Calibration: unavailable")

filtered["calibrated_model_probability"] = filtered.get("calibrated_model_probability", filtered["enhanced_model_probability"])
filtered["calibrated_yes_edge"] = filtered.get("calibrated_yes_edge", filtered["yes_edge"])
filtered["calibrated_no_edge"] = filtered.get("calibrated_no_edge", filtered["no_edge"])

filtered["model_probability_for_bet"] = np.where(
    filtered["bet_side"] == "YES",
    filtered["calibrated_model_probability"].fillna(filtered["enhanced_model_probability"]),
    (1 - filtered["calibrated_model_probability"]).fillna(1 - filtered["enhanced_model_probability"]),
)

filtered["best_ev"] = np.where(
    filtered["bet_side"] == "YES",
    filtered["calibrated_yes_edge"].fillna(filtered["yes_edge"]),
    filtered["calibrated_no_edge"].fillna(filtered["no_edge"]),
)

filtered["prob_gap"] = (
    filtered["calibrated_model_probability"].fillna(filtered["enhanced_model_probability"])
    - filtered["implied_probability"]
)
filtered["abs_prob_gap"] = filtered["prob_gap"].abs()
filtered["market_disagreement"] = np.where(
    filtered["prob_gap"] > 0, "model_higher", "market_higher"
)
filtered["model_health_flag"] = filtered.apply(classify_model_health, axis=1)

filtered["kelly_fraction_full"] = np.where(
    (filtered["bet_side"].isin(["YES", "NO"])) & (filtered["net_odds_for_bet"] > 0),
    (
        (filtered["net_odds_for_bet"] * filtered["model_probability_for_bet"])
        - (1 - filtered["model_probability_for_bet"])
    ) / filtered["net_odds_for_bet"],
    0.0,
)

filtered["kelly_fraction_full"] = (
    filtered["kelly_fraction_full"]
    .replace([np.inf, -np.inf], 0.0)
    .fillna(0.0)
    .clip(lower=0, upper=0.25)
)

filtered["kelly_bet_raw"] = bankroll * filtered["kelly_fraction_full"] * kelly_fraction

filtered["recommended_bet"] = filtered.apply(
    lambda row: watchlist_stake if row["row_type"] == "WATCHLIST" else compute_sane_recommended_bet(
        bankroll=float(bankroll),
        raw_kelly_bet=float(row.get("kelly_bet_raw", 0.0)),
        edge=float(abs(row.get("best_ev", 0.0))),
        calibrated_prob=float(row.get("calibrated_model_probability", 0.5)),
        calibration_metrics=calibration_metrics,
        min_bet_floor=float(min_bet_floor),
        max_bet_floor=50.0,
        max_edge_to_bet=0.20,
    ),
    axis=1,
)

filtered["profit_if_win_recommended"] = filtered["recommended_bet"] * filtered["net_odds_for_bet"]
filtered["expected_profit_recommended"] = np.where(
    filtered["row_type"] == "REAL BET",
    filtered["recommended_bet"] * abs(filtered["best_ev"]),
    np.nan,
)

filtered["bet_quality"] = np.select(
    [
        (filtered["row_type"] == "REAL BET") & (abs(filtered["best_ev"]) >= 0.05),
        (filtered["row_type"] == "REAL BET") & (abs(filtered["best_ev"]) >= 0.02),
        filtered["row_type"] == "WATCHLIST",
    ],
    [
        "high",
        "medium",
        "watchlist",
    ],
    default="low"
)

filtered["sort_score"] = np.where(
    filtered["row_type"] == "REAL BET",
    filtered["model_probability_for_bet"] * 0.7 + abs(filtered["best_ev"].fillna(0)) * 0.3,
    filtered["confidence"]
)

filtered = filtered.sort_values(
    ["row_type", "sort_score", "confidence"],
    ascending=[True, False, False]
)

display_df = filtered.head(top_n).copy()

st.title("WeatherEdge")

real_bets_df = display_df[display_df["row_type"] == "REAL BET"].copy()
watchlist_df = display_df[display_df["row_type"] == "WATCHLIST"].copy()

total_real_stake = float(real_bets_df["recommended_bet"].sum()) if len(real_bets_df) else 0.0
total_watchlist_stake = float(watchlist_df["recommended_bet"].sum()) if len(watchlist_df) else 0.0
total_expected_profit = float(real_bets_df["expected_profit_recommended"].fillna(0).sum()) if len(real_bets_df) else 0.0
total_expected_out = total_real_stake + total_expected_profit
expected_roi = (total_expected_profit / total_real_stake) if total_real_stake > 0 else 0.0
supported_count = int(display_df["model_supported"].sum()) if len(display_df) else 0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Markets", f"{len(display_df):,}")
k2.metric("Supported", f"{supported_count:,}")
k3.metric("Real bets", f"{len(real_bets_df):,}")
k4.metric("Watchlist", f"{len(watchlist_df):,}")
k5.metric("Avg real EV", f"{real_bets_df['best_ev'].abs().mean():.4f}" if len(real_bets_df) else "0.0000")

k6, k7, k8, k9 = st.columns(4)
k6.metric("Money in", f"${total_real_stake:,.2f}")
k7.metric("Expected profit", f"${total_expected_profit:,.2f}")
k8.metric("Expected out", f"${total_expected_out:,.2f}")
k9.metric("Expected ROI", f"{expected_roi:.2%}")

st.caption(
    f"Real bets: put in ${total_real_stake:,.2f} -> expected out ${total_expected_out:,.2f} "
    f"(profit ${total_expected_profit:,.2f}, ROI {expected_roi:.2%})"
)
st.caption(f"Watchlist manual stake total: ${total_watchlist_stake:,.2f}")

st.subheader("Model diagnostics")

ev_corr = safe_corr(filtered.get("best_ev").abs(), filtered.get("model_probability_for_bet"))

d1, d2, d3, d4, d5 = st.columns(5)
d1.metric("Avg prob gap", f"{filtered['abs_prob_gap'].mean():.4f}" if len(filtered) else "0.0000")
d2.metric("Strong disagreements", f"{(filtered['model_health_flag'] == 'strong_disagreement').sum():,}")
d3.metric("Calibration log loss", f"{calibration_metrics['log_loss']:.4f}" if calibration_metrics else "N/A")
d4.metric("Calibration Brier", f"{calibration_metrics['brier_score']:.4f}" if calibration_metrics else "N/A")
d5.metric("EV correlation", f"{ev_corr:.4f}" if pd.notna(ev_corr) else "N/A")

#st.error("DEBUG: YOU REACHED BENCHMARK")
st.subheader("Model benchmark")

benchmark_df = filtered.copy()

resolved_target_col = None
for c in ["resolved_outcome", "actual_outcome", "settled_outcome", "result"]:
    if c in benchmark_df.columns:
        resolved_target_col = c
        break

benchmark_results_df = pd.DataFrame(columns=[
    "model", "log_loss", "brier_score", "roc_auc", "avg_pred", "test_rows", "status"
])

benchmark_warning = None
benchmark_caption = None

if resolved_target_col is None:
    benchmark_warning = (
        "Benchmark disabled: no real resolved outcome column found. "
        "Current YES/NO bet decisions are strategy outputs, not ground-truth labels."
    )
else:
    benchmark_df["target"] = benchmark_df[resolved_target_col]

    if benchmark_df["target"].dtype == "object":
        benchmark_df["target"] = (
            benchmark_df["target"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({
                "yes": 1, "true": 1, "1": 1,
                "no": 0, "false": 0, "0": 0,
            })
        )

    benchmark_df["target"] = pd.to_numeric(benchmark_df["target"], errors="coerce")

    time_col = None
    for c in ["ev_updated_at", "fetched_at"]:
        if c in benchmark_df.columns:
            benchmark_df[c] = pd.to_datetime(benchmark_df[c], errors="coerce")
            if benchmark_df[c].notna().sum() > 0:
                time_col = c
                break

    feature_cols = [
        "implied_probability",
        "model_probability",
        "historical_probability",
        "target_low",
        "target_high",
        "month_num",
        "day_num",
        "hour_24",
    ]

    model_benchmark = benchmark_df.dropna(subset=["target"]).copy()
    model_benchmark = model_benchmark[model_benchmark["model_supported"]].copy()

    for col in feature_cols:
        if col not in model_benchmark.columns:
            model_benchmark[col] = np.nan

    def _group_text(v):
        if pd.isna(v):
            return "NA"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v).strip().lower()

    model_benchmark["benchmark_group"] = model_benchmark.apply(
        lambda row: " | ".join([
            _group_text(row.get("question")),
            _group_text(row.get("resolved_city")),
            _group_text(row.get("market_family")),
            _group_text(row.get("target_type")),
            _group_text(row.get("target_low")),
            _group_text(row.get("target_high")),
        ]),
        axis=1,
    )

    if len(model_benchmark) < 80:
        benchmark_warning = f"Not enough resolved labeled rows for benchmark comparison yet ({len(model_benchmark)} rows)."
    elif model_benchmark["target"].nunique() < 2:
        benchmark_warning = "Benchmark needs both outcome classes in resolved data."
    elif time_col is None:
        benchmark_warning = "Benchmark skipped: no usable timestamp column for grouped time holdout."
    else:
        group_time_df = (
            model_benchmark
            .groupby("benchmark_group", as_index=False)[time_col]
            .min()
            .sort_values(time_col)
            .reset_index(drop=True)
        )

        if len(group_time_df) < 10:
            benchmark_warning = f"Benchmark skipped: not enough distinct resolved groups ({len(group_time_df)})."
        else:
            split_idx = int(len(group_time_df) * 0.80)
            split_idx = max(1, min(split_idx, len(group_time_df) - 1))

            train_groups = set(group_time_df.iloc[:split_idx]["benchmark_group"])
            test_groups = set(group_time_df.iloc[split_idx:]["benchmark_group"])

            train_df = model_benchmark[model_benchmark["benchmark_group"].isin(train_groups)].copy()
            test_df = model_benchmark[model_benchmark["benchmark_group"].isin(test_groups)].copy()

            if len(train_df) < 50 or len(test_df) < 20:
                benchmark_warning = (
                    f"Benchmark skipped: grouped train/test split too small "
                    f"(train={len(train_df)}, test={len(test_df)})."
                )
            elif train_df["target"].nunique() < 2 or test_df["target"].nunique() < 2:
                benchmark_warning = "Benchmark skipped: both grouped train and test need both outcome classes."
            else:
                X_train = train_df[feature_cols].copy()
                y_train = train_df["target"].astype(int).copy()
                X_test = test_df[feature_cols].copy()
                y_test = test_df["target"].astype(int).copy()

                imputer = SimpleImputer(strategy="median")
                X_train_imp = imputer.fit_transform(X_train)
                X_test_imp = imputer.transform(X_test)

                benchmark_rows = []

                models = {
                    "Baseline": DummyClassifier(strategy="prior"),
                    "Random Forest": RandomForestClassifier(
                        n_estimators=200,
                        max_depth=6,
                        min_samples_leaf=10,
                        random_state=42,
                        n_jobs=-1,
                    ),
                }

                if HAS_XGBOOST:
                    models["XGBoost"] = XGBClassifier(
                        n_estimators=200,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        eval_metric="logloss",
                        random_state=42,
                    )

                for model_name, model in models.items():
                    try:
                        model.fit(X_train_imp, y_train)
                        pred_prob = model.predict_proba(X_test_imp)[:, 1]
                        pred_prob = np.clip(pred_prob, 1e-6, 1 - 1e-6)

                        benchmark_rows.append({
                            "model": model_name,
                            "log_loss": float(log_loss(y_test, pred_prob)),
                            "brier_score": float(brier_score_loss(y_test, pred_prob)),
                            "roc_auc": float(roc_auc_score(y_test, pred_prob)),
                            "avg_pred": float(np.mean(pred_prob)),
                            "test_rows": int(len(y_test)),
                            "status": "ok",
                        })
                    except Exception as model_error:
                        benchmark_rows.append({
                            "model": model_name,
                            "log_loss": np.nan,
                            "brier_score": np.nan,
                            "roc_auc": np.nan,
                            "avg_pred": np.nan,
                            "test_rows": int(len(y_test)),
                            "status": f"error: {str(model_error)[:80]}",
                        })

                benchmark_results_df = pd.DataFrame(benchmark_rows)

                if not benchmark_results_df.empty:
                    benchmark_results_df = benchmark_results_df.sort_values(
                        by=["log_loss", "brier_score"],
                        ascending=[True, True],
                        na_position="last"
                    ).reset_index(drop=True)

                benchmark_caption = (
                    f"Resolved grouped holdout: trained on {len(train_groups)} oldest groups / {len(train_df)} rows, "
                    f"tested on {len(test_groups)} newest groups / {len(test_df)} rows using `{time_col}`."
                )

if benchmark_warning:
    st.warning(benchmark_warning)

if benchmark_caption:
    st.caption(benchmark_caption)

if not benchmark_results_df.empty:
    valid_logloss = benchmark_results_df.dropna(subset=["log_loss"])
    valid_brier = benchmark_results_df.dropna(subset=["brier_score"])
    valid_auc = benchmark_results_df.dropna(subset=["roc_auc"])

    b1, b2, b3 = st.columns(3)

    if not valid_logloss.empty:
        best_logloss_row = valid_logloss.sort_values("log_loss").iloc[0]
        b1.metric("Best log loss", f"{best_logloss_row['log_loss']:.4f}")
        b1.caption(f"Winner: {best_logloss_row['model']}")
    else:
        b1.metric("Best log loss", "N/A")
        b1.caption("Winner: none")

    if not valid_brier.empty:
        best_brier_row = valid_brier.sort_values("brier_score").iloc[0]
        b2.metric("Best Brier", f"{best_brier_row['brier_score']:.4f}")
        b2.caption(f"Winner: {best_brier_row['model']}")
    else:
        b2.metric("Best Brier", "N/A")
        b2.caption("Winner: none")

    if not valid_auc.empty:
        best_auc_row = valid_auc.sort_values("roc_auc", ascending=False).iloc[0]
        b3.metric("Best ROC AUC", f"{best_auc_row['roc_auc']:.4f}")
        b3.caption(f"Winner: {best_auc_row['model']}")
    else:
        b3.metric("Best ROC AUC", "N/A")
        b3.caption("Winner: none")

    st.dataframe(
        benchmark_results_df,
        width="stretch",
        hide_index=True,
        column_config={
            "model": st.column_config.TextColumn("Model"),
            "log_loss": st.column_config.NumberColumn("Log loss", format="%.4f"),
            "brier_score": st.column_config.NumberColumn("Brier", format="%.4f"),
            "roc_auc": st.column_config.NumberColumn("ROC AUC", format="%.4f"),
            "avg_pred": st.column_config.NumberColumn("Avg pred", format="%.4f"),
            "test_rows": st.column_config.NumberColumn("Test rows", format="%d"),
            "status": st.column_config.TextColumn("Status", width="medium"),
        }
    )
else:
    st.caption("No trustworthy benchmark results to display yet.")

c1, c2 = st.columns(2)

with c1:
    st.subheader("Model support by family")
    family_counts = display_df.groupby(["market_family", "row_type"]).size().reset_index(name="count")
    if len(family_counts):
        fig_family = px.bar(
            family_counts,
            x="market_family",
            y="count",
            color="row_type",
            barmode="group",
            title="Displayed rows by model family"
        )
        fig_family.update_layout(margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_family, use_container_width=True)
    else:
        st.info("No rows available.")

with c2:
    st.subheader("Model vs market")
    scatter_df = filtered.dropna(subset=["implied_probability", "model_probability_for_bet"])
    if len(scatter_df):
        fig_scatter = px.scatter(
            scatter_df,
            x="implied_probability",
            y="model_probability_for_bet",
            color="row_type",
            symbol="display_side",
            hover_data=["question", "market_family", "row_type", "display_side", "best_ev"],
            title="Market probability vs calibrated model probability"
        )
        fig_scatter.add_shape(
            type="line",
            x0=0, y0=0, x1=1, y1=1,
            line=dict(dash="dash")
        )
        fig_scatter.update_layout(margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("No supported modeled rows available for the scatter plot.")

c3, c4 = st.columns(2)

with c3:
    st.subheader("Probability gap distribution")
    gap_df = filtered.dropna(subset=["prob_gap"])
    if len(gap_df):
        fig_gap = px.histogram(
            gap_df,
            x="prob_gap",
            color="row_type",
            nbins=40,
            title="Calibrated model probability - market probability"
        )
        fig_gap.update_layout(margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gap, use_container_width=True)
    else:
        st.info("No probability gap data available.")

with c4:
    st.subheader("EV by side")
    ev_df = filtered.dropna(subset=["best_ev"])
    if len(ev_df):
        fig_ev = px.box(
            ev_df.assign(best_ev_abs=ev_df["best_ev"].abs()),
            x="display_side",
            y="best_ev_abs",
            color="row_type",
            points="outliers",
            title="Expected value by chosen side"
        )
        fig_ev.update_layout(margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_ev, use_container_width=True)
    else:
        st.info("No EV data available.")

st.subheader("Recommended list")

show_cols = [
    "question",
    "row_type",
    "display_side",
    "market_family",
    "resolved_city",
    "target_type",
    "target_low",
    "target_high",
    "model_supported",
    "implied_probability",
    "enhanced_model_probability",
    "best_ev",
    "bet_quality",
    "confidence",
    "recommended_bet",
    "expected_profit_recommended",
]

st.dataframe(
    display_df[show_cols],
    use_container_width=True,
    hide_index=True,
    column_config={
        "question": st.column_config.TextColumn("Question", width="large"),
        "row_type": st.column_config.TextColumn("Type", width="small"),
        "display_side": st.column_config.TextColumn("Side", width="small"),
        "market_family": st.column_config.TextColumn("Family"),
        "resolved_city": st.column_config.TextColumn("City"),
        "target_type": st.column_config.TextColumn("Type"),
        "target_low": st.column_config.NumberColumn("Low", format="%.1f"),
        "target_high": st.column_config.NumberColumn("High", format="%.1f"),
        "model_supported": st.column_config.CheckboxColumn("✓"),
        "implied_probability": st.column_config.NumberColumn("Market %", format="%.3f"),
        "enhanced_model_probability": st.column_config.NumberColumn("Model %", format="%.3f"),
        "best_ev": st.column_config.NumberColumn("EV", format="%.3f"),
        "bet_quality": st.column_config.TextColumn("Quality"),
        "confidence": st.column_config.NumberColumn("Conf.", format="%.3f"),
        "recommended_bet": st.column_config.NumberColumn("Bet", format="$%.2f"),
        "expected_profit_recommended": st.column_config.NumberColumn("Exp Profit", format="$%.2f"),
    }
)