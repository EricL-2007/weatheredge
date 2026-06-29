# WeatherEdge

WeatherEdge is a probabilistic weather market decision intelligence platform built with Python, Supabase (PostgreSQL), and Streamlit. It scrapes live climate prediction markets from Kalshi, stores them in a Supabase-hosted PostgreSQL database via daily sync, applies a multi-signal probability model with conservative calibration, and surfaces bankroll-aware bet recommendations through an interactive dashboard deployed on Streamlit Cloud.

## Features

- Daily Kalshi API scrape across 322 climate-related series with batch upsert into Supabase
- Streamlit dashboard deployed on Streamlit Cloud with live Supabase connection
- Market filtering by city, market type, model family, and configurable bet rules
- Multi-signal model combining historical temperature baselines, market-implied probabilities, and existing model estimates
- Conservative shrinkage calibration with log loss and Brier score diagnostics
- Kelly-fractional bet sizing with real-bet vs. watchlist workflow
- Plotly charts: model vs. market scatter, probability gap distribution, EV by side, model family breakdown
- ML benchmark suite (Random Forest, XGBoost, Baseline) on resolved outcomes when available

## Stack

- Python
- Supabase (PostgreSQL)
- SQLAlchemy + psycopg2
- Streamlit (deployed on Streamlit Cloud)
- scikit-learn
- XGBoost
- Plotly
- pandas / numpy
- Kalshi Trade API v2
- python-dotenv

## How it works

### Daily sync (`src/jobs/daily_sync.py`)

1. Fetches all Kalshi series (~11,000+) from the Trade API v2
2. Filters to 322 climate-related series using keyword matching on title, subtitle, ticker, and category
3. Fetches all markets for each matched series (up to 500 per series)
4. Normalizes each market (extracts question, prices, implied probability, city, market type)
5. Upserts all rows into `public.market_data` in Supabase in batches of 500

A typical run inserts/updates ~18,000 market rows.

### Dashboard (`app.py`)

The dashboard loads climate markets from Supabase, then for each market:

- **Parses** the question text for thresholds, date/time, and direction (greater than / less than / range)
- **Infers city** from question text or the `city_name` column
- **Estimates hourly temperature** using city-specific seasonal baselines and a diurnal curve
- **Computes historical probability** via normal CDF over the estimated temperature distribution
- **Blends** historical probability (65%), market probability (10%), and stored model probability (25%) into an enhanced model estimate
- **Calibrates** the enhanced estimate using shrinkage toward market and coin-flip
- **Computes edge** for YES and NO sides; assigns bet direction based on configurable thresholds
- **Sizes bets** using fractional Kelly with floor and cap constraints
- **Classifies rows** as REAL BET or WATCHLIST based on edge and win probability filters

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/EricL-2007/weatheredge.git
cd weatheredge
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Set your Supabase connection string:

```
DATABASE_URL=postgresql+psycopg2://postgres:YOUR_ENCODED_PASSWORD@aws-1-us-west-2.pooler.supabase.com:6543/postgres?sslmode=require
```

For Streamlit Cloud deployment, add `DATABASE_URL` to your app's Streamlit secrets instead.

### 5. Run the daily sync

```bash
PYTHONPATH=. python -m src.jobs.daily_sync
```

### 6. Run the dashboard

```bash
PYTHONPATH=. streamlit run app.py
```

## Live app

[https://el2007weatheredge.streamlit.app](https://el2007weatheredge.streamlit.app)

## Daily workflow

### Morning

```bash
cd ~/weatheredge
source .venv/bin/activate
git pull origin main
PYTHONPATH=. python -m src.jobs.daily_sync
PYTHONPATH=. streamlit run app.py
```

### End of day

```bash
git add .
git commit -m "Daily WeatherEdge update"
git push origin main
```

## Calibration approach

The current calibration uses conservative shrinkage (blending toward market probability and 0.5) rather than row-by-row historical API calls. This keeps the dashboard fast and stable across reruns while reducing overconfidence in raw model outputs. Calibration quality is reported as log loss and Brier score in the sidebar and diagnostics panel.

## ML benchmark

When a resolved outcome column (`resolved_outcome`, `actual_outcome`, etc.) is present in the database, the dashboard runs a grouped time-holdout benchmark comparing:

- **Baseline** (prior class frequency)
- **Random Forest** (200 trees, max depth 6)
- **XGBoost** (200 rounds, learning rate 0.05)

Models are trained on the oldest 80% of resolved market groups and tested on the newest 20%, using features including implied probability, historical probability, threshold values, and date/time components.

## Current limitations

- Calibration is shrinkage-based, not a full historical backtest
- Temperature model covers 10 cities with seasonal baselines; non-temperature market families are unsupported
- Benchmark requires resolved ground-truth outcomes not yet available at scale
- Bet sizing is capped and simplified

## Recommended next improvements

- CSV export of filtered bets
- Expanded city and market family coverage
- Backtested calibration using Open-Meteo historical API
- Daily logging of reviewed opportunities
- Automated outcome resolution pipeline

## Security notes

Do not commit:

- `.env`
- `.venv/`
- Database credentials
- Local logs or output files

Use `.env.example` for safe public configuration guidance.

## License

MIT