# WeatherEdge

WeatherEdge is a Python, PostgreSQL, and Streamlit dashboard for identifying positive expected-value opportunities in weather prediction markets. It loads market data from a local database, applies model-driven probability estimates, calibrates those probabilities conservatively, and ranks opportunities with bankroll-aware bet sizing.

## Features

- Interactive dashboard built with Streamlit
- PostgreSQL-backed market storage
- Market filtering by city, market type, model family, and bet rules
- Conservative probability calibration
- Expected value and bankroll-based bet sizing
- KPI summary cards for money in, expected profit, and ROI
- “Real bet” vs watchlist workflow

## Stack

- Python
- PostgreSQL
- SQLAlchemy
- pandas
- numpy
- scikit-learn
- Plotly
- Streamlit
- psycopg2
- python-dotenv

## Project structure

```text
weatheredge/
├── src/
│   ├── __init__.py
│   ├── dashboard/
│   │   ├── __init__.py
│   │   └── app.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── weather_calibration.py
│   └── jobs/
│       └── daily_sync.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## What it does

The dashboard reads weather market rows from PostgreSQL, filters the market universe, and computes calibrated opportunities for decision support. The current version uses conservative shrinkage calibration rather than expensive live historical API calibration so the app stays fast and stable during reruns.

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/weatheredge.git
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

If `requirements.txt` does not exist yet, use:

```bash
pip install streamlit pandas numpy sqlalchemy psycopg2-binary python-dotenv scikit-learn plotly
```

Then save them:

```bash
pip freeze > requirements.txt
```

### 4. Configure environment variables

Copy the example file and fill in your local settings:

```bash
cp .env.example .env
```

Set:

```env
DATABASE_URL=postgresql+psycopg2://YOUR_USERNAME@localhost:5432/weatheredge
```

### 5. Make sure PostgreSQL is available

Test the database connection:

```bash
psql -h localhost -p 5432 -U YOUR_USERNAME -d weatheredge -c "SELECT COUNT(*) FROM market_data;"
```

## Run the dashboard

From the repo root:

```bash
source .venv/bin/activate
streamlit run src/dashboard/app.py
```

If you hit package import issues, this launch form is the safest:

```bash
PYTHONPATH=. streamlit run src/dashboard/app.py
```

## Daily workflow

This is the recommended day-to-day routine.

### Morning

```bash
cd ~/weatheredge
source .venv/bin/activate
git pull origin main
python -m src.jobs.daily_sync
streamlit run src/dashboard/app.py
```

### In the dashboard

Review these every day:

- Market count
- Supported market count
- Real bets vs watchlist
- Average real EV
- Money in
- Expected profit
- Expected ROI
- Top ranked opportunities

### Midday refresh

```bash
cd ~/weatheredge
source .venv/bin/activate
python -m src.jobs.daily_sync
streamlit cache clear
streamlit run src/dashboard/app.py
```

### End of day

```bash
cd ~/weatheredge
git status
git add .
git commit -m "Daily WeatherEdge update"
git push origin main
```

## Current calibration approach

The current version uses conservative shrinkage toward market probability and coin-flip probability instead of making expensive row-by-row historical weather API calls in the dashboard. This keeps runtime fast and avoids unstable rerun behavior while still reducing overconfidence in raw model outputs.

## Current limitations

- Calibration is conservative, not a full historical backtest
- Bet sizing is intentionally capped and simplified
- The dashboard is best used as a screening and prioritization tool
- Some payout assumptions may still need refinement

## Recommended next improvements

- CSV export of filtered bets
- Better default visible columns
- Top 10 / 25 / 50 row presets
- Daily logging of reviewed opportunities
- More robust payout-aware Kelly sizing
- Backtested calibration outside the live dashboard

## Security notes

Do not commit:
- `.env`
- `.venv/`
- database credentials
- local logs
- local output files

Use `.env.example` for safe public configuration guidance.

## Resume framing

This project is a good portfolio piece for:
- data engineering
- analytics engineering
- quant-style modeling
- dashboard development
- decision-support tooling

Suggested framing:

> Built a Python + PostgreSQL + Streamlit dashboard for weather prediction markets that computed model-vs-market edge, applied conservative probability calibration, and generated bankroll-aware bet recommendations with KPI reporting.

## License

Add a license before sharing publicly. MIT is a common simple choice for portfolio repositories.