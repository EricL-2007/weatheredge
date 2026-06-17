# Architecture

WeatherEdge pipeline:

1. Data ingestion from Open-Meteo and market APIs
2. Storage in PostgreSQL
3. Feature engineering for temporal, lag, rolling, and trend features
4. Forecasting models for rain classification and temperature regression
5. Probability calibration
6. SHAP explainability
7. Market ingestion and normalization
8. Event ranking engine
9. Streamlit dashboard

## Data Flow

Weather APIs -> PostgreSQL -> Feature Pipeline -> Models -> Calibration -> Rankings -> Dashboard
