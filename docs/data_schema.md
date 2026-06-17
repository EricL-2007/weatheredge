# Data Schema

## weather_data
Raw and recent hourly weather observations.

Columns:
- id
- date
- city
- temperature
- humidity
- pressure
- wind_speed
- precipitation
- cloud_cover

## feature_table
Engineered features derived from weather_data.

Examples:
- month
- week
- quarter
- season
- day_of_week
- temperature_1d
- temperature_3d
- temperature_7d
- precipitation_1d
- precipitation_3d
- precipitation_7d
- rolling averages
- temp_change_1d
- rain_trend_1d
- wind_trend_1d

## market_data
Prediction-market contracts and normalized implied probabilities.

Columns:
- market_id
- question
- implied_probability
- close_date
- source
- status

## ranked_opportunities
Joined model-vs-market opportunities.

Columns:
- market_id
- question
- model_probability
- market_probability
- edge
- confidence_score
- ranking_score
