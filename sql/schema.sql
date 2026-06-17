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

CREATE TABLE IF NOT EXISTS ranked_opportunities (
    id SERIAL PRIMARY KEY,
    market_id VARCHAR(255) UNIQUE,
    question TEXT,
    market_type VARCHAR(50),
    city_name VARCHAR(100),
    model_probability FLOAT,
    market_probability FLOAT,
    edge FLOAT,
    confidence_score FLOAT,
    ranking_score FLOAT,
    close_date TIMESTAMP,
    status VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);
