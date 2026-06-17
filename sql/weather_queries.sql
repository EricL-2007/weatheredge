-- 1. Hottest average temperatures by city and month
SELECT
    city,
    EXTRACT(MONTH FROM date) AS month,
    ROUND(AVG(temperature)::numeric, 2) AS avg_temp
FROM weather_data
GROUP BY city, EXTRACT(MONTH FROM date)
ORDER BY avg_temp DESC;

-- 2. Rainiest cities by total precipitation
SELECT
    city,
    ROUND(SUM(precipitation)::numeric, 2) AS total_precipitation
FROM weather_data
GROUP BY city
ORDER BY total_precipitation DESC;

-- 3. 24-hour rolling average temperature by city
SELECT
    city,
    date,
    temperature,
    ROUND(
        AVG(temperature) OVER (
            PARTITION BY city
            ORDER BY date
            ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
        )::numeric,
        2
    ) AS temp_24hr_rolling_avg
FROM weather_data
ORDER BY city, date;

-- 4. Temperature anomaly versus each city's average temperature
SELECT
    city,
    date,
    temperature,
    ROUND(
        (temperature - AVG(temperature) OVER (PARTITION BY city))::numeric,
        2
    ) AS temp_anomaly
FROM weather_data
ORDER BY city, date;

-- 5. Join weather with metadata
SELECT
    w.city,
    c.state,
    c.elevation,
    ROUND(AVG(w.temperature)::numeric, 2) AS avg_temp
FROM weather_data w
JOIN city_metadata c
    ON w.city = c.city
GROUP BY w.city, c.state, c.elevation
ORDER BY avg_temp DESC;

-- 6. Cities with meaningful precipitation
SELECT
    city,
    ROUND(AVG(precipitation)::numeric, 3) AS avg_precip
FROM weather_data
GROUP BY city
HAVING AVG(precipitation) > 0.05
ORDER BY avg_precip DESC;