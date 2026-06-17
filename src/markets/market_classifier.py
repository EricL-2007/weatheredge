import re

CITY_ALIASES = {
    "new york": "New York",
    "nyc": "New York",
    "los angeles": "Los Angeles",
    "la ": "Los Angeles",
    "chicago": "Chicago",
    "houston": "Houston",
    "austin": "Austin",
    "dallas": "Dallas",
    "denver": "Denver",
    "seattle": "Seattle",
}

def extract_city(text: str):
    if not text:
        return None
    t = text.lower()
    for alias, city in CITY_ALIASES.items():
        if alias in t:
            return city
    return None

def classify_market(question: str, subtitle: str = ""):
    text = f"{question or ''} {subtitle or ''}".lower()

    city_name = extract_city(text)

    if "hourly temperature" in text or ("temperature" in text and "hour" in text):
        return {"market_type": "hourly_temperature", "city_name": city_name}

    if "daily temperature" in text or "high temperature" in text or "daily high" in text:
        return {"market_type": "daily_temperature", "city_name": city_name}

    if "snow" in text or "rain" in text or "precipitation" in text:
        return {"market_type": "precipitation", "city_name": city_name}

    if "hurricane" in text or "tropical storm" in text:
        return {"market_type": "hurricane", "city_name": city_name}

    if "wildfire" in text or "earthquake" in text or "natural disaster" in text:
        return {"market_type": "natural_disaster", "city_name": city_name}

    if "climate change" in text or "global temperature" in text:
        return {"market_type": "climate_change", "city_name": city_name}

    return {"market_type": "unknown", "city_name": city_name}