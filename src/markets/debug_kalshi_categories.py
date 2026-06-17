import requests
import pandas as pd

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"

def get_json(url, params=None):
    headers = {"Accept": "application/json", "User-Agent": "weatheredge/1.0"}
    response = requests.get(url, params=params, headers=headers, timeout=60)
    response.raise_for_status()
    return response.json()

def main():
    urls = [
        f"{BASE_URL}/series/categories",
        f"{BASE_URL}/series",
    ]

    for url in urls:
        print(f"\n=== Fetching: {url} ===")
        try:
            payload = get_json(url, params={"limit": 20})
            print("Top-level keys:", list(payload.keys()))

            for k, v in payload.items():
                if isinstance(v, list):
                    print(f"Key '{k}' is a list with {len(v)} items.")
                    if len(v) > 0:
                        df = pd.DataFrame(v[:10])
                        print(df.head(10))
                elif isinstance(v, dict):
                    print(f"Key '{k}' is a dict with keys: {list(v.keys())[:20]}")
                else:
                    print(f"Key '{k}' =", v)
        except Exception as e:
            print("ERROR:", repr(e))

if __name__ == "__main__":
    main()
