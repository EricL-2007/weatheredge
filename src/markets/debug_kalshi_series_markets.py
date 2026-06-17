import requests

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
SERIES = ["KXDVHIGH", "KXHIGHHOU", "KXRAINMIAM", "KXHIGHMIA"]

def try_request(path, params):
    headers = {"Accept": "application/json", "User-Agent": "weatheredge/1.0"}
    url = f"{BASE_URL}{path}"
    print(f"\n=== GET {url} params={params} ===")
    r = requests.get(url, params=params, headers=headers, timeout=60)
    print("status:", r.status_code)
    try:
        data = r.json()
        print("keys:", list(data.keys()) if isinstance(data, dict) else type(data))
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, list):
                    print(f"{k}: list[{len(v)}]")
                    if len(v) > 0:
                        print("first item keys:", list(v[0].keys())[:20])
                        print("first item:", {kk: v[0].get(kk) for kk in list(v[0].keys())[:8]})
                        break
                else:
                    print(f"{k}: {type(v).__name__}")
    except Exception:
        print(r.text[:1000])

def main():
    for s in SERIES:
        try_request("/events", {"series_ticker": s, "limit": 5})
        try_request("/events", {"series": s, "limit": 5})
        try_request("/markets", {"series_ticker": s, "limit": 5})
        try_request("/markets", {"series": s, "limit": 5})

if __name__ == "__main__":
    main()
