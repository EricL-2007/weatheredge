from src.markets.daily_market_sync import main as sync_markets
from src.markets.ev import main as update_ev

def main():
    print("Running market sync...")
    sync_markets()
    print("Running EV update...")
    update_ev()
    print("Done.")

if __name__ == "__main__":
    main()
