import argparse
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from pathlib import Path
import zipfile
import io

def fetch_binance_data(symbol, interval, start_date, end_date, output_dir):
    """
    Fetch historical kline data from Binance Vision (monthly zip files)
    """
    symbol = symbol.upper()
    base_url = "https://data.binance.vision/data/spot/monthly/klines"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_dfs = []
    
    current_date = datetime.strptime(start_date, "%Y-%m")
    end_date_dt = datetime.strptime(end_date, "%Y-%m")
    
    print(f"🚀 Fetching {symbol} {interval} data from {start_date} to {end_date}...")
    
    while current_date <= end_date_dt:
        year = current_date.year
        month = current_date.month
        date_str = f"{year}-{month:02d}"
        
        file_name = f"{symbol}-{interval}-{date_str}.zip"
        url = f"{base_url}/{symbol}/{interval}/{file_name}"
        
        print(f"   ⬇️  Downloading {date_str}...", end="\r")
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    csv_name = z.namelist()[0]
                    with z.open(csv_name) as f:
                        # Binance columns: Open time, Open, High, Low, Close, Volume, Close time, ...
                        df = pd.read_csv(f, header=None)
                        df.columns = [
                            "timestamp", "open", "high", "low", "close", "volume", 
                            "close_time", "quote_asset_volume", "number_of_trades", 
                            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
                        ]
                        
                        # Keep only necessary columns
                        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
                        all_dfs.append(df)
            else:
                print(f"\n   ⚠️  Data not found for {date_str} (Status: {response.status_code})")
        except Exception as e:
            print(f"\n   ❌ Error downloading {date_str}: {e}")
            
        # Move to next month
        if month == 12:
            current_date = datetime(year + 1, 1, 1)
        else:
            current_date = datetime(year, month + 1, 1)
            
        time.sleep(0.5) # Be nice to the API
        
    if not all_dfs:
        print("\n❌ No data fetched.")
        return None
        
    print("\n   🔄 Merging and saving...")
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.sort_values("timestamp", inplace=True)
    
    # Convert timestamp to datetime if needed, but keeping as ms int is fine for now
    # final_df['timestamp'] = pd.to_datetime(final_df['timestamp'], unit='ms')
    
    output_file = output_path / f"{symbol.lower()}_usdt_{interval}_2023.csv"
    final_df.to_csv(output_file, index=False)
    
    print(f"✅ Saved {len(final_df)} rows to {output_file}")
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, required=True, help="Trading pair (e.g., BTCUSDT)")
    parser.add_argument("--interval", type=str, default="1m", help="Time interval (e.g., 1m, 1h)")
    parser.add_argument("--year", type=str, default="2023", help="Year to fetch (e.g., 2023)")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")
    
    args = parser.parse_args()
    
    # Fetch full year
    fetch_binance_data(
        args.symbol, 
        args.interval, 
        f"{args.year}-01", 
        f"{args.year}-12", 
        args.output_dir
    )
