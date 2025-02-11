import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta
from pathlib import Path

# Polygon.io API Key
API_KEY = "HJqdSyAiFzp4SUZB_6Ug7rODpAXlsXKo"

# Define save path in Documents/AAPL/
SAVE_DIR = Path.home() / "Documents" / "NVDA"
SAVE_DIR.mkdir(parents=True, exist_ok=True)  # Create folder if it doesn’t exist

# Function to fetch intraday data from Polygon.io
def fetch_intraday_data(symbol, interval="5min", total_years=5):
    base_url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/5/minute/"  # 5-minute interval
    all_data = []
    
    end_date = datetime.today()
    start_date = end_date - timedelta(days=total_years * 252)  # Approx. 252 trading days per year
    chunk_size = 90  # Fetch data in 3-month chunks to avoid API limits
    
    while start_date < end_date:
        chunk_end = min(start_date + timedelta(days=chunk_size), end_date)
        url = f"{base_url}{start_date.strftime('%Y-%m-%d')}/{chunk_end.strftime('%Y-%m-%d')}?adjusted=true&sort=asc&limit=50000&apiKey={API_KEY}"
        response = requests.get(url)
        data = response.json()
        
        if "results" not in data:
            print(f"Error fetching data for {symbol}: {data}")
            break
        
        # Convert response data into a DataFrame
        df = pd.DataFrame(data["results"])
        df["t"] = pd.to_datetime(df["t"], unit='ms')  # Convert timestamp
        df.set_index("t", inplace=True)
        all_data.append(df)
        
        # Move forward in time for the next request
        start_date = chunk_end
        
        # Respect API rate limits
        time.sleep(1)
    
    # Concatenate all collected data
    final_df = pd.concat(all_data)
    
    # Save the data to the AAPL folder in Documents
    file_path = SAVE_DIR / f"{symbol}_5min_data.csv"
    final_df.to_csv(file_path)
    
    print(f"Downloaded {len(final_df)} rows of {symbol} 5-minute data.")
    print(f"Saved to: {file_path}")

# Fetch AAPL 5-minute data for 5 years
fetch_intraday_data("NVDA")
