import yfinance as yf
import pandas as pd
import time

# Define date ranges to avoid rate limits
date_ranges = [
    ("1996-01-01", "2005-01-01"),
    ("2005-01-02", "2015-01-01"),
    ("2015-01-02", "2025-02-01")
]

# Initialize an empty DataFrame
vix_data = pd.DataFrame()

for start_date, end_date in date_ranges:
    print(f"Fetching data from {start_date} to {end_date}...")
    
    # Download VIX data in smaller chunks
    vix_chunk = yf.download("^VIX", start=start_date, end=end_date)
    vix_chunk.reset_index(inplace=True)
    vix_chunk.rename(columns={"Date": "date", "Close": "VIX_Close"}, inplace=True)
    
    # Append to main DataFrame
    vix_data = pd.concat([vix_data, vix_chunk])
    
    # Avoid hitting rate limits by pausing between requests
    time.sleep(30)  # Pause for 30 seconds

# Save to CSV
vix_data.to_csv("vix_data.csv", index=False)
print("✅ VIX data saved as vix_data.csv")
