import os
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

class DataCollector:
    def __init__(self):
        self.api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not self.api_key:
            raise ValueError("Alpha Vantage API key not found in environment variables")
        
        self.ts = TimeSeries(key=self.api_key, output_format='pandas')
        self.symbol = 'BHARTIARTL.BSE'  # Bharti Airtel on BSE
        
    def fetch_daily_data(self):
        """Fetch daily OHLC data"""
        try:
            data, meta_data = self.ts.get_daily(symbol=self.symbol, outputsize='compact')
            # Rename columns to match our expected format
            data.columns = ['open', 'high', 'low', 'close', 'volume']
            # Sort index to ensure chronological order
            data = data.sort_index()
            # Filter data to only include recent data (2024 onwards)
            data = data[data.index >= '2024-01-01']
            return data
        except Exception as e:
            print(f"Error fetching daily data: {e}")
            return None

    def save_data(self, data, filename):
        """Save data to CSV file"""
        if data is not None and not data.empty:
            data.to_csv(f'data/{filename}.csv')
            print(f"Data saved to data/{filename}.csv")

    def collect_all_data(self):
        """Collect and save all required data"""
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Fetch and save daily data
        daily_data = self.fetch_daily_data()
        if daily_data is not None:
            self.save_data(daily_data, 'daily_prices')

if __name__ == "__main__":
    collector = DataCollector()
    collector.collect_all_data() 