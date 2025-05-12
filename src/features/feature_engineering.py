import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from sklearn.preprocessing import MinMaxScaler

class FeatureEngineer:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def add_technical_indicators(self, df):
        """Add technical indicators to the price data"""
        # Moving Averages
        df['sma_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
        df['sma_50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
        df['ema_20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
        
        # RSI
        df['rsi'] = RSIIndicator(close=df['close']).rsi()
        
        # Bollinger Bands
        bollinger = BollingerBands(close=df['close'])
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_low'] = bollinger.bollinger_lband()
        df['bb_mid'] = bollinger.bollinger_mavg()
        
        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['price_change_5d'] = df['close'].pct_change(periods=5)
        
        # Volume features
        df['volume_change'] = df['volume'].pct_change()
        df['volume_sma_20'] = SMAIndicator(close=df['volume'], window=20).sma_indicator()
        
        # Additional technical indicators
        df['volatility'] = df['close'].rolling(window=20).std()
        df['momentum'] = df['close'] - df['close'].shift(10)
        
        return df
    
    def prepare_data(self, df, sequence_length=30):
        """Prepare data for LSTM model"""
        # Drop NaN values
        df = df.dropna()
        
        # Normalize features
        feature_columns = [col for col in df.columns if col not in ['close', 'volume']]
        df[feature_columns] = self.scaler.fit_transform(df[feature_columns])
        
        # Create sequences for LSTM
        X, y = [], []
        for i in range(len(df) - sequence_length):
            X.append(df[feature_columns].iloc[i:(i + sequence_length)].values)
            y.append(df['close'].iloc[i + sequence_length])
            
        return np.array(X), np.array(y)
    
    def inverse_transform(self, data):
        """Inverse transform the scaled data"""
        return self.scaler.inverse_transform(data)

def process_data(price_file='data/daily_prices.csv'):
    """Main function to process all data"""
    # Read data
    price_df = pd.read_csv(price_file, index_col=0, parse_dates=True)
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Add technical indicators
    price_df = engineer.add_technical_indicators(price_df)
    
    # Prepare data for model
    X, y = engineer.prepare_data(price_df)
    
    return X, y, price_df

if __name__ == "__main__":
    X, y, df = process_data()
    print(f"Processed data shape: X: {X.shape}, y: {y.shape}") 