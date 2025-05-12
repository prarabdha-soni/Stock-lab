import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.models.hybrid_model import HybridModel
from src.features.feature_engineering import FeatureEngineer

def load_latest_data():
    """Load the most recent data for prediction"""
    price_df = pd.read_csv('data/daily_prices.csv', index_col=0, parse_dates=True)
    return price_df

def prepare_prediction_data(price_df, sequence_length=30):
    """Prepare data for making predictions"""
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Add technical indicators
    price_df = engineer.add_technical_indicators(price_df)
    
    # Get the last sequence_length days of data
    last_sequence = price_df.iloc[-sequence_length:]
    
    # Prepare features
    feature_columns = [col for col in last_sequence.columns if col not in ['close', 'volume']]
    X = last_sequence[feature_columns].values.reshape(1, sequence_length, len(feature_columns))
    
    return X, price_df

def make_predictions(days_ahead=30):
    """Make predictions for the next n days"""
    try:
        # Load latest data
        price_df = load_latest_data()
        
        # Prepare prediction data
        X, price_df = prepare_prediction_data(price_df)
        
        # Load trained model
        model = HybridModel()
        model.load_models()
        
        # Make predictions
        predictions = model.predict(X, steps=days_ahead)
        
        # Create prediction dates
        last_date = price_df.index[-1]
        prediction_dates = [last_date + timedelta(days=i+1) for i in range(len(predictions))]
        
        # Create prediction DataFrame
        predictions_df = pd.DataFrame({
            'date': prediction_dates,
            'predicted_price': predictions
        })
        predictions_df.set_index('date', inplace=True)
        
        return predictions_df
    
    except Exception as e:
        print(f"Error making predictions: {e}")
        raise

def save_predictions(predictions_df, filename='predictions.csv'):
    """Save predictions to CSV file"""
    predictions_df.to_csv(f'data/{filename}')
    print(f"Predictions saved to data/{filename}")

if __name__ == "__main__":
    # Make predictions for next 30 days
    predictions_df = make_predictions(days_ahead=30)
    
    # Save predictions
    save_predictions(predictions_df)
    
    # Print predictions
    print("\nPredicted Prices for Next 30 Days:")
    print(predictions_df) 