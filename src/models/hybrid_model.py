import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

class HybridModel:
    def __init__(self, arima_order=(5,1,0), lstm_units=50, dropout_rate=0.2):
        self.arima_order = arima_order
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.arima_model = None
        self.lstm_model = None
        self.scaler = None
        
    def build_lstm_model(self, input_shape):
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(units=self.lstm_units, return_sequences=True, input_shape=input_shape),
            Dropout(self.dropout_rate),
            LSTM(units=self.lstm_units),
            Dropout(self.dropout_rate),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss=MeanSquaredError())
        return model
    
    def fit_arima(self, y):
        """Fit ARIMA model to the time series"""
        self.arima_model = ARIMA(y, order=self.arima_order)
        self.arima_model = self.arima_model.fit()
        return self.arima_model
    
    def get_arima_residuals(self, y):
        """Get residuals from ARIMA model"""
        arima_pred = self.arima_model.predict(start=0, end=len(y)-1)
        residuals = y - arima_pred
        return residuals
    
    def train(self, X, y, validation_split=0.2):
        """Train the hybrid model"""
        # Fit ARIMA model
        self.fit_arima(y)
        
        # Get ARIMA residuals
        residuals = self.get_arima_residuals(y)
        
        # Build and train LSTM model on residuals
        self.lstm_model = self.build_lstm_model(input_shape=(X.shape[1], X.shape[2]))
        
        # Train LSTM on residuals
        history = self.lstm_model.fit(
            X, residuals,
            epochs=50,
            batch_size=32,
            validation_split=validation_split,
            verbose=1
        )
        
        return history
    
    def predict(self, X, steps=30):
        """Make predictions using the hybrid model"""
        # Get ARIMA predictions
        arima_pred = self.arima_model.forecast(steps=steps)
        
        # Get LSTM predictions for residuals
        lstm_pred = self.lstm_model.predict(X)
        
        # Ensure predictions are the same length
        min_length = min(len(arima_pred), len(lstm_pred.flatten()))
        arima_pred = arima_pred[:min_length]
        lstm_pred = lstm_pred.flatten()[:min_length]
        
        # Combine predictions
        hybrid_pred = arima_pred + lstm_pred
        
        return hybrid_pred
    
    def evaluate(self, X, y_true):
        """Evaluate model performance"""
        y_pred = self.predict(X, steps=len(y_true))
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Calculate directional accuracy
        direction_true = np.diff(y_true) > 0
        direction_pred = np.diff(y_pred) > 0
        directional_accuracy = np.mean(direction_true == direction_pred)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'directional_accuracy': directional_accuracy
        }
    
    def save_models(self, path='models'):
        """Save both ARIMA and LSTM models"""
        os.makedirs(path, exist_ok=True)
        
        # Save ARIMA model
        joblib.dump(self.arima_model, f'{path}/arima_model.joblib')
        
        # Save LSTM model
        self.lstm_model.save(f'{path}/lstm_model.h5', save_format='h5')
    
    def load_models(self, path='models'):
        """Load both ARIMA and LSTM models"""
        # Load ARIMA model
        self.arima_model = joblib.load(f'{path}/arima_model.joblib')
        
        # Load LSTM model
        self.lstm_model = load_model(f'{path}/lstm_model.h5', 
                                   custom_objects={'MeanSquaredError': MeanSquaredError})

def train_and_evaluate(X, y, test_size=0.2):
    """Train and evaluate the hybrid model"""
    # Split data
    train_size = int(len(X) * (1 - test_size))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Initialize and train model
    model = HybridModel()
    history = model.train(X_train, y_train)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    
    # Save models
    model.save_models()
    
    return model, metrics, history

if __name__ == "__main__":
    from src.features.feature_engineering import process_data
    
    # Process data
    X, y, df = process_data()
    
    # Train and evaluate model
    model, metrics, history = train_and_evaluate(X, y)
    
    print("Model Evaluation Metrics:")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"Directional Accuracy: {metrics['directional_accuracy']:.2%}") 