import os
from src.features.feature_engineering import process_data
from src.models.hybrid_model import train_and_evaluate

def main():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    print("Processing data...")
    # Process data
    X, y, df = process_data()
    
    print("\nTraining model...")
    # Train and evaluate model
    model, metrics, history = train_and_evaluate(X, y)
    
    print("\nModel Evaluation Metrics:")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"Directional Accuracy: {metrics['directional_accuracy']:.2%}")
    
    print("\nModel saved successfully in the 'models' directory!")

if __name__ == "__main__":
    main() 