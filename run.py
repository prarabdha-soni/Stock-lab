import os
import warnings
import tensorflow as tf

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging
warnings.filterwarnings('ignore')  # Suppress Python warnings

# Disable oneDNN custom operations warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from app import app

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Run the app
    app.run_server(debug=True, host='0.0.0.0', port=8050) 