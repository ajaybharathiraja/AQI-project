import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Add the parent directory to the path so we can import the app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.extensions import db
from app.models.air_quality import AirQualityRecord

def train_lstm():
    print("Initializing Deep Learning Pipeline (LSTM)...")
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from sklearn.preprocessing import MinMaxScaler
        import joblib
    except ImportError:
        print("TensorFlow is not installed. Please install it using 'pip install tensorflow'.")
        return
        
    app = create_app()
    with app.app_context():
        print("Fetching chronological PM2.5 data from database...")
        # Get data sorted by timestamp for time-series sequencing
        records = db.session.query(
            AirQualityRecord.timestamp,
            AirQualityRecord.pm25
        ).filter(AirQualityRecord.pm25.isnot(None))\
         .order_by(AirQualityRecord.timestamp.asc()).all()
         
        if not records:
            print("No valid PM2.5 records found. Please ingest data first.")
            return
            
        print(f"Loaded {len(records)} records. Preprocessing for Deep Learning...")
        
        df = pd.DataFrame(records, columns=['timestamp', 'pm25'])
        data = df['pm25'].values.reshape(-1, 1)
        
        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        # Create sequences (lookback of 24 hours/steps)
        sequence_length = 24
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, 0])
            y.append(scaled_data[i, 0])
            
        X, y = np.array(X), np.array(y)
        
        # Reshape for LSTM [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        print("Splitting into Train and Test sets...")
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        print("Building LSTM Neural Network Architecture...")
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=25))
        model.add(Dense(units=1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        print("Training Model (Epochs: 5, Batch Size: 32)...")
        # We use a small epoch count here for demonstration/speed
        model.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_test, y_test))
        
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        model_path = os.path.join(models_dir, 'aqi_lstm.h5')
        scaler_path = os.path.join(models_dir, 'lstm_scaler.pkl')
        
        print("Saving Model and Scaler artifacts...")
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        
        print(f"Deep Learning Model successfully saved to: {model_path}")

if __name__ == '__main__':
    train_lstm()
