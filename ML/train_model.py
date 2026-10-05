import os
import sys
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Add the parent directory to the path so we can import the app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.extensions import db
from app.models.air_quality import AirQualityRecord

def train_aqi_model():
    print("Starting ML model training process...")
    app = create_app()
    with app.app_context():
        # Fetch data
        print("Querying database for records...")
        records = db.session.query(
            AirQualityRecord.station_id,
            AirQualityRecord.timestamp,
            AirQualityRecord.pm25
        ).filter(AirQualityRecord.pm25.isnot(None)).all()
        
        if not records:
            print("No records found with valid PM2.5 data. Run the data ingestion script first.")
            return
            
        print(f"Loaded {len(records)} records. Preparing features...")
        
        # Convert to DataFrame
        df = pd.DataFrame(records, columns=['station_id', 'timestamp', 'pm25'])
        
        # Feature Engineering
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        
        # Drop NaN values just in case
        df = df.dropna()
        
        # Define Features (X) and Target (y)
        features = ['month', 'day', 'dayofweek', 'station_id']
        X = df[features]
        y = df['pm25']
        
        print("Splitting dataset...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("Training XGBoost Regressor...")
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        print("Evaluating model...")
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        
        print(f"Model Evaluation Results:")
        print(f" - RMSE: {rmse:.2f}")
        print(f" - R2 Score: {r2:.2f}")
        
        # Save model
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, 'aqi_model.pkl')
        
        joblib.dump(model, model_path)
        print(f"Model successfully saved to {model_path}")

if __name__ == '__main__':
    train_aqi_model()
