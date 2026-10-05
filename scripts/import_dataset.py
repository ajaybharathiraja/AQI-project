import os
import sys
import glob
import pandas as pd
import numpy as np
from datetime import datetime

# Add the parent directory to the path so we can import the app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.extensions import db
from app.models.station import Station
from app.models.air_quality import AirQualityRecord
import traceback

def clean_column_names(df):
    """Normalize column names."""
    df.columns = df.columns.str.strip().str.lower()
    
    # Map common variations to standard names
    col_mapping = {
        'pm2.5': 'pm25', 'pm 2.5': 'pm25', 'pm25': 'pm25',
        'pm10': 'pm10', 'pm 10': 'pm10',
        'no': 'no', 'no2': 'no2', 'nox': 'nox',
        'nh3': 'nh3', 'so2': 'so2', 'co': 'co',
        'ozone': 'ozone', 'o3': 'ozone',
        'benzene': 'benzene', 'toluene': 'toluene',
        'xylene': 'xylene', 'mp-xylene': 'xylene',
        'rh': 'rh', 'wd': 'wd', 'sr': 'sr', 'bp': 'bp', 'at': 'at', 'rf': 'rf',
        'date': 'timestamp', 'time': 'timestamp', 'date/time': 'timestamp',
        'from date': 'timestamp', 'date time': 'timestamp'
    }
    
    new_cols = []
    for col in df.columns:
        if col in col_mapping:
            new_cols.append(col_mapping[col])
        else:
            # Fallback for exact match of "from date" in case of trailing spaces etc
            if 'from date' in col:
                new_cols.append('timestamp')
            elif 'date' in col and 'to date' not in col:
                new_cols.append('timestamp')
            else:
                new_cols.append(col)
                
    # Handle duplicates by only keeping the first occurrence (except for timestamp logic below)
    seen = set()
    deduped_cols = []
    for c in new_cols:
        if c in seen and c != 'timestamp':
            deduped_cols.append(f"{c}_duplicate")
        else:
            deduped_cols.append(c)
            seen.add(c)
    
    new_cols = deduped_cols
    
    # Handle duplicate timestamp columns if they exist (e.g. from date and to date both becoming timestamp)
    # We only want the first one to be timestamp, or we rename 'to date' to something else
    final_cols = []
    found_timestamp = False
    for col in new_cols:
        if col == 'timestamp':
            if found_timestamp:
                final_cols.append('end_timestamp')
            else:
                final_cols.append('timestamp')
                found_timestamp = True
        else:
            final_cols.append(col)
            
    df.columns = final_cols
    return df

def clean_and_ingest_data(base_dir):
    print("Starting data ingestion into MySQL...")
    app = create_app()
    
    with app.app_context():
        db.create_all()
        
        cities = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        
        total_inserted = 0
        
        for city in cities:
            city_path = os.path.join(base_dir, city)
            stations = [d for d in os.listdir(city_path) if os.path.isdir(os.path.join(city_path, d))]
            
            for station_name in stations:
                station_path = os.path.join(city_path, station_name)
                files = [f for f in os.listdir(station_path) if f.endswith(('.xlsx', '.csv'))]
                
                # Check if station exists, if not create
                station = Station.query.filter_by(city=city, station_name=station_name).first()
                if not station:
                    station = Station(city=city, station_name=station_name)
                    db.session.add(station)
                    db.session.commit()
                    print(f"Added station: {station_name} in {city}")
                
                for file in files:
                    file_path = os.path.join(station_path, file)
                    print(f"Processing {file_path}...")
                    try:
                        if file.endswith('.xlsx'):
                            preview = pd.read_excel(file_path, nrows=30, header=None)
                            header_idx = 0
                            for idx, row in preview.iterrows():
                                row_vals = [str(val).lower() for val in row.values if pd.notna(val)]
                                if len(row_vals) > 5 and any(x in y for x in ['date', 'time', 'from'] for y in row_vals):
                                    header_idx = idx
                                    break
                            df = pd.read_excel(file_path, header=header_idx)
                        else:
                            df = pd.read_csv(file_path)
                            
                        # Clean columns
                        df = clean_column_names(df)
                        
                        if 'timestamp' not in df.columns:
                            print(f"Skipping {file_path}, no timestamp column found after cleaning.")
                            continue
                            
                        # Normalize timestamp
                        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                        
                        # Drop rows where timestamp is NaT
                        df = df.dropna(subset=['timestamp'])
                        
                        # Remove duplicates based on timestamp
                        df = df.drop_duplicates(subset=['timestamp'])
                        
                        # Convert missing values to None for SQL insertion
                        df = df.replace({np.nan: None})
                        
                        # Prepare records
                        records = []
                        valid_columns = ['pm25', 'pm10', 'no', 'no2', 'nox', 'nh3', 'so2', 'co', 'ozone', 'benzene', 'toluene', 'xylene', 'rh', 'wd', 'sr', 'bp', 'at', 'rf']
                        
                        for _, row in df.iterrows():
                            # Check if record already exists
                            ts = row['timestamp']
                            
                            kwargs = {
                                'station_id': station.id,
                                'timestamp': ts
                            }
                            
                            for col in valid_columns:
                                if col in df.columns:
                                    val = row[col]
                                    if pd.notna(val) and val is not None and val != 'None' and val != '':
                                        try:
                                            kwargs[col] = float(val)
                                        except ValueError:
                                            pass
                                            
                            record = AirQualityRecord(**kwargs)
                            records.append(record)
                            
                        # Bulk insert using SQLAlchemy Core for speed
                        if records:
                            db.session.bulk_save_objects(records)
                            db.session.commit()
                            total_inserted += len(records)
                            print(f"Inserted {len(records)} records from {file}")
                            
                    except Exception as e:
                        db.session.rollback()
                        print(f"Error processing {file_path}: {e}")
                        traceback.print_exc()
                        
        print(f"Ingestion complete. Total records inserted: {total_inserted}")

if __name__ == '__main__':
    base_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "AIR QUALITY")
    clean_and_ingest_data(base_directory)
