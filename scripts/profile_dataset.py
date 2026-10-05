import os
import pandas as pd
import json
from collections import defaultdict
import datetime

def profile_dataset(base_dir):
    print("Starting dataset profiling...")
    cities = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    profiling = {
        "summary": {
            "num_files": 0,
            "num_cities": len(cities),
            "num_stations": 0,
            "num_years": set(),
            "date_range": {"min": None, "max": None},
            "available_parameters": set(),
            "total_rows": 0,
            "total_missing_values": 0,
            "total_duplicate_records": 0
        },
        "rows_per_file": {},
        "rows_per_city": defaultdict(int),
        "rows_per_station": defaultdict(int),
        "rows_per_year": defaultdict(int),
        "sampling_frequencies": set(),
        "data_types": {}
    }
    
    stations_set = set()
    
    for city in cities:
        city_path = os.path.join(base_dir, city)
        stations = [d for d in os.listdir(city_path) if os.path.isdir(os.path.join(city_path, d))]
        for station in stations:
            stations_set.add(station)
            station_path = os.path.join(city_path, station)
            files = [f for f in os.listdir(station_path) if f.endswith(('.xlsx', '.csv'))]
            
            for file in files:
                file_path = os.path.join(station_path, file)
                profiling["summary"]["num_files"] += 1
                
                try:
                    if file.endswith('.xlsx'):
                        # Skip header rows if necessary, some datasets have extra metadata rows
                        # We will read 10 rows first to find the actual header
                        preview = pd.read_excel(file_path, nrows=15)
                        header_idx = 0
                        # Usually the header contains Date or Date/Time
                        for idx, row in preview.iterrows():
                            row_vals = [str(val).lower() for val in row.values]
                            if any(x in y for x in ['date', 'time'] for y in row_vals if pd.notna(y)):
                                header_idx = idx
                                break
                        
                        df = pd.read_excel(file_path, header=header_idx)
                    else:
                        df = pd.read_csv(file_path)
                    
                    rows_in_file = len(df)
                    profiling["summary"]["total_rows"] += rows_in_file
                    profiling["rows_per_file"][file_path] = rows_in_file
                    profiling["rows_per_city"][city] += rows_in_file
                    profiling["rows_per_station"][station] += rows_in_file
                    
                    # Extract year from filename if possible
                    year_str = file.replace('.xlsx', '').replace('.csv', '')
                    if year_str.isdigit():
                        year = int(year_str)
                        profiling["summary"]["num_years"].add(year)
                        profiling["rows_per_year"][year] += rows_in_file
                    
                    duplicates = df.duplicated().sum()
                    profiling["summary"]["total_duplicate_records"] += int(duplicates)
                    
                    missing = df.isna().sum().sum()
                    profiling["summary"]["total_missing_values"] += int(missing)
                    
                    # Parameters and types
                    for col in df.columns:
                        profiling["summary"]["available_parameters"].add(str(col))
                        if str(col) not in profiling["data_types"]:
                            profiling["data_types"][str(col)] = str(df[col].dtype)
                            
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    
    profiling["summary"]["num_stations"] = len(stations_set)
    profiling["summary"]["num_years"] = list(profiling["summary"]["num_years"])
    profiling["summary"]["available_parameters"] = list(profiling["summary"]["available_parameters"])
    profiling["sampling_frequencies"] = list(profiling["sampling_frequencies"])
    
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "profiling_report.json")
    
    with open(report_path, 'w') as f:
        json.dump(profiling, f, indent=4)
        
    print(f"Profiling complete. Report saved to {report_path}")

if __name__ == '__main__':
    base_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "AIR QUALITY")
    profile_dataset(base_directory)
