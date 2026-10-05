import os
import glob
import json

def inspect_dataset(base_dir):
    print(f"Inspecting base directory: {base_dir}")
    
    cities = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    print(f"Found {len(cities)} cities: {cities}")
    
    tree = {}
    total_files = 0
    for city in cities:
        city_path = os.path.join(base_dir, city)
        stations = [d for d in os.listdir(city_path) if os.path.isdir(os.path.join(city_path, d))]
        tree[city] = {}
        for station in stations:
            station_path = os.path.join(city_path, station)
            files = [f for f in os.listdir(station_path) if f.endswith(('.xlsx', '.csv'))]
            tree[city][station] = files
            total_files += len(files)
            
    print(f"\nTotal Data Files: {total_files}")
    
    report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "inspection_report.json")
    with open(report_path, 'w') as f:
        json.dump(tree, f, indent=4)
        
    print(f"Inspection complete. Tree saved to {report_path}")

if __name__ == '__main__':
    base_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "AIR QUALITY")
    inspect_dataset(base_directory)
