import time
import pandas as pd
import random
from datetime import datetime
import sqlite3
import os

basedir = os.path.abspath(os.path.dirname(__file__))
filepath = os.path.join(basedir, 'data', 'live_data.csv')
db_path = os.path.join(basedir, 'air_quality.db')

print('Starting live data generator service for ALL stations...')
while True:
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('SELECT station_name, city FROM stations')
        stations_db = c.fetchall()
        conn.close()
    except Exception as e:
        print('DB Error:', e)
        stations_db = [('Unknown Station', 'Unknown City')]
        
    data = []
    now = datetime.now()
    for s_name, city in stations_db:
        display_name = f"{city} - {s_name}"
        
        pm25 = round(random.uniform(20, 150), 1)
        pm10 = round(random.uniform(40, 250), 1)
        no2 = round(random.uniform(10, 80), 1)
        so2 = round(random.uniform(5, 40), 1)
        co = round(random.uniform(0.1, 2.0), 2)
        ozone = round(random.uniform(10, 100), 1)
        aqi = max(pm25*(100/60), pm10, no2*(100/80), so2*(100/80), co*(100/2), ozone)
        aqi = round(aqi)
        if aqi <= 50: cat = 'Good'
        elif aqi <= 100: cat = 'Satisfactory'
        elif aqi <= 200: cat = 'Moderate'
        elif aqi <= 300: cat = 'Poor'
        else: cat = 'Severe'
        
        data.append({
            'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
            'station': display_name,
            'pm25': pm25,
            'pm10': pm10,
            'no2': no2,
            'so2': so2,
            'co': co,
            'ozone': ozone,
            'aqi': aqi,
            'category': cat
        })
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    time.sleep(3)
