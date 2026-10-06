from flask import Blueprint, render_template, jsonify, request
from flask_login import login_required, current_user
from app.extensions import db
from app.models.station import Station
from app.models.air_quality import AirQualityRecord
from sqlalchemy import func

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/')
@login_required
def index():
    if current_user.is_admin:
        from flask import redirect, url_for
        return redirect(url_for('admin.dashboard'))
        
    import random
    from sqlalchemy import func
    
    stations = Station.query.all()
    
    city_coords = {
        'Ariyalur': (11.1400, 79.0786),
        'Chengalpattu': (12.6841, 79.9836),
        'Chennai': (13.0827, 80.2707),
        'Coimbatore': (11.0168, 76.9558),
        'Cuddalur': (11.7480, 79.7714),
        'Dindigul': (10.3673, 77.9803),
        'Gummundipundi': (13.4072, 80.1171)
    }
    
    avg_pollutants = db.session.query(
        AirQualityRecord.station_id,
        func.avg(AirQualityRecord.pm25),
        func.avg(AirQualityRecord.pm10),
        func.avg(AirQualityRecord.no2),
        func.avg(AirQualityRecord.so2),
        func.avg(AirQualityRecord.co),
        func.avg(AirQualityRecord.ozone)
    ).group_by(AirQualityRecord.station_id).all()
    
    from app.utils.aqi_calculator import calculate_indian_aqi
    aqi_dict = {}
    for r in avg_pollutants:
        pm25, pm10, no2, so2, co, ozone = r[1], r[2], r[3], r[4], r[5], r[6]
        aqi_val, aqi_cat = calculate_indian_aqi(pm25, pm10, no2, so2, co, ozone)
        aqi_dict[r[0]] = {
            'aqi': round(aqi_val),
            'cat': aqi_cat,
            'pm25': round(pm25, 2) if pm25 else 0,
            'pm10': round(pm10, 2) if pm10 else 0,
            'no2': round(no2, 2) if no2 else 0,
            'so2': round(so2, 2) if so2 else 0,
            'co': round(co, 2) if co else 0,
            'ozone': round(ozone, 2) if ozone else 0
        }
    
    map_data = []
    for s in stations:
        base_lat, base_lon = city_coords.get(s.city, (13.0827, 80.2707))
        lat = s.latitude if s.latitude else base_lat + random.uniform(-0.03, 0.03)
        lon = s.longitude if s.longitude else base_lon + random.uniform(-0.03, 0.03)
        info = aqi_dict.get(s.id, {'aqi': 50, 'cat': 'Good', 'pm25': 0, 'pm10': 0, 'no2': 0, 'so2': 0, 'co': 0, 'ozone': 0})
        map_data.append({
            'name': s.station_name, 'city': s.city, 'lat': lat, 'lon': lon,
            'aqi': info['aqi'], 'cat': info['cat'], 'pm25': info['pm25'],
            'pm10': info['pm10'], 'no2': info['no2'], 'so2': info['so2'],
            'co': info['co'], 'ozone': info['ozone']
        })

    return render_template('dashboard/dashboard.html', map_data=map_data)

@dashboard_bp.route('/live')
@login_required
def live():
    return render_template('dashboard/live.html')

@dashboard_bp.route('/api/live')
@login_required
def api_live():
    import pandas as pd
    import os
    from datetime import datetime
    import random
    
    basedir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    filepath = os.path.join(basedir, 'data', 'live_data.csv')
    
    if os.path.exists(filepath) and not os.environ.get('VERCEL'):
        df = pd.read_csv(filepath)
        return jsonify(df.to_dict(orient='records'))
    else:
        # Mock Live Data generation since background worker doesn't run on Vercel
        try:
            stations = Station.query.limit(5).all()
        except:
            stations = []
        
        if not stations:
            return jsonify([])
            
        data = []
        now = datetime.now()
        for s in stations:
            pm25 = round(random.uniform(20, 150), 1)
            pm10 = round(random.uniform(40, 250), 1)
            no2 = round(random.uniform(10, 80), 1)
            so2 = round(random.uniform(5, 40), 1)
            co = round(random.uniform(0.1, 2.0), 2)
            ozone = round(random.uniform(10, 100), 1)
            
            from app.utils.aqi_calculator import calculate_indian_aqi
            aqi_val, cat = calculate_indian_aqi(pm25, pm10, no2, so2, co, ozone)
            
            data.append({
                'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
                'station': f"{s.city} - {s.station_name}",
                'pm25': pm25,
                'pm10': pm10,
                'no2': no2,
                'so2': so2,
                'co': co,
                'ozone': ozone,
                'aqi': round(aqi_val),
                'category': cat
            })
        return jsonify(data)

@dashboard_bp.route('/eda')
@login_required
def eda():
    from sqlalchemy import extract
    # Fetch cities for dropdown
    cities = db.session.query(Station.city).distinct().all()
    cities = [c[0] for c in cities if c[0]]
    
    stations = Station.query.all()
    
    # Fetch unique years
    years_query = db.session.query(extract('year', AirQualityRecord.timestamp).label('year')).distinct().all()
    years = sorted([int(y.year) for y in years_query if y.year], reverse=True)
    
    # Get available date range
    min_date = db.session.query(func.min(func.date(AirQualityRecord.timestamp))).scalar()
    max_date = db.session.query(func.max(func.date(AirQualityRecord.timestamp))).scalar()
    
    return render_template('dashboard/eda.html', cities=cities, stations=stations, years=years, min_date=min_date, max_date=max_date)

@dashboard_bp.route('/api/eda')
@login_required
def api_eda():
    from sqlalchemy import extract
    from app.utils.aqi_calculator import calculate_indian_aqi
    
    chart_type = request.args.get('chart_type', 'bar')
    pollutant_param = request.args.get('pollutant', 'pm25')
    pollutants = [p.strip() for p in pollutant_param.split(',') if p.strip()]
    city = request.args.get('city', '')
    station_id = request.args.get('station', '')
    year = request.args.get('year', '')
    
    col_map = {
        'pm25': AirQualityRecord.pm25,
        'pm10': AirQualityRecord.pm10,
        'no': AirQualityRecord.no,
        'no2': AirQualityRecord.no2,
        'nox': AirQualityRecord.nox,
        'so2': AirQualityRecord.so2,
        'co': AirQualityRecord.co,
        'ozone': AirQualityRecord.ozone,
        'nh3': AirQualityRecord.nh3,
        'benzene': AirQualityRecord.benzene,
        'toluene': AirQualityRecord.toluene,
        'xylene': AirQualityRecord.xylene,
        'aqi': AirQualityRecord.calculated_aqi
    }
    
    target_cols = []
    is_availability = False
    
    if 'availability' in pollutants:
        is_availability = True
    elif 'all' in pollutants:
        # Remove 'aqi' from 'all' list since it's on a different scale, or include it? Let's include everything except AQI for better chart scaling
        for key, col in col_map.items():
            if key != 'aqi':
                target_cols.append((key, col))
    else:
        for pol in pollutants:
            if pol in col_map:
                target_cols.append((pol, col_map[pol]))
    
    # Determine grouping based on filters
    if station_id:
        grouping_col = extract('month', AirQualityRecord.timestamp)
    elif city:
        grouping_col = Station.station_name
    else:
        grouping_col = Station.city

    # Build query
    select_items = [grouping_col.label('grouping')]
    if is_availability:
        select_items.append(func.count(AirQualityRecord.id).label('record_count'))
    else:
        for key, col in target_cols:
            select_items.append(func.avg(col).label(f'avg_{key}'))
            
    # Always get core pollutants for dynamic AQI calculation
    core_cols = [
        AirQualityRecord.pm25,
        AirQualityRecord.pm10,
        AirQualityRecord.no2,
        AirQualityRecord.so2,
        AirQualityRecord.co,
        AirQualityRecord.ozone
    ]
    for i, col in enumerate(core_cols):
        select_items.append(func.avg(col).label(f'core_{i}'))
        
    query = db.session.query(*select_items).select_from(Station).join(AirQualityRecord, Station.id == AirQualityRecord.station_id)

    if station_id:
        query = query.filter(Station.id == station_id)
    elif city:
        query = query.filter(Station.city == city)

    try:
        selected_date = request.args.get('date', '')
        if selected_date:
            query = query.filter(func.date(AirQualityRecord.timestamp) == selected_date)
        elif year:
            query = query.filter(extract('year', AirQualityRecord.timestamp) == int(year))
            
        results = query.group_by(grouping_col).all()
        
        labels = []
        aqi_info = []
        for r in results:
            if station_id:
                labels.append(f"Month {int(r[0])}" if r[0] else "Unknown")
            else:
                labels.append(r[0] if r[0] else 'Unknown')
                
            if not is_availability:
                # Core pollutants are the last 6 elements in the row
                avg_pm25 = r[-6] or 0
                avg_pm10 = r[-5] or 0
                avg_no2 = r[-4] or 0
                avg_so2 = r[-3] or 0
                avg_co = r[-2] or 0
                avg_ozone = r[-1] or 0
                
                # Use the extracted utility function for clean AQI calculation
                aqi_val, cat = calculate_indian_aqi(
                    pm25=avg_pm25, 
                    pm10=avg_pm10, 
                    no2=avg_no2, 
                    so2=avg_so2, 
                    co=avg_co, 
                    ozone=avg_ozone
                )
                
                aqi_info.append(f"{round(aqi_val)} - {cat}")
            else:
                aqi_info.append("N/A")
                
        if 'all' in pollutants or len(target_cols) > 1:
            datasets = []
            for i, (key, _) in enumerate(target_cols):
                if key == 'aqi':
                    data_values = []
                    for r in results:
                        avg_pm25, avg_pm10, avg_no2, avg_so2, avg_co, avg_ozone = r[-6] or 0, r[-5] or 0, r[-4] or 0, r[-3] or 0, r[-2] or 0, r[-1] or 0
                        aqi_val, _ = calculate_indian_aqi(pm25=avg_pm25, pm10=avg_pm10, no2=avg_no2, so2=avg_so2, co=avg_co, ozone=avg_ozone)
                        data_values.append(round(aqi_val))
                else:
                    # r[i+1] because r[0] is grouping_col
                    data_values = [round(r[i+1], 2) if r[i+1] else 0 for r in results]
                datasets.append({
                    'label': key.upper(),
                    'data': data_values
                })
            data = {
                'labels': labels,
                'is_all': True,
                'datasets': datasets,
                'pollutant_name': 'Multiple Pollutants' if 'all' not in pollutants else 'All Pollutants',
                'aqi_info': aqi_info
            }
        elif is_availability:
            data = {
                'labels': labels,
                'is_all': False,
                'values': [int(r[1]) if r[1] else 0 for r in results],
                'pollutant_name': 'Data Availability (Total Records)',
                'aqi_info': aqi_info
            }
        else:
            pol = pollutants[0] if pollutants else 'pm25'
            if pol == 'aqi':
                values = []
                for r in results:
                    avg_pm25, avg_pm10, avg_no2, avg_so2, avg_co, avg_ozone = r[-6] or 0, r[-5] or 0, r[-4] or 0, r[-3] or 0, r[-2] or 0, r[-1] or 0
                    aqi_val, _ = calculate_indian_aqi(pm25=avg_pm25, pm10=avg_pm10, no2=avg_no2, so2=avg_so2, co=avg_co, ozone=avg_ozone)
                    values.append(round(aqi_val))
            else:
                values = [round(r[1], 2) if r[1] else 0 for r in results]
            
            data = {
                'labels': labels,
                'is_all': False,
                'values': values,
                'pollutant_name': pol.upper(),
                'aqi_info': aqi_info
            }
        return jsonify(data)
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@dashboard_bp.route('/map')
@login_required
def map_view():
    import random
    from sqlalchemy import func
    
    stations = Station.query.all()
    
    # City coordinates mapping
    city_coords = {
        'Ariyalur': (11.1400, 79.0786),
        'Chengalpattu': (12.6841, 79.9836),
        'Chennai': (13.0827, 80.2707),
        'Coimbatore': (11.0168, 76.9558),
        'Cuddalur': (11.7480, 79.7714),
        'Dindigul': (10.3673, 77.9803),
        'Gummundipundi': (13.4072, 80.1171)
    }
    
    # Fetch average core pollutants per station to calculate true AQI
    avg_pollutants = db.session.query(
        AirQualityRecord.station_id,
        func.avg(AirQualityRecord.pm25),
        func.avg(AirQualityRecord.pm10),
        func.avg(AirQualityRecord.no2),
        func.avg(AirQualityRecord.so2),
        func.avg(AirQualityRecord.co),
        func.avg(AirQualityRecord.ozone)
    ).group_by(AirQualityRecord.station_id).all()
    
    from app.utils.aqi_calculator import calculate_indian_aqi
    aqi_dict = {}
    details_dict = {}
    for r in avg_pollutants:
        pm25, pm10, no2, so2, co, ozone = r[1], r[2], r[3], r[4], r[5], r[6]
        aqi_val, aqi_cat = calculate_indian_aqi(pm25, pm10, no2, so2, co, ozone)
        aqi_dict[r[0]] = {
            'aqi': round(aqi_val),
            'cat': aqi_cat,
            'pm25': round(pm25, 2) if pm25 else 0,
            'pm10': round(pm10, 2) if pm10 else 0,
            'no2': round(no2, 2) if no2 else 0,
            'so2': round(so2, 2) if so2 else 0,
            'co': round(co, 2) if co else 0,
            'ozone': round(ozone, 2) if ozone else 0
        }
    
    map_data = []
    for s in stations:
        # Default coords if none set
        base_lat, base_lon = city_coords.get(s.city, (13.0827, 80.2707))
        # Add slight jitter so multiple stations in same city don't perfectly overlap
        lat = s.latitude if s.latitude else base_lat + random.uniform(-0.03, 0.03)
        lon = s.longitude if s.longitude else base_lon + random.uniform(-0.03, 0.03)
        
        info = aqi_dict.get(s.id, {'aqi': 50, 'cat': 'Good', 'pm25': 0, 'pm10': 0, 'no2': 0, 'so2': 0, 'co': 0, 'ozone': 0})
        
        map_data.append({
            'name': s.station_name,
            'city': s.city,
            'lat': lat,
            'lon': lon,
            'aqi': info['aqi'],
            'cat': info['cat'],
            'pm25': info['pm25'],
            'pm10': info['pm10'],
            'no2': info['no2'],
            'so2': info['so2'],
            'co': info['co'],
            'ozone': info['ozone']
        })
    return render_template('dashboard/map.html', map_data=map_data)

@dashboard_bp.route('/alerts')
@login_required
def alerts():
    # In a real app, this would query the latest 24 hours of data.
    # For now, let's find the highest historical PM2.5 records to simulate active alerts.
    hazardous_records = db.session.query(
        AirQualityRecord, Station
    ).join(Station, AirQualityRecord.station_id == Station.id)\
     .filter(AirQualityRecord.pm25 > 150)\
     .order_by(AirQualityRecord.timestamp.desc())\
     .limit(20).all()
    
    return render_template('dashboard/alerts.html', alerts=hazardous_records)

@dashboard_bp.route('/calculator', methods=['GET', 'POST'])
@login_required
def calculator():
    from app.utils.aqi_calculator import calculate_indian_aqi
    
    result = None
    if request.method == 'POST':
        try:
            pm25 = float(request.form.get('pm25', 0) or 0)
            pm10 = float(request.form.get('pm10', 0) or 0)
            no2 = float(request.form.get('no2', 0) or 0)
            so2 = float(request.form.get('so2', 0) or 0)
            co = float(request.form.get('co', 0) or 0)
            ozone = float(request.form.get('ozone', 0) or 0)
            nh3 = float(request.form.get('nh3', 0) or 0)
            pb = float(request.form.get('pb', 0) or 0)
            co2 = float(request.form.get('co2', 0) or 0)
            voc = float(request.form.get('voc', 0) or 0)
            
            aqi_val, cat = calculate_indian_aqi(pm25, pm10, no2, so2, co, ozone, nh3, pb, co2, voc)
            
            result = {
                'aqi': round(aqi_val),
                'category': cat,
                'pm25': pm25,
                'pm10': pm10,
                'no2': no2,
                'so2': so2,
                'co': co,
                'ozone': ozone,
                'nh3': nh3,
                'pb': pb,
                'co2': co2,
                'voc': voc
            }
        except ValueError:
            pass # Handle invalid float conversion silently for now
            
    return render_template('dashboard/calculator.html', result=result)

@dashboard_bp.route('/export/csv')
@login_required
def export_csv():
    import csv
    from io import StringIO
    from flask import Response
    
    # Calculate average PM2.5 and PM10 per city
    results = db.session.query(
        Station.city,
        func.avg(AirQualityRecord.pm25).label('avg_pm25'),
        func.avg(AirQualityRecord.pm10).label('avg_pm10')
    ).join(AirQualityRecord, Station.id == AirQualityRecord.station_id)\
     .group_by(Station.city).all()
     
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['City', 'Average PM2.5 (µg/m³)', 'Average PM10 (µg/m³)'])
    
    for r in results:
        city = r[0] if r[0] else 'Unknown'
        pm25 = round(r[1], 2) if r[1] else 0
        pm10 = round(r[2], 2) if r[2] else 0
        writer.writerow([city, pm25, pm10])
        
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=aqi_report.csv"}
    )
