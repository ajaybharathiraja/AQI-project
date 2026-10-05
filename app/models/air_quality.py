from app.extensions import db
from datetime import datetime

class AirQualityRecord(db.Model):
    __tablename__ = 'air_quality_records'
    
    id = db.Column(db.Integer, primary_key=True)
    station_id = db.Column(db.Integer, db.ForeignKey('stations.id'), nullable=False, index=True)
    timestamp = db.Column(db.DateTime, nullable=False, index=True)
    
    # Pollutants
    pm25 = db.Column(db.Float, nullable=True)
    pm10 = db.Column(db.Float, nullable=True)
    no = db.Column(db.Float, nullable=True)
    no2 = db.Column(db.Float, nullable=True)
    nox = db.Column(db.Float, nullable=True)
    nh3 = db.Column(db.Float, nullable=True)
    so2 = db.Column(db.Float, nullable=True)
    co = db.Column(db.Float, nullable=True)
    ozone = db.Column(db.Float, nullable=True)
    benzene = db.Column(db.Float, nullable=True)
    toluene = db.Column(db.Float, nullable=True)
    xylene = db.Column(db.Float, nullable=True)
    
    # Weather / Environment parameters
    rh = db.Column(db.Float, nullable=True)
    wd = db.Column(db.Float, nullable=True)
    sr = db.Column(db.Float, nullable=True)
    bp = db.Column(db.Float, nullable=True)
    at = db.Column(db.Float, nullable=True)
    rf = db.Column(db.Float, nullable=True)
    
    # Calculated values (Not from ML, but standard calculation)
    calculated_aqi = db.Column(db.Float, nullable=True)
    aqi_category = db.Column(db.String(50), nullable=True)
    dominant_pollutant = db.Column(db.String(50), nullable=True)
    
    __table_args__ = (
        db.UniqueConstraint('station_id', 'timestamp', name='uix_station_time'),
        db.Index('idx_station_time', 'station_id', 'timestamp')
    )
    
    def __init__(self, **kwargs):
        super(AirQualityRecord, self).__init__(**kwargs)
        
    def __repr__(self):
        return f"<AirQualityRecord {self.timestamp} Station {self.station_id}>"
