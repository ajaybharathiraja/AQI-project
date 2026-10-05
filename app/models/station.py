from app.extensions import db

class Station(db.Model):
    __tablename__ = 'stations'
    
    id = db.Column(db.Integer, primary_key=True)
    city = db.Column(db.String(100), nullable=False, index=True)
    station_name = db.Column(db.String(200), nullable=False, index=True)
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    
    records = db.relationship('AirQualityRecord', backref='station_ref', lazy=True)
    
    def __init__(self, city=None, station_name=None, latitude=None, longitude=None):
        self.city = city
        self.station_name = station_name
        self.latitude = latitude
        self.longitude = longitude
    
    def __repr__(self):
        return f"<Station {self.station_name} ({self.city})>"
