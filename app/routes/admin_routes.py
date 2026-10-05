from flask import Blueprint, render_template, redirect, url_for, flash, abort
from flask_login import login_required, current_user
from app.extensions import db
from app.models.user import User
from app.models.station import Station
from app.models.air_quality import AirQualityRecord

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

@admin_bp.before_request
@login_required
def require_admin():
    if not current_user.is_admin:
        abort(403)

@admin_bp.route('/')
def dashboard():
    total_users = User.query.count()
    total_cities = db.session.query(Station.city).distinct().count()
    total_stations = Station.query.count()
    total_records = AirQualityRecord.query.count()
    
    return render_template('admin/dashboard.html', 
                           total_users=total_users,
                           total_cities=total_cities,
                           total_stations=total_stations,
                           total_records=total_records)

@admin_bp.route('/users')
def users():
    users_list = User.query.all()
    return render_template('admin/users.html', users=users_list)

@admin_bp.route('/datasets')
def datasets():
    return render_template('admin/datasets.html')

@admin_bp.route('/alerts')
def alerts():
    return render_template('admin/alerts.html')
