import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.extensions import db
from app.models.user import User

def create_test_admin():
    app = create_app()
    with app.app_context():
        email = "admin@test.com"
        password = "password123"
        
        # Create Admin
        admin = User.query.filter_by(email=email).first()
        if not admin:
            admin = User(name="Test Admin", email=email, role='ADMIN')
            admin.set_password(password)
            db.session.add(admin)
            
        # Create Normal User
        user_email = "user@test.com"
        user = User.query.filter_by(email=user_email).first()
        if not user:
            user = User(name="Test User", email=user_email, role='USER')
            user.set_password(password)
            db.session.add(user)
            
        db.session.commit()
        print("Test accounts created successfully!")

if __name__ == '__main__':
    create_test_admin()
