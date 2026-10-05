import os
import sys

# Add the parent directory to the path so we can import the app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.extensions import db
from app.models.user import User

def create_admin():
    app = create_app()
    with app.app_context():
        db.create_all()
        
        email = input("Enter admin email: ")
        
        if User.query.filter_by(email=email).first():
            print("User with this email already exists.")
            return
            
        name = input("Enter admin name: ")
        password = input("Enter admin password: ")
        
        admin = User(name=name, email=email, role='ADMIN')
        admin.set_password(password)
        
        db.session.add(admin)
        db.session.commit()
        
        print(f"Admin user {email} created successfully!")

if __name__ == '__main__':
    create_admin()
