import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev_key')
    
    # Database
    basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    
    # Check if running in Vercel
    if os.environ.get('VERCEL'):
        import gzip
        import shutil
        tmp_db = '/tmp/air_quality.db'
        gz_db = os.path.join(basedir, 'air_quality.db.gz')
        if not os.path.exists(tmp_db) and os.path.exists(gz_db):
            with gzip.open(gz_db, 'rb') as f_in:
                with open(tmp_db, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        db_uri = 'sqlite:///' + tmp_db
    else:
        db_uri = 'sqlite:///' + os.path.join(basedir, 'air_quality.db')

    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or db_uri
    SQLALCHEMY_TRACK_MODIFICATIONS = False

