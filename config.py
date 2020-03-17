import os
basedir = os.path.abspath(os.path.dirname(__file__))

class Config(object):
    DEBUG = False
    TESTING = False
    HOSTS_AND_PORTS = [('0.0.0.0', 61613)]
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    BASE_DIR = os.getcwd()
    CORS_HEADERS = 'Content-Type'
    JSON_AS_ASCII = False
    JSON_SORT_KEYS = False
    MEMORY_PERCENTAGE = 0.8
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    CELERY_BROKER_URL = 'amqp://guest:guest@localhost:5672/'
    CELERY_RESULT_BACKEND = '+'.join(['db', os.environ.get('DATABASE_URL')])
    # CELERY_RESULT_BACKEND = 'db+mysql+mysqldb://admin:password@localhost/recommender?use_unicode=1&charset=utf8'
    CELERY_IMPORTS = ('tasks', )
    CELERY_ANNOTATIONS = {}
    
class ProductionConfig(Config):
    DATABASE_URI = ''

class StagingConfig(Config):
    DEVELOPMENT = True
    DEBUG = True

class DevelopmentConfig(Config):
    DEVELOPMENT = True
    DEBUG = True
    PROFILE = True

class TestingConfig(Config):
    TESTING = True
