import os
basedir = os.path.abspath(os.path.dirname(__file__))

class Config(object):
    DEBUG = False
    TESTING = False
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    BASE_DIR = os.getcwd()
    CORS_HEADERS = 'Content-Type'
    JSON_AS_ASCII = False
    JSON_SORT_KEYS = False

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
