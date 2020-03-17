import os
import config
import flask
import stomp
import gc

import numpy as np
import pandas as pd
import traceback
import pickle

from flask import (Flask, session, g, json, Blueprint,flash, jsonify, redirect, render_template, request,
                   url_for, send_from_directory, send_file)
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename

from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

from dotenv import load_dotenv

# Load environment variables from the .env file with python-dotenv module
load_dotenv()
migrate = Migrate()

def create_app():

    # Initialize main Flask application and allow CORS
    app = Flask(__name__)
    cors = CORS(app)
    cors.init_app(app)

    # Load app config from config.py file, use config variable to point at STOMP/ActiveMQ host and ports
    app.config.from_object(os.environ['APP_SETTINGS'])

    from extensions import db
    from api import api

    with app.app_context():
        db.init_app(app)
        migrate.init_app(app, db)
        app.register_blueprint(api)

    return app

app = create_app()