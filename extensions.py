import sqlalchemy.dialects.mysql.pymysql

from flask_sqlalchemy import SQLAlchemy
from celery import Celery

db = SQLAlchemy()

def make_celery(app):
    celery = Celery(app.import_name, backend=app.config["CELERY_RESULT_BACKEND"], broker=app.config["CELERY_BROKER_URL"])
    celery.conf.update(app.config)
    TaskBase = celery.Task
    class ContextTask(TaskBase):
        abstract = True
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)
    celery.app = app
    celery.Task = ContextTask
    return celery