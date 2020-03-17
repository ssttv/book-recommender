FROM ubuntu:18.04
FROM python:3.7

RUN adduser --disabled-password --gecos '' book-recommender
VOLUME /vol/
WORKDIR /home/book-recommender

COPY dataset /vol/dataset
COPY requirements.txt requirements.txt

RUN apt-get update
RUN apt-get -y install build-essential python-scipy python-numpy
RUN python -m venv venv
RUN venv/bin/pip install --no-cache-dir --upgrade pip
RUN venv/bin/pip install --no-cache-dir Cython 
RUN venv/bin/pip install --no-cache-dir numpy
RUN venv/bin/pip install --no-cache-dir -r requirements.txt

COPY app.py app.py
COPY api.py api.py
COPY funcs funcs
COPY extensions.py extensions.py
COPY config.py config.py
COPY models.py models.py
COPY boot.sh boot.sh
RUN chmod +x boot.sh

ENV FLASK_APP app.py
ENV FLASK_ENV development
ENV APP_SETTINGS "config.DevelopmentConfig"
ENV DATABASE_URL="mysql+pymysql://admin:password@localhost:3306/recommender?use_unicode=1&charset=utf8mb4&binary_prefix=true"

# RUN chown -R book-recommender:book-recommender ./
# USER book-recommender

EXPOSE 5000
ENTRYPOINT ["./boot.sh"]