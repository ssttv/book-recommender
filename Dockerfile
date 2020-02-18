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
COPY config.py config.py
COPY stomp_receiver.py stomp_receiver.py
COPY memory_manager.py memory_manager.py
COPY boot.sh boot.sh
RUN chmod +x boot.sh

ENV FLASK_APP app.py
ENV FLASK_ENV development
ENV APP_SETTINGS "config.DevelopmentConfig"

# RUN chown -R book-recommender:book-recommender ./
# USER book-recommender

EXPOSE 5000
ENTRYPOINT ["./boot.sh"]