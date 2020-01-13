#!/bin/bash
source ./venv/bin/activate
exec gunicorn -b 0.0.0.0:5000 -w 1 app:app