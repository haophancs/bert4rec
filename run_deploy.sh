#!/bin/bash
source venv/bin/activate
echo "Open logs/celery.out to check if the predictor is ready!!"
echo "Open logs/uvicorn.out to get the running API url!!"
nohup python3 -u run_redis.py > logs/redis.out &
nohup celery -A run_api.celery worker --loglevel=info --pool=solo > logs/celery.out &
nohup uvicorn run_api:app --reload > logs/uvicorn.out &
