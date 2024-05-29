#!/bin/bash
source venv/bin/activate
nohup python3 -u run_redis.py > logs/redis.out &
nohup celery -A run_api.celery worker --loglevel=info --pool=solo > logs/celery.out &
nohup uvicorn run_api:app --reload > logs/uvicorn.out &

echo "Task are running in background, you can use the PIDs above or check 'jobs -l' to kill them if needed"
echo "> Open logs/redis.out to view the logs of redis server"
echo "> Open logs/celery.out to check if the celery with model predictor is ready"
echo "> Open logs/uvicorn.out to get the demo page url"
echo "Once the celery loaded successfully, we go to the demo page url"
