#!/bin/bash
python3 -m venv venv
source venv/bin/activate
mkdir -p resources/datasets resources/checkpoints resources/checkpoints resources/db logs
pip3 install -r requirements.txt
cp .env_org .env
python3 run_dump_data.py
wget -O resources/checkpoints/bert4rec_ml-25m_best.ckpt https://uithcm-my.sharepoint.com/:u:/g/personal/18520216_ms_uit_edu_vn/EVCvHZg7QFZGlis704IiPdIBMJxIK37tcVGUM9zY-LzlCw?e=tCgA0J&download=1

echo "Open logs/celery.out to check if the predictor is ready!!"
echo "Open logs/uvicorn.out to get the running API url!!"
nohup python3 -u run_redis.py > logs/redis.out &
nohup celery -A run_api.celery worker --loglevel=info --pool=solo > logs/celery.out &
nohup uvicorn run_api:app --reload > logs/uvicorn.out &
