#!/bin/bash
python3 -m venv venv
source venv/bin/activate
mkdir -p resources/datasets resources/checkpoints resources/checkpoints resources/db logs
pip3 install -r requirements.txt
python3 run_dump_data.py
gdown https://drive.google.com/u/0/uc?id=1gf4_zpHd4H-ZH625TvcSEEs95idxYGOa
mv bert4rec_ml-25m_best.ckpt resources/checkpoints/

echo "Open logs/celery.out to check if predictor is ready!!"
echo "Open logs/uvicorn.out to get the running API url!!"
nohup python3 -u run_redis.py > logs/redis.out &
nohup celery -A run_api.celery worker --loglevel=info --pool=solo > logs/celery.out &
nohup uvicorn run_api:app --reload > logs/uvicorn.out &
