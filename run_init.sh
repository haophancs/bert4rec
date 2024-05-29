#!/bin/bash
python3 -m venv venv
source venv/bin/activate
mkdir -p resources/datasets resources/checkpoints resources/db logs
pip3 install -r requirements.txt
cp .env_org .env
wget -O resources/checkpoints/bert4rec_ml-25m_best.ckpt https://uithcm-my.sharepoint.com/:u:/g/personal/18520216_ms_uit_edu_vn/EVCvHZg7QFZGlis704IiPdIBMJxIK37tcVGUM9zY-LzlCw?e=tCgA0J&download=1
wget -O resources/db/ml-25m.db https://uithcm-my.sharepoint.com/:u:/g/personal/18520216_ms_uit_edu_vn/EYa_K9yjDGdFtMKJtRZpHUwBsRtJSw1V5JedZJA6-Ch42g?e=yUDQaZ&download=1
