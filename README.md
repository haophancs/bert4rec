## Install and run at once
```
bash RUNME.sh
```

## Prepare environment
```
python3 -m venv venv
mkdir -p resources/datasets resources/checkpoints resources/checkpoints resources/db logs
pip3 install -r requirements.txt
```

## Prepare data
Open `.env` file and set `MOVIELENS_VERSION=ml-25m`, or `ml-100k`, `ml-1m`, `ml-10m`, `ml-20m` 
```
python3 dump_data.py
```

## Model checkpoint 

Note that only checkpoint trained on `ml-25m` provided.

To download pretrained checkpoint, run:
```
gdown https://drive.google.com/u/0/uc?id=1gf4_zpHd4H-ZH625TvcSEEs95idxYGOa
mv bert4rec_ml-25m_best.ckpt resources/checkpoints/
```
To train the model, run:
```
python3 run_train.py --batch_size 32 --hidden_size 128 --seq_length 120 --epochs 90
```

## Model usage examples 
```
python3 run_example.py
```

## Deploy recommendation API
```
python3 run_redis.py
```
Then open another terminal tab and run:
```
 celery -A run_api.celery worker --loglevel=info --pool=solo
```
Then open another terminal tab and run:
```
uvicorn run_api:app --reload
```