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
Open `test.ipynb file`

## Deploy recommendation API
```
python3 run_redis.py
```
Then open another terminal tab and run:
```
celery -A run_api_recsys.celery worker --loglevel=info   
```
Then open another terminal tab and run:
```
uvicorn run_api_recsys:app
```

## Deploy demo website
```
uvicorn run_website:app
```