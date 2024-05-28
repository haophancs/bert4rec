# PROJECT: EXPLOITING BERT4REC MODEL TO BUILD THE SEQUENTIAL RECOMMENDATION SYSTEM FOR MOVIELENS DATASETS
## Install and run at once
```
bash RUNME.sh
```

## Detailed steps
### Prepare environment
```
mkdir -p resources/datasets resources/checkpoints resources/checkpoints resources/db logs
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

### Prepare data
Open `.env` file and set `MOVIELENS_VERSION=ml-25m`, or `ml-100k`, `ml-1m`, `ml-10m`, `ml-20m` 
```
python3 run_dump_data.py
```

### Model checkpoint 

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
Note:
- `hidden_size`: token embedding dim
- `seq_length`: length of interaction sequence
- `epochs`: number of training epochs

### Model usage examples 
```
python3 run_example.py
```

### Deploy recommendation API
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
