# PROJECT: EXPLOITING BERT4REC MODEL TO BUILD THE SEQUENTIAL RECOMMENDATION SYSTEM FOR MOVIELENS DATASETS
## Install and deploy at once
```
bash run_deploy.sh
```

## Detailed guidelines
### Prepare environment
```
mkdir -p resources/datasets resources/checkpoints resources/db logs
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```
Prepare environment variables:
```
cp .env_org .env
# now you can edit .env
```

### Prepare data
Open `.env` file and set `MOVIELENS_VERSION=ml-25m`, or `ml-100k`, `ml-1m`, `ml-10m`, `ml-20m`
```
python3 run_dump_data.py
```

### Train and evaluate model

Note that only checkpoint trained on `ml-25m` provided.

To download pretrained checkpoint, run:
```
wget -O resources/checkpoints/bert4rec_ml-25m_best https://uithcm-my.sharepoint.com/:u:/g/personal/18520216_ms_uit_edu_vn/EVCvHZg7QFZGlis704IiPdIBMJxIK37tcVGUM9zY-LzlCw?e=tCgA0J&download=1
```
To train the model, run:
```
python3 run_train.py --batch_size 32 --hidden_size 128 --seq_length 120 --epochs 90
```

Note:
- `hidden_size`: token embedding dim
- `seq_length`: length of interaction sequence
- `epochs`: number of training epochs

To view Tensorboard while training model, run:
```
tensorboard --logdir logs --port 6006
```

### Model inference example
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

## Pre-commit

To install pre-commit simply run inside the shell:
```bash
pre-commit install
```

pre-commit is very useful to check code before publishing.
It's configured using .pre-commit-config.yaml file.

By default it runs:
* black (formats code);
* mypy (validates types);
* isort (sorts imports in all files);
* flake8 (spots possible bugs);
