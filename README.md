# SEQUENTIAL MOVIE RECOMMENDATION SYSTEM POWERED BY BERT4REC MODEL

## Introduction

This project aims to implement the BERT4Rec model, a neural network-based recommendation system that leverages the power of the Transformer architecture. The BERT4Rec model is trained on the MovieLens dataset to provide personalized movie recommendations based on users' interaction histories.

The project encompasses training the BERT4Rec model, evaluating its performance, and deploying a web-based recommendation API using FastAPI and Celery. The API allows users to obtain movie recommendations based on their past interactions with the system.

## Install and Deploy at Once

```
bash run_deploy.sh
```

## Detailed Guidelines

### Prepare Environment

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

Here's the section with notes for each environment variable:
```
MOVIELENS_VERSION=ml-25m  # Specify the MovieLens dataset version to use (ml-25m, ml-100k, ml-1m, ml-10m, or ml-20m)
DATASET_ROOT=./resources/datasets/  # Path to the directory where the dataset files will be stored
DATABASE_ROOT=./resources/db/  # Path to the directory where the database files will be stored

REDIS_HOST=localhost  # Host for the Redis server
REDIS_PORT=6379  # Port for the Redis server
REDIS_CELERY_DB=0  # Redis database number for Celery

RECSYS_HOST=0.0.0.0  # Host for the recommendation system API
RECSYS_PORT=8001  # Port for the recommendation system API
RECSYS_SEQ_LENGTH=120  # Length of the user interaction sequence used by the recommendation system
RECSYS_DEVICE=cpu  # Device to use for the recommendation system (cpu or cuda)
RECSYS_SECRET=secret_key  # Secret key for the recommendation system API

WEB_HOST=0.0.0.0  # Host for the web application
WEB_PORT=8000  # Port for the web application
```

### Prepare Data

Open the `.env` file and set `MOVIELENS_VERSION` to one of the following options: `ml-25m`, `ml-100k`, `ml-1m`, `ml-10m`, or `ml-20m`.

```
python3 run_dump_data.py
```

### Train and Evaluate Model

Note: Only the checkpoint trained on `ml-25m` is provided.

To download the pre-trained checkpoint, run:

```
wget -O resources/checkpoints/bert4rec_ml-25m_best.ckpt https://uithcm-my.sharepoint.com/:u:/g/personal/18520216_ms_uit_edu_vn/EVCvHZg7QFZGlis704IiPdIBMJxIK37tcVGUM9zY-LzlCw?e=tCgA0J&download=1
```

To train the model, run:

```
python3 run_train.py --batch_size 32 --hidden_size 128 --seq_length 120 --epochs 90
```

To test the model, run:
```
python3 run_train.py --epochs 0 --pretrained
```

Note:

- `hidden_size`: token embedding dimension
- `seq_length`: length of the interaction sequence
- `epochs`: number of training epochs

To view Tensorboard while training the model, run:

```
tensorboard --logdir logs --port 6006
```

### Model Inference Example

```
python3 run_example.py
```

### Deploy Recommendation API

```
python3 run_redis.py
```

Then, open another terminal tab and run:

```
celery -A run_api.celery worker --loglevel=info --pool=solo
```

Then, open another terminal tab and run:

```
uvicorn run_api:app --reload
```

## Pre-commit

To install pre-commit, simply run inside the shell:

```bash
pre-commit install
```

pre-commit is very useful for checking code before publishing. It's configured using the `.pre-commit-config.yaml` file.

By default, it runs:

- black (formats code)
- mypy (validates types)
- isort (sorts imports in all files)
- flake8 (spots possible bugs)
