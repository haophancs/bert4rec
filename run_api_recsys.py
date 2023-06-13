import logging
import os
import sys
from typing import List

from celery import Celery
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.reclib.helpers import Bert4RecPredictor

mode = 'celery' if 'celery' in sys.argv[0] else 'fastapi'

load_dotenv()

celery = Celery(
    'tasks',
    broker=f'redis://{os.getenv("REDIS_HOST")}:{os.getenv("REDIS_PORT")}/{os.getenv("REDIS_CELERY_DB")}',
    backend=f'redis://{os.getenv("REDIS_HOST")}:{os.getenv("REDIS_PORT")}/{os.getenv("REDIS_CELERY_DB")}'
)
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mount the static directory to serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Use Jinja2 templates
templates = Jinja2Templates(directory="templates")


@app.post('/recommend')
async def recommend(item_sequence: List[int], avoided_list: List[int] = None, k: int = 5):
    avoided_list = [] if avoided_list is None else avoided_list
    logger.info('Received recommendation request.')
    result = recommend_task.delay(item_sequence, avoided_list, k)
    logger.info('Recommendation task enqueued.')
    return {'task_id': result.id}


@app.get('/status/{task_id}')
def task_status(task_id: str):
    task = celery.AsyncResult(task_id)
    if task.state == 'SUCCESS':
        return {'status': 'completed', 'result': task.result}
    elif task.state == 'PENDING':
        return {'status': 'pending'}
    else:
        return {'status': 'failed'}


@app.get('/result/{task_id}')
def task_result(task_id: str):
    task = celery.AsyncResult(task_id)
    if task.state == 'SUCCESS':
        return {'result': task.result}
    else:
        return {'error': 'Recommendation task not completed.'}


@celery.task
def recommend_task(item_sequence, avoided_list, k):
    return predictor.predict(item_sequence, avoided_list)[:k]


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("recservice.html", {"request": request})


logger.info('Loading predictor...')
predictor = None

if mode == 'celery':
    predictor = Bert4RecPredictor(
        os.path.join('./resources/checkpoints/', f"bert4rec_{os.getenv('MOVIELENS_VERSION')}_best.ckpt"),
        data_root=os.getenv('DATABASE_ROOT'),
        data_name=os.getenv('MOVIELENS_VERSION'),
        seq_length=int(os.getenv('RECSYS_SEQ_LENGTH')),
        device=os.getenv('RECSYS_DEVICE')
    )
