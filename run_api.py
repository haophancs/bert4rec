import logging
import os
import sys
from typing import List, Optional

from celery import Celery
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.data.db import DatabaseRepository as DBRepo
from src.reclib.utils import BERT4RecPredictor

mode = "celery" if "celery" in sys.argv[0] else "fastapi"

load_dotenv()
db_root = os.getenv("DATABASE_ROOT")
movielens_version = os.getenv("MOVIELENS_VERSION")
seq_length = os.getenv("RECSYS_SEQ_LENGTH")
device = os.getenv("RECSYS_DEVICE")

assert db_root is not None
assert movielens_version is not None
assert seq_length is not None
assert device is not None
seq_length = int(seq_length)  # type: ignore

db = DBRepo(os.path.join(db_root, movielens_version + ".db"))

celery = Celery(
    "tasks",
    broker="redis://{0}:{1}/{2}".format(
        os.getenv("REDIS_HOST"),
        os.getenv("REDIS_PORT"),
        os.getenv("REDIS_CELERY_DB"),
    ),
    backend="redis://{0}:{1}/{2}".format(
        os.getenv("REDIS_HOST"),
        os.getenv("REDIS_PORT"),
        os.getenv("REDIS_CELERY_DB"),
    ),
)
app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

templates = Jinja2Templates(directory="templates")


@app.post("/recommend")
async def recommend(
    item_sequence: List[int],
    avoided_list: Optional[List[int]] = None,
    k: int = 5,
):
    if avoided_list is None:
        avoided_list = []
    logger.info("Received recommendation request.")
    result = recommend_task.delay(item_sequence, avoided_list, k)
    logger.info("Recommendation task enqueued.")
    return {"task_id": result.id}


@app.get("/status/{task_id}")
def task_status(task_id: str):
    task = celery.AsyncResult(task_id)
    if task.state == "SUCCESS":
        return {"status": "completed", "result": task.result}
    elif task.state == "PENDING":
        return {"status": "pending"}
    else:
        return {"status": "failed"}


@app.get("/result/{task_id}")
def task_result(task_id: str):
    task = celery.AsyncResult(task_id)
    if task.state == "SUCCESS":
        return {"result": task.result}
    else:
        return {"error": "Recommendation task not completed."}


@celery.task
def recommend_task(item_sequence, avoided_list, k):
    movie_ids = predictor.predict(item_sequence, avoided_list)[:k]
    return [
        db.get_movie_by_id(movie_id, ["movieId", "title", "genres"])
        for movie_id in movie_ids
    ]


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@celery.task
def get_movie_task(movieId: str):
    cols = ["movieId", "title", "genres"]
    movie_data = db.get_movie_by_id(movieId, cols)
    return {cols[i]: movie_data[i] for i in range(len(cols))}


@app.get("/movie/{movieId}")
def get_movie(movieId: int):
    logger.info("Received getting movie request.")
    result = get_movie_task.delay(movieId)
    logger.info("Getting movie task enqueued.")
    return {"taskId": result.id}


logger.info("Loading predictor...")
predictor = None

if mode == "celery":
    predictor = BERT4RecPredictor(
        os.path.join(
            "resources/checkpoints/",
            f"bert4rec_{movielens_version}_best.ckpt",
        ),
        data_root=db_root,
        data_name=movielens_version,
        seq_length=seq_length,
        device=device,
    )
