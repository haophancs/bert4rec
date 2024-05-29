import os
from typing import List

import pandas as pd
from dotenv import load_dotenv

from src.reclib.utils import BERT4RecPredictor

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

print("Loading model with pretrained weights...")
predictor = BERT4RecPredictor(
    os.path.join(
        "resources/checkpoints/",
        "bert4rec_{0}_best.ckpt".format(movielens_version),
    ),
    data_root=db_root,
    data_name=movielens_version,
    seq_length=seq_length,
    device=device,
)

movies = pd.read_csv("./resources/datasets/{0}/movies.csv".format(movielens_version))
movies.set_index("movieId", inplace=True)
movies.sample(5)

allMovieIds: List[int] = [115617, 112852, 89745, 112556]
allRecommends: List[int] = []

for k in range(1, len(allMovieIds) + 1):
    print("Input interacted movie sequence:")
    movieIds = allMovieIds[:k]
    print(movies.loc[movieIds].values)
    print("---------------------")

    recommends = predictor.predict(movieIds, allRecommends)

    allRecommends.extend(recommends[:5])

    print("Ranked list of next movie recommendation:")
    for movieId in recommends[:5]:
        print(movieId, movies.loc[movieId].values)
    print("=====================")
    print()
    print()
    print("=====================")
