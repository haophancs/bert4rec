import os

import pandas as pd
from dotenv import load_dotenv

from src.reclib.utils import BERT4RecPredictor

load_dotenv()

predictor = BERT4RecPredictor(
    os.path.join('./resources/checkpoints/', f"bert4rec_{os.getenv('MOVIELENS_VERSION')}_best.ckpt"),
    data_root=os.getenv('DATABASE_ROOT'),
    data_name=os.getenv('MOVIELENS_VERSION'),
    seq_length=int(os.getenv('RECSYS_SEQ_LENGTH')),
    device=os.getenv('RECSYS_DEVICE')
)

movies = pd.read_csv(f"./resources/datasets/{os.getenv('MOVIELENS_VERSION')}/movies.csv")
movies.set_index('movieId', inplace=True)
movies.sample(5)

allMovieIds = [115617, 112852, 89745, 112556]
allRecommends = []

for k in range(1, len(allMovieIds) + 1):
    print('Input interacted movie sequence:')
    movieIds = allMovieIds[:k]
    print(movies.loc[movieIds].values)
    print("---------------------")

    recommends = predictor.predict(movieIds, allRecommends)

    allRecommends.extend(recommends[:5])

    print('Ranked list of next movie recommendations:')
    for movieId in recommends[:5]:
        print(movieId, movies.loc[movieId].values)
    print('=====================')
    print()
    print()
    print('=====================')
