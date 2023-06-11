import os

from dotenv import load_dotenv

from src.utils.db import DatabaseRepository as DBRepo
from src.utils.movielens import read_movielens

if __name__ == '__main__':
    load_dotenv()
    db_root = os.getenv('DATABASE_ROOT')
    data_root = os.getenv('DATASET_ROOT')
    data_name = os.getenv('MOVIELENS_VERSION')
    db = os.path.join(db_root, f"{data_name}.db")

    interactions, users, movies = read_movielens(data_name, data_root)
    interactions = interactions.drop(columns=['rating'])

    db_repo = DBRepo(db)
    DBRepo.dump_from_df(interactions, 'user_interacted', db)
    DBRepo.dump_from_df(users, 'users', db)
    DBRepo.dump_from_df(movies, 'movies', db)
    db_repo.close_connection()
