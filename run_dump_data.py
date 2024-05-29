import os

from dotenv import load_dotenv

from src.data.db import DatabaseRepository as DBRepo
from src.data.movielens import read_movielens

if __name__ == '__main__':
    load_dotenv()
    db_root = os.getenv('DATABASE_ROOT')
    data_root = os.getenv('DATASET_ROOT')
    data_name = os.getenv('MOVIELENS_VERSION')
    db = os.path.join(db_root, f"{data_name}.db")

    interactions, users, movies = read_movielens(data_name, data_root)
    interactions = interactions.drop(columns=['rating'])

    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').fillna('1995').astype(int)

    popularity = interactions.groupby('movieId').agg({'userId': 'nunique'}).reset_index()
    popularity.rename(columns={'userId': 'number_of_users_interacted'}, inplace=True)
    total_users = interactions['userId'].nunique()
    popularity['popularity'] = popularity['number_of_users_interacted'] / total_users
    movies = movies.merge(popularity, on='movieId', how='left').drop(columns='number_of_users_interacted')

    db_repo = DBRepo(db)
    DBRepo.dump_from_df(interactions, 'interactions', db)
    DBRepo.dump_from_df(users, 'users', db)
    DBRepo.dump_from_df(movies, 'movies', db)
    db_repo.close_connection()
