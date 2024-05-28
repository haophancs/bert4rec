import os
import urllib.request
import zipfile

import pandas as pd


def read_ml_100k(data_root):
    dataset_name = "ml-100k"
    dataset_dir = os.path.join(data_root, dataset_name)
    if not os.path.exists(dataset_dir):
        download_and_extract(dataset_name, data_root)

    ratings_file = os.path.join(dataset_dir, "u.data")
    movies_file = os.path.join(dataset_dir, "u.item")
    genre_file = os.path.join(dataset_dir, "u.genre")

    ratings = pd.read_table(ratings_file, sep="\t", header=None, names=["userId", "movieId", "rating", "timestamp"])
    user_ids = ratings["userId"].unique()
    users = pd.DataFrame(
        {"userId": user_ids, "name": [f"dummy_{uid}" for uid in user_ids], "password": "dummy_password"})

    movies = pd.read_table(movies_file, sep="|", encoding="latin-1", header=None,
                           names=["movieId", "title", "release_date", "video_release_date", "imdb_url", "genre1",
                                  "genre2",
                                  "genre3", "genre4", "genre5", "genre6", "genre7", "genre8", "genre9", "genre10",
                                  "genre11",
                                  "genre12", "genre13", "genre14", "genre15", "genre16", "genre17", "genre18",
                                  "genre19"],
                           usecols=["movieId", "title", "genre1", "genre2", "genre3", "genre4", "genre5", "genre6",
                                    "genre7",
                                    "genre8", "genre9", "genre10", "genre11", "genre12", "genre13", "genre14",
                                    "genre15",
                                    "genre16", "genre17", "genre18", "genre19"])

    genre_mapping = {
        int(k) + 1: v[0] for k, v in
        pd.read_table(genre_file, sep='|', header=None).set_index(1).T.to_dict().items()
    }

    genre_columns = [f"genre{i}" for i in range(1, 19)]
    movies[genre_columns] = movies[genre_columns].replace({1: True, 0: False}).astype(bool)
    movies["genres"] = movies[genre_columns].apply(
        lambda row: "|".join([genre_mapping[i + 1] for i, v in enumerate(row) if v]), axis=1)
    movies = movies[['movieId', 'title', 'genres']]

    movies["imdbId"] = None
    movies["tmdbId"] = None

    return ratings, users, movies


def read_ml_1m(data_root):
    dataset_name = "ml-1m"
    dataset_dir = os.path.join(data_root, dataset_name)
    if not os.path.exists(dataset_dir):
        download_and_extract(dataset_name, data_root)

    ratings_file = os.path.join(dataset_dir, "ratings.dat")
    users_file = os.path.join(dataset_dir, "users.dat")
    movies_file = os.path.join(dataset_dir, "movies.dat")

    ratings = pd.read_table(ratings_file, sep="::", engine="python", header=None, encoding='latin',
                            names=["userId", "movieId", "rating", "timestamp"])
    user_ids = ratings["userId"].unique()
    users = pd.DataFrame(
        {"userId": user_ids, "name": [f"dummy_{uid}" for uid in user_ids], "password": "dummy_password"})

    movies = pd.read_table(movies_file, sep="::", engine="python", header=None, encoding='latin',
                           names=["movieId", "title", "genres"])

    movies["imdbId"] = None
    movies["tmdbId"] = None

    return ratings, users, movies


def read_ml_10m(data_root):
    dataset_name = "ml-10m"
    dataset_dir = os.path.join(data_root, dataset_name)
    if not os.path.exists(dataset_dir):
        download_and_extract(dataset_name, data_root)

    ratings_file = os.path.join(dataset_dir, "ratings.dat")
    movies_file = os.path.join(dataset_dir, "movies.dat")

    ratings = pd.read_csv(ratings_file, sep="::", engine="python", header=None,
                          names=["userId", "movieId", "rating", "timestamp"])
    user_ids = ratings["userId"].unique()
    users = pd.DataFrame(
        {"userId": user_ids, "name": [f"dummy_{uid}" for uid in user_ids], "password": "dummy_password"})

    movies = pd.read_csv(movies_file, sep="::", engine="python", header=None,
                         names=["movieId", "title", "genres"])

    movies["imdbId"] = None
    movies["tmdbId"] = None

    return ratings, users, movies


def read_ml_20m(data_root):
    dataset_name = "ml-20m"
    dataset_dir = os.path.join(data_root, dataset_name)
    if not os.path.exists(dataset_dir):
        download_and_extract(dataset_name, data_root)

    ratings_file = os.path.join(dataset_dir, "ratings.csv")
    movies_file = os.path.join(dataset_dir, "movies.csv")

    ratings = pd.read_csv(ratings_file, header=0, names=["userId", "movieId", "rating", "timestamp"])
    user_ids = ratings["userId"].unique()
    users = pd.DataFrame(
        {"userId": user_ids, "name": [f"dummy_{uid}" for uid in user_ids], "password": "dummy_password"})

    movies = pd.read_csv(movies_file, header=0, names=["movieId", "title", "genres"])

    links_file = os.path.join(dataset_dir, "links.csv")
    links = pd.read_csv(links_file, header=0, names=["movieId", "imdbId", "tmdbId"])

    movies = movies.merge(links, on="movieId")

    return ratings, users, movies


def read_ml_25m(data_root):
    dataset_name = "ml-25m"
    dataset_dir = os.path.join(data_root, dataset_name)
    if not os.path.exists(dataset_dir):
        download_and_extract(dataset_name, data_root)

    ratings_file = os.path.join(dataset_dir, "ratings.csv")
    movies_file = os.path.join(dataset_dir, "movies.csv")

    ratings = pd.read_csv(ratings_file, header=0, names=["userId", "movieId", "rating", "timestamp"])
    user_ids = ratings["userId"].unique()
    users = pd.DataFrame(
        {"userId": user_ids, "name": [f"dummy_{uid}" for uid in user_ids], "password": "dummy_password"})

    movies = pd.read_csv(movies_file, header=0, names=["movieId", "title", "genres"])

    links_file = os.path.join(dataset_dir, "links.csv")
    links = pd.read_csv(links_file, header=0, names=["movieId", "imdbId", "tmdbId"])

    movies = movies.merge(links, on="movieId")

    return ratings, users, movies


def download_and_extract(dataset_name, data_root):
    print(f"Downloading and extracting {dataset_name} dataset...")
    download_url = f"http://files.grouplens.org/datasets/movielens/{dataset_name}.zip"
    urllib.request.urlretrieve(download_url, f"{dataset_name}.zip")
    # os.system(f"wget {download_url}")
    with zipfile.ZipFile(f"{dataset_name}.zip", "r") as zip_ref:
        zip_ref.extractall(data_root)
    os.remove(f"{dataset_name}.zip")
    print("Download and extraction complete.")


def read_movielens(version, data_root):
    if version == 'ml-100k':
        return read_ml_100k(data_root)
    if version == 'ml-1m':
        return read_ml_1m(data_root)
    if version == 'ml-10m':
        return read_ml_10m(data_root)
    if version == 'ml-20m':
        return read_ml_20m(data_root)
    if version == 'ml-25m':
        return read_ml_25m(data_root)
    raise ValueError('Unsupported dataset')
