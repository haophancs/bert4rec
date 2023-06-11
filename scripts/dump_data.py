import json
import os

import redis
from dotenv import load_dotenv
from elasticsearch import Elasticsearch


def create_redis_database(ratings, users, movies, host='localhost', port=6379, password=None, db=0):
    r = redis.Redis(host=host, port=port, password=password, db=db)
    for _, row in ratings.iterrows():
        rating_data = {
            'userId': row['userId'],
            'movieId': row['movieId'],
            'rating': row['rating'],
            'timestamp': row['timestamp']
        }
        r.hset('ratings', row['userId'], json.dumps(rating_data))

    for _, row in users.iterrows():
        user_data = {
            'name': row['name'],
            'password': row['password']
        }
        r.hset('users', row['userId'], json.dumps(user_data))

    for _, row in movies.iterrows():
        movie_data = {
            'title': row['title'],
            'genres': row['genres'],
            'imdbId': row['imdbId'],
            'tmdbId': row['tmdbId']
        }
        r.hset('movies', row['movieId'], json.dumps(movie_data))

    print("Data has been successfully pushed to Redis.")


def create_elasticsearch_index(movies, index_name='movies'):
    load_dotenv()
    es_host = os.getenv('ES_HOST', 'localhost')
    es_port = int(os.getenv('ES_PORT', '9200'))
    es_username = os.getenv('ES_USERNAME', None)
    es_password = os.getenv('ES_PASSWORD', None)

    es = Elasticsearch(
        hosts=[{'host': es_host, 'port': es_port}],
        http_auth=(es_username, es_password) if es_username and es_password else None
    )

    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)

    es.indices.create(index=index_name)

    movie_documents = []
    for _, row in movies.iterrows():
        movie_document = {
            'title': row['title'],
            'genres': row['genres'],
            'imdbId': row['imdbId'],
            'tmdbId': row['tmdbId']
        }
        movie_documents.append(movie_document)

    bulk_body = [
        {
            '_index': index_name,
            '_source': movie_doc
        }
        for movie_doc in movie_documents
    ]
    es.bulk(body=bulk_body)
    print("Movies have been successfully indexed in Elasticsearch.")


if __name__ == '__main__':
    load_dotenv()
    host = os.getenv('REDIS_HOST', 'localhost')
    port = int(os.getenv('REDIS_PORT', '6379'))
    password = os.getenv('REDIS_PASSWORD', None)
    db = int(os.getenv('REDIS_DB', '0'))
    create_redis_database(ratings, users, movies, host=host, port=port, password=password, db=db)
    create_elasticsearch_index(movies)
