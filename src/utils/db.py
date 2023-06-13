import sqlite3
from datetime import datetime, timedelta
from typing import List


class DatabaseRepository:
    def __init__(self, db_file):
        self.conn = sqlite3.connect(db_file)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                interactionId INTEGER PRIMARY KEY AUTOINCREMENT,
                userId INTEGER,
                movieId INTEGER,
                timestamp TEXT,
                FOREIGN KEY (userId) REFERENCES users(userId),
                FOREIGN KEY (movieId) REFERENCES movies(movieId)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                userId INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                password TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS movies (
                movieId INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                genres TEXT,
                year INTEGER,
                popularity REAL,
                imdbId TEXT,
                tmdbId TEXT
            )
        ''')

        self.conn.commit()

    def get_interactions(self, cols='*'):
        if isinstance(cols, List):
            cols = ', '.join(cols)
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT {cols} FROM interactions")
        rows = cursor.fetchall()
        return rows

    def get_users(self, cols='*'):
        if isinstance(cols, List):
            cols = ', '.join(cols)
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT {cols} FROM users")
        rows = cursor.fetchall()
        return rows

    def get_movies(self, cols='*'):
        if isinstance(cols, List):
            cols = ', '.join(cols)
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT {cols} FROM movies")
        rows = cursor.fetchall()
        return rows

    def get_recent_user_interactions(self, user_id, k=None, minutes=None, cols='*'):
        if isinstance(cols, list):
            cols = ', '.join(cols)

        query = f'''
            SELECT {cols}
            FROM interactions
            WHERE userId = ?
        '''

        params = [user_id]

        if minutes is not None:
            end_timestamp = datetime.utcnow()
            start_timestamp = end_timestamp - timedelta(minutes=minutes)
            query += ' AND timestamp BETWEEN ? AND ?'
            params.extend([start_timestamp, end_timestamp])

        if k is not None:
            query += ' ORDER BY timestamp DESC LIMIT ?'
            params.append(k)

        cursor = self.conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return rows

    def get_movies_interacted_by_user(self, user_id, cols='*'):
        if isinstance(cols, List):
            cols = ', '.join(cols)
        cursor = self.conn.cursor()
        cursor.execute(f'''
            SELECT {cols}
            FROM movies
            INNER JOIN interactions ON movies.movieId = interactions.movieId
            WHERE interactions.userId = ?
        ''', (user_id,))
        rows = cursor.fetchall()
        return rows

    def get_users_interacted_with_movie(self, movie_id, cols='*'):
        if isinstance(cols, List):
            cols = ', '.join(cols)
        cursor = self.conn.cursor()
        cursor.execute(f'''
            SELECT {cols}
            FROM users
            INNER JOIN interactions ON users.userId = interactions.userId
            WHERE interactions.movieId = ?
        ''', (movie_id,))
        rows = cursor.fetchall()
        return rows

    def search_movie_by_title(self, title):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM movies WHERE title LIKE ?", ('%' + title + '%',))
        rows = cursor.fetchall()
        return rows

    def filter_movies_by_genres(self, genre, k=None):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM movies WHERE genres LIKE ?", ('%' + genre + '%',))
        if k is not None:
            rows = cursor.fetchmany(k)
        else:
            rows = cursor.fetchall()
        return rows

    def filter_movies_by_year(self, year, k=None):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM movies WHERE year = ?", (year,))
        if k is not None:
            rows = cursor.fetchmany(k)
        else:
            rows = cursor.fetchall()
        return rows

    def filter_movies_by_popularity(self, min_popularity, max_popularity, k=None):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM movies WHERE popularity BETWEEN ? AND ?", (min_popularity, max_popularity))
        if k is not None:
            rows = cursor.fetchmany(k)
        else:
            rows = cursor.fetchall()
        return rows

    def get_top_popularity_movies(self, k=None):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM movies ORDER BY popularity DESC")
        if k is not None:
            rows = cursor.fetchmany(k)
        else:
            rows = cursor.fetchall()
        return rows

    def get_top_latest_movies(self, k=None):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM movies ORDER BY year DESC")
        if k is not None:
            rows = cursor.fetchmany(k)
        else:
            rows = cursor.fetchall()
        return rows

    def close_connection(self):
        self.conn.close()

    @staticmethod
    def dump_from_df(df, table_name, db_file):
        conn = sqlite3.connect(db_file)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()
