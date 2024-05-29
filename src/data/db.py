import sqlite3
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Union


class DatabaseRepository:
    def __init__(self, db_file: str, check_same_thread=True):
        self.conn = sqlite3.connect(db_file, check_same_thread=check_same_thread)
        self.create_tables()

    def create_tables(self) -> None:
        """
        Create the necessary tables (interactions, users, movies) if they don't exist.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                interactionId INTEGER PRIMARY KEY AUTOINCREMENT,
                userId INTEGER,
                movieId INTEGER,
                timestamp TEXT,
                FOREIGN KEY (userId) REFERENCES users(userId),
                FOREIGN KEY (movieId) REFERENCES movies(movieId)
            )
        """,
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                userId INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                password TEXT
            )
        """,
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS movies (
                movieId INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                genres TEXT,
                year INTEGER,
                popularity REAL,
                imdbId TEXT,
                tmdbId TEXT
            )
        """,
        )

        self.conn.commit()

    def get_interactions(self, cols: Union[str, List[str]] = "*") -> List[Tuple]:
        """
        Retrieve interactions from the database.

        :param cols: A list of column names or "*" to select all columns
        :return: A list of tuples representing the interactions
        """
        if isinstance(cols, list):
            cols = ", ".join(cols)
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT {cols} FROM interactions")
        rows = cursor.fetchall()
        return rows

    def get_users(self, cols: Union[str, List[str]] = "*") -> List[Tuple]:
        """
        Retrieve users from the database.

        :param cols: A list of column names or "*" to select all columns
        :return: A list of tuples representing the users
        """
        if isinstance(cols, list):
            cols = ", ".join(cols)
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT {cols} FROM users")
        rows = cursor.fetchall()
        return rows

    def get_movies(self, cols: Union[str, List[str]] = "*") -> List[Tuple]:
        """
        Retrieve movies from the database.

        :param cols: A list of column names or "*" to select all columns
        :return: A list of tuples representing the movies
        """
        if isinstance(cols, list):
            cols = ", ".join(cols)
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT {cols} FROM movies")
        rows = cursor.fetchall()
        return rows

    def get_recent_user_interactions(
        self,
        user_id: int,
        k: Optional[int] = None,
        minutes: Optional[int] = None,
        cols: Union[str, List[str]] = "*",
    ) -> List[Tuple]:
        """
        Retrieve the recent interactions of a user.

        :param user_id: The user ID
        :param k: The maximum number of interactions to retrieve, defaults to None
        :param minutes: The maximum age of interactions in minutes, defaults to None
        :param cols: A list of column names or "*" to select all columns
        :return: A list of tuples representing the recent interactions
        """
        if isinstance(cols, list):
            cols = ", ".join(cols)

        query = f"""
            SELECT {cols}
            FROM interactions
            WHERE userId = ?
        """

        params = [user_id]

        if minutes is not None:
            end_timestamp = datetime.utcnow()
            start_timestamp = end_timestamp - timedelta(minutes=minutes)
            query += " AND timestamp BETWEEN ? AND ?"
            params.extend([start_timestamp, end_timestamp])  # type: ignore

        if k is not None:
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(k)

        cursor = self.conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return rows

    def get_movies_interacted_by_user(
        self,
        user_id: int,
        cols: Union[str, List[str]] = "*",
    ) -> List[Tuple]:
        """
        Retrieve the movies that a user has interacted with.

        :param user_id: The user ID
        :param cols: A list of column names or "*" to select all columns
        :return: A list of tuples representing the movies
        """
        if isinstance(cols, list):
            cols = ", ".join(cols)
        cursor = self.conn.cursor()
        cursor.execute(
            f"""
            SELECT {cols}
            FROM movies
            INNER JOIN interactions ON movies.movieId = interactions.movieId
            WHERE interactions.userId = ?
        """,
            (user_id,),
        )
        rows = cursor.fetchall()
        return rows

    def get_users_interacted_with_movie(
        self,
        movie_id: int,
        cols: Union[str, List[str]] = "*",
    ) -> List[Tuple]:
        """
        Retrieve the users who have interacted with a movie.

        :param movie_id: The movie ID
        :param cols: A list of column names or "*" to select all columns
        :return: A list of tuples representing the users
        """
        if isinstance(cols, list):
            cols = ", ".join(cols)
        cursor = self.conn.cursor()
        cursor.execute(
            f"""
            SELECT {cols}
            FROM users
            INNER JOIN interactions ON users.userId = interactions.userId
            WHERE interactions.movieId = ?
        """,
            (movie_id,),
        )
        rows = cursor.fetchall()
        return rows

    def get_movie_by_id(
        self,
        movie_id: int,
        cols: Union[str, List[str]] = "*",
    ) -> Optional[Tuple]:
        """
        Retrieve a movie by its ID.

        :param movie_id: The movie ID
        :param cols: A list of column names or "*" to select all columns
        :return: A tuple representing the movie, or None if not found
        """
        if isinstance(cols, list):
            cols = ", ".join(cols)
        cursor = self.conn.cursor()
        cursor.execute(
            f"""
            SELECT {cols}
            FROM movies
            WHERE movieId = ?
        """,
            (movie_id,),
        )
        row = cursor.fetchone()
        return row

    def search_movies_by_title(
        self,
        search_term: str,
        limit: Optional[int] = None,
    ) -> List[Tuple]:
        """
        Search for movies by their title.

        :param search_term: The title or partial title to search for
        :param limit: The maximum number of results to return
        :return: A list of tuples representing the matching movies
        """
        cursor = self.conn.cursor()
        query = "SELECT * FROM movies WHERE title LIKE ?"
        params = [f"%{search_term}%"]

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)  # type: ignore

        cursor.execute(query, params)
        rows = cursor.fetchall()
        return rows

    def get_movie_details(self, movie_id: int) -> Optional[Tuple]:
        """
        Retrieve detailed information about a movie.

        :param movie_id: The movie ID
        :return: A tuple representing the movie details, or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT movieId, title, genres, year, popularity, imdbId, tmdbId
            FROM movies
            WHERE movieId = ?
            """,
            (movie_id,),
        )
        row = cursor.fetchone()
        return row

    def search_movie_by_title(self, title: str) -> List[Tuple]:
        """
        Search for movies by their title.

        :param title: The title to search for (partial matches are allowed)
        :return: A list of tuples representing the matching movies
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM movies WHERE title LIKE ?", ("%" + title + "%",))
        rows = cursor.fetchall()
        return rows

    def filter_movies_by_genres(
        self,
        genre: str,
        k: Optional[int] = None,
    ) -> List[Tuple]:
        """
        Filter movies by their genres.

        :param genre: The genre to filter by (partial matches are allowed)
        :param k: The maximum number of movies to retrieve, defaults to None (no limit)
        :return: A list of tuples representing the matching movies
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM movies WHERE genres LIKE ?", ("%" + genre + "%",))
        if k is not None:
            rows = cursor.fetchmany(k)
        else:
            rows = cursor.fetchall()
        return rows

    def filter_movies_by_year(
        self,
        year: int,
        k: Optional[int] = None,
    ) -> List[Tuple]:
        """
        Filter movies by their release year.
        :param year: The release year to filter by
        :param k: The maximum number of movies to retrieve, defaults to None (no limit)
        :return: A list of tuples representing the matching movies
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM movies WHERE year = ?", (year,))
        if k is not None:
            rows = cursor.fetchmany(k)
        else:
            rows = cursor.fetchall()
        return rows

    def filter_movies_by_popularity(
        self,
        min_popularity: float,
        max_popularity: float,
        k: Optional[int] = None,
    ) -> List[Tuple]:
        """
        Filter movies by their popularity score.

        :param min_popularity: The minimum popularity score
        :param max_popularity: The maximum popularity score
        :param k: The maximum number of movies to retrieve, defaults to None (no limit)
        :return: A list of tuples representing the matching movies
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM movies WHERE popularity BETWEEN ? AND ?",
            (min_popularity, max_popularity),
        )
        if k is not None:
            rows = cursor.fetchmany(k)
        else:
            rows = cursor.fetchall()
        return rows

    def get_top_popularity_movies(self, k: Optional[int] = None) -> List[Tuple]:
        """
        Retrieve the top movies sorted by popularity score.

        :param k: The maximum number of movies to retrieve, defaults to None (no limit)
        :return: A list of tuples representing the top movies by popularity
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM movies ORDER BY popularity DESC")
        if k is not None:
            rows = cursor.fetchmany(k)
        else:
            rows = cursor.fetchall()
        return rows

    def get_top_latest_movies(self, k: Optional[int] = None) -> List[Tuple]:
        """
        Retrieve the latest movies sorted by release year.

        :param k: The maximum number of movies to retrieve, defaults to None (no limit)
        :return: A list of tuples representing the latest movies
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM movies ORDER BY year DESC")
        if k is not None:
            rows = cursor.fetchmany(k)
        else:
            rows = cursor.fetchall()
        return rows

    def close_connection(self) -> None:
        """
        Close the connection to the database.
        """
        self.conn.close()

    @staticmethod
    def dump_from_df(df, table_name: str, db_file: str) -> None:
        """
        Dump data from a pandas DataFrame into a table in the database.

        :param df: The pandas DataFrame containing the data
        :param table_name: The name of the table to create or replace
        :param db_file: The file path of the SQLite database
        """
        conn = sqlite3.connect(db_file)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.close()
