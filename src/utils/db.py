import sqlite3

from typing import List


class DatabaseRepository:
    def __init__(self, db_file):
        self.conn = sqlite3.connect(db_file)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_interacted (
                interactionId INTEGER PRIMARY KEY AUTOINCREMENT,
                userId INTEGER,
                movieId INTEGER,
                interaction INTEGER,
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
                imdbId TEXT,
                tmdbId TEXT
            )
        ''')

        self.conn.commit()

    def create_interaction(self, userId, movieId, interaction):
        timestamp = datetime.utcnow().isoformat()
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO user_interacted (userId, movieId, interaction, timestamp) VALUES (?, ?, ?, ?)",
            (userId, movieId, interaction, timestamp)
        )
        self.conn.commit()

    def create_user(self, name, password):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO users (name, password) VALUES (?, ?)",
            (name, password)
        )
        self.conn.commit()

    def create_movie(self, title, genres, imdbId, tmdbId):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO movies (title, genres, imdbId, tmdbId) VALUES (?, ?, ?, ?)",
            (title, genres, imdbId, tmdbId)
        )
        self.conn.commit()

    def get_interactions(self, cols='*'):
        if isinstance(cols, List):
            cols = ', '.join(cols)
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT {cols} FROM user_interacted")
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

    def update_interaction(self, interactionId, newinteraction):
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE user_interacted SET interaction = ? WHERE interactionId = ?",
            (newinteraction, interactionId)
        )
        self.conn.commit()

    def update_user(self, userId, newName, newPassword):
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE users SET name = ?, password = ? WHERE userId = ?",
            (newName, newPassword, userId)
        )
        self.conn.commit()

    def update_movie(self, movieId, newTitle, newGenres, newImdbId, newTmdbId):
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE movies SET title = ?, genres = ?, imdbId = ?, tmdbId = ? WHERE movieId = ?",
            (newTitle, newGenres, newImdbId, newTmdbId, movieId)
        )
        self.conn.commit()

    def delete_interaction(self, interactionId):
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM user_interacted WHERE interactionId = ?", (interactionId,))
        self.conn.commit()

    def delete_user(self, userId):
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM users WHERE userId = ?", (userId,))
        self.conn.commit()

    def delete_movie(self, movieId):
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM movies WHERE movieId = ?", (movieId,))
        self.conn.commit()

    def close_connection(self):
        self.conn.close()

    @staticmethod
    def dump_from_df(datadf, table_name, db_file):
        conn = sqlite3.connect(db_file)
        datadf.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()
