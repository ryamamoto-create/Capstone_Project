import sys
import os

# Add parent directory (Capstone root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from config import RATING_PATH, MOVIE_PATH
import time

# returns dataframe
def load_ratings():
    user_ids = []
    movie_ids = []
    ratings = []
    dates = []
    with open(RATING_PATH, encoding="utf-8") as f:
        for line in f:
            if line.endswith(":\n"):
                # Extract movie_id from the line, removing the trailing ":\n"
                movie_id = int(line[:-2])
            else:
                # Process rating data
                user, rating, date = line.strip().split(",")
                user_ids.append(int(user))
                movie_ids.append(movie_id)
                ratings.append(float(rating))
                dates.append(date)
    df = pd.DataFrame({
        "user_id": user_ids,
        "movie_id": movie_ids,
        "rating": ratings,
        "date": dates
    }).astype({"user_id": "int32", "movie_id": "int16", "rating": "int8", "date": str})

    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    return df

# returns dataframe
def load_movies():
    movies = []
    with open(MOVIE_PATH, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")

            movie_id = int(parts[0])
        
            # Handle missing year safely
            try:
                year = int(parts[1])
            except:
                year = None

            # Everything after column 2 is part of the title
            title = ",".join(parts[2:]).strip()

            # Remove trailing commas if present
            title = title.rstrip(",")

            movies.append((movie_id, year, title))
    return pd.DataFrame(movies, columns=["movie_id", "year", "title"]).astype({"movie_id": int, "year": "Int64", "title": str})

# saves parquet to data/processed/{filename}.parquet
def save_processed(df, filename):
    df.to_parquet(f"data/processed/{filename}.parquet", index=False)

# save_processed(load_ratings(), "ratings")

'''
# Testing the load_ratings function and analyzing the resulting dataframe
time_start = time.time()
df = load_ratings()
time_end = time.time()
# Time of the process
print(f"Time taken to load ratings: {time_end - time_start:.2f} seconds")
# What the dataframe looks like
print(df.head())
# Memory usage of the dataframe
print(f"Memory usage of ratings dataframe: {df.memory_usage(deep=True).sum()} bytes")
# shape of the dataframe
print(f"Shape of ratings dataframe: {df.shape}")
# data types of the dataframe
print(f"Data types of ratings dataframe:\n{df.dtypes}")
# Check for missing values
print(f"Missing values in ratings dataframe:\n{df.isnull().sum()}")
# Check for duplicates
print(f"Number of duplicate rows in ratings dataframe: {df.duplicated().sum()}")
# Check for unique users and movies
print(f"Number of unique users: {df['user_id'].nunique()}")
print(f"Number of unique movies: {df['movie_id'].nunique()}")
'''