import numpy as np
import pandas as pd
from config import RATING_PATH, MOVIE_PATH
# import time

'''
Due to the structure of the ratings file, we need to read it line by line and keep track of the current movie_id until we encounter the next one. 
We will store the user_id, movie_id, rating, and date in separate lists and then create a DataFrame from those lists. 
This approach allows us to efficiently parse the file without loading it entirely into memory at once, which is crucial given its size.
'''
def load_ratings():
    # Columns: user_id, movie_id, rating, date
    user_ids = []
    movie_ids = []
    ratings = []
    dates = []
    # Reading the file
    with open(RATING_PATH, encoding="utf-8") as f:
        # The file is structured such that movie_id lines end with ":\n", and the subsequent lines contain user ratings for that movie until the next movie_id line.
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
    # Convert extracted data into a DataFrame
    df = pd.DataFrame({
        "user_id": user_ids,
        "movie_id": movie_ids,
        "rating": ratings,
        "date": dates
    }).astype({"user_id": "int32", "movie_id": "int16", "rating": "int8", "date": str}) # Using smaller data types to save memory (1.86GB -> 405MB)

    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    return df

'''
The movies file is a CSV with the format: movie_id, year, title. However, some titles may contain commas, which can complicate parsing.
To handle this, we will read the file line by line, split on the first two commas to extract the movie_id and year, and then join the 
remaining parts to reconstruct the title.
'''
def load_movies():
    # Columns: movie_id, year, title
    movies = []
    # Reading the file
    with open(MOVIE_PATH, encoding="utf-8") as f:
        for line in f:
            # Split the line on commas, but only the first two commas to separate movie_id and year from the title
            parts = line.strip().split(",")

            # Extract movie_id
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

            # Append the parsed data to the movies list
            movies.append((movie_id, year, title))
    
    # Convert the list of movies into a DataFrame
    # Use appropriate data types for memory efficiency (movie_id as int, year as Int64 to allow for NaN, title as string)
    return pd.DataFrame(movies, columns=["movie_id", "year", "title"]).astype({"movie_id": int, "year": "Int64", "title": str})

# saves parquet to data/processed/{filename}.parquet
def save_processed(df, filename):
    # Save the processed DataFrame as a parquet file for efficient storage and faster loading in future analyses.
    df.to_parquet(f"data/processed/{filename}.parquet", index=False)

# Used for testing the load_movies function and analyzing the resulting dataframept
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