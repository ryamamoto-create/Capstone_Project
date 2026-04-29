import numpy as np
import pandas as pd
from preprocessing import train_test
from evaluation import rmse_function
from config import RATING_PARQUET_PATH

# Baseline model 1: Predict the global mean rating for all movies
def global_mean_model(train_df, test_df):
    # Find the global mean rating from the training data
    global_mean = train_df["rating"].mean()

    # Predict the global mean for all test samples
    preds = np.full(len(test_df), global_mean)

    # Calculate RMSE between the predictions and the actual ratings in the test set
    rmse = rmse_function(test_df["rating"], preds)

    # Return the RMSE value
    return rmse

# Baseline model 2: Predict the mean rating for each movie based on the training data
def movie_mean_model(train_df, test_df):
    # Calculate the mean rating for each movie in the training set
    movie_means = train_df.groupby("movie_id")["rating"].mean()

    # Predict the mean rating for each movie in the test set
    preds = test_df["movie_id"].map(movie_means).fillna(train_df["rating"].mean())

    # Calculate RMSE between the predictions and the actual ratings in the test set
    rmse = rmse_function(test_df["rating"], preds)

    # Return the RMSE value
    return rmse

'''
Baseline model 3: We know some movies are better than others and these will have higher average ratings.
We also know that some users tend to give higher ratings than others. We account for these effects in this
model by predicting that the rating is aproximately the global mean plus a movie bias (the average rating 
for that movie minus the global mean) plus a user bias (the average rating for that user minus the global mean).
'''
def user_movie_bias_model(train_df, test_df):
    # Calculate the global mean rating
    global_mean = train_df["rating"].mean()

    # Calculate the movie bias (average rating for each movie minus the global mean)
    movie_bias = train_df.groupby("movie_id")["rating"].mean() - global_mean

    # Calculate the user bias (average rating for each user minus the global mean and movie bias)
    user_bias = (
    train_df.assign(
        adjusted = train_df["rating"]
        - train_df["movie_id"].map(movie_bias)
        - global_mean
    )
    .groupby("user_id")["adjusted"]
    .mean()
    )

    # Predict the rating for each test sample as the global mean plus the movie bias and user bias
    # Updated to be vectorized for better performance
    preds = (
    global_mean
    + test_df["movie_id"].map(movie_bias)
    + test_df["user_id"].map(user_bias)
    )

    preds = preds.fillna(global_mean)

    # Calculate RMSE between the predictions and the actual ratings in the test set
    rmse = rmse_function(test_df["rating"], preds)

    return rmse