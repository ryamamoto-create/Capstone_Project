import numpy as np
from evaluation import rmse_function

# Baseline model 1: Predict the global mean rating for all movies
def global_mean_model(train_df, test_df):
    # Find the global mean rating from the training data
    global_mean = train_df["rating"].mean()

    # Predict the global mean for all test samples
    preds = np.full(len(test_df), global_mean)

    # Calculate RMSE between the predictions and the actual ratings in the test set
    rmse = rmse_function(test_df["rating"], preds)

    # Return the RMSE value
    return preds, rmse

# Baseline model 2: Predict the mean rating for each movie based on the training data
def movie_mean_model(train_df, test_df):
    # Calculate the mean rating for each movie in the training set
    movie_means = train_df.groupby("movie_id")["rating"].mean()

    # Predict the mean rating for each movie in the test set
    preds = test_df["movie_id"].map(movie_means).fillna(train_df["rating"].mean())

    # Calculate RMSE between the predictions and the actual ratings in the test set
    rmse = rmse_function(test_df["rating"], preds)

    # Return the RMSE value
    return preds, rmse

'''
Baseline model 3: We know some movies are better than others and these will have higher average ratings.
We also know that some users tend to give higher ratings than others. We account for these effects in this
model by predicting that the rating is aproximately the global mean plus a movie bias plus a user bias.
The model is regularized to prevent overfitting when calculating the biases, and the predictions are made 
in a vectorized way for better performance.
'''
def user_movie_bias_model(train_df, test_df):
    # Set a regularization parameter to prevent overfitting when calculating biases
    lambda_reg = 10

    # Calculate the global mean rating
    global_mean = train_df["rating"].mean()

    # Calculate the sum and count of ratings for each movie in the training set
    movie_stats = train_df.groupby("movie_id")["rating"].agg(["sum", "count"])

    # Calculate the movie bias (average rating for each movie minus the global mean, regularized to prevent overfitting)
    movie_bias = (movie_stats["sum"] - movie_stats["count"] * global_mean) / (movie_stats["count"] + lambda_reg)

    # Calculate the sum and count of adjusted ratings for each user in the training set, where the adjusted rating is the original rating minus the global mean and movie bias
    user_stats = (
    train_df.assign(
        adjusted = train_df["rating"]
        - train_df["movie_id"].map(movie_bias)
        - global_mean
    )
    .groupby("user_id")["adjusted"]
    .agg(["sum", "count"])
    )

    # Calculate the user bias (average rating for each user minus the global mean and movie bias)
    user_bias = user_stats["sum"] / (user_stats["count"] + lambda_reg)

    # Predict the rating for each test sample as the global mean plus the movie bias and user bias
    # Updated to be vectorized for better performance
    movie_b = test_df["movie_id"].map(movie_bias).fillna(0)
    user_b = test_df["user_id"].map(user_bias).fillna(0)

    preds = global_mean + movie_b + user_b

    # Calculate RMSE between the predictions and the actual ratings in the test set
    rmse = rmse_function(test_df["rating"], preds)

    return preds, rmse