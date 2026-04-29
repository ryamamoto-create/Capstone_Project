from preprocessing import train_test 
from config import RATING_PARQUET_PATH
from baseline import *  
import pandas as pd

# Read the parquet file and split it into training and testing sets
df = pd.read_parquet(RATING_PARQUET_PATH)
train_df, test_df = train_test(df)

# Evaluate the baseline models and print their RMSE values
print("Global Mean Model RMSE: " + str(round(global_mean_model(train_df, test_df), 4)))
print("Movie Mean Model RMSE: " + str(round(movie_mean_model(train_df, test_df), 4)))
print("User-Movie Bias Model RMSE: " + str(round(user_movie_bias_model(train_df, test_df), 4)))