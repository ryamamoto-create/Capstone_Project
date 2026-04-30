import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD
from evaluation import rmse_function 
import pickle
from config import SEED

'''
This file will train the SVD model using the Surprise library. The SVD model is a matrix 
factorization technique that can be used for collaborative filtering in recommendation systems. 
The function will take in the training and testing dataframes, convert them into the format 
required by the Surprise library, and then train the SVD model on the training data. 
Finally, it will evaluate the model on the testing data and return the RMSE value.
'''
def train_svd(train_df):
    # 1. Define the Reader - ensure the scale matches your actual data
    reader = Reader(rating_scale=(1, 5))
    
    # 2. Load the training data
    # Surprise expects exactly [user, item, rating] columns in this order
    data = Dataset.load_from_df(train_df[["user_id", "movie_id", "rating"]], reader)
    trainset = data.build_full_trainset()
    

    # 3. Initialize and train SVD
    model = SVD(n_factors=50, reg_all=0.05, n_epochs=20, random_state=SEED, verbose=True)
    model.fit(trainset)

    return model

def predict_svd(model, train_df, test_df):
    testset_tuples = list(
        test_df[["user_id", "movie_id", "rating"]]
        .itertuples(index=False, name=None)
    )

    predictions = model.test(testset_tuples)

    pred_df = pd.DataFrame(
    [(int(pred.uid), int(pred.iid), pred.est) for pred in predictions],
    columns=["user_id", "movie_id", "pred"]
    )

    merged = test_df.merge(pred_df, on=["user_id", "movie_id"], how="left")

    preds = merged["pred"].fillna(train_df["rating"].mean()).values
    return preds

def svd_model(train_df, test_df):
    model = train_svd(train_df)
    preds = predict_svd(model, train_df, test_df)
    rmse = rmse_function(test_df["rating"], preds)
    return preds, rmse, model

def save_model(model, filename):
    with open(filename, "wb") as f:
        pickle.dump(model, f)