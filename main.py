import os

import random
import pandas as pd

from preprocessing import train_test 
from config import RATING_PARQUET_PATH, SEED, SVD_PREDICTIONS_PATH, KNN_PREDICTIONS_PATH, TIME_BIAS_MODEL_PREDICTIONS_PATH
from baseline import global_mean_model, movie_mean_model, user_movie_bias_model, user_movie_time_bias_model
from svd import svd_model, save_model
from knn import knn_model, rmse_function
from evaluation import format_result

# Run all the baseline models and return their results
def run_baselines(train_df, test_df):
    results = []

    preds, rmse = global_mean_model(train_df, test_df)
    results.append(format_result("global_mean", preds, rmse))

    # Save predictions for analysis
    pd.DataFrame({
    "user_id": test_df["user_id"],
    "movie_id": test_df["movie_id"],
    "actual": test_df["rating"],
    "pred": preds
    }).to_parquet("predictions/global_mean_preds.parquet", index=False)

    preds, rmse = movie_mean_model(train_df, test_df)
    results.append(format_result("movie_mean", preds, rmse))

    # Save predictions for analysis
    pd.DataFrame({
    "user_id": test_df["user_id"],
    "movie_id": test_df["movie_id"],
    "actual": test_df["rating"],
    "pred": preds
    }).to_parquet("predictions/movie_mean_preds.parquet", index=False)

    preds, rmse = user_movie_bias_model(train_df, test_df)
    results.append(format_result("bias_model", preds, rmse))
    # Save predictions for analysis
    pd.DataFrame({
    "user_id": test_df["user_id"],
    "movie_id": test_df["movie_id"],
    "actual": test_df["rating"],
    "pred": preds
    }).to_parquet("predictions/bias_model_preds.parquet", index=False)

    preds, rmse = user_movie_time_bias_model(train_df, test_df)
    results.append(format_result("time_bias_model", preds, rmse))
    # Save predictions for analysis
    pd.DataFrame({
    "user_id": test_df["user_id"],
    "movie_id": test_df["movie_id"],
    "actual": test_df["rating"],
    "pred": preds
    }).to_parquet("predictions/time_bias_model_preds.parquet", index=False)
    
    return results

# Runs the SVD model and saves
def run_svd(train_df, test_df):
    preds, rmse, model = svd_model(train_df, test_df)
    save_model(model, "models/svd_model.pkl")

    # Save predictions for analysis
    pd.DataFrame({
    "user_id": test_df["user_id"],
    "movie_id": test_df["movie_id"],
    "actual": test_df["rating"],
    "pred": preds
    }).to_parquet("predictions/svd_preds.parquet", index=False)

    return [format_result("svd", preds, rmse, model=model)]

def run_knn(train_df, test_df):
    preds, rmse, model = knn_model(train_df, test_df)
    save_model(model, "models/knn_model.pkl")

    # Save predictions for analysis
    pd.DataFrame({
    "user_id": test_df["user_id"],
    "movie_id": test_df["movie_id"],
    "actual": test_df["rating"],
    "pred": preds
    }).to_parquet("predictions/knn_preds.parquet", index=False)
    return [format_result("knn", preds, rmse, model=model)]

# Prints the model and RMSE ordered by RMSE
def print_results(results):
    print("\n=== MODEL COMPARISON ===")
    for r in sorted(results, key=lambda x: x["rmse"]):
        print(f"{r['name']:20s} RMSE: {r['rmse']:.4f}")


'''
1. Load data
2. Train/test split
3. Run baseline models
4. Run advanced models (SVD, time-aware)
5. Collect results
6. Compare results in table
7. (Optional) build ensemble
8. Print + save report artifacts
'''
def main():
    # Create directories for models and predictions if they don't exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("predictions", exist_ok=True)

    # Set a random seed for reproducibility
    random.seed(SEED)

    # 1. Load data
    df = pd.read_parquet(RATING_PARQUET_PATH)

    # 2. Split
    train_df, test_df = train_test(df)

    # 3. Run models
    results = []
    results += run_baselines(train_df, test_df)

    # SVD is slow, toggle when needed
    RUN_SVD = False
    if RUN_SVD:
        results += run_svd(train_df, test_df)

    # KNN is also slow, toggle when needed
    RUN_KNN = False
    if RUN_KNN:
        results += run_knn(train_df, test_df)

    # 4. Print comparison
    print_results(results)

    # 5. Build ensemble, weights found from exploration notebook
    # Ensemble weights (SVD, KNN, Time Bias)
    w_svd, w_knn, w_bias = 0.6, 0.3, 0.1
    svd = pd.read_parquet(SVD_PREDICTIONS_PATH)
    knn = pd.read_parquet(KNN_PREDICTIONS_PATH)
    bias = pd.read_parquet(TIME_BIAS_MODEL_PREDICTIONS_PATH)

    df = svd.merge(knn, on=["user_id", "movie_id"], suffixes=("_svd", "_knn"))
    df = df.merge(bias, on=["user_id", "movie_id"])
    df = df.rename(columns={"pred": "pred_bias"})

    df["ensemble"] = (
    w_svd * df["pred_svd"]
    + w_knn * df["pred_knn"]
    + w_bias * df["pred_bias"]
    )

    rmse = rmse_function(df["actual_svd"], df["ensemble"])
    print(f"Ensemble RMSE: {rmse:.4f}")

# Run the main function
if __name__ == "__main__":
    main()