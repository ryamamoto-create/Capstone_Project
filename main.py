import random 
import pandas as pd

from preprocessing import train_test 
from config import RATING_PARQUET_PATH, SEED
from baseline import *  
from svd import svd_model, save_model
from evaluation import rmse_function

# Run all the baseline models and return their results
def run_baselines(train_df, test_df):
    results = []

    preds, rmse = global_mean_model(train_df, test_df)
    results.append(format_result("global_mean", preds, rmse))

    preds, rmse = movie_mean_model(train_df, test_df)
    results.append(format_result("movie_mean", preds, rmse))

    preds, rmse = user_movie_bias_model(train_df, test_df)
    results.append(format_result("bias_model", preds, rmse))

    return results

# Runs the SVD model and saves
def run_svd(train_df, test_df):
    preds, rmse, model = svd_model(train_df, test_df)
    save_model(model, "models/svd_model.pkl")
    return [format_result("svd", preds, rmse, model=model)]

# Formats the results into a consistent structure for easy comparison and reporting
def format_result(name, preds, rmse, model=None, params=None):
    return {
        "name": name,
        "preds": preds,
        "rmse": rmse,
        "model": model,
        "params": params
    }

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

    # 4. Print comparison
    print_results(results)

if __name__ == "__main__":
    main()