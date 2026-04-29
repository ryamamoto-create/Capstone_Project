import sys
import os

# Add parent directory (Capstone root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from config import SEED, RATING_PARQUET_PATH

df = pd.read_parquet(RATING_PARQUET_PATH)
test_df = df.groupby("movie_id").sample(n=1, random_state=SEED)
train_df = df.drop(test_df.index)

