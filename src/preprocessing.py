import sys
import os

# Add parent directory (Capstone root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from config import SEED

def train_test_split(df, random_state=SEED):
    # Group by movie_id and sample one rating per movie for the test set
    test_df = df.groupby("movie_id").sample(n=1, random_state=random_state)
    
    # The training set is the original dataframe minus the test set
    train_df = df.drop(test_df.index)
    
    return train_df, test_df


