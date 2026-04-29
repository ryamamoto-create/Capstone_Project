import pandas as pd
from config import SEED

'''
Creates a train-test split of the ratings dataframe, ensuring that each movie in the test set has exactly one rating in the training set. 
This is done by grouping the dataframe by 'movie_id' and sampling one rating per movie for the test set, while the remaining ratings form the training set.
'''
def train_test(df, random_state=SEED):
    # Group by movie_id and sample one rating per movie for the test set
    test_df = df.groupby("movie_id").sample(n=1, random_state=random_state)
    
    # The training set is the original dataframe minus the test set
    train_df = df.drop(test_df.index)
    
    return train_df, test_df
