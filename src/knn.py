import numpy as np
from surprise import Dataset, Reader, KNNWithMeans
from evaluation import rmse_function

def train_knn(train_df, k=40):
    reader = Reader(rating_scale=(1, 5))

    data = Dataset.load_from_df(
        train_df[["user_id", "movie_id", "rating"]],
        reader
    )

    trainset = data.build_full_trainset()

    # Item-based KNN is usually better for recommender systems
    sim_options = {
        "name": "cosine",
        "user_based": False  # item-item similarity
    }

    model = KNNWithMeans(
        k=k,
        sim_options=sim_options,
        verbose=True
    )

    model.fit(trainset)
    return model

def predict_knn(model, test_df):
    testset = list(
        test_df[["user_id", "movie_id", "rating"]]
        .itertuples(index=False, name=None)
    )

    preds = model.test(testset)
    preds = np.array([p.est for p in preds])

    return preds

def knn_model(train_df, test_df):
    model = train_knn(train_df)
    preds = predict_knn(model, test_df)

    rmse = rmse_function(test_df["rating"].values, preds)

    return preds, rmse, model