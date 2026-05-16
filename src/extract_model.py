import pickle
import pandas as pd

# Load model
with open("models/svd_model.pkl", "rb") as f:
    model = pickle.load(f)

rows = []

# Iterate through raw movie IDs
for raw_movie_id in model.trainset._raw2inner_id_items.keys():
    inner_id = model.trainset.to_inner_iid(raw_movie_id)

    embedding = model.qi[inner_id]

    row = {
        "movie_id": raw_movie_id
    }

    for i, value in enumerate(embedding):
        row[f"factor_{i}"] = value

    rows.append(row)

df = pd.DataFrame(rows)

df.to_parquet(
    "predictions/movie_embeddings.parquet",
    index=False
)

print(df.head())
print("Saved embeddings.")