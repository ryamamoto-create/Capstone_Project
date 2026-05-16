import pandas as pd
import umap.umap_ as umap
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from parsing import load_movies
from config import SEED

# Load embeddings
df = pd.read_parquet("predictions/movie_embeddings.parquet")

movie_ids = df["movie_id"]

# Load movies
movies_df = load_movies()

# Join data
df = df.merge(movies_df, on="movie_id")

factor_cols = [col for col in df.columns if col.startswith("factor_")]

X = df[factor_cols].values

# Run UMAP
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric="cosine",
    random_state=SEED
)

embedding = reducer.fit_transform(X)

df["x"] = embedding[:, 0]
df["y"] = embedding[:, 1]

# Plot
plt.figure(figsize=(12, 10))

plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    s=5,
    alpha=0.5
)

highlight_movies = [
    "Mean Girls",
    "The Godfather, Part II",
    "Back to the Future Part II",
    "Back to the Future Part III",
    "War Games",
    "Scooby-Doo's Spookiest Tales",
    "Shrek 2",
    "Finding Nemo (Widescreen)",
    "Star Trek: The Original Series: Vols. 1-15",
    "Sailor Moon Super S: Black Dream Hole",
    "Dragon Ball GT",
    "The Lord of the Rings",
    "Led Zeppelin: Inside Led Zeppelin 1968-1972",
    "George Carlin: Complaints and Grievances",
    "Texas Chainsaw Massacre: The Next Generation",
    "Scream",
    "Home Alone 2: Lost in New York",
    "Elf",
    "The Wire: Season 1",
    "Pride and Prejudice",
    "Tremors"
]

highlight_df = df[df["title"].isin(highlight_movies)]

for _, row in highlight_df.iterrows():
    plt.scatter(
        row["x"],
        row["y"],
        s=40
    )

    plt.text(
        row["x"],
        row["y"],
        row["title"],
        fontsize=9,
        color = "black",
        path_effects=[PathEffects.withStroke(linewidth=3, foreground="white")]
    )

plt.title("UMAP Projection of Movie Embeddings")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")

plt.savefig(
    "img/umap_movie_embeddings.png",
    dpi=300,
    bbox_inches="tight"
)