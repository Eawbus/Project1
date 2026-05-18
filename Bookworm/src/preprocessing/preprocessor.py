import pandas as pd
import numpy as np
import re
import os

INPUT_PATH = "data/raw/GoodReadsDataset.csv"
OUTPUT_PATH = "data/processed/cleaned_GoodReadsDataset.csv"

def clean_text(text):
    text = str(text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def clean_genres(g):
    if pd.isna(g):
        return ""

    g = str(g)
    g = g.replace("[", "").replace("]", "").replace("'", "")
    genres = [x.strip().title() for x in g.split(",") if x.strip() != ""]
    return ", ".join(genres)

def preprocess():
    print("Loading dataset...")
    df = pd.read_csv(INPUT_PATH)

    print(f"Original shape: {df.shape}")

    columns_map = {
    "title": "title",
    "author": "author",
    "description": "description",
    "genres": "genres",
    "rating": "rating",
    "numRatings": "num_ratings",
    "firstPublishDate": "year"
}

    available_cols = {k: v for k, v in columns_map.items() if k in df.columns}
    df = df[list(available_cols.keys())]
    df = df.rename(columns=available_cols)

    print(f"After column selection: {df.shape}")

    df = df.dropna(subset=["title", "description"])
    print(f"After dropping missing title/description: {df.shape}")

    df = df[df["description"].str.len() > 50]
    print(f"After removing short descriptions: {df.shape}")

    print("Cleaning text...")
    df["title"] = df["title"].apply(clean_text)
    df["description"] = df["description"].apply(clean_text)

    if "author" in df.columns:
        df["author"] = df["author"].apply(clean_text)

    if "genres" in df.columns:
        print("Cleaning genres...")
        df["genres"] = df["genres"].apply(clean_genres)
    else:
        df["genres"] = ""

    print("Cleaning numeric fields...")
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    if "num_ratings" in df.columns:
        df["num_ratings"] = pd.to_numeric(df["num_ratings"], errors="coerce")

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    df = df.dropna(subset=["rating", "num_ratings"])

    print("Removing duplicates...")
    df = df.drop_duplicates(subset=["title", "author"])

    df = df.reset_index(drop=True)
    df["book_id"] = df.index

    print(f"Final shape: {df.shape}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Cleaned data saved to {OUTPUT_PATH}")

# if __name__ == "__main__":
#     preprocess()