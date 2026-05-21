import faiss
import numpy as np
import os

EMBEDDINGS_PATH = "Bookworm/data/embeddings/embeddings.npy"
INDEX_PATH = "Bookworm/data/index/faiss.index"


def faiss_indexer():
   print("Loading embeddings...")
   embeddings = np.load(EMBEDDINGS_PATH)

   print(f"Embeddings shape: {embeddings.shape}")

   embeddings = embeddings.astype(np.float32)

   dimension = embeddings.shape[1]

   print("Creating FAISS index...")
   index = faiss.IndexFlatIP(dimension)

   print("Adding embeddings to index...")
   index.add(embeddings)

   os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

   print("Saving index...")
   faiss.write_index(index, INDEX_PATH)

   print(f"Index saved to {INDEX_PATH}")


if __name__ == "__main__":
   faiss_indexer()