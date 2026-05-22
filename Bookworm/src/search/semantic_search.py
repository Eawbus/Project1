import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from torch import norm
from torch import norm


class SemanticRetriever:
   def __init__(self, dataframe, model_path, index_path):
      self.df = dataframe

      self.model = SentenceTransformer(model_path)

      self.index = faiss.read_index(index_path)

   def normalize(self, vector):
      norm = np.linalg.norm(vector)

      if norm == 0:
         return vector

      return vector / norm

   def retrieve(self, query, k=50):
      query_embedding = self.model.encode([query])

      query_embedding = self.normalize(query_embedding)

      query_embedding = query_embedding.astype(np.float32)

      scores, indices = self.index.search(query_embedding, k)

      results = self.df.iloc[indices[0]].copy()

      results["semantic_score"] = scores[0]

      return results