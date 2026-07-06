"""
Configuration module for the Bookworm project
"""
import os
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Config:
   """Central configuration for the entire pipeline"""
    
   # Project paths
   BASE_DIR: Path = Path(__file__).parent.parent.parent
   DATA_DIR: Path = BASE_DIR / "data"
   MODELS_DIR: Path = BASE_DIR / "models"
   SRC_DIR: Path = BASE_DIR / "src"
   BENCHMARK_DIR: Path = BASE_DIR / "benchmarks"
    
   # Data paths
   RAW_DATA: Path = DATA_DIR / "raw" / "GoodReadsDataset.csv"
   PROCESSED_DATA: Path = DATA_DIR / "processed" / "cleaned_GoodReadsDataset.csv"
   EMBEDDINGS_FILE: Path = DATA_DIR / "embeddings" / "embeddings.npy"
   INDEX_FILE: Path = DATA_DIR / "index" / "faiss.index"
   DOCUMENTS_FILE: Path = DATA_DIR / "index" / "documents.pkl"
   BENCHMARK_QUERIES_CSV: Path = DATA_DIR / "benchmarks" / "benchmark_queries.csv"
   BENCHMARK_RESULTS_CSV: Path = DATA_DIR / "benchmarks" / "relevance_judgments.csv"
    
   # Model settings
   MODEL_NAME: str = "all-MiniLM-L6-v2"
   MODEL_PATH: Path = MODELS_DIR / "all-MiniLM-L6-v2"
    
   # Processing settings
   BATCH_SIZE: int = 64
   MAX_SEQ_LENGTH: int = 256
    
   # FAISS settings
   FAISS_INDEX_TYPE: str = "IVF100"  # "Flat", "IVF{n -> nlist}"
    
   # Column mapping for the dataset
   COLUMN_MAP: dict = None
    
   def __post_init__(self):
      """Initialize derived attributes"""
      if self.COLUMN_MAP is None:
         self.COLUMN_MAP = {
            "title": "title",
            "author": "author",
            "description": "description",
            "genres": "genres",
            "rating": "rating",
            "numRatings": "num_ratings",
            "firstPublishDate": "year"
         }
    
   def ensure_dirs(self):
      """Create all necessary directories"""
      directories = [
         self.DATA_DIR / "raw",
         self.DATA_DIR / "processed",
         self.DATA_DIR / "embeddings",
         self.DATA_DIR / "index",
         self.MODELS_DIR
      ]
      for path in directories:
         path.mkdir(parents=True, exist_ok=True)
      return self
    
   @classmethod
   def get_default(cls) -> 'Config':
      """Get default configuration instance"""
      return cls()


# Global configuration instance
config = Config()