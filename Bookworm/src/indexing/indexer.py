"""
Indexer module for building and searching FAISS indices
"""
import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Dict

# Relative import from sibling module
from src.configuration.config import config


class Indexer:
    """
    Handles FAISS index building and searching
    """
    
    def __init__(self, 
                 index_type: str = None,
                 config=config):
        self.config = config
        self.index_type = index_type or self.config.FAISS_INDEX_TYPE
        self.index = None
        self.documents = None  # Store document IDs or metadata
        self.embedding_dim = None
    
    def build_index(self, embeddings: np.ndarray, documents: Optional[List] = None):
        """
        Build a FAISS index from embeddings
        
        Args:
            embeddings: numpy array of shape (n, d)
            documents: Optional list of document IDs or metadata
            
        Returns:
            The built FAISS index
        """
        if embeddings.shape[0] == 0:
            raise ValueError("Cannot build index with empty embeddings")
        
        self.embedding_dim = embeddings.shape[1]
        self.documents = documents
        
        # Ensure float32 and normalized
        embeddings = embeddings.astype(np.float32)
        
        print(f"Building FAISS index ({self.index_type})...")
        print(f"Number of vectors: {embeddings.shape[0]}")
        print(f"Vector dimension: {embeddings.shape[1]}")
        
        # Create index based on type
        if self.index_type == "Flat":
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            
        elif self.index_type.startswith("IVF"):
            nlist = int(self.index_type.replace("IVF", ""))
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            print(f"Training IVF index with {nlist} centroids...")
            self.index.train(embeddings)
            
        elif self.index_type.startswith("HNSW"):
            hnsw_m = int(self.index_type.replace("HNSW", ""))
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, hnsw_m)
            
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # Add vectors to index
        self.index.add(embeddings)
        print(f"Index built. Contains {self.index.ntotal} vectors")
        
        return self.index
    
    def save_index(self, index_filepath: Path, documents_filepath: Optional[Path] = None):
        """
        Save FAISS index to disk
        
        Args:
            index_filepath: Path to save the FAISS index
            documents_filepath: Path to save the documents (optional)
        """
        if self.index is None:
            raise ValueError("No index to save")
        
        # Save FAISS index
        faiss.write_index(self.index, str(index_filepath))
        print(f"Index saved to {index_filepath}")
        
        # Save documents if provided
        if self.documents is not None and documents_filepath is not None:
            with open(documents_filepath, "wb") as f:
                pickle.dump(self.documents, f)
            print(f"Documents saved to {documents_filepath}")
    
    def load_index(self, index_filepath: Path, documents_filepath: Optional[Path] = None):
        """
        Load FAISS index from disk
        
        Args:
            index_filepath: Path to the FAISS index file
            documents_filepath: Path to the documents file (optional)
        """
        self.index = faiss.read_index(str(index_filepath))
        self.embedding_dim = self.index.d
        print(f"Index loaded. Contains {self.index.ntotal} vectors")
        
        # Load documents if available
        if documents_filepath is not None and documents_filepath.exists():
            with open(documents_filepath, "rb") as f:
                self.documents = pickle.load(f)
            print(f"Loaded {len(self.documents)} documents")
    
    def search(self, query_embeddings: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the index with query embeddings
        
        Args:
            query_embeddings: numpy array of shape (n_queries, d)
            k: Number of results to return per query
            
        Returns:
            scores: numpy array of shape (n_queries, k)
            indices: numpy array of shape (n_queries, k)
        """
        if self.index is None:
            raise ValueError("Index not loaded or built")
        
        # Ensure float32
        query_embeddings = query_embeddings.astype(np.float32)
        
        # Handle single query
        if len(query_embeddings.shape) == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        
        # Search
        scores, indices = self.index.search(query_embeddings, k)
        
        return scores, indices
    
    def search_single(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search with a single query
        
        Args:
            query_embedding: numpy array of shape (d,)
            k: Number of results to return
            
        Returns:
            scores: numpy array of shape (k,)
            indices: numpy array of shape (k,)
        """
        scores, indices = self.search(query_embedding, k)
        return scores[0], indices[0]
    
    def get_results_with_metadata(self, 
                                  scores: np.ndarray, 
                                  indices: np.ndarray, 
                                  metadata: List[Dict]) -> List[List[Dict]]:
        """
        Format search results with metadata
        
        Args:
            scores: scores from search
            indices: indices from search
            metadata: list of metadata objects (e.g., DataFrame rows)
            
        Returns:
            List of results per query, each result is a dict with score and metadata
        """
        results = []
        
        for query_idx in range(len(scores)):
            query_results = []
            for i, idx in enumerate(indices[query_idx]):
                if idx >= 0 and idx < len(metadata):
                    result = {
                        "rank": i + 1,
                        "score": float(scores[query_idx][i]),
                        "index": int(idx)
                    }
                    # Add metadata
                    if isinstance(metadata[idx], dict):
                        result.update(metadata[idx])
                    query_results.append(result)
            results.append(query_results)
        
        return results
    
    @property
    def n_total(self) -> int:
        """Return number of vectors in the index"""
        if self.index is None:
            return 0
        return self.index.ntotal