"""
Main pipeline orchestrator for building and searching the book index
Supports both semantic search and filter search (baseline)
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import time

from src.configuration.config import config
from src.preprocessing.preprocessor import Preprocessor
from src.embedding.embedder import Embedder
from src.indexing.indexer import Indexer
from src.filter_search.filter_search import FilterSearch


class Pipeline:
    """
    Main pipeline orchestrator for building and searching the book index
    Supports:
    - Semantic Search (FAISS + Sentence Transformers)
    - Filter Search (Keyword matching baseline)
    """
    
    def __init__(self, config=config):
        self.config = config
        self.preprocessor = Preprocessor(config)
        self.embedder = None
        self.indexer = None
        self.keyword_search_engine = FilterSearch(config)
        self.corpus_df = None
        
        # Ensure directories exist
        self.config.ensure_dirs()
    
    # ============ BUILD PHASE ============
    
    def build(self, force_rebuild: bool = False):
        """
        Build the complete index from raw data
        
        Args:
            force_rebuild: Force rebuild even if files exist
            
        Returns:
            self for method chaining
        """
        print("\n" + "="*60)
        print("BUILDING BOOK SEARCH INDEX")
        print("="*60)
        
        # Check if already built
        if not force_rebuild and self._is_built():
            print("\nIndex already exists. Use force_rebuild=True to rebuild.")
            return self
        
        # Step 1: Load and preprocess corpus
        print("\n[1/5] Loading and preprocessing corpus...")
        self.corpus_df = self._load_and_preprocess()
        
        # Step 2: Build text for embedding
        print("\n[2/5] Building text representations...")
        self.corpus_df["text"] = self.corpus_df.apply(
            self.preprocessor.build_corpus_text, axis=1
        )
        
        # Step 3: Generate embeddings
        print("\n[3/5] Generating embeddings...")
        self.embedder = Embedder(config=self.config)
        embeddings = self.embedder.encode(
            self.corpus_df["text"].tolist(),
            normalize=True,
            show_progress=True
        )
        
        # Save embeddings
        self.embedder.save_embeddings(embeddings, self.config.EMBEDDINGS_FILE)
        
        # Step 4: Build FAISS index
        print("\n[4/5] Building FAISS index...")
        self.indexer = Indexer(config=self.config)
        self.indexer.build_index(embeddings, self.corpus_df["book_id"].tolist())
        self.indexer.save_index(
            self.config.INDEX_FILE,
            self.config.DOCUMENTS_FILE
        )
        
        # Step 5: Build filter search index
        print("\n[5/5] Building filter search index...")
        self.keyword_search_engine.load_corpus(self.corpus_df)
        
        print("\n Build complete!")
        print(f"  - Processed {len(self.corpus_df)} books")
        print(f"  - Embeddings saved to {self.config.EMBEDDINGS_FILE}")
        print(f"  - FAISS index saved to {self.config.INDEX_FILE}")
        
        return self
    
    def _is_built(self) -> bool:
        """Check if the index is already built"""
        return (self.config.PROCESSED_DATA.exists() and 
                self.config.EMBEDDINGS_FILE.exists() and 
                self.config.INDEX_FILE.exists())
    
    def _load_and_preprocess(self) -> pd.DataFrame:
        """Load and preprocess the raw data"""
        # Check if processed file exists
        if self.config.PROCESSED_DATA.exists():
            print(f"Loading preprocessed data from {self.config.PROCESSED_DATA}")
            df = pd.read_csv(self.config.PROCESSED_DATA)
            print(f"Loaded {len(df)} records")
            return df
        
        # Load raw data
        print(f"Loading raw data from {self.config.RAW_DATA}")
        df = pd.read_csv(self.config.RAW_DATA)
        
        # Preprocess
        df = self.preprocessor.preprocess_corpus(df)
        
        # Save processed data
        df.to_csv(self.config.PROCESSED_DATA, index=False)
        print(f"Saved preprocessed data to {self.config.PROCESSED_DATA}")
        
        return df
    
    # ============ LOAD PHASE ============
    
    def load(self):
        """
        Load an existing index for searching
        
        Returns:
            self for method chaining
        """
        print("\n" + "="*60)
        print("LOADING BOOK SEARCH INDEX")
        print("="*60)
        
        # Check if built
        if not self._is_built():
            print("Index not found. Building from scratch...")
            return self.build()
        
        # Load corpus
        self.corpus_df = pd.read_csv(self.config.PROCESSED_DATA)
        print(f"Loaded corpus: {len(self.corpus_df)} books")
        
        # Build text for filter search
        self.corpus_df["text"] = self.corpus_df.apply(
            self.preprocessor.build_corpus_text, axis=1
        )
        
        # Load embedder
        self.embedder = Embedder(config=self.config)
        
        # Load FAISS indexer
        self.indexer = Indexer(config=self.config)
        self.indexer.load_index(
            self.config.INDEX_FILE,
            self.config.DOCUMENTS_FILE
        )
        
        # Load keyword search engine
        self.keyword_search_engine.load_corpus(self.corpus_df)
        
        print("\nLoad complete!")
        return self
    
    # ============ SEARCH METHODS ============
    
    def semantic_search(self, query: str, k: int = 10) -> List[Dict]:
        """
        Semantic search for books matching a query
        
        Args:
            query: Text query
            k: Number of results to return
            
        Returns:
            List of results with book metadata and similarity scores
        """
        if self.embedder is None or self.indexer is None:
            raise ValueError("Pipeline not loaded. Call load() first.")
        
        # Preprocess query
        query_clean = self.preprocessor.preprocess_query(query)
        
        # Generate embedding
        query_embedding = self.embedder.encode_query(query_clean)
        
        # Search
        scores, indices = self.indexer.search(query_embedding, k)
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.corpus_df):
                book = self.corpus_df.iloc[idx]
                results.append({
                    "rank": i + 1,
                    "score": float(scores[0][i]),
                    "book_id": int(book["book_id"]),
                    "title": book.get("title", "Unknown"),
                    "author": book.get("author", "Unknown"),
                    "genres": book.get("genres", ""),
                    "rating": float(book.get("rating", 0)),
                    "num_ratings": int(book.get("num_ratings", 0))
                })
        
        return results
    
    def keyword_search(self, query: str, k: int = 10) -> List[Dict]:
        """
        Keyword-based retrieval baseline.
        """

        return self.keyword_search_engine.search(query, k)
    
    # ============ COMPARISON METHODS ============
    
    def compare_search(self, query: str, k: int = 10) -> Dict:
        """
        Compare semantic search vs filter search for the same query
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            Dictionary containing both search results and comparison metrics
        """
        print(f"\n{'='*60}")
        print(f"SEARCH COMPARISON")
        print(f"Query: '{query}'")
        print(f"{'='*60}")
        
        # Semantic search
        print("\nSemantic Search Results:")
        print("-" * 60)
        semantic_start = time.time()
        semantic_results = self.semantic_search(query, k)
        semantic_time = time.time() - semantic_start
        semantic_ids = {r["book_id"] for r in semantic_results}
        
        # Display semantic results
        for book in semantic_results:
            print(f"{book['rank']}. {book['title']} by {book['author']} (Score: {book['score']:.4f})")
        
        # Filter search
        print("\nKeyword Filter Results:")
        print("-" * 60)
        keyword_start = time.time()
        keyword_results = self.keyword_search(query, k)
        keyword_time = (time.time() - keyword_start)
        keyword_ids = {r["book_id"] for r in keyword_results}
        
        # Display keyword results
        for book in keyword_results:
            print(f"{book['rank']}. {book['title']} by {book['author']} (Score: {book['score']:.4f})")
        
        # Calculate overlap
        overlap = semantic_ids.intersection(keyword_ids)
        
        comparison = {
            "query": query,
            "semantic": {
                "results": semantic_results,
                "time": semantic_time,
                "avg_score": np.mean([r["score"] for r in semantic_results]) if semantic_results else 0
            },
            "keyword": {
                "results": keyword_results,
                "time": keyword_time,
                "avg_score": np.mean([r["score"] for r in keyword_results]) if keyword_results else 0
            },
            "comparison": {
                "overlap_count": len(overlap),
                "overlap_percentage": (len(overlap) / k * 100) if k > 0 else 0,
                "semantic_only": len(semantic_ids - keyword_ids),
                "keyword_only": len(keyword_ids - semantic_ids)
            }
        }
        
        # Print comparison summary
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"Semantic Search Time:   {comparison['semantic']['time']:.4f}s")
        print(f"Keyword Search Time:     {comparison['keyword']['time']:.4f}s")
        print(f"Average Semantic Score: {comparison['semantic']['avg_score']:.4f}")
        print(f"Average Keyword Score:   {comparison['keyword']['avg_score']:.4f}")
        print(f"Overlap:                {comparison['comparison']['overlap_count']} of {k} results")
        print(f"Semantic Only Results:  {comparison['comparison']['semantic_only']}")
        print(f"Keyword Only Results:    {comparison['comparison']['keyword_only']}")
        
        return comparison
    
    def search_with_fallback(self, query: str, k: int = 10) -> List[Dict]:
        """
        Search with fallback: semantic first, then keyword if no results
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of results
        """
        semantic_results = self.semantic_search(query, k)
        
        if not semantic_results:
            print("No semantic results found, falling back to keyword search...")
            return self.keyword_search(query, k)
        
        return semantic_results
    
    def search_combined(self, query: str, k: int = 10) -> List[Dict]:
        """
        Combined search: merge semantic and keyword results with deduplication
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of combined results
        """
        semantic_results = self.semantic_search(query, k)
        keyword_results = self.keyword_search(query, k)
        
        # Combine and deduplicate
        seen_ids = set()
        combined = []
        
        # Add semantic results first
        for r in semantic_results:
            if r["book_id"] not in seen_ids:
                seen_ids.add(r["book_id"])
                combined.append(r)
        
        # Add keyword results
        for r in keyword_results:
            if r["book_id"] not in seen_ids:
                seen_ids.add(r["book_id"])
                combined.append(r)
        
        # Re-rank
        for i, r in enumerate(combined[:k], 1):
            r["rank"] = i
        
        return combined[:k]
    
    def get_book(self, book_id: int) -> Optional[Dict]:
        """
        Get a specific book by ID
        
        Args:
            book_id: Book ID
            
        Returns:
            Book details as dictionary or None if not found
        """
        if self.corpus_df is None:
            raise ValueError("Corpus not loaded. Call load() first.")
        
        book = self.corpus_df[self.corpus_df["book_id"] == book_id]
        if len(book) == 0:
            return None
        
        return book.iloc[0].to_dict()