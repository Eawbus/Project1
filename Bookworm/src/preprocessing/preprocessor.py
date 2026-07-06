"""
Preprocessor module for cleaning text data
"""
import re
import pandas as pd
from typing import List, Union

from src.configuration.config import config

class Preprocessor:
    """
    Handles text cleaning for both corpus documents and queries
    """
    
    def __init__(self, config=config):
        self.config = config
        self.html_pattern = re.compile(r"<.*?>")
        self.whitespace_pattern = re.compile(r"\s+")
        self.punctuation_pattern = re.compile(r"[^\w\s.,!?\-']")
    
    def clean_text(self, text: Union[str, float]) -> str:
        """
        Basic text cleaning for any string
        
        Args:
            text: Input text or NaN
            
        Returns:
            Cleaned text string
        """
        if pd.isna(text):
            return ""
        
        text = str(text)
        # Remove HTML tags
        text = self.html_pattern.sub("", text)
        # Remove extra whitespace
        text = self.whitespace_pattern.sub(" ", text)
        # Remove excessive punctuation (optional)
        # text = self.punctuation_pattern.sub("", text)
        return text.strip()
    
    def clean_genres(self, genres: Union[str, float]) -> str:
        """
        Clean genre string: remove brackets, split, and format
        
        Args:
            genres: Genre string or NaN
            
        Returns:
            Cleaned genre string
        """
        if pd.isna(genres):
            return ""
        
        genres = str(genres)
        # Remove brackets and quotes
        genres = genres.replace("[", "").replace("]", "").replace("'", "")
        # Split and clean each genre
        genre_list = [g.strip().title() for g in genres.split(",") if g.strip()]
        return ", ".join(genre_list)
    
    def build_corpus_text(self, row: pd.Series) -> str:
        """
        Build the full text representation for a corpus document
        Args: row: DataFrame row containing book data 
        Returns: Combined text string for embedding
        """
        title = row.get("title", "")
        description = row.get("description", "")
        genres = row.get("genres", "")
        
        # Clean each component
        title = self.clean_text(title)
        description = self.clean_text(description)
        genres = self.clean_genres(genres)
        
        # Combine with proper formatting
        parts = []
        if title:
            parts.append(title)
        if description:
            parts.append(description)
        if genres:
            parts.append(f"Genres: {genres}")
        
        return ". ".join(parts)
    
    def clean_query(self, query: str) -> str:
        """
        Simplified cleaning for queries
        Args: query: Raw query string
        Returns: Cleaned query string
        """
        if not query or pd.isna(query):
            return ""
        return self.clean_text(query)
    
    def preprocess_corpus(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full preprocessing pipeline for corpus data
        Args: df: Raw DataFrame
        Returns: Preprocessed DataFrame - .csv
        """
        print(f"Original shape: {df.shape}")
        
        # Select and rename columns
        available_cols = {k: v for k, v in self.config.COLUMN_MAP.items() 
                         if k in df.columns}
        df = df[list(available_cols.keys())]
        df = df.rename(columns=available_cols)
        print(f"After column selection: {df.shape}")
        
        # Drop rows without title or description
        df = df.dropna(subset=["title", "description"])
        print(f"After dropping missing semantics: {df.shape}")
        
        # Filter short descriptions
        df = df[df["description"].str.len() > 50]
        print(f"After removing short descriptions: {df.shape}")
        
        # Clean text fields
        print("Cleaning text fields...")
        df["title"] = df["title"].apply(self.clean_text)
        df["description"] = df["description"].apply(self.clean_text)
        
        if "author" in df.columns:
            df["author"] = df["author"].apply(self.clean_text)
        
        if "genres" in df.columns:
            df["genres"] = df["genres"].apply(self.clean_genres)
        else:
            df["genres"] = ""
        
        # Clean numeric fields
        print("Cleaning numeric fields...")
        for col in ["rating", "num_ratings", "year"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df = df.dropna(subset=["rating", "num_ratings"])
        
        # Remove duplicates
        print("Removing duplicates...")
        df = df.drop_duplicates(subset=["title", "author"])
        
        # Add ID
        df = df.reset_index(drop=True)
        df["book_id"] = df.index
        
        print(f"Final shape: {df.shape}")
        return df
    
    def preprocess_query(self, query: str) -> str:
        """
        Preprocess a single query
        Args: query: Raw query string
        Returns: Preprocessed query string
        """
        return self.clean_query(query)
    
    def preprocess_queries(self, queries: List[str]) -> List[str]:
        """
        Preprocess multiple queries
        Args: queries: List of query strings
        Returns: List of preprocessed query strings
        """
        return [self.preprocess_query(q) for q in queries]