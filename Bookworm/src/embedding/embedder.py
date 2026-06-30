"""
Embedder module for generating embeddings using SentenceTransformers
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Optional
from pathlib import Path

from src.configuration.config import config


class Embedder:
    """
    Handles embedding generation for both corpus documents and queries.
    """

    def __init__(
        self,
        model_name: str = None,
        model_path: Optional[Path] = None,
        batch_size: int = None,
        max_seq_length: int = None,
        config=config
    ):

        self.config = config

        self.model_name = model_name or self.config.MODEL_NAME
        self.model_path = model_path or self.config.MODEL_PATH

        self.batch_size = batch_size or self.config.BATCH_SIZE
        self.max_seq_length = (
            max_seq_length or self.config.MAX_SEQ_LENGTH
        )

        self.model = None

        self._load_model()

    def _load_model(self):
        """
        Load the SentenceTransformer model.
        """

        print(f"Loading model: {self.model_name}")

        try:

            if self.model_path.exists():

                self.model = SentenceTransformer(
                    str(self.model_path)
                )

            else:

                self.model = SentenceTransformer(
                    self.model_name
                )

        except Exception as e:

            print(
                f"Failed to load local model: {e}"
            )

            print(
                f"Falling back to {self.model_name}"
            )

            self.model = SentenceTransformer(
                self.model_name
            )

        self.model.max_seq_length = (
            self.max_seq_length
        )

        print(
            f"Model loaded. "
            f"Max sequence length: "
            f"{self.model.max_seq_length}"
        )

        print(
            f"Embedding dimension: "
            f"{self.model.get_embedding_dimension()}"
        )

    def encode(
        self,
        texts: List[str],
        normalize: bool = True,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode a list of texts into embeddings.

        Args:
            texts: List of text strings.
            normalize: Whether to L2-normalize embeddings.
            show_progress: Show progress bar.

        Returns:
            numpy array of embeddings.
        """

        if not texts:
            return np.array([])

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )

        return embeddings

    def encode_corpus(
        self,
        df: pd.DataFrame,
        text_column: str = "text"
    ) -> np.ndarray:
        """
        Encode an entire corpus DataFrame.

        Args:
            df: DataFrame containing corpus texts.
            text_column: Name of text column.

        Returns:
            numpy array of embeddings.
        """

        texts = df[text_column].tolist()

        return self.encode(
            texts,
            normalize=True
        )

    def encode_query(
        self,
        query: str
    ) -> np.ndarray:
        """
        Encode a single query.

        Args:
            query: Query string.

        Returns:
            Query embedding with shape (1, dim).
        """

        return self.encode(
            [query],
            normalize=True,
            show_progress=False
        )

    def encode_queries(
        self,
        queries: List[str]
    ) -> np.ndarray:
        """
        Encode multiple queries.

        Args:
            queries: List of query strings.

        Returns:
            numpy array of embeddings.
        """

        return self.encode(
            queries,
            normalize=True,
            show_progress=False
        )

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        filepath: Path
    ):
        """
        Save embeddings to disk.
        """

        filepath.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        np.save(filepath, embeddings)

        print(
            f"Embeddings saved to {filepath}"
        )

    def load_embeddings(
        self,
        filepath: Path
    ) -> np.ndarray:
        """
        Load embeddings from disk.
        """

        return np.load(filepath)

    @property
    def embedding_dim(self) -> int:
        """
        Return embedding dimension.
        """

        return (
            self.model
            .get_embedding_dimension()
        )

    @property
    def model_info(self) -> dict:
        """
        Return model metadata.
        """

        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "max_seq_length": self.max_seq_length,
            "batch_size": self.batch_size
        }