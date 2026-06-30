"""
Keyword-based retrieval baseline for comparison against semantic search.
"""

import pandas as pd
import re
from typing import List, Dict

from src.configuration.config import config


class FilterSearch:
    """
    Keyword-based search baseline.

    Uses:
    - Inverted index for candidate retrieval
    - Weighted scoring:
        Title match       = 3 points
        Genre match       = 2 points
        Description match = 1 point
    """

    def __init__(self, config=config):
        self.config = config

        self.corpus_df = None

        self.word_index = {}

        self._stopwords = self._get_stopwords()

    def _get_stopwords(self) -> set:
        """
        Common English stopwords plus
        retrieval-neutral book terms.
        """

        return {
            "a", "an", "the", "of", "and", "to", "for",
            "on", "at", "by", "with", "without", "from",
            "in", "into", "through", "during", "including",
            "among", "between", "after", "before", "about",
            "against", "along", "around", "as", "behind",
            "below", "beneath", "beside", "beyond", "but",
            "except", "off", "onto", "out", "outside",
            "over", "past", "since", "than", "toward",
            "under", "upon", "within", # Book-domain stopwords
            "book", "books", "story", "stories", "novel",
            "novels", "work", "works", "character","characters"
        }

    def load_corpus(self, df: pd.DataFrame):
        """
        Load corpus and build inverted index.
        """

        self.corpus_df = df.copy()

        if "text" not in self.corpus_df.columns:
            self.corpus_df["text"] = (
                self.corpus_df.apply(
                    self._build_text,
                    axis=1
                )
            )

        self._build_inverted_index()

        print(
            f"Filter search loaded with "
            f"{len(self.corpus_df)} books"
        )

        print(
            f"Inverted index contains "
            f"{len(self.word_index)} unique terms"
        )

    def _build_text(self, row: pd.Series) -> str:
        """
        Combined searchable text.
        """

        parts = []

        if pd.notna(row.get("title")):
            parts.append(str(row["title"]))

        if pd.notna(row.get("description")):
            parts.append(str(row["description"]))

        if pd.notna(row.get("genres")):
            parts.append(str(row["genres"]))

        return " ".join(parts).lower()

    def _tokenize(self, text: str) -> List[str]:
        """
        Convert text to searchable tokens.
        """

        if not text:
            return []

        words = re.findall(
            r"\b[a-z0-9]+\b",
            text.lower()
        )

        words = [
            w
            for w in words
            if (
                w not in self._stopwords
                and len(w) > 1
            )
        ]

        return words

    def _build_inverted_index(self):
        """
        Build inverted index:
        word -> set(document_ids)
        """

        self.word_index = {}

        for idx, row in self.corpus_df.iterrows():

            text = row.get("text", "")

            words = self._tokenize(text)

            for word in words:

                if word not in self.word_index:
                    self.word_index[word] = set()

                self.word_index[word].add(idx)

    def search(
        self,
        query: str,
        k: int = 10
    ) -> List[Dict]:
        """
        Keyword retrieval using weighted field matching.
        """

        if self.corpus_df is None:
            raise ValueError(
                "Corpus not loaded. "
                "Call load_corpus() first."
            )

        query_words = self._tokenize(query)

        if not query_words:
            return []

        candidate_docs = set()

        for word in query_words:

            if word in self.word_index:
                candidate_docs.update(
                    self.word_index[word]
                )

        doc_scores = {}

        for doc_idx in candidate_docs:

            row = self.corpus_df.iloc[doc_idx]

            title = str(
                row.get("title", "")
            ).lower()

            genres = str(
                row.get("genres", "")
            ).lower()

            description = str(
                row.get("description", "")
            ).lower()

            score = 0

            for word in query_words:

                if word in title:
                    score += 3

                if word in genres:
                    score += 2

                if word in description:
                    score += 1

            if score > 0:
                doc_scores[doc_idx] = score

        ranked_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]

        results = []

        max_possible_score = (
            len(query_words) * 6
        )

        for rank, (doc_idx, score) in enumerate(
            ranked_docs,
            start=1
        ):

            book = self.corpus_df.iloc[doc_idx]

            results.append({
                "rank": rank,
                "score": score / max_possible_score,
                "match_count": score,

                "book_id": int(
                    book.get(
                        "book_id",
                        doc_idx
                    )
                ),

                "title": book.get(
                    "title",
                    "Unknown"
                ),

                "author": book.get(
                    "author",
                    "Unknown"
                ),

                "genres": book.get(
                    "genres",
                    ""
                ),

                "rating": float(
                    book.get(
                        "rating",
                        0
                    )
                ),

                "num_ratings": int(
                    book.get(
                        "num_ratings",
                        0
                    )
                ),

                "year": (
                    int(book.get("year", 0))
                    if pd.notna(
                        book.get("year")
                    )
                    else None
                )
            })

        return results