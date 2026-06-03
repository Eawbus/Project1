import pandas as pd

from semantic_search import SemanticRetriever
from filter_search import MetadataFilter
from ranking import Ranker

class SearchEngine:

    def __init__(self, data_path, model_path, index_path):

        self.df = pd.read_csv(data_path)

        self.semantic = SemanticRetriever(
            self.df,
            model_path,
            index_path
        )

        self.filtering = MetadataFilter()

        self.ranker = Ranker()

    def search(
        self,
        query,
        title=None,
        author=None,
        genre=None,
        language=None,
        min_rating=None,
        semantic=True,
        k=10
    ):

        if semantic:
            results = self.semantic.retrieve(
                query,
                k=100
            )

        else:
            results = self.df.copy()

        results = self.filtering.filter(
            results,
            title=title,
            author=author,
            genre=genre,
            language=language,
            min_rating=min_rating
        )
        results = self.ranker.rank(results)

        return results.head(k)