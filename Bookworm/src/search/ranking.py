class Ranker:
   def rank(self, dataframe):
      if "semantic_score" in dataframe.columns:
         dataframe = dataframe.sort_values(
            by="semantic_score",
            ascending=False
            )

      return dataframe