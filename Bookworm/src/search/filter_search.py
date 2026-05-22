class MetadataFilter:
   def filter(self, dataframe, title=None, author=None, genre=None, language=None, min_rating=None):
      results = dataframe

      if title:
         results = results[
            results["title"]
            .str.contains(
                  title, 
                  case=False, 
                  na=False
               )
            ]

      if author:
         results = results[
            results["author"]
            .str.contains(
                  author,
                  case=False,
                  na=False
               )
            ]

      if genre:
         results = results[
            results["genres"]
            .str.contains(
                  genre,
                  case=False,
                  na=False
               )
            ]

      if language:
         results = results[
            results["language"]
            == language
            ]

      if min_rating:
         results = results[
            results["rating"]
            >= min_rating
         ]

      return results