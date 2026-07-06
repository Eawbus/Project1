"""
Main entry point for the Bookworm search engine
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pipeline.pipeline import Pipeline


def build_index():
    """Build the search index from scratch."""

    print("\n" + "=" * 60)
    print("BUILDING BOOK SEARCH INDEX")
    print("=" * 60 + "\n")

    pipeline = Pipeline()
    pipeline.build(force_rebuild=True)

    print("\n Index built successfully!")

    return pipeline


def interactive_search():
    """Interactive command-line interface."""

    print("\n" + "=" * 60)
    print("BOOKWORM")
    print("=" * 60)

    pipeline = Pipeline()

    try:
        pipeline.load()

    except FileNotFoundError:
        print("\nIndex not found. Building from scratch...")
        pipeline.build()

    while True:

        print("\n" + "-" * 60)
        print("1. Semantic Search")
        print("2. Keyword Search")
        print("3. Compare Both")
        print("4. Generate Benchmark Pool")
        print("5. Exit")

        choice = input("\nSelect option (1-5): ").strip()

        if choice in ["5", "exit", "quit"]:
            print("\n Goodbye!")
            break

        if choice == "1":
            query = input(
                "\n What kind of book are you looking for?\n> "
            ).strip()

            if not query:
                continue

            results = pipeline.semantic_search(query, k=5)

            display_results(
                results,
                "Semantic Search"
            )

        elif choice == "2":
            query = input(
                "\n Enter keywords:\n> "
            ).strip()

            if not query:
                continue

            results = pipeline.keyword_search(query, k=5)

            display_results(
                results,
                "Keyword Search"
            )

        elif choice == "3":

            query = input(
                "\nEnter query to compare:\n> "
            ).strip()

            if not query:
                continue

            pipeline.compare_search(
                query,
                k=5
            )

        elif choice == "4":
            generate_benchmark_pool(
                pipeline,
                pipeline.config.BENCHMARK_QUERIES_CSV
            )

def display_results(results, title):
    """Display search results."""

    if not results:
        print("\nNo results found.")
        return

    print(f"\n {title}")
    print("-" * 60)

    for result in results:

        print(
            f"{result['rank']}. "
            f"{result['title']} "
            f"by {result['author']}"
        )

        print(
            f"   Genres: "
            f"{result.get('genres', 'N/A')}"
        )

        print(
            f"   Rating: "
            f"{result.get('rating', 0):.2f}"
        )

        if "score" in result:
            print(
                f"   Score: "
                f"{result['score']:.4f}"
            )
        print()

def generate_benchmark_pool(
    pipeline,
    benchmark_csv,
    output_csv=None,
    k=5
):
    """
    Generate pooled relevance judgments for benchmark evaluation.

    Output format:
    query_id
    query
    book_id
    title
    author
    semantic_rank
    semantic_score
    keyword_rank
    keyword_score
    relevance
    """
    benchmark_csv = Path(benchmark_csv)

    if output_csv is None:
        output_csv = (
            benchmark_csv.parent /
            "relevance_judgments.csv"
        )

    benchmark = pd.read_csv(benchmark_csv)

    pool_rows = []

    print("\nGenerating benchmark pool...")
    print("-" * 60)

    for _, row in benchmark.iterrows():

        query_id = row["query_id"]
        query = row["query"]

        print(f"[{query_id}] {query}")

        # -----------------------------
        # Retrieve results
        # -----------------------------

        semantic_results = pipeline.semantic_search(
            query,
            k=k
        )

        keyword_results = pipeline.keyword_search(
            query,
            k=k
        )

        # -----------------------------
        # Build pooled table
        # -----------------------------

        pooled = {}

        #
        # Semantic results
        #
        for result in semantic_results:

            book_id = result["book_id"]

            pooled[book_id] = {
                "query_id": query_id,
                "query": query,
                "book_id": book_id,
                "title": result["title"],
                "author": result["author"],
                "semantic_rank": result["rank"],
                "semantic_score": result["score"],
                "keyword_rank": None,
                "keyword_score": None,
                "relevance": ""
            }

        #
        # Keyword results
        #
        for result in keyword_results:

            book_id = result["book_id"]

            if book_id not in pooled:

                pooled[book_id] = {
                    "query_id": query_id,
                    "query": query,
                    "book_id": book_id,
                    "title": result["title"],
                    "author": result["author"],
                    "semantic_rank": None,
                    "semantic_score": None,
                    "keyword_rank": result["rank"],
                    "keyword_score": result["score"],
                    "relevance": ""
                }

            else:

                pooled[book_id]["keyword_rank"] = result["rank"]
                pooled[book_id]["keyword_score"] = result["score"]

        # -----------------------------
        # Add to global pool
        # -----------------------------

        pool_rows.extend(
            pooled.values()
        )

    judgments = pd.DataFrame(pool_rows)

    judgments = judgments.sort_values(
        by=[
            "query_id",
            "semantic_rank",
            "keyword_rank"
        ],
        na_position="last"
    )

    judgments.to_csv(
        output_csv,
        index=False
    )

    print("\nPool generation complete.")
    print(f"Saved to: {output_csv}")
    print(f"Total pooled documents: {len(judgments)}")

    return judgments

def main():
    """Program entry point."""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "build":
            build_index()
        elif command == "search":
            if len(sys.argv) < 3:
                interactive_search()
                return

            query = " ".join(arg for arg in sys.argv[2:] if not arg.startswith("--"))

            pipeline = Pipeline()
            pipeline.load()

            if "--compare" in sys.argv:
                pipeline.compare_search(query, k=5)

            elif "--keyword" in sys.argv:
                results = pipeline.keyword_search(query, k=5)

                display_results(results, "Keyword Search")

            else:
                results = pipeline.semantic_search(query, k=5)

                display_results(results, "Semantic Search")

        elif command == "evaluate":
            print("Evaluation module not yet implemented.")

        else:
            print("\nUsage:")
            print("  python run.py build")
            print("  python run.py search \"query\"")
            print("  python run.py search \"query\" --keyword")
            print("  python run.py search \"query\" --compare")
            print("  python run.py evaluate")

    else:
        interactive_search()


if __name__ == "__main__":
    main()