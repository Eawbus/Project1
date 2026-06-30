#!/usr/bin/env python
"""
Main entry point for the Bookworm search engine
"""

import sys
from pathlib import Path

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
        print("4. Exit")

        choice = input("\nSelect option (1-4): ").strip()

        if choice in ["4", "q", "quit", "exit"]:
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
                "\n🔍 Enter query to compare:\n> "
            ).strip()

            if not query:
                continue

            pipeline.compare_search(
                query,
                k=5
            )

        else:
            print("Invalid option.")


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