"""
Evaluation module for Bookworm retrieval experiments.
Metrics:
- Precision@5
- MRR
- nDCG@5
- MAP
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2])
)

from src.configuration.config import config

class Evaluator:
    def __init__(self, config=config):
        self.config = config

    def precision_at_k(self, relevances, k=5):
        """
        Precision@k.
        Relevant = score > 0
        """
        relevances = relevances[:k]

        if len(relevances) == 0:
            return 0.0

        relevant = sum(r > 0 for r in relevances)
        return relevant / k

    def reciprocal_rank(self, relevances):
        """
        Reciprocal Rank.
        First relevant document (score > 0)
        """
        for i, rel in enumerate(relevances, start=1):
            if rel > 0:
                return 1.0 / i

        return 0.0

    def average_precision(self, relevances):
        """
        Average Precision.
        Relevant = score > 0
        """
        binary = [1 if r > 0 else 0 for r in relevances]

        total_relevant = sum(binary)
        if total_relevant == 0:
            return 0.0

        precisions = []

        for i in range(len(binary)):
            if binary[i] == 1:
                precision_i = sum(binary[: i + 1]) / (i + 1)
                precisions.append(precision_i)

        return np.mean(precisions)

    def dcg_at_k(self, relevances, k=5):
        """
        Discounted Cumulative Gain.
        Uses graded relevance (0, 1, 2).
        """
        relevances = relevances[:k]
        dcg = 0.0

        for i, rel in enumerate(relevances):
            rank = i + 1
            dcg += (2 ** rel - 1) / np.log2(rank + 1)

        return dcg

    def ndcg_at_k(self, relevances, k=5):
        """
        Normalized DCG@k.
        Uses graded relevance.
        """
        dcg = self.dcg_at_k(relevances, k)

        ideal = sorted(relevances, reverse=True)
        idcg = self.dcg_at_k(ideal, k)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def evaluate_system(self, judgments_df, system="semantic", include_failures=False):
        """
        Evaluate one retrieval system.
        Parameters
        ----------
        system : str
            "semantic" or "keyword"
        """
        rank_col = f"{system}_rank"

        metrics = {
            "precision@5": [],
            "mrr": [],
            "map": [],
            "ndcg@5": []
        }

        evaluated_queries = 0

        for _, group in judgments_df.groupby("query_id"):
            system_docs = group[group[rank_col].notna()].copy()

            if len(system_docs) == 0:
                if include_failures:
                    evaluated_queries += 1

                    metrics["precision@5"].append(0.0)
                    metrics["mrr"].append(0.0)
                    metrics["map"].append(0.0)
                    metrics["ndcg@5"].append(0.0)
                continue

            system_docs = system_docs.sort_values(rank_col)

            relevances = (
                system_docs["relevance"]
                .astype(int)
                .tolist()
            )

            has_relevant_retrieved = any(r > 0 for r in relevances)

            # Skip queries with no relevant documents
            if not any(group["relevance"] > 0):
                continue

            if include_failures:
                evaluated_queries += 1

                if not has_relevant_retrieved:
                    metrics["precision@5"].append(0.0)
                    metrics["mrr"].append(0.0)
                    metrics["map"].append(0.0)
                    metrics["ndcg@5"].append(0.0)

                else:
                    metrics["precision@5"].append(
                        self.precision_at_k(relevances, 5)
                    )

                    metrics["mrr"].append(
                        self.reciprocal_rank(relevances)
                    )

                    metrics["map"].append(
                        self.average_precision(relevances)
                    )

                    metrics["ndcg@5"].append(
                        self.ndcg_at_k(relevances, 5)
                    )

            else:
                if not has_relevant_retrieved:
                    continue

                evaluated_queries += 1

                metrics["precision@5"].append(
                    self.precision_at_k(relevances, 5)
                )

                metrics["mrr"].append(
                    self.reciprocal_rank(relevances)
                )

                metrics["map"].append(
                    self.average_precision(relevances)
                )

                metrics["ndcg@5"].append(
                    self.ndcg_at_k(relevances, 5)
                )

        results = {
            metric: np.mean(values) if values else 0.0
            for metric, values in metrics.items()
        }

        results["queries_evaluated"] = evaluated_queries

        return results
    
    def failure_analysis(self, judgments_df):
        """
        Analyze retrieval failures.
        """
        summary = {
            "both_failed": 0,
            "semantic_only": 0,
            "keyword_only": 0,
            "both_succeeded": 0
        }

        for _, group in judgments_df.groupby("query_id"):

            semantic_success = (
                (
                    group["semantic_rank"].notna()
                ) &
                (
                    group["relevance"] > 0
                )
            ).any()

            keyword_success = (
                (
                    group["keyword_rank"].notna()
                ) &
                (
                    group["relevance"] > 0
                )
            ).any()

            if semantic_success and keyword_success:
                summary["both_succeeded"] += 1

            elif semantic_success:
                summary["semantic_only"] += 1

            elif keyword_success:
                summary["keyword_only"] += 1

            else:
                summary["both_failed"] += 1

        return summary

    def coverage_statistics(self, failures):
        """
        Compute query-level coverage statistics.
        """
        total_queries = sum(failures.values())

        semantic_success = (
            failures["both_succeeded"]
            + failures["semantic_only"]
        )

        keyword_success = (
            failures["both_succeeded"]
            + failures["keyword_only"]
        )

        return {
        "total_queries": total_queries,
        "semantic_success_rate":
            semantic_success / total_queries,
        "semantic_failure_rate":
            1 - semantic_success / total_queries,
        "keyword_success_rate":
            keyword_success / total_queries,
        "keyword_failure_rate":
            1 - keyword_success / total_queries
        }

    def evaluate(self, judgments_csv):
        """
        Run full evaluation.
        """
        df = pd.read_csv(judgments_csv)

        if "relevance" not in df.columns:
            raise ValueError("Missing relevance column.")

        semantic_results = self.evaluate_system(
            df,
            "semantic",
            include_failures=False
        )

        keyword_results = self.evaluate_system(
            df,
            "keyword",
            include_failures=False
        )

        semantic_e2e = self.evaluate_system(
            df,
            "semantic",
            include_failures=True
        )

        keyword_e2e = self.evaluate_system(
            df,
            "keyword",
            include_failures=True
        )

        failures = self.failure_analysis(df)

        coverage = self.coverage_statistics(failures)

        print("\n" + "=" * 60)
        print("BOOKWORM EVALUATION")
        print("=" * 60)

        print("\nRANKING QUALITY (SUCCESSFUL QUERIES ONLY)")
        print("-" * 60)

        print(
            f"{'Metric':<15}"
            f"{'Semantic':<12}"
            f"{'Keyword':<12}"
        )

        for metric in [
            "precision@5",
            "mrr",
            "map",
            "ndcg@5"
        ]:
            print(
                f"{metric:<15}"
                f"{semantic_results[metric]:<12.4f}"
                f"{keyword_results[metric]:<12.4f}"
            )

        print(
            f"\nQueries Evaluated:"
            f" {semantic_results['queries_evaluated']} / "
            f"{keyword_results['queries_evaluated']}"
        )

        print("\nEND-TO-END PERFORMANCE (FAILURES INCLUDED)")
        print("-" * 60)

        print(
            f"{'Metric':<15}"
            f"{'Semantic':<12}"
            f"{'Keyword':<12}"
        )

        for metric in [
            "precision@5",
            "mrr",
            "map",
            "ndcg@5"
        ]:
            print(
                f"{metric:<15}"
                f"{semantic_e2e[metric]:<12.4f}"
                f"{keyword_e2e[metric]:<12.4f}"
            )

        precision_gain = (
            (
                semantic_e2e["precision@5"]
                - keyword_e2e["precision@5"]
            )
            / keyword_e2e["precision@5"]
        ) * 100

        print("\nFAILURE ANALYSIS")
        print("-" * 60)

        for key, value in failures.items():
            print(f"{key:<20}{value}")

        print("\nRELATIVE IMPROVEMENT")
        print("-" * 60)

        metrics = [
            "precision@5",
            "mrr",
            "map",
            "ndcg@5"
        ]

        for metric in metrics:

            semantic_score = semantic_e2e[metric]
            keyword_score = keyword_e2e[metric]

            absolute_gain = (
                semantic_score
                - keyword_score
            )

            if keyword_score > 0:

                relative_gain = (
                    absolute_gain
                    / keyword_score
                ) * 100

                print(
                    f"{metric:<15}"
                    f"{absolute_gain:+.4f} "
                    f"({relative_gain:+.1f}%)"
                )

            else:

                print(
                    f"{metric:<15}"
                    f"{absolute_gain:+.4f}"
                )

        print("\nSUCCESS RATES")
        print("-" * 60)

        print(
            f"Semantic Success Rate: "
            f"{coverage['semantic_success_rate']:.1%}"
        )

        print(
            f"Keyword Success Rate: "
            f"{coverage['keyword_success_rate']:.1%}"
        )

        print(
            f"Semantic Precision@5 improvement: "
            f"{precision_gain:.1f}%"
        )

        return {
            "semantic": semantic_results,
            "keyword": keyword_results,
            "failures": failures,
            "coverage": coverage
        }


if __name__ == "__main__":
    evaluator = Evaluator()

    benchmark_file = (
        config.DATA_DIR
        / "benchmarks"
        / "relevance_judgments.csv"
    )

    evaluator.evaluate(benchmark_file)

