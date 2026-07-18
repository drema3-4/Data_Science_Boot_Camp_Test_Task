import pandas as pd
from typing import Any
from tqdm.auto import tqdm

from services.store_service import StoreService


class ExperimentService:
    def __init__(self, store_service: StoreService):
        self.store_service = store_service

    def load_calibration_data(self, calibration_data_path: str) -> list[dict[str, Any]]:
        calibration_data = pd.read_feather(calibration_data_path)

        queries_relevants = [
            {
                "query": query,
                "relevant": list(map(int, relevant.split()))
            }

            for query, relevant in zip(
                calibration_data["query_text"].to_list(),
                calibration_data["ground_truth"].to_list()
            )
        ]

        return queries_relevants
    
    def recall_k(self, query: str, relevant: list[int], k: int) -> float:
        top_k = self.store_service.gen_candidates(query)[:k]

        found_relevant_count = sum([point.id in relevant for point in top_k]) + 0.0

        return found_relevant_count / len(relevant)

    def mean_recall_k(self, calibration_data_path: str, k: int) -> float:
        queries_relevants = self.load_calibration_data(calibration_data_path)

        progress_bar = tqdm(
            queries_relevants,
            desc=f"Расчёт Recall@{k}",
            unit="запрос",
            dynamic_ncols=True,
        )

        mean_rec_k = 0.0
        for processed_count, query_relevant in enumerate(
            progress_bar,
            start=1,
        ):
            query = query_relevant["query"]
            relevant = query_relevant["relevant"]

            mean_rec_k += self.recall_k(query, relevant, k)

            progress_bar.set_postfix(
                mean=f"{mean_rec_k / processed_count:.4f}"
            )

        return mean_rec_k / len(queries_relevants)
    
    def average_precision_k(self, query: str, relevant: list[int], k: int) -> float:
        top_k = self.store_service.reranking(query)[:k]

        found_articles_id = [article["article_id"] for article in top_k]

        sum_precision_k = sum(1.0 / i for i, article_id in enumerate(found_articles_id, start=1) if article_id in relevant)

        relevant_size = len(relevant)

        return sum_precision_k / (k if k <= relevant_size else relevant_size)

    def mean_average_precision_k(self, calibration_data_path: str, k: int) -> float:
        queries_relevants = self.load_calibration_data(calibration_data_path)

        progress_bar = tqdm(
            queries_relevants,
            desc=f"Расчёт MAP@{k}",
            unit="запрос",
            dynamic_ncols=True,
        )

        mean_avg_pre_k = 0.0
        for processed_count, query_relevant in enumerate(
            progress_bar,
            start=1,
        ):
            query = query_relevant["query"]
            relevant = query_relevant["relevant"]

            mean_avg_pre_k += self.average_precision_k(query, relevant, k)

            progress_bar.set_postfix(
                mean=(
                    f"{mean_avg_pre_k / processed_count:.4f}"
                )
            )

        return mean_avg_pre_k / len(queries_relevants)