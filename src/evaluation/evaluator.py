from retrieval.pipeline import SearchPipeline
from data.types import CalibrationSample
from evaluation.metrics import recall_k, map_k


class Evaluator:
    def __init__(self, pipeline: SearchPipeline):
        self.pipeline = pipeline

    def evaluate(
        self,
        calibration_samples: list[CalibrationSample],
        k: int = 10
    ) -> tuple[float, float]:
        queries = [
            calibration_sample.query for calibration_sample in calibration_samples
        ]
        relevants = [
            calibration_sample.ground_truth for calibration_sample in calibration_samples
        ]

        resultss = [
            self.pipeline.search(query, final_limit=k)
            for query in queries
        ]
        predicted_ids = [[result.article_id for result in results] for results in resultss]

        recall_k_score = recall_k(predicted_ids, relevants, k)
        map_k_score = map_k(predicted_ids, relevants, k)
        
        return (recall_k_score, map_k_score)
    
    def evaluate_reranker(
        self,
        calibration_samples: list[CalibrationSample],
        k: int = 10
    ) -> float:
        queries = [
            calibration_sample.query for calibration_sample in calibration_samples
        ]
        relevants = [
            calibration_sample.ground_truth for calibration_sample in calibration_samples
        ]

        resultss = [
            self.pipeline.search(query, final_limit=k)
            for query in queries
        ]
        predicted_ids = [[result.article_id for result in results] for results in resultss]

        map_k_score = map_k(predicted_ids, relevants, k)
        
        return map_k_score
    
    def evaluate_candidate_generation(
        self,
        calibration_samples: list[CalibrationSample],
        candidate_limit: int = 50,
        k: int = 50
    ) -> float:
        queries = [
            calibration_sample.query for calibration_sample in calibration_samples
        ]
        relevants = [
            calibration_sample.ground_truth for calibration_sample in calibration_samples
        ]

        resultss = [
            self.pipeline.search(query, final_limit=k, candidate_limit=candidate_limit, with_reranker=False)
            for query in queries
        ]
        predicted_ids = [[result.article_id for result in results] for results in resultss]

        recall_k_score = recall_k(predicted_ids, relevants, k)
        
        return recall_k_score