from pathlib import Path
import json
import sys
from datetime import datetime

from tqdm.auto import tqdm


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app.bootstrap import (
    build_chunked_body_search_pipeline,
    load_base_calibration_samples,
)
from evaluation.metrics import recall_k, map_k


RUNS_DIR = SRC_DIR / "runs"


def main():
    calibration_samples = load_base_calibration_samples()
    pipeline = build_chunked_body_search_pipeline(with_reranker=True)

    all_predictions = []
    all_relevants = []

    for sample in tqdm(
        calibration_samples,
        desc="Evaluating chunked body",
        unit="query",
    ):
        results = pipeline.search(
            query=sample.query,
            final_limit=10,
        )

        predicted_ids = [
            result.article_id
            for result in results
        ]

        all_predictions.append(predicted_ids)
        all_relevants.append(sample.ground_truth)

    candidate_predictions = []
    candidate_relevants = []

    for sample in tqdm(
        calibration_samples,
        desc="Evaluating chunked body candidates",
        unit="query",
    ):
        results = pipeline.search(
            query=sample.query,
            final_limit=50,
            candidate_limit=50,
            with_reranker=False,
        )

        predicted_ids = [
            result.article_id
            for result in results
        ]

        candidate_predictions.append(predicted_ids)
        candidate_relevants.append(sample.ground_truth)

    metrics = {
        "schema": "chunked_body",
        "recall@10": recall_k(all_predictions, all_relevants, 10),
        "map@10": map_k(all_predictions, all_relevants, 10),
        "candidate_recall@50": recall_k(candidate_predictions, candidate_relevants, 50),
    }

    run_dir = RUNS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S_chunked_body")
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(metrics)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
