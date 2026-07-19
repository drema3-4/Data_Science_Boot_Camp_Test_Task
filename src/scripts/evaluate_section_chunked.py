from pathlib import Path
import json
import sys
from datetime import datetime

from tqdm.auto import tqdm


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app.bootstrap import (
    build_section_chunked_search_pipeline,
    load_base_calibration_samples,
)
from evaluation.metrics import recall_k, map_k


RUNS_DIR = SRC_DIR / "runs"


def main():
    calibration_samples = load_base_calibration_samples()
    pipeline = build_section_chunked_search_pipeline(with_reranker=True)

    candidate_limit = 150
    prefetch_limit = 200
    final_limit = 10

    all_predictions = []
    all_relevants = []

    for sample in tqdm(
        calibration_samples,
        desc="Evaluating section chunked",
        unit="query",
    ):
        results = pipeline.search(
            query=sample.query,
            final_limit=final_limit,
            candidate_limit=candidate_limit,
            prefetch_limit=prefetch_limit,
            with_reranker=True,
        )

        predicted_ids = [
            result.article_id
            for result in results
        ]

        all_predictions.append(predicted_ids)
        all_relevants.append(sample.ground_truth)

    metrics = {
        "schema": "section_chunked",
        "candidate_limit": candidate_limit,
        "prefetch_limit": prefetch_limit,
        "final_limit": final_limit,
        "recall@10": recall_k(all_predictions, all_relevants, 10),
        "map@10": map_k(all_predictions, all_relevants, 10),
    }

    run_dir = RUNS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S_section_chunked")
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