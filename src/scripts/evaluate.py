from pathlib import Path
import json
import sys
from datetime import datetime


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app.bootstrap import build_evaluator, load_base_calibration_samples


RUNS_DIR = SRC_DIR / "runs"


def main():
    calibration_samples = load_base_calibration_samples()
    evaluator = build_evaluator(with_reranker=True)

    recall_at_10, map_at_10 = evaluator.evaluate(
        calibration_samples,
        k=10,
    )
    candidate_recall_at_50 = evaluator.evaluate_candidate_generation(
        calibration_samples,
        candidate_limit=50,
        k=50,
    )

    metrics = {
        "recall@10": recall_at_10,
        "map@10": map_at_10,
        "candidate_recall@50": candidate_recall_at_50,
    }

    run_dir = RUNS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S_base")
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