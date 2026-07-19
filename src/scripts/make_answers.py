from pathlib import Path
import json
import sys
from datetime import datetime
import pandas as pd
from tqdm.auto import tqdm


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app.bootstrap import load_base_test_samples, build_search_pipeline


ANSWERS_DIR = SRC_DIR / "answers"


def main():
    test_samples = load_base_test_samples()[:2]
    pipeline = build_search_pipeline(with_reranker=True)

    predicted_ids = [
        [
            str(predicted.article_id)
            for predicted in pipeline.search(test_sample.query_text, final_limit=10)
        ]
        for test_sample in tqdm(test_samples, desc="Generating answers", unit="query")
    ]

    results = pd.DataFrame(columns=["query_id", "answer"])
    for predicted, test_sample in zip(
        predicted_ids,
        test_samples
    ):
        results.loc[results.shape[0]] = [test_sample.query_id, " ".join(predicted)]

    answers_dir = ANSWERS_DIR / "base_model_answers"
    answers_dir.mkdir(parents=True, exist_ok=True)

    answers_path = answers_dir / "answer.csv"

    results.to_csv(answers_path, index=False)


if __name__ == "__main__":
    main()