from pathlib import Path
import sys


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app.bootstrap import build_chunked_body_search_pipeline


def main():
    pipeline = build_chunked_body_search_pipeline(with_reranker=True)

    results = pipeline.search(
        query="Как отправить заказ через Авито?",
        final_limit=10,
        candidate_limit=50,
        prefetch_limit=100,
    )

    for result in results:
        print(result.article_id, result.score, result.title)


if __name__ == "__main__":
    main()