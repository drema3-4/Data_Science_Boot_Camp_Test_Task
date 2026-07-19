from pathlib import Path
import sys


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app.bootstrap import (
    build_article_point_builder,
    build_base_collection_schema,
    build_model_factory,
    build_qdrant_store,
    load_base_processed_articles,
)


def main():
    schema = build_base_collection_schema()
    qdrant_store = build_qdrant_store()
    model_factory = build_model_factory()
    article_point_builder = build_article_point_builder(model_factory)

    processed_articles = load_base_processed_articles()
    points = article_point_builder.build_points(processed_articles, schema)

    qdrant_store.create_collection(schema)
    qdrant_store.upsert_points(
        schema.collection_name,
        points,
    )


if __name__ == "__main__":
    main()