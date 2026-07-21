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
from indexing.point_io import get_points_path, load_points, save_points


def main():
    schema = build_base_collection_schema()
    qdrant_store = build_qdrant_store()
    points_path = get_points_path(schema)

    if points_path.exists():
        print(f"Loading cached Qdrant points: {points_path}")
        points = load_points(points_path)
        print(f"Loaded points: {len(points)}")
    else:
        model_factory = build_model_factory()
        article_point_builder = build_article_point_builder(model_factory)

        processed_articles = load_base_processed_articles()
        points = article_point_builder.build_points(processed_articles, schema)
        save_points(points, points_path)
        print(f"Saved Qdrant points: {points_path}")

    qdrant_store.create_collection(schema)
    qdrant_store.upsert_points(
        schema.collection_name,
        points,
    )


if __name__ == "__main__":
    main()
