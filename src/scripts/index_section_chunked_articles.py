from pathlib import Path
import sys

from tqdm.auto import tqdm


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app.bootstrap import (
    build_article_point_builder,
    build_model_factory,
    build_qdrant_store,
    build_section_chunked_collection_schema,
    load_section_chunked_processed_articles,
)


def main():
    progress = tqdm(total=7, desc="Indexing section chunked articles", unit="step")

    schema = build_section_chunked_collection_schema()
    progress.update(1)

    qdrant_store = build_qdrant_store()
    progress.update(1)

    model_factory = build_model_factory()
    point_builder = build_article_point_builder(model_factory)
    progress.update(1)

    progress.set_description("Processing section chunks")
    processed_points = load_section_chunked_processed_articles()
    tqdm.write(f"Processed points: {len(processed_points)}")
    progress.update(1)

    progress.set_description("Creating collection")
    qdrant_store.create_collection(schema)
    progress.update(1)

    progress.set_description("Building Qdrant points")
    points = point_builder.build_points(processed_points, schema)
    tqdm.write(f"Built points: {len(points)}")
    progress.update(1)

    progress.set_description("Uploading to Qdrant")
    qdrant_store.upsert_points(schema.collection_name, points)
    progress.update(1)
    progress.close()


if __name__ == "__main__":
    main()
