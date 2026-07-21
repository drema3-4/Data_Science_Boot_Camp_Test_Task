import gzip
import json
from collections.abc import Iterable
from pathlib import Path

from qdrant_client import models

from indexing.collection_schema import CollectionSchema


SRC_DIR = Path(__file__).resolve().parents[1]
DEFAULT_POINTS_DIR = SRC_DIR / "data_sources" / "generated" / "points"


def get_points_path(
    schema: CollectionSchema,
    points_dir: Path = DEFAULT_POINTS_DIR,
) -> Path:
    return points_dir / f"{schema.collection_name}.jsonl.gz"


def _dump_point(point: models.PointStruct) -> dict:
    if hasattr(point, "model_dump"):
        return point.model_dump(mode="json")

    return point.dict()


def _load_point(raw_point: dict) -> models.PointStruct:
    if hasattr(models.PointStruct, "model_validate"):
        return models.PointStruct.model_validate(raw_point)

    return models.PointStruct.parse_obj(raw_point)


def save_points(
    points: Iterable[models.PointStruct],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")

    with gzip.open(temp_path, "wt", encoding="utf-8") as file:
        for point in points:
            json.dump(_dump_point(point), file, ensure_ascii=False)
            file.write("\n")

    temp_path.replace(path)


def load_points(path: Path) -> list[models.PointStruct]:
    with gzip.open(path, "rt", encoding="utf-8") as file:
        return [
            _load_point(json.loads(line))
            for line in file
        ]
