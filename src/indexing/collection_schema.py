from dataclasses import dataclass
from fastembed.common.model_description import ModelSource

from models.configs import SparseEmbedderConfig, DenseEmbedderConfig


@dataclass(frozen=True)
class VectorFieldConfig:
    name: str
    kind: str               # sparse or dense
    source_field: str
    model_config: object
    weight: float = 1.0

@dataclass(frozen=True)
class CollectionSchema:
    collection_name: str
    vector_fields: list[VectorFieldConfig]
    payload_fields: list[str]
    id_field: str = "article_id"

def make_base_schema(
    title_sparse_config: SparseEmbedderConfig,
    body_sparse_config: SparseEmbedderConfig,
    dense_config: DenseEmbedderConfig
):
    return CollectionSchema(
        collection_name="base_collection",
        vector_fields=[
            VectorFieldConfig(
                name="title_sparse",
                kind="sparse",
                source_field="title",
                model_config=title_sparse_config,
                weight=2.0
            ),
            VectorFieldConfig(
                name="body_sparse",
                kind="sparse",
                source_field="body_lexical",
                model_config=body_sparse_config,
                weight=1.0
            ),
            VectorFieldConfig(
                name="title_dense",
                kind="dense",
                source_field="title",
                model_config=dense_config,
                weight=2.0
            ),
            VectorFieldConfig(
                name="body_dense",
                kind="dense",
                source_field="body_plain",
                model_config=dense_config,
                weight=1.0
            )
        ],
        payload_fields=[
            "title",
            "body_plain"
        ],
        id_field="article_id"
    )