from qdrant_client import QdrantClient, models
from tqdm.auto import tqdm

from indexing.collection_schema import CollectionSchema


DEFAULT_UPSERT_BATCH_SIZE = 128


class QdrantStore:
    def __init__(self, url, timeout):
        self.client = QdrantClient(
            url=url,
            timeout=timeout
        )

    def create_collection(self, schema: CollectionSchema):
        self.client.create_collection(
            collection_name=schema.collection_name,
            vectors_config = {
                field.name: models.VectorParams(
                    size=field.model_config.dim,
                    distance=models.Distance.COSINE,
                )
                for field in schema.vector_fields
                if field.kind == "dense"
            },
            sparse_vectors_config = {
                field.name: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False),
                    modifier=models.Modifier.IDF,
                )
                for field in schema.vector_fields
                if field.kind == "sparse"
            }
        )

    def upsert_points(
        self,
        collection_name,
        points,
        batch_size: int = DEFAULT_UPSERT_BATCH_SIZE,
    ):
        for start in tqdm(
            range(0, len(points), batch_size),
            desc="Uploading Qdrant batches",
            unit="batch",
        ):
            batch = points[start:start + batch_size]

            self.client.upsert(
                collection_name=collection_name,
                points=batch,
                wait=True
            )
