from qdrant_client import models

from indexing.qdrant_store import QdrantStore
from models.factory import ModelFactory
from indexing.collection_schema import CollectionSchema


class HybridQdrantRetriever:
    def __init__(
        self,
        qdrant_store: QdrantStore,
        model_factory: ModelFactory,
        schema: CollectionSchema
    ):
        self.qdrant_store = qdrant_store
        self.model_factory = model_factory
        self.schema = schema

    def retrieve(
        self,
        query: str,
        candidate_limit: int = 50,
        prefetch_limit: int = 100
    ):
        prefetches = []
        weights = []

        for vector_field in self.schema.vector_fields:
            model = self.model_factory.make_model(vector_field.model_config)
            query_vector = model.embed_query(query)

            if vector_field.kind == "dense":
                qdrant_query = query_vector.tolist()

            elif vector_field.kind == "sparse":
                qdrant_query = models.SparseVector(
                    indices=query_vector.indices.tolist(),
                    values=query_vector.values.tolist(),
                )
            
            else:
                raise ValueError(f"Unknown vector field kind: {vector_field.kind}")

            prefetches.append(
                models.Prefetch(
                    query=qdrant_query,
                    using=vector_field.name,
                    limit=prefetch_limit,
                )
            )

            weights.append(vector_field.weight)

        response = self.qdrant_store.client.query_points(
            collection_name=self.schema.collection_name,
            prefetch=prefetches,
            query=models.RrfQuery(
                rrf=models.Rrf(
                    weights=weights
                )
            ),
            limit=candidate_limit,
            with_payload=True
        )

        return response.points