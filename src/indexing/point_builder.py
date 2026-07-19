from qdrant_client import models

from models.factory import ModelFactory
from data.types import BaseProcessedArticle
from indexing.collection_schema import CollectionSchema


class ArticlePointBuilder:
    def __init__(self, model_factory: ModelFactory):
        self.model_factory = model_factory

    def build_points(
        self,
        documents: list[BaseProcessedArticle],
        schema: CollectionSchema
    ):
        vectors_by_field = {}

        for vector_field in schema.vector_fields:
            texts = [
                getattr(document, vector_field.source_field)
                for document in documents
            ]

            model = self.model_factory.make_model(vector_field.model_config)

            if vector_field.kind == "dense":
                vectors = list(model.embed_documents(texts))
            elif vector_field.kind == "sparse":
                vectors = list(model.embed_documents(texts))
            else:
                raise ValueError(...)

            vectors_by_field[vector_field.name] = vectors

        points = []

        for index, document in enumerate(documents):
            point_vectors = {}

            for vector_field in schema.vector_fields:
                vector = vectors_by_field[vector_field.name][index]

                if vector_field.kind == "dense":
                    point_vectors[vector_field.name] = vector.tolist()

                elif vector_field.kind == "sparse":
                    point_vectors[vector_field.name] = models.SparseVector(
                        indices=vector.indices.tolist(),
                        values=vector.values.tolist(),
                    )

            payload = {
                field: getattr(document, field)
                for field in schema.payload_fields
            }

            points.append(
                models.PointStruct(
                    id=getattr(document, schema.id_field),
                    vector=point_vectors,
                    payload=payload,
                )
            )

        return points