from itertools import islice

from qdrant_client import models
from tqdm.auto import tqdm

from models.factory import ModelFactory
from data.types import BaseProcessedArticle
from indexing.collection_schema import CollectionSchema


EMBEDDING_PROGRESS_BATCH_SIZE = 128


class ArticlePointBuilder:
    def __init__(self, model_factory: ModelFactory):
        self.model_factory = model_factory

    def _collect_vectors_with_progress(self, vectors, total: int, desc: str):
        collected_vectors = []

        with tqdm(total=total, desc=desc, unit="text") as progress:
            while True:
                batch = list(islice(vectors, EMBEDDING_PROGRESS_BATCH_SIZE))
                if not batch:
                    break

                collected_vectors.extend(batch)
                progress.update(len(batch))

        return collected_vectors

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
                vectors = self._collect_vectors_with_progress(
                    model.embed_documents(texts),
                    total=len(texts),
                    desc=f"Embedding {vector_field.name}",
                )
            elif vector_field.kind == "sparse":
                vectors = self._collect_vectors_with_progress(
                    model.embed_documents(texts),
                    total=len(texts),
                    desc=f"Embedding {vector_field.name}",
                )
            else:
                raise ValueError(...)

            vectors_by_field[vector_field.name] = vectors

        points = []

        for index, document in enumerate(tqdm(
            documents,
            desc="Building Qdrant points",
            unit="point",
        )):
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
