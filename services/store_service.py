from qdrant_client import models

from services.store_manager import StoreManager
from services.document_processor import BaseDocumentProcessor
from core.config import StoreSettings


class StoreService:
    def __init__(self):
        self.store_manager = StoreManager()
        self.client = self.store_manager.get_client()
        self.sparse_model = self.store_manager.sparse_model
        self.dense_model = self.store_manager.dense_model

        self.base_document_processor = BaseDocumentProcessor()

    def __get_sparse_vectors__(self, texts: list[str]):
        return self.sparse_model.embed(texts)
    
    def __get_dense_vectors__(self, texts: list[str]):
        return self.dense_model.passage_embed(texts)
    
    def __make_points__(self, data_path) -> list[models.PointStruct]:
        data = self.base_document_processor.prepare_data(data_path)

        article_ids = data["article_id"].to_list()
        titles = data["title"].to_list()
        bodies = data["body"].to_list()

        title_dense_vectors = self.__get_dense_vectors__(titles)
        body_dense_vectors = self.__get_dense_vectors__(bodies)

        title_sparse_vectors = self.__get_sparse_vectors__(titles)
        body_sparse_vectors = self.__get_sparse_vectors__(bodies)

        points: list[models.PointStruct] = []

        for (
            article_id,
            title,
            body,
            title_dense,
            body_dense,
            title_sparse,
            body_sparse
        ) in zip(
            article_ids,
            titles,
            bodies,
            title_dense_vectors,
            body_dense_vectors,
            title_sparse_vectors,
            body_sparse_vectors
        ):
            points.append(
                models.PointStruct(
                    id=article_id,
                    vector={
                        StoreSettings.DENSE_TITLE_VECTOR_NAME: title_dense.tolist(),

                        StoreSettings.DENSE_BODY_VECTOR_NAME: body_dense.tolist(),

                        StoreSettings.SPARSE_TITLE_VECTOR_NAME: models.SparseVector(
                            indices=title_sparse.indices.tolist(),
                            values=title_sparse.values.tolist()
                        ),

                        StoreSettings.SPARSE_BODY_VECTOR_NAME: models.SparseVector(
                            indices=body_sparse.indices.tolist(),
                            values=body_sparse.values.tolist()
                        )
                    },
                    payload={
                        "title": title,
                        "body": body
                    }
                )
            )

        return points
    
    def load_data_in_store(self, data_path):
        points = self.__make_points__(data_path)

        self.client.upsert(
            collection_name=StoreSettings.COLLECTION_NAME,
            points=points,
            wait=True
        )

    def hybrid_article_search(
        self,
        query: str,
        limit: int = 10,
        prefetch_limit: int = 50
    ):
        dense_query = next(
            self.dense_model.query_embed(query)
        )

        sparse_query = next(
            self.sparse_model.query_embed(query)
        )

        sparse_vector = models.SparseVector(
            indices=sparse_query.indices.tolist(),
            values=sparse_query.values.tolist()
        )

        response = self.client.query_points(
            collection_name=StoreSettings.COLLECTION_NAME,
            prefetch=[
                # Поиск по заголовкам
                models.Prefetch(
                    query=sparse_vector,
                    using=StoreSettings.SPARSE_TITLE_VECTOR_NAME,
                    limit=prefetch_limit
                ),

                # Поиск по текстам статей
                models.Prefetch(
                    query=sparse_vector,
                    using=StoreSettings.SPARSE_BODY_VECTOR_NAME,
                    limit=prefetch_limit
                ),

                # Семантический поиск
                models.Prefetch(
                    query=dense_query.tolist(),
                    using=StoreSettings.DENSE_TITLE_VECTOR_NAME,
                    limit=prefetch_limit
                ),

                models.Prefetch(
                    query=dense_query.tolist(),
                    using=StoreSettings.DENSE_BODY_VECTOR_NAME,
                    limit=prefetch_limit
                )
            ],

            # Заголовок имеет больший вес
            query=models.RrfQuery(
                rrf=models.Rrf(
                    weights=[
                        2.0,  # title_sparse
                        1.0,  # body_sparse
                        2.0,  # title_dense
                        1.0   # body_dense
                    ]
                )
            ),

            limit=limit,
            with_payload=True
        )

        return response.points
    
    def hybrid_article_search_only_article_ids(
        self,
        query: str,
        limit: int = 10,
        prefetch_limit: int = 50
    ) -> list[list[int]]:
        results = self.hybrid_article_search(query, limit, prefetch_limit)

        article_ids = []
        for result in results:
            article_ids.append(result.id)

        return article_ids
    
    def hybrid_article_search_only_article_ids_print(
        self,
        query: str,
        limit: int = 10,
        prefetch_limit: int = 50
    ) -> list[list[int]]:
        results = self.hybrid_article_search(query, limit, prefetch_limit)

        for result in results:
            print(result.id, end=" ")
        
        print()

    def hybrid_article_search_print(
        self,
        query: str,
        limit: int = 10,
        prefetch_limit: int = 50
    ) -> None:
        results = self.hybrid_article_search(query, limit, prefetch_limit)

        for result in results:
            print(result.score)
            print(result.payload["title"])
            print(result.payload["body"][:200])
            print()