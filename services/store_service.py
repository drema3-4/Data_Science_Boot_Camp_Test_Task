from qdrant_client import models

from services.store_manager import StoreManager
from services.document_processor import BaseDocumentProcessor
from core.config import StoreSettings
from services.models_manager import ModelsManager


class StoreService:
    def __init__(self, store_manager: StoreManager, models_manager: ModelsManager):
        self.store_manager = store_manager
        self.client = self.store_manager.get_client()

        self.models_manager = models_manager
        self.title_sparse_model = self.models_manager.get_title_sparse_model()
        self.body_sparse_model = self.models_manager.get_body_sparse_model()
        self.gen_candidate_dense_model = self.models_manager.get_gen_candidate_dense_model()
        self.reranking_model = self.models_manager.get_reranking_model()

        self.base_document_processor = BaseDocumentProcessor()

    def __get_title_sparse_vectors__(self, texts: list[str]):
        return self.title_sparse_model.embed(texts)
    
    def __get_body_sparse_vectors__(self, texts: list[str]):
        return self.body_sparse_model.embed(texts)
    
    def __get_gen_candidate_dense_vectors__(self, texts: list[str]):
        return self.gen_candidate_dense_model.passage_embed(texts)
    
    def __make_points__(self, data_path) -> list[models.PointStruct]:
        data = self.base_document_processor.prepare_data(data_path)

        article_ids = data["article_id"].to_list()
        titles = data["title"].to_list()
        bodies = data["body_plain"].to_list()
        bodies_for_sparse = data["body_lexical"].to_list()
        bodies_for_dense = data["body_plain"].to_list()

        title_dense_vectors = self.__get_gen_candidate_dense_vectors__(titles)
        body_dense_vectors = self.__get_gen_candidate_dense_vectors__(bodies_for_dense)

        title_sparse_vectors = self.__get_title_sparse_vectors__(titles)
        body_sparse_vectors = self.__get_body_sparse_vectors__(bodies_for_sparse)

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

    def gen_candidates(
        self,
        query: str,
        candidate_limit: int=50,
        prefetch_limit: int=100
    ):
        gen_candidate_dense_query = next(
            self.gen_candidate_dense_model.query_embed(query)
        )

        title_sparse_query = next(
            self.title_sparse_model.query_embed(query)
        )
        title_sparse_vector = models.SparseVector(
            indices=title_sparse_query.indices.tolist(),
            values=title_sparse_query.values.tolist()
        )

        body_sparse_query = next(
            self.body_sparse_model.query_embed(query)
        )
        body_sparse_vector = models.SparseVector(
            indices=body_sparse_query.indices.tolist(),
            values=body_sparse_query.values.tolist()
        )

        response = self.client.query_points(
            collection_name=StoreSettings.COLLECTION_NAME,
            prefetch=[
                # Поиск по заголовкам
                models.Prefetch(
                    query=title_sparse_vector,
                    using=StoreSettings.SPARSE_TITLE_VECTOR_NAME,
                    limit=prefetch_limit
                ),

                # Поиск по текстам статей
                models.Prefetch(
                    query=body_sparse_vector,
                    using=StoreSettings.SPARSE_BODY_VECTOR_NAME,
                    limit=prefetch_limit
                ),

                # Семантический поиск
                models.Prefetch(
                    query=gen_candidate_dense_query.tolist(),
                    using=StoreSettings.DENSE_TITLE_VECTOR_NAME,
                    limit=prefetch_limit
                ),

                models.Prefetch(
                    query=gen_candidate_dense_query.tolist(),
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

            limit=candidate_limit,
            with_payload=True
        )

        return response.points
    
    def reranking(
        self,
        query: str,
        final_limit: int = 10,
        candidate_limit: int = 50
    ):
        candidates = self.gen_candidates(query, candidate_limit)

        documents = [
            (
                f"Заголовок: {candidate.payload['title']}\n"
                f"Текст: {candidate.payload['body']}"
            )
            for candidate in candidates
        ]

        pairs = [
            (query, document)
            for document in documents
        ]

        scores = self.reranking_model.predict(pairs)

        results = [
            {
                "article_id": candidate.id,
                "score": float(score),
                "title": candidate.payload["title"],
                "body": candidate.payload["body"]
            }
            for candidate, score in zip(candidates, scores)
        ]

        results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return results[:final_limit]
    
    def hybrid_article_search_only_article_ids(
        self,
        query: str,
        final_limit: int = 10,
        candidate_limit: int = 50
    ) -> list[list[int]]:
        results = self.reranking(query, final_limit, candidate_limit)

        article_ids = []
        for result in results:
            article_ids.append(result["article_id"])

        return article_ids
    
    def hybrid_article_search_only_article_ids_print(
        self,
        query: str,
        final_limit: int = 10,
        candidate_limit: int = 50
    ) -> list[list[int]]:
        results = self.reranking(query, final_limit, candidate_limit)

        for result in results:
            print(result["article_id"], end=" ")
        
        print()

    def hybrid_article_search_print(
        self,
        query: str,
        final_limit: int = 10,
        candidate_limit: int = 50
    ) -> None:
        results = self.reranking(query, final_limit, candidate_limit)

        for result in results:
            print(result["score"])
            print(result["title"])
            print(result["body"][:200])
            print()