from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.common.model_description import (
    ModelSource,
    PoolingType,
)

from core.config import StoreSettings


class StoreManager:
    """Инициализирует qdrant хранилище"""
    def __init__(self):
        self.client = QdrantClient(url=StoreSettings.QDRANT_URL, timeout=StoreSettings.TIMEOUT)

        self.__init_models__()
        
        if not self.client.collection_exists(StoreSettings.COLLECTION_NAME):
            self.__init_collection__()

    def __init_models__(self):
        TextEmbedding.add_custom_model(
            model=StoreSettings.GEN_CANDIDATE_DENSE_MODEL_NAME,
            pooling=PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(
                hf=StoreSettings.GEN_CANDIDATE_DENSE_MODEL_NAME,
            ),
            dim=StoreSettings.GEN_CANDIDATE_DENSE_MODEL_DIM,
            model_file=StoreSettings.GEN_CANDIDATE_DENSE_MODEL_MODEL_FILE,
        )

        self.gen_candidate_dense_model = TextEmbedding(
            model_name=StoreSettings.GEN_CANDIDATE_DENSE_MODEL_NAME,
        )

        self.title_sparse_model = SparseTextEmbedding(
            model_name=StoreSettings.SPARSE_MODEL_NAME,

            # Русские стоп-слова и Snowball stemmer для русского языка
            language=StoreSettings.SPARSE_TEXT_EMBEDDING_LANGUAGE,

            # Стандартные параметры BM25
            k=StoreSettings.SPARSE_TEXT_EMBEDDING_K,
            b=StoreSettings.SPARSE_TEXT_EMBEDDING_B,

            # Желательно подобрать под среднюю длину документов в корпусе
            avg_len=StoreSettings.TITLE_SPARSE_TEXT_EMBEDDING_AVG_LEN,
        )

        self.body_sparse_model = SparseTextEmbedding(
            model_name=StoreSettings.SPARSE_MODEL_NAME,

            # Русские стоп-слова и Snowball stemmer для русского языка
            language=StoreSettings.SPARSE_TEXT_EMBEDDING_LANGUAGE,

            # Стандартные параметры BM25
            k=StoreSettings.SPARSE_TEXT_EMBEDDING_K,
            b=StoreSettings.SPARSE_TEXT_EMBEDDING_B,

            # Желательно подобрать под среднюю длину документов в корпусе
            avg_len=StoreSettings.BODY_SPARSE_TEXT_EMBEDDING_AVG_LEN,
        )

    def __init_collection__(self):
        self.client.create_collection(
            collection_name=StoreSettings.COLLECTION_NAME,
            vectors_config={
                StoreSettings.DENSE_TITLE_VECTOR_NAME: models.VectorParams(
                    size=self.gen_candidate_dense_model.embedding_size,
                    distance=models.Distance.COSINE
                ),
                StoreSettings.DENSE_BODY_VECTOR_NAME: models.VectorParams(
                    size=self.gen_candidate_dense_model.embedding_size,
                    distance=models.Distance.COSINE
                ),
            },
            sparse_vectors_config={
                StoreSettings.SPARSE_TITLE_VECTOR_NAME: models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=False
                    ),
                    modifier=models.Modifier.IDF
                ),
                StoreSettings.SPARSE_BODY_VECTOR_NAME: models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=False
                    ),
                    modifier=models.Modifier.IDF
                )
            }
        )

    def get_client(self):
        return self.client
    
    def get_title_sparse_model(self):
        return self.title_sparse_model
    
    def get_body_sparse_model(self):
        return self.body_sparse_model
    
    def get_gen_candidate_dense_model(self):
        return self.gen_candidate_dense_model