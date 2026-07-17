from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding, TextEmbedding

from core.config import StoreSettings


class StoreManager:
    """Инициализирует qdrant хранилище"""
    def __init__(self):
        self.client = QdrantClient(url=StoreSettings.QDRANT_URL, timeout=StoreSettings.TIMEOUT)

        self.__init_models__()
        
        if not self.client.collection_exists(StoreSettings.COLLECTION_NAME):
            self.__init_collection__()

    def __init_models__(self):
        self.dense_model = TextEmbedding(
            model_name=StoreSettings.DENSE_MODEL_NAME,
        )

        self.sparse_model = SparseTextEmbedding(
            model_name=StoreSettings.SPARSE_MODEL_NAME,

            # Русские стоп-слова и Snowball stemmer для русского языка
            language=StoreSettings.SPARSE_TEXT_EMBEDDING_LANGUAGE,

            # Стандартные параметры BM25
            k=StoreSettings.SPARSE_TEXT_EMBEDDING_K,
            b=StoreSettings.SPARSE_TEXT_EMBEDDING_B,

            # Желательно подобрать под среднюю длину документов в корпусе
            avg_len=StoreSettings.SPARSE_TEXT_EMBEDDING_AVG_LEN,
        )

    def __init_collection__(self):
        self.client.create_collection(
            collection_name=StoreSettings.COLLECTION_NAME,
            vectors_config={
                StoreSettings.DENSE_TITLE_VECTOR_NAME: models.VectorParams(
                    size=self.dense_model.embedding_size,
                    distance=models.Distance.COSINE
                ),
                StoreSettings.DENSE_BODY_VECTOR_NAME: models.VectorParams(
                    size=self.dense_model.embedding_size,
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
    
    def get_sparse_model(self):
        return self.sparse_model
    
    def get_dense_model(self):
        return self.dense_model