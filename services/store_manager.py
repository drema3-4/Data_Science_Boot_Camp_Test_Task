from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.common.model_description import (
    ModelSource,
    PoolingType,
)

from core.config import StoreSettings
from services.models_manager import ModelsManager


class StoreManager:
    """Инициализирует qdrant хранилище"""
    def __init__(self, models_manager: ModelsManager):
        self.client = QdrantClient(url=StoreSettings.QDRANT_URL, timeout=StoreSettings.TIMEOUT)

        self.models_manager = models_manager
        self.gen_candidate_dense_model = self.models_manager.get_gen_candidate_dense_model()
        self.title_sparse_model = self.models_manager.get_title_sparse_model()
        self.body_sparse_model = self.models_manager.get_body_sparse_model()
        
        if not self.client.collection_exists(StoreSettings.COLLECTION_NAME):
            self.__init_collection__()

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