from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.common.model_description import (
    ModelSource,
    PoolingType,
)
import torch
from sentence_transformers import CrossEncoder

from core.config import StoreSettings


class ModelsManager:
    def __init__(self):
        self.__init_gen_candidate_models__()

        self.reranking_model_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__init_reranking_model__()        

    def __init_gen_candidate_models__(self):
        TextEmbedding.add_custom_model(
            model=StoreSettings.GEN_CANDIDATE_DENSE_MODEL_NAME,
            pooling=PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(
                hf=StoreSettings.GEN_CANDIDATE_DENSE_MODEL_NAME,
            ),
            dim=StoreSettings.GEN_CANDIDATE_DENSE_MODEL_DIM,
            model_file=StoreSettings.GEN_CANDIDATE_DENSE_MODEL_MODEL_FILE
        )

        self.gen_candidate_dense_model = TextEmbedding(
            model_name=StoreSettings.GEN_CANDIDATE_DENSE_MODEL_NAME
        )

        self.title_sparse_model = SparseTextEmbedding(
            model_name=StoreSettings.SPARSE_MODEL_NAME,

            # Русские стоп-слова и Snowball stemmer для русского языка
            language=StoreSettings.SPARSE_TEXT_EMBEDDING_LANGUAGE,

            # Стандартные параметры BM25
            k=StoreSettings.SPARSE_TEXT_EMBEDDING_K,
            b=StoreSettings.SPARSE_TEXT_EMBEDDING_B,

            # Желательно подобрать под среднюю длину документов в корпусе
            avg_len=StoreSettings.TITLE_SPARSE_TEXT_EMBEDDING_AVG_LEN
        )

        self.body_sparse_model = SparseTextEmbedding(
            model_name=StoreSettings.SPARSE_MODEL_NAME,

            # Русские стоп-слова и Snowball stemmer для русского языка
            language=StoreSettings.SPARSE_TEXT_EMBEDDING_LANGUAGE,

            # Стандартные параметры BM25
            k=StoreSettings.SPARSE_TEXT_EMBEDDING_K,
            b=StoreSettings.SPARSE_TEXT_EMBEDDING_B,

            # Желательно подобрать под среднюю длину документов в корпусе
            avg_len=StoreSettings.BODY_SPARSE_TEXT_EMBEDDING_AVG_LEN
        )

    def __init_reranking_model__(self):
        self.reranking_model = CrossEncoder(
            StoreSettings.RERANKING_MODEL_NAME,
            max_length=StoreSettings.RERANKING_MODEL_MAX_LENGTH,
            device=self.reranking_model_device
        )

    def get_gen_candidate_dense_model(self):
        return self.gen_candidate_dense_model
    
    def get_title_sparse_model(self):
        return self.title_sparse_model
    
    def get_body_sparse_model(self):
        return self.body_sparse_model
    
    def get_reranking_model(self):
        return self.reranking_model
    
    def get_reranking_model_device(self):
        return self.reranking_model_device