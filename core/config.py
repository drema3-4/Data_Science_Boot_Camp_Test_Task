class StoreSettings:
    QDRANT_URL = "http://127.0.0.1:6333"
    TIMEOUT = 30.0
    COLLECTION_NAME = "base_collection"

    DENSE_MODEL_NAME = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    SPARSE_MODEL_NAME = "Qdrant/bm25"
    SPARSE_TEXT_EMBEDDING_LANGUAGE = "russian"
    SPARSE_TEXT_EMBEDDING_K = 1.2
    SPARSE_TEXT_EMBEDDING_B = 0.75
    SPARSE_TEXT_EMBEDDING_AVG_LEN = 256.0

    SPARSE_TITLE_VECTOR_NAME = "title_sparse"
    SPARSE_BODY_VECTOR_NAME = "body_sparse"
    DENSE_TITLE_VECTOR_NAME = "title_dense"
    DENSE_BODY_VECTOR_NAME = "body_dense"