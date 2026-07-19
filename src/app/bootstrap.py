from pathlib import Path

import torch
from fastembed.common.model_description import ModelSource

from data.loaders import load_calibration_samples, load_raw_articles, load_test_samples
from data.processing import base_article_processor, chunked_article_processor, section_chunked_article_processor
from evaluation.evaluator import Evaluator
from indexing.collection_schema import CollectionSchema, make_base_schema, make_chunked_body_schema, make_section_chunked_schema
from indexing.point_builder import ArticlePointBuilder
from indexing.qdrant_store import QdrantStore
from models.configs import DenseEmbedderConfig, RerankerConfig, SparseEmbedderConfig
from models.factory import ModelFactory
from retrieval.pipeline import SearchPipeline
from retrieval.retrievers import HybridQdrantRetriever


SRC_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = SRC_DIR / "data_sources" / "raw"
ARTICLES_PATH = RAW_DATA_DIR / "articles.f"
CALIBRATION_PATH = RAW_DATA_DIR / "calibration.f"
TEST_PATH = RAW_DATA_DIR / "test.f"
QDRANT_URL = "http://127.0.0.1:6333"
QDRANT_TIMEOUT = 30.0


def build_title_sparse_config() -> SparseEmbedderConfig:
    return SparseEmbedderConfig(
        model_name="Qdrant/bm25",
        language="russian",
        k=1.2,
        b=0.75,
        avg_len=4.0,
    )


def build_body_sparse_config() -> SparseEmbedderConfig:
    return SparseEmbedderConfig(
        model_name="Qdrant/bm25",
        language="russian",
        k=1.2,
        b=0.75,
        avg_len=719.0,
    )


def build_dense_config() -> DenseEmbedderConfig:
    return DenseEmbedderConfig(
        model="intfloat/multilingual-e5-small",
        sources=ModelSource(
            hf="intfloat/multilingual-e5-small",
        ),
        dim=384,
        model_file="onnx/model.onnx",
    )


def build_reranker_config() -> RerankerConfig:
    return RerankerConfig(
        name="Qwen/Qwen3-Reranker-0.6B",
        max_length=512,
        device=("cuda" if torch.cuda.is_available() else "cpu"),
    )


def build_base_collection_schema() -> CollectionSchema:
    dense_config = build_dense_config()

    return make_base_schema(
        build_title_sparse_config(),
        build_body_sparse_config(),
        dense_config,
    )


def build_qdrant_store() -> QdrantStore:
    return QdrantStore(QDRANT_URL, QDRANT_TIMEOUT)


def build_model_factory() -> ModelFactory:
    model_factory = ModelFactory()
    model_factory.reg_model(build_dense_config())

    return model_factory


def build_article_point_builder(model_factory: ModelFactory) -> ArticlePointBuilder:
    return ArticlePointBuilder(model_factory)


def build_hybrid_qdrant_retriever(
    qdrant_store: QdrantStore,
    model_factory: ModelFactory,
    schema: CollectionSchema,
) -> HybridQdrantRetriever:
    return HybridQdrantRetriever(
        qdrant_store,
        model_factory,
        schema,
    )


def build_search_pipeline(with_reranker: bool = True) -> SearchPipeline:
    schema = build_base_collection_schema()
    qdrant_store = build_qdrant_store()
    model_factory = build_model_factory()
    retriever = build_hybrid_qdrant_retriever(
        qdrant_store,
        model_factory,
        schema,
    )
    reranker = None

    if with_reranker:
        reranker = model_factory.make_model(build_reranker_config())

    return SearchPipeline(
        retriever=retriever,
        reranker=reranker,
        schema=schema,
    )


def build_evaluator(with_reranker: bool = True) -> Evaluator:
    return Evaluator(
        build_search_pipeline(with_reranker=with_reranker),
    )


def load_base_processed_articles():
    raw_articles = load_raw_articles(str(ARTICLES_PATH))

    return base_article_processor(raw_articles)


def load_base_calibration_samples():
    return load_calibration_samples(str(CALIBRATION_PATH))


def load_base_test_samples():
    return load_test_samples(str(TEST_PATH))


def build_chunked_body_collection_schema() -> CollectionSchema:
    dense_config = build_dense_config()

    return make_chunked_body_schema(
        build_title_sparse_config(),
        build_body_sparse_config(),
        dense_config,
    )


def load_chunked_body_processed_articles():
    raw_articles = load_raw_articles(str(ARTICLES_PATH))

    return chunked_article_processor(
        raw_articles,
        chunk_size=180,
        overlap=40,
    )


def build_chunked_body_search_pipeline(with_reranker: bool = True) -> SearchPipeline:
    schema = build_chunked_body_collection_schema()
    qdrant_store = build_qdrant_store()
    model_factory = build_model_factory()

    retriever = build_hybrid_qdrant_retriever(
        qdrant_store,
        model_factory,
        schema,
    )

    reranker = None
    if with_reranker:
        reranker = model_factory.make_model(build_reranker_config())

    return SearchPipeline(
        retriever=retriever,
        reranker=reranker,
        schema=schema,
    )


def build_chunked_body_evaluator(with_reranker: bool = True) -> Evaluator:
    return Evaluator(
        build_chunked_body_search_pipeline(with_reranker=with_reranker),
    )


def build_section_chunked_collection_schema() -> CollectionSchema:
    dense_config = build_dense_config()

    return make_section_chunked_schema(
        build_title_sparse_config(),
        build_body_sparse_config(),
        dense_config,
    )


def load_section_chunked_processed_articles():
    raw_articles = load_raw_articles(str(ARTICLES_PATH))

    return section_chunked_article_processor(
        raw_articles,
        chunk_size=160,
        overlap=30,
    )


def build_section_chunked_search_pipeline(
    with_reranker: bool = True
) -> SearchPipeline:
    schema = build_section_chunked_collection_schema()
    qdrant_store = build_qdrant_store()
    model_factory = build_model_factory()

    retriever = build_hybrid_qdrant_retriever(
        qdrant_store,
        model_factory,
        schema,
    )

    reranker = None
    if with_reranker:
        reranker = model_factory.make_model(build_reranker_config())

    return SearchPipeline(
        retriever=retriever,
        reranker=reranker,
        schema=schema,
    )


def build_section_chunked_evaluator(with_reranker: bool = True) -> Evaluator:
    return Evaluator(
        build_section_chunked_search_pipeline(with_reranker=with_reranker),
    )