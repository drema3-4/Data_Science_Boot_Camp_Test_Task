from models.configs import SparseEmbedderConfig, DenseEmbedderConfig, RerankerConfig
from models.embedders import (
    make_sparse_embedder,
    make_dense_embedder,
    reg_dense_model,
    SparseEmbedder,
    DenseEmbedder
)
from models.rerankers import make_reranker, Reranker


class ModelFactory:
    def make_model(self, config):
        if isinstance(config, SparseEmbedderConfig):
            return SparseEmbedder(
                make_sparse_embedder(config)
            )

        if isinstance(config, DenseEmbedderConfig):
            return DenseEmbedder(
                make_dense_embedder(config)
            )

        if isinstance(config, RerankerConfig):
            return Reranker(
                make_reranker(config)
            )

        raise ValueError(f"Unknown model config: {config}")

    def reg_model(self, config):
        if isinstance(config, DenseEmbedderConfig):
            reg_dense_model(config)
        else:
            raise ValueError(f"Unknown model config: {config}")