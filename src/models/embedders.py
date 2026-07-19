from fastembed import SparseTextEmbedding, TextEmbedding

def make_sparse_embedder(config):
    return SparseTextEmbedding(
        model_name=config.model_name,
        language=config.language,
        k=config.k,
        b=config.b,
        avg_len=config.avg_len
    )

def reg_dense_model(config):
    TextEmbedding.add_custom_model(
        model=config.model,
        pooling=config.pooling,
        normalization=config.normalization,
        sources=config.sources,
        dim=config.dim,
        model_file=config.model_file
    )

def make_dense_embedder(config):
    return TextEmbedding(
        model_name=config.model
    )

class SparseEmbedder:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts: list[str]):
        return self.model.embed(texts)

    def embed_query(self, query: str):
        return next(self.model.query_embed(query))

class DenseEmbedder:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts: list[str]):
        return self.model.passage_embed(texts)

    def embed_query(self, query: str):
        return next(self.model.query_embed(query))