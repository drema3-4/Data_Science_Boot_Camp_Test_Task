from sentence_transformers import CrossEncoder

def make_reranker(config):
    return CrossEncoder(
        config.name,
        max_length=config.max_length,
        device=config.device
    )

class Reranker:
    def __init__(self, model):
        self.model = model

    def score(self, query: str, documents: list[str]) -> list[float]:
        pairs = [(query, document) for document in documents]
        return self.model.predict(pairs)