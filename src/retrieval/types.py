from dataclasses import dataclass


@dataclass
class PipelineResponse:
    article_id: int
    score: float
    retrieval_score: float
    title: str
    body: str