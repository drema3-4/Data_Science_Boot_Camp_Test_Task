from dataclasses import dataclass


@dataclass
class RawArticle:
    article_id: int
    title: str
    body_html: str

@dataclass
class BaseProcessedArticle:
    article_id: int
    title: str
    body_plain: str
    body_lexical: str

@dataclass
class ChunkProcessedArticle:
    point_id: int
    article_id: int
    chunk_id: int
    title: str
    body_plain: str
    body_lexical: str

@dataclass
class SectionChunkProcessedArticle:
    point_id: int
    article_id: int
    chunk_id: int
    point_type: str
    title: str
    body_plain: str
    body_lexical: str
    rerank_text: str

@dataclass
class CalibrationSample:
    query: str
    ground_truth: list[int]

@dataclass
class TestSample:
    query_id: int
    query_text: str