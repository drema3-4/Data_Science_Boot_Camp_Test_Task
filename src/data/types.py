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
class CalibrationSample:
    query: str
    ground_truth: list[int]

@dataclass
class TestSample:
    query_id: int
    query_text: str