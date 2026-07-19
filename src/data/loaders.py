import pandas as pd

from data.types import RawArticle, CalibrationSample, TestSample


def load_raw_articles(data_path: str) -> list[RawArticle]:
    data = pd.read_feather(data_path)

    raw_articles: list[RawArticle] = [
        RawArticle(
            article_id=article_id,
            title=title,
            body_html=body_html
        )
        for article_id, title, body_html in zip(
            data["article_id"].to_list(),
            data["title"].to_list(),
            data["body"].to_list()
        )
    ]

    return raw_articles

def load_calibration_samples(data_path: str) -> list[CalibrationSample]:
    data = pd.read_feather(data_path)

    calibration_samples: list[CalibrationSample] = [
        CalibrationSample(
            query=query,
            ground_truth=list(map(int, ground_truth.split()))
        )
        for query, ground_truth in zip(
            data["query_text"].to_list(),
            data["ground_truth"].to_list()
        )
    ]

    return calibration_samples

def load_test_samples(data_path: str) -> list[TestSample]:
    data = pd.read_feather(data_path)

    test_samples = [
        TestSample(
            query_id=query_id,
            query_text=query_text
        )
        for query_id, query_text in zip(
            data["query_id"].to_list(),
            data["query_text"].to_list()
        )
    ]

    return test_samples