import re
from bs4 import BeautifulSoup

from data.types import RawArticle, BaseProcessedArticle


def plain_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    plain_text = soup.get_text(separator=" ", strip=True)

    return plain_text

def lexical_text_from_plain_text(plain_text: str) -> str:
    text = plain_text.lower()

    words = re.findall(r"[а-яёa-z0-9]+", text)
    lexical_text = " ".join(words)

    return lexical_text

def base_article_processor(raw_articles: list[RawArticle]) -> list[BaseProcessedArticle]:
    base_processed_articles = [
        BaseProcessedArticle(
            article_id=raw_article.article_id,
            title=raw_article.title,
            body_plain=plain_text_from_html(raw_article.body_html),
            body_lexical=lexical_text_from_plain_text(
                plain_text_from_html(raw_article.body_html)
            )
        )
        for raw_article in raw_articles
    ]

    return base_processed_articles