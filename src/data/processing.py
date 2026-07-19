import re
from bs4 import BeautifulSoup

from data.types import RawArticle, BaseProcessedArticle, ChunkProcessedArticle, SectionChunkProcessedArticle


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

def split_text_by_words(text: str, chunk_size: int = 180, overlap: int = 40) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    if not words:
        return [""]

    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        if chunk_words:
            chunks.append(" ".join(chunk_words))

        if start + chunk_size >= len(words):
            break

    return chunks

def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def extract_html_blocks_with_headings(html: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "input"]):
        tag.decompose()

    blocks = []
    current_heading = ""

    useful_tags = ["h1", "h2", "h3", "p", "li", "td", "th", "label"]

    for tag in soup.find_all(useful_tags):
        text = normalize_spaces(tag.get_text(" ", strip=True))
        if not text:
            continue

        tag_classes = set(tag.get("class") or [])
        is_heading = (
            tag.name in {"h1", "h2", "h3"}
            or "spoiler-title" in tag_classes
            or "tab-label" in tag_classes
        )

        if is_heading:
            current_heading = text
            blocks.append((current_heading, text))
        else:
            blocks.append((current_heading, text))

    return blocks

def build_word_chunks_from_blocks(
    blocks: list[tuple[str, str]],
    chunk_size: int = 160,
    overlap: int = 30,
) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    texts = []

    for heading, block_text in blocks:
        if heading:
            texts.append(f"Раздел: {heading}. {block_text}")
        else:
            texts.append(block_text)

    full_text = " ".join(texts)
    return split_text_by_words(full_text, chunk_size=chunk_size, overlap=overlap)

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

def chunked_article_processor(
    raw_articles: list[RawArticle],
    chunk_size: int = 180,
    overlap: int = 40,
) -> list[ChunkProcessedArticle]:
    processed_chunks = []

    for raw_article in raw_articles:
        body_plain = plain_text_from_html(raw_article.body_html)
        chunks = split_text_by_words(body_plain, chunk_size, overlap)

        for chunk_id, chunk_text in enumerate(chunks):
            search_text = (
                f"Заголовок: {raw_article.title}\n"
                f"Текст: {chunk_text}"
            )

            processed_chunks.append(
                ChunkProcessedArticle(
                    point_id=raw_article.article_id * 10_000 + chunk_id,
                    article_id=raw_article.article_id,
                    chunk_id=chunk_id,
                    title=raw_article.title,
                    body_plain=search_text,
                    body_lexical=lexical_text_from_plain_text(search_text),
                )
            )

    return processed_chunks

def section_chunked_article_processor(
    raw_articles: list[RawArticle],
    chunk_size: int = 160,
    overlap: int = 30,
) -> list[SectionChunkProcessedArticle]:
    processed_points = []

    for raw_article in raw_articles:
        body_plain = plain_text_from_html(raw_article.body_html)
        blocks = extract_html_blocks_with_headings(raw_article.body_html)
        chunks = build_word_chunks_from_blocks(blocks, chunk_size, overlap)

        point_index = 0

        overview_text = (
            f"Заголовок: {raw_article.title}\n"
            f"Текст: {' '.join(body_plain.split()[:350])}"
        )

        processed_points.append(
            SectionChunkProcessedArticle(
                point_id=raw_article.article_id * 100_000 + point_index,
                article_id=raw_article.article_id,
                chunk_id=point_index,
                point_type="overview",
                title=raw_article.title,
                body_plain=overview_text,
                body_lexical=lexical_text_from_plain_text(overview_text),
                rerank_text=overview_text,
            )
        )
        point_index += 1

        for chunk_text in chunks:
            search_text = (
                f"Заголовок: {raw_article.title}\n"
                f"Текст: {chunk_text}"
            )

            processed_points.append(
                SectionChunkProcessedArticle(
                    point_id=raw_article.article_id * 100_000 + point_index,
                    article_id=raw_article.article_id,
                    chunk_id=point_index,
                    point_type="section_chunk",
                    title=raw_article.title,
                    body_plain=search_text,
                    body_lexical=lexical_text_from_plain_text(search_text),
                    rerank_text=search_text,
                )
            )
            point_index += 1

    return processed_points
