import pandas as pd
import re
from bs4 import BeautifulSoup


class BaseDocumentProcessor:
    def __init__(self):
        pass

    def prepare_data(self, data_path: str) -> pd.DataFrame:
        data = pd.read_feather(data_path)

        data["body"] = data["body"].apply(lambda x: self.__html_to_words__(x))

        return data.copy(deep=True)

    def __html_to_words__(self, html: str) -> list[str]:
        soup = BeautifulSoup(html, "html.parser")

        # Удаляем содержимое служебных тегов
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        # Получаем обычный текст
        text = soup.get_text(separator=" ", strip=True)

        # Приводим к нижнему регистру
        text = text.lower()

        # Оставляем слова и числа
        words = re.findall(r"[а-яёa-z0-9]+", text)

        return " ".join(words)