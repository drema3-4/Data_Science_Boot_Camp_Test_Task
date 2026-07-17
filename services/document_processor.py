import pandas as pd
import re
from bs4 import BeautifulSoup


class BaseDocumentProcessor:
    def __init__(self):
        pass

    def prepare_data(self, data_path: str) -> pd.DataFrame:
        data = pd.read_feather(data_path)

        data["body_html"] = data["body"]
        data["body_plain"] = data["body"].apply(lambda x: self.__get_text_from_html__(x))
        data["body_lexical"] = data["body"].apply(lambda x: self.__leave_only_text_and_numbers__(x))

        return data.copy(deep=True)

    def __get_text_from_html__(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")

        # Удаляем содержимое служебных тегов
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        # Получаем обычный текст
        text = soup.get_text(separator=" ", strip=True)

        return " ".join(text)
    
    def __leave_only_text_and_numbers__(self, plain_text: str) -> str:
        text = plain_text.split(" ")

        # Приводим к нижнему регистру
        text = text.lower()

        # Оставляем слова и числа
        words = re.findall(r"[а-яёa-z0-9]+", text)

        return " ".join(words)