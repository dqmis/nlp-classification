from typing import Any, List, Optional

from nltk.stem import SnowballStemmer
from sklearn.base import BaseEstimator, TransformerMixin


class Stemmer(BaseEstimator, TransformerMixin):
    def __init__(self, language: str = "english"):
        self._stemmer = SnowballStemmer(language)
        self.language = language

    def fit(self, x: List[str], y: Optional[List[Any]] = None) -> "Stemmer":
        return self

    def transform(self, x: List[str], y: Optional[List[Any]] = None) -> List[str]:
        return [self._stemmer.stem(word) for word in x]
