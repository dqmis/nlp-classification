from typing import List

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.review.models.components.stemmer import Stemmer

_classifiers = {
    "logreg": LogisticRegression(),
    "xgboost": XGBClassifier(),
    "nb": BernoulliNB(),
}


class MlClassifier:
    def __init__(self, language: str = "english", classifier: str = "logreg"):
        self.language = language
        self.pipeline = Pipeline(
            [
                ("stemmer", Stemmer()),
                ("cv", CountVectorizer(stop_words=language)),
                ("classifyer", _classifiers[classifier]),
            ]
        )

    def fit(self, X: List[str], y: List[int]) -> None:
        self.pipeline.fit(X, y)

    def predict(self, X: List[str]) -> List[int]:
        predictions: List[int] = self.pipeline.predict(X)
        return predictions
