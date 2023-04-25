from typing import Any, Dict, List, Optional

import eli5
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.models.components.stemmer import Stemmer

_classifiers = {
    "logreg": LogisticRegression,
    "xgboost": XGBClassifier,
    "nb": BernoulliNB,
}


def _parse_search_to_model_params(params: Dict[str, Any]) -> Dict[str, Any]:
    filtered_dict: Dict[str, Any] = {}
    for key, value in params.items():
        if "classifyer__" in key:
            filtered_dict[key.split("classifyer__")[-1]] = value
    return filtered_dict


def _parse_model_params_to_search(params: Dict[str, Any]) -> Dict[str, Any]:
    filtered_dict: Dict[str, Any] = {}
    for key, value in params.items():
        filtered_dict["classifyer__" + key] = value
    return filtered_dict


class MlClassifier:
    def __init__(
        self, language: Optional[str] = "english", classifier_name: str = "logreg"
    ):
        self.language = language
        self.classifier_name = classifier_name
        self.pipeline = self._define_pipeline()

    def hyperparam_search(
        self,
        X: List[str],
        y: List[int],
        search_params: Dict[str, Any],
        n_iter: int = 100,
        cv: int = 5,
        random_state: int = 42,
    ) -> None:
        """
        Search for the best parameters for the classifier.
        To pass search_params, define them with prefix "classifyer__" and
        pass them as a dictionary.
        """

        search = RandomizedSearchCV(
            self.pipeline,
            _parse_model_params_to_search(search_params),
            n_iter=n_iter,
            cv=cv,
            random_state=random_state,
        )
        search.fit(X, y)

        self.pipeline = self._define_pipeline(
            _parse_search_to_model_params(search.best_estimator_.get_params())
        )

    def _define_pipeline(
        self,
        classifier_params: Optional[Dict[str, Any]] = None,
    ) -> Pipeline:
        cv_params = {}
        if self.language:
            cv_params["stop_words"] = self.language
        if classifier_params is None:
            classifier = _classifiers[self.classifier_name]()
        else:
            classifier = _classifiers[self.classifier_name](**classifier_params)
        return Pipeline(
            [
                ("stemmer", Stemmer()),
                ("cv", CountVectorizer(**cv_params)),
                ("classifyer", classifier),
            ]
        )

    def get_feature_importance(self, top: int = 30) -> eli5.base.Explanation:
        return eli5.explain_weights(
            self.pipeline.named_steps["classifyer"],
            top=top,
            feature_names=self.pipeline.named_steps["cv"].get_feature_names_out(),
        )

    def fit(self, X: List[str], y: List[int]) -> None:
        self.pipeline.fit(X, y)

    def predict(self, X: List[str]) -> List[int]:
        predictions: List[int] = self.pipeline.predict(X)
        return predictions
