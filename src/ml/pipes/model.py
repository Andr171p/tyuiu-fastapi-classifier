import joblib
import numpy as np
import pandas as pd
from typing import Any, Union, Self
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from src.config import settings


class BinaryClassifierModel(BaseEstimator, ClassifierMixin):
    def __init__(self) -> None:
        self._model: RandomForestClassifier = joblib.load(
            filename=settings.ml.classifier_path
        )

    def fit(
            self,
            X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.DataFrame, np.ndarray]
    ) -> Self:
        self._model.fit(
            X=X,
            y=y
        )
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> ...:
        check_is_fitted(self, "_model")
        return self._model.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> ...:
        check_is_fitted(self, "_model")
        return self._model.predict_proba(X)

    def score(self, X, y):
        ...