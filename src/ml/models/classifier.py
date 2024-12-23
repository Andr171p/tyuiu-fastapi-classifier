from typing import Union, Self

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from src.config import settings


class Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self) -> None:
        self._clf = joblib.load(settings.ml.clf)

    def fit(
            self,
            X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.DataFrame, np.ndarray]
    ) -> Self:
        self._clf.fit(
            X=X,
            y=y
        )
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> ...:
        check_is_fitted(self, "_clf")
        return self._clf.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> ...:
        check_is_fitted(self, "_clf")
        return self._clf.predict_proba(X)
