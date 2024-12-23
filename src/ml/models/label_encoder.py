from typing import Self, Literal

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import settings


class LabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self._feature: Literal["gender"] = "gender"
        self._le = joblib.load(settings.ml.label)

    def fit(self, X: pd.DataFrame, y=None) -> Self:
        self._le.fit(X[self._feature])
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        labeled = self._le.transform(X[self._feature])
        X[self._feature] = labeled
        return X
