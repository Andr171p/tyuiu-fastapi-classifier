import pandas as pd
from typing import Self
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import settings
from src.utils import load_dill


class LabelsImputer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self._imputer = load_dill(
            filename=settings.transformers.imputer_path
        )

    def fit(self, X: pd.DataFrame, y=None) -> Self:
        self._imputer.fit(X)
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        labeled = self._imputer.transform(X)
        return labeled
