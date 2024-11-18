import pickle
import pandas as pd
from typing import Self
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer

from src.config import settings


class LabelsImputer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self._imputer = pickle.load(
            file=settings.pipe.imputer_path
        )

    def fit(self, X: pd.DataFrame) -> Self:
        self._imputer.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        labeled = self._imputer.transform(X)
        return labeled
