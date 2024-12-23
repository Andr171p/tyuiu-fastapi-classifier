from typing import List, Union, Self

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import settings


class StandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self._scaler = joblib.load(settings.ml.scaler)

    def fit(
            self,
            X: Union[pd.DataFrame, List[List[float]]],
            y=None
    ) -> Self:
        self._scaler.fit(X)
        return self

    def transform(
            self,
            X: Union[pd.DataFrame, List[List[float]]]
    ) -> pd.DataFrame:
        scaled = self._scaler.transform(X)
        scaled = pd.DataFrame(
            data=scaled,
            columns=X.columns
        )
        return scaled
