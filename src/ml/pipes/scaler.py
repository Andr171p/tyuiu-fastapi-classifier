import numpy as np
import pandas as pd
from typing import List, Union, Self
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from src.config import settings
from src.utils import load_pkl


class Scaler(BaseEstimator, TransformerMixin):
    def __init__(self, copy: bool = True) -> None:
        self._scaler: StandardScaler = load_pkl(
            filename=settings.pipe.scaler_path
        )

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
    ) -> np.ndarray:
        scaled = self._scaler.transform(X)
        return scaled
