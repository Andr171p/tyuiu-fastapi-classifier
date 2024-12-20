import pandas as pd
from typing import List, Self
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

from src.config import settings
from src.utils import load_pkl


class OHE(BaseEstimator, TransformerMixin):
    def __init__(self, features: List[str]) -> None:
        self._features = features
        self._ohe: OneHotEncoder = load_pkl(
            filename=settings.transformers.ohe_path
        )

    def fit(
            self,
            X: pd.DataFrame,
            y=None
    ) -> Self:
        X_features = X[self._features]
        self._ohe.fit(X_features)
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X_features = X[self._features]
        encoded = self._ohe.transform(X_features)
        ohe_columns = self._ohe.get_feature_names_out().tolist()
        X_encoded = pd.DataFrame(
            data=encoded,
            index=X.index,
            columns=ohe_columns
        )
        X_dropped = X.drop(
            columns=self._features,
            axis=1
        )
        X_transformed = pd.concat([X_dropped, X_encoded], axis=1)
        print(X_transformed.values)
        return X_transformed
