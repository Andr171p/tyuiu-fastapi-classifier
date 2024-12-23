from typing import List, Self

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import settings


class OneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self._features = settings.ml.features
        self._ohe = joblib.load(settings.ml.ohe)

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
        return X_transformed
