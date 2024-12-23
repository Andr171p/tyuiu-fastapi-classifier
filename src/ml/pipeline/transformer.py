from typing import Union, List, Dict, Any

from sklearn.pipeline import Pipeline

from src.ml.base.transformer import BaseTransformer
from src.ml.models import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler
)


class Transformer(BaseTransformer):
    def __init__(self) -> None:
        self._transformer = Pipeline([
            ("label_encoder", LabelEncoder()),
            ("one_hot_encoder", OneHotEncoder()),
            ("standard_scaler", StandardScaler())
        ])

    def fit_transform(
            self,
            X: Union[List[Dict[str, Any]], Dict[str, Any]]
    ) -> Any:
        return self._transformer.fit_transform()

    def transform(
            self,
            X: Union[List[Dict[str, Any]], Dict[str, Any]]
    ) -> Any:
        return self._transformer.transform()
