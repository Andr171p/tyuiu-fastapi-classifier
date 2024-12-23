from typing import Any, Self

from sklearn.pipeline import Pipeline

from src.ml.base import BaseClassifier
from src.ml.models import Classifier
from src.ml.pipeline.transformer import Transformer
from src.ml.utils import transform_data


class ModelPipeline(Transformer, BaseClassifier):
    def __init__(self) -> None:
        super().__init__()
        self._classifier = Pipeline([
            ("transformer", self._transformer),
            ("classifier", Classifier())
        ])

    def fit(self, X: Any, y: Any) -> Self:
        self._classifier.fit()
        return self._classifier

    def predict_proba(self, X: Any) -> Any:
        data = transform_data(X)
        print(data)
        y = self._classifier.predict_proba(X=data)
        return y
