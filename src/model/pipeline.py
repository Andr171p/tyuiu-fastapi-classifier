from typing import Any, Optional
from sklearn.pipeline import Pipeline
from functools import singledispatchmethod

from src.model.transformers.imputer import Imputer
from src.model.transformers.labels import LabelsImputer
from src.model.transformers.ohe import OHE
from src.model.transformers.scaler import Scaler
from src.model.transformers.classifier import ClassifierModel
from src.config import settings


class ModelPipeline:
    def __init__(self) -> None:
        self._pipeline: Optional[Pipeline] = None

    def create_pipeline(self) -> None:
        transformers = Pipeline([
            ("labels_imputer", LabelsImputer()),
            ("one_hot_encoder", OHE(features=settings.transformers.features)),
            ("scaler", Scaler())
        ])
        self._pipeline = Pipeline([
            ("transformers", transformers),
            ("classifier", ClassifierModel())
        ])

    @singledispatchmethod
    def predict(self, X) -> Any:
        raise NotImplementedError(f"Method 'predict' not implemented for type {type(X)}")

    @predict.register
    def _(self, X: dict) -> ...:
        imputer = Imputer()
        if self._pipeline is None:
            raise ValueError("Pipeline is not created")
        x = imputer.transform(X)
        y = self._pipeline.predict_proba(x)
        return y

    @predict.register
    def _(self, X: list) -> ...:
        imputer = Imputer()
        if self._pipeline is None:
            raise ValueError("Pipeline is not created")
        x = imputer.transform(X)
        y = self._pipeline.predict_proba(x)
        return y
