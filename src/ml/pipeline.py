from typing import Any, Optional
from sklearn.pipeline import Pipeline
from functools import singledispatchmethod

from src.ml.transformers.imputer import Imputer
from src.ml.transformers.labels import LabelsImputer
from src.ml.transformers.ohe import OHE
from src.ml.transformers.scaler import Scaler
from src.ml.transformers.classifier import ClassifierModel

from src.config import settings

features = ['education', 'study_form', 'reception_form', 'speciality']

pipeline = Pipeline([
    ("binary_imputer", LabelsImputer()),
    ("one_hot_encoder", OHE(features=features)),
    ("standard_scaler", Scaler())
])


from src.app.schemas.user import UserSchema

user = UserSchema(
    gender='М',
    hostel='да',
    gpa=4.5,
    priority=1,
    exams_points=310,
    bonus_points=0,
    education='Среднее общее образование',
    study_form='Очная',
    reception_form='Общий конкурс',
    speciality='12.03.01 Приборостроение'
)


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


p = ModelPipeline()
p.create_pipeline()
print(p.predict(user.model_dump()))