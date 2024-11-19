from typing import List
from pathlib import Path
from pydantic_settings import BaseSettings


BASE_DIR: Path = Path(__file__).resolve().parent.parent


class TransformersSettings(BaseSettings):
    ohe_path: Path = BASE_DIR / "models" / "transformers" / "one-hot-encoder-new.pkl"
    scaler_path: Path = BASE_DIR / "models" / "transformers" / "standard-scaler-new.pkl"
    imputer_path: Path = BASE_DIR / "models" / "transformers" / "binary-imputer-new.dill"
    features: List[str] = ['education', 'study_form', 'reception_form', 'speciality']


class ClassifierSettings(BaseSettings):
    classifier_path: Path = BASE_DIR / "models" / "classifiers" / "random-forrest-classifier-new.joblib"


class Settings(BaseSettings):
    transformers: TransformersSettings = TransformersSettings()
    classifier: ClassifierSettings = ClassifierSettings()


settings = Settings()
