from typing import List
from pathlib import Path
from pydantic_settings import BaseSettings


BASE_DIR: Path = Path(__file__).resolve().parent.parent


class MLSettings(BaseSettings):
    ohe: Path = BASE_DIR / "trained_models" / "one-hot-encoder.joblib"
    scaler: Path = BASE_DIR / "trained_models" / "standard-scaler.joblib"
    label: Path = BASE_DIR / "trained_models" / "label-encoder.joblib"
    clf: Path = BASE_DIR / "trained_models" / "classifier.joblib"

    features: List[str] = ['direction']


class APISettings(BaseSettings):
    name: str = "Classifier API"
    prefix: str = "/api/v1"


class Settings(BaseSettings):
    ml: MLSettings = MLSettings()
    api_v1: APISettings = APISettings()


settings = Settings()
