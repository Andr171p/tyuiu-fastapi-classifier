from pathlib import Path
from pydantic_settings import BaseSettings


BASE_DIR: Path = Path(__file__).resolve().parent.parent


class PipesSettings(BaseSettings):
    ohe_path: Path = BASE_DIR / "pipes" / "one-hot-encoder.pkl"
    scaler_path: Path = BASE_DIR / "pipes" / "standard-scaler.pkl"
    imputer_path: Path = BASE_DIR / "pipes" / "binary-imputer.dill"


class ModelsSettings(BaseSettings):
    classifier_path: Path = BASE_DIR / "models" / "random-forrest-model.joblib"


class Settings(BaseSettings):
    pipe: PipesSettings = PipesSettings()
    ml: ModelsSettings() = ModelsSettings()


settings = Settings()
