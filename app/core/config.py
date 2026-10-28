from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "AdoptaNet ML Service"
    VERSION: str = "2.1.0"
    API_V1_STR: str = "/v1"

    # Model config
    MODEL_DIR: Path = Path(__file__).resolve().parent.parent / "models" / "vision"
    MODEL_FILENAME: str = "adoptanet_vision_dino.onnx"
    CONFIG_FILENAME: str = "model_config.json"
    RECOMMENDATION_MODEL_DIR: Path = Path(__file__).resolve().parent.parent / "models" / "recommendations"

    # Thresholds
    MIN_SPECIES_CONFIDENCE: float = 0.80
    MIN_COSINE_SIMILARITY: float = 0.35

    # Host & Port
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    model_config = SettingsConfigDict(case_sensitive=True)

settings = Settings()
