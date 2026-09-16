from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    service_name: str = "product-category-service"
    service_port: int = 8000
    log_level: str = "INFO"

    redis_url: str = "redis://redis:6379/0"
    cache_ttl_seconds: int = 3600
    triton_url: str = "triton:8000"
    triton_model_name: str = "product_category"
    triton_timeout_seconds: float = 10.0

    vectorizer_path: Path = Path("/artifacts/vectorizer.joblib")
    metadata_path: Path = Path("/artifacts/metadata.json")
    event_list_key: str = "prediction_events"
    event_list_max_size: int = 10_000


settings = Settings()
