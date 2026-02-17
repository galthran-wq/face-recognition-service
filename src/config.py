from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "python-service-template"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    cors_origins: list[str] = ["*"]
    metrics_enabled: bool = True

    # Face provider settings
    face_provider: str = "insightface"
    face_use_gpu: bool = False
    face_ctx_id: int = 0
    face_det_size: tuple[int, int] = (640, 640)
    face_model_name: str = "buffalo_l"
    face_model_dir: str = "~/.insightface"
    face_max_batch_size: int = 20

    @field_validator("face_det_size", mode="before")
    @classmethod
    def parse_det_size(cls, v: object) -> tuple[int, int]:
        if isinstance(v, str):
            parts = v.split(",")
            return (int(parts[0].strip()), int(parts[1].strip()))
        return v  # type: ignore[return-value]


settings = Settings()
