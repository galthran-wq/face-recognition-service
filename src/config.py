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
    face_max_batch_size: int = 64
    face_use_tensorrt: bool = False
    face_trt_cache_path: str = "/models/trt_cache"
    # TensorRT optimization-profile bounds for dynamic-batch models (recognition,
    # genderage). Without a profile the TRT EP builds a fresh engine for every
    # distinct batch size it sees (~30-75 s, in-process only, lost on restart) —
    # so batch endpoints stutter on each new face-count. A single profile lets
    # TRT build one engine spanning [1, max_batch], cached to disk and reused for
    # any batch in range. Recognition batch == total faces in a request, so
    # max_batch must cover the largest expected face count; larger calls are
    # chunked to max_batch. Set max_batch<=0 to disable profiles (legacy behavior).
    face_trt_max_batch: int = 256
    face_trt_opt_batch: int = 16
    # Pad-to-square fallback for frame-filling faces missed by RetinaFace anchors.
    face_pad_fallback_border_px: int = 100
    face_pad_fallback_fill: int = 128

    @field_validator("face_det_size", mode="before")
    @classmethod
    def parse_det_size(cls, v: object) -> tuple[int, int]:
        if isinstance(v, str):
            parts = v.split(",")
            return (int(parts[0].strip()), int(parts[1].strip()))
        return v  # type: ignore[return-value]


settings = Settings()
