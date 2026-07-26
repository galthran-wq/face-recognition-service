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
    # Max in-flight inference calls per process. ORT sessions are thread-safe;
    # values >1 let the CPU stages (decode, align, NMS) of one request overlap
    # the GPU stages of another instead of serializing the whole pipeline.
    face_inference_concurrency: int = 2
    # Hard cap on the CUDA EP arena, in GiB (0 = unlimited). The arena grows
    # under load and never shrinks; capping it keeps multi-instance-per-GPU
    # deployments from creeping into OOM.
    face_cuda_gpu_mem_limit_gb: float = 0.0
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
    # Experimental: capture TRT execution into CUDA graphs to cut kernel-launch
    # overhead on small models. Re-captures on every batch-shape change, so only
    # worth enabling for workloads with stable shapes — measure before keeping.
    face_trt_cuda_graph: bool = False
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
