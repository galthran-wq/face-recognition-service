from pydantic import BaseModel

# --- Request schemas ---


class ImageRequest(BaseModel):
    image_b64: str


class BatchRequest(BaseModel):
    images: list[ImageRequest]


# --- Shared schemas ---


class BoundingBoxSchema(BaseModel):
    x: float
    y: float
    width: float
    height: float


# --- Per-endpoint face schemas ---


class DetectFaceSchema(BaseModel):
    bbox: BoundingBoxSchema
    det_score: float


class EmbedFaceSchema(DetectFaceSchema):
    embedding: list[float]


class AnalyzeFaceSchema(EmbedFaceSchema):
    age: float | None = None
    gender: str | None = None
    race: str | None = None
    race_probs: dict[str, float] | None = None


# --- Single-image responses ---


class DetectResponse(BaseModel):
    faces: list[DetectFaceSchema]
    face_count: int


class EmbedResponse(BaseModel):
    faces: list[EmbedFaceSchema]
    face_count: int


class AnalyzeResponse(BaseModel):
    faces: list[AnalyzeFaceSchema]
    face_count: int


# --- Batch responses ---


class DetectBatchResultItem(BaseModel):
    index: int
    faces: list[DetectFaceSchema]
    face_count: int
    error: str | None = None


class EmbedBatchResultItem(BaseModel):
    index: int
    faces: list[EmbedFaceSchema]
    face_count: int
    error: str | None = None


class AnalyzeBatchResultItem(BaseModel):
    index: int
    faces: list[AnalyzeFaceSchema]
    face_count: int
    error: str | None = None


class DetectBatchResponse(BaseModel):
    results: list[DetectBatchResultItem]
    total_faces: int


class EmbedBatchResponse(BaseModel):
    results: list[EmbedBatchResultItem]
    total_faces: int


class AnalyzeBatchResponse(BaseModel):
    results: list[AnalyzeBatchResultItem]
    total_faces: int
