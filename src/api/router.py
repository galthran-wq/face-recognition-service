from fastapi import APIRouter

from src.api.endpoints import faces, health

router = APIRouter()
router.include_router(health.router, tags=["health"])
router.include_router(faces.router)
