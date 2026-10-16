from fastapi import APIRouter

from app.api.v1.recommendations import router as recommendations_router
from app.api.v1.vision import router as vision_router

api_router = APIRouter()
api_router.include_router(vision_router)
api_router.include_router(recommendations_router)
