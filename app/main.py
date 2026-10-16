from contextlib import asynccontextmanager
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.api import api_router
from app.core.config import settings
from app.services.vision_service import get_vision_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load ONNX model into memory once
    service = get_vision_service()
    print(f"✔ Model loaded successfully: {service.config['nombre']} v{service.config['version']}")
    yield
    # Shutdown logic if needed
    print("✔ Shutting down AdoptaNet ML Service.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Machine Learning and Computer Vision inference microservice for AdoptaNet.",
    lifespan=lifespan,
)

# Enable CORS for frontend and backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", status_code=status.HTTP_200_OK, tags=["Health"])
async def health_check():
    service = get_vision_service()
    return {
        "status": "healthy",
        "service": "adoptanet-ml-service",
        "version": settings.VERSION,
        "vision_model": {
            "name": service.config["nombre"],
            "version": service.config["version"],
            "embedding_dimension": 384,
            "ready": True,
        },
    }


app.include_router(api_router, prefix=settings.API_V1_STR)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
