from typing import List
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.schemas.vision_schemas import (
    EmbeddingQueryResponse,
    VisionPredictionResponse,
)
from app.services.vision_service import get_vision_service

router = APIRouter(prefix="/vision", tags=["Vision AI"])


@router.post(
    "/autocomplete",
    response_model=VisionPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract physical attributes and visual embedding for pet registration",
)
async def autocomplete_pet_attributes(
    files: List[UploadFile] = File(
        ...,
        description="List of pet photos (1 to 6 images)",
    ),
) -> VisionPredictionResponse:
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one image file is required.",
        )

    service = get_vision_service()
    images_bytes: List[bytes] = []

    for file in files[:6]:
        content = await file.read()
        if content:
            images_bytes.append(content)

    if not images_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded files are empty.",
        )

    try:
        return service.predict_attributes(images_bytes)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


@router.post(
    "/embedding",
    response_model=EmbeddingQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Validate pet photo and extract 384-d normalized embedding for similarity search",
)
async def extract_query_embedding(
    file: UploadFile = File(
        ...,
        description="Single photo of a dog or cat to search similar pets",
    ),
) -> EmbeddingQueryResponse:
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    service = get_vision_service()
    try:
        return service.extract_search_embedding(content)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )
