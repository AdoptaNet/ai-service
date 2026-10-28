from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.schemas.recommendation_schemas import (
    RecommendationDossierRequest,
    RecommendationDossierResponse,
    RecommendationMatchRequest,
    RecommendationMatchResponse,
)
from app.services.recommendation_service import get_recommendation_service

router = APIRouter(prefix="/recommendations", tags=["Recommendations AI"])


@router.post(
    "/match",
    response_model=RecommendationMatchResponse,
    status_code=status.HTTP_200_OK,
    summary="Calcular ranking híbrido de compatibilidad entre adoptante y mascotas",
)
async def compute_recommendations(payload: RecommendationMatchRequest) -> RecommendationMatchResponse:
    """
    Ejecuta el pipeline híbrido de recomendación de AdoptaNet:
    - Nivel 0: Exclusión por reglas duras de bienestar animal.
    - Nivel 1a: Motor colaborativo EASE con promedio demográfico.
    - Nivel 1b: Compatibilidad ordinal (ajuste estilo de vida y hogar vs. requerimientos de la mascota).
    - Nivel 2: Ponderación gamma (0.9 compatibilidad + 0.1 motor).
    - Nivel 3: Reranking por larga estancia en albergue y generación de explicabilidad (motivos XAI y alertas).
    """
    service = get_recommendation_service()
    try:
        resultado = service.recommend(
            adopter=payload.adopter,
            pets=payload.pets,
            k=payload.k,
            exclude_pet_ids=payload.exclude_pet_ids,
        )
        return RecommendationMatchResponse(**resultado)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Error en los atributos del adoptante o mascota: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en el motor de inferencia de recomendaciones: {str(e)}",
        )


@router.post(
    "/dossier",
    response_model=RecommendationDossierResponse,
    status_code=status.HTTP_200_OK,
    summary="Generar ficha técnica detallada de compatibilidad (XAI, pros, contras, alertas) para el albergue",
)
async def compute_dossier(payload: RecommendationDossierRequest) -> RecommendationDossierResponse:
    """
    Evalúa un par específico (adoptante - mascota) para proveer al albergue
    los puntos a favor, en contra, nivel de compatibilidad y banderas rojas a verificar en la entrevista.
    """
    service = get_recommendation_service()
    try:
        ficha = service.get_dossier(
            adopter=payload.adopter,
            pet_id=payload.pet_id,
            pet_data=payload.pet_data,
        )
        return RecommendationDossierResponse(**ficha)
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Error en los atributos para generar el dossier: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error calculando la ficha de compatibilidad: {str(e)}",
        )
