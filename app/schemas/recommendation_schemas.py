from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RecommendedPetItem(BaseModel):
    posicion: int = Field(..., description="Posición en el ranking (1-based)")
    pet_id: str = Field(..., description="Identificador único de la mascota")
    mascota: str = Field(..., description="Descripción breve de la mascota")
    score: float = Field(..., description="Puntaje de afinidad global calculado por el modelo")
    p_interes: float = Field(..., description="Probabilidad estimada de interés del adoptante")
    compatibilidad: float = Field(..., description="Compatibilidad pura sin sesgo cultural (0 a 1)")
    motivos: List[str] = Field(default_factory=list, description="Motivos explicativos a favor (XAI)")
    alertas: List[str] = Field(default_factory=list, description="Alertas o advertencias para la entrevista de adopción")


class ExcludedPetItem(BaseModel):
    pet_id: str
    reglas: List[str]


class RecommendationMatchRequest(BaseModel):
    adopter: Dict[str, Any] = Field(..., description="Respuestas del cuestionario del adoptante según ESQUEMA_ADOPTANTE")
    pets: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Listado opcional de mascotas disponibles para actualizar el catálogo dinámico"
    )
    k: int = Field(default=5, ge=1, le=100, description="Cantidad máxima de recomendaciones solicitadas")
    exclude_pet_ids: Optional[List[str]] = Field(default=None, description="Lista de pet_id a excluir del ranking")


class RecommendationMatchResponse(BaseModel):
    modelo: Dict[str, str] = Field(..., description="Metadatos del modelo")
    adoptante: Optional[str] = Field(default=None, description="Identificador del adoptante si aplica")
    recomendaciones: List[RecommendedPetItem] = Field(default_factory=list, description="Lista ordenada de mascotas recomendadas")
    n_viables: int = Field(..., description="Total de mascotas viables que superaron las reglas duras")
    excluidas: List[ExcludedPetItem] = Field(default_factory=list, description="Mascotas excluidas por reglas duras")
    reglas_incumplidas: List[str] = Field(default_factory=list, description="Reglas incumplidas si no hubo viables")
    advertencia: Optional[str] = Field(default=None, description="Mensaje de advertencia si no hay candidatos idóneos")
    tiempo_ms: float = Field(..., description="Tiempo de inferencia en milisegundos")


class RecommendationDossierRequest(BaseModel):
    adopter: Dict[str, Any] = Field(..., description="Respuestas del cuestionario del adoptante")
    pet_id: str = Field(..., description="Identificador de la mascota a evaluar")
    pet_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Datos de la mascota si no reside en el catálogo cargado"
    )


class RecommendationDossierResponse(BaseModel):
    pet_id: str
    mascota: str
    verdict: str = Field(default="APTO PARA ENTREVISTA", description="Veredicto del modelo")
    compatibility: float = Field(..., description="Compatibilidad pura sin sesgo")
    interest_probability: float = Field(..., description="Probabilidad estimada de interés del adoptante")
    positive_factors: List[str] = Field(default_factory=list, description="Puntos a favor de la adopción")
    negative_factors: List[str] = Field(default_factory=list, description="Factores en contra identificados")
    alerts: List[str] = Field(default_factory=list, description="Puntos de atención para la entrevista")
    violated_rules: List[str] = Field(default_factory=list, description="Reglas duras de bienestar animal violadas")
