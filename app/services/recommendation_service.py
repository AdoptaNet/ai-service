from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings

# Asegurar importación del recomendador exportado
rec_dir = Path(__file__).resolve().parent.parent / "models" / "recommendations"
if str(rec_dir) not in sys.path:
    sys.path.insert(0, str(rec_dir))

from recomendador import ModeloRecomendacion  # noqa: E402


class RecommendationService:
    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = model_dir or getattr(settings, "RECOMMENDATION_MODEL_DIR", rec_dir)
        self.modelo = ModeloRecomendacion(str(self.model_dir), verbose=False)

    def recommend(
        self,
        adopter: Dict[str, Any],
        pets: Optional[List[Dict[str, Any]]] = None,
        k: int = 5,
        exclude_pet_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Calcula las k mascotas más afines para el adoptante dado.
        Si se suministra una lista de mascotas (pets), actualiza el catálogo dinámicamente.
        """
        if pets:
            self.modelo.actualizar_catalogo(pets, verbose=False)

        res = self.modelo.recomendar(
            adoptante=adopter,
            k=k,
            excluir=exclude_pet_ids,
        )
        return res

    def get_dossier(
        self,
        adopter: Dict[str, Any],
        pet_id: str,
        pet_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Genera la ficha técnica de compatibilidad (XAI, pros, contras, alertas) para el albergue.
        """
        if pet_data and (pet_id not in self.modelo.catalogo.index):
            if "pet_id" not in pet_data:
                pet_data["pet_id"] = pet_id
            self.modelo.agregar_mascota(pet_data)

        ficha_data = self.modelo.ficha(adopter, pet_id, mostrar=False)
        return {
            "pet_id": ficha_data["pet_id"],
            "mascota": ficha_data["mascota"],
            "verdict": ficha_data.get("veredicto", "APTO PARA ENTREVISTA"),
            "compatibility": ficha_data["compatibilidad"],
            "interest_probability": ficha_data["p_interes"],
            "positive_factors": ficha_data["a_favor"],
            "negative_factors": ficha_data["en_contra"],
            "alerts": ficha_data["alertas"],
            "violated_rules": ficha_data["reglas_incumplidas"],
        }


# Global singleton
recommendation_service: Optional[RecommendationService] = None


def get_recommendation_service() -> RecommendationService:
    global recommendation_service
    if recommendation_service is None:
        recommendation_service = RecommendationService()
    return recommendation_service
