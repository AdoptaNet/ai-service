import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.recommendation_service import get_recommendation_service


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_adopter():
    service = get_recommendation_service()
    return service.modelo.ejemplo()


def test_recommendation_service_initialization():
    service = get_recommendation_service()
    assert service.modelo is not None
    assert service.modelo.config["nombre"] == "adoptanet-recomendador"
    assert len(service.modelo.catalogo) > 0


def test_recommendation_match_endpoint(client, sample_adopter):
    response = client.post(
        "/v1/recommendations/match",
        json={
            "adopter": sample_adopter,
            "k": 3,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "recomendaciones" in data
    assert len(data["recomendaciones"]) == 3
    assert data["n_viables"] > 0

    first = data["recomendaciones"][0]
    assert "pet_id" in first
    assert "score" in first
    assert "compatibilidad" in first
    assert "motivos" in first
    assert isinstance(first["motivos"], list)


def test_recommendation_dossier_endpoint(client, sample_adopter):
    # Usar una mascota del catálogo existente
    service = get_recommendation_service()
    first_pet_id = str(service.modelo.catalogo.index[0])

    response = client.post(
        "/v1/recommendations/dossier",
        json={
            "adopter": sample_adopter,
            "pet_id": first_pet_id,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["pet_id"] == first_pet_id
    assert "verdict" in data
    assert "compatibility" in data
    assert "interest_probability" in data
    assert "positive_factors" in data
    assert "alerts" in data
    assert "negative_factors" in data
    assert "violated_rules" in data


def test_health_check_includes_recommendations(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "recommendation_model" in data
    assert data["recommendation_model"]["ready"] is True
