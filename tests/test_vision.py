import io
from pathlib import Path
from PIL import Image
from app.services.vision_service import VisionService


def create_dummy_image() -> bytes:
    img = Image.new("RGB", (256, 256), color=(120, 80, 50))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_vision_service_loads_and_predicts():
    service = VisionService()
    dummy_bytes = create_dummy_image()

    result = service.predict_attributes([dummy_bytes])

    assert result.model.name == "adoptanet-vision"
    assert result.model.version == "2.1.0"
    assert result.images_processed == 1
    assert result.attributes.species.confidence > 0
    assert result.embedding is not None
    assert len(result.embedding.vector) == 384
    assert result.inference_time_ms > 0


def test_vision_service_embedding_query():
    service = VisionService()
    dummy_bytes = create_dummy_image()

    result = service.extract_search_embedding(dummy_bytes)

    assert result.dimension == 384
    assert result.version == "2.1.0"
    assert isinstance(result.valid, bool)
