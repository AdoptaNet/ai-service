from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class AttributeOption(BaseModel):
    value: str
    label: str
    probability: float


class AttributeSuggestion(BaseModel):
    value: Union[str, List[str]]
    label: Union[str, List[str]]


class SingleAttributeResult(BaseModel):
    value: Optional[str] = None
    label: Optional[str] = None
    confidence: float
    autocompleted: bool
    suggestion: AttributeSuggestion
    options: List[AttributeOption] = []
    note: Optional[str] = None


class MultipleAttributeResult(BaseModel):
    value: Optional[List[str]] = None
    label: Optional[List[str]] = None
    confidence: float
    autocompleted: bool
    suggestion: AttributeSuggestion
    options: List[AttributeOption] = []
    note: Optional[str] = None


class AttributesPrediction(BaseModel):
    species: SingleAttributeResult
    fur: SingleAttributeResult
    size: SingleAttributeResult
    colors: MultipleAttributeResult


class EmbeddingResult(BaseModel):
    vector: List[float]
    dimension: int = 384
    version: str = "2.1.0"


class ModelMetadata(BaseModel):
    name: str
    version: str


class VisionPredictionResponse(BaseModel):
    model: ModelMetadata
    images_processed: int
    images_discarded: List[Dict[str, Any]] = []
    attributes: AttributesPrediction
    autocompleted_fields: List[str] = []
    review_fields: List[str] = []
    manual_fields: List[str] = []
    embedding: Optional[EmbeddingResult] = None
    inference_time_ms: float


class EmbeddingQueryResponse(BaseModel):
    valid: bool
    message: Optional[str] = None
    detected_species: Optional[str] = None
    species_confidence: Optional[float] = None
    vector: Optional[List[float]] = None
    dimension: int = 384
    version: str = "2.1.0"
    inference_time_ms: float
