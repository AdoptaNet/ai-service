from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

from app.core.config import settings
from app.schemas.vision_schemas import (
    AttributeOption,
    AttributeSuggestion,
    AttributesPrediction,
    EmbeddingQueryResponse,
    EmbeddingResult,
    ModelMetadata,
    MultipleAttributeResult,
    SingleAttributeResult,
    VisionPredictionResponse,
)

ImageBytesOrPIL = Union[bytes, Image.Image]


def open_image(img: ImageBytesOrPIL) -> Image.Image:
    if isinstance(img, Image.Image):
        im = img
    elif isinstance(img, (bytes, bytearray)):
        im = Image.open(io.BytesIO(img))
    else:
        im = Image.open(img)
    im = ImageOps.exif_transpose(im)
    return im.convert("RGB")


def preprocess_image(im: Image.Image, resize: int, crop: int, mean: List[float], std: List[float]) -> np.ndarray:
    w, h = im.size
    if w <= h:
        nw, nh = resize, int(resize * h / w)
    else:
        nw, nh = int(resize * w / h), resize
    im = im.resize((nw, nh), Image.BILINEAR)
    left = int(round((nw - crop) / 2.0))
    top = int(round((nh - crop) / 2.0))
    im = im.crop((left, top, left + crop, top + crop))
    x = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
    std_arr = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
    return ((x - mean_arr) / std_arr).astype(np.float32)


class VisionService:
    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = model_dir or settings.MODEL_DIR
        config_path = self.model_dir / settings.CONFIG_FILENAME
        model_path = self.model_dir / settings.MODEL_FILENAME

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        self.config = json.loads(config_path.read_text(encoding="utf-8"))

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )

        pre = self.config["preprocesamiento"]
        self._pre = (
            int(pre["resize_lado_corto"]),
            int(pre["crop_centro"]),
            pre["mean"],
            pre["std"],
        )
        self._input_name = self.config["entrada"]["nombre"]
        self._attributes = self.config["atributos"]
        self._emb_spec = self.config.get("embedding")
        self._output_names = [a["salida"] for a in self._attributes.values()] + (
            [self._emb_spec["salida"]] if self._emb_spec else []
        )

    def _prepare_tensors(
        self, images: Iterable[ImageBytesOrPIL]
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        images_list = list(images)[: self.config.get("max_imagenes", 10)]
        tensors, discarded = [], []
        for i, img in enumerate(images_list):
            try:
                tensors.append(preprocess_image(open_image(img), *self._pre))
            except Exception as e:
                discarded.append({"index": i, "reason": f"{type(e).__name__}: {str(e)}"})

        if not tensors:
            raise ValueError("No valid image could be processed.")

        return np.stack(tensors).astype(np.float32), discarded

    @staticmethod
    def _normalize_mean_vector(vectors: np.ndarray) -> np.ndarray:
        mean_vec = vectors.mean(axis=0)
        norm = np.linalg.norm(mean_vec) + 1e-12
        return mean_vec / norm

    @staticmethod
    def _build_options(classes: List[Dict[str, Any]], probs: np.ndarray) -> List[AttributeOption]:
        sorted_indices = np.argsort(-probs)
        return [
            AttributeOption(
                value=classes[i]["valor"],
                label=classes[i]["etiqueta"],
                probability=round(float(probs[i]), 3),
            )
            for i in sorted_indices
        ]

    def _process_single_attribute(
        self, spec: Dict[str, Any], probs: np.ndarray
    ) -> SingleAttributeResult:
        options = self._build_options(spec["clases"], probs)
        top = options[0]
        is_confident = float(probs.max()) >= spec["umbral"]
        return SingleAttributeResult(
            value=top.value if is_confident else None,
            label=top.label if is_confident else None,
            confidence=top.probability,
            autocompleted=bool(is_confident),
            suggestion=AttributeSuggestion(value=top.value, label=top.label),
            options=options,
            note=spec.get("nota"),
        )

    def _process_multiple_attribute(
        self, spec: Dict[str, Any], probs: np.ndarray
    ) -> MultipleAttributeResult:
        classes = spec["clases"]
        thresholds = np.array(spec["umbrales"], dtype=np.float32)
        sorted_indices = [i for i in np.argsort(-probs)]
        selected = [i for i in sorted_indices if probs[i] >= thresholds[i]][: spec.get("max_seleccion", 3)]
        
        suggested = selected or (
            [i for i in sorted_indices if probs[i] >= 0.5][: spec.get("max_seleccion", 3)]
            or sorted_indices[:1]
        )

        selected_values = [classes[i]["valor"] for i in selected] or None
        selected_labels = [classes[i]["etiqueta"] for i in selected] or None
        suggested_values = [classes[i]["valor"] for i in suggested]
        suggested_labels = [classes[i]["etiqueta"] for i in suggested]

        min_conf = round(float(min(probs[i] for i in suggested)), 3) if suggested else 0.0

        return MultipleAttributeResult(
            value=selected_values,
            label=selected_labels,
            confidence=min_conf,
            autocompleted=bool(selected),
            suggestion=AttributeSuggestion(value=suggested_values, label=suggested_labels),
            options=self._build_options(classes, probs),
            note=spec.get("nota"),
        )

    def predict_attributes(
        self, images: List[ImageBytesOrPIL], include_embedding: bool = True
    ) -> VisionPredictionResponse:
        t0 = time.perf_counter()
        tensors, discarded = self._prepare_tensors(images)
        outputs = self.session.run(self._output_names, {self._input_name: tensors})

        attribute_probs = {
            name: out.mean(axis=0) for name, out in zip(self._attributes.keys(), outputs)
        }

        species_res = self._process_single_attribute(
            self._attributes["especie"], attribute_probs["especie"]
        )
        fur_res = self._process_single_attribute(
            self._attributes["pelaje"], attribute_probs["pelaje"]
        )
        size_res = self._process_single_attribute(
            self._attributes["tamano"], attribute_probs["tamano"]
        )
        colors_res = self._process_multiple_attribute(
            self._attributes["colores"], attribute_probs["colores"]
        )

        attributes_pred = AttributesPrediction(
            species=species_res,
            fur=fur_res,
            size=size_res,
            colors=colors_res,
        )

        autocompleted = []
        review = []
        for name, attr in [
            ("species", species_res),
            ("fur", fur_res),
            ("size", size_res),
            ("colors", colors_res),
        ]:
            if attr.autocompleted:
                autocompleted.append(name)
            else:
                review.append(name)

        embedding_res = None
        if self._emb_spec and include_embedding:
            norm_vector = self._normalize_mean_vector(outputs[-1])
            embedding_res = EmbeddingResult(
                vector=[round(float(v), 6) for v in norm_vector],
                dimension=int(norm_vector.shape[0]),
                version=self.config["version"],
            )

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        return VisionPredictionResponse(
            model=ModelMetadata(name=self.config["nombre"], version=self.config["version"]),
            images_processed=int(tensors.shape[0]),
            images_discarded=discarded,
            attributes=attributes_pred,
            autocompleted_fields=autocompleted,
            review_fields=review,
            manual_fields=self.config.get("campos_manuales", []),
            embedding=embedding_res,
            inference_time_ms=elapsed_ms,
        )

    def extract_search_embedding(
        self, image: ImageBytesOrPIL
    ) -> EmbeddingQueryResponse:
        t0 = time.perf_counter()
        tensors, _ = self._prepare_tensors([image])
        outputs = self.session.run(self._output_names, {self._input_name: tensors})

        species_probs = outputs[0][0]  # first batch element
        species_spec = self._attributes["especie"]
        species_res = self._process_single_attribute(species_spec, species_probs)

        confidence = species_res.confidence
        species_val = species_res.suggestion.value if isinstance(species_res.suggestion.value, str) else species_res.suggestion.value[0]

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        # Confidence check for dog/cat
        if confidence < settings.MIN_SPECIES_CONFIDENCE:
            return EmbeddingQueryResponse(
                valid=False,
                message="No dog or cat was clearly detected in the image.",
                detected_species=species_val,
                species_confidence=confidence,
                vector=None,
                dimension=384,
                version=self.config["version"],
                inference_time_ms=elapsed_ms,
            )

        norm_vector = self._normalize_mean_vector(outputs[-1])
        return EmbeddingQueryResponse(
            valid=True,
            message="Dog or cat identified successfully.",
            detected_species=species_val,
            species_confidence=confidence,
            vector=[round(float(v), 6) for v in norm_vector],
            dimension=int(norm_vector.shape[0]),
            version=self.config["version"],
            inference_time_ms=elapsed_ms,
        )


# Global instance
vision_service: Optional[VisionService] = None


def get_vision_service() -> VisionService:
    global vision_service
    if vision_service is None:
        vision_service = VisionService()
    return vision_service
