# AdoptaNet ML Service

Microservicio de Machine Learning e Inferencia de Visión Computacional para la plataforma AdoptaNet.

Desarrollado en **FastAPI** y acelerado con **ONNX Runtime (CPU Execution Provider)** para inferencias matriciales de baja latencia (~30–50 ms por imagen en CPU).

---

## Capacidades

1. **Autocompletado de Registro (`POST /v1/vision/autocomplete`)**:
   - Analiza de 1 a 6 fotografías de una mascota.
   - Extrae rasgos fenotípicos:
     - `species`: Perro o gato.
     - `fur`: Corto o largo.
     - `size`: Pequeño, mediano o grande (tamaño adulto estimado).
     - `colors`: Hasta 3 colores predominantes (negro, marrón, crema, gris, blanco).
   - Genera un vector de embedding normalizado ($R^{384}$) a partir del Vision Transformer (**DINOv2 ViT-S/14**).
   - Discrimina automáticamente los campos autocompletables con alta confianza respecto a los que requieren confirmación humana.

2. **Búsqueda por Similitud Visual (`POST /v1/vision/embedding`)**:
   - Recibe una imagen subida por un adoptante.
   - Valida la confianza de especie: si no se reconoce un perro o gato con confianza $\ge 80\%$, rechaza la búsqueda amigablemente.
   - Si es válida, retorna el vector normalizado de 384 dimensiones para que el backend ejecute la consulta coseno con `pgvector`.

3. **Recomendaciones de Compatibilidad (`POST /v1/recommendations/match`)**:
   - Módulo preparado para el algoritmo híbrido de afinidad adoptante-mascota.

---

## Ejecución Local

### Opción A: Entorno Virtual Python (Python 3.12)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Opción B: Con Docker

```bash
docker build -t adoptanet-ml-service .
docker run -p 8000:8000 adoptanet-ml-service
```

---

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health` | Chequeo de salud del servicio y verificación del modelo en memoria |
| `POST` | `/v1/vision/autocomplete` | Inferencia multi-foto para el formulario de registro de mascotas |
| `POST` | `/v1/vision/embedding` | Extracción de vector unitario de 384-d para búsqueda por imagen |
| `POST` | `/v1/recommendations/match` | Endpoint de compatibilidad y ranking híbrido |

Documentación interactiva disponible en: `http://localhost:8000/docs` (Swagger UI).
