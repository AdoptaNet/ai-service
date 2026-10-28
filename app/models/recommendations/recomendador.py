"""
Adoptanet Recomendador - mascotas recomendadas para un adoptante a partir de su cuestionario.

Dependencias: numpy, pandas  (no necesita scikit-learn ni scipy; el entrenamiento está en
recomendador_hibrido_mascotas.ipynb, que escribe este archivo y exporta los parámetros).

    from recomendador import ModeloRecomendacion
    modelo = ModeloRecomendacion("export_recomendador")    # carga parámetros y catálogo una vez
    r = modelo(adoptante)                                  # top-k con motivos y alertas (y lo muestra)
    r = modelo.recomendar(adoptante, k=5)                  # solo el dict (lo que recibe el backend)
    modelo.ficha(adoptante, "pet_026")                     # ficha del par para el albergue
    modelo.recomendar_lote([a1, a2, ...], eta_carga=1.0)   # varios adoptantes, repartiendo la demanda
    modelo.puntuar(adoptantes)                             # matriz de scores (adoptantes x mascotas)
    modelo.agregar_mascota({...}); modelo.quitar_mascota("pet_026")   # catálogo, sin reentrenar

Niveles: 0 reglas duras (fuera del ML) -> 1a motor EASE (historial demográfico) -> 1b compatibilidad
(logística ordinal) -> 2 mezcla gamma -> 3 boost por larga estancia, disponibles y reparto de carga.
"""
from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd

# ---------------- esquema de entrada (única fuente de verdad: el notebook usa estas mismas definiciones) ----------------
# tipo: categoria (valores cerrados), multiple (varios valores separados por "|" o lista, o "indistinto"),
#       binario (0/1), entero, numero, texto (libre, se pasa a minúsculas). nulo=True admite None.
ESQUEMA_ADOPTANTE = {
    "departamento": {"tipo": "texto"},
    "zona_climatica": {"tipo": "categoria", "valores": ["costa", "sierra", "selva"]},
    "tipo_vivienda": {"tipo": "categoria", "valores": ["departamento", "casa", "casa_terreno"]},
    "tenencia_propia": {"tipo": "binario"},
    "permiso_mascotas": {"tipo": "binario"},
    "area_exterior": {"tipo": "binario"},
    "m2_exterior": {"tipo": "numero", "min": 0, "nulo": True},
    "cercado_seguro": {"tipo": "binario"},
    "n_integrantes": {"tipo": "entero", "min": 1},
    "presencia_ninos": {"tipo": "binario"},
    "edad_min_ninos": {"tipo": "entero", "min": 0, "max": 17, "nulo": True, "nota": "obligatorio si hay niños"},
    "otras_mascotas_perro": {"tipo": "binario"},
    "otras_mascotas_gato": {"tipo": "binario"},
    "horas_fuera_dia": {"tipo": "numero", "min": 0, "max": 24},
    "nivel_actividad": {"tipo": "categoria", "valores": ["sedentario", "moderado", "activo"]},
    "experiencia_previa": {"tipo": "categoria", "valores": ["ninguna", "alguna", "amplia"]},
    "presupuesto_mensual": {"tipo": "categoria", "valores": ["bajo", "medio", "alto"]},
    "tiempo_adiestramiento": {"tipo": "categoria", "valores": ["baja", "media", "alta"]},
    "pref_especie": {"tipo": "categoria", "valores": ["perro", "gato", "indistinto"]},
    "pref_tamano": {"tipo": "multiple", "valores": ["pequeno", "mediano", "grande"]},
    "pref_edad": {"tipo": "multiple", "valores": ["cachorro", "joven", "adulto", "senior"]},
    "pref_sexo": {"tipo": "categoria", "valores": ["macho", "hembra", "indistinto"]},
    "pref_energia": {"tipo": "entero", "min": 1, "max": 5},
    "apertura_nec_especiales": {"tipo": "binario"},
    "apertura_senior": {"tipo": "binario"},
}
_NIVELES = ["bajo", "medio", "alto"]
ESQUEMA_MASCOTA = {
    "especie": {"tipo": "categoria", "valores": ["perro", "gato"]},
    "sexo": {"tipo": "categoria", "valores": ["macho", "hembra"]},
    "categoria_edad": {"tipo": "categoria", "valores": ["cachorro", "joven", "adulto", "senior"]},
    "tamano": {"tipo": "categoria", "valores": ["pequeno", "mediano", "grande"]},
    "tipo_pelaje": {"tipo": "categoria", "valores": ["corto", "medio", "largo"]},
    "nivel_energia": {"tipo": "entero", "min": 1, "max": 5},
    "sociable_ninos": {"tipo": "categoria", "valores": _NIVELES + ["desconocido"]},
    "sociable_perros": {"tipo": "categoria", "valores": _NIVELES + ["desconocido"]},
    "sociable_gatos": {"tipo": "categoria", "valores": _NIVELES + ["desconocido"]},
    "sociable_desconocidos": {"tipo": "categoria", "valores": _NIVELES},
    "nivel_ladrido": {"tipo": "categoria", "valores": _NIVELES},
    "entrenamiento_basico": {"tipo": "categoria", "valores": ["nulo", "parcial", "completo"]},
    "estado_salud": {"tipo": "categoria", "valores": ["sano", "cronica", "necesidad_especial"]},
    "esterilizado": {"tipo": "binario"},
    "vacunas_completas": {"tipo": "binario"},
    "departamento_refugio": {"tipo": "texto"},
    "zona_climatica_refugio": {"tipo": "categoria", "valores": ["costa", "sierra", "selva"]},
    "req_area_exterior": {"tipo": "binario"},
    "req_cercado": {"tipo": "binario"},
    "req_sin_ninos": {"tipo": "binario"},
    "dias_en_refugio": {"tipo": "entero", "min": 0},
}
_SI = {"1", "true", "si", "sí", "s", "yes"}
_NO = {"0", "false", "no", "n"}


def _normalizar(registro: dict, esquema: dict) -> tuple[dict, list]:
    """Valida y convierte los tipos de un registro según el esquema. Devuelve (registro_limpio, problemas)."""
    out, problemas = {}, []
    for campo, spec in esquema.items():
        v = registro.get(campo)
        if v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and not v.strip()):
            if spec.get("nulo"):
                out[campo] = np.nan
            elif "defecto" in spec:
                out[campo] = spec["defecto"]
            else:
                problemas.append(f"falta '{campo}'")
            continue
        tipo = spec["tipo"]
        try:
            if tipo == "binario":
                s = str(v).strip().lower()
                if isinstance(v, (bool, np.bool_)) or s in _SI | _NO or s in ("1.0", "0.0"):
                    v = int(bool(v)) if isinstance(v, (bool, np.bool_)) else int(s in _SI or s == "1.0")
                else:
                    raise ValueError
            elif tipo in ("entero", "numero"):
                x = float(v)
                if tipo == "entero":
                    if x != int(x):
                        raise ValueError
                    x = int(x)
                if ("min" in spec and x < spec["min"]) or ("max" in spec and x > spec["max"]):
                    problemas.append(f"'{campo}'={v} fuera de rango [{spec.get('min', '')}, {spec.get('max', '')}]")
                    continue
                v = x
            elif tipo == "categoria":
                v = str(v).strip().lower()
                if v not in spec["valores"]:
                    problemas.append(f"'{campo}'='{v}' no es válido; usa uno de {spec['valores']}")
                    continue
            elif tipo == "multiple":
                partes = v if isinstance(v, (list, tuple, set)) else str(v).split("|")
                partes = [str(p).strip().lower() for p in partes if str(p).strip()]
                malas = [p for p in partes if p not in spec["valores"] + ["indistinto"]]
                if malas or not partes:
                    problemas.append(f"'{campo}'={v!r} no es válido; usa 'indistinto' o valores de {spec['valores']}")
                    continue
                v = "indistinto" if "indistinto" in partes else "|".join(dict.fromkeys(partes))
            else:  # texto
                v = str(v).strip().lower()
        except (TypeError, ValueError):
            problemas.append(f"'{campo}'={v!r} no es un {tipo} válido")
            continue
        out[campo] = v
    return out, problemas


def validar_adoptante(adoptante: dict) -> dict:
    """Devuelve el adoptante con los tipos normalizados. Lanza ValueError con todos los problemas juntos."""
    a, problemas = _normalizar(dict(adoptante), ESQUEMA_ADOPTANTE)
    if a.get("presencia_ninos") == 0:
        a["edad_min_ninos"] = 0                                   # así viene en los datos de entrenamiento
    elif a.get("presencia_ninos") == 1 and pd.isna(a.get("edad_min_ninos")):
        problemas.append("falta 'edad_min_ninos' (hay niños)")
    if not problemas and a["area_exterior"] == 0 and pd.notna(a["m2_exterior"]) and a["m2_exterior"] > 0:
        problemas.append("'m2_exterior' > 0 pero 'area_exterior' = 0")
    if problemas:
        raise ValueError("Adoptante inválido: " + "; ".join(problemas))
    return a


def validar_mascota(mascota: dict) -> dict:
    """Igual que validar_adoptante, para una mascota del catálogo. Conserva 'pet_id' y las columnas extra
    (nombre, foto, etc.), que no usa el modelo pero se devuelven en las recomendaciones."""
    m, problemas = _normalizar(dict(mascota), ESQUEMA_MASCOTA)
    pid = mascota.get("pet_id")
    if pid is None or not str(pid).strip():
        problemas.insert(0, "falta 'pet_id'")
    if problemas:
        raise ValueError(f"Mascota {pid or ''} inválida: " + "; ".join(problemas))
    extra = {k: v for k, v in mascota.items() if k not in ESQUEMA_MASCOTA and k != "pet_id"}
    return {"pet_id": str(pid).strip(), **m, **extra}


# ---------------- nivel 0: reglas duras de bienestar animal ----------------
def reglas_pares(X):
    """X: una fila por par (adoptante, mascota). Devuelve una columna booleana por regla."""
    return pd.DataFrame({
        "La vivienda no permite mascotas": X.permiso_mascotas == 0,
        "Requiere área exterior y el hogar no tiene": (X.req_area_exterior == 1) & (X.area_exterior == 0),
        "Requiere cercado seguro y el hogar no tiene": (X.req_cercado == 1) & (X.cercado_seguro == 0),
        "No apta para hogares con niños": (X.presencia_ninos == 1)
                                           & ((X.req_sin_ninos == 1) | (X.sociable_ninos == "bajo")),
        "No se lleva con perros y el hogar tiene perro": (X.otras_mascotas_perro == 1) & (X.sociable_perros == "bajo"),
        "No se lleva con gatos y el hogar tiene gato": (X.otras_mascotas_gato == 1) & (X.sociable_gatos == "bajo"),
        "Necesidad especial sin disposición del adoptante": (X.estado_salud == "necesidad_especial")
                                                             & (X.apertura_nec_especiales == 0),
    }, index=X.index)


def pares(A, P):
    """Producto cartesiano adoptantes × mascotas (orden: adoptante, luego mascota)."""
    Ar = A.loc[A.index.repeat(len(P))].reset_index()
    Pr = P.iloc[np.tile(np.arange(len(P)), len(A))].reset_index()
    return pd.concat([Ar, Pr], axis=1)


# ---------------- codificación para el motor (nivel 1a) ----------------
CAT_MASCOTA = ["especie", "sexo", "categoria_edad", "tamano", "tipo_pelaje", "nivel_energia", "sociable_ninos",
               "sociable_perros", "sociable_gatos", "sociable_desconocidos", "nivel_ladrido",
               "entrenamiento_basico", "estado_salud", "zona_climatica_refugio"]
BIN_MASCOTA = ["esterilizado", "vacunas_completas", "req_area_exterior", "req_cercado", "req_sin_ninos"]
CAT_ADOPTANTE = ["departamento", "zona_climatica", "tipo_vivienda", "nivel_actividad", "experiencia_previa",
                 "presupuesto_mensual", "tiempo_adiestramiento", "pref_especie", "pref_sexo", "pref_energia",
                 "tenencia_propia", "permiso_mascotas", "area_exterior", "cercado_seguro", "presencia_ninos",
                 "otras_mascotas_perro", "otras_mascotas_gato", "apertura_nec_especiales", "apertura_senior"]


def codificar_mascotas(P):
    return pd.concat([pd.get_dummies(P[CAT_MASCOTA].astype(str)), P[BIN_MASCOTA]], axis=1).astype(float)


def codificar_adoptantes(A):
    rangos = pd.DataFrame({
        "m2_exterior": pd.cut(A.m2_exterior, [-1, 0, 30, 80, np.inf], labels=["0", "1-30", "31-80", "80+"])
                         .astype(str).replace("nan", "sin_dato"),
        "horas_fuera": pd.cut(A.horas_fuera_dia, [-1, 4, 8, 12, 24], labels=["0-4", "5-8", "9-12", "13+"]).astype(str),
        "integrantes": pd.cut(A.n_integrantes, [0, 1, 3, 99], labels=["1", "2-3", "4+"]).astype(str),
        "edad_ninos": np.where(A.presencia_ninos == 0, "sin_ninos",
                               pd.cut(A.edad_min_ninos, [-1, 5, 11, 99], labels=["0-5", "6-11", "12+"]).astype(str)),
    }, index=A.index)
    return pd.concat([pd.get_dummies(pd.concat([A[CAT_ADOPTANTE].astype(str), rangos], axis=1)),
                      A.pref_tamano.str.get_dummies("|").add_prefix("pref_tamano_"),
                      A.pref_edad.str.get_dummies("|").add_prefix("pref_edad_")], axis=1).astype(float)


def nombre_mascota(pet_id, m) -> str:
    """Nombre legible: usa la columna 'nombre' del catálogo si existe; si no, una descripción."""
    desc = f"{m['especie']} {m['categoria_edad']} {m['tamano']} ({m['sexo']}, energía {m['nivel_energia']})"
    nombre = m.get("nombre") if hasattr(m, "get") else None
    return f"{nombre} · {desc}" if isinstance(nombre, str) and nombre.strip() else f"{pet_id} · {desc}"


# ---------------- features del par para la compatibilidad (nivel 1b) ----------------
_NIVEL = {"bajo": 0.0, "medio": 0.5, "alto": 1.0, "desconocido": 0.5}


def _ord(s, mapa):
    return s.astype(str).map(mapa).astype(float)


def _b(cond):
    return np.asarray(cond, dtype=float)


def _contiene(pref, valor):
    return np.array([str(v) in str(p).split("|") for p, v in zip(pref, valor)])


def features_compatibilidad(X):
    """Features del par adoptante × mascota. X debe tener columnas de adopters y de pets."""
    f = {}
    ene = (X.nivel_energia - 1) / 4
    act = _ord(X.nivel_actividad, {"sedentario": 0, "moderado": .5, "activo": 1})
    f["energia_vs_actividad"] = (ene - act).abs()
    f["energia_vs_preferida"] = (ene - (X.pref_energia - 1) / 4).abs()

    talla = _ord(X.tamano, {"pequeno": 0, "mediano": .5, "grande": 1})
    hogar = (_ord(X.tipo_vivienda, {"departamento": 0, "casa": .5, "casa_terreno": 1}) + X.area_exterior
             + (X.m2_exterior.fillna(0) / 100).clip(0, 1)) / 3
    f["talla_vs_espacio"] = (talla - hogar).abs()
    f["grande_sin_exterior"] = _b((X.tamano == "grande") & (X.area_exterior == 0))

    entren = _ord(X.entrenamiento_basico, {"nulo": 0, "parcial": .5, "completo": 1})
    demanda = (_b(X.categoria_edad == "cachorro") + (1 - entren) + ene + _b(X.estado_salud != "sano")) / 4
    disponible = 1 - (X.horas_fuera_dia / 16).clip(0, 1)
    f["demanda_vs_disponibilidad"] = (demanda - disponible).abs()
    f["deficit_atencion"] = (demanda - disponible).clip(lower=0)

    dificultad = (ene + (1 - entren) + _b(X.sociable_desconocidos == "bajo")
                  + _b(X.estado_salud == "necesidad_especial")) / 4
    f["dificultad_sobre_experiencia"] = (dificultad - _ord(X.experiencia_previa,
                                                           {"ninguna": 0, "alguna": .5, "amplia": 1})).clip(lower=0)
    f["sin_entrenamiento_poco_tiempo"] = (1 - entren) * (1 - _ord(X.tiempo_adiestramiento,
                                                                 {"baja": 0, "media": .5, "alta": 1}))

    f["hay_ninos"] = _b(X.presencia_ninos)
    f["ninos_x_sociable"] = X.presencia_ninos * _ord(X.sociable_ninos, _NIVEL)
    f["ninos_pequenos_x_energia"] = X.presencia_ninos * _b(X.edad_min_ninos <= 5) * ene
    f["hay_perro"] = _b(X.otras_mascotas_perro)
    f["perro_x_sociable"] = X.otras_mascotas_perro * _ord(X.sociable_perros, _NIVEL)
    f["hay_gato"] = _b(X.otras_mascotas_gato)
    f["gato_x_sociable"] = X.otras_mascotas_gato * _ord(X.sociable_gatos, _NIVEL)

    f["mismo_departamento"] = _b(X.departamento == X.departamento_refugio)
    f["misma_zona"] = _b(X.zona_climatica == X.zona_climatica_refugio)

    for nombre, pref, valor in [("especie", X.pref_especie, X.especie), ("tamano", X.pref_tamano, X.tamano),
                                ("edad", X.pref_edad, X.categoria_edad), ("sexo", X.pref_sexo, X.sexo)]:
        indistinto = (pref == "indistinto").to_numpy()
        ok = _contiene(pref, valor)
        f[f"{nombre}_coincide"] = _b(ok & ~indistinto)
        f[f"{nombre}_no_coincide"] = _b(~ok & ~indistinto)

    f["senior_x_apertura"] = _b(X.categoria_edad == "senior") * X.apertura_senior
    f["salud_x_apertura"] = _b(X.estado_salud != "sano") * X.apertura_nec_especiales
    f["salud_x_presupuesto_bajo"] = _b(X.estado_salud != "sano") * _b(X.presupuesto_mensual == "bajo")
    f["ladrido_en_departamento"] = _b(X.nivel_ladrido == "alto") * _b(X.tipo_vivienda == "departamento")

    # Sesgo cultural: atributos intrínsecos de la mascota, no compatibilidad
    f["es_cachorro"] = _b(X.categoria_edad == "cachorro")
    f["es_pequeno"] = _b(X.tamano == "pequeno")
    f["es_senior"] = _b(X.categoria_edad == "senior")
    f["salud_cronica"] = _b(X.estado_salud == "cronica")
    f["necesidad_especial"] = _b(X.estado_salud == "necesidad_especial")

    f["viola_regla_dura"] = _b(reglas_pares(X).any(axis=1))
    f["posicion"] = X["posicion_mostrada"] if "posicion_mostrada" in X else np.nan
    return pd.DataFrame(f, index=X.index).astype(float)


GRUPOS = {
    "energia": ["energia_vs_actividad"], "pref_energia": ["energia_vs_preferida"],
    "espacio": ["talla_vs_espacio", "grande_sin_exterior"],
    "atencion": ["demanda_vs_disponibilidad", "deficit_atencion"],
    "experiencia": ["dificultad_sobre_experiencia", "sin_entrenamiento_poco_tiempo"],
    "ninos": ["hay_ninos", "ninos_x_sociable", "ninos_pequenos_x_energia"],
    "mascotas": ["hay_perro", "perro_x_sociable", "hay_gato", "gato_x_sociable"],
    "geo": ["mismo_departamento", "misma_zona"],
    "pref_especie": ["especie_coincide", "especie_no_coincide"],
    "pref_tamano": ["tamano_coincide", "tamano_no_coincide"],
    "pref_edad": ["edad_coincide", "edad_no_coincide"],
    "pref_sexo": ["sexo_coincide", "sexo_no_coincide"],
    "apertura": ["senior_x_apertura", "salud_x_apertura"],
    "otros": ["salud_x_presupuesto_bajo", "ladrido_en_departamento"],
    "sesgo": ["es_cachorro", "es_pequeno", "es_senior", "salud_cronica", "necesidad_especial"],
    "reglas": ["viola_regla_dura"], "posicion": ["posicion"],
}

# ---------------- explicaciones: motivos y alertas ----------------
FRASES = {   # feature: (frase si suma, frase si resta)
    "energia_vs_actividad": ("Su energía encaja con tu nivel de actividad", "Su energía no encaja con tu nivel de actividad"),
    "energia_vs_preferida": ("Tiene la energía que buscas", "Su energía difiere de la que buscas"),
    "talla_vs_espacio": ("Su talla es adecuada para tu espacio", "Su talla no es ideal para tu espacio"),
    "grande_sin_exterior": (None, "Es grande y no tienes área exterior"),
    "demanda_vs_disponibilidad": ("Tu tiempo en casa cubre lo que necesita", "Necesita más atención de la que puedes dar"),
    "deficit_atencion": (None, "Necesita más atención de la que puedes dar"),
    "dificultad_sobre_experiencia": ("Tu experiencia es suficiente para su manejo", "Su manejo exige más experiencia"),
    "sin_entrenamiento_poco_tiempo": (None, "Necesita adiestramiento y tienes poco tiempo"),
    "ninos_x_sociable": ("Es sociable con niños", "Poco sociable con niños"),
    "perro_x_sociable": ("Se lleva bien con perros", "Poco sociable con perros"),
    "gato_x_sociable": ("Se lleva bien con gatos", "Poco sociable con gatos"),
    "mismo_departamento": ("Está en tu mismo departamento", None),
    "misma_zona": ("Está en tu misma zona climática", None),
    "especie_coincide": ("Es la especie que prefieres", None),
    "especie_no_coincide": (None, "No es la especie que prefieres"),
    "tamano_coincide": ("Es del tamaño que prefieres", None),
    "tamano_no_coincide": (None, "No es del tamaño que prefieres"),
    "edad_coincide": ("Es de la edad que prefieres", None),
    "edad_no_coincide": (None, "No es de la edad que prefieres"),
    "sexo_coincide": ("Es del sexo que prefieres", None),
    "senior_x_apertura": ("Estás abierto a adoptar un senior", None),
    "salud_x_apertura": ("Estás abierto a cuidados especiales", None),
    "salud_x_presupuesto_bajo": (None, "Su salud requiere gastos y tu presupuesto es bajo"),
    "ladrido_en_departamento": (None, "Ladra mucho y vives en departamento"),
}

FRASES_ALBERGUE = {   # feature: frases en 3ra persona para la evaluación del albergue sobre el postulante
    "energia_vs_actividad": ("Su nivel de energía encaja con la actividad del postulante", "Su energía no encaja con el nivel de actividad del postulante"),
    "energia_vs_preferida": ("Tiene la energía que busca el postulante", "Su energía difiere de la que busca el postulante"),
    "talla_vs_espacio": ("Su talla es adecuada para la vivienda del postulante", "Su talla no es ideal para el espacio de la vivienda"),
    "grande_sin_exterior": (None, "Es grande y el postulante no cuenta con área exterior"),
    "demanda_vs_disponibilidad": ("El tiempo en casa del postulante cubre lo que la mascota necesita", "La mascota necesita más atención del tiempo disponible del postulante"),
    "deficit_atencion": (None, "La mascota necesita más atención de la que el postulante puede dedicarle"),
    "dificultad_sobre_experiencia": ("El postulante cuenta con experiencia suficiente para su manejo", "El manejo de la mascota exige mayor experiencia de la que posee el postulante"),
    "sin_entrenamiento_poco_tiempo": (None, "Necesita adiestramiento y el postulante dispone de poco tiempo libre"),
    "ninos_x_sociable": ("Es sociable con niños", "Poco sociable con niños"),
    "perro_x_sociable": ("Se lleva bien con perros", "Poco sociable con perros"),
    "gato_x_sociable": ("Se lleva bien con gatos", "Poco sociable con gatos"),
    "mismo_departamento": ("Ubicada en el mismo departamento que el postulante", None),
    "misma_zona": ("Ubicada en la misma zona climática que el postulante", None),
    "especie_coincide": ("Coincide con la especie que busca el postulante", None),
    "especie_no_coincide": (None, "Difiere de la especie que busca el postulante"),
    "tamano_coincide": ("Coincide con el tamaño que busca el postulante", None),
    "tamano_no_coincide": (None, "Difiere del tamaño que busca el postulante"),
    "edad_coincide": ("Coincide con la edad que busca el postulante", None),
    "edad_no_coincide": (None, "Difiere de la edad que busca el postulante"),
    "sexo_coincide": ("Coincide con el sexo que busca el postulante", None),
    "senior_x_apertura": ("Postulante con apertura a adoptar una mascota senior", None),
    "salud_x_apertura": ("Postulante con apertura a cuidados especiales de salud", None),
    "salud_x_presupuesto_bajo": (None, "Su salud requiere gastos y el presupuesto del postulante es bajo"),
    "ladrido_en_departamento": (None, "Nivel de vocalización alto y el postulante reside en departamento"),
}
BINARIAS = {c for c in FRASES if c not in ("energia_vs_actividad", "energia_vs_preferida", "talla_vs_espacio",
                                           "demanda_vs_disponibilidad", "deficit_atencion",
                                           "dificultad_sobre_experiencia", "sin_entrenamiento_poco_tiempo")}


def motivos(contrib, valores, n=3, signo=+1, para_albergue: bool = False) -> list:
    """Frases de las features que más suman (signo=+1) o restan (signo=-1) al score de compatibilidad."""
    diccionario = FRASES_ALBERGUE if para_albergue else FRASES
    out = []
    for f in (contrib * signo).sort_values(ascending=False).index:
        if len(out) == n or contrib[f] * signo <= 0:
            break
        if f not in diccionario or (f in BINARIAS and valores[f] == 0):
            continue
        frase = diccionario[f][0 if signo > 0 else 1]
        if frase and frase not in out:
            out.append(frase)
    return out


def alertas(x) -> list:
    """Riesgos que el albergue debería revisar en la entrevista (no excluyen, informan)."""
    al = []
    act = {"sedentario": 1.5, "moderado": 3, "activo": 4.5}[x.nivel_actividad]
    if abs(x.nivel_energia - act) >= 2:
        al.append(f"Energía {x.nivel_energia}/5 para un hogar {x.nivel_actividad}")
    if x.horas_fuera_dia >= 10 and (x.categoria_edad == "cachorro" or x.nivel_energia >= 4):
        al.append(f"Pasa {x.horas_fuera_dia:.0f} h fuera y la mascota demanda atención")
    if x.experiencia_previa == "ninguna" and (x.entrenamiento_basico == "nulo" or x.estado_salud != "sano"):
        al.append("Sin experiencia previa y la mascota requiere manejo")
    for cond, campo, texto in [(x.presencia_ninos, "sociable_ninos", "niños"),
                               (x.otras_mascotas_perro, "sociable_perros", "perros"),
                               (x.otras_mascotas_gato, "sociable_gatos", "gatos")]:
        if cond == 1 and x[campo] == "desconocido":
            al.append(f"Sociabilidad con {texto} desconocida: evaluar en visita")
    if x.estado_salud != "sano" and x.presupuesto_mensual == "bajo":
        al.append("Condición de salud con presupuesto bajo")
    if x.tenencia_propia == 0:
        al.append("Vivienda no propia: pedir permiso escrito del propietario")
    if x.departamento != x.departamento_refugio:
        al.append(f"Traslado {x.departamento_refugio} → {x.departamento}")
    if pd.isna(x.m2_exterior) and x.tamano != "pequeno":
        al.append("Falta el dato de m² exteriores")
    return al


# ---------------- niveles 2 y 3: mezcla y reranking ----------------
def minmax_filas(M):
    lo, hi = M.min(1, keepdims=True), M.max(1, keepdims=True)
    return (M - lo) / np.where(hi - lo > 0, hi - lo, 1)


def rerank_carga(S, k, eta):
    """Greedy por adoptante: penaliza mascotas ya recomendadas en el lote (carga / carga esperada)."""
    S = S.copy()
    carga = np.zeros(S.shape[1])
    esperada = max(S.shape[0] * k / S.shape[1], 1)
    for i in range(len(S)):
        S[i] -= eta * carga / esperada
        top = np.argsort(-S[i])[:k]
        carga[top[np.isfinite(S[i, top])]] += 1
    return S


def _normalize(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.where(n > 0, n, 1)


def _sigmoide(z):
    return 1 / (1 + np.exp(-z))


# ---------------- exportación (la llama el notebook de entrenamiento) ----------------
NOMBRE_MODELO = "adoptanet-recomendador"
ARCHIVOS = {"parametros": "parametros.npz", "catalogo": "catalogo.csv"}


def exportar(rec, carpeta: Union[str, Path] = "export_recomendador", version: str = "1.0.0",
             metricas: Optional[dict] = None, dataset: str = "", ejemplo_adoptante: Optional[dict] = None) -> Path:
    """Guarda un RecomendadorAdopcion entrenado como model_config.json + parametros.npz + catalogo.csv.
    No usa pickle: lo exportado se carga con ModeloRecomendacion sin el notebook ni scikit-learn."""
    m, c = rec.motor_, rec.compat_
    if m.engine != "ease":
        raise NotImplementedError("Solo se exporta el motor 'ease' (el 'svd' no se puede recalcular al cambiar el catálogo)")
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)

    # TF-IDF de los atributos de mascota con el idf del entrenamiento (fijo, como un vectorizador ajustado)
    Fm = codificar_mascotas(rec.mascotas_).reindex(index=m.item_ids_, columns=m.feature_names_, fill_value=0)
    Fv = Fm.to_numpy(float)
    idf = np.log((1 + len(Fv)) / (1 + (Fv > 0).sum(0))) + 1
    if not np.allclose(_normalize(Fv * idf), m.F_):
        raise RuntimeError("La codificación de mascotas no reproduce motor_.F_ del entrenamiento")
    gram = (m.Xpos_.T @ m.Xpos_).toarray()   # EASE solo necesita XᵀX: con él se recalcula B para cualquier catálogo

    np.savez_compressed(
        carpeta / ARCHIVOS["parametros"],
        motor_gram=gram, motor_H=m.H_, motor_global=np.asarray(m.global_hist_).ravel(), motor_idf=idf,
        compat_media=c.scaler_.mean_, compat_escala=c.scaler_.scale_,
        compat_coef=np.vstack([mod.coef_[0] for mod in c.modelos_]),
        compat_intercepto=np.array([mod.intercept_[0] for mod in c.modelos_]),
        compat_es_sesgo=c.es_sesgo_)
    cat = rec.mascotas_.copy()
    cat.index.name = "pet_id"
    cat.to_csv(carpeta / ARCHIVOS["catalogo"], encoding="utf-8")

    config = {
        "nombre": NOMBRE_MODELO,
        "version": version,
        "fecha_entrenamiento": date.today().isoformat(),
        "dataset": dataset,
        "archivos": ARCHIVOS,
        "niveles": {
            "0": "Reglas duras de bienestar animal (fuera del ML): excluyen el par",
            "1a": "Motor EASE con historial demográfico (cold start de adoptante)",
            "1b": "Compatibilidad: logística ordinal P(r>=1) + P(r>=2) sobre features del par",
            "2": "score = gamma * minmax(compatibilidad) + (1 - gamma) * motor",
            "3": "+ eta_estancia * percentil(días en refugio); disponibles; reparto de carga en lote",
        },
        "hiperparametros": {
            "motor": {"engine": m.engine, "reg": m.reg, "beta": m.beta, "alpha": m.alpha,
                      "lambda_demo": m.lambda_demo, "neg_weight": m.neg_weight, "pos_threshold": m.pos_threshold},
            "C": rec.C, "gamma": rec.gamma, "peso_sesgo": rec.peso_sesgo, "eta_estancia": rec.eta_estancia,
            "eta_carga": rec.eta_carga, "umbral_minimo": rec.umbral_minimo, "k": rec.k,
        },
        "motor": {"items": [str(i) for i in m.item_ids_], "features_mascota": [str(f) for f in m.feature_names_],
                  "features_adoptante": [str(f) for f in m.demo_cols_]},
        "compatibilidad": {"features": [str(f) for f in c.cols_], "posicion_media": float(c.pos_media_)},
        "reglas_duras": list(reglas_pares(pd.DataFrame(columns=[*ESQUEMA_ADOPTANTE, *ESQUEMA_MASCOTA])).columns),
        "esquema_adoptante": ESQUEMA_ADOPTANTE,
        "esquema_mascota": ESQUEMA_MASCOTA,
        "ejemplo_adoptante": ejemplo_adoptante,
        "metricas_test": metricas or {},
    }
    (carpeta / "model_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=float),
                                               encoding="utf-8")
    return carpeta


# ---------------- inferencia ----------------
AdoptanteInput = Union[dict, pd.Series]


class ModeloRecomendacion:
    """Modelo exportado listo para usar: se carga una vez y luego solo se le pasan adoptantes (dict)."""

    def __init__(self, model_dir: Union[str, Path] = "export_recomendador",
                 catalogo: Union[pd.DataFrame, Iterable[dict], None] = None, verbose: bool = True):
        self.model_dir = Path(model_dir)
        self.config = json.loads((self.model_dir / "model_config.json").read_text(encoding="utf-8"))
        hp = self.config["hiperparametros"]
        self.hiperparametros = {k: hp[k] for k in ("gamma", "peso_sesgo", "eta_estancia", "eta_carga",
                                                   "umbral_minimo", "k")}
        self._motor = hp["motor"]
        with np.load(self.model_dir / self.config["archivos"]["parametros"]) as p:
            self._p = {k: p[k] for k in p.files}
        self._items = {pid: j for j, pid in enumerate(self.config["motor"]["items"])}
        self._feat_mascota = self.config["motor"]["features_mascota"]
        self._feat_adoptante = self.config["motor"]["features_adoptante"]
        self._feat_compat = self.config["compatibilidad"]["features"]
        self._catalogo_csv = self.model_dir / self.config["archivos"]["catalogo"]
        if catalogo is None:
            catalogo = pd.read_csv(self._catalogo_csv, dtype={"pet_id": str})
        self.actualizar_catalogo(catalogo, guardar=False, verbose=False)
        self.verbose = verbose
        if verbose:
            print(f"ModeloRecomendacion listo: {self.config['nombre']} {self.config['version']} | "
                  f"catálogo {len(self.catalogo)} mascotas")

    def __repr__(self):
        return (f"ModeloRecomendacion({self.config['nombre']} {self.config['version']}, "
                f"catálogo={len(self.catalogo)} mascotas, {self.hiperparametros})")

    # ---------------- catálogo ----------------
    def actualizar_catalogo(self, mascotas: Union[pd.DataFrame, Iterable[dict]], guardar: bool = False,
                            verbose: bool = True) -> pd.DataFrame:
        """Reemplaza el catálogo completo (p. ej. con las mascotas que el backend lee de su base de datos)."""
        registros = (mascotas.reset_index() if isinstance(mascotas, pd.DataFrame) and "pet_id" not in mascotas
                     else mascotas)
        registros = registros.to_dict("records") if isinstance(registros, pd.DataFrame) else list(registros)
        filas, errores = [], []
        for r in registros:
            try:
                filas.append(validar_mascota(r))
            except ValueError as e:
                errores.append(str(e))
        if errores:
            raise ValueError("Catálogo inválido:\n  " + "\n  ".join(errores))
        cat = pd.DataFrame(filas).set_index("pet_id")
        if cat.index.duplicated().any():
            raise ValueError(f"pet_id repetidos: {sorted(cat.index[cat.index.duplicated()].unique())}")
        if cat.empty:
            raise ValueError("El catálogo está vacío")
        self.catalogo = cat
        self._preparar_motor()
        if guardar:
            self.guardar_catalogo()
        if verbose:
            print(f"Catálogo actualizado: {len(cat)} mascotas ({self._nuevas} sin historial de entrenamiento)")
        return cat

    def agregar_mascota(self, mascota: dict, guardar: bool = True) -> dict:
        """Agrega (o reemplaza, si el pet_id ya existe) una mascota, sin reentrenar. Por defecto guarda el catálogo."""
        m = validar_mascota(mascota)
        pid = m["pet_id"]
        existia = pid in self.catalogo.index
        registros = self.catalogo.reset_index().to_dict("records")
        registros = [m if r["pet_id"] == pid else r for r in registros] + ([] if existia else [m])
        self.actualizar_catalogo(registros, guardar=guardar, verbose=False)
        print(f"{'Actualizada' if existia else 'Agregada'}: {nombre_mascota(pid, self.catalogo.loc[pid])} | "
              f"catálogo {len(self.catalogo)} mascotas")
        return m

    def quitar_mascota(self, pet_id: str, guardar: bool = True) -> bool:
        """Quita una mascota del catálogo (p. ej. cuando se adopta). Por defecto guarda el catálogo."""
        if pet_id not in self.catalogo.index:
            return False
        if len(self.catalogo) == 1:
            raise ValueError("No se puede dejar el catálogo vacío")
        self.catalogo = self.catalogo.drop(index=pet_id)
        self._preparar_motor()
        if guardar:
            self.guardar_catalogo()
        print(f"Quitada: {pet_id} | catálogo {len(self.catalogo)} mascotas")
        return True

    def guardar_catalogo(self, ruta: Union[str, Path, None] = None):
        self.catalogo.to_csv(ruta or self._catalogo_csv, encoding="utf-8")

    def _preparar_motor(self):
        """Recalcula EASE para el catálogo actual. Con el catálogo de entrenamiento da exactamente el modelo
        entrenado; una mascota nueva queda como un ítem sin interacciones (cold start de mascota: su score de
        motor es 0 y la recomendación depende de la compatibilidad); una quitada sale de la matriz."""
        cat, p = self.catalogo, self._p
        ids = list(cat.index)
        cj = [j for j, pid in enumerate(ids) if pid in self._items]          # posición en el catálogo
        tj = [self._items[ids[j]] for j in cj]                               # posición en el entrenamiento
        n = len(ids)
        F = codificar_mascotas(cat[list(ESQUEMA_MASCOTA)]).reindex(columns=self._feat_mascota, fill_value=0)
        self._F = _normalize(F.to_numpy(float) * p["motor_idf"])
        G = np.zeros((n, n))
        G[np.ix_(cj, cj)] = p["motor_gram"][np.ix_(tj, tj)]
        G += self._motor["beta"] ** 2 * (self._F @ self._F.T)
        G[np.diag_indices_from(G)] += self._motor["reg"]
        P = np.linalg.inv(G)
        B = -P / np.diag(P)
        np.fill_diagonal(B, 0.0)
        self._B = B
        self._H = np.zeros((p["motor_H"].shape[0], n))
        self._H[:, cj] = p["motor_H"][:, tj]
        self._global = np.zeros(n)
        self._global[cj] = p["motor_global"][tj]
        self._estancia = cat.dias_en_refugio.rank(pct=True).to_numpy()
        self._nuevas = n - len(cj)

    # ---------------- entrada ----------------
    def validar(self, adoptante: AdoptanteInput) -> dict:
        """Adoptante con tipos normalizados, o ValueError con todos los problemas (útil para el formulario)."""
        return validar_adoptante(dict(adoptante))

    def ejemplo(self) -> dict:
        """Un adoptante de ejemplo con todos los campos (para probar el modelo)."""
        return dict(self.config.get("ejemplo_adoptante") or {})

    def _adoptantes(self, adoptantes) -> pd.DataFrame:
        if isinstance(adoptantes, pd.DataFrame):
            ids = list(adoptantes.index)
            registros = adoptantes.to_dict("records")
        else:
            registros = [dict(a) for a in ([adoptantes] if isinstance(adoptantes, (dict, pd.Series)) else adoptantes)]
            ids = [r.get("respondent_id", f"adoptante_{i}") for i, r in enumerate(registros)]
        filas, errores = [], []
        for i, r in zip(ids, registros):
            try:
                filas.append(validar_adoptante(r))
            except ValueError as e:
                errores.append(f"{i}: {e}")
        if errores:
            raise ValueError("\n".join(errores))
        return pd.DataFrame(filas, index=pd.Index(ids, name="respondent_id"))

    # ---------------- componentes del score ----------------
    def _componentes(self, A: pd.DataFrame, peso_sesgo: float):
        cat = self.catalogo[list(ESQUEMA_MASCOTA)]
        shape = (len(A), len(cat))
        # Nivel 1a: motor en cold start (sin historial): historial demográfico -> EASE (+ contenido si alpha < 1)
        D = codificar_adoptantes(A).reindex(columns=self._feat_adoptante, fill_value=0).to_numpy(float)
        demo = (D @ self._H) / np.maximum(D.sum(1, keepdims=True), 1)
        demo[demo.sum(1) == 0] = self._global
        cf = minmax_filas(demo @ self._B)
        cb = minmax_filas(_normalize(demo @ self._F) @ self._F.T)
        a = self._motor["alpha"]
        motor = a * cf + (1 - a) * cb
        # Nivel 1b: compatibilidad del par
        Xp = pares(A, cat)
        Fc = features_compatibilidad(Xp)
        Fc["posicion"] = self.config["compatibilidad"]["posicion_media"]   # neutraliza el sesgo de posición
        Fc = Fc[self._feat_compat]
        Z = (Fc.to_numpy(float) - self._p["compat_media"]) / self._p["compat_escala"]
        coef, b0, sesgo = self._p["compat_coef"], self._p["compat_intercepto"], self._p["compat_es_sesgo"]
        logit = Z @ coef.T + b0                                            # (pares, 2): r>=1 y r>=2
        parte_sesgo = Z[:, sesgo] @ coef[:, sesgo].T
        comp = lambda ps: _sigmoide(logit - (1 - ps) * parte_sesgo).sum(1).reshape(shape)
        reglas = reglas_pares(Xp)
        return {"motor": motor, "comp": comp(peso_sesgo), "comp_pura": comp(0.0),
                "p1": _sigmoide(logit[:, 0]).reshape(shape), "p2": _sigmoide(logit[:, 1]).reshape(shape),
                "inv": reglas.any(axis=1).to_numpy().reshape(shape), "reglas": reglas, "Xp": Xp, "Fc": Fc,
                "contrib": Z * coef[1]}

    def _score(self, c, gamma, eta_estancia, disponibles=None, excluir=None):
        S = gamma * minmax_filas(c["comp"]) + (1 - gamma) * c["motor"] + eta_estancia * self._estancia
        S = np.where(c["inv"], -np.inf, S)
        fuera = np.zeros(S.shape[1], bool)
        if disponibles is not None:
            fuera |= ~self.catalogo.index.isin(list(disponibles))
        if excluir is not None:
            fuera |= self.catalogo.index.isin(list(excluir))
        S[:, fuera] = -np.inf
        return S

    def _params(self, **kw):
        return {k: (self.hiperparametros[k] if v is None else v) for k, v in kw.items()}

    # ---------------- API ----------------
    def puntuar(self, adoptantes, disponibles=None, excluir=None, gamma=None, peso_sesgo=None,
                eta_estancia=None, eta_carga=None) -> pd.DataFrame:
        """Matriz de scores (adoptantes × mascotas del catálogo). -inf = excluida (regla dura o no disponible).
        Sin umbral mínimo: es la misma matriz que RecomendadorAdopcion.puntuar del notebook."""
        h = self._params(gamma=gamma, peso_sesgo=peso_sesgo, eta_estancia=eta_estancia, eta_carga=eta_carga)
        A = self._adoptantes(adoptantes)
        S = self._score(self._componentes(A, h["peso_sesgo"]), h["gamma"], h["eta_estancia"], disponibles, excluir)
        if h["eta_carga"] > 0:
            S = rerank_carga(S, self.hiperparametros["k"], h["eta_carga"])
        return pd.DataFrame(S, index=A.index, columns=self.catalogo.index)

    def recomendar(self, adoptante: AdoptanteInput, k: Optional[int] = None, disponibles=None, excluir=None,
                   gamma=None, peso_sesgo=None, eta_estancia=None, umbral_minimo=None) -> dict:
        """Top-k para un adoptante nuevo (solo el cuestionario). Respuesta JSON-serializable para el backend.
        disponibles: pet_id que se pueden recomendar (None = todo el catálogo). excluir: pet_id a omitir
        (p. ej. ya vistas). Los demás parámetros sobrescriben los hiperparámetros solo en esta llamada."""
        return self._recomendar_lote([adoptante], k, disponibles, excluir, gamma, peso_sesgo, eta_estancia,
                                     umbral_minimo, eta_carga=0.0)[0]

    def recomendar_lote(self, adoptantes, k: Optional[int] = None, disponibles=None, excluir=None,
                        eta_carga: Optional[float] = None, **kw) -> list:
        """Varios adoptantes a la vez (lista de dicts o DataFrame). Con eta_carga > 0 reparte la demanda para
        que no todos reciban las mismas mascotas (el orden de la lista da la prioridad)."""
        return self._recomendar_lote(adoptantes, k, disponibles, excluir, eta_carga=eta_carga, **kw)

    def _recomendar_lote(self, adoptantes, k=None, disponibles=None, excluir=None, gamma=None, peso_sesgo=None,
                         eta_estancia=None, umbral_minimo=None, eta_carga=None) -> list:
        t0 = time.perf_counter()
        h = self._params(k=k, gamma=gamma, peso_sesgo=peso_sesgo, eta_estancia=eta_estancia,
                         umbral_minimo=umbral_minimo, eta_carga=eta_carga)
        A = self._adoptantes(adoptantes)
        c = self._componentes(A, h["peso_sesgo"])
        S = self._score(c, h["gamma"], h["eta_estancia"], disponibles, excluir)
        viables = np.isfinite(S).sum(1)
        if h["umbral_minimo"] is not None:
            S[c["p1"] < h["umbral_minimo"]] = -np.inf        # mejor recomendar menos que recomendar mal
        if h["eta_carga"] > 0:
            S = rerank_carga(S, h["k"], h["eta_carga"])
        nP, ids = len(self.catalogo), self.catalogo.index
        salida = []
        for i, aid in enumerate(A.index):
            top = [j for j in np.argsort(-S[i])[:h["k"]] if np.isfinite(S[i, j])]
            filas = [i * nP + j for j in top]
            reglas_i = c["reglas"].iloc[i * nP:(i + 1) * nP]
            excl = [{"pet_id": ids[j], "reglas": list(reglas_i.columns[reglas_i.iloc[j].to_numpy()])}
                    for j in np.flatnonzero(c["inv"][i])]
            if top:
                advertencia = None
            elif viables[i] == 0 and c["inv"][i].all():
                advertencia = "Sin mascotas viables para este hogar: ninguna cumple las reglas de bienestar"
            elif viables[i] == 0:
                advertencia = "Sin mascotas disponibles para este hogar"
            else:
                advertencia = "Ninguna mascota viable alcanza la compatibilidad mínima"
            salida.append({
                "modelo": {"nombre": self.config["nombre"], "version": self.config["version"]},
                "adoptante": None if str(aid).startswith("adoptante_") else aid,
                "recomendaciones": [{
                    "posicion": n + 1,
                    "pet_id": ids[j],
                    "mascota": nombre_mascota(ids[j], self.catalogo.iloc[j]),
                    "score": round(float(S[i, j]), 4),
                    "p_interes": round(float(c["p2"][i, j]), 3),
                    "compatibilidad": round(float(c["comp_pura"][i, j] / 2), 3),
                    "motivos": motivos(pd.Series(c["contrib"][f], index=self._feat_compat), c["Fc"].iloc[f]),
                    "alertas": alertas(c["Xp"].iloc[f]),
                } for n, (j, f) in enumerate(zip(top, filas))],
                "n_viables": int(viables[i]),
                "excluidas": excl,
                "reglas_incumplidas": (sorted({r for e in excl for r in e["reglas"]}, key=list(c["reglas"]).index)
                                       if not top else []),
                "advertencia": advertencia,
                "tiempo_ms": round((time.perf_counter() - t0) * 1000, 1),
            })
        return salida

    def ficha(self, adoptante: AdoptanteInput, pet_id: str, mostrar: bool = True) -> dict:
        """Ficha del par para el albergue: reglas, interés, compatibilidad, a favor / en contra y alertas."""
        if pet_id not in self.catalogo.index:
            raise KeyError(f"{pet_id} no está en el catálogo")
        A = self._adoptantes(adoptante)
        j = self.catalogo.index.get_loc(pet_id)
        c = self._componentes(A, self.hiperparametros["peso_sesgo"])
        viol = c["reglas"].iloc[j]
        contrib = pd.Series(c["contrib"][j], index=self._feat_compat)
        compat, p_int = float(c["comp_pura"][0, j] / 2), float(c["p2"][0, j])
        al = alertas(c["Xp"].iloc[j])
        if viol.any():
            veredicto = "NO RECOMENDABLE"
        elif compat >= 0.5 and len(al) <= 2:
            veredicto = "APTO PARA ENTREVISTA"
        else:
            veredicto = "REVISAR CON CUIDADO"
        f = {
            "pet_id": pet_id,
            "mascota": nombre_mascota(pet_id, self.catalogo.loc[pet_id]),
            "dias_en_refugio": int(self.catalogo.loc[pet_id, "dias_en_refugio"]),
            "veredicto": veredicto,
            "reglas_incumplidas": list(viol.index[viol.to_numpy()]),
            "p_interes": round(p_int, 3),
            "compatibilidad": round(compat, 3),
            "a_favor": motivos(contrib, c["Fc"].iloc[j], n=4, para_albergue=True),
            "en_contra": motivos(contrib, c["Fc"].iloc[j], n=4, signo=-1, para_albergue=True),
            "alertas": al,
        }
        if mostrar:
            print(f"Mascota: {f['mascota']} | días en refugio: {f['dias_en_refugio']}")
            print(f"Veredicto sugerido: {veredicto}")
            print(f"  Reglas duras    : {', '.join(f['reglas_incumplidas']) or 'cumple todas'}")
            print(f"  Interés del adoptante: {p_int:.2f}")
            print(f"  Compatibilidad  : {compat:.2f}  (sin sesgo cultural, 0-1)")
            print(f"  A favor         : {'; '.join(f['a_favor']) or '—'}")
            print(f"  En contra       : {'; '.join(f['en_contra']) or '—'}")
            print("  Alertas         : " + ("\n                    ".join(al) if al else "—"))
        return f

    @staticmethod
    def tabla(resultado: dict) -> pd.DataFrame:
        """Las recomendaciones de un resultado como DataFrame (motivos y alertas unidos en texto)."""
        df = pd.DataFrame(resultado["recomendaciones"])
        if df.empty:
            return df
        for col, sep in (("motivos", "; "), ("alertas", " | ")):
            df[col] = df[col].map(sep.join)
        return df.set_index("posicion")

    def mostrar(self, resultado: dict):
        try:
            from IPython.display import display
        except ImportError:
            display = print
        if resultado["advertencia"]:
            print(f"⚠ {resultado['advertencia']}")
            if resultado["reglas_incumplidas"]:
                print("  Reglas incumplidas: " + ", ".join(resultado["reglas_incumplidas"]))
            return
        print(f"Top {len(resultado['recomendaciones'])} de {resultado['n_viables']} mascotas viables "
              f"({len(resultado['excluidas'])} excluidas por reglas de bienestar) en {resultado['tiempo_ms']} ms")
        with pd.option_context("display.max_colwidth", 120, "display.width", 250):
            display(self.tabla(resultado))

    def __call__(self, adoptante: AdoptanteInput, k: Optional[int] = None, mostrar: bool = True, **kw) -> dict:
        r = self.recomendar(adoptante, k=k, **kw)
        if mostrar:
            self.mostrar(r)
        return r


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    modelo = ModeloRecomendacion(Path(__file__).parent)
    entrada = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")) if len(sys.argv) > 1 else modelo.ejemplo()
    print(json.dumps(modelo.recomendar(entrada), ensure_ascii=False, indent=2))
