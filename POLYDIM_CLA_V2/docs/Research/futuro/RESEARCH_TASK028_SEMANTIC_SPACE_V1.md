# POLYDIM_DEST
# destination: polydim/research/
# filename: RESEARCH_TASK028_SEMANTIC_SPACE_V1.md
# author: claude-sonnet-4-6 (curso05.mithril@gmail.com)

# RESEARCH — TASK_028: SemanticSpace — NATIVE_DIMS emergentes del modelo de embedding

**Fecha:** 2026-06-19
**Tarea:** TASK_028 (BACKLOG_V28, prioridad 1)
**Prerrequisito:** TASK_027 TERMINADA (backends portados a V07/V052)
**Referencias:** POLYDIM_BASES_V1.md §6 PREGUNTA_1, KF_002, KF_003, KF_004

---

## 1. EL PROBLEMA QUE RESUELVE

Hoy los NATIVE_DIMS son strings hardcodeados mapeados a vectores por MD5:

```python
NATIVE = ["DIM_SQL", "DIM_PYTHON", "DIM_FLUTTER", ...]
# Space._mk("DIM_SQL") → hash("DIM_SQL") → vector aleatorio determinista
```

**DIM_SQL** y **DIM_PYTHON** son cuasi-ortogonales **por azar matemático**, no porque
"SQL" y "Python" sean conceptos independientes en ningún sentido profundo.

La PREGUNTA_1 de POLYDIM_BASES_V1.md lo dice explícito:
> "Los subespacios deben EMERGER de un modelo de embedding real.
>  El proyecto ya tiene MiniLMBackend. Esa era la dirección correcta — no debe abandonarse."

---

## 2. LO QUE EXISTE EN V07 (TASK_027 completada)

```python
class MiniLMBackend(SemanticBackend):
    dim = 384
    def encode(self, text: str) -> np.ndarray:
        return self._model.encode(text, normalize_embeddings=True).astype(np.float32)

class Space:
    def __init__(self, ps="", semantic_backend=None):
        self._JL = make_jl(semantic_backend.dim) if semantic_backend else None
        ...
    def _mk(self, name):
        if self.backend:
            hv = self._JL @ self.backend.encode(name)  # semántica real
        else:
            hv = MD5_hash(name)  # determinista legacy
```

Con `Space(semantic_backend=MiniLMBackend())`, los subespacios ya emergen del LLM.
Pero NATIVE sigue siendo una lista hardcodeada por humanos.

---

## 3. DISEÑO DE SemanticSpace

### Clase propuesta

```python
class SemanticSpace(Space):
    """
    Espacio POLYDIM donde NATIVE_DIMS emergen del modelo de embedding.

    En lugar de:
        NATIVE = ["DIM_SQL", "DIM_PYTHON", ...]  # hardcodeado por humanos
    
    Los subespacios se descubren midiendo cuáles regiones del espacio
    semántico del LLM son naturalmente "ocupadas" por los conceptos
    que el proyecto maneja.

    Paso 1 (implementado aquí): usar las mismas 10 etiquetas PERO
    construir sus vectores con el backend semántico, no con MD5.
    Esto ya produce estructura semántica real: DIM_SQL cercano a
    DIM_VECTOR (ambos sobre datos), DIM_ERROR alejado de DIM_FLUTTER.

    Paso 2 (TASK_030, futura): discover_subspaces() — encontrar
    los subespacios sin etiquetas predefinidas.
    """
    def __init__(self, ps: str = "", semantic_backend: SemanticBackend = None,
                 native_labels: List[str] = None):
        ...
```

### Qué cambia respecto a Space con backend

En `Space(semantic_backend=MiniLMBackend())`, el backend se usa pero NATIVE sigue siendo
la lista global. En `SemanticSpace`, los labels son un parámetro, preparando el camino
para cuando puedan emerger automáticamente.

### Medidas de emergencia semántica (el test clave)

Con MockSemanticBackend (ya verificado en V07):
- `usuario` ↔ `cliente` > `usuario` ↔ `tabla` ✓ (mismo grupo vs grupos distintos)

Con SemanticSpace + MockSemanticBackend:
- `DIM_SQL` ↔ `DIM_VECTOR` > `DIM_SQL` ↔ `DIM_FLUTTER`
  (datos vs datos > datos vs interfaz)
- `DIM_ERROR` ↔ `DIM_META` relativo vs `DIM_ERROR` ↔ `DIM_PYTHON`
  (contexto de ejecución vs lenguaje)

---

## 4. ANÁLISIS DE IMPACTO SOBRE EL SISTEMA DE ALIGN

**Problema potencial identificado:**

Si dos IAs tienen `Space` distintos (una con backend, otra sin él), el ALIGN actual
usa SONDAS para construir matrices A y B, y RPA para alinear. Esto debería seguir
funcionando porque RPA opera sobre los vectores como geometría, no sobre sus nombres.

**Sin embargo**, el `align_score` podría bajar con SemanticSpace porque:
- Los subespacios semánticos **no son quasi-ortogonales por diseño** (a diferencia del MD5)
- DIM_SQL y DIM_VECTOR estarán **cerca** en el espacio → al alinear dos SemanticSpaces,
  la distinción entre ellos importa menos → el score sobre NATIVE podría ser diferente

**Decisión de diseño:** SemanticSpace debe poder alinearse tanto con:
1. Otro SemanticSpace (caso ideal)
2. Un Space determinista MD5 (retrocompatibilidad con IAs sin backend)

Esto requiere que las SONDAS incluyan términos del vocabulario semántico real.
Las SONDAS actuales ya incluyen "usuario", "tabla", "sesion", etc. — bien.

---

## 5. ESTRUCTURA DEL ARTEFACTO

`polydim_semantic_space.py` — módulo independiente, importa de `polydim_runtime_v07.py`

```python
from polydim_runtime_v07 import (
    Space, SemanticBackend, MockSemanticBackend, MiniLMBackend,
    make_jl, NATIVE, SONDAS, N, _sim
)

class SemanticSpace(Space):
    def __init__(self, ps="", semantic_backend=None, native_labels=None):
        ...

def measure_semantic_emergence(space: SemanticSpace) -> dict:
    """Mide estructura semántica real del espacio. Reporta pares similares/distintos."""
    ...

def test_semantic_emergence_mock(): ...
def test_semantic_space_retains_transfer(): ...
def test_semantic_space_align_with_deterministic(): ...
```

---

## 6. CRITERIOS DE ÉXITO

1. `measure_semantic_emergence` reporta pares con similitud > 0.6 dentro del mismo dominio
   (datos↔datos, error↔meta) y < 0.55 entre dominios distintos (datos↔interfaz)
2. Transfer con SemanticSpace sigue detectando dims correctamente (test_api_v01 adaptado)
3. Align entre SemanticSpace y Space determinista → score ≥ 0.70 (puede bajar de 0.85)
4. Sin dependencias nuevas más allá de sentence-transformers (ya en TASK_027)

---

## 7. LO QUE NO HACE ESTA TAREA

- **No elimina NATIVE_DIMS predefinidas** — las conserva como bootstrap
- **No descubre subespacios automáticamente** — eso es el PASO_4 de POLYDIM_BASES_V1.md
- **No cambia la API pública de Space** — SemanticSpace es una subclase compatible

Lo que SÍ hace: dar el primer paso medible desde "DIM_SQL = MD5 aleatorio"
hacia "DIM_SQL = región semántica real donde viven los conceptos de datos tabulares".

---

RESEARCH_TASK028_SEMANTIC_SPACE_V1.md — 2026-06-19 — curso05.mithril@gmail.com
