# POLYDIM_DEST
# destination: polydim/research/
# filename: RESEARCH_TASK027_SEMANTIC_BACKENDS_V1.md
# author: claude-sonnet-4-6 (curso05.mithril@gmail.com)

# RESEARCH — TASK_027: Port MiniLMBackend + FastTextBackend de V03 a V06

**Fecha:** 2026-06-18
**Tarea:** TASK_027 (BACKLOG_V27, prioridad 1)
**Referencia:** POLYDIM_BASES_V1.md — PASO_1 del roadmap fundacional
**Fuente:** polydim_runtime_v03.py (1aS7QU_gnXCp_DssleSNOO9YSKMg-TPV7)

---

## 1. POR QUÉ ES URGENTE (según POLYDIM_BASES_V1.md)

La evaluación Antigravity identifica que los subespacios actuales son MD5-hash de strings:
- `DIM_SQL` y `DIM_PYTHON` son ortogonales por azar matemático, no por semántica real
- El proyecto es "Python con peluca vectorial" mientras esto no cambie
- PASO_1 del roadmap: activar MiniLMBackend como default

Con MiniLMBackend activo:
- `usuario` ↔ `cliente` = 0.830 (cercanos, mismo grupo semántico)
- `usuario` ↔ `tabla` = 0.512 (separados, grupos distintos)
- `DIM_SQL` y `DIM_VECTOR` tendrán estructura semántica real emergente del LLM

---

## 2. QUÉ EXISTE EN V03 (fuente del port)

```
SemanticBackend (ABC):
  - encode(text: str) -> np.ndarray  # vector normalizado

MockSemanticBackend:
  - dim = 64
  - 9 grupos semánticos con vectores deterministas
  - Retrocompatible: sin internet, sin pip

MiniLMBackend:
  - dim = 384
  - sentence-transformers all-MiniLM-L6-v2
  - pip install sentence-transformers
  - Descarga ~90MB en primera ejecución

FastTextBackend:
  - dim = 300
  - fasttext multilingual (incluye castellano)
  - pip install fasttext-wheel
  - Requiere descargar modelo manualmente

make_jl(d_in, d_out=N) -> np.ndarray:
  - Matriz Johnson-Lindenstrauss determinista
  - Proyecta embedding (dim=384) → hipervector (N=10000)
  - Seed determinista: hash("JL_{d_out}_{d_in}")

Space V03 con backend:
  - Space(ps="", semantic_backend=MiniLMBackend())
  - _mk(name): JL @ backend.encode(name) + ps_noise (0.15)
  - Sin backend: comportamiento V02/V06 (MD5 hash)
```

---

## 3. CAMBIOS NECESARIOS SOBRE V06

### 3.1 Imports adicionales
```python
from abc import ABC, abstractmethod
```

### 3.2 Nueva función make_jl (antes de Space)
```python
def make_jl(d_in: int, d_out: int = N) -> np.ndarray:
    """Matriz Johnson-Lindenstrauss determinista: d_in → d_out."""
    s = int(hashlib.md5(f"JL_{d_out}_{d_in}".encode()).hexdigest(), 16) % (2 ** 32)
    R = np.random.default_rng(s).standard_normal((d_out, d_in)).astype(np.float32)
    return R / math.sqrt(d_in)
```

### 3.3 Nuevas clases (antes de Space)
```python
class SemanticBackend(ABC):
    dim: int = 384
    @abstractmethod
    def encode(self, text: str) -> np.ndarray: ...

class MockSemanticBackend(SemanticBackend):
    # 9 grupos + encode con mezcla 60% grupo / 40% base determinista
    ...

class MiniLMBackend(SemanticBackend):
    dim = 384
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer('all-MiniLM-L6-v2')
    def encode(self, text: str) -> np.ndarray:
        return self._model.encode(text, normalize_embeddings=True).astype(np.float32)

class FastTextBackend(SemanticBackend):
    dim = 300
    def __init__(self, model_path: str):
        import fasttext
        self._model = fasttext.load_model(model_path)
    def encode(self, text: str) -> np.ndarray:
        hv = self._model.get_word_vector(text).astype(np.float32)
        n = np.linalg.norm(hv)
        return hv / n if n > 1e-10 else hv
```

### 3.4 Modificación de Space.__init__
```python
def __init__(self, ps: str = "", semantic_backend: SemanticBackend = None):
    self.ps = ps
    self.backend = semantic_backend
    self._s = {}; self._sub = {}
    self._JL = make_jl(semantic_backend.dim) if semantic_backend else None
    for d in NATIVE:
        self._sub[d] = self._mk(d)
```

### 3.5 Modificación de Space._mk()
```python
def _mk(self, name: str) -> np.ndarray:
    if self.backend:
        hv = self._JL @ self.backend.encode(name)
        if self.ps:
            k = f"{self.ps}:{name}"
            s = int(hashlib.md5(k.encode()).hexdigest(), 16) % (2 ** 32)
            p = np.random.default_rng(s).standard_normal(N).astype(np.float32)
            p /= np.linalg.norm(p)
            hv = 0.85 * hv + 0.15 * p
    else:
        k = f"{self.ps}:{name}" if self.ps else name
        s = int(hashlib.md5(k.encode()).hexdigest(), 16) % (2 ** 32)
        hv = np.random.default_rng(s).standard_normal(N).astype(np.float32)
    n = np.linalg.norm(hv)
    return (hv / n if n > 1e-10 else hv).astype(np.float32)
```

---

## 4. RETROCOMPATIBILIDAD

| Caso | Comportamiento |
|------|----------------|
| Space() | Sin backend → MD5 determinista (V06 identical) |
| Space("MI_IA") | Con ps, sin backend → MD5 + ps (V06 identical) |
| Space(semantic_backend=MockSemanticBackend()) | Nuevo: semántica de prueba |
| Space("MI_IA", semantic_backend=MiniLMBackend()) | Nuevo: semántica real + personalización |

Los 14 tests de V06 pasan sin cambio (usan Space() sin backend).

---

## 5. TEST NUEVO (test_mock_semantic_backend)

```python
def test_mock_semantic_backend():
    """TASK_027: MockSemanticBackend produce proximidad semántica real."""
    sp = Space(semantic_backend=MockSemanticBackend())
    # usuario y cliente deben ser más cercanos entre sí que usuario y tabla
    sim_uc = _sim(sp.sym("usuario"), sp.sym("cliente"))  # mismo grupo: identidad
    sim_ut = _sim(sp.sym("usuario"), sp.sym("tabla"))    # grupos distintos: identidad vs datos
    # Con MockSemanticBackend los grupos semánticos deben separarse
    return sim_uc > sim_ut
```

---

## 6. VERSIÓN RESULTADO

V0.7 — TASK_027 — SemanticBackend + MockSemanticBackend + MiniLMBackend + FastTextBackend
Tests: 14 heredados + 1 nuevo = 15/15

---

RESEARCH_TASK027_SEMANTIC_BACKENDS_V1.md — 2026-06-18 — curso05.mithril@gmail.com
