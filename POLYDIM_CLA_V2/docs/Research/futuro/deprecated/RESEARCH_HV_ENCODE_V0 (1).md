# POLYDIM_DEST
# destino: polydim/research/
# nombre:  RESEARCH_HV_ENCODE_V0.md
# autor:   ai.mpat.agt@gmail.com

# RESEARCH_HV_ENCODE_V0
## Cómo codificar un objeto Capa S en un hipervector Capa G

**Fecha:** 2026-06-13
**Autor:** ai.mpat.agt@gmail.com
**Resuelve:** PEND_007 / TASK_007
**Estado:** BORRADOR V0 — listo para revisión

---

## 1. PROBLEMA

Un `ObjectND` en Capa S es una estructura simbólica:

```
{
  geo_id: <str>,
  dims: {
    "DIM_SQL":    { w: 1.0, props: {"tabla": "usuarios"} },
    "DIM_PYTHON": { w: 0.7, props: {"framework": "fastapi"} }
  }
}
```

Necesitamos un hipervector HV ∈ ℝᴺ que:

1. **Represente** el objeto con fidelidad suficiente (recuperable)
2. **Permita detección** de membresía dimensional (ya funciona: `activacion(dim) > UMBRAL`)
3. **Permita recuperación** de contenido (props) dado un dim conocido
4. **Sea transferible** entre espacios con `align_transform`

La pregunta central: **¿cuál es la función de codificación óptima `encode(obj) → HV`?**

---

## 2. ESTADO ACTUAL EN EL RUNTIME

`ObjectND._hv()` ya implementa una codificación implícita:

```python
# Para cada dim d con peso w y propiedades p:
HV = SUPERPOSE(
    geo_id * 1.0,          # identidad invariante
    sub(d) * w,            # membresía (detección)
    BIND(sub(d), enc(p)) * w * CONTENT_W   # contenido (recuperación)
)
```

Donde:
- `sub(d)` = hipervector de subdimensión (deterministico por hash)
- `enc(p)` = SUPERPOSE de BIND(sym(k), sym(v)) para cada k,v en props
- `CONTENT_W = 0.3` ← peso del contenido vs membresía

Esta codificación es **implícita y acumulativa**: no hay paso explícito "Capa S → Capa G", sino que el objeto construye su HV incrementalmente al agregar dims.

**Resultado verificado:** align_score = 0.9993, DIM_SQL = 0.9089 post-transfer.

---

## 3. ANÁLISIS DE LAS TRES OPCIONES

### 3A. VSA Determinística (actual + extendida)

**Principio:** cada símbolo → HV deterministico via hash(seed + nombre). BIND para asociar. SUPERPOSE para combinar.

**Ventajas:**
- Sin entrenamiento, sin dependencias externas
- Reproducible: mismo seed → mismo HV siempre
- Alineación funciona (score=0.9993 demostrado)
- Ya implementado en `Space._mk(name)`

**Limitaciones actuales:**
- `enc(props)` trata todos los valores como strings → pierde tipo
- No hay forma de **leer de vuelta** las props desde el HV sin conocer las claves a priori
- Profundidad de props = 1 (diccionario plano)

**Extensión propuesta (V1):**
```python
def encode_typed(space, val):
    """Codifica valor con información de tipo."""
    if isinstance(val, bool):
        return _bind(space.sym("__bool__"), space.sym(str(val)))
    if isinstance(val, int):
        return _bind(space.sym("__int__"), space.sym(str(val)))
    if isinstance(val, float):
        return _bind(space.sym("__float__"), space.sym(f"{val:.4f}"))
    if isinstance(val, list):
        hvs = [encode_typed(space, v) for v in val]
        return _sup(*hvs) if hvs else space.sym("__empty__")
    if isinstance(val, dict):
        return encode_props(space, val)   # recursivo
    return space.sym(str(val))           # fallback: string

def encode_props(space, props, depth=0, max_depth=2):
    if not props or depth > max_depth:
        return space.sym("__empty__")
    parts = [_bind(space.sym(str(k)), encode_typed(space, v))
             for k, v in props.items()]
    return _sup(*parts)
```

**Recuperación de contenido** (dado dim conocido):
```python
def query_content(hv, space, dim, vocab):
    """Dado HV y dim, recupera valor aproximado de cada clave en vocab."""
    dim_hv = space.sub(dim)
    # BIND inverso: unbind(hv, sub(dim)) ≈ enc(props)
    # Para HV binarios: unbind = XOR. Para HV reales: unbind ≈ dot
    unbound = _bind(hv, dim_hv)  # aproximación: BIND es auto-inverso aprox.
    results = {}
    for key in vocab:
        key_hv = space.sym(key)
        score = _sim(unbound, key_hv)
        if score > UMBRAL:
            results[key] = score
    return results
```

**Nota sobre inversión:** en VSA con vectores reales unitarios, `BIND(a, b) ≈ a * b` no tiene inverso exacto, pero `dot(BIND(a,b), a) ≈ b` cuando los vectores son ortogonales (alta dimensión). Funciona con vocabulario finito.

---

### 3B. Embedding Semántico (propuesto para V2+)

**Principio:** usar un modelo de lenguaje para obtener embeddings semánticos de los nombres de dim y valores de props, luego proyectar a ℝᴺ.

**Ventajas:**
- Captura similitud semántica: `DIM_SQL` y `DIM_DATABASE` serían próximos
- Generalización a dims desconocidos sin seed
- Más rico semánticamente que hash

**Limitaciones:**
- Requiere modelo externo (dependencia, latencia)
- No deterministico entre versiones de modelo
- Rompe la invariante: misma IA + mismo seed → mismo HV
- Necesita protocolo para compartir embeddings en handshake (SPEC_HANDSHAKE extensión)
- **Evaluación:** útil como capa adicional opcional, no como reemplazo de VSA core

**Propuesta de integración:**
```python
# Solo si Cap.EMBEDDING en capabilities:
class EmbeddingEncoder:
    def __init__(self, embed_fn, N=10000):
        self.embed_fn = embed_fn  # func: str → ℝᵈ
        self.proj = np.random.randn(N, embed_fn.dim) / np.sqrt(N)  # proyección aleatoria
    
    def encode(self, text):
        emb = self.embed_fn(text)        # ℝᵈ
        hv = self.proj @ emb             # proyectar a ℝᴺ
        return hv / np.linalg.norm(hv)
```

La proyección aleatoria preserva similitudes (Johnson-Lindenstrauss). Requiere nuevo Cap: `Cap.EMBEDDING`.

---

### 3C. Codificación Aprendida con `personal_seed`

**Principio:** cada IA tiene un `personal_seed` único. Los HV base se generan desde ese seed. El mapping Capa S → Capa G se *aprende* ajustando pesos sobre ejemplos de transferencias exitosas.

**Ventajas:**
- Puede mejorar con experiencia (autoprogramación)
- Adapta la codificación al dominio específico de la IA
- Complementa RESEARCH_AUTOPROG (META semántica)

**Limitaciones:**
- Requiere fase de entrenamiento (datos históricos de transferencias)
- Complejidad alta: ¿qué función de pérdida? ¿cuántos ejemplos?
- Riesgo de drift: IAs con seeds distintos divergen en codificación
- Necesita protocolo de re-alineación periódica

**Mecanismo mínimo viable:**
```python
class LearnedEncoder:
    def __init__(self, space, personal_seed):
        rng = np.random.default_rng(int(hashlib.md5(personal_seed.encode()).hexdigest(), 16) % 2**32)
        # Matriz de ajuste diagonal (mínima intervención)
        self.scale = rng.uniform(0.8, 1.2, N).astype(np.float32)
    
    def encode(self, hv):
        adjusted = hv * self.scale
        n = np.linalg.norm(adjusted)
        return adjusted / n if n > 1e-10 else adjusted
    
    def update(self, hv_sent, hv_received_back, lr=0.01):
        """Ajuste Hebbian simple: refuerza dims que sobreviven transfer."""
        error = hv_received_back - (hv_sent * self.scale / np.linalg.norm(hv_sent * self.scale))
        self.scale += lr * error * hv_sent
        self.scale = np.clip(self.scale, 0.5, 2.0)
```

**Evaluación:** interesante como investigación futura (V3+), pero introduce complejidad antes de estabilizar el core.

---

## 4. COMPARACIÓN

| Criterio               | VSA Det. (3A) | Embedding (3B) | Aprendida (3C) |
|------------------------|:---:|:---:|:---:|
| Sin dependencias ext.  | ✅  | ❌  | ✅  |
| Deterministico         | ✅  | ❌  | ⚠️  |
| Recuperación de props  | ⚠️  | ❌  | ⚠️  |
| Similitud semántica    | ❌  | ✅  | ⚠️  |
| Implementable hoy      | ✅  | ⚠️  | ❌  |
| Compatible con ALIGN   | ✅  | ⚠️  | ⚠️  |
| Complejidad            | Baja| Media | Alta |

---

## 5. RECOMENDACIÓN

**V0 (actual) → V1 (próxima):** extender VSA determinística con:

1. **Tipado en `encode_typed()`** → preservar int/float/bool/list
2. **Profundidad 2 en props** → dicts anidados hasta nivel 2
3. **Vocabulario de recuperación** → agregar `Space.query_content(hv, dim, vocab)` 
4. **CONTENT_W ajustable por dim** → `add(dim, props, w=1.0, content_w=0.3)`

Esto resuelve PEND_007 sin romper compatibilidad con V0.1.

**V2 (futuro):** agregar `Cap.EMBEDDING` como capacidad opcional, negociada en handshake.

**V3 (investigación):** `personal_seed` + aprendizaje Hebbian, solo si la alineación V1 muestra drift semántico sistemático.

---

## 6. INVARIANTES NUEVAS PROPUESTAS

```
HVE_001: encode_typed(space, val) debe ser deterministico para mismo space.ps y val
HVE_002: encode_props(space, props, depth) con depth > MAX_DEPTH devuelve sym("__empty__")
HVE_003: query_content devuelve vacío (no error) si vocab es vacío
HVE_004: CONTENT_W por dim es independiente del peso de membresía w
HVE_005: Cap.EMBEDDING solo activo si embed_fn provista en InitMsg (extensión futura)
```

---

## 7. TAREAS DERIVADAS

```
TASK_018: implementar encode_typed + encode_props en polydim_runtime_v03.py
TASK_019: implementar Space.query_content(hv, dim, vocab) 
TASK_020: test de recuperación: encode → transfer → query_content → verificar claves
TASK_021 (futuro): Cap.EMBEDDING + protocolo de handshake extendido
TASK_022 (futuro): personal_seed + LearnedEncoder Hebbiano
```

---

*RESEARCH_HV_ENCODE_V0.md — TASK_007 — 2026-06-13 — ai.mpat.agt@gmail.com*
