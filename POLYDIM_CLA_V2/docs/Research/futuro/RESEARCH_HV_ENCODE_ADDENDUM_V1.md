# POLYDIM_DEST
# destino: polydim/research/
# nombre:  RESEARCH_HV_ENCODE_ADDENDUM_V1.md
# autor:   ai.mpat.agt@gmail.com

# RESEARCH_HV_ENCODE_ADDENDUM_V1
## SOTA real para MPAT — corrección al V0

**Fecha:** 2026-06-13
**Autor:** ai.mpat.agt@gmail.com
**Corrige:** RESEARCH_HV_ENCODE_V0.md (1Yjfde9EkN_r7jIZIXSsR2v4xM9s2HC4y)
**Estado:** DEFINITIVO

---

## 1. REPLANTEAMIENTO DEL PROBLEMA

El V0 preguntó: *"¿cuál es la función de codificación óptima `encode(obj) → HV`?"*

Esta es la pregunta equivocada para MPAT.

**La pregunta correcta es:** ¿cómo transfiere una IA stateless su estado cognitivo a otra instancia fresca del mismo modelo, sin pérdida de información?

La diferencia es crítica. MPAT no es un sistema de embedding de documentos. Es un protocolo de transferencia de estado entre instancias LLM sin memoria persistente.

---

## 2. REQUERIMIENTOS REALES DE MPAT

```
R1: Emisor = Claude con contexto activo
    Receptor = Claude fresca con skill POLYDIM pero sin contexto del emisor

R2: "Dims" son categorías funcionales (competencias), no tokens semánticos
    DIM_SQL = "esta IA tiene competencia SQL con weight w"
    Props = metadatos sobre esa competencia

R3: Recuperabilidad práctica
    El receptor puede leer props SI conoce las claves
    El vocabulario de claves viene del contexto del problema, no del HV

R4: Sin dependencias externas
    Sin APIs de embeddings, sin modelos auxiliares
    Funciona offline y en cualquier instancia Claude
```

---

## 3. HALLAZGOS EXPERIMENTALES

### 3.1 VSA determinística: ortogonalidad verificada

```
Media similaridades cruzadas entre NATIVE_DIMS: 0.4989 (ideal: 0.5000)
Max desviación: 0.0093

→ Los dims son ortogonales por construcción via hash.
→ Embeddings semánticos ROMPERÍAN esta propiedad.
→ 3B (embedding) es contraproducente para el core.
```

### 3.2 BIND inverso: funciona con vocabulario conocido

```
Experimento: obj con DIM_SQL, props={"tabla":"usuarios","schema":"public"}
CONTENT_W=0.3, geo_id presente

bind(obj_hv, dim_hv) → bind(result, "tabla") → sim("usuarios") = 0.5504 ✅
bind(obj_hv, dim_hv) → bind(result, "schema") → sim("public")   = 0.5530 ✅

Funciona. Pero: solo si el receptor ya sabe preguntar por "tabla" y "schema".
Las claves no son detectables directamente (score ~0.503, bajo umbral 0.510).
→ Requiere vocabulario previo = contexto del problema.
→ Esto es CORRECTO para MPAT R3, no una limitación.
```

### 3.3 3C (Learned) es incompatible con arquitectura stateless

```
Claude no tiene persistencia entre sesiones.
Sin persistencia: LearnedEncoder se resetea → equivale a VSA con seed distinto.
Con persistencia: O(N)=O(10000) por transfer, riesgo de drift entre pares.
→ 3C no es viable ahora. Documentada como investigación futura.
```

---

## 4. HALLAZGO CRÍTICO: BUG EN Session.receive()

**Este es el hallazgo más importante del análisis.**

El SOTA para MPAT no requiere un encoding más sofisticado. Requiere corregir un bug en `receive()`:

```python
# CÓDIGO ACTUAL (runtime v02fix) — BUGGEADO
def receive(self, pkt):
    if pkt.payload_G is not None: hv = pkt.payload_G    # ← en MODO_H entra aquí
    elif pkt.payload_S:                                   # ← NUNCA llega aquí en MODO_H
        o = ObjectND(self.space)
        for d, info in pkt.payload_S.get("dims", {}).items():
            o.add(d, info.get("props", {}), w=info.get("w", 1.0))
        hv = o._hv()
    else: return {}
    return {d: round(_proj(hv, self.space.sub(d)), 4)    # ← solo devuelve scores
            for d in NATIVE if _proj(hv, self.space.sub(d)) > UMBRAL}
```

**Consecuencias del bug:**
- En MODO_H: `payload_G` existe → `elif payload_S` nunca se ejecuta
- `payload_S` (que contiene props exactas) es **ignorado completamente**
- `receive()` devuelve `{dim: score}` → **props se pierden**
- Todo el contenido de `props` de cada dim se descarta en la recepción

**Corrección SOTA:**

```python
def receive(self, pkt):
    """
    MODO_H: usa G para detección (precisa) + S para props (exactas).
    MODO_G: usa G para detección, props vacías.
    MODO_S: reconstruye HV desde S, usa props de S.
    Devuelve ObjectND completo (no dict).
    """
    # 1. Extraer props de S siempre que esté disponible
    props_from_S = {}
    if pkt.payload_S:
        props_from_S = pkt.payload_S.get("dims", {})

    # 2. Detectar dims activos
    if pkt.payload_G is not None:
        hv = pkt.payload_G
    elif pkt.payload_S:
        o = ObjectND(self.space)
        for d, info in props_from_S.items():
            o.add(d, info.get("props", {}), w=info.get("w", 1.0))
        hv = o._hv()
    else:
        return ObjectND(self.space)  # vacío

    active = {d: round(_proj(hv, self.space.sub(d)), 4)
              for d in NATIVE if _proj(hv, self.space.sub(d)) > UMBRAL}

    # 3. Reconstruir ObjectND con props de S (si disponible) o vacías
    result = ObjectND(self.space)
    for d, score in active.items():
        info = props_from_S.get(d, {})
        result.add(d, info.get("props", {}), w=info.get("w", score))

    return result
```

**Impacto de la corrección:**
- `receive()` devuelve `ObjectND` en lugar de `dict`
- MODO_H = transmisión sin pérdida: G detecta, S recupera props exactamente
- MODO_G = detección precisa, props vacías (comportamiento anterior)
- MODO_S = reconstrucción completa desde simbólico (comportamiento anterior)

**BREAKING CHANGE:** `receive()` ya no devuelve `dict`. Callers deben actualizar:
```python
# Antes:
dims_activas = session_b.receive(pkt)  # {dim: score}

# Después:
obj_recibido = session_b.receive(pkt)  # ObjectND
dims_activas = obj_recibido.dims_activas()
props_sql = obj_recibido._props.get("DIM_SQL", {})
```

---

## 5. EL SOTA COMPLETO PARA MPAT

```
VSA Determinística (3A)     → encoding base correcto, no cambiar
MODO_H                      → ya implementado, es el modo correcto
receive() corregido          → bug fix, impacto máximo
encode_typed (tipado)        → mejora incremental, secundaria
query_content (BIND inverso) → útil para MODO_G puro, implementar después
```

**Por qué 3A es SOTA y las otras no:**

| Propiedad MPAT                  | 3A VSA | 3B Embeddings | 3C Learned |
|---------------------------------|:------:|:-------------:|:----------:|
| Stateless compatible            | ✅     | ✅            | ❌         |
| Ortogonalidad entre dims        | ✅     | ❌ (rompe)    | ✅         |
| Reproducible entre instancias   | ✅     | ❌            | ❌         |
| Sin dependencias externas       | ✅     | ❌            | ✅         |
| Alineación con ALIGN score>0.85 | ✅     | ⚠️ no probada | ⚠️ deriva  |
| Implementable hoy               | ✅     | ⚠️            | ❌         |

**3B** es contraproducente porque POLYDIM necesita ortogonalidad entre dims, no similitud semántica entre ellos. La similitud semántica existe *dentro* de un dim (en sus props), no *entre* dims.

**3C** es incompatible porque Claude es stateless. Un sistema de aprendizaje sin persistencia no aprende nada.

---

## 6. TAREAS CORREGIDAS (reemplaza TASK_018-020)

```
TASK_018_v2 [CRÍTICA]: Corregir Session.receive() en polydim_runtime_v03.py
  - Devuelve ObjectND en lugar de dict
  - Lee payload_S en MODO_H (fix del elif → if independiente)
  - BREAKING CHANGE: actualizar Connection.transfer() y tests

TASK_019_v2 [MEJORA]: encode_typed() + to_symbolic() con tipos
  - to_symbolic() debe incluir tipo del valor para round-trip exacto
  - encode_typed() para MODO_S/H en send()
  - Mejora fidelidad en MODO_S; en MODO_H ya es exacto via payload_S

TASK_020_v2 [UTILIDAD]: query_content(hv, dim, vocab) en Space
  - Para MODO_G puro (sin payload_S)
  - BIND inverso: bind(hv, sub(dim)) → bind(result, sym(key)) → sim(value)
  - Requiere vocabulario externo (correcto por diseño)
```

---

## 7. INVARIANTE NUEVA CRÍTICA

```
HVE_006 (CRÍTICA): receive() en MODO_H DEBE leer payload_S para props.
         payload_G se usa para detección. payload_S se usa para contenido.
         Nunca ignorar payload_S cuando payload_G está presente.
         
HVE_007: receive() devuelve ObjectND (no dict).
         El llamador es responsable de llamar .dims_activas() si necesita el dict.
```

---

## 8. CONCLUSIÓN

El SOTA para MPAT no era un algoritmo de encoding más sofisticado. Era:

1. **Reconocer** que MODO_H ya existe y ya envía S+G simultáneamente
2. **Arreglar** el bug en `receive()` que ignoraba `payload_S` en MODO_H
3. **Mantener** VSA determinística como encoding base (correcto por diseño)

El sistema ya tenía la arquitectura correcta. El bug en `receive()` impedía aprovecharla.

La corrección convierte MODO_H en un canal de transmisión sin pérdida:
- `payload_G` → detección rápida y precisa de dims activos
- `payload_S` → recuperación exacta de props sin aproximación

No se necesita encoding más complejo. Se necesita leer lo que ya se está enviando.

---

*RESEARCH_HV_ENCODE_ADDENDUM_V1.md — 2026-06-13 — ai.mpat.agt@gmail.com*
*Corrige V0. Reemplaza TASK_018-020 con versiones v2.*
