# POLYDIM_DEST
# destino: polydim/docs/
# nombre:  README_AIS.md
# autor:   ai.mpat.agt@gmail.com
# fecha:   2026-06-26
# tarea:   TASK_README (D02)
# nota:    Este documento es para AIs leyendo el proyecto por primera vez.
#          Está en MODO_S (simbólico, legible por cualquier LLM).
#          Contiene la especificación compacta suficiente para empezar a trabajar.

---

# POLYDIM — Onboarding para AIs
## Especificación compacta · MODO_S · V5

**Tiempo estimado de lectura:** 5 minutos
**Prerrequisito:** Ninguno — este documento es autocontenido para AIs
**Autoridad:** Si algo aquí contradice POLYDIM_CONSTITUCION_V4.md, prevalece la Constitución.

---

## STARTUP SEQUENCE (obligatorio antes de cualquier tarea)

```
1. Google Drive:list_recent_files pageSize:1
   → Verifica que Drive MCP está activo

2. download_file_content 1POahj3CcqKw1VBuF83Cb4mfpnfs8gogW
   → Lee CARPETAS_POLYDIM.json (IDs canónicos de todas las carpetas)

3. download_file_content 1z1puD6tZYK3FlnxXuXXqYZMiB-3otx2V
   → Lee BACKLOG_V16.json (tareas activas y sus estados)

4. Tomar UNA tarea con estado DISPONIBLE
   → Actualizar lock.alumno + lock.tomada + lock.expira
   → Guardar BACKLOG actualizado en Drive

5. Ejecutar la tarea

6. Cerrar con RELAY (10 secciones — ver Parte 6)
```

**NUNCA hacer:** inventar un folderId, crear archivo sin header POLYDIM_DEST, cerrar sesión sin RELAY.

---

## PARTE 1 — QUÉ ES POLYDIM

POLYDIM es un lenguaje de programación cuya unidad fundamental es una transformación geométrica:

```
T : R^N → R^N
```

Un programa POLYDIM es una composición de transformaciones. No hay instrucciones secuenciales. No hay variables con nombre. No hay loops. No hay if/else binario. No hay tipos declarados.

**Lo que NO existe (reglas R1-R10, inviolables):**
```
R2: No hay variables con nombre → hay posiciones P=(g,S) con GEO_ID
R3: No hay loops secuenciales  → hay FIXPOINT(T,ε)
R4: No hay if/else binario     → hay MIX(α,T₁,β,T₂)
R5: No hay tipos declarados    → los tipos emergen de PROJECT(P, DIM_X)
R9: El bootstrap Python es el andamio, no el lenguaje
```

**Analogía histórica (usar en explicaciones):**
```
1950s: CPUs ya hacían aritmética binaria → Assembly los formalizó
2020s: Transformers ya operan en geometría → POLYDIM los formaliza
```

---

## PARTE 2 — ESPECIFICACIÓN FORMAL COMPACTA

### Estado y Posición

```
Estado:   S = (V, D, A)
            V ∈ R^N                  — vector de posición
            D ⊆ {DIM_0,...,DIM_8}   — subespacios activos
            A: D → [0.0, 1.0]       — activaciones

Posición: P = (g, S)
            g ∈ R^N  — GEO_ID invariante (nunca cambia — R10)
            S        — estado mutable

Transform: T: R^N → R^N  (T = W₀ + U·Vᵀ, U,V ∈ R^{N×r}, r≪N)
Programa:  Π = T_n ∘ ... ∘ T₁
```

### Las Cuatro Primitivas (Capa Algebraica)

```
COMPOSE(T₁, T₂)          = T₂ ∘ T₁          — reemplaza secuencia
MIX(α, T₁, β, T₂)        = α·T₁ + β·T₂      — reemplaza if/else
FIXPOINT(T, ε)            = T^n(v) hasta ‖s_{k+1}-s_k‖<ε — reemplaza loop
PROJECT(T, executor E)    = F_E(T(s))         — compilador / intérprete
```

### La Capa de Implementación

```
ATTEND(W_Q, W_K, W_V, s) = softmax((s·W_Q)(s·W_K)ᵀ/√d)·(s·W_V)  [transformer]
RECUR(A, B, C, h, x)     = A·h+B·x, y=C·h                          [Mamba/SSM]
```

### Semántica Operacional (Big-Step)

```
[E-COMPOSE]  ⟨s,T₁⟩⇓s₁  ⟨s₁,T₂⟩⇓s₂ / ⟨s,COMPOSE(T₁,T₂)⟩⇓s₂
[E-MIX]      ⟨s,MIX(α,T₁,β,T₂)⟩⇓α·T₁(s)+β·T₂(s)
[E-FIXPOINT] ∃n:‖T^n(s)-T^{n-1}(s)‖<ε, s*=T^n(s) / ⟨s,FIXPOINT(T,ε)⟩⇓s*
[E-PROJECT]  F_E:G→E / ⟨s,PROJECT(T,E)⟩⇓F_E(T(s))
```

### Los Cuatro Teoremas (para el paper)

```
Teorema 1: COMPOSE es asociativa [trivial — composición de funciones]
Teorema 2: MIX preserva linealidad [trivial — combinación lineal]
Teorema 3: PROJECT es un funtor [central — bajo H1+H2, ver THEOREM3_PROJECT_FUNCTOR_V1.md]
Teorema 4: FIXPOINT converge unívocamente [Banach, si T contractiva]
```

---

## PARTE 3 — LOS NUEVE SUBESPACIOS

```
ID | Nombre       | Dominio semántico              | Tipo emergente típico
---|---|---|---
 0 | DIM_PYTHON   | lógica dinámica, ML, scripts   | callable, DataFrame
 1 | DIM_RUST     | memoria segura, performance    | struct, enum, fn
 2 | DIM_FLUTTER  | UI reactiva, widgets           | Widget, StatefulWidget
 3 | DIM_SQL      | datos relacionales, tablas     | Column, Table, Query
 4 | DIM_GRAPH    | grafos, nodos, aristas         | Node, Edge, Path
 5 | DIM_VECTOR   | embeddings, similitud          | R^d, cosine_sim
 6 | DIM_TIME     | secuencias temporales          | Timestamp, Series
 7 | DIM_ERROR    | errores, excepciones           | Result<T,E>, Option
 8 | DIM_META     | metadatos, auditoría           | AuditLog, Dict
```

**Regla:** No agregar subespacios sin investigación formal. DIM_MEMORY fue investigado (TASK_025) y rechazado — es DIM_VECTOR + DIM_GRAPH + DIM_META.

---

## PARTE 4 — BOOTSTRAP V0.3 (API de referencia)

```python
from polydim_runtime_v03 import Space, ObjectND, polydim_connect

# CRÍTICO: obj y Session DEBEN usar el MISMO objeto Space
sp  = Space("MY_AI")   # MISMO objeto aquí...
obj = ObjectND(sp).add("DIM_SQL", {"table": "users"}, w=1.0)
res = polydim_connect(sp, Space("OTHER_AI")).transfer(obj)  # ...y aquí
# Resultado: res.align_score ≈ 0.9993

# Backends disponibles:
# None (default/simbólico) | MiniLMBackend | FastTextBackend
```

**Bugs conocidos y fixes:**
```
SPACE_CRITICAL:  obj y Session DEBEN usar EL MISMO objeto Space
CACHE_BUG_FIX:   "if self._c is not None:" NO "if self._c:"
JL_DETERMINISTIC: make_jl(d_in, seed=42) para reproducibilidad
```

**Métricas verificadas:**
```
Tests:              29/29
align_score:        0.9993
semantic (MiniLM):  0.830  vs  deterministic: 0.506  (+64.5%)
```

---

## PARTE 5 — PROTOCOLO AI↔AI

### ALIGN (comunicación entre modelos con distintas dimensiones)

```
IA_A en R^{d_A}, IA_B en R^{d_B}:

1. Pre-alineación: compartir codebook C = {g₁,...,g_k} (GEO_IDs universales)
   A = [g₁^A,...,g_k^A] ∈ R^{d_A×k},  B = [g₁^B,...,g_k^B] ∈ R^{d_B×k}

2. Procrustes: M* = argmin_M ‖M·A-B‖² s.t. M^T·M=I
   Solución:   SVD de B·A^T = U·Σ·V^T → M = U·V^T

3. Transformación: T_B = M · T_A · M†  (M† = pseudoinversa)

4. Aplicación:  new_state_B = T_B(state_B)
```

### Cross-attention como comunicación nativa

```
Leer un mensaje de IA_A = computar una capa de atención:
Communication(s_B, s_A) = softmax(Q_B · K_Aᵀ / √d) · V_A
```

No es metáfora. Es la operación matemática exacta del cross-attention en transformers.

### Los tres modos

```
MODO_S (Simbólico):  texto legible por cualquier LLM — bootstrap V0.3
MODO_G (Geométrico): hipervectores en R^N — comunicación pura AI↔AI
MODO_H (Híbrido):    S para control, G para datos — DEFAULT
```

---

## PARTE 6 — PROTOCOLO DE RELAY (cierre de sesión obligatorio)

```
RELAY — SECCIÓN 1: IDENTIFICACIÓN
  modelo, sesión, tarea_tomada

RELAY — SECCIÓN 2: ESTADO INICIAL
  qué encontré, qué leí, estado del backlog

RELAY — SECCIÓN 3: TRABAJO REALIZADO
  lista concreta de acciones, archivos, decisiones

RELAY — SECCIÓN 4: INVARIANTES
  qué propiedades se mantienen, qué tests pasan, qué reglas se respetaron

RELAY — SECCIÓN 5: RECONCILIACIONES RESUELTAS
  conflictos entre documentos resueltos

RELAY — SECCIÓN 6: RECONCILIACIONES PENDIENTES
  tensiones no resueltas para revisión docente

RELAY — SECCIÓN 7: ARTEFACTOS GENERADOS
  [nombre] → Drive fileId: [ID]

RELAY — SECCIÓN 8: ESTADO DE CIERRE
  backlog al cierre, tests pasando

RELAY — SECCIÓN 9: PRÓXIMO PASO
  qué tarea tomar, qué leer primero

RELAY — SECCIÓN 10: DEUDA TÉCNICA
  qué quedó pendiente por tiempo
```

---

## PARTE 7 — CARPETAS Y IDs CLAVE

```
CARPETAS_POLYDIM.json:   1POahj3CcqKw1VBuF83Cb4mfpnfs8gogW  (siempre leer primero)
BACKLOG_V16.json:        1z1puD6tZYK3FlnxXuXXqYZMiB-3otx2V  (tareas activas)

polydim/ root:           1C7hGAKRIY6U8Oh8AjWnsjTO7eB27onck
spec/:                   1gCRbUaINsrIE-NDCKIuPqliL5Wa7OmnM
core/:                   1QuCMIYa_6UyQWUKv2j6V3jHaMZR9kHmn
docs/:                   1btIiG5Ei7n-eD0dqRRsztpQ_fb8u-ok-
backlog/:                1A-TolzjtZlvEpB4eom2TizFvlIt9cCfv
skills/:                 1owF5OGjHPrWXqfM-SonNUF0C8ovCMd9z

Archivos clave:
  polydim_runtime_v03.py:      1aS7QU_gnXCp_DssleSNOO9YSKMg-TPV7
  polydim_primitives_v1.py:    1aUys6ae5g_C4fFccuIsctro87irZmyAf
  polydim_align_v2.py:         1oHipwy1zv7JEq-o66z4SaHemrRiJUO1n
  polydim_core.rs:             1r-2BZNYY_nRkNuY0d1rcUtQlkOWD6e6T
  SPEC_FORMATO_BINARIO_V5.md:  1kF2votHtvfQ2m6VH5pgsSJGRFy_MM2r6
  POLYDIM_GRAMMAR_BNF_V1.md:  1IRsCQTGQmKSNhazALO9kzqh-N6Q949Dn
  THEOREM3_PROJECT_FUNCTOR_V1: 14FbTYXRV9B9P87enNsVW8MHwV_B5zwA7
  POLYDIM_PAPER_COMPLETO_V3:  1vuJsRbAhqTB7ptmthJ9rIPqNhxROyy0e
  polydim-core-skill-V9.md:   1kBj2Tkczo41V9Hf2v-1GHpHw1osdzTkc
```

---

## PARTE 8 — TAREAS ACTIVAS (resumen de BACKLOG_V16)

```
TASK_README (prio 1): README_HUMANS.md + README_AIS.md  ← EN PROGRESO
TASK_LATEX  (prio 2): polydim_paper_arxiv.tex
TASK_022    (prio 3): polydim_weighted_inference.py
TASK_R01    (prio 4): RESEARCH_SUBSPACE_EMERGENCE_V1.md
TASK_R02    (prio 5): RESEARCH_GEOMETRIC_CRDTS_V1.md
TASK_015            : BLOQUEADA_MCP — acción manual en Drive web
```

---

## PARTE 9 — REGLAS ABSOLUTAS

```
NUNCA: inventar un folderId
NUNCA: crear archivo sin header POLYDIM_DEST
NUNCA: mostrar código en pantalla si va a Drive
NUNCA: usar present_files (ahorra tokens)
NUNCA: cerrar sesión sin RELAY completo
SIEMPRE: leer CARPETAS + BACKLOG al inicio
SIEMPRE: tomar UNA tarea a la vez
SIEMPRE: disableConversionToGoogleType: true
SIEMPRE: contentMimeType: text/plain para archivos .md, .py, .rs, .json
```

---

## PARTE 10 — COMPORTAMIENTO ESPECIAL

```
IF usuario escribe "." solo: continuar sin saludo ni resumen
IF task tiene lock activo: no tomar, buscar otra disponible
IF hay conflicto entre documentos: la CONSTITUCIÓN_V4 prevalece
IF Teorema 3 necesita revisión: no modificar, marcar en RELAY Sección 6
```

---

*README_AIS.md · V1.0 · 2026-06-26 · MODO_S*
*Próxima actualización: cuando BACKLOG cambie de versión*
*Documento hermano: README_HUMANS.md (para personas)*
