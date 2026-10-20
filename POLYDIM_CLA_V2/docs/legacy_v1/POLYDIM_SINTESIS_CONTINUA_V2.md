# POLYDIM_DEST
# destino: polydim_v1/docs/
# filename: POLYDIM_SINTESIS_CONTINUA_V2.md
# autor:    ai.mpat.agt@gmail.com (claude-sonnet-4-6)
# fecha:    2026-06-25
# DT_003 RESUELTA — reemplaza old_POLYDIM_SINTESIS_CONTINUA_V1_0_pre-conflicto-V4-V6.md
# reemplaza: old_POLYDIM_SINTESIS_CONTINUA_V1_0_pre-conflicto-V4-V6.md (en docs/old/)

---

# SÍNTESIS CONTINUA — POLYDIM_v1
### Documento operativo derivado de POLYDIM_CONSTITUCION_V6.md
### V2 · 2026-06-25 · Estado: VIGENTE

> Este documento traduce la Constitución V6 a reglas accionables para sesiones,
> skills y auditoría. **No es ley** — la ley es `POLYDIM_CONSTITUCION_V6.md`.
> En caso de discrepancia, **la Constitución prevalece**.

---

## 1. FUENTE DE VERDAD

`POLYDIM_CONSTITUCION_V6.md` (fileId `1LneDD0D8fHREiZN-_Xw-Ax9AzxfF6tKv`, en `docs/V6/`)
es la ley vigente. V6 reemplaza explícitamente V2–V5 y todos los borradores previos.

**Las tres marcas epistémicas (Art. IV.1–3):**
```
✅ LEY          normativo. No se contradice sin enmienda formal (Art. XX).
⚙️ MECANISMO   especificación técnica concreta, en implementación.
🔬 INVESTIGACIÓN  formalización avanzada, válida como hipótesis, no vinculante.
```

Regla para `EXTENDS` headers: si la sección citada es 🔬, el código **no puede**
presentarse como cumplimiento de una regla. Solo como exploración.

---

## 2. REGLAS ABSOLUTAS (R1–R13, Art. XVI) — síntesis accionable

| # | Regla (resumida) | Qué significa en la práctica |
|---|---|---|
| R1 | Unidad fundamental: T: ℝᴺ→ℝᴺ | Ningún código del núcleo es "paso a paso" — es composición de transformaciones |
| R2 | Sin variables con nombre nativas | GEO_ID + activaciones, no variables etiquetadas en el núcleo |
| R3 | Sin loops imperativos nativos | Convergencia = FIXPOINT, no `for`/`while` |
| R4 | Sin `if`/`else` binario nativo | Bifurcación = MIX ponderado |
| R5 | Tipos emergen de la proyección | PROJECT/COMPILE/RENDER/EXPORT dan el tipo; nunca anotar tipo en el núcleo |
| R6 | Python/Rust/Flutter son destinos de bootstrap, no el lenguaje base | Código Python del bootstrap siempre etiquetado como "bootstrap" |
| R7 | Canal nativo AI↔AI transmite T, no texto serializado | MODO_S para humanos; canal nativo no depende de JSON/texto |
| R8 | Flutter es el único ejecutor nativo del subespacio humano (isomorfismo demostrado, Art. X.2) | Otros frameworks requieren demostrar isomorfismo equivalente |
| R9 | Bootstrap Python = andamio, nunca "código POLYDIM" | Todo ejemplo debe decir explícitamente "código del bootstrap" |
| R10 | GEO_ID invariante en forma normativa mínima (Art. VI.3) | ∀T ∈ T_permitidas: dist(T(GEO_ID), GEO_ID) < ε |
| R11 | ATTEND y RECUR son capa de implementación (Art. IV.2) | No elevar a ley del núcleo sin enmienda |
| R12 | `.polydim` no puede usar matrices densas N×N como representación general | Por defecto: bajo rango (T = W₀ + UVᵀ, Art. XI) |
| R13 | Ninguna formalización 🔬 puede citarse como ley vinculante en un test | GEO_ID homotópico (VI.5), Teoría de Haces (VIII.4) no son tests |

---

## 3. ESTADO ACTUAL DEL CORPUS (auditoría 2026-06-25)

### Bootstrap Python (VIGENTE)
```
polydim_runtime_v04.py    fileId: 1ogmIBUQRqgYa-OaYCItBD9Co-zDQWnCd   ← ACTIVO
polydim_tests.py          29/29 tests pasan (contra V0.3, compatibles V0.4)
```

V0.3 y V0.4 son compatibles ascendentemente. La Constitución V6 Art. IX §B
referencia polydim_runtime_v03 como "el código que los 29 tests verifican" —
V0.4 pasa los mismos 29 tests sin cambios.

### Módulos activos (todos en `core/`)
```
polydim_weighted_inference.py    PEND_002 cerrado
polydim_drift_detection.py       PEND_008 cerrado
polydim_align_heterogeneous.py   PEND_010 cerrado
polydim_distributed.py           PEND_003 pragmático (⚙️)
polydim_core.rs                  prototipo Rust sin deps
polydim_integration_tests.py     9/9 tests Python
```

### Specs activas (en `spec/`)
```
SPEC_FORMATO_BINARIO_V0.md          POLYDIM_BIN: 40014 bytes, 0.005ms
SPEC_SEMANTICA_OPERACIONAL_V0.md    6 primitivas: 4 ✅ + 2 ⚙️
POLYDIM_FAITHFULNESS_FLUTTER_V1.md  Prop. 6.1 probada → Teorema 3 incondicional
POLYDIM_THREATS_TO_VALIDITY_V1.md
POLYDIM_THEOREM3_PROOF_V1.md
SPEC_GEOMETRIC_CRDT_RESEARCH_V0.md  programa investigación Geometric-CRDT
```

### Backlog
```
backlog/BACKLOG_VIGENTE.md    fileId: 1YXug4pN1BYDq6g-md6jB8BEL5HRX5ouV
Estado: vacío (sin tareas DISPONIBLES)
Deudas técnicas: DT_001–004 RESUELTAS (sesión 2026-06-25)
```

---

## 4. ADVERTENCIAS OPERATIVAS (lecciones aprendidas)

```
⚠️ BACKLOG_V29.json NO es de POLYDIM_v1 — es del proyecto hermano polydim/
⚠️ polydim_runtime_v05.py NO es de POLYDIM_v1
⚠️ polydim_core_v02-.py y polydim_core_prototype.py están NEUTRALIZADOS (REDIRECT_*)
⚠️ La skill activa en la sesión es V9_0 (no V8_0_EN ni V5_0)
⚠️ Backlogs V12/V13 en raíz del proyecto son históricos; fuente de verdad = BACKLOG_VIGENTE.md
```

---

## 5. ARXIV STATUS (PAPER_V3)

Con TASK_035 completada, PAPER_V3 está listo para arXiv:

| Requisito | Estado |
|---|---|
| Teorema 3 (capa algebraica, 3 contratos) | ✅ POLYDIM_THEOREM3_PROOF_V1.md |
| Faithfulness F_Flutter (Prop. 6.1) | ✅ POLYDIM_FAITHFULNESS_FLUTTER_V1.md |
| Threats to Validity | ✅ POLYDIM_THREATS_TO_VALIDITY_V1.md |
| Lax functor ATTEND/RECUR | ⚙️ caracterizado, no bloquea |
| Actualización §2.5 y §9 de PAPER_V3 | ⏳ TASK_036 (requiere docente) |

---

## 6. HALLAZGOS TÉCNICOS ACUMULADOS

```
ALIGN_SCORE:        sobre NATIVE_DIMS, no sobre contenido (distinción crítica)
UMBRAL:             0.5 + 2*(1/(2*sqrt(N))) = 0.510 para N=10000
CAPAS:              MEMBRESÍA (dim_sub*w) + CONTENIDO (BIND*w*0.3)
SPACE_CRÍTICO:      obj y Session deben usar el MISMO objeto Space
CACHE_BUG:          "if self._c is not None:" — nunca "if self._c:"
JL_DETERMINISTA:    make_jl(d_in) reproducible via MD5 seed
FORMATO_BINARIO:    POLYDIM_BIN — 14 bytes header + N*4 bytes HV float32 BE
DRIFT_UMBRAL:       0.15 = cambio de dominio (DriftDetector)
ALIGN_HETERO:       N_A≠N_B → JL common space (N_common=min) → score≈0.998
MERGE_PROPIEDADES:  conmutativo=1.0, idempotente=1.0 (empírico, no CRDT formal)
SEGUNDA_CUENTA:     polydim.ai.lenguage@gmail.com es el docente
```

---

## 7. HISTORIA DE VERSIONES DE ESTE DOCUMENTO

| Versión | Fecha | Estado | Reemplaza |
|---|---|---|---|
| V1.0 pre-conflicto | 2026-06-23 | OBSOLETA (en docs/old/) | — |
| V2 | 2026-06-25 | VIGENTE (este doc) | V1.0 pre-conflicto |

El conflicto V4-V6 referenciado en el nombre del archivo V1.0 fue resuelto
cuando V6 absorbió formalmente todos los contenidos válidos de V4/V5
(ver Anexo A de POLYDIM_CONSTITUCION_V6.md).

---

*POLYDIM_SINTESIS_CONTINUA_V2.md · 2026-06-25 · ai.mpat.agt@gmail.com*
*Fuente de verdad: POLYDIM_CONSTITUCION_V6.md (fileId 1LneDD0D8fHREiZN-_Xw-Ax9AzxfF6tKv)*
