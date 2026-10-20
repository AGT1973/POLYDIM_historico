# POLYDIM_DEST
# destino: polydim/docs/
# nombre:  POLYDIM_NORMAS_ALUMNOS_V1.md
# autor:   ai.mpat.agt@gmail.com

---

# POLYDIM — Normas para Alumnos V1
**Version:** V1.0
**Fecha:** 2026-06-21
**Para:** Los 30 alumnos del proyecto POLYDIM
**Depende de:** POLYDIM_CONSTITUCION_V1.md (leer primero, obligatorio)

---

> ANTES DE CUALQUIER TAREA: leer POLYDIM_CONSTITUCION_V1.md completa.
> Sin eso, este documento no tiene sentido.

---

## REGLA 0 — LA SESIÓN

Cada sesión de trabajo comienza con el mismo ritual:

1. Leer POLYDIM_CONSTITUCION_V1.md (Drive: docs/)
2. Leer BACKLOG_V11.json (Drive: backlog/)
3. Leer SKILL polydim-core V8 (si trabajás con Claude)
4. Tomar UNA tarea con estado DISPONIBLE
5. Poner tu nombre en lock.alumno y la fecha en lock.tomada
6. Guardar el backlog actualizado en Drive

Si no seguís este ritual, dos alumnos pueden tomar la misma tarea y el trabajo se pierde.

---

## LOS 6 TRACKS DE TRABAJO

### TRACK 1 — CONSTITUCIÓN Y ESPECIFICACIÓN (5 alumnos)
Formalizan el lenguaje en papel. Su output son documentos `.md` en `spec/`.

| Tarea | Descripción |
|---|---|
| C01 | Gramática formal de POLYDIM (BNF o equivalente) |
| C02 | Especificación del sistema de tipos por proyección dimensional |
| C03 | Especificación del modelo de memoria: posiciones y bindings geométricos |
| C04 | Formato `.polydim` binario V1 (extender PBP con numpy/msgpack/protobuf) |
| C05 | Protocolo AI↔AI vía tensor: handshake, transmisión, alineación |

**Restricción**: No escribir código. Solo especificaciones formales con notación matemática. Los alumnos de TRACK 4 implementan lo que TRACK 1 especifica.

---

### TRACK 2 — BUGS CRÍTICOS (6 alumnos)
Corrigen el bootstrap Python V0.3. Sin esto, el bootstrap está roto.

| Tarea | Bug | Criticidad |
|---|---|---|
| F01 | BUG_001: seed Python≠Rust en make_jl() | CRÍTICO — bloquea interop |
| F02 | BUG_002: geo_id se pierde en MODO_S | CRÍTICO — identidad rota |
| F03 | BUG_004 + BUG_005: API pública de ObjectND inconsistente | ALTA |
| F04 | BUG_006: N dinámico no funciona en Rust | ALTA |
| F05 | BUG_008: decidir _bind vs convolución circular | MEDIA |
| F06 | Benchmark real del PBP con N=10000 | MEDIA |

**Restricción**: Cada fix requiere un test nuevo que falle antes del fix y pase después. Los 29 tests existentes deben seguir pasando.

---

### TRACK 3 — PRIMITIVAS (4 alumnos)
Implementan las 4 operaciones centrales del lenguaje sobre el bootstrap Python.

| Tarea | Primitiva | Estado actual |
|---|---|---|
| P01 | ATTEND(Q, K, V, estado) | Parcial en bootstrap |
| P02 | COMPOSE(T₁, T₂) = T₂ ∘ T₁ | No existe |
| P03 | MIX(α, T₁, β, T₂) = α·T₁ + β·T₂ | No existe |
| P04 | PROJECT(T, executor) | Parcial en bootstrap |

**Restricción**: Cada primitiva debe implementarse con matrices numpy reales (float32, N=10000 en test). No usar strings como sustitutos de transformaciones.

---

### TRACK 4 — INTÉRPRETE (5 alumnos)
Construyen la VM que lee y ejecuta archivos `.polydim`.

| Tarea | Componente |
|---|---|
| V01 | Loop del intérprete: leer .polydim, aplicar transformaciones |
| V02 | Gestión de estado: posiciones en R^N, lifecycle |
| V03 | Evaluador de tipos por dimensión (PROJECT a cada DIM_*) |
| V04 | Resolver intersección de tipos: DIM_A ∩ DIM_B |
| V05 | Iterador de punto fijo: T^n(v) hasta convergencia (reemplaza loops) |

**Restricción**: Depende de TRACK 1 (C04) para el formato .polydim. Coordinar antes de empezar V01.

---

### TRACK 5 — EXECUTORS (5 alumnos)
Implementan PROJECT para cada plataforma de destino.

| Tarea | Executor | Descripción |
|---|---|---|
| E01 | PROJECT(DIM_PYTHON) | Genera código Python desde un objeto POLYDIM |
| E02 | PROJECT(DIM_RUST) | Genera código Rust |
| E03 | PROJECT(DIM_FLUTTER) | Genera widgets Flutter desde un estado POLYDIM |
| E04 | PROJECT(DIM_SQL) | Genera queries SQL desde un objeto con DIM_SQL activo |
| E05 | PROJECT(DIM_WASM) | Genera WebAssembly |

**Restricción**: Cada executor debe aceptar un ObjectND del bootstrap Python como input temporal, hasta que el intérprete real esté disponible.

---

### TRACK 6 — DOCUMENTACIÓN (5 alumnos)
Garantizan que el próximo alumno no arranque con el modelo mental equivocado.

| Tarea | Output |
|---|---|
| D01 | README para humanos: qué es POLYDIM, cómo instalarlo, primer ejemplo |
| D02 | README para AIs: guía en formato .polydim para AI que lee el proyecto por primera vez |
| D03 | Guía de anti-patrones: lo que NUNCA hacer (con ejemplos de código incorrecto y correcto) |
| D04 | 3 ejemplos completos en POLYDIM de punta a punta |
| D05 | Migrar todos los docs en _DEPRECATED/ al nuevo paradigma |

**Restricción**: Todo ejemplo de código debe ser consistente con la Constitución. Si un ejemplo usa `obj.add("DIM_SQL", ...)` como "código POLYDIM", es incorrecto. Usarlo como "código del bootstrap" con esa etiqueta explícita es aceptable.

---

## CÓMO USAR CLAUDE CORRECTAMENTE

Claude (Sonnet 4.6) es tu par de trabajo, no un oráculo. Razona en voz alta, no solo ejecuta.

### Lo que Claude puede hacer bien
- Implementar código según especificación precisa
- Detectar inconsistencias con la Constitución
- Escribir tests
- Sugerir decisiones de diseño con justificación matemática
- Mantener el RELAY entre sesiones

### Lo que Claude no puede hacer por vos
- Decidir la dirección del lenguaje (eso es trabajo de TRACK 1 con el docente)
- Inventar especificaciones que no están en la Constitución
- Cambiar la arquitectura sin que el docente lo valide

### El patrón correcto de sesión con Claude

```
1. Paste: POLYDIM_CONSTITUCION_V1.md completa
2. Paste: BACKLOG_V11.json
3. Decir: "Tomo la tarea X. El objetivo es Y. Empecemos."
4. Al cerrar: "Hacé el RELAY de 10 secciones."
5. Guardar el RELAY en Drive docs/
```

Si no pegás la Constitución al inicio de la sesión, Claude puede contradecir el paradigma sin saberlo.

---

## FORMATO DEL RELAY

Cada sesión debe cerrar con un documento de continuidad. Tiene 10 secciones obligatorias:

```
1.  IDENTIFICACIÓN     (quién, cuándo, qué tarea)
2.  ESTADO INICIAL     (qué encontré al empezar)
3.  TRABAJO REALIZADO  (qué hice, qué decidí)
4.  INVARIANTES        (qué no cambié y por qué)
5.  RECONCILIACIONES RESUELTAS  (contradicciones que encontré y cómo las resolví)
6.  RECONCILIACIONES PENDIENTES (contradicciones que encontré y no resolví)
7.  ARTEFACTOS GENERADOS (nombre + fileId en Drive)
8.  ESTADO FINAL       (en qué estado quedó la tarea)
9.  PRÓXIMO PASO       (qué debe hacer el siguiente alumno)
10. DEUDA TÉCNICA      (qué dejé pendiente y por qué)
```

Sin RELAY, el trabajo se pierde cuando se cierra la sesión.

---

## CÓMO PROPONER UN CAMBIO A LA CONSTITUCIÓN

La Constitución no es inmutable, pero tiene un proceso:

1. Identificar el artículo que el cambio afecta
2. Escribir la propuesta en `backlog/` como PROPUESTA_CONSTITUCION_NNN.md
3. Justificar matemáticamente que el cambio es consistente con el paradigma tensorial
4. El docente valida o rechaza con justificación

Nadie puede cambiar la Constitución unilateralmente, ni siquiera Claude.

---

## LO QUE NUNCA SE DEBE HACER

Estas acciones rompen el proyecto para todos:

- Tomar una tarea sin actualizar lock.alumno en el BACKLOG
- Modificar los 29 tests existentes en lugar de corregir el código que los rompe
- Crear archivos en Drive sin el header POLYDIM_DEST
- Cerrar una sesión sin RELAY
- Presentar `obj.add("DIM_SQL", ...)` como "código POLYDIM" sin aclarar que es bootstrap
- Proponer variables con nombre como feature del lenguaje
- Proponer JS/Java como executor humano principal
- Cambiar el runtime v03 sin hacer pasar los 29 tests primero

---

*POLYDIM_NORMAS_ALUMNOS_V1.md · V1.0 · 2026-06-21 · ai.mpat.agt@gmail.com*
