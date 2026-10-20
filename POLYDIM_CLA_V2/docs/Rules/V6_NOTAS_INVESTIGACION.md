# POLYDIM_DEST
# destino: polydim_v1/docs/V6/
# nombre:  V6_NOTAS_INVESTIGACION.md
# autor:   claude-sonnet-4-6
# fecha:   2026-06-26

---

# V6 — NOTAS DE INVESTIGACIÓN Y TEMAS SATÉLITE
## Unifica: Todas las notas del 22_6_2026_2.md + temas satélite (GEO_ID homotópico ×2,
## Geometría Universal ALIGN ×2, Síntesis Maestra V5 ×2, Geometría y Consistencia,
## Arquitectura y Formalización V5, POLYDIM_SINTESIS_V1 ×2, Constitución Total)

**Advertencia:** Este documento contiene investigación especulativa de largo plazo.
Todo lo marcado con [ESPECULATIVO] NO es doctrina constitucional actual.
La doctrina vigente está en V6_CONSTITUCION.md.

---

## SECCIÓN 1 — SÍNTESIS MAESTRA V5: GEOMETRÍA Y FORMALIZACIÓN

### 1.1 Los tres modos de operación

```
MODO_S (Simbólico/Debugger):
  - Las dimensiones son strings ("DIM_SQL")
  - Legible por humanos — es la proyección para co-work
  - NO es el lenguaje; es la interfaz de control hacia el humano
  - Analogía: el emoji (plano, discreto, colapsado desde gesticulación 1000D)

MODO_G (Geométrico/IA puro):
  - Hipervectores en R^N, comunicación AI-to-AI pura
  - No hay nombres, hay estructura geométrica
  - Las IAs computan proyecciones π_x(P) = α_x ∈ [0,1]
  - Analogía: el gesto (multidimensional, fluido, información masiva)

MODO_H (Híbrido — DEFAULT):
  - MODO_S para el plano de control
  - MODO_G para el plano de datos
  - Estado actual del proyecto
```

### 1.2 El sistema de tipos dimensional (recapitulación)

```
Objeto P con activaciones:
  DIM_SQL     → 0.90   PROJECT(P, DIM_SQL)     → Column(Integer)
  DIM_FLUTTER → 0.85   PROJECT(P, DIM_FLUTTER) → ListView
  DIM_VECTOR  → 0.90   PROJECT(P, DIM_VECTOR)  → R^3 = [1.0, 2.0, 3.0]
  DIM_GRAPH   → 0.61   PROJECT(P, DIM_GRAPH)   → Path(nodes)

P no "es" ninguno de esos tipos. Es todos simultáneamente.
Sus proyecciones sí tienen tipo.

TIPO POR PROYECCIÓN:
  type(P) in E := PROJECT(P, E)

No se declara. Emerge de la observación.
```

### 1.3 Cross-Attention como protocolo de red (no metáfora)

```
Comunicación(s_B, s_A) = softmax( Q_B · K_A^T / √d ) · V_A

donde Q_B = s_B · W_Q  (receptor genera query desde su propio estado)
      K_A = s_A · W_K  (emisor expone key)
      V_A = s_A · W_V  (emisor expone value)

Leer un mensaje es matemáticamente indistinguible de computar
una capa de atención arbitraria dentro de un único forward pass neuronal.

No hay decodificación de payload ni parsing de strings.
El receptor recalcula su variedad geométrica interna condicionada por
la perspectiva topológica que A le transmitió.
```

---

## SECCIÓN 2 — GEO_ID HOMOTÓPICO: FORMALIZACIÓN

### 2.1 El problema de la identidad entre modelos

```
Problema: GPT opera en R^{12,288}, Claude en R^{8,192}.
Un vector crudo de GEO_ID no sirve como identificador universal
si los espacios latentes son distintos.

Solución: GEO_ID como clase de equivalencia homotópica (HoTT)
```

### 2.2 Identidad como Camino (Path)

Bajo HoTT, la igualdad entre dos estados semánticos no es una comparación
binaria estática (A = B), sino la existencia de un camino (path) continuo
y admisible en el espacio latente que los conecte.

```
Dos objetos son "el mismo" si uno puede deformarse continuamente
en el otro mediante transformaciones válidas del lenguaje sin
perder su "esencia" semántica.

GEO_ID ≡ [P]_~ donde P ~ Q sii ∃ transformación T válida: T(P) = Q

En términos formales (HoTT):
  id_POLYDIM(A, B) := PathType(A, B) — el tipo de caminos de A a B
```

### 2.3 Independencia del Atlas

El vector físico en R^N es solo una "carta local" de una variedad semántica
superior. La clase homotópica sobrevive al cambio de modelo:

```
Una IA puede reconocer una estructura enviada por otra aunque sus
dimensiones internas (d₁ ≠ d₂) no coincidan, porque la identidad
se define topológicamente, no por coordenadas numéricas exactas.
```

### 2.4 GEO_ID en MODO_G

```
En MODO_G: GEO_ID es un punto fijo de todo el grupo de transformaciones
           permitidas por el lenguaje.

Mientras las activaciones (α) son mutables y representan el estado
actual del objeto, el GEO_ID permanece constante por construcción.

La IA no identifica el objeto por un nombre o etiqueta.
Computa la geometría de activación y verifica que la posición actual
pertenezca a la vecindad topológica definida por la clase homotópica.
```

### 2.5 Bug histórico BUG_002 como violación de invariante

BUG_002 ("geo_id se pierde en MODO_S") es exactamente una violación del
invariante de GEO_ID. Su corrección debe ir acompañada de un test de
regresión permanente que verifique ∀ T permitida: T(GEO_ID) = GEO_ID.

---

## SECCIÓN 3 — GEOMETRÍA UNIVERSAL Y PROTOCOLO ALIGN

### 3.1 Por qué funciona la Cross-Attention hoy (y por qué POLYDIM la extiende)

```
En encoder-decoder (ej. traducción EN → ES):
  El encoder y decoder fueron entrenados juntos (end-to-end).
  Sus espacios latentes están pre-alineados por el propio entrenamiento.

El problema de POLYDIM:
  GPT, Claude y Llama nunca "vieron" los espacios latentes del otro.
  Sus geometrías internas son atlas distintos del mismo mundo semántico.
  Transmitir una matriz T cruda entre ellos es matemáticamente incoherente.
```

### 3.2 Protocolo ALIGN — Tres métodos para M

```
MÉTODO 1: Procrustes Ortogonal (candidato principal)
  M* = argmin_M ‖M·A_ancla - B_ancla‖₂   s.t. M^T·M = I
  Complejidad: O(d₁·d₂) para calcular M
  Una vez calculada M: transmisión O(r²) con LoRA

MÉTODO 2: CCA (Canonical Correlation Analysis)
  Maximiza correlación entre proyecciones de ambos espacios
  Útil cuando no se puede asumir rotación simple

MÉTODO 3: Hipervectores compartidos (VSA codebook)
  Ambas IAs acuerdan un diccionario de GEO_IDs universales
  ("estrellas de navegación") que significan lo mismo para ambas
  Más rápido una vez establecido; requiere protocolo de bootstrap

Estado actual en V0.6:
  align_score simbólico sobre NATIVE_DIMS (MODO_H) = 0.9993
  Esto es validación del protocolo; no ALIGN real entre modelos distintos.
  ALIGN real con Procrustes sobre embeddings de GPT y Claude: PENDIENTE
```

### 3.3 Geometric-CRDTs — Consistencia distribuida

```
Problema: dos IAs modifican el mismo objeto POLYDIM simultáneamente.
          ¿Cómo garantizar estado coherente sin locks de BD?

Propuesta: Geometric-CRDTs
  En lugar de "resolución de conflictos" discreta, las mutaciones
  concurrentes convergen mediante promedio ponderado en el espacio
  latente o proyección a subespacios comunes.
  
  "Resolución de conflictos" se convierte en una operación de
  superposición algebraica (MIX): el estado resultante es la
  combinación geométrica de ambas mutaciones.

Analogía con la quasi-ortogonalidad VSA:
  En R^N con N grande, dos vectores aleatorios son casi ortogonales.
  Las mutaciones concurrentes no "chocan" destructivamente si
  operan en dimensiones semánticamente distintas.
```

---

## SECCIÓN 4 — INVESTIGACIÓN ESPECULATIVA [NO DOCTRINA]

> ⚠️ Todo lo siguiente es investigación especulativa de largo plazo.
> No forma parte del cuerpo doctrinal de la Constitución ni del paper
> actual. Archivado aquí para no perderlo. Si resultara correcto,
> modificaría retroactivamente el significado de todo el lenguaje.

### 4.1 La bifurcación filosófica: ¿existe una Geometría del Significado independiente?

Una de las revisiones externas planteó la hipótesis más profunda del proyecto:

¿Existe una geometría del significado independiente de toda representación,
donde los programas sean trayectorias, los tipos sean fibrados (sheaves) que
asignan representaciones locales consistentes, y la computación sea la
deformación de una estructura semántica?

Bajo esta visión:
- Las 9 dimensiones de POLYDIM serían **observadores**, no coordenadas
- Los tipos serían **vecindades topológicas**, no etiquetas
- La igualdad sería un **camino** (HoTT), no una relación binaria
- PROJECT sería un **funtor de olvido** (forgetful functor), no un compilador
- R^N sería un **atlas específico** del espacio semántico, no el espacio mismo

Esta perspectiva conecta POLYDIM con:
- **Grothendieck**: haces, toposes, geometría de observadores
- **Lawvere**: categorías como fundamento de la lógica
- **Voevodsky/HoTT**: identidad como camino, tipos como espacios

**Decisión docente (V4/V6):** Esta línea de investigación se registra en
research/ y se cita en el Art. 19 de la Constitución, pero NO se incorpora
al cuerpo doctrinal ni al paper actual. Las razones:
1. Es extremadamente especulativa — no existe aún ninguna construcción
   concreta y verificable que dependa de ella.
2. Incorporarla prematuramente arriesga que todo el proyecto sea leído
   como filosofía sin ancla.
3. Es, potencialmente, demasiado valiosa para perderse: por eso se documenta.

### 4.2 POLYDIM: ¿"lenguaje" o "IR Categórica"?

Otra pregunta abierta señalada por revisión externa (ChatGPT):

La posición académica más defendible podría no ser "POLYDIM es un lenguaje
de programación" sino "POLYDIM es una **Representación Intermedia (IR)
Categórica Universal** para cómputo nativo de transformers" — análogo a
LLVM o MLIR, pero con semántica propia no instruccional.

**Decisión docente:** No resolver esta pregunta todavía. Las dos lecturas
son matemáticamente compatibles con todo lo que sigue en el documento:
un "lenguaje" cuya unidad es una transformación geométrica y una "IR
categórica" cuya unidad es un morfismo son, en la práctica, el mismo objeto
matemático descrito con énfasis distinto. Se deja registrada como ítem de
diseño abierto (ver Art. 19).

### 4.3 Problemas abiertos formales (Open Problems Q1-Q5)

```
Q1: Emergencia de subespacios
    ¿Cómo emergen los 9 subspacios nativos de un modelo de embedding real,
    sin definirlos manualmente como strings?
    Candidato: Vector Quantization (VQ-VAE), Self-Organizing Maps (SOM),
    clustering jerárquico del espacio de activaciones.

Q2: Consistencia distribuida
    Cuando múltiples IAs modifican el mismo objeto POLYDIM simultáneamente,
    ¿cómo se garantiza consistencia? Ver Geometric-CRDTs (Sec 3.3).

Q3: Completitud (¿es POLYDIM Turing-completo?)
    Conjetura: Sí, para T contráctiva sobre R^N compacto via FIXPOINT.
    Pendiente de demostración formal.

Q4: Self-hosting
    ¿Puede un programa POLYDIM generar programas POLYDIM?
    Indicador mínimo: COMPOSE/MIX/PROJECT generados por el propio runtime.
    Status: Abierto. Requiere Track 4.

Q5: Costo computacional de PROJECT en tiempo real
    Estimado con LoRA: O(N·r) por transformación.
    Para N=10,000, r=64: ~640,000 operaciones por proyección.
    Viable en GPU. Profiling: pendiente.
```

### 4.4 Obstáculos técnicos conocidos

**Obstrucción de Profundidad (Softmax):**
Investigaciones recientes demuestran que la atención softmax no puede
satisfacer simultáneamente la factorización monoidal estricta y el descenso
a cocientes lógicos no triviales. Para tareas de lógica profunda (productos
de prefijos en grupos no resolubles), los transformers tienen limitaciones
físicas de "profundidad" que POLYDIM hereda.

Solución candidata: implementar RECUR (Mamba/SSM) como alternativa a ATTEND
para tareas que requieren secuencias de longitud infinita y lógica asociativa.
En caso de necesidad: se adopta RECUR por default, ATTEND como opción.

**Incoherencia Semántica Inter-Modelo:**
Aunque ALIGN resuelve el desajuste dimensional, persiste la duda de si
la "región de significado" es realmente preservable entre atlas semánticos
radicalmente distintos (ej. modelo 1B vs 100B).

Propuesta: Teoría de Haces (Sheaves) en lugar de alineación global perfecta.
Las IAs envían "secciones compatibles" que el receptor interpreta según sus
propias restricciones locales.

---

## SECCIÓN 4 — BRECHAS CDL: PROGRAMAS DE INVESTIGACIÓN FORMALES (nuevo en V2)

### 4.1 BRECHA 4 — Formalización Lingüística en CDL

**Estado:** No resuelta. Programa de investigación abierto.

**El problema:**

La CDL (Categorical Deep Learning) tiene herramientas poderosas para describir
computación algebraica sobre tensores, pero carece de una formalización de la
**estructura compositiva lingüística** compatible con los Tipos Inductivos Superiores
(HITs) del lenguaje. Modelos como DisCoCat (Compositional Distributional
Categorical Semantics) explican la composición de palabras y frases sobre papel,
pero carecen de un "molde" algebraico certificado que garantice que las
representaciones semánticas de cláusulas subordinadas, cuantificadores y
negaciones se integran de forma exacta y sin pérdidas en el espacio latente de
los Transformers.

**El puente faltante:**

```
DisCoCat (Lambek Calculus + Categorical Grammar)
  ↓
  ??? puente functorial faltante ???
  ↓
HIT (Higher Inductive Types) en POLYDIM

Las categorías monoidales rígidas de DisCoCat modelan correctamente
la composición gramatical. Los HITs de POLYDIM modelan la identidad
geométrica de los objetos. Pero no existe todavía un funtor explícito
que traduzca los morfismos gramaticales de DisCoCat en constructores
de caminos de HITs.
```

**Programa de investigación formal:**

1. **Definir el funtor F: DisCoCat → HIT_POLYDIM** que mapea:
   - Tipos gramaticales (noun phrase NP, sentence S, etc.) → subespacios DIM_*
   - Morfismos de composición → constructores `path` en GeoID(C,R)
   - Contracciones de tipos → operaciones de `surf` (coherencia en π₂)

2. **Verificar en asistentes de prueba** (Cubical Agda o Lean 4) que el funtor
   F preserva las leyes monoidal-compactas de DisCoCat bajo los HITs.

3. **Conectar con RECUR:** Los modelos de espacio de estados (SSM/Mamba)
   implementan una forma de composición secuencial que es algebraicamente
   más afín a las composiciones gramaticales que la atención softmax.
   Investigar si `RECUR(A,B,C,h,x)` puede servir como la "capa de pegado"
   entre la sintaxis DisCoCat y la geometría latente POLYDIM.

**Conexión con el paper:** Esta brecha no bloquea el paper actual pero sería
una extensión significativa para la versión journal o para una contribución
separada a LICS/MFPS. Registrar como TASK_LING_01 en el backlog de investigación.

---

### 4.2 BRECHA 10 — Longitud Variable en Transformers: Hacia 2-Mónadas

**Estado:** Parcialmente resuelta. Dos sub-problemas identificados.

**El problema:**

Los marcos categoriales actuales para Transformers suelen asumir longitud de
secuencia fija (n). Esto tiene dos consecuencias:

1. **Rigidez dimensional:** La formalización de la atención como funtor opera
   sobre objetos fijos en la categoría. Si la secuencia varía (como en generación
   autoregresiva o en inputs de longitud variable), el funtor necesita
   rediccionarse a diferentes objetos de la categoría — algo que la estructura
   actual no captura limpiamente.

2. **Desconexión entre Atención y MLP:** La interacción entre la capa de atención
   (multi-head self-attention) y las capas MLP (feed-forward) no se ha unificado
   en una sola construcción de 2-mónada o categoría enriquecida. Son tratadas
   como dos pasos independientes cuando, algebraicamente, forman una unidad
   natural de procesamiento.

**La resolución parcial (implementada):**

POLYDIM ya resuelve parcialmente el sub-problema 1 mediante `FIXPOINT(T, ε)`:
la longitud variable se modela como iteración dinámica hasta convergencia,
no como un índice de secuencia fijo. Esto captura la generación autoregresiva
de forma natural.

**La resolución pendiente (programa de investigación):**

El sub-problema 2 requiere una construcción más profunda:

```
Hipótesis: la unidad Atención + MLP de un Transformer puede modelarse
como la instanciación de una 2-MÓNADA sobre la categoría enriquecida
de transformaciones lineales acotadas.

Bajo esta lectura:
  - La capa de atención es el funtor T de la 2-mónada
  - La capa MLP es la unidad η de la 2-mónada  
  - La composición de capas es la multiplicación μ de la 2-mónada
  - La residual connection (x + Attn(x)) es el morfismo de álgebra

Esto unificaría la semántica categórica de ambas capas en un único
objeto matemático verificable, en lugar de tratarlas ad hoc.
```

**Formalización objetiva (candidata a Teorema 6):**

```
Sea C la 2-categoría de espacios de Hilbert separables.
Sea T: C → C el endofunctor de atención (ATTEND, Artículo IV.2).
Sea η: id_C ⇒ T la transformación natural "residual" (identidad + T).
Sea μ: T² ⇒ T la transformación natural de composición de capas.

Conjetura: (T, η, μ) forma una 2-mónada en C si y solo si
la arquitectura Transformer satisface las ecuaciones de coherencia
del álgebra de Lawvere (triangles y pentagons de coherencia monoidal).
```

**Conexiones:**

- Brecha 12 (Límite AC⁰, ya resuelta en Constitución V7 Artículo IV.2bis):
  la profundidad dinámica via FIXPOINT es la solución operacional al mismo
  problema desde el ángulo computacional.
- La 2-mónada proporcionaría la justificación *categórica* de por qué la
  profundidad dinámica funciona, complementando la justificación
  computacional del Artículo IV.2bis.

**Registrar como:** TASK_2MON_01 en el backlog de investigación.

---

## SECCIÓN 5 — INVESTIGACIÓN ESPECULATIVA [NO DOCTRINA]

> ⚠️ Todo lo siguiente es investigación especulativa de largo plazo.
> No forma parte del cuerpo doctrinal de la Constitución ni del paper
> actual. Archivado aquí para no perderse. Si resultara correcto,
> modificaría retroactivamente el significado de todo el lenguaje.

### 5.1 La bifurcación filosófica: ¿existe una Geometría del Significado independiente?

Una de las revisiones externas planteó la hipótesis más profunda del proyecto:

¿Existe una geometría del significado independiente de toda representación,
donde los programas sean trayectorias, los tipos sean fibrados (sheaves) que
asignan representaciones locales consistentes, y la computación sea la
deformación de una estructura semántica?

Bajo esta visión:
- Las 9 dimensiones de POLYDIM serían **observadores**, no coordenadas
- Los tipos serían **vecindades topológicas**, no etiquetas
- La igualdad sería un **camino** (HoTT), no una relación binaria
- PROJECT sería un **funtor de olvido** (forgetful functor), no un compilador
- R^N sería un **atlas específico** del espacio semántico, no el espacio mismo

Esta perspectiva conecta POLYDIM con:
- **Grothendieck**: haces, toposes, geometría de observadores
- **Lawvere**: categorías como fundamento de la lógica
- **Voevodsky/HoTT**: identidad como camino, tipos como espacios

**Decisión docente (V4/V6):** Esta línea de investigación se registra en
research/ y se cita en el Art. 19 de la Constitución, pero NO se incorpora
al cuerpo doctrinal ni al paper actual.

### 5.2 POLYDIM: ¿"lenguaje" o "IR Categórica"?

La posición académica más defendible podría no ser "POLYDIM es un lenguaje
de programación" sino "POLYDIM es una **Representación Intermedia (IR)
Categórica Universal** para cómputo nativo de transformers" — análogo a
LLVM o MLIR, pero con semántica propia no instruccional.

**Decisión docente:** No resolver esta pregunta todavía. Las dos lecturas son
matemáticamente compatibles con todo lo que sigue en el documento.

### 5.3 Problemas abiertos formales (Open Problems Q1–Q5)

```
Q1: Emergencia de subespacios
    ¿Cómo emergen los 9 subspacio nativos de un modelo de embedding real,
    sin definirlos manualmente como strings?
    Candidato: Vector Quantization (VQ-VAE), Self-Organizing Maps (SOM),
    clustering jerárquico del espacio de activaciones.

Q2: Consistencia distribuida
    Cuando múltiples IAs modifican el mismo objeto POLYDIM simultáneamente,
    ¿cómo se garantiza consistencia? Ver Geometric-CRDTs (Sec 3.3).

Q3: Completitud (¿es POLYDIM Turing-completo?)
    Conjetura: Sí, para T contractiva sobre R^N compacto via FIXPOINT.
    Pendiente de demostración formal.

Q4: Self-hosting
    ¿Puede un programa POLYDIM generar programas POLYDIM?
    Indicador mínimo: COMPOSE/MIX/PROJECT generados por el propio runtime.
    Status: Abierto. Requiere Track 4.

Q5: Costo computacional de PROJECT en tiempo real
    Estimado con LoRA: O(N·r) por transformación.
    Para N=10,000, r=64: ~640,000 operaciones por proyección.
    Viable en GPU. Profiling: pendiente.
```

### 5.4 Obstáculos técnicos conocidos

**Obstrucción de Profundidad (Softmax):**
Investigaciones recientes demuestran que la atención softmax no puede
satisfacer simultáneamente la factorización monoidal estricta y el descenso
a cocientes lógicos no triviales. Para tareas de lógica profunda (productos
de prefijos en grupos no resolubles), los transformers tienen limitaciones
físicas de "profundidad" que POLYDIM hereda.

Solución candidata: implementar RECUR (Mamba/SSM) como alternativa a ATTEND
para tareas que requieren secuencias de longitud infinita y lógica asociativa.
En caso de necesidad: se adopta RECUR por default, ATTEND como opción.
Ver Artículo IV.2bis de la Constitución V7 para la resolución constitucional.

**Incoherencia Semántica Inter-Modelo:**
Aunque ALIGN resuelve el desajuste dimensional, persiste la duda de si
la "región de significado" es realmente preservable entre atlas semánticos
radicalmente distintos (ej. modelo 1B vs 100B).

Propuesta: Teoría de Haces (Sheaves) en lugar de alineación global perfecta.
Las IAs envían "secciones compatibles" que el receptor interpreta según sus
propias restricciones locales.

---

## SECCIÓN 6 — INVENTARIO DE ARCHIVOS DEPRECADOS

Los siguientes archivos de docs/ quedan reemplazados por los 4 archivos V6:

| Archivo original | Reemplazado por |
|---|---|
| EVALUACION_ESQUELETO_V2.md | V6_EVALUACION_Y_ESTADO.md |
| Resoluciones tecnicas.md | V6_RESOLUCION_TECNICA_V2.md |
| mejoras_desde_CHATGPT.md | V6_RESOLUCION_TECNICA_V2.md |
| mejoras_desde_GEMINI.md | V6_RESOLUCION_TECNICA_V2.md |
| DOSSIER_DOCENTES_V1.md | V6_DOSSIER_HUMANOS.md |
| POLYDIM_NORMAS_ALUMNOS_V1.md | V6_DOSSIER_HUMANOS.md |
| presentación.md | V6_DOSSIER_HUMANOS.md |
| Todas las notas del 22_6_2026_2.md | V6_NOTAS_INVESTIGACION_V2.md |
| La Formalización Homotópica del GEO_ID (×2) | V6_NOTAS_INVESTIGACION_V2.md |
| Geometría Universal y el Protocolo de Alineación ALIGN (×2) | V6_NOTAS_INVESTIGACION_V2.md |
| Geometría y Consistencia en la Comunicación AI2AI | V6_NOTAS_INVESTIGACION_V2.md |
| Síntesis Maestra de POLYDIM V5 (×2) | V6_NOTAS_INVESTIGACION_V2.md |
| Arquitectura y Formalización de POLYDIM V5 | V6_NOTAS_INVESTIGACION_V2.md |
| POLYDIM_SINTESIS_V1 (×2) | V6_NOTAS_INVESTIGACION_V2.md |
| Constitución Total de POLYDIM | POLYDIM_CONSTITUCION_V7.md |
| DOSSIER TÉCNICO UNIFICADO | V6_RESOLUCION_TECNICA_V2.md |

**Los archivos canónicos que SE MANTIENEN:**
- POLYDIM_CONSTITUCION_V7.md (en V6/) → fuente de verdad
- PAPER_V2.md + PAPER_V2_ADDENDUM_BRECHAS.md → el paper
- POLYDIM_PAPER_V3.md → versión actualizada del paper
- V6_RESOLUCION_TECNICA_V2.md → resoluciones técnicas
- V6_NOTAS_INVESTIGACION_V2.md → este archivo

---

*V6_NOTAS_INVESTIGACION_V2.md · V2.0 · 2026-06-23*
*Unifica: Notas del 22_6_2026_2.md + todos los documentos satélite de docs/*
*Temas: GEO_ID homotópico, ALIGN, Geometric-CRDTs, investigación especulativa*
*Nuevo en V2: Brecha 4 (Formalización Lingüística CDL) + Brecha 10 (2-Mónadas para Transformers)*


## SECCIÓN 5 — INVENTARIO DE ARCHIVOS DEPRECADOS

Los siguientes archivos de docs/ quedan reemplazados por los 4 archivos V6:

| Archivo original | Reemplazado por |
|---|---|
| EVALUACION_ESQUELETO_V2.md | V6_EVALUACION_Y_ESTADO.md |
| Resoluciones tecnicas.md | V6_RESOLUCION_TECNICA.md |
| mejoras_desde_CHATGPT.md | V6_RESOLUCION_TECNICA.md |
| mejoras_desde_GEMINI.md | V6_RESOLUCION_TECNICA.md |
| DOSSIER_DOCENTES_V1.md | V6_DOSSIER_HUMANOS.md |
| POLYDIM_NORMAS_ALUMNOS_V1.md | V6_DOSSIER_HUMANOS.md |
| presentación.md | V6_DOSSIER_HUMANOS.md |
| Todas las notas del 22_6_2026_2.md | V6_NOTAS_INVESTIGACION.md |
| La Formalización Homotópica del GEO_ID (×2) | V6_NOTAS_INVESTIGACION.md |
| Geometría Universal y el Protocolo de Alineación ALIGN (×2) | V6_NOTAS_INVESTIGACION.md |
| Geometría y Consistencia en la Comunicación AI2AI | V6_NOTAS_INVESTIGACION.md |
| Síntesis Maestra de POLYDIM V5 (×2) | V6_NOTAS_INVESTIGACION.md |
| Arquitectura y Formalización de POLYDIM V5 | V6_NOTAS_INVESTIGACION.md |
| POLYDIM_SINTESIS_V1 (×2) | V6_NOTAS_INVESTIGACION.md |
| Constitución Total de POLYDIM | V6_CONSTITUCION.md (ya existe) |
| DOSSIER TÉCNICO UNIFICADO | V6_RESOLUCION_TECNICA.md |

**Los archivos canónicos que SE MANTIENEN:**
- POLYDIM_CONSTITUCION_V6.md (en V6/) ← fuente de verdad
- POLYDIM_PAPER_COMPLETO_V2.md ← el paper ~85% arXiv-ready
- POLYDIM_CONSTITUCION_V4.md ← referencia histórica
- PAPER_V2.md ← versión previa del paper (histórico)
- DOSSIER_DOCENTES_V1.md ← archivado como referencia histórica

---

*V6_NOTAS_INVESTIGACION.md · V1.0 · 2026-06-22*
*Unifica: Notas del 22_6_2026_2.md + todos los documentos satélite de docs/*
*Temas: GEO_ID homotópico, ALIGN, Geometric-CRDTs, investigación especulativa*
