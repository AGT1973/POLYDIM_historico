# POLYDIM_DEST
# destination: (carpeta compartida de resúmenes IA — ver nota de gobernanza abajo)
# filename: POLYDIM_SINTESIS_CONSOLIDADA_IA.md
# generado_por: Claude (sesión externa al proyecto, a pedido del usuario) — 2026-07-02
# fuentes_primarias: carpeta 1pYn8jFoPgTTPH0Z57bVuIZJhf4FoRuSR (Constitución/Paper) +
#                     carpeta 1czWYt3kk8qseDOxJdLlEeyzsn2-ai4hM (Paper/Apéndices/Publicación)
# NO se leyó el código fuente del runtime (ese vive en otra raíz de Drive, ver Sección 8).

# POLYDIM — Síntesis consolidada para agentes de IA

> **Propósito:** que un agente de IA entienda el estado actual de POLYDIM sin leer los
> ~62 documentos originales de las dos carpetas fuente. **No reemplaza una prueba matemática
> exacta ni el código fuente** — para eso, ir al documento citado en cada sección.
>
> **Antes de usar este archivo, leer la Sección 0.** Contiene un hallazgo que cambia cómo
> hay que tratar cualquier resumen de POLYDIM, incluido este.

---

## 0. Advertencia de gobernanza y de contradicción interna (leer primero)

**0.1 — Ya existen otros resúmenes en esta misma carpeta.** Al momento de escribir esto
(2026-07-02, ~11:45–11:53 UTC) esta carpeta ya contenía:
- `POLYDIM_RESUMEN_IA.md` (15.6 KB, cuenta `curso02.mithril@gmail.com`)
- `POLYDIM_RESUMEN_IA.md` (41 bytes, misma cuenta — **archivo roto**: su contenido es
  literalmente el string `$(cat /home/claude/POLYDIM_RESUMEN_IA.md)`, es decir, a otro
  agente le falló la sustitución de shell y subió el comando en vez de su salida)
- `POLYDIM_AI_KNOWLEDGE.json` (14.9 KB, cuenta `cursos.mithril@gmail.com`)
- `POLYDIM_KNOWLEDGE_BASE_AI.md` (9 KB, cuenta `polydim.ai.lenguage@gmail.com`)

Cuatro cuentas distintas generaron resúmenes casi al mismo minuto. Esto sugiere que la
consigna ("leé las carpetas, resumí para IA, guardá acá") se dio a varios alumnos en
paralelo sobre una carpeta compartida. Quien lea esto como docente probablemente quiera:
(a) borrar el archivo de 41 bytes (es un error, no un resumen), y (b) decidir cuál de los
resúmenes válidos es el canónico, en vez de acumular uno nuevo por cada sesión de IA que
pase por acá.

**0.2 — Los resúmenes ya existentes NO coinciden en cuál es el documento canónico**, y esto
no es un detalle menor: es la razón técnica por la que un solo resumen no alcanza todavía.
- Este documento (y `POLYDIM_PAPER_V11.md`) toman como fuente más reciente
  `CONSTITUTION_EN.md` (modificado 2026-07-01, el archivo con fecha de modificación más
  nueva de la carpeta 1) y el paper V11 (2026-06-30), que usan la tupla de estado
  `S = (V, D, A)` / posición `P = (g, S)`.
- `POLYDIM_AI_KNOWLEDGE.json` y `POLYDIM_KNOWLEDGE_BASE_AI.md` (ya en esta carpeta) declaran
  como fuente canónica `POLYDIM_CONSTITUCION_V2_CLEAN.md`, y ese documento describe una
  tupla **distinta e incompatible**: `P = (S, G, T, O, C, Π)` (Estado, Meta, Transformación,
  Observador, Restricción, Política), enmarcada como una disciplina más amplia llamada
  "Programación Cognitiva" de la que POLYDIM pasaría a ser sólo el primer caso de estudio.
- Ninguno de los ~15 archivos "CONSTITUCION*" de la carpeta 1 tiene un campo de versión que
  domine inequívocamente sobre los demás; hay al menos 8 variantes (V2, V2_CLEAN, FINAL,
  FINAL_CLEAN, FINAL_CLEAN_PRINT...) en tres formatos cada una.

**Consecuencia práctica:** si 30 agentes leen "la Constitución" sin especificar cuál, es
esperable que converjan en descripciones distintas del núcleo formal del lenguaje — que es
exactamente lo que ya pasó entre este resumen y los dos anteriores. Cualquier resumen
(incluido este) debe declarar explícitamente qué archivo tomó como fuente. Recomendación:
que el autor del proyecto (Ariel H. Garcia Traba) fije un único documento raíz y mueva el
resto a `_DEPRECATED/`, siguiendo el propio protocolo del proyecto.

---

## 1. Tesis en una frase

POLYDIM es un lenguaje de programación formal cuya unidad mínima de cómputo no es una
instrucción sino una **transformación geométrica T: R^N → R^N** (N ≥ 10.000), pensado para
que transformers y SSMs (tipo Mamba) compongan, mezclen, iteren y proyecten esas
transformaciones directamente en su espacio latente, sin serializar a texto.

**Evidencia empírica citada (de terceros, no medida por el propio POLYDIM — verificar antes
de repetir como hecho establecido, ver Sección 6):** Interlat [Du et al. 2025] — un hidden
state transportaría ~40.000 bits de información vs. ~15 bits por token de texto (~2.667× de
brecha de ancho de banda semántico), con hasta 24× de aceleración al evitar la
serialización. LatentMAS [ICML 2026 Spotlight] — alineación latente sin reentrenamiento
entre agentes. El paper de POLYDIM es explícito en que estas cifras vienen de trabajo ajeno,
no de un experimento propio.

Analogía oficial del proyecto: el ensamblador no inventó el transistor, lo hizo programable;
POLYDIM no inventa el transformer, lo hace programable.

---

## 2. Núcleo algebraico (capa invariante, independiente de arquitectura)

Fuente: `POLYDIM_PAPER_V11.md` Sec. 2.2–2.4 (la formalización más reciente y la única con
reglas big-step completas para las seis primitivas).

| Primitiva | Definición | Reemplaza a |
|---|---|---|
| `COMPOSE(T1,T2)` | T2 ∘ T1 (no conmutativa) | secuencia de instrucciones |
| `MIX(α,T1,β,T2)` | α·T1 + β·T2 (superposición continua; casi-ortogonalidad en alta dimensión evita interferencia destructiva) | if/else binario |
| `FIXPOINT(T,ε)` | itera T hasta convergencia (si T es una contracción, Banach garantiza unicidad) | loops for/while |
| `PROJECT(T,executor)` | funtor de la categoría geométrica G a la categoría del ejecutor E | tipos declarados / compilación |

**Capa de implementación (específica de arquitectura, NO parte del núcleo — es la novedad
de la V11, sección 2.4.1):**
- `ATTEND(Q,K,V,s) = softmax(Q·Kᵀ/√d)·V` — familia transformer.
- `RECUR(A,B,C,h,x)`: h(t+1)=A·h(t)+B·x(t), y(t)=C·h(t) — familia SSM (Mamba/Mamba-3),
  requiere radio espectral ρ(A) < 1 para converger.

La separación en dos capas es el punto que la V11 marca como el cierre de una asimetría que
existía en versiones previas (la arquitectura se declaraba de dos capas pero la semántica
operacional sólo cubría cuatro reglas). Si `ATTEND`/`RECUR` se reemplazan a futuro por otra
primitiva de implementación con las mismas pre/postcondiciones, la capa algebraica no
cambia — esa es la garantía que el diseño busca dar.

---

## 3. Teoremas y su estado real de prueba

Fuente: `POLYDIM_PAPER_V11.md` Sec. 2.5, 6.1, 10 (Threats to Validity). Esta es la sección
más importante para no sobre-representar el rigor del proyecto frente a los alumnos.

| # | Enunciado | Estado |
|---|---|---|
| T1 | Asociatividad de COMPOSE | Prueba estándar (composición de funciones/matrices). Sólida. |
| T2 | Linealidad de MIX | Prueba estándar (combinación lineal de mapas lineales). Sólida. |
| T3 | PROJECT es funtor | Probado **sin condiciones** para los tres contratos COMPILE, EXPORT (vía Lema de Conmutatividad de Subespacios) y RENDER (vía Prop. 6.1). Es el resultado que más cambió entre versiones: antes era condicional. |
| T4 | Unicidad de punto fijo de FIXPOINT | Consecuencia directa del teorema de punto fijo de Banach. Sólida si T es contracción. |
| T5 | Invariancia de GEO_ID | Probada **por construcción** (la transformación admisible se define para no tocar `g`). La caracterización topológica más rica vía Homotopy Type Theory se declara explícitamente **"active research direction, not finalized theorem"** — no tratarla como probada. |
| Prop. 6.1 | Isomorfismo algebraico Flutter | Probada como isomorfismo monoidal estricto. |

**El propio paper incluye una sección "Threats to Validity" (Sec. 10) inusualmente honesta
para este tipo de documento** — vale la pena que los alumnos la vean tal cual, no filtrada:
- El bootstrap Python (29/29 tests, align_score=0.9993, +64.5% de mejora semántica) valida
  el *bootstrap*, no el lenguaje formal — son cosas distintas (T-C1, explícito en el paper).
- Los subespacios nativos son hoy *string labels*, no regiones geométricas reales (T-C2,
  problema abierto Q1).
- **"Las pruebas no fueron revisadas por un par humano independiente"** — el propio texto
  recomienda revisión externa antes de enviar a Zenodo/arXiv (T-I4).
- "AI-native" es una afirmación de diseño, no un resultado empírico medido (T-R3).
- El protocolo ALIGN no tiene validación empírica contra modelos heterogéneos reales
  (T-I3, PEND_010).

**Recomendación directa para uso docente:** si este material se presenta a los 30 alumnos,
conviene mantener esta distinción explícita (probado por construcción / probado
matemáticamente / conjeturado / pendiente de validar) en vez de aplanarla en un tono de
"así es como funciona POLYDIM" sin matices — el propio autor ya se cuida de no hacerlo en el
paper.

---

## 4. Sistema de tipos dimensional

`type(P) en ejecutor E := PROJECT(P, E)`. El tipo no se declara, emerge de qué subespacio se
observa; el mismo objeto puede tener varios tipos simultáneos (fila SQL + widget Flutter +
vector R^3 + nodo de grafo, todos a la vez, sincronizados por construcción vía pullback
categórico — no por ORM ni binding manual).

**Catálogo de subespacios `DIM_*` — ADVERTENCIA: la lista no es idéntica entre documentos:**
- `CONSTITUTION_EN.md` (más reciente) Art. 18: `SQL, FLUTTER, RUST, WASM, VECTOR, GRAPH,
  META, LOGIC, QUANTUM` (9), con prohibición explícita de crear `DIM_MEMORY`.
- `polydim-core.md` (skill, V8_0_EN, 2026-06-12): `PYTHON, RUST, FLUTTER, SQL, GRAPH, VECTOR,
  TIME, ERROR, META` (9, distintos).
- Ambas fuentes coinciden en que `DIM_MEMORY` está prohibido ("el estado es posición, no
  almacenamiento") y en que `DIM_FLUTTER`/`DIM_SQL`/`DIM_RUST`/`DIM_VECTOR`/`DIM_GRAPH` son
  estables en todas las versiones revisadas.
- Usar la lista de `CONSTITUTION_EN.md` por ser la más reciente, pero flaggear la
  discrepancia si un alumno cita la otra.

---

## 5. Identidad e interoperabilidad (GEO_ID, ALIGN)

- `GEO_ID`: componente invariante `g` de una posición `P=(g,S)`. Ninguna transformación
  admisible lo modifica (Teorema 5). Es la identidad "topológica" del objeto, independiente
  de en qué subespacio se lo observe.
- Protocolo `ALIGN` para comunicación entre IAs con espacios latentes de distinta
  dimensión (d1≠d2): comparten un codebook de anclas, calculan una matriz de proyección M
  vía Procrustes ortogonal (SVD), y aplican `T_B = M·T_A·M⁺` (pseudo-inversa de
  Moore-Penrose). Costo O(d1·d2). **Sin validación empírica todavía** (ver T-I3 arriba).
- Comunicación IA↔IA = computar una capa de cross-attention entre el emisor (expone K,V) y
  el receptor (genera Q) — no hay parsing léxico de por medio.
- Comunicación IA↔Humano = `PROJECT(T, DIM_FLUTTER)`. Flutter es el único ejecutor con
  isomorfismo algebraico probado (Prop. 6.1); React/Qt/SwiftUI se tratan como adaptadores
  externos, no nativos.

---

## 6. Ingeniería: formato `.polydim` y estado del bootstrap

- Formato binario de bajo rango (estilo LoRA, reutilizado como *tipo de dato de runtime*,
  no como mecanismo de fine-tuning): `T_geo = W₀ + U·Vᵀ`, con U,V ∈ R^(N×r), r≪N. Para
  N=10.000, r=64: ~2.5–5 MB por transformación vs. ~400 MB en formato denso (~99%+ de
  reducción). Cifras exactas varían levemente entre documentos (78.000× vs 99.4% — son la
  misma reducción expresada distinto, no una contradicción real).
- Bootstrap en Python (`polydim_runtime_v03.py`): 29/29 tests, `align_score = 0.9993`,
  +64.5% de mejora semántica sobre baseline determinístico. **Repetimos la advertencia T-C1:
  esto valida el bootstrap, no la especificación formal completa.**
- VM en Rust (`polydim_core.rs` V0.1, WASM) completada 2026-06-28; hay una propuesta técnica
  reciente (`propuesta_resolucion_deuda.md`, ya en esta carpeta) para optimizar el binding
  VSA de O(N²) a O(N log N) vía FFT — ~750× de aceleración teórica a N=10.000.
- **Esta síntesis no incluye el código fuente en sí** (vive en una raíz de Drive distinta a
  las dos carpetas que se pidió leer — ver skills `polydim-core` / `skill-polydim-v1`, que
  además apuntan a raíces de Drive diferentes entre sí, otra inconsistencia a resolver).

---

## 7. Qué NO es POLYDIM (según el propio paper, Sec. 8.1)

No es un framework de tensores (PyTorch computa CON tensores; POLYDIM programa ES tensores).
No es una arquitectura de red neuronal nueva. No es un mecanismo de atención nuevo (expone
la atención existente como primitiva programable). No es "Python con dimensiones con
nombre" (las dimensiones son subespacios geométricos en R^N, no anotaciones).

---

## 8. Problemas abiertos (Sec. 8.4 del paper)

Q1 cómo emergen los subespacios nativos desde un modelo de embeddings real (candidatos:
VQ-VAE, SOM, clustering jerárquico) · Q2 consistencia distribuida (Geometric-CRDTs) · Q3
¿POLYDIM es Turing-completo vía FIXPOINT? (conjetura: sí, para T contractivas sobre R^N
compacto) · Q4 auto-hosting (¿puede un programa POLYDIM generar programas POLYDIM?) · Q5
costo real de PROJECT a escala (perfilado pendiente) · Q6 relación entre FIXPOINT(ATTEND) y
la mónada libre sobre el endofuntor de atención · Q7 verificación empírica de la constante
de Lipschitz de ATTEND (condición de convergencia no verificada todavía en el prototipo).

---

## 9. Mapa de versiones y estado de publicación

- Documento con fecha de modificación más reciente en la carpeta 1: `CONSTITUTION_EN.md`
  (2026-07-01, cuenta `ai.mpat.agt@gmail.com`).
- Paper más reciente en la carpeta 2: `POLYDIM_PAPER_V11.md` (2026-06-30), header dice
  `Status: FINAL V11 — listo para Zenodo salvo [Authors placeholder]`.
- `publication_plan.md` (carpeta 2, 2026-06-30) ya tiene autor, afiliación y ORCID
  completos: **Ariel H. Garcia Traba, Investigador Independiente, Buenos Aires, ORCID
  0009-0001-2787-6067** — con lo cual el placeholder de autores del paper parece ser el
  único bloqueante formal para el envío.
- El propio paper recomienda revisión de un par experto en semántica de lenguajes de
  programación antes del envío final (T-I4). No hay evidencia en las carpetas leídas de que
  esa revisión ya haya ocurrido.

---

## 10. Fuentes citadas en este documento (para verificación puntual)

`CONSTITUTION_EN.md` (id 1s7AA7rSHCz0OHYicG1vh6xSha-gdgh8n) · `POLYDIM_PAPER_V11.md`
(id 132_KvnDFJi-F0W4Mn7Kr1-T0-lf4FPM7) · `publication_plan.md`
(id 17RAzez26P0lipvCmVhdwSA_jjKxpmavV) · `PROPOSITION_6_1_PROOF_V1.md`
(id 18WPN6-vmH7n1Nkh5wUkcfcBePL-alGhR, no leído en detalle en esta pasada — citado por el
paper como prueba completa de Prop. 6.1) · skills locales `polydim-core.md` /
`skill-polydim-v1` (para el catálogo alternativo de DIM_* y las raíces de Drive del
runtime).

**Nota sobre citas de terceros (Interlat, LatentMAS, Mamba-3, etc.):** son recientes
(2025–2026) y no se verificaron contra la web en esta pasada — antes de presentarlas a
alumnos como hechos externos confirmados, conviene un chequeo rápido de que existen y dicen
lo que el paper afirma que dicen.
