# POLYDIM — RESUMEN PARA CONSUMO DE AGENTES DE IA (no leer con ojos humanos, optimizado para tokens)

> Generado: 2026-07-02. Fuente: síntesis de ~40 documentos en dos carpetas Drive de POLYDIM
> (Constitución + Paper). Autor del proyecto: Ariel H. Garcia Traba (Independent Researcher,
> Buenos Aires, ORCID 0009-0001-2787-6067, github.com/AGT1973). Documento raíz que prevalece
> en caso de contradicción: `CONSTITUCION_V2.md` / `POLYDIM_CONSTITUCION_V2_CLEAN.md`.
> **Los agentes IA (alumnos) deben leer SOLO este archivo para entender POLYDIM, no los
> originales**, salvo que necesiten una prueba matemática exacta o el código fuente.

---

## 0. QUÉ ES POLYDIM EN UNA FRASE

POLYDIM es un lenguaje de programación formal cuya unidad mínima de cómputo no es una
instrucción de máquina de Von Neumann sino una **transformación geométrica T: R^N → R^N**
(N ≥ 10.000), diseñado para que las IAs (transformers, SSMs tipo Mamba) programen y se
comuniquen directamente en su espacio latente, sin serializar a texto.

Tesis central: los LLM "piensan" en miles de dimensiones simultáneas pero se ven forzados a
comunicarse por texto secuencial (tokens). Esto se llama **impedance mismatch / impedance gap**.
Dato duro citado (Interlat, Du et al. 2025, AAAI 2026): un hidden state transporta ~40.000 bits
de información vs ~15 bits por token de texto → **brecha de ancho de banda semántico de 2.667x**;
se descarta >99,9% del contenido semántico en cada serialización; evitarla logra hasta **24x**
de aceleración de inferencia (LatentMAS, ICML 2026 Spotlight, alineación latente sin reentrenar).

Analogía oficial del proyecto: "el ensamblador no inventó el transistor, lo hizo programable.
POLYDIM no inventa el transformer, lo hace programable."

---

## 1. LAS CUATRO PRIMITIVAS ALGEBRAICAS (núcleo invariante, independiente de arquitectura)

| Primitiva | Definición | Reemplaza a |
|---|---|---|
| `COMPOSE(T1,T2)` | = T2 ∘ T1 (no conmutativa) | secuencia de instrucciones |
| `MIX(α,T1,β,T2)` | = α·T1 + β·T2 (superposición continua, ortogonalidad en alta dimensión) | if/else binario |
| `FIXPOINT(T,ε)` | itera T hasta convergencia (Banach) | loops for/while |
| `PROJECT(T,executor)` | funtor covariante P: G → E_executor; es el "compilador" | tipos declarados / compilación |

`PROJECT` proyecta a ejecutores concretos: `DIM_FLUTTER` (UI nativa), `DIM_RUST`, `DIM_WASM`,
`DIM_SQL`, `DIM_PYTHON`, `DIM_GRAPH`, `DIM_VECTOR`, `DIM_TIME`, `DIM_ERROR`, `DIM_META`.
El **tipo emerge de la proyección**, no se declara (el mismo objeto puede ser 4 tipos a la vez
según qué subespacio se observe).

### Capa de implementación (específica de arquitectura, NO parte del núcleo algebraico)
- `ATTEND(Q,K,V,s) = softmax(Q·Kᵀ/√d)·V` — para Transformers.
- `RECUR` — para SSMs tipo Mamba-3 (ICLR 2026); requiere radio espectral ρ(A) < 1 para garantizar
  convergencia (heredado de la parametrización A = exp(−Δ·Λ) de la familia Mamba/S4).
Estas dos son "primitivas opacas": la especificación no exige saber cómo se implementan
internamente, solo sus contratos de entrada/salida. Esto hace que POLYDIM sea robusto a
transiciones arquitectónicas (transformer → SSM → lo que venga después).

### Comunicación AI↔AI (sin texto)
```
IA_A: estado_A → genera T → transmite T
IA_B: recibe T → aplica T a estado_B → nuevo estado_B
Communication(sB, sA) = softmax(QB·KA^T/√d)·VA
```
Leer un mensaje = computar una capa de atención. Protocolo `ALIGN` (basado en sondas
compartidas / Procrustes) traduce y alinea espacios vectoriales heterogéneos entre IAs distintas
sin perder semántica.

### Comunicación AI↔Humano
`PROJECT(estado, DIM_FLUTTER)` → widget nativo. Flutter es el "executor" nativo del subespacio
humano (isomorfismo algebraico directo, Proposition 6.1, probado). React/SwiftUI/Qt son
executors externos que requieren capa de adaptación.

---

## 2. LO QUE NO EXISTE EN POLYDIM (y qué lo reemplaza)

| No existe | Existe en su lugar |
|---|---|
| Variables con nombre | Posiciones en R^N con identidad geométrica (GEO_ID) |
| Funciones f(x) | Transformaciones T: R^N→R^N que se componen |
| Loop for/while | Convergencia a punto fijo T^n(v) |
| Condicional if/else | Gate continuo α·T1+(1−α)·T2 |
| Tipo declarado | Tipo emergente por proyección |
| Instrucción secuencial | Composición algebraica T2∘T1 |
| Mensaje de texto entre IAs | Transmisión directa de T |

`GEO_ID`: hipervector base único e invariante = identidad topológica del objeto, nunca cambia.
`Activaciones`: pesos continuos en [0.0, 1.0] por subespacio (mutan; no es reasignación, es
evolución del estado geométrico). `Binding`: vector de relación direccional en R^N.

### Formato de archivo `.polydim`
Binario tensorial (NO texto, NO bytecode secuencial):
```
[HEADER] magic 0x504F4C59 | version | N | n_transforms | precision_flags
[GEO_ID] float32[N] identidad invariante
[TRANSFORMS] Wi ∈ R^{N×r} low-rank (LoRA-style: T = W0 + U·Vᵀ, r << N)
[ACTIVATIONS] float32[K] pesos por subespacio
[PROJECTIONS] enum[] executors objetivo
```
Ejecución: el runtime aplica el bloque tensorial al vector de estado local. **No hay program
counter.** Ganancia LoRA: N=10.000, r=256 → ~20MB por transformación (vs ~400MB denso);
con r=64 la reducción medida es de 99,4% (400MB → ~2,5MB).

---

## 3. FUNDAMENTOS FORMALES — TEOREMAS Y AXIOMAS

### Teoremas algebraicos (paper técnico, versión más reciente V11/SOTA_UNIFIED)
- **T1**: COMPOSE es asociativo — T3∘(T2∘T1) = (T3∘T2)∘T1.
- **T2**: MIX preserva linealidad.
- **T3**: PROJECT es un funtor F: G → E, probado **incondicionalmente** para tres contratos:
  COMPILE y EXPORT (vía el Lema de Conmutatividad de Subespacios) y RENDER (vía el Isomorfismo
  Algebraico de Flutter, Proposition 6.1, probado).
- **T4**: FIXPOINT converge a un punto fijo único si T es una contracción (Teorema de Banach).
- **Teorema de Adecuación** (§2.6, nuevo en V9+): equivalencia entre semántica operacional y
  denotacional para los 6 constructores primitivos del núcleo, probado por inducción
  estructural (cf. Winskel 1993); se extiende a cualquier expresión finita vía COMPOSE.
- Reglas big-step T-C4 (ATTEND, requiere constante de Lipschitz L<1) y T-C5 (RECUR, ρ(A)<1)
  formalizan las primitivas de implementación como reglas operacionales opacas (añadidas en V11).
- Semántica denotacional: ⟦·⟧: POLYDIM → Para(Vect) (categoría de morfismos paramétricos);
  ATTEND es un endofunctor paramétrico sobre Vect; apilar n capas = mónada libre sobre ese
  endofunctor (cita: Gavranovic 2024, arXiv:2501.02931).

### Marco teórico ampliado: "Programación Cognitiva" (V2.0, evolución posterior a Fase XXXII)
A partir de esta fase el objeto de estudio deja de ser solo "construir un lenguaje" y pasa a
fundar la disciplina de **Programación Cognitiva**. POLYDIM queda re-encuadrado como el
**primer caso de estudio matemático** de esa disciplina.

- **Tupla del programa cognitivo**: P = (S, G, T, O, C, Π)
  - S = Estado Cognitivo; G = Meta/Goal; T = Transformación (S→S);
    O = Observador (S→M, espacio de significado); C = Restricción (C(S)=true);
    Π = Política (regla para elegir la siguiente transformación, modificable en ejecución).
- **Siete axiomas**: (I) todo programa tiene estado representacional; (II) todo estado es
  observable; (III) toda transformación produce otro estado válido; (IV) las transformaciones
  son componibles; (V) la ejecución persigue metas explícitas; (VI) toda ejecución es evaluable
  contra restricciones; (VII) la política Π puede modificarse durante la ejecución (deliberación).
- **Teorema de Preservación Cognitiva** (mayor desafío abierto de la disciplina): bajo qué
  condiciones un proceso de razonamiento preserva la meta G sin violar C — equivalente
  matemáticamente a erradicar "alucinación" y "goal drift". **No está resuelto**; es el
  problema abierto central.
- **Ley de Aislamiento (Sandboxing)**: los agentes experimentales (alumnos/IA) NO deben
  modificar el corpus teórico raíz directamente; deben operar en entornos aislados `/Alumnos/`.
- Reinterpretación clave: "No programamos secuencias algorítmicas deterministas, diseñamos
  procesos de razonamiento bajo restricciones y objetivos verificables."
- Documento de Constitución V10 original (Arts. I–XI) sigue vigente en sus reglas técnicas y
  algebraicas, pero reinterpretado bajo este nuevo teorema.

---

## 4. ESTADO EMPÍRICO / DE IMPLEMENTACIÓN (a la fecha de los documentos, ~1 jul 2026)

- Bootstrap en Python (`polydim_runtime_v03.py`): **29/29 tests** pasando; `align_score = 0.9993`;
  **+64,5% de mejora semántica** sobre baseline determinístico. Es andamio (scaffolding), NO es
  el lenguaje final — es "Python simulando POLYDIM", no sintaxis nativa.
- VM en Rust (`polydim_core.rs`, `polydim_fft_bind.rs`): compilando y testeada al 100%, con
  binding optimizado vía FFT (O(N log N)), sin dependencias externas; parser `.polydim` V5.
  Corre eficientemente incluso en hardware modesto (se menciona un Athlon 6000+ como referencia
  de baja gama) — argumento de autonomía de cómputo local/edge.
- Ejecutor SQL (`polydim_executor_sql.py`) verificó ejecutablemente las leyes de funtor de
  PROJECT (`verify_functor_law_a/b()`), 29/29 casos, elevando el Teorema 3 de "proof sketch" a
  parcialmente machine-verified.
- Ruta de evolución declarada: Hoy = Python bootstrap (API simbólica) → Próximo = VM Rust/WASM
  nativa → Futuro = POLYDIM self-host (el lenguaje se escribe a sí mismo).
- Existen múltiples specs formales versionadas (spec binario, semántica operacional, semántica
  denotacional) y un paper académico con ~11 revisiones (V5→V9→V10→V11→SOTA_UNIFIED_V1),
  evolucionando iterativamente vía tareas etiquetadas TASK_0XX ejecutadas por distintos agentes
  IA (Claude, Gemini) bajo distintas cuentas de curso (curso02/03/05.mithril, ai.mpat.agt).

## 4bis. AMENAZAS A LA VALIDEZ (honestidad epistémica declarada por el propio proyecto)
- **Interna**: las pruebas de T1–T5 y el Teorema de Adecuación fueron escritas por un agente de
  IA y NO han sido revisadas por un par humano independiente (aunque siguen métodos estándar:
  inducción estructural de Plotkin, Banach, propiedades de funtores).
- **Externa**: el bootstrap es Python, no el lenguaje de producción; los benchmarks de tamaño y
  rendimiento están acotados a ese prototipo. ATTEND/RECUR asumen pesos fijos en inferencia; no
  hay garantías para tiempo de entrenamiento (fine-tuning/RLHF).
- **Construct**: la condición ρ(A)<1 para RECUR no es universal (solo garantizada por
  construcción en la familia Mamba/S4); para otros A debe verificarse numéricamente.
- La identidad geométrica GEO_ID se apoya en topología con una formalización HoTT **en curso**
  (dirección de investigación activa, no cerrada).

---

## 5. NOVEDADES / SITUACIÓN ACTUAL DE PUBLICACIÓN (lo más reciente, relevante para "qué hacer ahora")

- **POPL 2027** (revisión doble ciega): deadline de envío **9 de julio de 2026, 23:59 AoE**.
  El scope encaja directamente ("contributions to theory/design/understanding/implementation/
  application of programming languages"). Acción: subir PDF sin nombre/institución a
  popl27.hotcrp.com, marcar topics cs.PL + formal semantics. Afiliación "Independent Researcher"
  es aceptada por el sistema.
- **Zenodo**: sin restricciones de afiliación, DOI inmediato y permanente (operado por CERN,
  financiado por la UE), indexado en Google Scholar. Recomendado subir cuanto antes.
- **arXiv**: desde enero 2026 exige email institucional + coautoría previa en arXiv, o un
  endorsement de un autor establecido en cs.PL — ruta difícil para investigador independiente;
  estrategia sugerida: publicar primero en Zenodo y conseguir un endorser.
- **Empresas objetivo (Tier 1)**: Anthropic (research@anthropic.com, por su trabajo en Claude
  Code / arquitectura multiagente), Google DeepMind (MCP, A2A), Meta AI/FAIR (VSA, fundamento
  matemático de MIX). **Tier 2**: Modular (Chris Lattner — analogía con LLVM/MLIR), Hugging
  Face, Cohere/Mistral.
- **Instituciones académicas receptivas**: en Argentina, DC-UBA y FAMAF (UNC); posible vía
  CONICET si hay tracción; internacionalmente CMU LTI y MIT CSAIL como contactos naturales tras
  tracción en POPL/arXiv.
- Kit de publicación ya redactado para LinkedIn (post corto + artículo largo), hilo de 8 tweets
  para X/Twitter, README de GitHub, y checklist de campos para Zenodo — listos para copiar y
  pegar cuando haya DOI.
- Existen **dos líneas de versión del paper en paralelo**: la línea "V5→V9→V10→V11" (framing
  técnico puro, "impedance gap" + Interlat/LatentMAS) y una línea más nueva "PAPER_V2 /
  POLYDIM_ES" que añade el framing de "Cognitive Programming" como marco teórico paraguas antes
  de presentar POLYDIM como primer caso de estudio. Ambas coexisten en el Drive; la V2.0 de la
  Constitución declara que, en caso de contradicción, ella prevalece.
- Hay una versión completa en español del paper (`POLYDIM_ES.md`) con el mismo contenido
  técnico que la versión en inglés.

---

## 6. LIMITACIONES DE ESTE RESUMEN (respuesta honesta a "¿esto es todo lo que hay que saber?")

**No, este resumen no sustituye el 100% del corpus, pero cubre lo esencial para que una IA
entienda qué es POLYDIM, cómo razona, y qué está pasando ahora mismo con el proyecto.**
Concretamente, este resumen NO incluye (por ser código/detalle de implementación, no
conocimiento conceptual):
- El código fuente completo de la VM Rust (`polydim_core.rs`, `polydim_fft_bind.rs`) ni del
  bootstrap Python.
- Las ~28 especificaciones formales detalladas archivo por archivo (binaria, operacional,
  denotacional) con sus pruebas matemáticas completas línea por línea.
- El archivo `polydim_paper_arxiv_final.tex` (fuente LaTeX exacta para arXiv) y `supplementary.zip`.
- Contenido histórico de versiones intermedias descartadas (V3, V4, V6, V7, V8 del paper) —
  solo se refleja su resultado acumulado en la versión más reciente.
- Documentos de una carpeta paralela `/Alumnos/` (sandboxing) mencionada en la Constitución V2,
  que no fue provista para esta síntesis.

Si un alumno (IA) necesita **implementar** algo (escribir código, verificar una prueba exacta,
citar un teorema con precisión formal para un paper), debe pedir el documento fuente específico
citado aquí (nombre de archivo) en vez de confiar solo en este resumen. Para **entender el
paradigma, discutirlo, razonar sobre él o generar contenido de alto nivel**, este resumen es
suficiente y evita gastar tokens leyendo ~40 documentos con alta redundancia (mismo contenido
duplicado en .md/.html/.docx/.pdf).

---

## 7. GLOSARIO RÁPIDO
- **Impedance mismatch/gap**: pérdida de información al forzar comunicación IA↔IA o IA↔humano
  a pasar por texto plano en vez de geometría nativa.
- **Gesto vs. Emoji**: metáfora del proyecto — el "gesto" (1000+ dimensiones) es lo que la IA
  piensa realmente; el "emoji" (texto, 2D) es la versión degradada que usamos hoy.
- **DIM_X**: subespacio nativo de observación (SQL, FLUTTER, RUST, GRAPH, VECTOR, TIME, ERROR, META).
- **Cognitive Programming**: disciplina científica paraguas propuesta en V2.0; POLYDIM es su
  primer caso de estudio formal.
- **Sandboxing / Ley de Aislamiento**: regla constitucional que prohíbe a agentes experimentales
  modificar el corpus teórico raíz; deben trabajar en `/Alumnos/`.
