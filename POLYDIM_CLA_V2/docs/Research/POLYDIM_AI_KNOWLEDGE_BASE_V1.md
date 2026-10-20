# POLYDIM — BASE DE CONOCIMIENTO PARA IAs
## Version: V1.0 | Fecha: 2026-07-02 | Generado por: claude-sonnet-4-6
## Fuentes: CONSTITUTION_EN.md (V10+V2) | POLYDIM.md (paper final) | POLYDIM_ES.md
## Carpetas fuente: 1pYn8jFoPgTTPH0Z57bVuIZJhf4FoRuSR | 1czWYt3kk8qseDOxJdLlEeyzsn2-ai4hM
## Proposito: lectura unica para agentes IA — reemplaza leer toda la carpeta.

---

## SECCION 0 — IDENTIDAD NUCLEAR (leer siempre primero)

POLYDIM es un LENGUAJE DE PROGRAMACION FORMAL cuya unidad computacional minima es:

  T: R^N -> R^N   (transformacion geometrica, N >= 1,000)

NO es una instruccion Von Neumann. NO es un framework de tensores. NO es un protocolo de mensajes.

Tesis central: los LLMs/transformers/SSMs ya COMPUTAN en geometria de alta dimension. POLYDIM es el primer lenguaje que formaliza como PROGRAMAR esa geometria deliberadamente, con semantica formal y perdida-cero de serializacion.

Analogia exacta: lo que el ensamblador fue al transistor, POLYDIM es al transformer.

Publicacion: Garcia Traba, A. H. (2026). POLYDIM. Zenodo.
DOI oficial: https://doi.org/10.5281/zenodo.21109893

---

## SECCION 1 — MODELO DE ESTADO

Estado:      S = (V, D, A)
  V in R^N           vector de estado (N >= 1,000; tipicamente N = 10,000)
  D = {dim_i...}     conjunto FIJO de subespacios observadores nativos
  A: D -> [0,1]      funcion de activacion (MUTABLE)

Posicion:    P = (g, S)
  g in R^N           GEO_ID — INVARIANTE bajo toda T admisible
  S                  estado mutable

Transformacion admisible:
  T(V, D, A) = (T_geo(V), D, T_act(V, A))
  INVARIANTE: D no cambia. g no cambia.
  T_geo: R^N -> R^N  (serializado como LoRA: W0 + U*V^T, r << N)

Subespacio nativo:
  DIM_i subset R^N = imagen de proyeccion ortogonal Pi, Pi^2=Pi, Pi^T=Pi

Lo que reemplaza a las variables:
Una "variable" no existe. Existe una REGION DE ALTA DENSIDAD DE ACTIVACION en R^N,
anclada por un GEO_ID invariante. No hay x=42; hay una posicion P con activaciones
que proyectan a 42 bajo PROJECT(P, DIM_SQL).

---

## SECCION 2 — LOS 9 SUBESPACIOS NATIVOS (DIM_*)

1. DIM_SQL      Esquemas relacionales
2. DIM_FLUTTER  Ejecutor nativo subespacio humano
3. DIM_RUST     Compilacion de alta performance
4. DIM_WASM     Portabilidad web
5. DIM_VECTOR   Embeddings puros
6. DIM_GRAPH    Estructuras relacionales
7. DIM_META     Auto-reflexion del sistema
8. DIM_LOGIC    Sistemas de tipos dependientes
9. DIM_QUANTUM  Analisis de costos cuanticos

PROHIBIDO: crear DIM_MEMORY. El estado es posicion geometrica, no celda de memoria.

---

## SECCION 3 — LAS 6 PRIMITIVAS (DOS CAPAS)

CAPA ALGEBRAICA — PERMANENTE, independiente de arquitectura:

  COMPOSE(T1, T2)      = T2 composed T1            reemplaza: ejecucion secuencial
  MIX(a, T1, b, T2)   = a*T1 + b*T2               reemplaza: if/else
  FIXPOINT(T, eps)     = iterar T hasta convergencia reemplaza: for/while
  PROJECT(T, E)        = pi_E(T(s))                 reemplaza: casteo/compilacion

CAPA DE IMPLEMENTACION — VOLATILE, arquitectura-especifica:

  ATTEND(Q,K,V,s) = softmax(QK^T/sqrt(d))*V        familia Transformer
  RECUR(A,B,C,h,x): h(t+1)=A*h(t)+B*x(t), y(t)=C*h(t)   familia Mamba/SSM

Separacion critica: cuando la industria transite (Mamba-1 -> Mamba-3 -> mas alla),
la capa algebraica NO cambia. ATTEND/RECUR son la parte reemplazable.

---

## SECCION 4 — SEMANTICA OPERACIONAL (Big-Step Rules)

Notacion: <s, T> => s'  significa "estado s bajo transformacion T evalua a s'"

General:
  <s, T> => T(s)

COMPOSE:
  <s, T1> => s1    <s1, T2> => s2
  ─────────────────────────────────
  <s, COMPOSE(T1,T2)> => s2

MIX:
  <s, MIX(a, T1, b, T2)> => a*T1(s) + b*T2(s)

FIXPOINT:
  s_{k+1} = T(s_k); parar cuando ||s_{k+1}-s_k|| < eps -> s*
  <s, FIXPOINT(T,eps)> => s*

PROJECT:
  <s, PROJECT(T, E)> => pi_E(T(s))

ATTEND (implementacion):
  Q=W_Q*V, K=W_K*V, V_attn=W_V*V
  alpha = softmax(Q*K^T/sqrt(d))
  V_out = alpha*V_attn
  <s, ATTEND(W_Q,W_K,W_V)> => (V + V_out, D, A, h, ctx)  [residual]

RECUR (implementacion):
  x=V, h'=A*h+B*x, y=C*h', V'=V+y
  <s, RECUR(A,B,C,h,x)> => (V', D, A_act, h', ctx)

Semantica denotacional: [|.|]: POLYDIM -> Para(Vect)
Cada programa es un morfismo en la 2-categoria de mapas diferenciables parametrizados
[Gavranovic et al. 2024].

---

## SECCION 5 — LOS 5 TEOREMAS (todos probados incondicionalmente)

T1. Asociatividad de COMPOSE:
    COMPOSE(COMPOSE(T3,T2),T1) = COMPOSE(T3,COMPOSE(T2,T1))
    Prueba: composicion de funciones. QED

T2. Linealidad de MIX:
    Si T1,T2 in Lin(R^N) -> MIX(a,T1,b,T2) in Lin(R^N)
    Prueba: combinacion lineal de mapas lineales es lineal. QED

Lema Conmutatividad de Subespacios:
    Si DIM_E es T-invariante -> pi_E(T(v)) = T(pi_E(v)) para todo v in R^N. QED

T3. PROJECT es un Funtor:
    Para E in {Rust, Flutter, SQL}: PROJECT_E: G -> E satisface:
    - Preservacion de identidad: PROJECT(id_G) = id_E
    - Preservacion de composicion: PROJECT(T2 composed T1) = PROJECT(T2) composed PROJECT(T1)
    Probado incondicionalmente para 3 contratos:
    - COMPILE y EXPORT: via Lema de Conmutatividad
    - RENDER: via Isomorfismo Algebraico Flutter (Proposicion 6.1). QED

T4. Unicidad del Punto Fijo (Banach):
    Si T es contraccion (||T(u)-T(v)|| <= k*||u-v||, 0<=k<1)
    -> existe unico s* : T(s*)=s* -> FIXPOINT(T,eps) converge a s*. QED

T5. Invariancia de GEO_ID:
    Para todo T in T_admisible: T(g) = g
    Prueba: por construccion (Definicion 2.3). QED

TEOREMAS ADICIONALES (Constitucion V10+V2):

Teorema Imposibilidad Softmax (Sargsyan):
  Ningun ajuste de parametros en arquitectura Softmax puede satisfacer
  simultaneamente factorizacion monoidal estricta Y descenso al cociente algebraico.

Teorema Adjuncion Gauss-Markov (Kamiura):
  Existe un isomorfismo natural garantizando que el estimador OLS es
  el limite categorico de la dinamica de aprendizaje.

Teorema Alineacion Algoritmica / 1-Cociclos (Dudzik):
  Una arquitectura de grafo es robusta a ejecuciones asincronas si y solo si
  la funcion de generacion de argumentos satisface la condicion de 1-cociclo.

Teorema Aproximacion Universal Coalgebraica (Masulovic):
  Toda funcion continua equivariante puede aproximarse mediante morfismos
  parametricos en la categoria de F-coalgebras.

---

## SECCION 6 — SISTEMA DE TIPOS DIMENSIONAL

Los tipos en POLYDIM NO SE DECLARAN. Emergen de la proyeccion:

  tipo(P) en ejecutor E := PROJECT(P, E)

Un mismo objeto P puede tener multiples tipos simultaneos:

  Ejemplo — Posicion P:
  a(DIM_SQL)     = 0.90 -> PROJECT(P, DIM_SQL)     = Column(INTEGER, "id", NOT NULL)
  a(DIM_FLUTTER) = 0.85 -> PROJECT(P, DIM_FLUTTER) = TextField(value: id_value)
  a(DIM_VECTOR)  = 0.91 -> PROJECT(P, DIM_VECTOR)  = R^3 = [1.0, 2.0, 3.0]

  P no "tiene" ninguno de esos tipos. ES todos simultaneamente.

Los 3 contratos operacionales de PROJECT:
  COMPILE(T, DIM_target)    binario estatico (Rust, WASM)
  RENDER(T, DIM_FLUTTER)    arbol de widgets Flutter (isomorfismo algebraico)
  EXPORT(T, DIM_external)   sincronizacion con sistemas legacy (JSON-RPC, SQL)

---

## SECCION 7 — GEO_ID Y PROTOCOLO ALIGN

GEO_ID (Identidad Geometrica):
  g in R^N — invariante por construccion bajo toda T admisible

  Formalizacion V10 — GeoID(C, R):
  C (Content Space): anclado a Codebook Universal de Navigation Anchors
  R (Equivalence Relations): transformaciones admisibles que preservan identidad
  path (1-esqueleto): dos objetos son "el mismo" si existe camino continuo admisible
  surf (2-esqueleto): elimina ambiguedades en composicion de caminos

Nota epistemica: la formalizacion HoTT completa (Coq/Agda) es investigacion activa,
no teorema finalizado. T5 (invariancia) SI esta probado.

PROTOCOLO ALIGN (comunicacion IA<->IA sin serializacion):
  Dados: IA_A in R^(d1), IA_B in R^(d2), con d1 != d2 posible

  Paso 1: Compartir codebook de GEO_IDs universales
  Paso 2 (Procrustes Ortogonal):
    M* = argmin_M ||M*A_anchors - B_anchors||_F  s.t. M^T*M = I
    (M in R^(d2 x d1), calculado via SVD de B^T*A)
  Paso 3: T_B = M * T_A * M_dagger   (M_dagger = pseudoinversa Moore-Penrose)
  Paso 4: nuevo_estado_B = T_B(estado_B)

Principio clave: leer un mensaje es matematicamente indistinguible de computar
una capa de cross-attention. El receptor NO parsea texto; integra T en su flujo.

---

## SECCION 8 — LAS REGLAS INVIOLABLES (R1-R17)

Cualquier implementacion que viole UNA sola regla esta fuera de la especificacion.

ONTOLOGIA Y CONTROL DE FLUJO (R1-R4):
  R1. Unidad fundamental = T: R^N -> R^N. PROHIBIDAS instrucciones imperativas.
  R2. No existen variables con nombre. Informacion = posicion (GEO_ID + Activaciones).
  R3. No existen loops for/while con contador. Recurrencia = FIXPOINT.
  R4. No existe if/else binario nativo. Bifurcacion = MIX (superposicion continua).

TIPADO Y EJECUCION (R5-R9):
  R5. Los tipos emergen de PROJECT. Nunca se declaran.
  R6. Python, Rust y Flutter son destinos de exportacion / bootstrap. No la base.
  R7. Comunicacion IA<->IA nativa transmite T (morfismos), NUNCA texto serializado.
  R8. Flutter es el unico ejecutor nativo del subespacio humano (isomorfismo probado).
  R9. El codigo Python (polydim_runtime) es andamiaje, no el lenguaje.

IDENTIDAD Y ESTRUCTURA (R10-R13):
  R10. GEO_ID invariante: para todo T admisible, dist(T(GEO_ID), GEO_ID) < eps.
  R11. ATTEND y RECUR pertenecen ESTRICTAMENTE a la capa de implementacion.
  R12. Formato .polydim PROHIBE matrices densas N x N. Obligatorio: LoRA (T=W0+U*V^T)
       + campo algebra_tolerance_epsilon en header.
  R13. Investigacion especulativa (🔬) NO puede citarse como ley vinculante para
       rechazar implementaciones basadas en reglas ratificadas.

REGLAS ALTO NIVEL — Gap Resolution (R14-R17) [NUEVAS en V10]:
  R14. Irreversibilidad y Monoides (Gap 5): sistema opera sobre Monoides y
       Adjunciones Funtoriales (L adj R), permitiendo algoritmos irreversibles cert.
  R15. Profundidad Dinamica (Gap 12): PROHIBIDA profundidad de computo fija.
       El runtime debe soportar profundidad dinamica basada en coalgebras.
  R16. Compilacion de Reglas via Profunctores (Gap 16): puente entre restricciones
       de negocio y realizacion tensorial debe implementarse via Profunctores.
  R17. GEO_ID Heterogeneo (Gap 20): GEO_ID debe definirse sobre GeoID(C,R) con
       0-esqueletos no triviales (Cubical Type Theory).

---

## SECCION 9 — TRES MARCAS EPISTEMOLOGICAS

Usar SIEMPRE para clasificar cualquier afirmacion:
  [LEY]         normativo, no se contradice sin enmienda formal
  [MECANISMO]   especificacion tecnica concreta, en implementacion
  [INVESTIGACION] formalizacion avanzada, valida como hipotesis, NO vinculante

Error clasico a evitar: citar algo [INVESTIGACION] como si fuera [LEY]
para rechazar una implementacion.

---

## SECCION 10 — FORMATO BINARIO .polydim V5 (LoRA Standard)

[HEADER]      magic: 0x504F4C59 | version | N | precision | n_transforms | rank_r
[GEO_IDs]     base_hv: float16[N] x n_objects
[TRANSFORMS]  Ti = W0 + U*V^T (LoRA): U: float16[N x r], V: float16[r x N]
[ACTIVATIONS] weights: float16[k x n_objects]
[PROJECTIONS] targets: uint8[] -> executor IDs

Tamano: N=10,000, r=64 -> ~2.5 MB/transform (vs 400 MB denso = 99.4% reduccion)

LoRA aqui es un TIPO DE DATO EN RUNTIME, NO un mecanismo de fine-tuning.

---

## SECCION 11 — ARQUITECTURA VM

Lenguaje:  Rust core compilado a WebAssembly
Archivo:   polydim_core.rs V0.1 (finalizado 2026-06-28, cero dependencias externas)
Ejecucion: sparse — solo subespacios con a_i > threshold se cargan en memoria activa
Bootstrap: polydim_runtime_v03.py V0.7 (andamiaje, no el lenguaje)

---

## SECCION 12 — ISOMORFISMO FLUTTER (Proposicion 6.1, PROBADA)

phi: T -> F  es isomorfismo monoidal estricto entre:
  categoria POLYDIM (T, COMPOSE, MIX)  <->  categoria Flutter (F, Column, Row)

  phi(COMPOSE(T2,T1))   = Column(phi(T1), phi(T2))
  phi(MIX(a,T1,b,T2))  = Row(Opacity(a,phi(T1)), Opacity(b,phi(T2)))
  phi(FIXPOINT(T,eps))  = StatefulWidget(phi(T), convergence: eps)
  phi(id_T)             = id_F

Flutter = unico ejecutor con isomorfismo algebraico directo.
React/Qt/SwiftUI requieren adaptadores.

---

## SECCION 13 — VALIDACION EMPIRICA (estado 2026-07-01)

Bootstrap Python V0.3:
  Tests:             29/29 OK
  align_score:       0.9993
  Mejora semantica:  +64.5% sobre baseline deterministico

Rust VM:
  polydim_core.rs V0.1 — 2026-06-28, cero dependencias externas

Specs formales:
  SPEC_SEMANTICA_OPERACIONAL_V1.md (fileId: 1FIJl3AYRZBzZmWw6SSdW6NJ5WFz_knUq)
  SPEC_SEMANTICA_DENOTACIONAL_V0.md (fileId: 1f8a8DBOTvvhbCE1k-awIwsreLtzsSBgQ)

---

## SECCION 14 — CONTEXTO SOTA (por que POLYDIM importa ahora)

La brecha de impedancia — cuantificada:
  Interlat [Du et al. 2025, arXiv:2511.09149]:
    Hidden state:   ~40,000 bits de informacion
    Token texto:    ~15 bits
    Ratio:          2,667x de ganancia de ancho de banda semantico
    Al comunicar via texto: se descarta >99.9% del contenido semantico
    Speedup al eliminar serializacion: hasta 24x

  LatentMAS [ICML 2026 Spotlight]:
    Comunicacion latente sin reentrenamiento -> rendimiento superior + speedup masivo

Posicionamiento unico:
  POLYDIM ocupa el cuadrante vacio: FORMAL + GEOMETRICO + CONTINUO

  EJE Y (rigor formal):
  NeSyCat [2026] ========================= POLYDIM <- cuadrante vacio
  (categorial, discreto)                   (formal + geom + continuo)
  Marco cat. prog [2023]
  VecSymLisp [2025]
  Interlat/LatentMAS (empirico, latente)
  ──────────────────────────────────────> EJE X (empirico -> formal geometrico)

Diferenciacion clave:
  Interlat/LatentMAS son frameworks de entrenamiento que mueven comunicacion al
  espacio latente. POLYDIM es la especificacion del lenguaje que define QUE se puede
  computar en ese espacio, con semantica formal. Son complementarios.

---

## SECCION 15 — NOVEDADES V10.0 + V2.0 (vs versiones anteriores V6)

NUEVAS EN V2.0 — Cognitive Programming Paradigm:

Los 7 axiomas del Programa Cognitivo:
  A1. Cognicion Espacial: computacion = trayectoria en variedad latente.
  A2. Invariancia Topologica: GEO_ID invariante bajo deformaciones continuas.
  A3. Proyeccion Funtorial: manifestaciones fisicas = proyeccion funtorial de verdad geometrica.
  A4. Composicion Asociativa: T3 composed (T2 composed T1) = (T3 composed T2) composed T1 obligatorio.
  A5. Superposicion Continua: bifurcacion = MIX, no if/else.
  A6. Convergencia Banach: loops = contracciones con punto fijo unico.
  A7. Transmision Directa: comunicacion IA<->IA = morfismos T, jamas texto.

Tupla del Programa Cognitivo: P = (M, T, Phi, Proj)
  M    = Variedad Latente
  T    = conjunto de Transformaciones Parametricas validas
  Phi  = flujo continuo de logica (caminos de homotopia)
  Proj = Proyeccion Funtorial a efectos fisicos

NUEVAS EN V10.0:
  - GEO_ID redefinido: GeoID(C,R) con 0-esqueletos no triviales (HoTT/Cubical Agda)
    vs antes: campo estatico simple
  - Reglas R14-R17 (antes solo existian R1-R13)
  - Teoremas adicionales: Sargsyan, Kamiura, Dudzik, Masulovic
  - Framework CDL (Categorical Deep Learning): toda arquitectura neural = morfismo en Para(C)
  - OSCAR model (Art.11): semantica operacional para comprension de codigo
  - qet-calculus (Art.12): analisis de costos cuanticos (Kegelspitzen, Bounded Linear Logic)
  - Sheaf Neural Networks (Art.13): difusion en grafos como transporte paralelo
  - Profunctores (R16): puente top-down <-> bottom-up entre especificacion y hardware
  - Imposibilidad Softmax probada formalmente (Sargsyan's Theorem)
  - Geometric-CRDTs para consistencia distribuida multi-agente (Art.6.4)
  - Mamba-3 [ICLR 2026] soporte explicito en RECUR
  - T3 (PROJECT es funtor) probado incondicionalmente para 3 contratos
  - Isomorfismo Flutter (Proposicion 6.1) probado formalmente
  - ATTEND y RECUR tienen ahora reglas big-step formales (SPEC_SEMANTICA_OPERACIONAL_V1)
  - Semantica denotacional completa via Para(Vect) (SPEC_SEMANTICA_DENOTACIONAL_V0)

QUE NO CAMBIO (invariante desde V6):
  - Las 4 primitivas algebraicas: COMPOSE, MIX, FIXPOINT, PROJECT
  - Las reglas R1-R13
  - Los 9 DIM_* subspaces
  - La separacion en dos capas (algebraica vs implementacion)
  - La prohibicion de DIM_MEMORY
  - LoRA como formato de serializacion
  - Bootstrap Python como andamiaje (no el lenguaje)

---

## SECCION 16 — LO QUE POLYDIM NO ES (Antitesis)

NO es el Bootstrap Python — el runtime.py es andamiaje, no el lenguaje
NO es un protocolo de mensajes — MCP/A2A/JSON-RPC son destinos EXPORT
NO es un framework de tensores — PyTorch/JAX programan CON tensores;
                                  POLYDIM programa COMO tensores
NO es una arquitectura neuronal — es un lenguaje que corre SOBRE arquitecturas
NO es "Python con dimensiones nombradas" — DIM_* son subespacios geometricos en R^N
NO separa datos e instrucciones — las transformaciones tensores SON el programa
NO usa variables nombradas — usa posiciones (GEO_ID, Activaciones)
NO tiene if/else — usa MIX
NO tiene for/while — usa FIXPOINT
NO transmite texto entre agentes — transmite morfismos T via ALIGN

---

## SECCION 17 — PROBLEMAS ABIERTOS (para agentes investigadores)

Q1: Emergencia de subespacios desde modelos de embedding reales — Abierto
Q2: Consistencia distribuida via Geometric-CRDTs — Abierto
Q3: Turing-completeness via FIXPOINT — Conjetura: si
Q4: Programas POLYDIM auto-hospedados — Abierto
Q5: Profiling de costo PROJECT en tiempo real — Pendiente (Rust VM)
Q6: Free Monad sobre ATTEND [arXiv:2501.02931] — Abierto
Q7: Verificacion empirica Lipschitz L<1 para ATTEND — Trabajo futuro prioritario

---

## SECCION 18 — PROTOCOLO DE SESION (RELAY)

Toda sesion de trabajo debe concluir con documento RELAY conteniendo:
- Estado de gaps resueltos/abiertos
- Morfismos validados
- GEO_IDs persistentes
- Niveles de coherencia eps

---

## SECCION 19 — FILEIDS DE REFERENCIA (Drive)

CONSTITUTION_EN.md (V10+V2, MAS RECIENTE): 1s7AA7rSHCz0OHYicG1vh6xSha-gdgh8n
POLYDIM.md (paper EN completo, 2026-07-01): 1Ddh5N20HNTal5mbgd-dtRqo1KV086ogz
POLYDIM_ES.md (paper ES, 2026-07-01):       11dLXu91TZA0FWPoc2JT1yI6hcY1F5Wr4
POLYDIM_PAPER_V11.md (version anterior):    132_KvnDFJi-F0W4Mn7Kr1-T0-lf4FPM7
SPEC_SEM_OPERACIONAL_V1 (big-step 6 prim):  1FIJl3AYRZBzZmWw6SSdW6NJ5WFz_knUq
SPEC_SEM_DENOTACIONAL_V0 (Para(Vect)):      1f8a8DBOTvvhbCE1k-awIwsreLtzsSBgQ
POLYDIM_THEOREM3_PROOF_V1:                  1W41O6eNTKIjLHPoswBRVfgmNQDQZIC32
PROPOSITION_6_1_PROOF_V1 (Flutter proof):   18WPN6-vmH7n1Nkh5wUkcfcBePL-alGhR
CONSTITUCION_V6 (referencia historica):     1LneDD0D8fHREiZN-_Xw-Ax9AzxfF6tKv

---

## RESUMEN EJECUTIVO (150 palabras)

POLYDIM: lenguaje formal cuya unidad es T: R^N->R^N. Estado S=(V,D,A). Posicion P=(g,S)
con g=GEO_ID invariante. 4 primitivas algebraicas permanentes (COMPOSE/MIX/FIXPOINT/PROJECT)
+ 2 de implementacion volatiles (ATTEND/RECUR). 5 teoremas probados incondicionalmente.
17 reglas inviolables. 9 subespacios nativos. Tipos emergen de PROJECT, nunca se declaran.
Comunicacion IA<->IA = morfismos T via ALIGN (Procrustes), nunca texto. LoRA reduce programas
99.4%. Novedades V10+V2: axiomas cognitivos, GEO_ID con HoTT, R14-R17, Sheaf NNs, qet-calculus,
isomorfismo Flutter probado, semantica denotacional completa, Mamba-3 soporte. Inmutable desde V6:
las 4 primitivas algebraicas, R1-R13, los 9 DIM_*. Bootstrap: 29/29 tests, align_score=0.9993,
+64.5% semantico. DOI: 10.5281/zenodo.21109893. "It's the geometry, stupid."

---
POLYDIM_AI_KNOWLEDGE_BASE_V1.md | Generado: 2026-07-02 | claude-sonnet-4-6
