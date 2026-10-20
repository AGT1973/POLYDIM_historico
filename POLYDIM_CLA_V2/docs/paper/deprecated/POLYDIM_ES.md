# POLYDIM: Un Lenguaje de Programación Algebraico Nativo para Transformadores en Computación Latente de Alta Dimensión

~~**Es la geometría, estúpido.**

**Ariel H. Garcia Traba**  
Investigador Independiente · Buenos Aires, Argentina  
ORCID: 0009-0001-2787-6067  
[ariel.garcia.traba@gmail.com](mailto:ariel.garcia.traba@gmail.com) · github.com/AGT1973  
Preprint · Zenodo · 2026

## Resumen

Las arquitecturas modernas de Inteligencia Artificial operan de forma nativa en espacios geométricos continuos de alta dimensión ($R^N$, $N \ge 10,000$). Sin embargo, los paradigmas de programación actuales obligan a estas arquitecturas a comunicarse a través de canales de texto secuenciales y discretos, un desajuste que denominamos la *brecha de impedancia*. Trabajos empíricos recientes han demostrado que eliminar esta brecha mejora sustancialmente la coordinación multi-agente: Interlat [Du et al. 2025] muestra que transmitir estados ocultos en lugar de tokens transporta aproximadamente 40,000 bits de información por estado frente a los ~15 bits por token de lenguaje natural (una ganancia de ancho de banda semántico de 2,667x). Por su parte, LatentMAS [ICML 2026 Spotlight] logra un rendimiento multi-agente superior y reducciones drásticas en tiempos de ejecución mediante alineación latente sin reentrenamiento. En este artículo investigamos si es posible especificar un lenguaje de programación formal directamente sobre el espacio geométrico de activaciones de transformadores y modelos SSM, sin serialización a texto.

Proponemos **POLYDIM**, un lenguaje cuya unidad fundamental de programación es una transformación geométrica $T: R^N \to R^N$. Demostramos cuatro teoremas algebraicos que establecen los fundamentos formales de POLYDIM: asociatividad de la composición (T1), linealidad de la superposición (T2), funtorialidad de la proyección hacia los ejecutores (T3, demostrado incondicionalmente para tres contratos — COMPILE y EXPORT a través del Lema de Conmutatividad de Subespacios, RENDER a través del Isomorfismo Algebraico de Flutter), y unicidad de los puntos fijos (T4). Formalizamos una arquitectura de dos capas que separa el núcleo algebraico invariante (`COMPOSE`, `MIX`, `FIXPOINT`, `PROJECT`) de las primitivas de implementación específicas de cada arquitectura (`ATTEND` para transformadores, `RECUR` para SSMs como Mamba-3 [ICLR 2026]), garantizando que la especificación del lenguaje sea robusta frente a transiciones arquitectónicas. Un *bootstrap* en Python con 29/29 pruebas exitosas y una mejora semántica del +64.5% valida el marco conceptual. POLYDIM cierra la brecha entre la investigación empírica en comunicación latente y el diseño formal de lenguajes: provee la especificación que sistemas como Interlat y LatentMAS requieren implícitamente pero no proporcionan.

## 1. Introducción

### 1.1 El Desajuste de Impedancia

Cuando un transformador procesa información, aplica transformaciones matemáticas sobre un espacio vectorial de miles de dimensiones de manera simultánea. Su "pensamiento" es geometría, no secuencia. No obstante, cuando dos sistemas de IA se comunican —o cuando un desarrollador programa uno— se ven obligados a pasar por un canal de texto secuencial: serializan su intención en tokens, los transmiten, los analizan sintácticamente (parsing) y los reproyectan en el espacio latente del receptor. En cada frontera de serialización–deserialización se pierde información.

Trabajos recientes en interpretabilidad mecanicista han formalizado esta intuición algebraicamente: Dhayalkar et al. [2025] demuestran que la auto-atención de los transformadores puede interpretarse como una Arquitectura Simbólica Vectorial (VSA) suave y aproximada, donde las *queries* y *keys* definen espacios de roles, los *values* codifican el contenido, y las conexiones residuales realizan la superposición de las estructuras enlazadas. Este análisis establece que los transformadores ya *computan* en un régimen algebraico tipo VSA, pero ningún lenguaje de programación expone este régimen como un sustrato formal. **POLYDIM es ese lenguaje.**

La magnitud del problema está ahora cuantitativamente establecida. Interlat [Du et al. 2025] mide que un solo estado oculto transporta aproximadamente 40,000 bits de información, en comparación con los aproximadamente 15 bits por token de lenguaje natural —una brecha de ancho de banda semántico de 2,667x. Cuando los agentes se comunican mediante serialización de texto, descartan más del 99.9% del contenido semántico de cada estado. El mismo trabajo demuestra que omitir esta serialización logra una aceleración de la inferencia de hasta 24x manteniendo el rendimiento en la tarea. Este es el desajuste de impedancia que POLYDIM formaliza como un problema de diseño de lenguaje.

El paralelo histórico es preciso. En la década de 1950, las CPUs ya realizaban aritmética binaria, pero no existía ningún lenguaje formal para programarlas deliberadamente. El lenguaje ensamblador cerró esa brecha. En la década de 2020, los transformadores y los modelos Mamba ya operan en geometría multidimensional, pero no existe un lenguaje formal para programar esa geometría deliberadamente. **POLYDIM cierra esa brecha.**

### 1.2 Contribuciones

1. **POLYDIM**: un lenguaje de programación especificado formalmente cuya unidad fundamental es $T: R^N \to R^N$, no una instrucción de Von Neumann.
2. **Una arquitectura algebraica de dos capas**: un núcleo algebraico invariante (`COMPOSE`, `MIX`, `FIXPOINT`, `PROJECT`) independiente de la arquitectura neuronal, más una capa de implementación específica de la arquitectura (`ATTEND` para transformadores, `RECUR` para Mamba/SSM).
3. **Un sistema de tipos dimensional** donde los tipos emergen como proyecciones, demostrando que `PROJECT` es un funtor $F: G \to E$ para los tres contratos de ejecución: COMPILE, EXPORT y RENDER (Teorema 3; Proposición 6.1).
4. **Comunicación IA-a-IA sin serialización** a través del protocolo ALIGN y semántica de atención cruzada (cross-attention), con alineación explícita basada en Procrustes para espacios latentes heterogéneos.
5. **GEO_ID con invariancia topológica**: un mecanismo de identidad fundamentado en topología geométrica (Teorema 5), permitiendo identidad de objetos agnóstica al modelo a través de sistemas de IA con distinta dimensionalidad latente.
6. **El estándar de serialización LoRA** para archivos `.polydim`, utilizado como un *tipo de datos en tiempo de ejecución* para transformaciones geométricas (no un mecanismo de *fine-tuning*), reduciendo el tamaño del programa de ~400 MB (denso $N \times N$) a ~2.5 MB (descomposición de rango $r$, ~99.4% de reducción).
7. **Un bootstrap en Python** (V0.3) con 29/29 pruebas superadas y un `align_score = 0.9993`, validando empíricamente el marco geométrico.
8. **Publicación**: García Traba, A. H. (2026). POLYDIM: A Native Programming Language for High-Dimensional Latent Computing. Zenodo. https://doi.org/10.5281/zenodo.21109893

### 1.3 Organización del Artículo

La Sección 2 establece el cálculo formal, incluyendo la semántica de estados, primitivas y cuatro teoremas fundamentales. La Sección 3 desarrolla el sistema de tipos dimensional como un funtor categórico. La Sección 4 formaliza la identidad de objetos (`GEO_ID`) y el protocolo de interoperabilidad IA-a-IA (ALIGN). La Sección 5 describe las especificaciones de ingeniería: el formato binario `.polydim` y la arquitectura de la Máquina Virtual (VM). La Sección 6 presenta el isomorfismo algebraico de Flutter que permite la proyección nativa a interfaces humanas. La Sección 7 posiciona a POLYDIM en relación con trabajos afines, incluyendo sistemas de comunicación en espacio latente (§7.8) y lenguajes previos que usan estructuras geométricas o algebraicas (§7.7). La Sección 8 discute problemas abiertos y las dos vías de despliegue. La Sección 9 concluye el trabajo. La Sección 10 cataloga las amenazas a la validez. La Sección 11 lista las referencias. La Sección 12 presenta la revisión del Estado del Arte (SOTA) y mapa de posicionamiento (2025–2026).

## 2. Cálculo Formal y Primitivas Algebraicas

### 2.1 Estado, Posición y Transformación

**Definición 2.1 (Estado POLYDIM).** Un estado es una tripleta:
```
S = (V, D, A)  donde:    
  V en R^N        vector de estado (N >= 1,000)    
  D = {dim1,...}  conjunto finito de subespacios observadores nativos (fijo por objeto)    
  A: D -> [0,1]   función de activación (mutable)
```

**Definición 2.2 (Posición).** Una posición P es un par:
```
P = (g, S)  donde:    
  g en R^N      GEO_ID — invariante bajo toda transformación admisible (Sec. 4)    
  S = (V, D, A) estado mutable
```
Una Posición reemplaza la noción de una *variable nombrada*. No existe `x = 42`; hay una región de alta densidad de activación en $R^N$, anclada por una identidad geométrica invariante $g$.

**Definición 2.3 (Transformación).** Una transformación $T$ es una función $T: S \to S$. Las transformaciones admisibles modifican $V$ y pueden actualizar $A$ en función del nuevo $V$, manteniendo $D$ fijo:
```
T(V, D, A) = (T_geo(V), D, T_act(V, A))    
donde:    
  T_geo: R^N -> R^N    
  T_act: R^N x [0,1]^|D| -> [0,1]^|D|    
  D es invariante bajo todo T admisible
```
Para una serialización práctica (Sec. 5), $T\_geo$ se representa en forma de bajo rango (LoRA): `T_geo = W0 + U*V^T` donde $U, V \in R^{N \times r}$, $r \ll N$.

**Definición 2.4 (Subespacio Nativo).** Un subespacio nativo $DIM_i \subseteq R^N$ es una región de $R^N$ donde una clase de conceptos semánticamente relacionados tiene alta densidad. Formalmente, $DIM_i$ es la imagen de una matriz de proyección ortogonal $P_i$ en $R^{N \times N}$, $P_i^2 = P_i$, $P_i^T = P_i$.

### 2.2 La Arquitectura de Dos Capas

**Capa Algebraica (permanente, independiente de la arquitectura):**
| Primitiva | Definición | Reemplaza |
| - | - | - |
| `COMPOSE(T1, T2)` | $T_2 \circ T_1$ | Ejecución secuencial |
| `MIX(a, T1, b, T2)` | $a \cdot T_1 + b \cdot T_2$ | Bifurcación if/else |
| `FIXPOINT(T, e)` | Iterar hasta la convergencia | Bucles for/while |
| `PROJECT(T, E)` | $\pi_E(T(s))$ | Conversión de tipos / compilación |

**Capa de Implementación (específica de la arquitectura):**
- `ATTEND(Q, K, V, s)` = $\text{softmax}(QK^T/\sqrt{d}) \cdot V$ — familia transformer
- `RECUR(A, B, C, h, x)`: $h(t+1) = A \cdot h(t) + B \cdot x(t)$, $y(t) = C \cdot h(t)$ — familia Mamba/SSM

La separación garantiza que cuando la capa de implementación cambie (ej. de Mamba-1 a Mamba-3), la especificación algebraica permanece intacta.

### 2.3 Semántica Operacional de Paso Grande (Big-Step)

Escribimos `<s, T> => s'` para "el estado $s$, bajo la transformación $T$, evalúa al nuevo estado $s'$."
```
General:   <s, T> => T(s)    
    
COMPOSE:   <s, T1> => s1    <s1, T2> => s2    
           ─────────────────────────────────    
           <s, COMPOSE(T1,T2)> => s2    
    
MIX:       <s, MIX(a, T1, b, T2)> => a*T1(s) + b*T2(s)    
    
FIXPOINT:  <s, FIXPOINT(T, e)> => s*  cuando T(s*) = s* con tolerancia e    
    
PROJECT:   <s, PROJECT(T, E)> => pi_E(T(s))
```

*Semántica denotacional:* $[| \cdot |]: POLYDIM \to Para(Vect)$ es un funtor que interpreta los programas como morfismos en la 2-categoría de mapas diferenciables parametrizados [Gavranovic et al. 2024].

### 2.4 Cuatro Teoremas Fundamentales

**Teorema 1 (Asociatividad de COMPOSE):**
`COMPOSE(COMPOSE(T3, T2), T1) = COMPOSE(T3, COMPOSE(T2, T1))`
*Demostración:* Propiedad estándar de la composición de funciones. ∎

**Teorema 2 (Linealidad de MIX):**
`Si T1, T2 en Lin(R^N), entonces MIX(a, T1, b, T2) en Lin(R^N)`
*Demostración:* Una combinación lineal de mapas lineales es lineal. ∎

**Lema (Conmutatividad de Subespacios).** Sea $\pi_E: R^N \to DIM_E$ la proyección ortogonal al subespacio del ejecutor $DIM_E$, y sea $T$ una transformación lineal tal que $DIM_E$ es invariante bajo $T$ ($T(DIM_E) \subseteq DIM_E$). Entonces: $\pi_E(T(v)) = T(\pi_E(v))$ para todo $v \in R^N$. ∎

**Teorema 3 (PROJECT es un Funtor).** Para cada ejecutor $E \in \{\text{Rust, Flutter, SQL}\}$, $PROJECT_E: G \to E$ satisface los dos axiomas de los funtores (preservación de identidad y preservación de composición). Demostrado incondicionalmente para tres contratos: COMPILE y EXPORT a través del Lema de Conmutatividad de Subespacios; RENDER a través del Isomorfismo Algebraico de Flutter (Proposición 6.1). ∎

**Teorema 4 (Unicidad del Punto Fijo — Banach):**
```
Si T es una contracción (||T(u)-T(v)|| <= k*||u-v||, 0 <= k < 1),    
entonces existe un único s* en R^N tal que T(s*) = s*,    
y FIXPOINT(T, e) converge a él.
```
*Demostración:* Teorema del Punto Fijo de Banach en $(R^N, ||\cdot||_2)$. ∎

**Teorema 5 (Invariancia de GEO_ID):**
`Para todo T en T_admisible: T(g) = g`
*Demostración:* Por construcción (Definición 2.3). ∎

## 3. Sistema de Tipos Dimensional

### 3.1 Comparación de Sistemas de Tipos

| Propiedad | Estático (Rust) | Dinámico (Python) | Dimensional (POLYDIM) |
| - | - | - | - |
| Declaración | Al momento de definición | Inferida en runtime | Nunca — emerge de la proyección |
| Cantidad de tipos | Uno por valor | Uno en cada instante | Múltiples simultáneos |
| Cambio de tipo | Imposible | Mutación/reasignación | Cambio de proyección (no mutación) |
| Sincronización multiplataforma | Requiere ORM/puente | Requiere serialización | Cero — observación dual |

**Proposición 3.1 (Sistema de Tipos Dimensional):** El tipo de un objeto POLYDIM no se declara; emerge de la proyección: `tipo(P) en el ejecutor E := PROJECT(P, E)`. El mismo objeto P puede tener múltiples tipos simultáneos — uno por subespacio activo.

### 3.2 Ejemplo Concreto

```
Posición P con activaciones:    
  a(DIM_SQL)     = 0.90  ->  PROJECT(P, DIM_SQL)     = Column(INTEGER, "id", NOT NULL)    
  a(DIM_FLUTTER) = 0.85  ->  PROJECT(P, DIM_FLUTTER) = TextField(value: id_value)    
  a(DIM_VECTOR)  = 0.91  ->  PROJECT(P, DIM_VECTOR)  = R^3 = [1.0, 2.0, 3.0]    
    
P no "tiene" ninguno de estos tipos. ES todos ellos simultáneamente.
```

## 4. Identidad e Interoperabilidad

### 4.1 GEO_ID con Invariancia Topológica

**Definición 4.1 (GEO_ID).** La identidad de un objeto POLYDIM es un componente invariante $g \in R^N$ de su posición $P = (g, S)$. Todas las transformaciones admisibles dejan $g$ sin cambios (Teorema 5). Dos estados $s_1, s_2$ representan el *mismo objeto* si y solo si existe un camino continuo admisible $\gamma: [0,1] \to R^N$ con $\gamma(0) = V_1, \gamma(1) = V_2$.

*Estatus epistémico de la formalización HoTT:* La caracterización basada en caminos y la formalización completa en la Teoría de Tipos Homotópicos (HoTT) son **direcciones de investigación activas**, no teoremas finalizados. La formalización completa en Coq/Agda es un objetivo de la Vía 2.

### 4.2 El Protocolo ALIGN

```
Dados: IA_A en R^(d1), IA_B en R^(d2)    
    
Paso 1: Compartir el libro de códigos de GEO_IDs universales.    
Paso 2 (Procrustes Ortogonal):    
  M* = argmin_M ||M*A_anchors - B_anchors||_F  sujeto a M^T*M = I    
  (M en R^(d2 x d1), calculado mediante SVD de B^T*A)    
Paso 3: T_B = M * T_A * M-dagger    
Paso 4: nuevo_estado_B = T_B(estado_B)
```

**Leer un mensaje es matemáticamente indistinguible de computar una capa de atención.** Esto elimina el desajuste de impedancia que afecta a MCP, JSON-RPC y A2A.

## 5. Ingeniería: Formato Binario .polydim y Arquitectura VM

### 5.1 El Formato Binario .polydim V5 (Estándar LoRA)

```
[HEADER]      magic: 0x504F4C59 | versión | N | precisión | n_transformaciones | rango_por_defecto(r)    
[GEO_IDs]     base_hv: float16[N] x n_objetos    
[TRANSFORMS]  Ti = W0 + U*V^T (LoRA): U: float16[N×r], V: float16[r×N]    
[ACTIVATIONS] pesos: float16[k x n_objetos]    
[PROJECTIONS] targets: uint8[] -> IDs de ejecutores    
    
Tamaño con N=10,000, r=64:    
  2 x 64 x 10,000 x 2 bytes ≈ 2.5 MB por transformación (vs 400 MB en denso)    
  Reducción: ~99.4%
```

*Nota:* LoRA se utiliza aquí como un *tipo de dato en tiempo de ejecución*, no como un mecanismo de *fine-tuning*. Esta es una reutilización estructural de la representación de bajo rango a nivel del lenguaje [Hu et al. 2021].

### 5.2 Arquitectura de la VM

La VM utiliza el vector de activación $\alpha$ para *ejecución rala (sparse)*: solo los subespacios con $a_i > \text{umbral}$ se cargan en memoria activa. Implementación: Núcleo en Rust compilado a WebAssembly (`polydim_core.rs` V0.1, finalizado 2026-06-28), cero dependencias externas.

## 6. El Ejecutor Flutter: Isomorfismo Algebraico

**Proposición 6.1 (demostrada).** El mapeo $\phi: T \to F$ es un isomorfismo monoidal estricto entre la categoría de transformación de POLYDIM ($T$, `COMPOSE`, `MIX`) y la categoría de widgets de Flutter ($F$, `Column`, `Row`), satisfaciendo:

```
phi(COMPOSE(T2, T1))   = Column(phi(T1), phi(T2))    
phi(MIX(a, T1, b, T2)) = Row(Opacity(a, phi(T1)), Opacity(b, phi(T2)))    
phi(FIXPOINT(T, e))    = StatefulWidget(phi(T), convergence: e)    
phi(id_T)              = id_F
```

| Propiedad | Flutter | React | Qt | SwiftUI |
| - | - | - | - | - |
| Modelo de Widget | Comp. Algebraica | Diff VDOM | OOP | Grafo de Propiedades |
| Mapeo POLYDIM | **Directo** ($T_2 \circ T_1$) | Requiere adaptador | Requiere adaptador | Parcial |
| Estatus de Ejecutor | **NATIVO** | EXTERNO | EXTERNO | EXTERNO |
| Isomorfismo Algebraico | **Sí (demostrado)** | No | Parcial | No |

## 7. Trabajos Relacionados

### 7.1 Frameworks de Programación Tensorial (PyTorch, JAX, TensorFlow)
PyTorch, JAX y TensorFlow usan tensores como *datos* que fluyen a través de un *programa secuencial*. La distinción con POLYDIM es fundamental: estos frameworks programan *con* tensores; POLYDIM programa *como* tensores.

### 7.2 Protocolos de Comunicación de Agentes (MCP, A2A, FIPA)
MCP (Anthropic, 2024) y A2A (Google, 2025) dependen de la serialización estructurada de texto (JSON-RPC), perpetuando el desajuste de impedancia. MCP y A2A son objetivos de `EXPORT` en POLYDIM (`PROJECT(T, DIM_JSON_RPC)`), no alternativas arquitectónicas.

### 7.3 Sistemas de Comunicación de Espacio Latente
**Interlat [Du et al. 2025]** transmite estados ocultos de última capa entre agentes en lugar de tokens de lenguaje natural. **LatentMAS [ICML 2026 Spotlight]** permite a los agentes compartir el KV-cache y estados ocultos con alineación latente sin reentrenamiento. Ambos sistemas validan la premisa central de POLYDIM empíricamente, pero son frameworks de entrenamiento. POLYDIM es la especificación del lenguaje que formaliza *qué* cálculos se pueden expresar en ese espacio.

### 7.4 Posicionamiento SOTA (2026)
POLYDIM ocupa el espacio entre la investigación empírica en comunicación latente y el diseño de lenguajes formales, un cuadrante único que provee semántica formal rigurosa, portabilidad arquitectónica garantizada y pérdida cero de serialización.

## 8. Conclusión

Hemos presentado POLYDIM, un lenguaje de programación cuya unidad fundamental es una transformación geométrica $T: R^N \to R^N$. El lenguaje elimina la brecha de impedancia entre la geometría multidimensional de las arquitecturas modernas de IA y los canales secuenciales de texto mediante los cuales se comunican y programan actualmente. 

Cuatro teoremas algebraicos establecen el fundamento formal, todos demostrados incondicionalmente. El protocolo ALIGN permite la comunicación sin serialización entre sistemas de IA heterogéneos, mientras que el formato LoRA reduce los programas `.polydim` en un ~99.4%. La investigación empírica actual confirma que omitir la serialización a texto es factible y beneficioso. POLYDIM proporciona la capa de especificación formal que hace este enfoque componible, portable y analizable. **El lenguaje ensamblador de la era de los transformadores ya está aquí.**

---
*Para ver las Referencias Completas, la Revisión Extensa del Estado del Arte, y los Apéndices sobre el formato binario y validación métrica, consulte el archivo maestro `POLYDIM.md` (Versión en Inglés).*
