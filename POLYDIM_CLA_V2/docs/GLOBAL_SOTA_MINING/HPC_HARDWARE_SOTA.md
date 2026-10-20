# Estado del Arte (SOTA): Aceleración Hardware/Silicio de Operaciones Ortogonales Complejas

## Resumen Ejecutivo
La optimización en variedades topológicas (Manifolds de Stiefel, transformadas de Cayley, Espinores) está atravesando un cambio de paradigma radical en el hardware de IA moderno. Las arquitecturas de silicio actuales (NVIDIA Tensor Cores, Cerebras CS-3, Groq LPU, Tenstorrent) están severamente penalizadas por la inversión directa de matrices $O(N^3)$ (necesaria para Cayley puro o descomposiciones QR/SVD). 

Para evadir este cuello de botella y explotar los motores GEMM (General Matrix Multiply), los frameworks y compiladores (XLA, CUDA, TT-Metalium) implementan **identidades algebraicas (Woodbury)** y **aproximaciones iterativas libres de inversión (Series de Neumann, Newton-Schulz)**.

---

## 1. El Problema Matemático vs. La Arquitectura del Silicio
*   **Transformada de Cayley:** Parametriza matrices ortogonales $\mathbf{R} = (\mathbf{I} - \mathbf{A})^{-1}(\mathbf{I} + \mathbf{A})$. Su cuello de botella ineludible es la inversa.
*   **Variedad de Stiefel ($St(n, p)$):** Esencial para mantener pesos ortogonales. La optimización tradicional requiere "retracciones" (ej. QR) que son secuenciales, lentas y hostiles para las GPUs.
*   **Restricción del Hardware:** Tensor Cores, WSE (Cerebras) y LPUs (Groq) están diseñados para multiplicar y sumar masivamente (MACs). Carecen de instrucciones nativas eficientes para solvers lineales pesados como `inv()` o `svd()`.

---

## 2. Estrategias de Evasión de $O(N^3)$

### A. Identidad de Woodbury y Curvatura (K-FAC, NGD)
La Identidad de Woodbury calcula la inversa de una matriz modificada por actualizaciones de bajo rango, reduciendo la complejidad drásticamente (ej. de $O(N^3)$ a $O(N^2K)$).
*   **XLA / JAX:** XLA no tiene una "primitiva de Woodbury", pero al codificar la identidad en JAX, el compilador fusiona (kernel fusion) los tensores en secuencias de GEMMs ultrarrápidas, esquivando la inversión pesada y manteniendo saturadas las ALUs.

### B. Cayley-Neumann (Aproximación Inversion-Free)
Para modelos de fine-tuning ortogonal (como OFTv2), la matriz inversa de Cayley $(\mathbf{I} - \mathbf{A})^{-1}$ se reemplaza por la **Serie de Neumann** (aproximación por sumas de potencias).
*   **Impacto de Hardware:** Convierte una inversión estancada en iteraciones de multiplicaciones matriz-matriz (MVM/MMM). Esto es música para los Tensor Cores, permitiendo transformaciones topológicas en tiempo real.

### C. Iteraciones Newton-Schulz (Stiefel sin Retracción)
Para mantener los pesos en la variedad de Stiefel, el SOTA está abandonando el QR en favor de las **iteraciones Newton-Schulz** (usado recientemente en el optimizador **Muon** para pre-entrenamiento de LLMs). 
*   Este método es 100% "retraction-free", no requiere inversas y depende estrictamente de multiplicaciones. Encaja perfectamente en la jerarquía de memoria y ejecución de GPUs y TPUs.

---

## 3. Panorama por Arquitectura / Fabricante

### NVIDIA (CUDA / Tensor Cores)
NVIDIA lidera la ejecución. Librerías personalizadas (`Cayley_SGD`) mapean las matemáticas hacia enfoques "Matrix-Free" o "Input-Centric". Los Tensor Cores aceleran Neumann y Newton-Schulz de forma nativa porque reducen las complejas reglas de ortogonalidad a un denso pipeline de operaciones de suma-multiplicación flotante (FP16/BF16).

### Cerebras (CS-3 WSE)
El Wafer-Scale Engine posee una arquitectura Dataflow que domina el SpMM (Álgebra Lineal Dispersa). Calcular una inversa pura obligaría a una sincronización global catastrófica a nivel de latencia. En su lugar, el uso de Woodbury a través del **Cerebras Software Language (CSL)** descompone el problema en fragmentos locales que fluyen por el silicio sin congestionar el interconnect.

### Tenstorrent (Tensix Cores & TT-Metalium)
Los núcleos Tensix basados en RISC-V están optimizados para programación escalar combinada con aceleración tensorial. El código open-source de **TT-Metal** hace referencias explícitas a **kernels de "Woodbury / Neumann series"**. Esto demuestra que Tenstorrent incorpora la evasión algorítmica directamente en su base compilable, evitando ahogar el núcleo con instrucciones pseudo-invertidas complejas.

### Groq (LPU)
La LPU es puramente determinista, ejecutando sin caché dinámica usando masivas reservas de SRAM. Los algoritmos como la descomposición QR o inversiones exactas destruyen el scheduling de su compilador (al depender de bifurcaciones de datos). Al usar Newton-Schulz (Stiefel) o Woodbury, Groq obtiene un **grafo computacional predecible (bucles de MVMs exactos)**, logrando que el compilador programe el paso de datos de forma impecable y sin bloqueos de ciclo.

### XLA y Ecosistema de Compilación
Los compiladores no "conocen" la topología compleja de forma nativa, pero la ingestión de identidades como Woodbury permite reescribir grafos computacionales lentos en operaciones primitivas que el hardware devora rápidamente.

---

## Conclusión
La industria no construye silicio específico para espacios no euclidianos o transformadas de Cayley directas; lo que está sucediendo es una **traducción arquitectónica**. Todo cálculo topológico, de Stiefel o Espinores que requería $O(N^3)$, está siendo refactorizado por matemáticos hacia algoritmos "inversion-free" (Woodbury, Neumann, Newton-Schulz). Esto permite que conceptos de alta dimensionalidad se ejecuten sobre las mismas unidades hardware que calculan atención densa, usando su ancho de banda masivo para resolver iteraciones puras de producto de matrices.
