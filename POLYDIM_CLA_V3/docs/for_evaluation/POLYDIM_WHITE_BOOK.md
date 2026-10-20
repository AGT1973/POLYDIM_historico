# POLYDIM V3: White Book (Arquitectura Matemática y Topológica)

**Estado:** Propuesta Arquitectónica Teórica
**Ámbito:** Arquitecturas sub-cuadráticas para NLP y Sistemas Multi-Agente

## 1. Abstracto Académico y Contexto SOTA

El cuello de botella de la caché Clave-Valor (KV-Cache Wall) en la arquitectura clásica del Transformer ($\mathcal{O}(N^2)$) ha impulsado la búsqueda de modelos eficientes para contextos largos. El estado del arte actual (SOTA) ha resuelto en gran medida este problema de latencia mediante **State Space Models (SSMs)** como Mamba y arquitecturas de atención lineal como **SparseK Attention**. Estas arquitecturas logran complejidad lineal $\mathcal{O}(N)$ comprimiendo el contexto en estados ocultos de dimensión fija o pre-filtrando los tokens relevantes.

Sin embargo, los SSMs clásicos aplican proyecciones lineales que comprimen el estado a costa de degradar las relaciones geométricas exactas de alta dimensión (Catastrophic Forgetting en contextos extremos). **POLYDIM V3** se propone como una investigación arquitectónica teórica que busca resolver este problema combinando las **Álgebras de Clifford (Isometría)** y la **Teoría de Haces / Laplacianos de Hodge (Consenso)**. Nuestra tesis no es inventar la geometría, sino proponer su aplicación rigurosa como mecanismo de enrutamiento para evitar la pérdida entrópica del estado latente.

## 2. La Arquitectura POLYDIM V3 (Motor Híbrido)

A diferencia de Mamba, que depende de descensos de gradiente continuo sobre matrices dinámicas, POLYDIM V3 divide la ejecución en dos espacios ortogonales:
- **Espacio Estadístico:** Extracción de features (Redes Neuronales, SwiGLU).
- **Espacio Geométrico:** Enrutamiento determinista y sin pérdida (Clifford/Hodge).

### 2.1 Fase de Ingestión: Lema de Johnson-Lindenstrauss
Mapeamos tokens a una hiperesfera $\mathcal{S}^{D-1}$ en $\mathbb{R}^{D}$ (donde $D \approx 10,000$). Aplicamos una proyección ortogonal determinista (GEO_ID) que conserva las distancias topológicas latentes. 

### 2.2 Fase de Enrutamiento: Isometría de Clifford
*Basado en el teorema clásico de W.K. Clifford (1878) y las formulaciones de Hestenes (1984).*

La pérdida de información (colapso semántico) ocurre cuando aplicamos transformaciones no isométricas a la memoria. En POLYDIM, actualizamos el estado geométrico usando exclusivamente **Rotores de Clifford** (producto de dos reflexiones de Householder).

**Proposición Aplicada:** Si el estado del contexto está codificado en un tensor $X \in \mathcal{S}^{D-1}$, cualquier actualización $X' = R X \tilde{R}$ garantiza matemáticamente que $\|X'\|_2 = \|X\|_2 = 1.0$. Aplicamos esto para asegurar que el vector de estado jamás se desvanezca ni explote numéricamente en contextos de $1M+$ tokens.

### 2.3 Fase de Consenso: Filtros de Chebyshev sobre Laplacianos de Hodge
Para que los tokens distribuidos obtengan atención global sin calcular la matriz densa $\mathcal{O}(N^2)$, utilizamos el **Laplaciano de Hodge Disperso** ($\tilde{\Delta_0}$).

Sabemos que aplicar difusión pura en GNNs causa *Oversmoothing* (Kipf & Welling, 2016). Para evitar que todos los tokens colapsen en el $\ker(\Delta_0)$, proponemos:
1. **Truncamiento:** Usar polinomios de Chebyshev de orden bajo ($K \ll N$).
2. **Regularización de Dirichlet:** Un término de anclaje $\beta(X_0 - X)$ que fuerza al estado a no olvidar su *GEO_ID* original.

## 3. Resolución de Problemas y Limitaciones

### 3.1 El Problema del Shadow Loss
Definimos *Shadow Loss* no como la proyección inicial de embeddings, sino como la pérdida de entropía que ocurre cuando dos agentes (LLMs) se comunican mediante **texto discreto plano** (1D). El protocolo **PMTP (Polydim Matrix Transfer Protocol)** en Rust serializa los tensores internos *Block-Sparse Row (BSR)* y los transmite directamente por red, eliminando el texto plano y preservando el estado geométrico.

### 3.2 Análisis de Complejidad y Crossover Point
POLYDIM opera en $\mathcal{O}(N \cdot k \cdot D)$. Con un $D$ masivo (10,000), para secuencias cortas ($N < 1,000$), POLYDIM es matemáticamente **más lento** que un Transformer clásico ($N^2 < N \cdot 10,000$). 
Nuestra arquitectura sólo tiene sentido asintótico cuando $N \gg D$ (ej. $N = 1,000,000$), donde superamos a la atención densa, compitiendo en el mismo nicho asintótico que Mamba o SparseK.

## 4. Conclusión y Futuro Empírico
Este documento establece la viabilidad matemática de un LLM basado en isometría de Clifford y Laplacianos de Hodge. Sin embargo, carece de validación a escala SOTA. El trabajo futuro requiere abandonar simulaciones *toy* para entrenar un modelo base de 130M parámetros en *The Pile* y medir su *Perplexity* contra Mamba-1/3 y Transformers con Flash Attention 2.
