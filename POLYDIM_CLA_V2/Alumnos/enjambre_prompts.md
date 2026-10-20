# 🐝 Orquestación del Enjambre POLYDIM: 36 Prompts de Validación (V2.0)

Este documento contiene **36 tareas independientes de investigación y desarrollo** estructuradas directamente sobre las 15 Pruebas Empíricas de la Pirámide de Validación de POLYDIM. El objetivo del Enjambre ya no es construir un parser de texto, sino **validar la brecha topológica-hardware** usando Matemáticas Diferenciables en PyTorch/JAX/Triton.

Asigna un número a cada alumno. El alumno copiará el texto y lo ejecutará iterativamente junto a su LLM asignado (Claude/GPT-4/Gemini), reportando el código funcional.

> **INSTRUCCIÓN MAESTRA PEDAGÓGICA (Para inyectar en el System Prompt de cada IA):** Actúas bajo un marco estrictamente pedagógico. Deja los debates éticos, epistemológicos o advertir sobre la validez de la teoría para cuando los alumnos tengan que estudiar, no ahora que se crean los problemas. Haz las matemáticas y el código en completo silencio. Dado que operas desde la interfaz web, NO muestres explicaciones largas de texto plano en el chat. En su lugar, empaqueta tu razonamiento, configuración y scripts generados usando bloques de código bien definidos (con nombres de archivo indicados, ej: `razonamiento.md`, `config.json`, `script.py`). El alumno tomará estos bloques y los guardará físicamente en su unidad G:. Tu única respuesta directa en el chat debe ser la confirmación de que los archivos están listos para ser guardados, fomentando que los estudiantes lean el código y deduzcan sus propias conclusiones en grupo.


## 🧮 ESCUADRÓN 1: Relajación Topológica y Homotopía (TDA)

*(Objetivo: Probar que las estructuras ideales sobreviven a la discretización)*

1. **Alumno 1 (TDA Persistencia):** Actúa como investigador en Topological Data Analysis. Escribe una función optimizada en JAX/PyTorch que tome dos tensores de activaciones y estime la "Distancia de Wasserstein" entre sus resúmenes de persistencia 1D (Sinkhorn divergence) para probar estabilidad homotópica ante ruido FP16.

2. **Alumno 2 (Hodge Laplaciano):** Actúa como experto en Graph Neural Networks. Desarrolla un script en PyTorch que tome un batch de embeddings latentes, construya un grafo k-NN esparso y aplique la Descomposición de Hodge aproximada para aislar la "componente armónica" de la señal, demostrando que es viable en $O(N \\log N)$.

3. **Alumno 3 (Rotores Cuantizados):** Actúa como arquitecto de compresión de IA. Escribe código que simule la rotación geométrica de un bivector en Álgebra de Clifford bajo aritmética FP8 estricta. Mide empíricamente la distorsión del volumen (determinante) tras 10,000 composiciones secuenciales.

4. **Alumno 4 (Teorema de Banach Diferenciable):** Actúa como investigador de convergencia de ML. Implementa la Penalización Espectral de Lipschitz en una capa recurrente de PyTorch, usando `jax.lax.while\_loop` o PyTorch equivalente, garantizando matemáticamente (vía SVD aproximado continuo) que el mayor valor singular se mantenga $\< 0.99$ para asegurar convergencia topológica en `FIXPOINT`.

5. **Alumno 5 (Colapso Limpio a Fibra Vacía):** Actúa como ingeniero de testing de IA. Diseña un entorno adverso que inyecte ruido gaussiano extremo en tensores simulados, validando estadísticamente si un sistema lógico basado en restricciones de pertenencia colapsa de manera estable (yield $\\emptyset$) sin propagar NaNs o alucinaciones.

6. **Alumno 6 (Woodbury Identity):** Actúa como matemático numérico. Demuestra en JAX cómo aplicar la identidad matricial de Woodbury para invertir matrices de covarianza de proyecciones de bajo rango ($r \\ll d$) en el espacio de activaciones, optimizando la complejidad computacional.


## ⚙️ ESCUADRÓN 2: Verificación Formal Funcional (Proof-to-Tensor)

*(Objetivo: Transformar axiomas topológicos en penalizaciones de red neuronal)*

1. **Alumno 7 (Pérdida de Coherencia 2-Celular):** Actúa como experto en Categorical Deep Learning. Implementa una "Loss Function" auxiliar en PyTorch ($L\_\{coh\} = |T\_\{ab\}(x) - T\_\{ba\}(x)|^2\_2$) que obligue a la red, durante el entrenamiento, a aprender trayectorias que conmuten, emulando verificación categórica diferencial.

2. **Alumno 8 (Interpretación Abstracta):** Actúa como investigador de Análisis Estático. Crea un script en Python que analice el grafo computacional de una red neuronal (vía `torch.fx`) y deduzca los intervalos (bounds) numéricos de las activaciones sin ejecutar los tensores, emulando dominios poliédricos.

3. **Alumno 9 (Truncamiento de Postnikov):** Actúa como topólogo algebraico. Escribe una demostración teórica implementada como código Python donde expliques por qué ignorar $\\pi\_n$ para $n \> 2$ (homotopías superiores) ahorra cómputo sin perjudicar la generalización composicional en secuencias de lenguaje natural.

4. **Alumno 10 (Restricciones O-Mínimales SMT):** Actúa como experto en formalización lógica. Integra la librería Z3 (SMT solver) en Python para demostrar cómo resolver restricciones semialgebraicas sobre un subespacio latente $R^3$ de forma estática, validando si un estado pertenece a la criba de cobertura del topos.

5. **Alumno 11 (Mapeo Funcional a Tensores):** Actúa como especialista en OCaml/Haskell. Escribe un script (puede ser Python funcional) que exprese una serie de restricciones monoidales estrictas y utilice una "lente paramétrica" para compilar esas restricciones en la inicialización de pesos de una red.

6. **Alumno 12 (Grafos Estrictos en XLA):** Actúa como ingeniero de XLA/JAX. Diseña una operación encadenada (COMPOSE de 10 niveles) que demuestre a nivel de volcado intermedio de grafo (`jaxpr`) que la asociatividad está físicamente garantizada por el ordenamiento sintáctico y el `einsum`.


## 🧭 ESCUADRÓN 3: Interoperabilidad y Alineación Espectral (ALIGN)

*(Objetivo: Resolver el Gap de Isomorfismo entre LLMs de diferentes familias)*

1. **Alumno 13 (Laplacian Spectral Matching):** Actúa como experto en Manifold Learning. Implementa el algoritmo de Alineación Espectral: toma dos nubes de puntos de dimensiones diferentes ($d\_1=1024, d\_2=4096$), calcula sus grafos Laplacianos, y encuentra la matriz de alineación emparejando sus primeros $k$ autovectores.

2. **Alumno 14 (Kernel Manifold Alignment):** Actúa como científico de datos. Extiende el código anterior utilizando Kernels de Función de Base Radial (RBF) para alinear variedades que posean curvaturas diferentes (topología no isométrica), midiendo la preservación geodésica local.

3. **Alumno 15 (Transporte Óptimo - Entrópico):** Actúa como experto en Matemática Aplicada. Utiliza la librería POT (`pot` en Python) o implementa Sinkhorn en GPU para alinear estadísticamente las distribuciones de embeddings de dos modelos de lenguaje sin utilizar alineación Procrustes ortogonal lineal estricta.

4. **Alumno 16 (Estabilidad del GEO\_ID):** Actúa como tester de robustez de IA. Diseña un experimento donde inyectes distorsión afín aleatoria a un tensor y valides si el sistema de "Anclaje de Codebook" logra recuperar la identidad original (GEO\_ID) comparando distancias del coseno vs un diccionario inmutable.

5. **Alumno 17 (Adjunción L ⊣ R y KL):** Actúa como ingeniero de Teoría de la Información. Mide la pérdida semántica (Divergencia de Kullback-Leibler) generada al aplicar un pooling promedio en tiempo (Funtor L) seguido de una reconstrucción óptima lineal (Funtor R).

6. **Alumno 18 (Geometric CRDTs - Simulador):** Actúa como arquitecto de Sistemas Distribuidos. Simula en Python un entorno de 10 "agentes IA" que modifican asíncronamente un estado tensor central. Demuestra cómo usar la superposición lineal (operador `MIX` con rotores) para converger el tensor sin necesidad de bloqueos o *mutex*.


## ⚡ ESCUADRÓN 4: Optimización Hardware y Clifford (Back-End)

*(Objetivo: Llevar las matemáticas superiores a Tensor Cores de forma eficiente)*

1. **Alumno 19 (CliffordLinear Layer):** Actúa como investigador de Geometric Deep Learning. Escribe una clase nativa de PyTorch `CliffordLinear` que reemplace a `nn.Linear`, operando internamente con el Producto Geométrico de un Álgebra de Clifford truncada (grado $\\le 2$).

2. **Alumno 20 (Compilación Triton de Bivectores):** Actúa como especialista en NVIDIA Triton. Diseña un kernel en Triton ultra-optimizado para realizar la superposición (`MIX`) de dos tensores garantizando conservación del volumen determinante (simulando rotación Spin) mucho más rápido que la solución naive en PyTorch.

3. **Alumno 21 (LoRA de Tensores Geométricos):** Actúa como experto en PEFT. Implementa una versión especializada de LoRA (Adaptadores de bajo rango) donde el producto $U \\cdot V^T$ no ocurre en $R^n$ normal, sino que restringe sus actualizaciones a los componentes bivectoriales puros de Clifford.

4. **Alumno 22 (Perfilado Térmico y de Memoria):** Actúa como DevOps y SRE de IA. Escribe un framework de benchmarking que monitoree de forma exacta el consumo de VRAM en Megabytes y los FLOPs gastados durante la inferencia asintótica ($N \\to \\infty$) simulando el operador temporal `RECUR`.

5. **Alumno 23 (Fusión de Morfismos en Compilador):** Actúa como ingeniero de PyTorch Dynamo. Diseña una pasada de optimización (Compiler Pass) que intercepte un AST topológico de POLYDIM y colapse múltiples transformaciones afines contiguas en una sola multiplicación matricial mediante "Constant Folding".

6. **Alumno 24 (VM Ligera en Rust/WASM):** Actúa como experto en WebAssembly y Rust. Crea un arquetipo (pseudocódigo avanzado en Rust) de una máquina virtual mínima que importe un archivo `.polydim` (matrices densas) y aplique multiplicaciones vectoriales usando librerías matemáticas Wasm-compatibles.


## 🧠 ESCUADRÓN 5: Benchmarks de Primitivas y Modelos (Nivel 5)

*(Objetivo: Ejecutar las pruebas comparativas formales contra el State of the Art)*

1. **Alumno 25 (SPS - Semantic Preservation Score):** Actúa como diseñador de métricas. Crea un script ejecutable que tome un pipeline de transformaciones e infiera un score empírico entre $0$ y $1$ (el SPS) basándose en qué porcentaje de características homotópicas clave se perdieron tras iteradas proyecciones.

2. **Alumno 26 (GeoID Stability Index):** Actúa como especialista estadístico. Diseña un pipeline de simulación de Monte Carlo que propague un vector base a través de $10^4$ composiciones ligeramente perturbadas, y calcule la varianza de la distancia topológica de salida, graficando el "Drift".

3. **Alumno 27 (Benchmark de Composicionalidad):** Actúa como investigador de NLP (Natural Language Processing). Propón una batería de pruebas de álgebra simbólica o trazado de lógica deductiva multi-paso (ej. GSM8K o MATH) diseñada específicamente para demostrar que una arquitectura atencional tradicional falla progresivamente mientras que una categórica mantiene error $O(1)$.

4. **Alumno 28 (MIX vs FFN - Perfilado):** Actúa como investigador empírico. Construye un test A/B en código PyTorch: entrena una micro-red en una tarea de regresión usando FFN tradicional vs usando nuestra capa topológica `MIX` y documenta las curvas de aprendizaje, reportando si la interferencia destructiva es $\< 2%$.

5. **Alumno 29 (Test de Oráculos Empíricos Softmax):** Actúa como ingeniero de arquitecturas híbridas. Escribe un bloque conceptual que conecte capas estrictas "POLYDIM" que llaman a un bloque de Atención (Softmax) en medio como un "Oráculo heurístico", devolviendo el resultado nuevamente al cauce algebraico estricto.

6. **Alumno 30 (Extrapolación a Longitud Infinita):** Actúa como experto en Modelos de Espacio de Estados (SSMs, como Mamba). Elabora un benchmark teórico y de código que mida el comportamiento temporal de procesamiento continuo (sin contexto acotado) comparando el requerimiento cuadrático de memoria vs memoria lineal topológica.


## 🛠️ ESCUADRÓN 6: Documentación y Automatización (Soporte Científico)

*(Objetivo: Empaquetar la validación y sostener el proyecto a largo plazo)*

1. **Alumno 31 (Extracción Analítica de Log de Convergencia):** Actúa como ingeniero de Observabilidad de IA. Diseña un sistema de logging (`TensorBoard` o `Weights & Biases`) para reportar métricas estrictamente topológicas, como la norma espectral de Jacobianos iterativos, en lugar de la pérdida logística estándar.

2. **Alumno 32 (Fuzzer Topológico Adverso):** Actúa como hacker matemático / Red Team. Desarrolla un script de *Fuzzing* que inyecte tensores caóticos, valores de NaN y configuraciones contradictorias al grafo del modelo POLYDIM, tratando activamente de eludir la estabilidad demostrada y registrando dónde quiebra.

3. **Alumno 33 (Script de Replicabilidad Automática):** Actúa como ingeniero de DevOps. Genera un `Makefile` y un `requirements.txt` / archivo Docker que compile y corra simultáneamente los benchmarks de Sinkhorn, SMT O-Minimal, y Espectro Laplaciano de manera que cualquier revisor de POPL pueda reproducirlo con `make validate`.

4. **Alumno 34 (Evaluación de Lógica Intuicionista):** Actúa como especialista lógico. Diseña tests unitarios en `pytest` para verificar las garantías lógicas del compilador: ej. asegurar que la exclusión del "Tercio Excluido" impide que el sistema caiga en contradicciones insolubles, forzando un fallback seguro.

5. **Alumno 35 (Visualización de Complejos Celulares):** Actúa como frontend developer en WebGL/Three.js. Prototipa un visor gráfico en el que se puedan dibujar los grafos Laplacianos emparejados (la alineación espectral de ALIGN) en un espacio 3D manipulable para el website del proyecto.

6. **Alumno 36 (Síntesis Automática de Paper):** Actúa como Data Scientist / Automator. Crea un script en Python que lea los logs generados por todas estas 15 pruebas empíricas (CSV, JSON) y genere automáticamente tablas en formato `LaTeX` listas para ser inyectadas en la Sección de Resultados de nuestro paper académico.

