# POLYDIM_DEST: docs/paper/research_notes/
# TÍTULO: Auditoría Crítica de Claude - Softmax Categórico y Derivación Física

## 1. El Hueco del Softmax No Lineal (O'Neill y Giry)
La semántica categórica de Q/K/V está cerrada por O'Neill (2025) en `Para(Vect)`. Pero la no-linealidad del softmax sigue abierta. Las tres vías candidatas para formalizar el forward pass del softmax son:
1. Variedad con esquinas (probabilidad simplex).
2. Categoría de conjuntos convexos/afines.
3. Categorías probabilísticas (Teoría de la medida, **Monoides de Giry** / Categorías de Markov).
*Nota:* Gavranović (Categorical Cybernetics) categoriza el backward pass (gradientes vía lentes), no el softmax del forward pass.

## 2. Kan Extensions y Spinor Embeddings
- **Kan Extension Transformers (Mahadevan 2026):** Formaliza la atención como una extensión de Kan ponderada. Confirma que la intuición de "Kan Horn Filling" de POLYDIM era correcta pero acotada.
- **Álgebra de Clifford en Alta Dimensión:** Escalar rotores a 10,000-D genera una explosión combinatoria de $2^{10000}$. La solución de POLYDIM de truncar el grado a 10 es heurística. La literatura más cercana a semántica con Clifford son los **Spinor Embeddings (White 2024)**.

## 3. La Falsa Analogía Física (Pezzaglia/Castro)
El track de "Topos de Grothendieck" cita a Pezzaglia (1996) y Castro (C-space) como base fundamental. Esto es un error metodológico fatal: esos trabajos pertenecen a la física teórica de los 90s/00s (teoría de cuerdas, relatividad polidimensional) y **no tienen ningún puente publicado hacia redes neuronales**. Usar esto como base para IA debilita la credibilidad del documento.

## 4. Hyperdimensional Computing (HDC / VSA)
La capa práctica del track "Cognitive Programming" (`bind/mix`, alineación) es funcional porque en realidad es **Vector Symbolic Architectures (VSA)** encubierto. Funciona por sí solo sin necesidad de la envoltura teórica de Topos/Física.
