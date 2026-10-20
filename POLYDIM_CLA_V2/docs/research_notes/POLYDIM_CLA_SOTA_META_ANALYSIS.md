# Meta-Análisis del SOTA: Árboles de Investigación y Vacíos Genuinos

**Propuesta de frente de investigación — POLYDIM-CLA / MPAT_V3_02**
**Fuente:** Auditoría externa independiente (Claude)

---

## 1. El Diagnóstico del Ecosistema

POLYDIM-CLA no opera en un frente virgen. Los pilares que componen la arquitectura original tienen programas de investigación paralelos, activos y con validación empírica. El mapa de la literatura actual se divide en cuatro árboles:

### Árbol 1: Topos + Lógica Intuicionista sobre Redes (Desde 2021)
Lo que POLYDIM define como el espacio latente (Topos de Grothendieck, \(\Omega\) como clasificador de subobjetos, lógica de Heyting) ya existe en:
- **Belfiore & Bennequin (2021):** *"Topos and Stacks of Deep Neural Networks"* y *"The Topos of Transformer Networks"*.
- **Sheaf Neural Networks (2020-2022):** Hansen & Gebhart, Bodnar et al. (NeurIPS 2022 - *Neural Sheaf Diffusion*). Resuelven la inconsistencia lógica/geométrica usando haces celulares sobre grafos con laplacianos de haz entrenables.
- **Sheaf-Theoretic Planning (2026):** Aplica esta lógica intuicionista a sistemas multiagente autónomos.

**Gap concreto:** La línea Sheaf-NN tiene código y benchmarks (e.g., mitigación de oversmoothing). POLYDIM-CLA necesita demostrar que el `FIXPOINT` real converge usando restricción/pegado de secciones con un Laplaciano de haz entrenable.

### Árbol 2: Rotores de Clifford / Multivectores (Ya en Producción)
El uso de Álgebra Geométrica en Redes Neuronales es una frontera altamente activa:
- **GATr (Geometric Algebra Transformer, 2023):** Representa entradas y estados ocultos como multivectores en álgebra geométrica proyectiva, con atención equivariante.
- **LaB-GATr (2024):** Extensión escalable para mallas biomédicas masivas.
- **Clifford Group Equivariant Neural Networks (2023):** Garantías formales de equivarianza.
- **Versor (2026):** Arquitecturas de secuencia basadas puramente en rotores.

**Conjugación posible:** La invariancia descubierta en POLYDIM (la subálgebra par es cerrada, la truncación de grado \(\le 2\) no lo es) es exactamente el tipo de invariante que GATr necesita para normalizaciones sensibles a grado. 

### Árbol 3: Gradiente Natural (Fisher-Rao + Woodbury + K-FAC)
Terreno maduro, no es frontera. Se basa en Martens & Grosse (2015). El SOTA actual en Optimización de Transporte a gran escala usa Sinkhorn entrópico con aproximaciones low-rank + diagonal, y la comunidad LLM usa optimizadores tipo Shampoo. La aplicación de Woodbury para la distancia de Mahalanobis en Sinkhorn es correcta, pero acotada.

### Árbol 4: VSA/HDC (Vector Symbolic Architectures)
Sigue siendo un problema abierto. La integración LLM+NeSy vía VSA choca contra el "memory wall" al escalar con LLMs de miles de millones de parámetros. No hay integración a escala productiva probada.

---

## 2. La Verdadera Frontera de Investigación (El Hueco SOTA)

El árbol real de investigación muestra que cada comunidad vive en su silo:

```text
Topos/Haces sobre redes ──┬── Belfiore-Bennequin (topos de transformers)
                          └── Sheaf NNs (Hansen-Gebhart) → Sheaf-Theoretic Planning

Álgebra geométrica en NN ─┬── Clifford layers (Brandstetter 2023)
                          ├── GATr / LaB-GATr / Versor (escalables)
                          └── Clifford Group Equivariant NN (garantías)

Simbólico distribuido ────── VSA/HDC (Cuello de botella de memoria)
```

**La laguna real de POLYDIM-CLA es conectar estos árboles.**
Nadie en la literatura está combinando simultáneamente:
1. Lógica interna de topos para invariantes de seguridad.
2. Multivectores de Clifford entrenables a escala (GATr).
3. Binding simbólico VSA.

### La Pregunta Fundamental para POLYDIM-CLA:
> **¿Puede un rotor de Clifford (arquitectura GATr) servir como el *restriction map* de un haz celular (Sheaf-NN), unificando la equivarianza geométrica con la consistencia lógica local-global?**

Eso no existe publicado. Ahí hay frontera genuina.
