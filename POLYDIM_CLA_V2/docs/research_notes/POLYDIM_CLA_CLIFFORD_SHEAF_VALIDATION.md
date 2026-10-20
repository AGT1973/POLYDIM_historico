# Validación SOTA: Rotores de Clifford como Mapas de Restricción en Haces Celulares

**Propuesta de frente de investigación — POLYDIM-CLA / MPAT_V3_02**
**Fuente:** Trazas de razonamiento y búsqueda SOTA (Auditoría externa 2026)

---

## 1. La Hipótesis del "Santo Grial"

**Pregunta fundamental:** *¿Puede un rotor de Clifford (arquitectura GATr) servir como el 'restriction map' de un haz celular (Sheaf-NN), unificando equivarianza geométrica con consistencia lógica local-global?*

Esta pregunta busca fusionar:
- **Equivarianza geométrica:** Vía Álgebra Geométrica (GATr, multivectores, rotores de Clifford).
- **Consistencia local-global:** Vía Teoría de Haces Celulares (Sheaf Neural Networks, Laplaciano de Haz).

---

## 2. Validación Empírica en la Literatura (2020-2026)

La auditoría realizó búsquedas profundas cruzando ambos dominios y confirmó que la intersección es **genuinamente novedosa y publicable**. 

### A. El lado de los Haces (Sheaf NNs) y Laplacianos
- **Bodnar et al. (2022) - *Neural Sheaf Diffusion*:** Prueba la existencia de soluciones armónicas no triviales para casos de mapas de restricción ortogonales. Esto valida que restringir los mapas a transformaciones ortogonales (la acción de un rotor) tiene sentido en la teoría de haces.
- **Singer & Wu - *Vector diffusion maps and the connection Laplacian* / Cheeger Inequality:** Modela las aristas de un grafo como transformaciones ortogonales (holonomía), alineándose perfectamente con la perspectiva de teoría de gauge (gauge theory) necesaria para POLYDIM.
- **arXiv:2605.06395 (Mayo 2026) - *Consistent Geometric Deep Learning via Hilbert Bundles and Cellular Sheaves*:** El paper más reciente unificando teoría de fibrados (bundle theory) y haces celulares.

### B. El lado de Clifford y Álgebra Geométrica
- **GATr (Geometric Algebra Transformer):** Opera sobre multivectores donde los rotores (versores de grado par) actúan vía *sandwich product* para preservar equivarianza bajo el grupo ortogonal. 
- **Clifford Group Equivariant Neural Networks (2023):** Confirma que la acción del grupo forma un automorfismo ortogonal que respeta la graduación del multivector.

---

## 3. Análisis Estructural: "La Trampa" de la Ortogonalidad

La respuesta superficial es "un rotor es una matriz ortogonal y las Sheaf-NNs usan mapas ortogonales, luego funciona". Sin embargo, el análisis riguroso revela cuatro restricciones arquitectónicas críticas (la "trampa"):

### Paso 1: El Lado Bueno (Analogía de la Brújula)
La intuición es correcta. La acción sandwich de un rotor (\(x \mapsto R x R^{-1}\)) es un mapa ortogonal estructurado en bloques. Los mapas ortogonales ya se usan en Sheaf-NNs por eficiencia y por su analogía con el transporte paralelo. El rotor actúa como un "dial de conversión" equivariante que alinea mediciones locales relativas.

### Paso 2: El Problema de Dominio (Stalks Homogéneos)
Un rotor conjuga un espacio contra sí mismo: mapea \(F(v) \to F(v)\), pero un mapa de restricción en un haz debe mapear el nodo a la arista \(F(v) \to F(e)\). 
Para que el rotor sirva como mapa de restricción, **hay que forzar que \(F(e)\) sea una copia nominal del mismo espacio graduado que \(F(v)\)**. Esto elimina la ventaja inicial del haz de tolerar stalks de distinta dimensión. 

### Paso 3: Preservación de Grado (La limitación heterogénea)
Si quisiéramos que el nodo \(F(v)\) fuera de grado 1 (vectores) y la arista \(F(e)\) de grado 2 (bivectores), un rotor puro **no sirve**. El grupo de Clifford respeta la graduación del multivector (una rotación nunca convierte un vector en un bivector). 
**Solución arquitectónica:** Para recuperar la heterogeneidad de grado/dimensión sin perder equivarianza, el mapa de restricción no puede ser un rotor puro, sino **productos exteriores o geométricos contra un multivector de referencia**, que sí son equivariantes y sí cambian el grado.

### Paso 4: Precio en Expresividad
Los rotores solo cubren \(SO(n)\) (conexo, preserva orientación), que es un subconjunto mucho más estricto que \(GL(d, \mathbb{R})\) e incluso que el grupo ortogonal general. Estar limitado a \(SO(n)\) asegura (vía Bodnar 2022) la existencia de secciones armónicas reales, lo que garantiza que el Laplaciano del haz funcionará bien, pero restringe la expresividad de la red neuronal.

---

## 4. Conclusión de Novelty (Vacío Comprobado)

Mientras que las álgebras de Clifford y las redes de haces han sido exploradas extensamente por separado, **combinar productos geométricos/rotores de Clifford como mapas de restricción de un haz celular es genuinamente novedoso**. 

### Aplicación Directa a POLYDIM-CLA (Cohomología de Haces + Clifford)
Esta arquitectura permite medir la deriva de estado latente AI2AI midiendo la **holonomía del multivector y la consistencia del haz**. 
En un ciclo de comunicación entre tres agentes, componer los mapas de restricción (rotores/productos geométricos) alrededor del ciclo:
- Produce la identidad (holonomía trivial) \(\implies\) **Consistencia Semántica Global** (ker del Laplaciano \(= 0\)).
- Produce una rotación/multivector residual (obstrucción) \(\implies\) **Desacuerdo / Deriva Dimensional** (medido por la Energía de Dirichlet del Laplaciano de Haz).

**Veredicto:** Este modelo riguroso de restricción equivariante (Frente B+D) es el núcleo matemático definitivo para el paper en POPL, reemplazando la "Cota de Preservación Compuesta" genérica por una métrica topológica y algebraica exacta.
