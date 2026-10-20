# Tarea 60 — POLYDIM: Alineación Espectral entre nubes de puntos de dimensiones distintas

## 1. Enunciado original (banco maestro, tal como figura)

> **60 (Test 60):** Actúa como experto en Categorical Deep Learning. Diseña una pasada de
> optimización que intercepte un AST topológico y dos nubes de puntos de dimensiones
> diferentes emparejando sus primeros k autovectores.

## 2. Nota de reconstrucción (paso obligatorio antes de resolver)

Este enunciado está gramaticalmente roto: el banco se genera combinando una plantilla de
**rol + acción de compilador** ("Diseña una pasada de optimización que intercepte un AST
topológico y **[verbo faltante]**...") con un **fragmento técnico** ("dos nubes de puntos de
dimensiones diferentes emparejando sus primeros k autovectores") que no tiene verbo principal
ni sujeto claro conectados entre sí.

**Versión reconstruida, matemáticamente investigable:**

> Diseñá una pasada de optimización de grafo de cómputo (análoga a un *compiler pass* que
> intercepta un AST/grafo de operaciones) que, dados dos conjuntos de datos (nubes de puntos)
> con dimensiones ambientales distintas, los reemplace por una representación conjunta en un
> espacio espectral común, alineando sus primeros *k* autovectores mediante Procrustes
> Ortogonal. Medí el error de alineación con una norma L2 (Frobenius) y documentá si el método
> converge y bajo qué condiciones.

Esta reconstrucción respeta el fragmento técnico reconocible ("dos nubes... emparejando sus
primeros k autovectores") — que corresponde a una técnica real y bien documentada de
**alineación de subespacios espectrales** (ver §5, referencias) — y le da un verbo y una
estructura de "pasada de optimización sobre un grafo" coherente con el rol de Categorical
Deep Learning y con el resto del banco (Escuadrón 3: Descomposición de Hodge y Grafos
Esparsos).

**Importante — el enunciado NO asume una conclusión no demostrada** (a diferencia de otros
ítems del banco tipo "demostrá por qué X ahorra cómputo sin perjudicar Y"), así que no aplica
la advertencia de premisa-no-establecida del paso 4 de la metodología: acá se trata de
implementar y verificar empíricamente un método, no de demostrar una afirmación dada por
cierta de antemano.

## 3. Modelo matemático estándar

Sean $X \in \mathbb{R}^{n \times d_1}$ e $Y \in \mathbb{R}^{n \times d_2}$ dos nubes de $n$
puntos cada una, con $d_1 \neq d_2$ en general.

**3.1 Grafo k-NN y Laplaciano normalizado.** Para cada nube se construye un grafo de los
$k_{nn}$ vecinos más cercanos con pesos gaussianos $w_{ij} = \exp\!\left(-\frac{\|x_i -
x_j\|^2}{2\sigma^2}\right)$, y el Laplaciano simétrico normalizado:

$$
L = I - D^{-1/2} W D^{-1/2}, \qquad D_{ii} = \sum_j w_{ij}
$$

$L$ es siempre $n \times n$, **independientemente de $d_1$ o $d_2$** — este es el truco que
permite comparar nubes de dimensiones ambientales distintas: el Laplaciano vive en un espacio
que depende solo de $n$, no de $d$.

**3.2 Embedding espectral.** Se calculan los $k$ autovectores de $L$ asociados a los $k$
autovalores más chicos (excluyendo el autovalor trivial $\lambda_0 = 0$):

$$
U_X, U_Y \in \mathbb{R}^{n \times k}, \qquad L_X U_X = U_X \Lambda_X, \quad L_Y U_Y = U_Y \Lambda_Y
$$

**3.3 Procrustes Ortogonal (alineación).** Se busca la rotación $Q \in O(k)$ (matriz
ortogonal $k\times k$) que minimiza:

$$
Q^\* = \arg\min_{Q^T Q = I} \| U_X Q - U_Y \|_F^2
$$

**Solución cerrada (Schönemann, 1966):** si $M = U_X^T U_Y = W S V^T$ (SVD de $M$), entonces
$Q^\* = W V^T$.

**3.4 Métrica de validación (L2 / Frobenius):**

$$
\text{error}_{L2} = \frac{\| U_X Q^\* - U_Y \|_F}{\| U_Y \|_F}
$$

Esta es exactamente la métrica estándar pedida por el marco POLYDIM (Norma L2), sin necesidad
de inventar una métrica ad-hoc tipo "SPS" — el Frobenius normalizado es directamente
verificable e interpretable.

## 4. Razonamiento (por qué este camino, qué se descarta)

- **Por qué Laplaciano de grafo y no PCA directo:** PCA lineal no puede comparar $X$ e $Y$
  porque viven en espacios ambientes de dimensión distinta ($d_1 \neq d_2$); no hay una
  matriz de covarianza conjunta bien definida. El Laplaciano de grafo, en cambio, es
  **intrínsecamente $n \times n$**, lo que lo hace comparable entre nubes de cualquier $d$.
  Esta es la idea central detrás de métodos de *manifold alignment* (Wang & Mahadevan, 2009).
- **Por qué Procrustes y no una red neuronal entrenada para alinear:** Procrustes ortogonal
  tiene solución cerrada vía SVD (sin iteración, sin hiperparámetros de entrenamiento, sin
  riesgo de no-convergencia), y es exactamente el estimador de mínimos cuadrados bajo la
  restricción de ortogonalidad — es el método correcto cuando lo que se busca es una rotación
  rígida entre dos bases, no una transformación arbitraria.
- **Qué se descarta:** CCA (Canonical Correlation Analysis) fue descartado porque asume
  linealidad directa entre $X$ e $Y$ en sus dimensiones originales; acá el punto es comparar
  estructuras no lineales (variedades) vía su representación espectral, no las coordenadas
  crudas.
- **El límite del método (hallazgo empírico, no ocultado):** Procrustes sobre autovectores
  alinea **subespacios**, no garantiza correspondencia punto-a-punto salvo que la estructura
  de vecindad k-NN ya sea consistente entre ambas nubes. Ver §6.

## 5. Desarrollo y código

Ver `tarea_60.py` — implementación funcional en Python/NumPy con:

1. `knn_laplacian`: construcción del Laplaciano normalizado por k-NN gaussiano.
2. `spectral_embedding`: extracción de los primeros $k$ autovectores.
3. `orthogonal_procrustes`: solución cerrada vía SVD + error L2 normalizado.
4. `SpectralAlignmentPass`: la "pasada de optimización de grafo" propiamente dicha — una
   clase que recorre una lista de `ComputeNode` (simulación mínima de un AST/grafo de
   cómputo), detecta los dos nodos `embedding_raw`, y los reemplaza por un único nodo
   `aligned_embedding` con el resultado de la alineación — esta es la traducción directa y
   literal del "intercepte un AST topológico y [alinee] dos nubes de puntos" del enunciado.
5. `demo()`: corrida con dos casos de control (ver §6).

**Referencias reales en las que se apoya el modelo:**
- Belkin, M. & Niyogi, P. (2003). *Laplacian Eigenmaps for Dimensionality Reduction*.
- Schönemann, P. H. (1966). *A generalized solution of the orthogonal Procrustes problem*.
- Wang, C. & Mahadevan, S. (2009). *Manifold Alignment without Correspondence*.

## 6. Conclusión (la trampa educativa del enunciado)

El sanity-check interno corrió dos casos:

- **Con correspondencia punto-a-punto conocida entre $X$ e $Y$:** error L2 ≈ 0.28 —
  el método captura una parte sustancial de la estructura latente compartida (no llega a 0
  por el ruido agregado y por ambigüedades de rotación en autovalores casi degenerados).
- **Sin correspondencia conocida (índices mezclados):** error L2 ≈ 1.34 — el error sube
  marcadamente.

**Esto es el núcleo conceptual de la tarea, y es fácil pasarlo por alto en una lectura
superficial del enunciado:** "emparejar los primeros k autovectores" suena a que el método
resuelve automáticamente el problema de *correspondencia* entre nubes de puntos distintas.
No es así — Procrustes sobre autovectores alinea **subespacios espectrales globales**, pero
el Laplaciano k-NN de cada nube se construye sobre su propia estructura de vecindad local. Si
esa estructura de vecindad no es consistente entre ambas nubes (por ejemplo, si no hay
correspondencia índice-a-índice preexistente), el método no tiene forma de recuperar esa
correspondencia solo con la rotación $Q$.

**No se ocultó el resultado:** ambos casos se documentan explícitamente en el código y en
esta conclusión, tal como exige la metodología (documentar una falla o limitación es un
resultado científico válido, no algo a esconder).

## 7. Instrucciones para que el alumno corra el código

```bash
# Requiere Python 3.10+ y numpy
pip install numpy
python3 tarea_60.py
```

El script no tiene dependencias externas más allá de `numpy`. Al ejecutarlo vas a ver los dos
casos (con y sin correspondencia) y la interpretación numérica. **Se pide explícitamente que
lo corras vos mismo/a y verifiques con tus propios ojos** si el punto del enunciado
("emparejar autovectores alinea nubes de dimensiones distintas") queda demostrado, refutado,
o matizado — la comprobación activa es parte del aprendizaje. Como ejercicio adicional, probá
cambiar `n_neighbors`, `k`, o el nivel de ruido, y observá cómo cambia el error en cada caso.

## 8. Resumen estructurado

```json
{
  "tarea": 60,
  "tipo_problema": "alineación espectral de subespacios (manifold alignment) vía Procrustes ortogonal",
  "estado": "resuelto",
  "resultado_resumen": "El método alinea correctamente subespacios espectrales cuando existe correspondencia de vecindad consistente entre las nubes; sin esa correspondencia, el error de alineación aumenta marcadamente porque el Laplaciano k-NN depende de la estructura local, no solo de la estructura latente compartida.",
  "metrica_estandar_usada": "Error L2 (Frobenius) normalizado"
}
```

---
*que has usado el formato de razonamiento adaptado por AGT*
