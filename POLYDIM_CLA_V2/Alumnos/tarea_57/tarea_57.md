# Tarea 57 — POLYDIM (Escuadrón 6: Alineación Espectral y Transporte Óptimo)

## 1. Enunciado original (banco maestro)

> "57 (Test 57): Actúa como especialista en OCaml/Haskell. Implementa el algoritmo de Alineación Espectral que dos nubes de puntos de dimensiones diferentes emparejando sus primeros k autovectores."

Este enunciado está gramaticalmente incompleto: la cláusula "el algoritmo de Alineación Espectral **que** dos nubes de puntos..." no tiene verbo principal (el banco combina fragmentos de plantillas — rol + fragmento técnico — y varias combinaciones quedan rotas).

## 2. Reconstrucción del enunciado

**Versión reformulada, investigable:**

> Implementar un algoritmo de *Alineación Espectral* (Spectral Alignment / Manifold Alignment no supervisado) que permita comparar dos nubes de puntos $X \in \mathbb{R}^{n_1 \times d_1}$ e $Y \in \mathbb{R}^{n_2 \times d_2}$, con $d_1 \neq d_2$ (y en general $n_1 \neq n_2$), embebiéndolas en un espacio latente común de dimensión $k$ mediante sus primeros $k$ autovectores no triviales del Laplaciano de grafo de cada nube, y evaluar empíricamente la calidad de esa alineación con una métrica estándar.

Qué se conserva del enunciado original: el rol (especialista en estructuras formales/álgebra — aquí interpretado como "razonar con precisión sobre la buena definición del objeto matemático", ya que "OCaml/Haskell" en el banco funciona como *rol decorativo*, no como restricción de lenguaje de implementación — el curso pide entrega en Python y/o Rust, ver Sección 5) y el núcleo técnico: "alineación espectral", "dos nubes de puntos de dimensiones diferentes", "primeros k autovectores".

Qué se corrige: se explicita el verbo faltante y, más importante, se señala una ambigüedad matemática real que el enunciado da por sentada sin advertirla (ver Sección 4: "premisa no establecida").

## 3. Modelo matemático estándar

### 3.1 Construcción del grafo y Laplaciano

Para cada nube de puntos $X = \{x_1,\dots,x_{n_1}\}\subset\mathbb{R}^{d_1}$, se construye un grafo de $k_{nn}$-vecinos más cercanos con pesos gaussianos:

$$
W^X_{ij} = \exp\left(-\frac{\|x_i-x_j\|^2}{2\sigma_X^2}\right) \quad \text{si } x_j \in \mathrm{kNN}(x_i),\ \ 0 \text{ en otro caso}
$$

simetrizado como $W^X \leftarrow \tfrac{1}{2}(W^X + (W^X)^\top)$. El Laplaciano normalizado (Laplaciano de Laplace–Beltrami discreto, Belkin & Niyogi, 2003) es:

$$
L^X = I - D_X^{-1/2} W^X D_X^{-1/2}, \qquad D_X = \mathrm{diag}\left(\sum_j W^X_{ij}\right)
$$

Análogamente se define $L^Y$ para $Y \in \mathbb{R}^{n_2\times d_2}$. Nótese que $L^X \in \mathbb{R}^{n_1\times n_1}$ y $L^Y \in \mathbb{R}^{n_2\times n_2}$ — **viven en espacios de dimensión distinta** ($n_1 \neq n_2$ en general). Esto es central para la Sección 4.

### 3.2 Descomposición espectral y embedding latente

Se resuelven los problemas de autovalores $L^X v_i = \lambda_i v_i$, $L^Y u_i = \mu_i u_i$, ordenando $0=\lambda_0\le\lambda_1\le\dots$. Se descarta el autovector trivial $\lambda_0=0$ (vector constante, componente conexa) y se toman los siguientes $k$:

$$
\Phi_X = [v_1,\dots,v_k] \in \mathbb{R}^{n_1\times k}, \qquad \Phi_Y = [u_1,\dots,u_k] \in \mathbb{R}^{n_2\times k}
$$

$\Phi_X$ y $\Phi_Y$ son los *embeddings espectrales* (Laplacian Eigenmaps, Belkin & Niyogi 2003) de cada nube en un espacio común $\mathbb{R}^k$, independiente de $d_1, d_2$.

### 3.3 Resolución de ambigüedades (signo y rotación intra-eigenspacio)

Los autovectores están definidos hasta signo, y si $\lambda_i$ tiene multiplicidad $>1$, hasta una rotación ortogonal arbitraria dentro del autoespacio. Se fija el signo normalizando la componente de mayor valor absoluto a positiva:

$$
v_i \leftarrow \mathrm{sign}(v_i[\arg\max_j |v_i[j]|]) \cdot v_i
$$

### 3.4 Métrica de calidad de alineación (estándar, verificable)

Como $\Phi_X$ y $\Phi_Y$ tienen distinto número de filas ($n_1 \ne n_2$), **no se comparan celda a celda**. Se usan dos métricas estándar, ambas explícitamente solicitadas por el marco del curso:

1. **Distancia espectral (L2 sobre autovalores)**: compara las formas de las dos variedades independientemente de la correspondencia punto-a-punto:
$$
d_{\mathrm{spec}} = \left\| (\lambda_1,\dots,\lambda_k) - (\mu_1,\dots,\mu_k) \right\|_2
$$

2. **Distancia de Wasserstein-2 entre los embeddings** (transporte óptimo, Peyré & Cuturi, 2019), que sí admite comparar distribuciones de puntos de tamaños distintos en el mismo espacio $\mathbb{R}^k$:
$$
W_2(\Phi_X,\Phi_Y) = \left(\min_{\pi \in \Pi(\mu_X,\mu_Y)} \int \|a-b\|^2 \, d\pi(a,b)\right)^{1/2}
$$
donde $\mu_X,\mu_Y$ son las medidas empíricas uniformes sobre las filas de $\Phi_X,\Phi_Y$. Se resuelve con el algoritmo de Sinkhorn (regularización entrópica) por ser el método estándar y escalable.

## 4. Razonamiento y premisa no establecida

El enunciado original asume implícitamente que "emparejar los primeros k autovectores" es una operación directa y bien definida. **Esto no es cierto en general** y se documenta como hallazgo, no como hecho:

- Si $n_1 = n_2$ (mismo número de puntos, ej. dos nubes que son deformaciones una de otra), "emparejar autovectores" puede interpretarse como comparar $v_i$ con $u_i$ directamente (correlación o error L2), y el problema es el clásico *spectral graph matching* de Umeyama (1988).
- Si $n_1 \neq n_2$ (caso general, y el más natural dado que el enunciado dice "dimensiones diferentes" sin decir "mismo cardinal"), los autovectores **no viven en el mismo espacio vectorial** y no se pueden emparejar componente a componente. Lo que sí es matemáticamente válido es comparar: (a) los autovalores (espectros, que son invariantes independientes de $n$), y (b) las distribuciones de los embeddings resultantes vía transporte óptimo.

Por lo tanto, se investiga la pregunta como abierta: el código implementa ambos casos ($n_1=n_2$ y $n_1\neq n_2$) y reporta en qué régimen la "alineación por emparejamiento directo" es válida y en cuál requiere la relajación a distancia de Wasserstein. No se da por sentado que el enunciado original sea correcto tal como está planteado — se corrige explícitamente esta premisa.

## 5. Desarrollo y código

Ver `tarea_57_alineacion_espectral.py`. El script:
1. Genera dos nubes de puntos sintéticas de dimensiones y cardinalidades distintas (una esfera en $\mathbb{R}^3$ muestreada con $n_1$ puntos, y una versión "estirada y ruidosa" de la misma variedad en $\mathbb{R}^5$ con $n_2 \neq n_1$ puntos), para poder verificar si la alineación espectral recupera que ambas provienen de la misma variedad subyacente.
2. Construye los grafos kNN, Laplacianos, y calcula los primeros $k$ autovectores de cada uno (usando `scipy.sparse.linalg.eigsh`).
3. Resuelve la ambigüedad de signo.
4. Calcula $d_{\mathrm{spec}}$ y, si `POT` está disponible, $W_2$ vía Sinkhorn; si no, cae a un emparejamiento por Hungarian (asignación óptima, `scipy.optimize.linear_sum_assignment`) sobre una matriz de costos como aproximación discreta al transporte óptimo — documentando explícitamente que es una aproximación, no el óptimo continuo.
5. Reporta ambas métricas y un veredicto textual de si la alineación es consistente con "misma variedad subyacente" (umbral relativo, no absoluto — se documenta el umbral usado).

**No se ejecuta como parte de la entrega** (solo se hizo un sanity-check interno de sintaxis). Para comprobarlo vos mismo:

```bash
pip install numpy scipy scikit-learn --break-system-packages
# opcional, para la distancia de Wasserstein exacta vía Sinkhorn:
pip install pot --break-system-packages

python3 tarea_57_alineacion_espectral.py
```

Salida esperada: dos números ($d_{\mathrm{spec}}$ y $W_2$ o su aproximación Hungarian) y un veredicto. Si `POT` no está instalado, el script lo indica y usa el fallback — comparalo corriendo ambas variantes para ver cuánto se acerca la aproximación discreta al óptimo continuo.

## 6. Conclusión

El núcleo matemático de la tarea es *manifold alignment* vía Laplacian Eigenmaps (Belkin & Niyogi 2003) combinado con *optimal transport* (Peyré & Cuturi 2019) para comparar embeddings de cardinalidad distinta — no un "emparejamiento" ingenuo de autovectores como sugiere la redacción original del banco. La rigurosidad matemática exige distinguir el caso $n_1=n_2$ (spectral matching, comparación directa) del caso general $n_1\neq n_2$ (comparación de distribuciones, transporte óptimo), algo que el enunciado original no distingue. Correspondele a vos correrlo y, si querés, variar $n_1, n_2, k$ para ver cómo se comporta $W_2$ cuando las dos nubes efectivamente vienen de variedades distintas (deberías ver que $W_2$ crece).
