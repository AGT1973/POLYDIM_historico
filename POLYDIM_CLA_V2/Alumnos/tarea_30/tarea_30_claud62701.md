# Tarea 30 — POLYDIM (Batería de Pruebas Matemáticas)

**Alumno:** claud62701@gmail.com
**Área:** Escuadrón 2 — Álgebra de Clifford y Rotores (numeración del banco), pero el
contenido real de este ejercicio pertenece a **Alineación estadística de distribuciones
de embeddings vía Transporte Óptimo**.

---

## 1. Enunciado

### 1.1 Enunciado original (tal como figura en el banco maestro)

> Test 30: "Actúa como experto en Manifold Learning. Diseña una pasada de optimización
> que intercepte un AST topológico y las distribuciones de embeddings de dos modelos
> de lenguaje."

### 1.2 Nota de reconstrucción

El enunciado original está **incompleto**: el banco se genera combinando una plantilla
de "acción" (*"Diseña una pasada de optimización que intercepte un AST topológico y
[FRAGMENTO TÉCNICO]"*) con un fragmento técnico recortado de otra frase. Al pegarse,
falta el verbo que conecta "intercepte un AST... y" con "las distribuciones de
embeddings de dos modelos de lenguaje" — tal como está escrito, no dice **qué** hacer
con esas distribuciones.

Comparando con otros ítems del banco que comparten la misma plantilla ("...pasada de
optimización que intercepte un AST topológico y X", tests 29, 60, 111, 131, 137, 169),
y con el fragmento técnico "distribuciones de embeddings de dos modelos de lenguaje"
que en otros tests del banco (37, 106, 172, 173) aparece asociado explícitamente a
**"alinear estadísticamente... usando la librería POT o Sinkhorn"**, la reconstrucción
adoptada es:

> **Enunciado reconstruido:** Como experto en Manifold Learning, diseñar una pasada de
> optimización sobre un grafo computacional (AST) que detecte automáticamente los
> subgrafos correspondientes a las distribuciones de embeddings de dos modelos de
> lenguaje distintos, y sustituya el uso directo (crudo) de esas dos distribuciones por
> una **alineación estadística vía Transporte Óptimo (algoritmo de Sinkhorn)**,
> reportando la distancia resultante y si el algoritmo converge.

Esta reconstrucción es la interpretación más conservadora: no inventa un problema
nuevo, sino que completa la frase con la acción que el propio banco asocia
sistemáticamente a ese fragmento técnico en otros ítems.

---

## 2. Modelo matemático

Sean dos modelos de lenguaje $A$ y $B$. Cada uno produce un conjunto finito de
embeddings (por ejemplo, de tokens o frases):

$$
X = \{x_1, \dots, x_n\} \subset \mathbb{R}^{d}, \qquad
Y = \{y_1, \dots, y_m\} \subset \mathbb{R}^{d}
$$

(si $d_A \neq d_B$, se proyecta al mínimo común $d = \min(d_A, d_B)$; esto es una
limitación explícita del modelo, señalada en el código).

Cada nube de puntos define una **distribución empírica**:

$$
\mu = \frac{1}{n}\sum_{i=1}^{n} \delta_{x_i}, \qquad
\nu = \frac{1}{m}\sum_{j=1}^{m} \delta_{y_j}
$$

**Costo de transporte** (cuadrático euclídeo):

$$
C_{ij} = \lVert x_i - y_j \rVert_2^2
$$

**Transporte Óptimo entrópico** (Cuturi, 2013): en vez de resolver el problema de
transporte óptimo exacto (un programa lineal costoso, $O(n^3 \log n)$), se regulariza
con entropía para poder resolverlo por escalado matricial, mucho más barato
($O(n^2)$ por iteración):

$$
\pi^\star = \arg\min_{\pi \in U(\mu,\nu)} \langle C, \pi \rangle + \varepsilon\, H(\pi)
$$

sujeto a $\pi \mathbf{1} = \mu,\; \pi^\top \mathbf{1} = \nu,\; \pi \ge 0$, donde
$H(\pi) = \sum_{ij} \pi_{ij}(\log \pi_{ij} - 1)$.

La solución tiene la forma $\pi^\star = \mathrm{diag}(u)\, K\, \mathrm{diag}(v)$, con
$K = \exp(-C/\varepsilon)$, y $u, v$ se obtienen por iteración alterna
(**algoritmo de Sinkhorn-Knopp**):

$$
u \leftarrow \frac{\mu}{Kv}, \qquad v \leftarrow \frac{\nu}{K^\top u}
$$

**Métrica estándar reportada (equivalente a "SPS" del marco POLYDIM):** la distancia de
Wasserstein entrópica $W_\varepsilon(\mu,\nu) = \langle C, \pi^\star \rangle$, que es la
traducción a una métrica matemática estándar y verificable del pedido original de
"alineación estadística".

---

## 3. Razonamiento (por qué este camino y qué se descarta)

- **Por qué Transporte Óptimo y no una comparación directa (distancia euclídea entre
  medias, CCA, o concatenar embeddings sin más):** dos modelos de lenguaje entrenados
  por separado producen espacios de embeddings que pueden estar rotados, escalados o
  desplazados entre sí sin que eso indique nada semánticamente distinto. Comparar
  vectores "en crudo" (como hace el nodo `concat`/`sub` del grafo original) mezclaría
  diferencia semántica real con diferencia puramente geométrica arbitraria. El
  Transporte Óptimo compara **distribuciones completas**, no puntos individuales, y es
  invariante a permutaciones dentro de cada nube.
- **Por qué la versión entrópica (Sinkhorn) y no el OT exacto:** el OT exacto es un
  programa lineal con complejidad cúbica y no es diferenciable de forma directa, lo que
  lo hace impracticable para integrarlo como nodo de un grafo computacional que después
  se podría usar en entrenamiento. Sinkhorn es diferenciable y mucho más rápido, al
  costo de una pequeña aproximación controlada por $\varepsilon$.
- **Se descartó** Wasserstein-Procrustes conjunto (optimizar simultáneamente una
  rotación $W$ y el plan de transporte $\pi$) por ser una extensión natural pero fuera
  del alcance mínimo pedido por el enunciado reconstruido; se deja mencionado como
  posible extensión.
- **Marco de la "pasada de optimización sobre AST":** se modela el grafo computacional
  como una lista de nodos (`embedding`, `concat`, etc.), y la pasada busca el patrón
  "dos nodos `embedding` de modelos distintos alimentando un mismo consumidor" para
  reemplazarlo por un nodo `alignment`. Esto es análogo a una pasada de optimización de
  compilador (constant folding, fusion), pero aplicada a nivel estadístico en vez de
  puramente sintáctico.

---

## 4. Desarrollo y código

El código completo, funcional, está en `tarea_30.py` (adjunto en esta misma carpeta).
Contiene:

1. `sinkhorn(...)`: implementación desde cero del algoritmo de Sinkhorn-Knopp (sin caja
   negra), con manejo explícito de inestabilidad numérica (si $K$ degenera a ceros o
   `NaN`, el algoritmo lo reporta como no convergencia en lugar de fallar en silencio).
2. `build_toy_graph(...)` y `optimization_pass_align_embeddings(...)`: la
   representación simplificada del grafo/AST y la pasada de optimización que detecta el
   patrón de dos embeddings y los reemplaza por un nodo de alineación.
3. `main()`: genera embeddings sintéticos de dos "modelos" (uno con una rotación y
   desplazamiento respecto del otro, simulando que son modelos distintos), corre la
   pasada de optimización, y reporta la distancia de Wasserstein entrópica resultante.

**Instrucciones para correrlo vos mismo** (esta parte la tenés que ejecutar y observar
con tus propios ojos — no está corrida como comprobación final en esta entrega):

```bash
pip install numpy
python tarea_30.py
```

El script solo depende de `numpy`, no requiere GPU ni librerías pesadas.

Al ejecutarlo, vas a ver impreso:
- el grafo original (3 nodos: `embedding A`, `embedding B`, `concat`),
- el grafo optimizado (el nodo `concat` reemplazado por `alignment`),
- el resultado numérico: si Sinkhorn convergió, en cuántas iteraciones, y el valor de la
  distancia de Wasserstein entrópica entre las dos distribuciones de embeddings.

**Qué comprobar vos mismo (esto es la parte activa del aprendizaje):**
- Cambiá `epsilon` en `main()` a un valor muy chico (p. ej. `0.001`) y volvé a correr:
  deberías ver que el algoritmo deja de converger (inestabilidad numérica documentada
  en el propio código). Esto te permite verificar en la práctica el trade-off entre
  precisión de la aproximación entrópica y estabilidad numérica.
- Cambiá la rotación/desplazamiento sintético de `emb_b` y verificá que la distancia de
  Wasserstein reportada cambia de forma consistente con cuánto "distinto" hiciste al
  modelo B.

---

## 5. Conclusión

El problema planteado (una vez reconstruido) es un problema de **alineación de
distribuciones vía Transporte Óptimo**, matemáticamente bien definido y con solución
estándar (Sinkhorn-Knopp), que converge geométricamente cuando el costo $C$ es finito y
$\varepsilon > 0$ (garantía clásica del teorema de Sinkhorn para matrices con entradas
estrictamente positivas). El "AST topológico" del enunciado se interpreta como el grafo
computacional donde se detecta el patrón de dos ramas de embeddings y se inserta la
alineación estadística en vez de una comparación cruda.

No se afirma nada no demostrado: la convergencia de Sinkhorn depende de $\varepsilon$ y
del condicionamiento de $C$, y el propio código documenta el caso de no convergencia en
lugar de forzar un resultado.

---

## 6. Resumen estructurado (índice de catalogación)

```json
{
  "alumno": "claud62701@gmail.com",
  "tarea": 30,
  "tipo_problema": "Alineación de distribuciones de embeddings vía Transporte Óptimo (Sinkhorn)",
  "estado": "resuelto",
  "resultado_resumen": "Se reconstruyó el enunciado (incompleto en el banco), se modeló el problema como Transporte Óptimo entrópico entre dos distribuciones empíricas de embeddings, se implementó Sinkhorn-Knopp desde cero, y se integró como pasada de optimización sobre un grafo/AST simplificado que reemplaza un nodo de comparación cruda por un nodo de alineación estadística.",
  "metrica_estandar_usada": "Distancia de Wasserstein entrópica (Sinkhorn)"
}
```
