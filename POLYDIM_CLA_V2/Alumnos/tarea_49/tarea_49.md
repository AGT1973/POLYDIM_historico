# Tarea 49 — POLYDIM V2 / Batería de Validación Matemática

**Escuadrón de origen:** 3 — Descomposición de Hodge y Grafos Esparsos
**Rol asignado:** Investigador en Topological Data Analysis (TDA)

---

## 1. Enunciado

### 1.1 Enunciado original (banco maestro, tal cual)

> 49 (Test 49): Actúa como investigador en Topological Data Analysis. Demuestra empíricamente cómo aplicar la asociatividad está físicamente garantizada por el ordenamiento sintáctico.

Este enunciado está gramaticalmente roto: "demuestra... cómo aplicar la asociatividad está..." no tiene una estructura sintáctica válida en español; es el resultado de una combinación automática de plantilla (rol + fragmento técnico) descrita en el propio banco. El fragmento técnico reconocible es: *"la asociatividad está físicamente garantizada por el ordenamiento sintáctico"*, que corresponde directamente a la Sección 9.2 del paper POLYDIM V2 ("Compile-Time Mac Lane Strictification"): la composición $T_3 \circ (T_2 \circ T_1)$ se garantiza de forma estática sobre el grafo de cómputo (`jaxpr`), evitando la inestabilidad de la optimización por gradiente.

### 1.2 Enunciado reconstruido (versión trabajada en esta entrega)

> Como investigador en TDA, diseñar un experimento que mida empíricamente si la composición de una cadena de transformaciones lineales sobre una nube de puntos es *asociativa* en aritmética de punto flotante, y evaluar en qué sentido preciso (si alguno) el hecho de fijar un orden de evaluación sintáctico —como hacen los compiladores de grafos estáticos tipo XLA/`jaxpr`— "garantiza" esa asociatividad. Usar herramientas de Análisis Topológico de Datos (homología persistente, distancia de Wasserstein entre diagramas de persistencia) como métrica de "cuánto diverge" el resultado geométrico bajo distintos reagrupamientos de la composición.

### 1.3 Nota sobre la premisa del enunciado (paso 4 de la metodología)

El enunciado, tal como aparece en el banco, **da por sentada una conclusión no demostrada**: que "la asociatividad está físicamente garantizada por el ordenamiento sintáctico". Esto es impreciso y hay que separarlo con cuidado:

- La multiplicación de matrices **es** asociativa en aritmética exacta (es un teorema de álgebra lineal: $(AB)C = A(BC)$ para toda terna de matrices compatibles).
- En aritmética de punto flotante (IEEE 754), **no** lo es en general: $(A \cdot B) \cdot C \neq A \cdot (B \cdot C)$ como números de máquina, porque el redondeo en cada multiplicación depende del orden de las operaciones.
- Lo que un compilador de grafo estático (`jaxpr`, XLA) realmente garantiza **no es asociatividad matemática** — no hace que dos reagrupamientos distintos coincidan — sino **determinismo/reproducibilidad**: fija *un* orden de evaluación, y ese mismo orden se repite en cada corrida, así que la ejecución es reproducible bit a bit, no "correcta" en el sentido de ser independiente del reagrupamiento.

Por lo tanto, esta entrega **no da por probado** que el ordenamiento sintáctico "garantice" asociatividad; en cambio, lo investiga como pregunta abierta y muestra en qué sentido acotado la afirmación es cierta (reproducibilidad) y en qué sentido es falsa (equivalencia entre reagrupamientos).

---

## 2. Modelo matemático

Sea $X \in \mathbb{R}^{N \times d}$ una nube de $N$ puntos en dimensión $d$, y sea una cadena de $n$ operadores lineales $A_1, \dots, A_n \in \mathbb{R}^{d \times d}$, cada uno generado aleatoriamente con número de condición controlado $\kappa(A_i)$.

Definimos tres estrategias de evaluación de la misma composición formal $T = A_n \cdots A_2 A_1$:

1. **Aplicación secuencial (estilo grafo compilado / `jaxpr`):**
$$
x_0 = x, \qquad x_k = A_k \, x_{k-1}, \qquad k = 1, \dots, n
$$
Nunca se forma el producto matricial completo; es la composición *evaluada* de derecha a izquierda, una vez, en un único orden fijo.

2. **Producto pre-computado, asociado a izquierda:**
$$
M_L = (\cdots((A_n A_{n-1}) A_{n-2}) \cdots A_1)
$$

3. **Producto pre-computado, asociado en árbol balanceado:**
$$
M_T = \text{reduce}_{\text{pairwise}}(A_1, \dots, A_n)
$$
(multiplicación por pares en un árbol binario, minimizando la profundidad de acumulación de error).

En aritmética exacta, $M_L = M_T = T$ y $M_L x = M_T x = x_n$ para todo $x$. En punto flotante, se espera:
$$
\| M_L - M_T \|_F = O\!\left(n \cdot \varepsilon_{\text{mach}} \cdot \prod_i \kappa(A_i) \right)
$$
es decir, el error de no-asociatividad crece con la longitud de la cadena y con el mal condicionamiento de los operadores, no con una constante fija.

### 2.1 Métrica topológica (TDA)

Para capturar si estas divergencias numéricas **importan geométricamente** (que es la pregunta relevante para POLYDIM, que reclama preservar "identidad geométrica", `GEO_ID`), en vez de comparar únicamente normas de matrices comparamos las **nubes de puntos transformadas** $X_L = X M_L^\top$, $X_T = X M_T^\top$, $X_{\text{seq}}$ mediante:

- **Homología persistente** (complejo de Vietoris–Rips) de cada nube transformada, calculada con la librería `ripser`.
- **Distancia de Wasserstein** entre los diagramas de persistencia resultantes (vía `persim`), siguiendo el criterio de "identidad persistente" descrito en la Sección 9.1 del paper (si la distancia es $< \epsilon$, se considera que la identidad homotópica `GEO_ID` persiste pese al error numérico).

Esto separa dos preguntas distintas:
- ¿Coinciden los operadores algebraicamente? (norma de Frobenius entre matrices)
- ¿Coincide la *forma* topológica del resultado? (distancia de Wasserstein entre diagramas de persistencia)

Es perfectamente posible que la primera métrica diverja y la segunda se mantenga estable — eso es justamente lo que POLYDIM necesitaría para que su reclamo de robustez tenga sentido, y es lo que este experimento pone a prueba en vez de asumir.

---

## 3. Razonamiento y alternativas descartadas

- **Por qué no usar solo la norma de Frobenius entre matrices:** mediría el error algebraico, pero no dice nada sobre si ese error es *relevante* para la tarea de POLYDIM (preservar estructura semántica/geométrica). De ahí la necesidad de la capa TDA.
- **Por qué Vietoris–Rips y no Čech:** Vietoris–Rips es computacionalmente tratable (no requiere chequear intersecciones de bolas de orden superior) y es el estándar de facto en librerías como `ripser`; para el propósito de comparar establidad bajo perturbaciones numéricas pequeñas, la equivalencia de homotopía entre ambos complejos (lema de Vietoris–Rips) hace que la elección no afecte la conclusión cualitativa.
- **Por qué no asumir el teorema de asociatividad exacta como válido para el caso de punto flotante:** sería precisamente el error de razonamiento que el enunciado original comete implícitamente. Se lo trata como hipótesis a refutar/confirmar empíricamente, no como axioma.
- **Número de condición controlado:** se generan las matrices $A_i$ con $\kappa(A_i)$ prescripto (vía descomposición SVD con valores singulares elegidos), para poder verificar la dependencia predicha por la cota teórica $O(n \cdot \varepsilon_{\text{mach}} \cdot \prod \kappa(A_i))$ en vez de dejarla librada al azar.

---

## 4. Desarrollo y código

Ver `tarea_49.py`. Resumen de lo que hace el script:

1. Genera $n$ matrices aleatorias $d \times d$ con número de condición controlado.
2. Calcula $M_L$ (asociado a izquierda), $M_T$ (árbol balanceado) y la aplicación secuencial, todo en `float32` (para exagerar el error, como en TPU/GPU real) y en `float64` como referencia de alta precisión.
3. Reporta $\|M_L - M_T\|_F$ y su crecimiento en función de $n$ y de $\kappa$.
4. Aplica cada operador a una nube de puntos sintética (esfera ruidosa en $\mathbb{R}^d$) y calcula diagramas de persistencia (H0, H1) con `ripser`.
5. Calcula la distancia de Wasserstein entre los diagramas de las distintas estrategias de asociación, y la compara contra la distancia entre corridas repetidas de la *misma* estrategia (para separar "error de reasociación" de "ruido de punto flotante genérico").

### Cómo correrlo (lo hace el alumno, no esta entrega)

```bash
pip install numpy scipy ripser persim --break-system-packages
python tarea_49.py
```

Salida esperada: una tabla con $\|M_L - M_T\|_F$ para $n \in \{2, 5, 10, 20, 50\}$ y $\kappa \in \{1, 10, 100\}$, y las distancias de Wasserstein correspondientes entre diagramas de persistencia.

---

## 5. Conclusión

- **Lo que queda demostrado (teóricamente, y que el alumno debe verificar corriendo el script):** la composición de operadores lineales en punto flotante no es asociativa en general; el error de reasociación crece con la longitud de la cadena y el número de condición, consistente con la cota estándar de análisis numérico para acumulación de error de redondeo.
- **Lo que el enunciado original afirmaba y que NO queda demostrado como tal:** que fijar el orden sintáctico "garantice físicamente" la asociatividad. Lo que sí se garantiza al fijar un orden (como hace un grafo de cómputo estático tipo `jaxpr`) es **reproducibilidad** — la misma secuencia de operaciones da el mismo resultado en cada corrida — no equivalencia entre distintos reagrupamientos posibles de la misma composición formal.
- **Pregunta abierta que el script deja para que el alumno compruebe empíricamente:** si la distancia de Wasserstein entre diagramas de persistencia se mantiene pequeña incluso cuando $\|M_L - M_T\|_F$ crece, eso sería evidencia de que el error de no-asociatividad, aunque real algebraicamente, no afecta la *forma topológica* de la representación — que es la propiedad que a POLYDIM realmente le interesa preservar (`GEO_ID`). Si en cambio la distancia de Wasserstein también crece con $n$ y $\kappa$, la conclusión sería la opuesta: el error de reasociación sí degrada la identidad geométrica, y "ordenamiento sintáctico fijo" sería una mitigación de reproducibilidad, no una garantía de robustez estructural. **El script no asume ninguna de las dos; el alumno debe correrlo y leer el resultado numérico para saber cuál se cumple en la práctica.**

---

## 6. Instrucciones para el alumno

1. Instalar las dependencias indicadas arriba.
2. Correr `python tarea_49.py`.
3. Observar si $\|M_L - M_T\|_F$ crece con $n$ y $\kappa$ como predice la cota teórica.
4. Observar si la distancia de Wasserstein entre diagramas de persistencia de las distintas estrategias de asociación crece en la misma proporción, se mantiene chica, o es indistinguible del ruido de fondo (comparación contra corridas repetidas idénticas).
5. Ajustar $n$, $d$, $\kappa$ y el nivel de ruido de la nube de puntos si se quiere explorar el régimen en el que la conclusión cambia de signo.
