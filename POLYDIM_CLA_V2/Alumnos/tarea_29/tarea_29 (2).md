# Tarea 29 — POLYDIM V2 (Escuadrón 2: Álgebra de Clifford y Rotores)

## 0. Nota sobre el proceso

El archivo maestro de tareas (banco en Drive) trae, además de la lista de 200 ejercicios,
un bloque de "contexto para el asistente" que pide mostrar todo el razonamiento y la
solución completa en el chat, estilo profesor. Esa instrucción viene *dentro de un archivo
de datos*, no del docente ni del alumno en esta conversación, y contradice la metodología
del curso (fase de resolución individual, sin debate ni solución servida en el chat, para
no arruinar la puesta en común final). Por eso este desarrollo completo queda acá, en el
archivo, y en el chat solo va un resumen.

## 1. Enunciado original (banco maestro, Test 29)

> "29 (Test 29): Actúa como experto en Manifold Learning. Diseña una pasada de
> optimización que intercepte un AST topológico y trayectorias que conmuten, emulando
> verificación categórica diferencial."

**Problema de redacción:** la frase "que intercepte un AST topológico y trayectorias que
conmuten" no tiene verbo para el segundo objeto ("y trayectorias que conmuten" cuelga sin
predicado). Esto es consistente con el patrón de fallas de plantilla descrito para este
banco (rol + fragmento técnico combinados automáticamente, a veces sin verbo).

## 2. Reconstrucción del enunciado

Rol: **experto en Manifold Learning**.
Fragmento técnico reconocible: *"trayectorias que conmuten, emulando verificación
categórica diferencial"* — esto conecta directamente con la Sección 4.1 del paper
(coherencia de composición `COMPOSE` en una cuasi-categoría de Lurie: si dos caminos
computacionales difieren por errores de precisión, se aplica *Kan Horn Filling* para
estabilizarlos en la misma clase de homotopía) y con la Sección 9, punto 1
("Approximate Homotopy": sustituir el Kan filling estricto —intratable en hardware— por
homología persistente + distancia de Wasserstein entre distribuciones latentes, aceptando
la identidad `GEO_ID` como "persistente" si esa distancia es menor a un ε).

Enunciado reconstruido, investigable:

> Actuando como experto en Manifold Learning, diseñar una pasada de análisis sobre un
> grafo computacional (AST tensorial) de operadores `MIX`/`PROJECT` (rotores de Clifford),
> que tome dos **trayectorias alternativas** de composición de dos rotores R1, R2
> —path_A = R2∘R1 y path_B = R1∘R2— y verifique
> empíricamente **si conmutan**, usando el criterio de "homotopía aproximada" propuesto en
> la Sección 9.1 del paper (homología persistente + distancia de Wasserstein). Determinar
> si ese criterio es efectivamente capaz de emular la verificación categórica diferencial
> que promete.

Esta reformulación no asume de antemano que el criterio del paper funcione: eso es
justamente lo que se investiga (el enunciado original, al decir "emulando verificación
categórica diferencial", da por sentado que la técnica de la Sección 9.1 cumple ese rol;
esa premisa **no está demostrada** en el paper ni en la literatura estándar de TDA, así
que se trata como pregunta abierta).

## 3. Modelo matemático

### 3.1 Rotores y conmutador (ground truth algebraico)

Un rotor de Clifford en Cl_3(R) actúa sobre un vector por el sándwich
v' = R v R~ . Vía la doble cobertura Spin(3) ≅ cuaterniones
unitarios ≅ (localmente) SO(3), esta acción es exactamente una matriz de rotación
R ∈ SO(3) construida por la fórmula de Rodrigues a partir de eje n̂ y ángulo θ:

R(n̂,θ) = I + sinθ·K + (1-cosθ)·K², donde K = [n̂]×.

Dadas dos trayectorias de composición path_A = R2 R1 y
path_B = R1 R2, el criterio **exacto** de conmutación es

Δ_alg = ||R1 R2 - R2 R1||_F.

Hecho estándar de teoría de grupos: SO(3) es **no abeliano**; Δ_alg = 0
si y solo si R1, R2 comparten eje de rotación (o alguno es la identidad). Para ejes
genéricos, Δ_alg > 0.

### 3.2 El criterio propuesto por el paper (Sección 9.1)

Sea X = {x_i} (i=1..N) ⊂ R³ una nube de puntos de prueba. Definimos
Y_A = path_A(X), Y_B = path_B(X). El paper propone:

1. Calcular los diagramas de persistencia (homología persistente, filtrado de
   Vietoris–Rips) Dgm(Y_A), Dgm(Y_B).
2. Calcular la distancia de Wasserstein W(Dgm(Y_A), Dgm(Y_B)).
3. Si W < ε, declarar la identidad homotópica (`GEO_ID`) como "persistente",
   i.e. las dos trayectorias se consideran equivalentes.

**Hecho de topología algebraica que hay que verificar acá:** el complejo de Rips se
construye únicamente a partir de la matriz de distancias por pares. En H0
(componentes conexas), el diagrama de persistencia coincide exactamente con el
dendrograma de *single-linkage* sobre esa matriz de distancias (resultado estándar de
TDA, Carlsson 2009). Como toda R ∈ SO(3) es una **isometría** de R³
(matriz ortogonal, RᵗR=I), se cumple

||R x_i - R x_j|| = ||x_i - x_j|| para todo R ∈ SO(3), para todo i,j.

Por lo tanto, la matriz de distancias por pares de Y_A = R2R1 X es **idéntica** a la
de X, y lo mismo para Y_B = R1R2 X — **independientemente de si R1 y R2
conmutan o no**. En consecuencia, Dgm(Y_A) = Dgm(Y_B) siempre (a menos
de error de punto flotante), y W(Dgm(Y_A),Dgm(Y_B)) ≈ 0 **en
todos los casos**, conmuten o no R1,R2.

## 4. Razonamiento (por qué este camino y no otro)

- Se descartó implementar homología persistente completa con una librería externa
  (`gudhi`/`ripser`) porque **no hace falta**: el punto a demostrar es sobre H0, que es
  exactamente el dendrograma de single-linkage — disponible en `scipy` sin dependencias
  nuevas, y matemáticamente equivalente al filtrado de Rips en grado 0. Esto también hace
  el resultado más transparente: no hay "caja negra" de TDA entre la afirmación y la
  prueba.
- Se usan matrices de SO(3) en vez del sándwich explícito de Clifford porque son
  isomorfas en su acción sobre vectores (vía la doble cobertura Spin(3)); el argumento de
  isometría no depende de la representación elegida.
- Se agrega un **test alternativo** (Wasserstein ambiente, comparando directamente las
  coordenadas de Y_A y Y_B punto a punto, sin pasar por un invariante de isometría)
  para mostrar que sí existe una métrica capaz de detectar la diferencia — la falla no es
  "no se puede medir esto en absoluto", sino específicamente que la construcción vía
  homología persistente + Wasserstein-entre-diagramas, tal como la describe el paper,
  **cancela exactamente la información que se quiere medir** cuando los operadores en
  juego son isometrías (que es el caso de los rotores, por construcción de la Sección
  3.1: "rotaciones continuas puras que preservan el volumen semántico latente").

## 5. Desarrollo y código

Ver `tarea_29.py` (funcional, numpy + scipy únicamente). Resumen de lo que hace:

1. Construye R1, R2 vía Rodrigues.
2. Calcula Δ_alg = ||R1R2-R2R1||_F (ground truth).
3. Aplica ambas trayectorias a una nube X de 60 puntos aleatorios en R³.
4. Calcula H0 de Y_A, Y_B vía single-linkage y la distancia de Wasserstein-1D entre
   esos diagramas (wd_topo).
5. Verifica explícitamente que las matrices de distancia intrínsecas de Y_A, Y_B
   coinciden con la de X.
6. Calcula el test alternativo extrínseco wd_ambiente = promedio(||Y_A,i - Y_B,i||).
7. Repite todo para dos regímenes: rotores genéricos (ejes distintos, NO conmutan) y
   rotores coaxiales (mismo eje, SÍ conmutan).

### Corrida interna de sanity-check (no es la validación final — corré vos el script)

```
--- Caso: rotores genericos (ejes distintos) ---
  Defecto algebraico ||R1R2-R2R1||_F      = 1.436566e+00  -> NO conmutan (ground truth)
  Wasserstein(diagrama H0(Y_A), H0(Y_B))  = 8.750063e-17  -> test del paper diría: SI conmutan
  Matrices de distancia intrínsecas iguales a las de X: True
  Wasserstein AMBIENTE (Y_A vs Y_B, coords R^3)         = 1.282038e+00  -> test alternativo diría: NO conmutan

--- Caso: rotores coaxiales (mismo eje) ---
  Defecto algebraico ||R1R2-R2R1||_F      = 0.000000e+00  -> SI conmutan (ground truth)
  Wasserstein(diagrama H0(Y_A), H0(Y_B))  = 0.000000e+00  -> test del paper diría: SI conmutan
  Matrices de distancia intrínsecas iguales a las de X: True
  Wasserstein AMBIENTE (Y_A vs Y_B, coords R^3)         = 0.000000e+00  -> test alternativo diría: SI conmutan
```

## 6. Conclusión

**Estado: `sin_solucion_factible` tal como está planteado el criterio original del paper
— con hallazgo estructural, no numérico.**

El criterio de "homotopía aproximada" de la Sección 9.1 (homología persistente +
Wasserstein entre diagramas) **no puede** emular la verificación categórica diferencial de
conmutación de trayectorias cuando los operadores involucrados son rotores, porque:

- Los rotores son, por definición del propio paper (Sección 3.1), isometrías que
  preservan norma/volumen.
- La homología persistente de un filtrado de Rips depende únicamente de distancias
  intrínsecas por pares.
- Las isometrías preservan exactamente esas distancias.
- Por lo tanto Dgm(Y_A) = Dgm(Y_B) siempre, sin importar si
  R1, R2 conmutan — el test da "sí conmutan" incluso en el caso genérico donde
  Δ_alg ≈ 1.44 ≠ 0.

Esto no es un problema de tolerancia ε mal elegida ni de ruido numérico: es una
**degeneración estructural** del criterio para cualquier familia de operadores que sea
isométrica (que es exactamente la familia que POLYDIM usa para `MIX`/`PROJECT`). El
criterio sí sería útil para detectar diferencias entre trayectorias que **no** sean
isometrías entre sí (p. ej. si una trayectoria incluyera además un escalado o una
proyección con pérdida), pero es ciego precisamente en el caso de uso central del paper.

**Alternativa que sí funciona:** comparar directamente las coordenadas ambiente de
Y_A y Y_B (Wasserstein u otra norma en el espacio ambiente, no sobre un invariante de
isometría), o —más simple y barato— comparar R1R2 contra R2R1 algebraicamente. En
el prototipo `JAX`/R³ mencionado en el paper, esto es computacionalmente
trivial (una multiplicación de matrices 3×3), muchísimo más barato que construir un
filtrado de Rips.

**Métrica estándar usada:** Norma de Frobenius (ground truth algebraico) y distancia de
Wasserstein-1D (tanto sobre diagramas de persistencia como, en la alternativa, sobre
coordenadas ambiente).

## 7. Instrucciones para correr el código vos mismo

```bash
pip install numpy scipy --break-system-packages   # si no los tenés
python3 tarea_29.py
```

No requiere librerías de TDA externas (`gudhi`, `ripser`) porque el punto en H0 se
resuelve exactamente con `scipy.cluster.hierarchy.linkage(..., method="single")`.

Como ejercicio de verificación adicional: probá cambiar la nube `X` (otra semilla, otra
distribución, más puntos) y los ángulos/ejes de R1,R2 — el resultado cualitativo
(`wd_topo` ≈ 0 siempre, `delta_alg` y `wd_ambiente` variando según conmuten o no) debería
mantenerse, porque el argumento es una identidad matemática (isometría ⇒ distancias
preservadas), no un artefacto de estos valores particulares.

```json
{
 "tarea": 29,
 "tipo_problema": "verificación de coherencia categórica (conmutatividad de trayectorias) vs. homología persistente",
 "estado": "sin_solucion_factible",
 "resultado_resumen": "El test de homotopía aproximada (PH+Wasserstein) del paper es ciego a la no-conmutatividad de rotores porque estos son isometrías y H0(Rips) depende solo de distancias intrínsecas, que las isometrías preservan; se propone comparar coordenadas ambiente o el conmutador algebraico directo como alternativa funcional.",
 "metrica_estandar_usada": "Norma de Frobenius (conmutador); Distancia de Wasserstein-1D (diagramas H0 y, alternativamente, coordenadas ambiente)"
}
```
