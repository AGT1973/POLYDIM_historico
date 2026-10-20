# Tarea 52

## 1. Enunciado original (banco maestro)

> "52 (Test 52): Actúa como científico de datos. Utiliza la librería POT o Sinkhorn
> para alinear estadísticamente la superposición garantizando conservación del
> volumen determinante."

**Nota de reconstrucción.** El enunciado, tal como está en el banco, es
gramaticalmente incompleto: "alinear estadísticamente la superposición
garantizando..." no tiene un objeto claro sobre qué se alinea, y mezcla dos
fragmentos de plantillas distintas (uno sobre transporte óptimo/Sinkhorn, otro
sobre "superposición que garantiza conservación del volumen determinante",
fragmento que en el banco también aparece asociado a otras tareas sobre
rotores). Reconstruyo el enunciado combinando el rol indicado (científico de
datos) con el núcleo técnico reconocible: transporte óptimo regularizado para
alinear estadísticamente dos distribuciones, más una superposición rígida que
conserve el volumen (es decir, sin reflexión ni escalado).

**Enunciado reconstruido:**

> Dadas dos nubes de puntos (muestras de dos distribuciones empíricas) en R^d,
> relacionadas por una transformación rígida desconocida, hallar la
> correspondencia estadística óptima entre ambas mediante transporte óptimo
> regularizado (algoritmo de Sinkhorn, librería POT), y a partir de esa
> correspondencia calcular la transformación de superposición (rotación +
> traslación) que mejor alinea una nube sobre la otra, garantizando que dicha
> transformación **conserve el volumen** — esto es, que sea una rotación
> propia con determinante igual a +1 (sin reflexiones).

## 2. Modelo matemático

Sean $X = \{x_i\}_{i=1}^n \subset \mathbb{R}^d$ y $Y = \{y_j\}_{j=1}^m \subset \mathbb{R}^d$,
con distribuciones empíricas uniformes $\mu = \frac{1}{n}\sum_i \delta_{x_i}$,
$\nu = \frac{1}{m}\sum_j \delta_{y_j}$.

**Paso A — Transporte óptimo entrópico (Sinkhorn, Cuturi 2013):**

$$
P^\* = \arg\min_{P \in U(\mu,\nu)} \langle P, C \rangle - \varepsilon H(P)
$$

donde $C_{ij} = \|x_i - y_j\|^2$, $U(\mu,\nu)$ es el politopo de transporte
(marginales fijas $\mu$, $\nu$), $H(P) = -\sum_{ij} P_{ij}\log P_{ij}$ es la
entropía y $\varepsilon>0$ el parámetro de regularización. Se resuelve por
iteración de Sinkhorn-Knopp (escalado alternado de filas/columnas).

**Paso B — Correspondencia baricéntrica:** a partir de $P^\*$, cada $x_i$ se
asocia a un punto sintético $\hat{y}_i = \frac{1}{\sum_j P^\*_{ij}} \sum_j P^\*_{ij}\, y_j$.

**Paso C — Procrustes ortogonal restringido a rotaciones propias (Kabsch 1976,
Umeyama 1991):**

$$
(R^\*, t^\*) = \arg\min_{R \in SO(d),\, t \in \mathbb{R}^d} \sum_i \| R x_i + t - \hat{y}_i \|^2
$$

con $SO(d) = \{R : R^TR = I,\; \det(R) = +1\}$. Restringir a $SO(d)$ (y no al
grupo ortogonal completo $O(d)$, que incluye reflexiones con $\det=-1$) es
justamente lo que garantiza la **conservación del volumen**: toda matriz
ortogonal preserva normas y ángulos, pero solo las de determinante $+1$
preservan también la orientación; el volumen con signo de cualquier región se
multiplica por $\det(R)$, así que fijar $\det(R)=+1$ asegura conservación
exacta del volumen (sin reflexión).

**Solución cerrada:** centrando $X$, $\hat{Y}$ y definiendo
$H = X_c^T \hat{Y}_c$, con SVD $H = U\Sigma V^T$:

$$
R^\* = V \, \mathrm{diag}(1,\dots,1,\det(VU^T)) \, U^T
$$

Esta corrección de signo en la última componente es exactamente lo que impide
que la solución colapse a una reflexión cuando el ajuste "naive" ($R=VU^T$)
tendría $\det=-1$.

**Métrica de verificación:** $\det(R^\*)$ (debe dar $+1$ dentro de tolerancia
numérica) y el error RMS de alineación $\frac{1}{\sqrt n}\|XR^{*T}+t^\* - \hat Y\|_F$.

## 3. Razonamiento

- Se descartó modelar la alineación como un simple ajuste de mínimos
  cuadrados directo (sin transporte óptimo) porque el enunciado pide
  explícitamente "alinear estadísticamente" dos conjuntos, lo cual sugiere que
  no hay correspondencia punto-a-punto conocida de antemano; Sinkhorn resuelve
  precisamente ese problema de correspondencia entre distribuciones.
- Se descartó usar el grupo ortogonal completo $O(d)$ en el Procrustes: eso
  permitiría reflexiones, que invierten el signo del volumen. Como el
  enunciado exige "conservación del volumen determinante", la restricción a
  $SO(d)$ (det $=+1$) es la que corresponde matemáticamente, no una elección
  arbitraria.
- Se usó la formulación entrópica (Sinkhorn) en vez del transporte óptimo
  exacto (programación lineal) porque es diferenciable, numéricamente estable
  y es la variante que la consigna nombra explícitamente ("Sinkhorn").

## 4. Desarrollo y código

Ver `tarea_52.py`. Contiene:
- Generación sintética de dos nubes relacionadas por una rotación propia
  conocida (para poder verificar si el método la recupera).
- Cálculo del plan de transporte con `ot.sinkhorn` (librería POT).
- Correspondencia baricéntrica.
- Procrustes con corrección de signo para garantizar $\det(R)=+1$.
- Cálculo de métricas de verificación.

El código fue revisado con un chequeo de sintaxis interno (compila sin
errores), pero **no fue ejecutado como parte de esta entrega**: la
comprobación empírica (correrlo y ver si `det(R) ~ 1` y el error de
alineación es bajo) la hace el alumno.

### Instrucciones para correrlo

```bash
pip install numpy scipy pot --break-system-packages
python3 tarea_52.py
```

Salida esperada: valores de `det(R)` cercano a `1.0`, el error de alineación
(RMS) y la distancia entre la rotación recuperada y la rotación verdadera
usada para generar los datos sintéticos.

## 5. Conclusión

El modelo propuesto combina dos herramientas estándar y bien establecidas en
la literatura (transporte óptimo entrópico vía Sinkhorn, y el problema de
Procrustes ortogonal restringido a rotaciones propias vía Kabsch/Umeyama) para
resolver el enunciado reconstruido: alinear estadísticamente dos nubes de
puntos con una transformación que garantiza conservación del volumen. La
verificación empírica queda a cargo del alumno corriendo el script: si
`det(R)` da aproximadamente `1.0` y el error de alineación es bajo, el punto
del enunciado queda demostrado; si `det(R)` se aleja de `1.0` o el error es
alto, hay que revisar el `epsilon` de regularización del Sinkhorn o el nivel
de ruido de los datos.
