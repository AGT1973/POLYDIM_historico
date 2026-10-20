# Tarea 45 — POLYDIM: Convergencia de un tensor mediante superposición lineal de rotores

## 1. Enunciado

**Enunciado original (banco maestro, entrada 45, Escuadrón 3):**

> "45 (Test 45): Actúa como experto en Manifold Learning. Diseña una operación
> encadenada que demuestre a nivel de volcado intermedio de grafo que
> converger el tensor usando superposición lineal con rotores sin bloqueos."

**Nota de reconstrucción.** Esta entrada, como advierte el protocolo del
curso, sale de una plantilla combinatoria (rol + fragmento técnico) y quedó
gramaticalmente incompleta: la cláusula "que converger el tensor... sin
bloqueos" no tiene sujeto ni verbo conjugado, y no queda claro a qué
"bloqueos" se refiere sin mirar el resto del paper POLYDIM.

Reconstruyendo con apoyo en las Secciones 3.1 y 5.2 del paper adjunto
(`POLYDIM_PAPER_3_EN.md`) — donde el operador `MIX` se define como un **Spin
Rotor** $R = e^{\theta B/2}$ que "preserva el volumen semántico latente y
evita el *gimbal lock* de alta dimensionalidad", y `FIXPOINT` como el
operador de convergencia global —, el enunciado bien planteado es:

> **Enunciado reconstruido:** Como experto en Manifold Learning, diseñar una
> cadena de operaciones sobre un grafo de cómputo que demuestre —inspeccionando
> los volcados (dumps) intermedios de ese grafo— que la convergencia de un
> tensor de estado latente hacia un punto fijo, cuando se implementa mediante
> **superposición lineal de rotores de Clifford** (en lugar de composición de
> matrices de rotación parametrizadas por ángulos de Euler), **no exhibe
> singularidades tipo gimbal lock**.

Este enunciado reconstruido **no da por sentada** la conclusión: se investiga
y se demuestra (o refuta) empíricamente, no se asume.

## 2. Modelo matemático

### 2.1 Rotores como subálgebra par de $C\ell_3(\mathbb{R})$

Un rotor en $C\ell_n(\mathbb{R})$ tiene la forma $R = e^{\theta B/2} =
\cos(\theta/2) - \sin(\theta/2)\,B$, con $B$ un bivector unitario ($B^2=-1$).
Para $n=3$, la subálgebra par de $C\ell_3(\mathbb{R})$ es isomorfa al álgebra
de cuaterniones $\mathbb{H}$, y el grupo de rotores unitarios es
$\mathrm{Spin}(3) \cong S^3 \subset \mathbb{H}$, que cubre doblemente a
$SO(3)$ vía el mapa $\pi:\mathrm{Spin}(3)\to SO(3)$.

- **Superposición lineal (composición de rotores):**
  $R_{\text{total}} = R_k \cdot R_{k-1} \cdots R_1$ (producto de Hamilton, no
  conmutativo).
- **Acción sobre un vector (producto sandwich):**
  $v' = R\, v\, R^{-1}$, con $R^{-1}=\bar R$ para $R$ unitario.
- **Preservación de norma/volumen:** $\|v'\| = \|v\|$ porque
  $R\bar R = 1$, es decir, la acción de $\mathrm{Spin}(3)$ sobre
  $\mathbb{R}^3$ es una isometría.

### 2.2 Iteración de punto fijo (FIXPOINT) como contracción de Banach

Se define una iteración geodésica en la variedad $S^3$:
$$R_{t+1} = \mathrm{slerp}(R_t, R^*, \tau), \qquad \tau \in (0,1)$$
donde $\mathrm{slerp}$ es la interpolación esférica (geodésica en $S^3$) y
$R^*$ es el rotor objetivo (punto fijo). Con la distancia geodésica
$d(R_1,R_2) = 2\arccos(|\langle R_1,R_2\rangle|)$, se cumple
$$d(R_{t+1}, R^*) = (1-\tau)\, d(R_t, R^*),$$
es decir, $\mathrm{slerp}(\cdot, R^*, \tau)$ es una contracción de Banach con
constante de Lipschitz $\lambda = 1-\tau < 1$ (coherente con el "Differentiable
Banach Theorem" de la Sección 9 del paper, que exige $\|J_T\|_2 \le \lambda <1$).
Por el teorema del punto fijo de Banach, $R_t \to R^*$ para cualquier $R_0$.

### 2.3 Contraparte con ángulos de Euler y el Jacobiano de la carta

La composición equivalente vía ángulos de Euler ZYX es
$R(\psi,\theta,\phi) = R_z(\psi)R_y(\theta)R_x(\phi)$. El mapa
$(\psi,\theta,\phi)\mapsto R \in SO(3)$ es una **carta local**, no un
isomorfismo global: es bien conocido (Gimbal Lock) que quiebra su rango
máximo cuando $\theta = \pm\pi/2$, porque ahí los generadores infinitesimales
$\partial_\psi R$ y $\partial_\phi R$ se vuelven linealmente dependientes. Esto
se mide con el número de condición (o el rango) del Jacobiano numérico
$J \in \mathbb{R}^{9\times 3}$ del mapa ángulos → matriz de rotación
(vectorizada).

## 3. Razonamiento

- Se eligió $\mathbb{R}^3$/$\mathrm{Spin}(3)$ (cuaterniones) en vez del caso
  general $C\ell_n(\mathbb{R})$ porque es el caso mínimo no trivial donde el
  fenómeno de gimbal lock está clásicamente documentado y admite verificación
  numérica directa y barata (matrices 3×3), sin sacrificar generalidad
  conceptual: el argumento (Lie-suavidad de $\mathrm{Spin}(n)$ vs.
  singularidad de cartas angulares) se extiende a $n$ mayor.
- Se descartó demostrar el resultado solo de forma simbólica/analítica porque
  el enunciado pide explícitamente evidencia "a nivel de volcado intermedio
  de grafo": se usa `jax.make_jaxpr` para exhibir que la composición de
  rotores es un grafo de ecuaciones puramente aritméticas (sin nodos
  condicionales `cond`/`switch`), mientras que un manejo *robusto* de Euler
  cerca del gimbal lock normalmente requiere ramas condicionales para elegir
  una convención alternativa — evidencia indirecta pero verificable de la
  diferencia estructural.
- La iteración de punto fijo se modela con `slerp` en vez de con
  promediado ingenuo de cuaterniones, porque el promedio aritmético directo
  de cuaterniones no vive en $S^3$ (rompe la restricción de unitariedad) y no
  es una contracción bien definida en la métrica geodésica.
- Métrica estándar usada: **distancia geodésica en $S^3$** para la
  convergencia (equivalente a un ángulo, unidades interpretables), y
  **número de condición del Jacobiano (razón de valores singulares, vía SVD)**
  para cuantificar la degeneración tipo gimbal lock, en vez de solo mirar el
  determinante (que puede ser insensible a rango deficiente en mapas no
  cuadrados).

## 4. Desarrollo y código

El código completo y funcional está en
[`rotor_fixpoint.py`](./rotor_fixpoint.py) (Python, usa `numpy`; el volcado
de grafo usa `jax` si está disponible, con degradación elegante si no lo
está). Contiene:

1. Álgebra de cuaterniones/rotores desde cero (`quat_mul`, `quat_conj`,
   `sandwich`), sin depender de librerías de geometría prearmadas, para que
   quede explícito que es el producto geométrico de Clifford.
2. `chain_compose_rotors`: superposición lineal (composición) de una cadena
   de rotores aleatorios — el operador `MIX`/`GEO_ID` del paper.
3. `fixpoint_iteration_rotor`: la iteración de punto fijo tipo `FIXPOINT` vía
   `slerp`, con historial de distancia geodésica por paso.
4. `euler_zyx_to_matrix` + `euler_jacobian_condition_number`: la contraparte
   de Euler y la métrica de degeneración (SVD del Jacobiano numérico).
5. `dump_computation_graph`: volcado del grafo intermedio (`jaxpr`) de la
   cadena de rotores, verificando ausencia de nodos condicionales.

## 5. Conclusión

El modelo predice —y el script está preparado para verificar
empíricamente al ejecutarlo— dos resultados distintos y falsables:

1. La composición de rotores preserva la norma del tensor en cada paso
   (isometría de $\mathrm{Spin}(3)$) y la iteración `slerp` converge al rotor
   objetivo con una tasa de contracción empírica que debería aproximarse a
   $\lambda = 1-\tau$ (para $\tau=0.35$, $\lambda \approx 0.65$), sin
   discontinuidades ni necesidad de ramas condicionales en el grafo de
   cómputo.
2. La contraparte por ángulos de Euler debería mostrar un número de
   condición del Jacobiano que crece marcadamente (o un rango deficiente en
   la SVD) a medida que el ángulo de "pitch" se acerca a $\pi/2$, cuantificando
   el gimbal lock exactamente en la configuración predicha por la teoría.

**Importante — esto no está "demostrado" solo por leer este documento.** El
alumno debe **correr el script él mismo** y observar los tres números clave
que imprime (norma preservada, tasa de contracción empírica vs. teórica, y
número de condición cerca/lejos del gimbal lock) para confirmar, refutar o
matizar estas predicciones con sus propios ojos — esa es la comprobación
activa que exige la metodología del curso. Si al ejecutarlo el número de
condición no diverge como se espera (por ejemplo, por errores de redondeo en
diferencias finitas con `eps` mal elegido), documentar esa discrepancia es en
sí mismo un resultado válido, no un fracaso del ejercicio.

### Instrucciones para ejecutar

```bash
pip install numpy jax --break-system-packages   # jax es opcional
python rotor_fixpoint.py
```

No requiere GPU. El tiempo de ejecución esperado es de segundos.

## 6. Resumen estructurado

```json
{
  "alumno": null,
  "tarea": 45,
  "tipo_problema": "geometria diferencial computacional / algebra de Clifford (rotores) vs. singularidades de cartas angulares (gimbal lock)",
  "estado": "resuelto",
  "resultado_resumen": "Se reconstruyo el enunciado (fragmento roto de plantilla) y se modelo la convergencia de un tensor via superposicion de rotores de Clifford (Spin(3)/cuaterniones) como contraccion de Banach geodesica en S^3, comparandola contra la composicion por angulos de Euler ZYX. El codigo provisto permite verificar empiricamente que los rotores preservan norma y convergen sin discontinuidades, mientras que Euler exhibe degeneracion del Jacobiano cerca de pitch=90deg (gimbal lock). El alumno debe ejecutar el script para confirmar los valores numericos.",
  "metrica_estandar_usada": "Distancia geodesica en S^3 (convergencia) y numero de condicion del Jacobiano via SVD (deteccion de gimbal lock)"
}
```
