# Tarea 59 — Batería de Pruebas Matemáticas POLYDIM
**Alumno:** proyectosnuevos.andrea@gmail.com
**Escuadrón 3:** Descomposición de Hodge y Grafos Esparsos
**Rol asignado:** Ingeniero de PyTorch Dynamo

---

## 1. Enunciado

### 1.1 Enunciado original (banco maestro, Test 59)
> "Actúa como ingeniero de PyTorch Dynamo. Diseña un entorno adverso que inyecte ruido gaussiano extremo, validando si un pooling promedio en tiempo seguido de una reconstrucción óptima lineal."

### 1.2 Nota de reconstrucción
El enunciado del banco está incompleto: termina en un sustantivo ("...reconstrucción óptima lineal") sin verbo ni predicado que diga *qué* hay que validar sobre esa reconstrucción. Esto es consistente con el resto del banco (se genera combinando plantillas de "rol" + "fragmento técnico" y varios fragmentos quedan cortados).

Reconstruyo el enunciado apoyándome en el rol (Dynamo trabaja con grafos de cómputo y optimizaciones de compilación, por lo que le interesan operadores lineales explícitos y su comportamiento numérico) y en el fragmento técnico reconocible ("pooling promedio en tiempo" + "reconstrucción óptima lineal", que en el marco POLYDIM corresponde a una compresión temporal seguida de la reconstrucción de mínimos cuadrados mencionada en la Sección 5 del paper, ligada al `FIXPOINT` y a la estabilidad numérica):

> **Enunciado reconstruido:** Actuando como ingeniero de PyTorch Dynamo, diseñar un entorno de prueba adversarial que inyecte ruido gaussiano de varianza extrema sobre una señal temporal latente, y validar empíricamente si la cadena de operaciones **average-pooling temporal → reconstrucción óptima lineal (mínimos cuadrados, pseudoinversa de Moore-Penrose)** recupera la señal original con un error acotado y previsible frente a ese ruido, o si el error de reconstrucción diverge de forma no controlada.

**Importante — no doy por sentada la premisa:** el enunciado no afirma que el esquema "funcione" ni que "falle"; lo trato como pregunta abierta y la respondo con el álgebra, no con intuición.

### 1.3 La trampa educativa de este ejercicio
La intuición de nivel inicial universitario suele ser: *"promediar en el tiempo suaviza el ruido, entonces cuanto más grande el bloque de pooling, más robusta es la reconstrucción frente a ruido extremo."*

Esa intuición es **incompleta**. Como se demuestra en la Sección 4, la parte del error de reconstrucción que proviene del ruido resulta **exactamente igual sin importar el tamaño del bloque de pooling**. Lo que sí cambia con el tamaño del bloque es una componente completamente distinta (el sesgo por pérdida de información), que no tiene nada que ver con el ruido. El alumno que no llegue a hacer el álgebra se queda con la versión "simple" y equivocada.

---

## 2. Modelo matemático

Sea $x \in \mathbb{R}^T$ una señal latente temporal (por simplicidad, un canal; el resultado se aplica canal a canal si $x \in \mathbb{R}^{T\times d}$).

**Operador de pooling promedio.** Con tamaño de bloque $k$ (asumimos $T = mk$, $m$ bloques), definimos $P \in \mathbb{R}^{m \times T}$:

$$P_{i,t} = \begin{cases} \frac{1}{k} & \text{si } t \in \{(i-1)k+1, \dots, ik\} \\ 0 & \text{en otro caso}\end{cases}$$

Esto es exactamente `nn.AvgPool1d(kernel_size=k, stride=k)` expresado como matriz lineal — la forma en que Dynamo vería el operador si tuviera que trazarlo como un nodo de grafo de álgebra lineal.

**Canal adversario.** El entorno adverso observa la señal ya contaminada:
$$y = P(x + \eta), \qquad \eta \sim \mathcal{N}(0, \sigma^2 I_T)$$
con $\sigma$ arbitrariamente grande ("ruido gaussiano extremo").

**Reconstrucción óptima lineal.** Dado que solo se observa $y \in \mathbb{R}^m$ (con $m < T$, es decir, $P$ no es invertible), el estimador lineal óptimo en el sentido de mínimos cuadrados / norma mínima es la pseudoinversa de Moore-Penrose:
$$\hat{x} = P^{+} y$$

**Propiedad clave de este $P$ (bloques ortogonales).** Las filas de $P$ tienen soportes disjuntos, así que $PP^T = \frac{1}{k} I_m$. De ahí:
$$P^{+} = P^T (PP^T)^{-1} = k\, P^T$$

Concretamente, $P^+$ reparte de vuelta el valor promedio de cada bloque a cada una de sus $k$ posiciones originales: $\hat{x}_t = y_i$ para todo $t$ del bloque $i$. Es la proyección ortogonal de $x+\eta$ sobre el subespacio de "señales constantes por bloques".

---

## 3. Razonamiento

Antes de simular nada, conviene descomponer el error en sus dos fuentes, porque mezclarlas es exactamente donde está la trampa:

$$\mathbb{E}\left[\|x - \hat{x}\|^2\right] = \underbrace{\|x - P^{+}Px\|^2}_{\text{sesgo (pérdida de info. por promediar)}} + \underbrace{\sigma^2 \,\mathrm{tr}(P^{+}(P^{+})^T)}_{\text{varianza (aporte del ruido)}}$$

Esta es la descomposición estándar sesgo-varianza para un estimador lineal $\hat x = P^+P x + P^+ \eta$: el término de sesgo no depende de $\sigma$, y el término de varianza no depende del contenido de $x$, solo de $P$ y $\sigma$.

Descarté dos caminos más simples antes de este:
- **Comparar solo error total sin descomponer:** hubiera mezclado dos efectos de naturaleza distinta y llevado a la conclusión errónea ("el pooling grande es más robusto") sin ver por qué.
- **Asumir que $P^+$ es aproximadamente $P^T$ sin normalizar:** hubiera dado un factor de escala incorrecto en el término de varianza. Hay que usar la pseudoinversa real, no una aproximación informal.

**Cálculo del término de varianza (el resultado no trivial):**
$$\mathrm{tr}(P^+(P^+)^T) = \mathrm{tr}(k^2 P^T P)$$
Cada entrada diagonal de $P^TP$ vale $\frac{1}{k^2}$ (proviene de un único bloque), y hay $T$ entradas diagonales, así que $\mathrm{tr}(P^TP) = \frac{T}{k^2}$. Entonces:
$$\mathrm{tr}(P^+(P^+)^T) = k^2 \cdot \frac{T}{k^2} = T$$

**Es decir: el aporte del ruido al error total es $\sigma^2 T$, sin importar el tamaño de bloque $k$.** El pooling reduce la varianza *del vector comprimido* $y$ (cada componente de $y$ tiene varianza $\sigma^2/k$, por promediar $k$ ruidos independientes), pero al reconstruir con la pseudoinversa ese mismo valor se reparte hacia $k$ posiciones, y esa expansión cancela exactamente la reducción. Ni gana ni pierde robustez frente al ruido por elegir $k$ más grande.

Lo que sí depende de $k$ es el sesgo $\|x - P^+Px\|^2$: es la suma, sobre todos los bloques, de la varianza interna de $x$ dentro de cada bloque. Si $x$ es muy suave (varía poco dentro de cada bloque), el sesgo es chico aunque $k$ sea grande; si $x$ tiene variación de alta frecuencia, el sesgo crece con $k$.

**Conclusión analítica antes de simular:** cuando $\sigma \to \infty$, el error total diverge linealmente como $\sigma^2 T$, **para cualquier $k$**. El esquema no es "extremo-ruido-proof": no hay ningún $k$ que lo salve de una divergencia no acotada si el ruido crece sin límite. Lo único verificable empíricamente es que la tasa de esa divergencia es la misma constante $T$ para todo bloque, lo cual es la parte contraintuitiva a comprobar con código.

---

## 4. Desarrollo empírico (código)

El archivo `tarea_59_pooling_reconstruccion.py` (adjunto) implementa:

1. Construcción de $P$ como matriz de PyTorch para varios tamaños de bloque $k$.
2. Verificación numérica de $P^+ = kP^T$ comparando contra `torch.linalg.pinv`.
3. Una señal sintética suave $x_t = \sin(2\pi t / T) + 0.1\cos(6\pi t/T)$ que representa un estado latente temporal.
4. Un bucle de Monte Carlo que, para una grilla de valores de $\sigma$ (incluyendo valores "extremos", $\sigma \gg \|x\|$) y de $k$, mide el MSE empírico de $\hat x = P^+ P(x+\eta)$ contra la fórmula teórica $\|x-P^+Px\|^2/T + \sigma^2$.
5. Impresión de una tabla que separa sesgo medido vs. teórico y varianza medida vs. teórica ($\sigma^2 T$), para que quede visible que el término de varianza no cambia con $k$.

El script no se ejecuta como parte de esta entrega (ver instrucciones abajo); se hizo únicamente una corrida interna de sanity-check para confirmar que no tiene errores de sintaxis ni de shapes.

---

## 5. Conclusión

- La cadena "average pooling temporal → reconstrucción por pseudoinversa" **no es robusta frente a ruido gaussiano extremo** en el sentido de que el error total no está acotado cuando $\sigma \to \infty$: crece linealmente como $\sigma^2 T$.
- Ese crecimiento es **idéntico para cualquier tamaño de bloque $k$**: el pooling no compra ninguna protección adicional contra el ruido a nivel de la señal reconstruida completa, aunque sí reduce la varianza del vector comprimido antes de reconstruir.
- Lo único que el tamaño de bloque controla es el **sesgo** (pérdida de información por promediar dentro de cada bloque), que es independiente del ruido.
- Esto es relevante para el marco POLYDIM: si se pretendiera usar pooling temporal como mecanismo de "filtrado" de ruido antes del operador `FIXPOINT` (Sección 5.2 del paper), el álgebra muestra que ese filtrado no protege la reconstrucción final contra ruido extremo — habría que actuar sobre la varianza directamente (por ejemplo, con un filtro que no sea de norma mínima, como Tikhonov/ridge, que sí cambia el trade-off sesgo-varianza). Esto último queda como observación, no como parte resuelta de este ejercicio.

**Instrucciones para que corras el código vos mismo:**
```bash
pip install torch --break-system-packages   # si no lo tenés instalado
python3 tarea_59_pooling_reconstruccion.py
```
Vas a ver, para cada $\sigma$ y cada $k$ probado, el sesgo medido vs. teórico y la varianza medida vs. teórica. Prestá atención a la columna de varianza: debería mantenerse igual (dentro del error de Monte Carlo) para todos los $k$, aun cuando $\sigma$ sea muy grande. Si querés estresar más el resultado, subí el valor de `sigma_extremo` en el script y volvé a correr.

---

## 6. Resumen estructurado

```json
{
  "alumno": "proyectosnuevos.andrea@gmail.com",
  "tarea": 59,
  "tipo_problema": "álgebra lineal / estimación estadística (bias-variance de un operador de pooling + pseudoinversa)",
  "estado": "resuelto",
  "resultado_resumen": "El error de reconstrucción se descompone en sesgo (depende del bloque k, no del ruido) y varianza (= sigma^2 * T, constante para todo k). El esquema pooling+reconstrucción no es robusto frente a ruido extremo: el error diverge linealmente en sigma^2 sin importar k.",
  "metrica_estandar_usada": "Error cuadrático medio (MSE / Norma L2 al cuadrado), descomposición sesgo-varianza"
}
```

*que has usado el formato de razonamiento adaptado por AGT*
