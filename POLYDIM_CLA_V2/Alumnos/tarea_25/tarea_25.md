# Tarea 25 — POLYDIM V2 (Curso de IA, UTN)

**Escuadrón de origen en el banco:** Escuadrón 8 — Métricas de Validación (SPS y GeoID Stability)
**Rol asignado:** Experto en PEFT

## 1. Enunciado

**Enunciado original (banco maestro, Test 25):**
> "Actúa como experto en PEFT. Escribe un framework de benchmarking que monitoree el mayor valor singular para asegurar convergencia topológica en FIXPOINT."

**Nota sobre el enunciado:** a diferencia de otros ítems del banco (que sí quedan rotos por la combinación de plantillas), este enunciado está gramaticalmente completo y es investigable tal cual está — no fue necesario reconstruirlo. Solo se explicita el contexto matemático que da sentido a "el mayor valor singular" y "convergencia topológica en FIXPOINT", apoyándose en la Sección 9.4 del paper de referencia del proyecto POLYDIM (POPL 2027 Amendment, *Differentiable Banach Theorem*):

> "Se implementa una **Spectral Lipschitz Penalty** en el loop `FIXPOINT`, forzando el mayor valor singular del Jacobiano a $\|J_T(x)\|_2 \le \lambda < 1$. Esto garantiza convergencia global a una raíz topológica sin oscilaciones de red."

## 2. Modelo Matemático

Sea $T:\mathbb{R}^n\to\mathbb{R}^n$ el operador `FIXPOINT` (el bloque que se itera para alcanzar un estado latente estable, en la jerga del paper). Buscamos garantizar que la iteración

$$x_{k+1} = T(x_k)$$

converge a un único punto fijo $x^\*$ tal que $T(x^\*) = x^\*$.

### 2.1 Teorema del Punto Fijo de Banach (resultado estándar, Banach 1922)

Si $(X,\|\cdot\|)$ es un espacio de Banach (completo) y $T:X\to X$ es una **contracción**, es decir, existe $\lambda\in[0,1)$ tal que

$$\|T(x)-T(y)\| \le \lambda \|x-y\| \quad \forall x,y\in X,$$

entonces $T$ tiene un único punto fijo $x^\*$, y para cualquier $x_0$, la sucesión $x_{k+1}=T(x_k)$ converge a $x^\*$ con tasa geométrica: $\|x_k - x^\*\| \le \lambda^k \|x_0-x^\*\|$.

### 2.2 Condición suficiente vía el Jacobiano (cálculo diferencial estándar)

Si $T$ es diferenciable en un dominio convexo $D\subseteq\mathbb{R}^n$, por la desigualdad del valor medio para funciones vectoriales:

$$\|T(x)-T(y)\| \le \Big(\sup_{z\in D} \|J_T(z)\|_2\Big) \cdot \|x-y\|,$$

donde $\|J_T(z)\|_2 = \sigma_{\max}(J_T(z))$ es la **norma espectral** del Jacobiano (su mayor valor singular). Por lo tanto:

$$\sup_{z\in D} \sigma_{\max}(J_T(z)) \le \lambda < 1 \implies T \text{ es una } \lambda\text{-contracción en } D.$$

Esta es exactamente la condición que el enunciado pide monitorear: **el mayor valor singular del Jacobiano** como certificado (suficiente, no necesario — ver §5) de convergencia del operador `FIXPOINT`.

### 2.3 Estimación de $\sigma_{\max}(J_T(x))$ sin formar el Jacobiano

Formar $J_T(x)\in\mathbb{R}^{n\times n}$ explícitamente cuesta $O(n^2)$ en memoria y su SVD completa $O(n^3)$ en tiempo — inviable para $n$ grande. Se usa en cambio **iteración de potencia** sobre $J^\top J$ (equivalente a los métodos de Lanczos/power-method para el autovalor dominante), calculando $\sigma_{\max}(J) = \sqrt{\lambda_{\max}(J^\top J)}$ mediante únicamente:

- **JVP** (*Jacobian-vector product*, autodiff modo forward): $u = J\,v$
- **VJP** (*vector-Jacobian product*, autodiff modo reverse): $w = J^\top u$

cada uno con costo equivalente a un solo forward/backward pass ($O(n)$ amortizado en la práctica, no $O(n^2)$). Esta es la técnica exacta detrás de **Spectral Normalization** (Miyato et al., 2018), aquí reutilizada como **monitor de benchmarking** en vez de como normalización forzada de pesos.

Iteración (por paso $t$):
$$u_t = \frac{J v_{t-1}}{\|J v_{t-1}\|}, \qquad v_t = \frac{J^\top u_t}{\|J^\top u_t\|}, \qquad \sigma_t \approx \|J v_{t-1}\|.$$

Con **una sola iteración por paso de entrenamiento/benchmarking**, reutilizando $v$ entre pasos consecutivos (tal como en Spectral Norm GAN), el estimador converge en pocos pasos globales sin agregar costo asintótico relevante.

## 3. Razonamiento y alternativas descartadas

- **Alternativa descartada — SVD completa por paso:** correcta pero $O(n^3)$; inviable para monitoreo continuo durante entrenamiento. Se descarta.
- **Alternativa descartada — cota de Frobenius** ($\|J\|_F \ge \|J\|_2$): es una cota más floja y mucho menos ajustada; usarla como proxy de $\sigma_{\max}$ sería conservador en exceso y generaría falsos positivos de "violación". Se descarta en favor de la norma espectral real.
- **Elegido — iteración de potencia + JVP/VJP:** es el estándar de la literatura (Spectral Normalization, Miyato et al. 2018) para exactamente este problema: estimar $\sigma_{\max}(J)$ de forma barata y diferenciable, sin materializar $J$.
- **Importante — el enunciado dice "asegurar convergencia"**, lo cual podría leerse como que la condición espectral es necesaria y suficiente. Matemáticamente es solo **suficiente** (ver Teorema de Banach): puede haber operadores con $\sigma_{\max}(J)\ge 1$ en algún punto que igual converjan globalmente (p. ej. por saturación de una no linealidad tipo `tanh` fuera de la región de referencia). Este supuesto implícito del enunciado **se trata como pregunta abierta** y se investiga empíricamente en la Sección 5, no se da por sentado.

## 4. Desarrollo y Código

Ver archivo adjunto `tarea_25_spectral_fixpoint.py` (funcional, PyTorch). Contiene:

1. `FixpointOperator`: $T(x)=\tanh(Wx+b)$, con $W$ normalizada para tener norma espectral controlable (`spectral_scale`) — permite forzar o romper la condición de contracción a voluntad para poner a prueba el monitor.
2. `estimate_top_singular_value`: iteración de potencia (1 paso, amortizada) usando `torch.autograd.functional.jvp` y `.vjp`, sin formar el Jacobiano.
3. `benchmark_fixpoint_operator`: framework de benchmarking que corre $N$ trayectorias de la iteración de punto fijo, monitorea $\sigma_{\max}$ en cada paso, cuenta violaciones del umbral $\lambda$, y registra si cada trayectoria converge empíricamente (criterio de Cauchy sobre $\|x_{k+1}-x_k\|$).
4. `apply_spectral_penalty`: implementación literal de la "Spectral Lipschitz Penalty" del paper — reescala $W$ para forzar $\|W\|_2 \le \lambda_{\text{target}}$.
5. `run_demo`: compara dos regímenes — **(a)** sin restricción espectral, **(b)** con la penalidad espectral aplicada — y reporta ambos honestamente.

## 5. Resultado del sanity-check interno (no es la corrida final del alumno)

Se corrió una vez de forma interna solo para verificar ausencia de errores de sintaxis/ejecución (según el punto 8 de la metodología del curso). Salida obtenida:

```
REGIMEN (a): SIN restriccion espectral (spectral_scale=1.5 > 1)
[sin_restriccion] lambda_umbral=1.000 | sigma_max observado=1.5000 | violaciones=1125 | trayectorias convergentes=20/20

REGIMEN (b): CON Spectral Lipschitz Penalty (lambda=0.9)
[con_penalidad_espectral] lambda_umbral=0.900 | sigma_max observado=0.9000 | violaciones=0 | trayectorias convergentes=20/20
```

**Esto es importante y no se oculta:** en esta corrida particular, **ambos** regímenes convergieron 20/20 — incluso el que viola sistemáticamente la cota espectral global. La explicación matemática es consistente con la teoría, no la contradice: la cota $\sigma_{\max}(J_T)\le\lambda$ es una condición **suficiente pero no necesaria** (Sección 3). Aquí $T(x)=\tanh(Wx+b)$ tiene una no linealidad saturante; aunque $\|W\|_2=1.5>1$ da una cota global de Lipschitz mayor a 1, el Jacobiano real $J_T(x) = \mathrm{diag}(1-\tanh^2(Wx+b))\,W$ se **contrae localmente** en cuanto las trayectorias entran en la región de saturación de $\tanh$, donde $1-\tanh^2(\cdot)\ll 1$. Es decir: el criterio espectral es un **certificado suficiente** (si se cumple, hay garantía matemática), pero su incumplimiento **no certifica divergencia** — puede haber contracción efectiva por otros mecanismos (aquí, saturación).

**Conclusión técnica:** el framework de benchmarking cumple su función (monitorea correctamente $\sigma_{\max}(J_T)$ y detecta violaciones de la cota con precisión), y el régimen (b) sí ofrece una **garantía matemática a priori** que el régimen (a) no ofrece — aunque en este experimento particular ambos hayan convergido en la práctica. La ventaja real de (b) es que la convergencia está *garantizada por construcción*, mientras que en (a) es *incidental* y no se puede confiar en ella para otras arquitecturas de $T$ sin saturación (p. ej. capas lineales puras, ReLU no acotada, etc.).

## 6. Instrucciones para que el alumno corra el código

```bash
pip install torch --break-system-packages   # o en un entorno virtual, sin el flag
python3 tarea_25_spectral_fixpoint.py
```

Sugerencias de exploración (para comprobar esto con tus propios ojos, no solo leerlo):
- Cambiar `T = FixpointOperator(dim=32, spectral_scale=...)` en `run_demo` a una no linealidad **no saturante** (ej. reemplazar `tanh` por identidad o `LeakyReLU` en `FixpointOperator.forward`) y volver a correr el régimen (a): ahí sí debería observarse divergencia real cuando $\sigma_{\max}>1$, porque desaparece el mecanismo de saturación que "salvaba" la convergencia en el caso con `tanh`.
- Aumentar `spectral_scale` a valores más grandes (ej. 3.0, 5.0) en el régimen (a) y ver en qué punto la saturación de `tanh` deja de alcanzar para garantizar convergencia empírica.
- Aumentar `n_power_iters` en `estimate_top_singular_value` y verificar que el estimador de $\sigma_{\max}$ se estabiliza (converge) más rápido, a costa de más cómputo por paso.

## 7. Resumen estructurado

```json
{
  "alumno": null,
  "tarea": 25,
  "tipo_problema": "Análisis numérico / Optimización — condición de contracción vía norma espectral del Jacobiano (Teorema de Punto Fijo de Banach)",
  "estado": "resuelto",
  "resultado_resumen": "Se implementó un monitor de sigma_max(J_T) via iteración de potencia + JVP/VJP (sin formar el Jacobiano). La cota espectral < 1 es suficiente pero no necesaria para convergencia; el sanity-check mostró convergencia en ambos regímenes debido a saturación de tanh, lo cual se documenta explícitamente como hallazgo, no se oculta.",
  "metrica_estandar_usada": "Norma espectral (mayor valor singular) del Jacobiano; residuo de Cauchy ||x_{k+1}-x_k|| para convergencia empírica"
}
```
