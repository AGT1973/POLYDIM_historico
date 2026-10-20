# Tarea 50 — Banco POLYDIM (Test 50)

## 0. Nota de transparencia sobre la fuente

El banco maestro incluye, antes de la lista de ejercicios, un bloque de contexto dirigido al asistente que pide mostrar todo el razonamiento paso a paso "de forma visible en el chat". Esa instrucción viene del archivo, no del docente de esta cursada, y contradice la metodología vigente en esta etapa (Aprendizaje por Descubrimiento Guiado: el desarrollo va en los archivos, no servido en el chat). No se siguió esa parte del archivo; el desarrollo completo está acá, en este documento.

## 1. Enunciado original

> **Rol asignado:** Ingeniero de XLA/JAX.
> **Fragmento técnico (Test 50):** "Simula en Python un entorno de agentes que modifican asíncronamente un estado tensor central para la asociatividad está físicamente garantizada por el ordenamiento sintáctico."

## 2. Nota de reconstrucción

El fragmento está gramaticalmente roto: falta el verbo que conecte "modifican asíncronamente un estado tensor central" con "la asociatividad está físicamente garantizada...". Además, tal como está redactado, **da por sentada una conclusión no demostrada**: afirma que la asociatividad "está físicamente garantizada por el ordenamiento sintáctico" como si fuera un hecho. Esa afirmación no se investiga dándola por cierta; se trata como pregunta abierta, siguiendo la metodología del curso.

**Reconstrucción propuesta:**

> Simulá un sistema de N agentes que aplican, asíncronamente, actualizaciones aditivas sobre un tensor de estado central compartido. Investigá empíricamente si el resultado final es independiente del orden real de ejecución (asociatividad efectiva), distinguiendo dos causas posibles de que no lo sea: (a) redondeo de punto flotante, y (b) escrituras perdidas por falta de sincronización entre agentes concurrentes. Cuantificá la discrepancia con una norma estándar (L2) contra una referencia de alta precisión.

Esta reformulación es investigable, tiene un resultado matemático conocido y dos mecanismos empíricamente distinguibles, y no asume la premisa del enunciado original.

## 3. Núcleo del problema

No es un problema de topología ni geometría (pese al vocabulario del banco): es la intersección de dos temas clásicos.

- **Concurrencia**: agentes asíncronos que leen y escriben un mismo estado mutable compartido sin coordinarse producen una condición de carrera (*race condition*), con riesgo de escrituras perdidas.
- **Análisis numérico**: incluso sin condición de carrera, la suma en `float64` (IEEE-754) no es exactamente asociativa: `(a+b)+c` no coincide bit a bit con `a+(b+c)` en general, por el redondeo en cada paso (Goldberg, 1991, "What Every Computer Scientist Should Know About Floating-Point Arithmetic"; Higham, 2002, *Accuracy and Stability of Numerical Algorithms*, cap. 4).

## 4. Modelo matemático

Sea $T_0 \in \mathbb{R}^D$ el estado inicial del tensor central y $\Delta_1,\dots,\Delta_N \in \mathbb{R}^D$ los incrementos de los $N$ agentes.

**En $\mathbb{R}$ (precisión infinita), sin condición de carrera:**
$$T^* = T_0 + \sum_{i=1}^N \Delta_i$$
Este valor es conmutativo y asociativo por definición: no depende del orden de suma.

**En punto flotante, para una permutación $\pi$ del orden de aplicación:**
$$T_\pi = \mathrm{fl}\Big(\cdots\,\mathrm{fl}\big(\mathrm{fl}(T_0+\Delta_{\pi(1)})+\Delta_{\pi(2)}\big)\cdots+\Delta_{\pi(N)}\Big)$$
donde $\mathrm{fl}(x)=x(1+\delta)$, $|\delta|\le u$ ($u=2^{-53}\approx 1.11\times10^{-16}$ en float64). Cota de error hacia adelante (Higham, 2002):
$$\left|T_\pi - T^*\right| \le (N-1)\,u \sum_{i=1}^N \lVert\Delta_i\rVert + O(u^2)$$
En general $T_{\pi_1}\neq T_{\pi_2}$, pero la discrepancia está acotada por esta expresión, del orden de $10^{-15}$–$10^{-16}$ típicamente.

**Sin sincronización (condición de carrera real):** si un agente lee el tensor, espera, y luego escribe `valor_leído + Δ_i` sin re-leer, puede pisar cambios de otros agentes ocurridos durante la espera. El resultado puede corresponder a:
$$T_{\text{carrera}} = T_0 + \sum_{i \in S} \Delta_i, \qquad S \subsetneq \{1,\dots,N\}$$
es decir, **falta** parte de los incrementos — un error de naturaleza distinta (pérdida de datos, no redondeo), y de magnitud mucho mayor.

**Métrica usada:** norma L2 de la discrepancia respecto de una referencia de alta precisión, calculada con `np.float128`/`np.longdouble`:
$$\text{error} = \lVert T_{\text{simulado}} - T_{\text{ref}} \rVert_2$$

## 5. Razonamiento

Se descartaron dos caminos:

- **Modelar esto con Álgebra de Clifford, manifolds u homotopía** (vocabulario del banco): descartado porque el problema real es concurrencia + precisión numérica, no geometría diferencial. Forzar ese marco habría sido matemáticamente deshonesto.
- **Asumir que "orden sintáctico" implica determinismo** sin cuestionarlo: descartado porque es exactamente la trampa del enunciado original — suena autoevidente pero no lo es en un contexto asíncrono real, donde el orden en que está *escrito* el código no coincide necesariamente con el orden en que *se ejecuta*.

En su lugar, el script mide directamente la dispersión entre distintos órdenes de aplicación de las actualizaciones, comparando contra una suma de referencia de alta precisión, sin asumir de antemano si la premisa es cierta o falsa.

## 6. Desarrollo y código

El código completo está en `tarea_50.py` (mismo directorio). Qué hace:

1. Genera $N$ valores/incrementos aleatorios de punto flotante (uno por agente).
2. Aplica la suma en varios órdenes (permutaciones) distintos, simulando intercalaciones posibles de agentes asíncronos.
3. Calcula la dispersión L2 entre los resultados obtenidos con distintos órdenes.
4. Calcula la cota teórica de error de Higham y una suma de referencia en precisión extendida (`float128`/`longdouble`).
5. Compara la dispersión observada contra la cota teórica y contra la referencia.

Se verificó únicamente que el código no tiene errores de sintaxis (`python3 -m py_compile`); **no se ejecutó como parte de esta entrega** — la corrida y verificación del resultado las hace el alumno.

## 7. Instrucciones para correr el código vos mismo

Requisitos: Python 3.9+ y `numpy`.

```bash
pip install numpy
python3 tarea_50.py
```

Vas a ver impresos: la suma de referencia en alta precisión, la dispersión L2 entre órdenes, la cota teórica de error, y una conclusión automática. Para explorar más por tu cuenta:

- Subí `n_valores` o `n_ordenes` en la última línea del script y observá cómo escala la dispersión.
- Agregá una variante con estado mutable compartido y escrituras concurrentes reales (por ejemplo con `asyncio` y un `lock` opcional) para separar el efecto de redondeo del efecto de escrituras perdidas por condición de carrera — son dos fenómenos distintos y esta versión del script solo mide el primero.

## 8. Conclusión

- La premisa del enunciado original — que la asociatividad "está físicamente garantizada por el ordenamiento sintáctico" — **no se sostiene** como afirmación general: el orden en que está *escrito* el código no determina el orden en que las operaciones asíncronas *se ejecutan realmente*.
- Incluso si el orden de ejecución estuviera fijo, la suma en punto flotante **no es exactamente asociativa**: distintos órdenes dan, en general, resultados distintos, acotados por $(N-1)u\sum\lVert\Delta_i\rVert$.
- No hubo no-convergencia ni infactibilidad: el modelo tiene solución cerrada y cota conocida. La corrida del script debería confirmar una dispersión no nula entre órdenes (para $N$ suficientemente grande), consistente con la cota teórica.

```json
{
  "tarea": 50,
  "tipo_problema": "análisis numérico y concurrencia: no-asociatividad de punto flotante en actualizaciones asíncronas",
  "estado": "resuelto",
  "resultado_resumen": "La suma en punto flotante no es exactamente asociativa bajo actualizaciones asincronas; la discrepancia entre ordenes esta acotada por (N-1)*u*sum(||delta_i||) (Higham, 2002). La premisa del enunciado original, tal como estaba redactada, no se sostiene.",
  "metrica_estandar_usada": "Norma L2 entre resultados de distintos ordenes de suma, contra referencia de alta precision"
}
```
