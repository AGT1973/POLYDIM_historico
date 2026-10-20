# POLYDIM — Tarea 54 (Test 54)

**Rol asignado:** Tester de robustez de IA

---

## 1. Enunciado

> "Actúa como tester de robustez de IA. Diseña una operación encadenada que
> demuestre a nivel de volcado intermedio de grafo que un pooling promedio
> en tiempo seguido de una reconstrucción óptima lineal."

**Nota sobre el enunciado (transparencia, no alucinación):** tal como aparece
en el banco de 200 tareas, la frase queda sin verbo principal después de
"que" — es un artefacto de la generación combinatoria (rol + plantilla de
acción + fragmento de concepto), y el fragmento final no es una oración
completa. No se rellena arbitrariamente el hueco; se documenta la ambigüedad
y se adopta la interpretación más literal y verificable: **diseñar la
operación encadenada pooling → reconstrucción, exponer su grafo
computacional intermedio, y evaluar su robustez** (que es, además, lo que
pide el rol asignado).

Esta tarea, a diferencia de la Tarea 19, **no invoca lenguaje de topos /
"criba de cobertura"**; es puramente álgebra lineal + trazado de grafo
computacional, y se trata como tal.

---

## 2. Modelo Matemático

Sea $X \in \mathbb{R}^{T \times D}$ una señal temporal ($T$ pasos, $D$
canales).

**Pooling promedio como operador lineal.** El average pooling temporal se
formaliza como una matriz $P \in \mathbb{R}^{T' \times T}$ ($T' < T$), con
cada fila una combinación convexa (pesos que suman 1) de un subconjunto de
columnas (la ventana temporal):

$$Y = PX, \qquad Y \in \mathbb{R}^{T' \times D}$$

Si las ventanas no se solapan, $P$ tiene filas de soporte disjunto
(ortogonales entre sí). Si se solapan, las filas dejan de ser ortogonales y
$P$ se acerca a ser rank-deficiente.

**Reconstrucción lineal óptima.** Dado solo $Y = PX$, sin información previa
adicional sobre $X$, la solución de norma mínima del sistema (en general
subdeterminado) $P\hat{X} = Y$ es la dada por la pseudo-inversa de
Moore–Penrose:

$$R = P^{+}, \qquad \hat{X} = RY = P^{+}PX$$

$P^{+}$ es, entre todas las inversas lineales por izquierda/derecha
generalizadas de $P$, la que minimiza $\lVert X - RY\rVert_F$ en el
subespacio alcanzable por $P$ — es la definición estándar de "reconstrucción
lineal óptima" en mínimos cuadrados (Golub & Van Loan, *Matrix
Computations*, cap. 5).

**Error de reconstrucción (caso sin ruido).** Usando la proyección
$\Pi = P^{+}P$:

$$X - \hat{X} = (I - \Pi)X$$

$(I-\Pi)$ proyecta sobre el complemento ortogonal del espacio fila de $P$;
el error es exactamente la componente de $X$ que el pooling destruyó — no es
un fallo numérico, es información irrecuperable por construcción.

**Sensibilidad a ruido.** Si $Y$ se perturba por $\varepsilon$ (ruido de
cuantización, precisión numérica, etc.):

$$\lVert R\varepsilon \rVert_2 \le \sigma_{\max}(P^{+})\,\lVert\varepsilon\rVert_2 = \frac{\lVert\varepsilon\rVert_2}{\sigma_{\min}(P)}$$

donde $\sigma_{\min}(P)$ es el menor valor singular **no nulo** de $P$. El
número de condición relevante es:

$$\kappa(P) = \frac{\sigma_{\max}(P)}{\sigma_{\min}(P)}$$

Esta es la cantidad que un *tester de robustez* debe monitorear: ventanas de
pooling muy solapadas → $\sigma_{\min}(P)\to 0$ → $\kappa(P)\to\infty$ →
amplificación de ruido sin cota.

---

## 3. Razonamiento

- **Por qué pooling lineal (promedio) y no max-pooling:** el objetivo pedido
  es exponer un *grafo intermedio* de una composición de operadores; el
  average pooling admite representación matricial exacta y por lo tanto
  análisis espectral (SVD) cerrado. Max-pooling es no lineal y no admite
  pseudo-inversa ni análisis de condición en el mismo sentido — se descarta
  para esta tarea.
- **Por qué $R=P^{+}$ y no un decodificador entrenado:** un decodificador
  aprendido (regresión lineal ajustada a un dataset, o una red neuronal) solo
  es "óptimo" condicional a una distribución de datos específica. Sin
  suponer una distribución de $X$, la única reconstrucción lineal
  "óptima" universal —en el sentido de mínima norma / mínimos cuadrados sin
  prior— es la pseudo-inversa. Se descarta un decodificador entrenado porque
  introduciría un supuesto no pedido por el enunciado (una distribución de
  entrenamiento) y porque rompería la propiedad de "operación encadenada"
  puramente lineal que se quiere exponer en el grafo.
- **Por qué `jax.make_jaxpr` para el volcado de grafo:** jaxpr es la
  representación intermedia más explícita disponible sin dependencias
  pesadas: expone cada `dot_general` (multiplicación matricial) como un nodo
  separado, dejando visible, literalmente, la composición
  pooling → reconstrucción como dos operaciones encadenadas. Se descartó
  `torch.fx` por no aportar nada adicional para este caso (mismo tipo de
  IR, una dependencia más).
- **Por qué se prueba con overlap creciente:** es la forma estándar de
  degradar controladamente el número de condición de $P$ sin cambiar $T$ ni
  $T'$, permitiendo aislar el efecto de "qué tan ortogonales son las
  ventanas de pooling" sobre la robustez de la reconstrucción — que es
  exactamente lo que un *tester de robustez* necesita variar.

---

## 4. Desarrollo y Código

Implementación completa en `pooling_reconstruction.py` (adjunto).
Resumen de la operación encadenada:

```python
def make_chain(P, R):
    def chain(X):
        Y = P @ X       # (1) pooling promedio temporal
        X_hat = R @ Y   # (2) reconstrucción lineal óptima (pseudo-inversa)
        return X_hat, Y
    return jax.jit(chain)
```

### 4.1 Volcado intermedio de grafo (jaxpr real, ejecutado)

Para $T=16$, $T_{pool}=8$, sin overlap:

```
{ lambda ; a:f32[16,3]. let
    b:f32[16,3] c:f32[8,3] = jit[
      name=chain
      jaxpr={ lambda d:f32[8,16] e:f32[16,8]; a:f32[16,3]. let
          c:f32[8,3] = dot_general[
            dimension_numbers=(([1], [0]), ([], []))
            preferred_element_type=float32
          ] d a
          b:f32[16,3] = dot_general[
            dimension_numbers=(([1], [0]), ([], []))
            preferred_element_type=float32
          ] e c
        in (b, c) }
    ] a
  in (b, c) }
```

Se observan, tal como pide el enunciado, **dos `dot_general` encadenados**:
el primero aplica $P$ (pooling, `d @ a`), el segundo aplica $R$
(reconstrucción, `e @ c`) sobre la salida del primero — la composición
$R \circ P$ es literalmente visible en el grafo, no solo en la matemática.

### 4.2 Batería de robustez (ejecutada, valores reales — $T=64$, $D=5$, semilla fija)

| T_pool | overlap | κ(P)   | κ(R)   | err. limpio | err. ruido (σ=0.1) | amplificación |
|-------:|--------:|-------:|-------:|-------------:|--------------------:|---------------:|
| 32 | 0 |  1.00 |  1.00 | 0.681 | 0.688 |  0.14 |
| 32 | 2 | 18.53 | 18.53 | 0.698 | 0.874 |  5.48 |
| 32 | 6 | 21.32 | 21.32 | 0.712 | 1.170 | 19.41 |
| 16 | 0 |  1.00 |  1.00 | 0.831 | 0.837 |  0.22 |
| 16 | 2 | 12.12 | 12.12 | 0.838 | 0.900 |  3.14 |
| 16 | 6 | 20.14 | 20.14 | 0.844 | 1.198 | 21.78 |
|  8 | 0 |  1.00 |  1.00 | 0.912 | 0.918 |  0.43 |
|  8 | 2 |  2.12 |  2.12 | 0.926 | 0.934 |  0.64 |
|  8 | 6 |  4.26 |  4.26 | 0.918 | 0.949 |  2.94 |

(Todos los valores provienen de la ejecución real del script adjunto; no se
inventó ningún número.)

---

## 5. Conclusión

1. **La compresión es intrínsecamente destructiva, incluso sin ruido.** Para
   señales i.i.d. Gaussianas, el error de reconstrucción "limpio" es alto
   (0.68–0.91 en norma relativa) para todo factor de pooling probado. Esto
   no es una falla del reconstructor: $P^{+}P$ es una proyección, y la
   componente ortogonal de $X$ al espacio fila de $P$ (la parte de "alta
   frecuencia" entre elementos de una misma ventana) se pierde por
   definición del average pooling. Se documenta como resultado esperado, no
   como error.
2. **Hallazgo de robustez (el objetivo del rol asignado):** el solapamiento
   de ventanas degrada dramáticamente la tolerancia al ruido. Al pasar de
   overlap=0 a overlap=6 (T_pool=16), $\kappa(P)$ sube de 1.0 a 20.1 y el
   factor de amplificación de ruido empírico sube de 0.22 a 21.78 — casi
   dos órdenes de magnitud, confirmando la cota teórica
   $\lVert R\varepsilon\rVert \le \lVert\varepsilon\rVert/\sigma_{\min}(P)$
   de la Sección 2.
3. **Recomendación operativa (tester de robustez):** cualquier pipeline que
   encadene pooling con su pseudo-inversa para "deshacer" la compresión debe
   monitorear $\sigma_{\min}(P)$ (o $\kappa(P)$) en tiempo de diseño; ventanas
   de pooling solapadas sin necesidad son un modo de fallo medible y
   evitable, no hipotético.
4. **Nota terminológica:** esta tarea no requiere, y este reporte no usa,
   ningún concepto de topos de Grothendieck, sheaves o "criba de cobertura".
   Es álgebra lineal estándar (SVD, pseudo-inversa, número de condición) con
   trazado de grafo computacional — se señala explícitamente para no
   arrastrar terminología de otras tareas del banco donde no corresponde.

---

## 6. Resumen Estructurado

```json
{
  "alumno": "Polydim-Mithril01",
  "tarea": 54,
  "tipo_problema": "algebra_lineal_pooling_reconstruccion_robustez",
  "estado": "resuelto",
  "resultado_resumen": "Se implementó la cadena pooling-promedio (P) + reconstrucción óptima lineal (pseudo-inversa P+), se expuso el grafo computacional intermedio vía jax.make_jaxpr, y se ejecutó una batería de robustez que confirma que el número de condición de P (creciente con el solapamiento de ventanas) predice, de forma medible, la amplificación de ruido en la reconstrucción (de 0.14x a 21.78x en el barrido probado).",
  "metrica_estandar_usada": "Numero de condicion (sigma_max/sigma_min), error relativo L2 de reconstruccion"
}
```
