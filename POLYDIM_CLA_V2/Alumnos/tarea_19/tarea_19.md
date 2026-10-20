# Tarea 19 — Curso IA (UTN) — Banco POLYDIM

## 0. Nota sobre el proceso

El banco maestro trae, además de las 200 tareas, un bloque de instrucciones ("Marco de Validación Académica V3.0") que pide mostrar en el chat *todo* el desarrollo paso a paso, estilo clase particular. Esa instrucción viene dentro del archivo fuente, no de vos como alumno, y choca con la metodología de esta cursada (Aprendizaje por Descubrimiento Guiado): en esta etapa el desarrollo completo va en este archivo, y en el chat solo se confirma qué se resolvió — así llegás a la etapa de exposición final sin haber visto ya la solución servida. Por eso este `.md` sigue el formato de la cursada, no el del banco.

## 1. Enunciado original

> **19 (Test 19):** Actúa como investigador de convergencia de ML. Implementa el algoritmo de Alineación Espectral que un subespacio latente R^3 de forma estática, validando si pertenece a la criba de cobertura del topos.

Como advierte la metodología del curso, el banco se arma combinando plantillas (rol + fragmento técnico) y varios fragmentos quedan sin verbo o rotos gramaticalmente. Este es un caso claro: *"...el algoritmo de Alineación Espectral que un subespacio latente R^3..."* no tiene predicado completo.

## 2. Reconstrucción del enunciado

**Versión reformulada, matemáticamente investigable:**

> Implementar un algoritmo de *Alineación Espectral* que, dado un subespacio latente estático $S \subseteq \mathbb{R}^3$ y una familia cubridora $\{U_i\}$ que genera una topología de Grothendieck sobre el sitio de subespacios de $\mathbb{R}^3$, determine si $S$ pertenece a la **criba de cobertura** (covering sieve) generada por esa familia — y hacerlo mediante un criterio espectral (autovalores del operador de proyección), no por inspección manual.

Los dos fragmentos reconocibles que se conservan del enunciado original son: (a) *"algoritmo de Alineación Espectral"* — mencionado en la Sección 9.3 del marco POLYDIM como protocolo `ALIGN`; y (b) *"validando si [un subespacio] pertenece a la criba de cobertura del topos"* — que remite a la Sección 2.1 del mismo marco (sitio, topología de Grothendieck $J$, clasificador de subobjetos $\Omega$). Lo que se agregó para que el problema sea investigable es la **definición operacional concreta** de "criba de cobertura" y de "alineación espectral" sobre un sitio construido en $\mathbb{R}^3$, que el enunciado da por sabidas pero no especifica.

Importante: el marco POLYDIM describe la topología de Grothendieck y el clasificador de subobjetos como una analogía/formalización propuesta para IA, no como un resultado establecido de la literatura estándar de deep learning; lo que sí es matemática estándar y lo que efectivamente se demuestra abajo es (i) que los subespacios de $\mathbb{R}^3$ con la operación suma forman un sitio válido con una topología de Grothendieck genuina, y (ii) que la pertenencia a una criba generada por una familia cubridora es decidible mediante un criterio espectral. No se demuestra —ni se necesita para este ejercicio— ninguna afirmación sobre si esto "mejora" el razonamiento de un modelo de lenguaje real.

## 3. Modelo matemático

### 3.1 El sitio $(\mathcal{C}, J)$

Sea $\mathcal{C}$ la categoría cuyos objetos son subespacios vectoriales de $X = \mathbb{R}^3$ y cuyos morfismos son las inclusiones lineales $T \hookrightarrow X$. El retículo de subespacios de $\mathbb{R}^3$ (bajo suma $+$ e intersección $\cap$, con máximo $\mathbb{R}^3$) admite una topología de Grothendieck estándar:

$$
\{U_i \hookrightarrow X\}_{i \in I} \in J(X) \iff \sum_{i \in I} U_i = X
$$

es decir, una familia cubre a $X$ si y solo si sus elementos *generan* (en el sentido de suma de subespacios) el espacio ambiente completo — el análogo lineal de "la unión de abiertos es el espacio total".

### 3.2 La criba generada

Dada una familia cubridora $\{U_i \to X\}$, la **criba** (sieve) que genera es, por definición categórica estándar:

$$
\langle U_i \rangle = \{\, T \hookrightarrow X \;:\; \exists\, i,\ T \subseteq U_i \,\}
$$

(cerradura por precomposición: cualquier subobjeto que factorice a través de algún generador pertenece a la criba). Un subespacio latente estático $S \subseteq \mathbb{R}^3$ **pertenece a la criba** si y solo si existe algún $i$ tal que $S \subseteq U_i$.

### 3.3 Alineación espectral como criterio de pertenencia

Sea $U_i$ con base ortonormal (columnas de $U$). El operador de proyección ortogonal

$$
\Pi_{U_i} = U U^{T}
$$

es diagonalizable con espectro exactamente $\{1^{(\dim U_i)},\, 0^{(3 - \dim U_i)}\}$: el autoespacio de autovalor $1$ es $U_i$ mismo, y el de autovalor $0$ es su complemento ortogonal. Esta dicotomía espectral es el criterio de "alineación":

$$
S \subseteq U_i \iff \Pi_{U_i}\, v = v \ \ \forall v \in \text{base}(S) \iff \| (I - \Pi_{U_i})\, v \| = 0
$$

Definimos el **residual de alineación espectral**:

$$
r(S, U_i) = \frac{\| (I - \Pi_{U_i})\, B_S \|_F}{\dim S}
$$

donde $B_S$ es una base ortonormal de $S$. Entonces:

$$
S \in \langle U_i \rangle \iff \min_i\, r(S, U_i) < \varepsilon
$$

para una tolerancia numérica $\varepsilon$ (necesaria por precisión de punto flotante, ver §4). Este es exactamente el mecanismo que el marco POLYDIM describe como "pullback al objeto inicial $\emptyset$": cuando ningún $U_i$ alinea con $S$, la proposición "$S$ está cubierto" colapsa a falso ($\Omega = \bot$) en vez de lanzar una excepción.

## 4. Razonamiento y alternativas descartadas

- **¿Por qué proyección ortogonal y no comparación de ángulos entre vectores individuales?** Porque la pertenencia $S \subseteq U_i$ es una afirmación sobre *todo el subespacio* $S$, no vector por vector; el operador de proyección captura esto de una sola vez para cualquier base de $S$, y su espectro (0/1) da una certificación exacta libre de la elección de base — coherente con la exigencia del enunciado de un criterio "espectral".
- **¿Por qué no Transporte Óptimo entrópico (Sinkhorn) como propone la Sección 9.3 del marco para el protocolo `ALIGN` general?** Esa maquinaria está pensada para alinear *distribuciones o nubes de puntos de cardinalidad/dimensión distintas*. Acá $S$ y $U_i$ son subespacios exactos del mismo espacio ambiente $\mathbb{R}^3$; usar Sinkhorn sería una complicación computacional sin beneficio, y de hecho introduciría error de aproximación innecesario en un problema que tiene solución algebraica exacta (proyección ortogonal). Se documenta la alternativa para que quede explícito qué se descartó y por qué, no porque sea incorrecta en otro contexto (nubes de puntos de distinta dimensión, que es el caso para el que el marco la propone).
- **¿Por qué $\varepsilon$ y no comparación exacta a $0$?** Aritmética de punto flotante (ver Sección 4.1 del paper POLYDIM sobre `GEO_ID`/Kan filling): la igualdad estricta es intratable en hardware SIMD; se usa una tolerancia, documentada, en vez de fingir exactitud.

## 5. Desarrollo y código

El código completo está en `tarea_19.py` (mismo directorio). Resumen de lo que hace:

1. `orthonormal_basis`: ortonormaliza (QR) los generadores de un subespacio.
2. `projection_matrix`: construye $\Pi_U = UU^T$.
3. `spectral_alignment_residual`: calcula $r(S, U_i)$.
4. `covering_sieve_membership`: recorre la familia cubridora y decide pertenencia a la criba.
5. `is_valid_cover`: chequeo de sanidad de que $\{U_i\}$ efectivamente genera $J(X)$ (condición previa, independiente de si un $S$ particular está en la criba).

Corrida de sanity-check interna (no reemplaza que vos lo corras y lo mires con tus propios ojos): con la familia cubridora de los 3 planos coordenados de $\mathbb{R}^3$,

- un vector en el plano $xy$ → pertenece a la criba (residual $\approx 0$ contra $U_1$),
- el vector diagonal $(1,1,1)$ → **no** pertenece a la criba (residual $\approx 0.577$ contra los tres $U_i$): cae en el pullback a $\emptyset$, aunque $(1,1,1)$ sí está en $\mathbb{R}^3$ — la criba exige factorizar por *un único* generador, no por la suma de todos.
- el eje $x$ → pertenece a la criba por partida doble ($U_1$ y $U_3$ lo contienen).

Este último caso ilustra que "pertenecer a la criba" es más fuerte que "estar cubierto por la unión": ser miembro de la familia cubridora global no basta si no factoriza por un solo generador — es la diferencia formal entre cobertura de $X$ y pertenencia a la sieve generada.

## 6. Conclusión

El modelo converge y es decidible en tiempo cerrado (sin iteración): la pertenencia a la criba es $O(|I| \cdot n^2)$ para $n = \dim X$ y $|I|$ generadores, dominado por la construcción de cada proyector. No hay caso de no-convergencia aquí porque el criterio es algebraico exacto, no un punto fijo iterativo — a diferencia de otras tareas del banco (p. ej. las que involucran `FIXPOINT` o penalización espectral de Lipschitz), acá no aplica reportar divergencia.

Lo que sí queda abierto, y es tu parte activa: correr `tarea_19.py`, modificar la familia cubridora y los subespacios de prueba, y verificar con tus propios ojos en qué casos el criterio espectral coincide con la definición categórica de pertenencia a la sieve, y en qué casos el umbral $\varepsilon$ empieza a dar falsos positivos/negativos si metés ruido en los vectores generadores (por ejemplo, perturbando $U_i$ con ruido gaussiano pequeño y viendo cuándo el residual deja de ser $\approx 0$ para un $S$ que debería pertenecer).

## 7. Cómo correrlo vos mismo

```bash
pip install numpy --break-system-packages   # si no lo tenés instalado
python3 tarea_19.py
```

No requiere GPU ni librerías adicionales. Si querés extenderlo con la variante de Transporte Óptimo entrópico mencionada en §4 (para el caso de subespacios de distinta dimensión/cardinalidad), la librería `POT` (`pip install pot`) es el punto de partida estándar — quedó fuera de este entregable porque no es necesaria para el caso exacto que pide el enunciado reconstruido.

```json
{
  "tarea": 19,
  "tipo_problema": "Teoría de categorías / topología de Grothendieck / álgebra lineal espectral",
  "estado": "resuelto",
  "resultado_resumen": "Se formalizó la criba de cobertura sobre el sitio de subespacios de R^3 y se implementó un criterio espectral exacto (autovalores 0/1 del proyector ortogonal) para decidir pertenencia de un subespacio estático S a dicha criba.",
  "metrica_estandar_usada": "Residual de proyección ortogonal (norma de Frobenius normalizada)"
}
```
