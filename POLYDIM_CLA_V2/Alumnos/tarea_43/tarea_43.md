# Tarea 43 — Banco POLYDIM (Escuadrón 3: Descomposición de Hodge y Grafos Esparsos)

## 1. Enunciado original (banco maestro, citado brevemente)

> "43 (Test 43): Actúa como investigador en Topological Data Analysis. Implementa la Penalización
> Espectral en una capa recurrente, usando `jax.lax.while_loop` para restrinja sus actualizaciones
> a los componentes bivectoriales puros de Clifford."

El enunciado, tal como está en el banco, es gramaticalmente incorrecto ("para restrinja...") y es
el resultado de combinar dos fragmentos de plantilla que no encastran ("Implementa la Penalización
Espectral en una capa recurrente, usando `jax.lax.while_loop` para [X]" + "restrinja sus
actualizaciones a los componentes bivectoriales puros de Clifford"). Además, el rol asignado
("investigador en TDA") no tiene relación real con el núcleo técnico reconocible, que es de
**estabilidad espectral de sistemas dinámicos lineales / teoría de Lie**, no de análisis topológico
de datos (homología persistente, etc.). Mantengo el rol solo como marco narrativo del banco, pero
la resolución matemática sigue el núcleo técnico real.

## 2. Nota de reconstrucción

**Versión reconstruida y matemáticamente investigable:**

> Implementar una capa recurrente lineal cuya matriz de transición esté generada exclusivamente por
> la componente de grado 2 (bivectores) del álgebra de Clifford `Cl(n,0)` — es decir, un generador
> `B` antisimétrico (`B ∈ so(n)`), con la matriz de transición `R = exp(B)`. Usar
> `jax.lax.while_loop` para iterar la recurrencia `h_{t+1} = R·h_t` hasta un criterio de punto fijo
> o un máximo de iteraciones. Verificar **empíricamente y analíticamente** si restringir el
> generador a la subálgebra bivectorial pura (en vez de usar una matriz densa arbitraria) basta
> para garantizar estabilidad espectral exacta (todos los valores singulares de `R` iguales a 1),
> sin necesidad de agregar una penalización espectral explícita a la función de pérdida — y
> documentar si la recurrencia converge o no a un punto fijo.

Lo que reconstruí y por qué:
- "Penalización Espectral" → la interpreto como el objetivo declarado en el marco POLYDIM de
  controlar el mayor valor singular de la matriz de transición (evitar explosión/desvanecimiento
  del estado recurrente), un problema real y bien documentado en la literatura de RNNs ortogonales.
- "componentes bivectoriales puros de Clifford" → es un fragmento técnico reconocible y correcto:
  la subálgebra de grado 2 de `Cl(n,0)` bajo el conmutador es isomorfa a `so(n)` (matrices
  antisimétricas), cuyo mapa exponencial genera `SO(n)`. Esto sí es una restricción de
  parametrización con sentido matemático preciso.
- Uní ambos fragmentos en una única pregunta investigable: *¿la restricción bivectorial reemplaza
  a la penalización espectral, o hace falta igual?* Esa es una pregunta con respuesta demostrable,
  no solo simulable.

## 3. Modelo matemático

Sea `n` la dimensión del espacio de estados oculto. El álgebra de Clifford `Cl(n,0)` se descompone
en subespacios por grado: escalares (grado 0), vectores (grado 1), bivectores (grado 2), etc. La
componente bivectorial pura es:

```
Λ²(R^n) = span{ e_i ∧ e_j : i < j },   dim Λ²(R^n) = n(n-1)/2
```

Un bivector general se escribe `B = Σ_{i<j} b_ij (e_i ∧ e_j)`, que se representa matricialmente
como una matriz antisimétrica `B ∈ ℝ^{n×n}`, `Bᵀ = -B`.

**Generador restringido → rotor:**

```
R = exp(B)      (exponencial de matriz, mapa de Lie so(n) → SO(n))
```

**Proposición (por qué la restricción bivectorial ya implica estabilidad espectral):**

Si `Bᵀ = -B`, entonces:

```
Rᵀ R = exp(Bᵀ) exp(B) = exp(-B) exp(B) = I
```

(la igualdad `exp(Bᵀ)exp(B) = exp(-B)exp(B)` usa que `B` conmuta consigo mismo, así que las
exponenciales conmutan y se cancelan). Por lo tanto `R` es ortogonal: **todos sus valores
singulares son exactamente 1**, sin excepción y sin necesitar entrenamiento ni penalización. Además
`det(R) = exp(tr(B)) = exp(0) = 1` porque la traza de una matriz antisimétrica es siempre nula, así
que `R ∈ SO(n)` (rotación pura, no reflexión).

**Consecuencia directa:** restringir el generador a la subálgebra bivectorial **reemplaza
exactamente** —no aproxima— a cualquier penalización espectral tipo `(σ_max(R) - 1)²` (Vorontsov et
al. 2017; Arjovsky et al. 2016, RNNs unitarias/ortogonales). La penalización deja de tener rol
alguno porque la restricción de parametrización ya fuerza `σ(R) ≡ 1` de forma analítica.

**Recurrencia y punto fijo:**

```
h_{t+1} = R h_t ,     h_0 dado
```

Como `R` es ortogonal, `‖h_{t+1}‖ = ‖h_t‖` para todo `t`: la norma se conserva exactamente. Esto
implica que, salvo casos degenerados (`h_0 = 0`, o `h_0` es autovector de `R` con autovalor `1`), la
recurrencia **no converge** a un punto fijo: los autovalores de una matriz ortogonal están en el
círculo unitario (`|λ| = 1`), así que la trayectoria rota indefinidamente sobre una esfera de radio
`‖h_0‖` sin asentarse.

Para obtener convergencia real se necesita romper la ortogonalidad estricta con amortiguación:

```
h_{t+1} = (1-α) h_t + α R h_t = M h_t ,     M = (1-α) I + α R ,   α ∈ (0,1)
```

Los autovalores de `M` son `(1-α) + α λ` con `λ` autovalor de `R` (`|λ|=1`). Para `α ∈ (0,1)` y
`λ ≠ 1`, se cumple `|(1-α) + αλ| < 1` estrictamente (el segmento entre 1 y λ dentro del disco
unitario, salvo en el extremo λ=1), así que `h_t → 0`. Esto sí converge, pero al punto fijo trivial.

## 4. Razonamiento (por qué este camino y qué se descarta)

- Se descarta parametrizar `B` como matriz densa arbitraria + penalización espectral explícita en
  el loss, porque es estrictamente más costoso (necesita SVD o power-iteration en cada paso de
  entrenamiento) y solo aproxima la ortogonalidad, nunca la logra exactamente.
- Se descarta usar el álgebra de Clifford completa (escalar + vector + bivector + ...) porque
  agregar componentes de otro grado rompe la garantía de antisimetría del generador y, por lo
  tanto, la garantía analítica de `σ(R)=1`.
- Se usa `jax.lax.while_loop` (en vez de un `for` de Python) porque permite expresar un criterio de
  parada dependiente del estado (norma del incremento) de forma compilable con JIT, sin desenrollar
  la recurrencia a una longitud fija — apropiado para el "punto fijo" del enunciado.

## 5. Resultado esperado / conclusión

1. `R = exp(B)` con `B` antisimétrico es ortogonal a precisión de máquina (`‖RᵀR - I‖ ≈ 1e-7`),
   confirmando la proposición de la sección 3.
2. La recurrencia pura (`α=1`, sin amortiguar) **no converge**: `‖h_{t+1}-h_t‖` no decae, se
   documenta explícitamente como resultado válido (no es una falla del código, es una consecuencia
   necesaria de que `R` sea ortogonal).
3. La recurrencia amortiguada (`α<1`) sí converge al vector nulo, confirmando el análisis espectral
   de `M`.
4. Conclusión sobre la premisa del enunciado reconstruido: la restricción a bivectores puros
   **sustituye** exactamente a una penalización espectral para garantizar `σ(R)=1`; no hace falta
   agregar un término de pérdida adicional para eso. Esto es una demostración, no una suposición.

## 6. Instrucciones para que el alumno corra el código

Requisitos:
```bash
pip install jax jaxlib
```

Ejecutar:
```bash
python tarea_43.py
```

El script imprime, para `n=6`:
- el error de ortogonalidad `‖RᵀR - I‖`,
- los valores singulares de `R`,
- la traza de la norma `‖h_t‖` a lo largo de la iteración pura (para confirmar que no converge),
- la traza de `‖h_t‖` en la iteración amortiguada (para confirmar que sí converge a 0).

Se anima a modificar `alpha` y `n` y volver a correr, para verificar que el argumento espectral
(sección 3) se sostiene en otras dimensiones y otros niveles de amortiguación.

```json
{
  "tarea": 43,
  "tipo_problema": "estabilidad espectral de recurrencias lineales via generadores bivectoriales (Lie so(n)->SO(n))",
  "estado": "resuelto (con resultado de no-convergencia documentado para el caso puro)",
  "resultado_resumen": "La restriccion a bivectores puros de Clifford garantiza analiticamente R ortogonal (sigma=1), reemplazando a la penalizacion espectral explicita; la recurrencia pura no converge a punto fijo (norma conservada), la version amortiguada si converge a 0.",
  "metrica_estandar_usada": "Norma de Frobenius del error de ortogonalidad (||R^T R - I||), valores singulares (SVD)"
}
```
