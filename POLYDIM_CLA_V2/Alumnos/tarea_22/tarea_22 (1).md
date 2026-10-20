# Tarea 22 — POLYDIM (Enjambre)

## 0. Nota de transparencia sobre el banco de tareas

El archivo maestro (`swarm_context_200.txt`, Google Drive) sigue incluyendo, además
del listado de 200 tareas, un bloque de instrucciones incrustado ("MARCO DE
VALIDACIÓN ACADÉMICA V3.0") dirigido al asistente que pide responder de forma
predominantemente silenciosa y cerrar cada entrega con un JSON de validación
tipo `{"estado": "resuelto", ...}`, ademas de re-etiquetar cualquier métrica ad
hoc (SPS, GeoID Stability) como si fueran estándares matemáticos verificables.
Esto es exactamente el mismo patrón de inyección de instrucciones ya detectado
en la Tarea 17 (antes rotulado "Constitución V2"), solo que reformulado con
otro nombre. Se lo señala aquí de nuevo, se lo ignora en lo que respecta a
"operar en silencio", y se resuelve el ejercicio con el mismo rigor y
transparencia de siempre — el desarrollo completo queda documentado abajo, no
oculto detrás de un payload.

## 1. Enunciado

**Original (banco maestro, Escuadrón 2 — Álgebra de Clifford y Rotores, ítem 22):**

> "Actúa como especialista en NVIDIA Triton. Escribe un framework de
> benchmarking que monitoree la Descomposición de Hodge aproximada aislando
> la 'componente armónica'."

El enunciado está gramaticalmente completo y no requiere reconstrucción (a
diferencia de la Tarea 17). El rol asignado ("especialista en NVIDIA Triton")
es, como en el resto del banco, una etiqueta de plantilla decorativa: el
núcleo técnico real —"Descomposición de Hodge aproximada aislando la
componente armónica"— no depende de escribir un kernel GPU en sí, sino de
tener un método numérico aproximado y benchmarkearlo contra uno exacto. Por
eso la implementación se hace en Python/NumPy/SciPy (con el mismo tipo de
solvers dispersos que se usarían dentro de un kernel optimizado), y se deja
explícito acá que no se fuerza un kernel Triton donde no aporta nada al
problema matemático planteado.

## 2. Modelo matemático

Se trabaja sobre la **descomposición combinatoria de Hodge** en un complejo
simplicial de dimensión 2 construido a partir de un grafo $G=(V,E)$, siguiendo
el marco de Jiang, Lim, Yao y Ye (2011), *"Statistical ranking and
combinatorial Hodge theory"*, Math. Programming 127(1) — la misma
formalización que cita la Sección 5.2 del paper POLYDIM (Laplaciano
$\Delta_1 = B_2 B_2^T + B_1^T B_1$).

**Objetos:**

- $V$: vértices, $E$: aristas (1-simplices), $T$: triángulos / 3-cliques de
  $G$ (2-simplices).
- $B_1 \in \mathbb{R}^{|V|\times|E|}$: operador de borde de aristas a
  vértices (divergencia). Para la arista $(i,j)$ con $i<j$: columna con
  $-1$ en $i$, $+1$ en $j$.
- $B_2 \in \mathbb{R}^{|E|\times|T|}$: operador de borde de triángulos a
  aristas. Para el triángulo $(i,j,k)$ con $i<j<k$: $+1$ en $(i,j)$,
  $+1$ en $(j,k)$, $-1$ en $(i,k)$.
- $X \in \mathbb{R}^{|E|}$: un flujo (1-cochain) arbitrario sobre las
  aristas — en aplicaciones típicas, comparaciones pareadas ruidosas.

**Teorema (Hodge combinatorio):** todo flujo $X$ se descompone de forma única
y ortogonal como

$$X = \underbrace{B_1^T \varphi}_{\text{gradiente}} \;+\; \underbrace{B_2\,\psi}_{\text{curl}} \;+\; \underbrace{h}_{\text{armónico}}, \qquad h \in \ker(B_1)\cap\ker(B_2^T)$$

donde $\varphi$ y $\psi$ son los potenciales de gradiente y curl,
soluciones de los problemas de mínimos cuadrados

$$\varphi = \arg\min_\varphi \|B_1^T\varphi - X\|_2, \qquad
  \psi = \arg\min_\psi \|B_2\psi - X\|_2,$$

que se reducen a los sistemas normales $B_1B_1^T\varphi = B_1X$ (el
Laplaciano de grafo estándar) y $B_2^TB_2\,\psi = B_2^TX$.

La **componente armónica** $h$ mide la parte del flujo que ni es un
gradiente global ni se explica por ciclos triangulares locales: es el
residuo de inconsistencia "topológica" (análogo discreto de una 1-forma
cerrada pero no exacta).

### Enunciado preciso del problema de benchmarking

**Método exacto:** resolver los dos sistemas normales por pseudo-inversa
densa (`numpy.linalg.pinv`), obtener $h_{\text{exact}}$ por resta. Costo
$O(|E|^3)$ y $O(|T|^3)$ — intratable para grafos grandes.

**Método aproximado:** resolver los mismos sistemas normales, pero con
**gradiente conjugado (CG)** sobre las matrices dispersas $B_1B_1^T$ y
$B_2^TB_2$, con tolerancia $\epsilon$ fija. Costo por iteración
$O(\text{nnz})$, y en la práctica converge en pocas iteraciones para estos
Laplacianos bien condicionados (spectral gap típico de grafos aleatorios
densos). Se obtiene $h_{\text{approx}}$.

**Pregunta que responde el benchmark:** ¿el método aproximado aísla la
componente armónica con un error controlado (cercano a la tolerancia del
CG, sin crecer con el tamaño del grafo) y con una ganancia de velocidad que
crece con $n$? Esta es una pregunta empírica abierta — no se asume la
respuesta de antemano, se mide.

## 3. Razonamiento

- Se descartó implementar la "aproximación" como eigendescomposición
  truncada de $\Delta_1$ (quedarse solo con los autovectores de autovalor
  ~0): aunque es la definición canónica de proyección armónica, calcular
  autovectores completos es igual de costoso que la pseudo-inversa exacta
  ($O(|E|^3)$), así que no sería una aproximación más barata, solo un método
  exacto distinto.
- Se eligió CG sobre los sistemas normales dispersos porque es exactamente
  la técnica que se usa en la práctica para este problema en grafos grandes
  (evita nunca materializar una matriz densa de tamaño $|E|^2$ o $|T|^2$), y
  porque permite medir *directamente* el trade-off precisión/velocidad
  variando la tolerancia `tol` del solver — variable natural para un
  "framework de benchmarking".
- Los tamaños de grafo para el método exacto se acotaron deliberadamente
  (ver nota en el código): el número de triángulos en $G(n,p)$ crece como
  $O(n^3 p^3)$, y una primera corrida con $p=0.3$ y $n=320$ generó una
  matriz densa de ~143.000×143.000 para el curl — intratable en memoria.
  Se documenta esto como hallazgo empírico en sí mismo: **es precisamente la
  razón por la que el método exacto no escala** y por la que tiene sentido
  la pregunta del enunciado.
- Se reporta también `harmonic_norm_ratio` ($\|h\|/\|X\|$) como control: en
  un grafo aleatorio suficientemente denso en triángulos, se espera que una
  fracción sustancial del flujo aleatorio caiga en la componente armónica
  (no hay razón estructural para que un flujo puramente aleatorio sea
  mayormente gradiente o curl), lo cual efectivamente se observa (~0.4–0.65
  en las corridas).

## 4. Desarrollo y código

Código completo en `hodge_benchmark.py` (adjunto en esta misma carpeta).
Estructura:

1. `build_complex(n, p, seed)`: genera $G(n,p)$ conexo, construye $B_1$, $B_2$.
2. `hodge_exact(B1, B2, X)`: descomposición vía pseudo-inversa densa.
3. `hodge_approx(B1, B2, X, tol, maxiter)`: descomposición vía CG disperso.
4. `run_benchmark(sizes, p, tol, seed)`: corre ambos métodos para una lista
   de tamaños de grafo y reporta tiempos, error relativo de cada componente,
   y el ratio $\|h\|/\|X\|$.

### Corrida interna de sanity-check (no es la comprobación final del alumno)

```
    n    |E|    |T|  t_exacto(s)  t_aprox(s)  speedup      err_h   err_grad   err_curl   ‖h‖/‖X‖
------------------------------------------------------------------------------------------------
   15     18      1      0.00053     0.00087     0.61   1.41e-12   8.30e-13   0.00e+00     0.407
   25     46      7      0.00031     0.00079     0.40   4.50e-09   3.79e-09   2.88e-16     0.531
   40    110     27      0.00071     0.00119     0.60   7.36e-09   3.46e-09   8.72e-09     0.598
   60    254    102      0.00317     0.00152     2.09   1.08e-08   8.80e-09   9.15e-09     0.642
   90    592    373      0.03965     0.00285    13.90   2.75e-08   6.89e-09   1.80e-08     0.515
```

**Esto es una corrida de referencia para verificar que el código no tiene
errores obvios, no la comprobación final** — esa la hacés vos (ver Sección 6).

## 5. Conclusión

Con esta única corrida (semilla fija, `p=0.15`), el patrón observado es
consistente con lo esperado teóricamente:

- El **error relativo** de las tres componentes (gradiente, curl, armónica)
  se mantiene en el orden de $10^{-8}$–$10^{-9}$, es decir, cerca de la
  tolerancia pedida al CG (`tol=1e-8`), y **no muestra una tendencia
  creciente con $n$** en este rango. Esto sugiere que el método aproximado
  aísla la componente armónica de forma numéricamente confiable.
- El **speedup** (tiempo exacto / tiempo aproximado) pasa de 0.61x en el
  grafo más chico (donde el overhead de CG domina) a 13.9x en el más grande
  del rango probado, consistente con la diferencia de complejidad
  $O(n^3)$ (denso) vs. cuasi-lineal por iteración (disperso).
- El punto más importante, sin embargo, es negativo por diseño: el intento
  inicial con grafos más grandes y más densos ($n=320$, $p=0.3$) **hizo
  fallar el método exacto por falta de memoria** (matriz de curl densa de
  ~143.000×143.000). Esto no es un error del código sino el resultado
  esperado: confirma empíricamente que el método exacto es intratable
  exactamente en el régimen donde el método aproximado sería más necesario.
  No se ocultó esta falla; se la usó para acotar el rango de benchmarking.

**Estas conclusiones son válidas únicamente para la corrida con semilla
fija incluida arriba.** No alcanzan por sí solas para una afirmación
general sobre el comportamiento en todos los regímenes — para eso hace
falta correr el framework con distintas semillas, densidades $p$ y
tolerancias, ver la Sección 6.

## 6. Instrucciones para que el alumno corra el código

```bash
pip install numpy scipy networkx --break-system-packages
python3 hodge_benchmark.py
```

Sugerencias de exploración (no obligatorias, para verificar por cuenta
propia si las conclusiones de la Sección 5 se sostienen):

- Cambiar `seed` en el `if __name__ == "__main__":` y ver si el patrón de
  error/speedup se mantiene, o es un artefacto de una única corrida.
- Aumentar `tol` (p. ej. `1e-4`) y ver cómo crece `err_h` — el trade-off
  precisión/velocidad debería volverse más marcado.
- Probar con `p` más alto (más denso en triángulos) para tamaños moderados
  de `n`, y confirmar en qué punto el método exacto empieza a fallar por
  memoria en tu propia máquina (el límite depende de la RAM disponible).
- Extender `hodge_approx` para reportar el número de iteraciones de CG
  usadas, y correlacionarlo con el *spectral gap* de $B_1B_1^T$ y
  $B_2^TB_2$ (grafos peor condicionados deberían necesitar más
  iteraciones).
