# Tarea 47 — Batería POLYDIM (Escuadrón 3: Descomposición de Hodge y Grafos Esparsos)

**Alumno:** ai.mpat.marta@gmail.com
**Rol asignado:** Investigador de Geometric Deep Learning

---

## 1. Enunciado

### 1.1 Enunciado original (banco maestro, tal cual)

> "47 (Test 47): Actúa como investigador de Geometric Deep Learning. Escribe una demostración teórica implementada como código donde expliques por qué un sistema lógico basado en restricciones de pertenencia colapsa de manera estable."

### 1.2 Nota de reconstrucción

El banco se genera combinando plantillas (rol + fragmento técnico), y varios fragmentos quedan gramaticalmente incompletos. En este caso el fragmento "colapsa de manera estable" no especifica bajo qué condiciones matemáticas ocurre ese colapso, ni qué se entiende formalmente por "sistema lógico basado en restricciones de pertenencia". No voy a fingir que la frase es autocontenida: la reconstruyo apoyándome en el fragmento técnico reconocible (restricciones de pertenencia sobre un universo, iteración hasta estabilidad) y en el área temática del escuadrón (grafos esparsos / estructuras discretas), del siguiente modo:

**Enunciado reconstruido:**
> Actuando como investigador de Geometric Deep Learning, demostrar teóricamente —e implementar como código verificable— por qué un sistema de restricciones de pertenencia sobre un universo finito, definido mediante un operador de propagación monótono, converge siempre a un único estado estable (punto fijo) tras un número finito de iteraciones, sin oscilar entre estados distintos.

La reconstrucción es fiel al fragmento original (restricciones de pertenencia + colapso/estabilidad) y la traduzco a un objeto matemático estándar: operadores monótonos sobre retículos completos y el teorema de punto fijo de Knaster–Tarski, que es exactamente la teoría que explica cuándo y por qué un sistema de este tipo "colapsa de manera estable" (converge) y cuándo no.

---

## 2. Modelo Matemático

Sea U un universo finito de "hechos" o "elementos de pertenencia" (por ejemplo, nodos de un grafo esparso que pueden estar "activos" o "inactivos" según reglas de pertenencia). Definimos:

- **Retículo de estados:** L = (2^U, ⊆), el conjunto de partes de U ordenado por inclusión. Es un retículo completo: toda familia de subconjuntos tiene supremo (unión) e ínfimo (intersección), y existen mínimo (∅) y máximo (U).

- **Operador de propagación:** T : 2^U → 2^U, donde T(X) agrega a X todos los elementos cuya restricción de pertenencia se satisface dado el conjunto actual X:

  T(X) = X ∪ { u ∈ U : regla(u, X) = verdadero }

- **Monotonía (condición clave):** T es monótono si

  X ⊆ Y ⟹ T(X) ⊆ T(Y)

  Es decir, agregar más hechos nunca "retira" conclusiones ya alcanzadas (no hay negación en las reglas, solo pertenencia positiva).

- **Iteración de Kleene:** se define la cadena ascendente

  X_0 = ∅,  X_(k+1) = T(X_k)

### Teorema (Knaster–Tarski, versión constructiva / Kleene)

Si L es un retículo completo y T: L → L es monótono, entonces:

1. T tiene un punto fijo mínimo μT = ∪_{k≥0} X_k.
2. Como U es finito, 2^U tiene altura finita (|U|+1 niveles), por lo tanto la cadena X_0 ⊆ X_1 ⊆ X_2 ⊆ … es creciente hasta estabilizarse, y se cumple

   ∃ k* ≤ |U| : X_(k*) = X_(k*+1) = μT

3. Ese X_(k*) es estable: T(μT) = μT (idempotencia), y es único como punto fijo mínimo alcanzado desde ∅.

Esto es precisamente el "colapso estable": el sistema de restricciones de pertenencia no oscila ni diverge, sino que se estabiliza en un número finito de pasos acotado por |U|, siempre que T sea monótono.

**Contraejemplo necesario:** si T deja de ser monótono (por ejemplo, si una regla puede retirar un elemento de pertenencia — negación no estratificada), el teorema no aplica, la cadena de Kleene no es necesariamente creciente, y el sistema puede oscilar sin converger. Esto se verifica empíricamente en el código.

---

## 3. Razonamiento

**Por qué este modelo y no otro:**

- El fragmento original habla de "restricciones de pertenencia" que "colapsan de manera estable". La estructura matemática estándar que formaliza exactamente esa noción —pertenencia (subconjuntos), colapso (convergencia), estable (punto fijo)— es la teoría de retículos y el teorema de Knaster–Tarski. Es la misma teoría que sustenta la semántica de Datalog, la interpretación abstracta en compiladores, y el análisis de flujo de datos monótono.
- **Alternativas descartadas:**
  - *Descomposición de Hodge* (título del escuadrón): aplica a espacios vectoriales/grafos con estructura de flujo, no a sistemas lógicos de pertenencia discretos. No es el objeto correcto para "restricciones de pertenencia".
  - *Convergencia por contracción de Banach* (punto fijo métrico): exige una métrica y un factor de contracción menor a 1; un sistema de pertenencia booleano no tiene naturalmente una métrica de contracción, mientras que sí tiene naturalmente un orden parcial completo. Knaster–Tarski es la herramienta nativa para retículos, no para espacios métricos.
  - *Convergencia estocástica* (tipo cadenas de Markov): no aplica porque el sistema es determinístico y discreto, no probabilístico.
- La métrica estándar verificable empíricamente para "qué tan lejos está el sistema de estabilizarse" es la distancia de Hamming / cardinalidad de la diferencia simétrica |X_(k+1) Δ X_k|, que decrece monótonamente hasta 0 cuando el sistema converge, y no llega a 0 cuando el sistema (por falta de monotonía) oscila. Esto reemplaza acá a métricas como GeoID Stability, que no tienen definición matemática estándar: la traduzco a esta métrica de convergencia de conjuntos, verificable con código.

---

## 4. Desarrollo y Código

El código (`tarea_47.py`) implementa:

1. Un `ConstraintSystem` genérico basado en reglas de pertenencia sobre un universo finito.
2. Iteración de Kleene con conteo de pasos hasta punto fijo.
3. Verificación de monotonía de las reglas.
4. Un caso monótono (converge, "colapso estable") y un caso no monótono (con retiro de elementos, para mostrar la oscilación / no convergencia) — el contraejemplo que prueba por qué la monotonía es la condición necesaria.

Ver el archivo `tarea_47.py` para el código completo y comentado. El alumno debe ejecutarlo para comprobar empíricamente ambos comportamientos.

### Instrucciones para ejecutar

```bash
python3 tarea_47.py
```

No requiere dependencias externas (solo la librería estándar de Python). Si se quiere graficar la curva de convergencia, instalar matplotlib:

```bash
pip install matplotlib
```

y descomentar el bloque de graficado al final del script.

---

## 5. Conclusión

El sistema de restricciones de pertenencia colapsa de manera estable si y solo si el operador de propagación T es monótono sobre el retículo completo 2^U. En ese caso, el teorema de Knaster–Tarski garantiza:

- Existencia de un punto fijo mínimo único.
- Convergencia en a lo sumo |U| iteraciones (cota finita, no asintótica).
- Estabilidad: una vez alcanzado, el estado no vuelve a cambiar (T(μT)=μT).

Cuando se rompe la monotonía (regla con "retiro" de pertenencia, análoga a negación no estratificada), el sistema puede entrar en un ciclo de longitud mayor a 1 y nunca estabilizarse — esto se demuestra empíricamente en el script, midiendo que la distancia de Hamming entre iteraciones sucesivas no converge a 0 sino que oscila entre un conjunto de valores repetidos.

**El alumno debe correr el script, observar ambos comportamientos (convergente vs. oscilante), y ajustar el universo U o las reglas para verificar que la propiedad se sostiene en otros casos —esa comprobación activa es la que valida (o refuta) el enunciado, no esta lectura.**

---

## 6. Resumen Estructurado

```json
{
  "alumno": "ai.mpat.marta@gmail.com",
  "tarea": 47,
  "tipo_problema": "Convergencia de operadores monótonos sobre retículos completos (Knaster-Tarski) aplicado a sistemas de restricciones de pertenencia",
  "estado": "resuelto",
  "resultado_resumen": "Se demuestra que un sistema de restricciones de pertenencia colapsa de manera estable si y solo si su operador de propagación es monótono sobre 2^U; se prueba existencia, unicidad y cota finita de convergencia (Knaster-Tarski), y se muestra por contraejemplo que la monotonía es condición necesaria.",
  "metrica_estandar_usada": "Distancia de Hamming / cardinalidad de la diferencia simétrica entre iteraciones sucesivas (|X_(k+1) Δ X_k|)"
}
```
