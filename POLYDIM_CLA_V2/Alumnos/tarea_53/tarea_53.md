# Tarea 53 — Restricciones SMT (Z3) sobre la inicialización de pesos de una red

## 1. Enunciado

**Original (banco maestro, Escuadrón 3 — Test 53):**

> "Actúa como especialista en OCaml/Haskell. Integra la librería Z3 (SMT solver) para
> demostrar cómo resolver restricciones sobre la inicialización de pesos de una red."

**Nota sobre el enunciado:** a diferencia de otros ítems del banco (que combinan
fragmentos de plantillas y quedan gramaticalmente rotos), este enunciado es
coherente y se entiende bien. Lo único que falta especificar es **cuáles son
las restricciones** sobre la inicialización de pesos — el enunciado no las
define. Elegí un conjunto concreto, estándar en la literatura de
inicialización de redes, para que el problema sea investigable:

1. **Preservación de varianza** (Glorot & Bengio, 2010 — inicialización
   "Xavier"): la energía total de la matriz de pesos debe corresponder a una
   varianza objetivo determinada por `fan_in` y `fan_out`.
2. **Cuasi-ortogonalidad de columnas** (Saxe, McClelland & Ganguli, 2013 —
   "dynamical isometry"): las columnas de `W` deben ser aproximadamente
   ortonormales, dentro de una tolerancia `ε`.
3. **Cota por entrada**: cada peso individual está acotado en magnitud.

El SMT solver (Z3) se usa para responder, de forma *simbólica y exacta* (no
por muestreo aleatorio como hace un inicializador estándar de PyTorch), si
existe una matriz de pesos que satisfaga las tres restricciones
simultáneamente, y para explorar cómo escala el costo de esa búsqueda al
crecer la dimensión de la matriz.

## 2. Modelo matemático

Sea $W \in \mathbb{R}^{n \times m}$ la matriz de pesos de una capa lineal,
con $n$ = `fan_in` y $m$ = número de columnas (unidades de salida
proyectadas; en este experimento tomamos $n=m$ para simplificar).

**Restricción de energía (Glorot/Xavier):**

$$\sum_{r=1}^{n}\sum_{c=1}^{m} W_{rc}^2 \;=\; \frac{2}{\text{fan\_in}+\text{fan\_out}} \cdot n \cdot m$$

**Restricción de cuasi-ortogonalidad de columnas (Saxe et al.):** para cada
columna $i$,

$$\left| \sum_{r} W_{ri}^2 - 1 \right| \le \varepsilon$$

y para cada par de columnas $i \neq j$,

$$\left| \sum_{r} W_{ri} W_{rj} \right| \le \varepsilon$$

**Cota por entrada:**

$$|W_{rc}| \le b \quad \forall r,c$$

Este es un **CSP (Constraint Satisfaction Problem) sobre los reales**, no
lineal (las restricciones de energía y ortogonalidad son cuadráticas en las
variables $W_{rc}$). Z3 lo resuelve con su motor `nlsat` (aritmética real no
lineal), a diferencia de SAT/SMT lineal estándar.

## 3. Razonamiento

- **Por qué Z3 y no un inicializador aleatorio estándar**: un inicializador
  como `torch.nn.init.orthogonal_` genera *una* muestra que aproximadamente
  cumple ortogonalidad, pero no garantiza — ni puede certificar — que sea la
  única solución posible, ni explora el espacio de soluciones factibles de
  forma exhaustiva. Un SMT solver responde de manera decidible (sat / unsat /
  unknown) si el conjunto de restricciones tiene *alguna* solución, lo cual
  es útil como herramienta de verificación formal, no como método de
  entrenamiento en producción.
- **Por qué el caso 2×2 y 3×3, y no una red real**: las restricciones son
  polinomios cuadráticos, y el problema general de decidir satisfacibilidad
  de restricciones polinomiales sobre los reales (teoría de Tarski) tiene
  complejidad doblemente exponencial en el peor caso. Matrices de tamaño
  realista (cientos o miles de parámetros) son completamente inviables para
  un SMT solver de propósito general — por eso el experimento se diseña para
  **mostrar el techo de escalabilidad**, no para reemplazar la
  inicialización real de una red.
- **Alternativas descartadas**: podría plantearse el problema como
  optimización continua (mínimos cuadrados con penalización de
  ortogonalidad, resuelta con descenso de gradiente) — eso es exactamente lo
  que hacen los métodos de inicialización reales, y es tratable a cualquier
  escala. La diferencia con el enfoque SMT es que gradiente da un óptimo
  aproximado sin garantía de existencia de solución exacta, mientras que Z3
  da una respuesta binaria certificada (sat/unsat) para el sistema de
  restricciones tal como está planteado. El enunciado pide explícitamente
  Z3, así que el experimento prioriza esa vía y documenta su costo.

## 4. Desarrollo y código

El script `tarea_53.py` (adjunto) construye el CSP descripto arriba con la
librería `z3-solver` de Python y lo resuelve para dos tamaños:

- Caso 2×2 (`fan_in = fan_out = 2`)
- Caso 3×3 (`fan_in = fan_out = 3`)

con un timeout de 10 segundos por caso, para poder documentar si el solver
converge, no converge (unknown) o determina infactibilidad (unsat).

**Instrucciones para correrlo vos mismo:**

```bash
pip install z3-solver
python3 tarea_53.py
```

Vas a ver, para cada caso, si el estado es `sat` (y en ese caso la matriz de
pesos concreta que satisface las tres restricciones), `unsat` (no existe tal
matriz con los parámetros elegidos) o `unknown` (el solver no terminó dentro
del timeout — indicio de que el costo del nlsat está explotando).

## 5. Conclusión (a completar por vos con tu propia corrida)

No ejecuté el script como parte de esta entrega — eso te toca a vos, para que
veas el resultado con tus propios ojos (así lo pide la metodología del
curso). Lo que sí puedo anticipar, en base a la teoría:

- El caso 2×2 es razonablemente tratable para `nlsat` y debería resolver en
  segundos.
- El caso 3×3 ya tiene 9 variables reales con restricciones cuadráticas
  acopladas (energía global + 3 normas de columna + 3 productos internos
  cruzados) — es plausible que tarde sensiblemente más, o incluso llegue a
  `unknown` con el timeout dado. Si eso pasa, **no es un error del script**:
  es el resultado esperado y documentable de la explosión combinatoria del
  álgebra real no lineal a medida que crece la dimensión — el propio
  fenómeno que el enunciado pedía demostrar.
- Si tu corrida da `sat` en ambos casos, fijate el valor de `ε` usado
  (0.15): es una tolerancia relativamente laxa. Achicarlo (por ejemplo a
  0.05) es un buen experimento adicional para ver en qué punto el sistema
  pasa a `unsat` o a `unknown`.

Ajustá `eps_orth`, `bound` y `timeout_ms` en el script y volvé a correrlo si
querés explorar el punto exacto donde cambia el resultado — esa exploración
activa es la que da el aprendizaje real acá, no la lectura pasiva de este
documento.

## 6. Resumen estructurado

```json
{
  "alumno": null,
  "tarea": 53,
  "tipo_problema": "CSP no lineal sobre reales (SMT) para verificación de esquemas de inicialización de pesos",
  "estado": "a_verificar_por_el_alumno",
  "resultado_resumen": "Se modela la inicialización de pesos (Glorot + cuasi-ortogonalidad Saxe et al. + cota por entrada) como un CSP cuadrático sobre reales, resuelto con Z3 (nlsat) para matrices 2x2 y 3x3. El script no fue ejecutado como parte de la entrega; el alumno debe correrlo y documentar sat/unsat/unknown.",
  "metrica_estandar_usada": "Satisfacibilidad SMT (sat/unsat/unknown) + norma L2 de columnas para cuasi-ortogonalidad"
}
```
