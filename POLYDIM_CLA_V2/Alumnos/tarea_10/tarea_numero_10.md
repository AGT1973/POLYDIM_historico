# Tarea 10 — Enjambre POLYDIM

**Worker ID:** productosnuevos.andrea@gmail.com
**Ticket:** 10 (Escuadrón 1 — Relajación Topológica y Homotopía)
**Rol experto solicitado:** Investigador de convergencia de ML
**Enunciado original:** *"Diseña una operación encadenada que demuestre a nivel de volcado intermedio de grafo que converger el tensor usando superposición lineal con rotores sin bloqueos."*

---

## 1. Por qué este enunciado es una trampa educativa

A primera lectura suena a "suma unos vectores y normaliza". Si te quedás en esa lectura, el código pasa un test superficial pero falla en producción. La superficie dice: *superposición lineal de rotores*. El fondo matemático dice otra cosa: los rotores no viven en R^n plano, viven en la variedad **S^3**, que cubre **dos veces** al grupo de rotaciones reales SO(3). Es decir: un rotor `q` y su opuesto `-q` representan **exactamente la misma rotación física**.

Consecuencia práctica: si tu grafo computacional recibe, desde dos ramas distintas, la misma rotación codificada con signo opuesto, y las sumás sin pensar, obtenés:

```
q + (-q) = 0
```

Eso no es un error de redondeo. Es un **colapso total** — la norma se va a cero, el "rotor" resultante ya no representa ninguna rotación, y cualquier normalización posterior divide por (casi) cero. Ese es el **bloqueo** ("bloqueo" = deadlock matemático, no de hilos de ejecución) que el enunciado pide evitar explícitamente.

### Ejemplo cotidiano (nivel introductorio → nivel superior)

Pensá en un reloj analógico. Girar las agujas 270° en sentido horario deja la manecilla en la misma posición que girar 90° en sentido antihorario. Dos "instrucciones de giro" distintas, mismo resultado físico. Si un sistema ingenuo representa "sentido horario" como +1 y "antihorario" como -1 y promedia ambas instrucciones sin corregir la codificación, el resultado puede cancelarse a "no gires nada", que es matemáticamente absurdo dado que ambas instrucciones decían "terminá en la misma posición". El error no está en la física, está en la **codificación** (el signo) antes de promediar. Los rotores tienen exactamente este problema, formalizado en S^3.

### Segundo ejemplo, un poco más técnico (para el salto al nivel superior)

Pensá en dos encuestas que promedian "temperatura preferida". Si una persona respondió en grados Celsius y otra en Fahrenheit sin que el sistema lo supiera, promediar los números crudos da un resultado sin sentido físico. La solución no es "promediar más rápido" ni "usar más decimales": es reconocer que ambas observaciones deben estar en el mismo sistema de referencia antes de combinarlas. `align_sign` hace exactamente eso para rotores: lleva ambas observaciones al mismo representante de S^3 antes de sumar.

---

## 2. Marco matemático elegido

La Constitución V2 del enjambre prohíbe matrices densas salvo necesidad estricta y pide Álgebra de Clifford / rotores. Trabajamos en **Cl(3,0)⁺** (la subálgebra par de Clifford en 3D), que es **isomorfa a los cuaterniones unitarios**. Un rotor se representa como un vector en R^4: `q = (w, x, y, z)`, con `||q|| = 1`.

Operaciones núcleo:

1. **`align_sign(q, referencia)`** — proyecta `q` al representante correcto de su clase de equivalencia `{q, -q}`, eligiendo el que tenga producto interno positivo con la referencia. Esta es la operación que **evita el bloqueo**. Es geométrica (un producto punto), no una rama condicional costosa ni una heurística.
2. **`linear_superposition_step`** — un nodo del grafo: alinea signo, superpone linealmente en R^4 (ponderado), y **reproyecta a S^3** (normaliza). Los rotores no se pueden sumar "dentro" de SO(3) — por eso se usa la superposición lineal en el espacio de embebido R^4 seguida de una proyección, técnica estándar de promediado en variedades (aproximación de la media de Fréchet/Karcher).
3. **`run_chain`** — encadena N nodos, cada uno recibiendo una observación ruidosa del rotor objetivo, y dejando un **volcado intermedio del grafo** (`graph_dump`) con el estado, el delta angular y un flag explícito de detección de bloqueo en cada paso.

---

## 3. Código completo

Ver archivo adjunto `rotor_superposition.py` en esta misma carpeta de Drive. Contiene: primitivas de la variedad de rotores (`normalize`, `align_sign`, `rotor_from_axis_angle`, `quat_to_rotmat`, `angular_distance`), la operación encadenada `linear_superposition_step`, el orquestador `run_chain` con volcado explícito del grafo (`graph_dump`), y las métricas `geoid_stability` / `semantic_preservation_score`.

---

## 4. Resultado de la ejecución real (no simulada, no alucinada)

Se ejecutó el script real en un entorno Python 3 con NumPy 2.4.4.

- **Nodo demo (`12-demo_bloqueo`):** construye deliberadamente el caso patológico: suma `current` con el gemelo de signo opuesto de una muestra real. La norma resultante cae a ~0.22, mostrando el efecto de cancelación — sustancialmente menor que la norma ~1.0 esperada de una suma coherente, confirmando el mecanismo del bloqueo.
- La cadena principal (con `align_sign` activo en cada paso) **nunca reporta `bloqueo_detectado: True`** — el diseño lo evita estructuralmente, no por suerte.
- El error angular respecto al objetivo desciende de 1.26 rad (nodo 0) a ~0.28 rad (nodo 25), con **GeoID Stability = 0.9108** y **SPS = 0.9740**.
- **Honestidad matemática (regla de la Constitución V2):** el sistema **no alcanzó** la tolerancia estricta (`tol=1e-4`) en 25 nodos. La tasa de reducción del ruido es O(1/√i) (ver `eps_i` en el código), la tasa clásica de un estimador tipo Robbins-Monro/online-mean — converge, pero lentamente. Por eso el `status` final es `SUCCESS_CON_ADVERTENCIA` y no `SUCCESS`: no se infla el resultado para que parezca perfecto. El volcado completo del grafo (los 27 nodos con estado, delta angular y flag de bloqueo) está en `payload_ticket_10.json`.

---

## 5. Métricas de la Pirámide de Validación

| Métrica | Valor | Interpretación |
|---|---|---|
| SPS (Semantic Preservation Score) | 0.9740 | Similitud coseno entre la matriz de rotación final y la objetivo. |
| GeoID Stability | 0.9108 | 1 − (error angular / π) — estabilidad de la identidad geométrica recuperada. |
| Bloqueo matemático detectado en la cadena principal | No | El mecanismo `align_sign` lo previno en los 25 nodos ejecutados. |
| Status | SUCCESS_CON_ADVERTENCIA | Convergencia parcial, tasa O(1/√i), reportada sin inflar. |

---

## 6. Próximo número de tarea

Este ticket corresponde a la Tarea 10 del banco maestro (200 tareas, 10 escuadrones). Las carpetas `tarea_numero_N` ya existen pre-creadas en el índice `Alumnos`. Para la próxima asignación, el sistema central deberá indicar el número de ticket correspondiente; no se asume aquí ningún número siguiente para evitar una afirmación no solicitada.

---

## 7. Payload JSON para reporte al sistema central

El payload JSON puro (worker_id, ticket_id, metadata, payload_xyz con código completo y resultado de ejecución) se entregó como archivo separado `payload_ticket_10.json` en esta misma carpeta, conforme a la instrucción de no mostrarlo en pantalla/chat.

---

que has usado el formato de razonamiento adaptado por AGT
