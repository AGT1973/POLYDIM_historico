# Nota técnica — Evaluación de SPIDER aplicado a ATTEND
**Proyecto:** POLYDIM · **Base documental real:** `POLYDIM_CONSTITUCION_FINAL_FINAL.docx` (Artículos 4, 9, 11, 14 y 15)

---

## 1. ¿ATTEND tiene un patrón de acceso fijo? (Paso 2)

La Constitución define `ATTEND` de forma mínima, en el glosario del Artículo 4: *"primitiva de implementación que actualiza el estado s mediante un mecanismo de atención (softmax)"*. No hay una restricción explícita sobre qué posiciones puede consultar cada posición.

Matemáticamente esto importa porque en cualquier mecanismo de atención softmax, el peso `α_ij` entre la posición `i` y la posición `j` se calcula en tiempo de ejecución a partir del contenido:

```
α_ij = softmax_j( q_i · k_j / √d )
```

`q_i` y `k_j` son vectores que dependen de los datos de entrada, no de la posición en abstracto. Esto tiene una consecuencia directa: **no existe ningún subconjunto de índices `j` que pueda descartarse de antemano**, porque hasta que no se calculan `q_i · k_j` para todo `j`, no se sabe cuáles pesos serán relevantes. Es atención completa, sin ventana ni máscara fija.

**Ejemplo cotidiano:** pensá en un profesor que, al corregir un examen, decide primero quién copió de quién *comparando todas las respuestas entre todos los alumnos*. No puede decidir de antemano "solo voy a comparar al alumno 3 con el 4 y el 5" — necesita ver el contenido de las 30 respuestas para saber cuáles se parecen. Ese es ATTEND tal como está definido: no hay vecindario fijo, hay comparación total dependiente del contenido.

**Conclusión del Paso 2:** ATTEND, tal cual está en la Constitución, es atención completa sobre todo el vector de estado — no tiene patrón fijo conocido antes de ejecutar. Por lo tanto, según el criterio de la Sección 3.2/3.3 de la consigna, **SPIDER no aplica directamente**.

---

## 2. Definición formal de ATTEND_local (Paso 3)

La Constitución sí formaliza una notación operacional de tipo big-step, pero no en el lugar que asumía la consigna: está en el Artículo 11 (Semántica Operacional del modelo OSCAR), con la forma `⟨instrucción, s⟩ → s′` (flecha simple, no `⇓`). Uso esa notación real, no una inventada:

**Regla base — ATTEND completo (tal como está en la Constitución):**

```
⟨ATTEND(Q,K,V), s⟩ → s[ i ↦ Σ_{j=1..N} softmax_j(q_i·k_j/√d) · v_j ]   para toda posición i ∈ {1..N}
```

**Regla propuesta — ATTEND_local(Q,K,V,w):**

```
⟨ATTEND_local(Q,K,V,w), s⟩ → s[ i ↦ Σ_{j ∈ N_w(i)} softmax_j(q_i·k_j/√d) · v_j ]   para toda posición i ∈ {1..N}

donde N_w(i) = { j ∈ {1..N} : |i − j| ≤ w }
```

La única diferencia formal con la regla base es el conjunto de índices sobre el que se suma: `{1..N}` completo versus `N_w(i)`, un vecindario de tamaño fijo `2w+1` centrado en `i` (o solo hacia atrás, `{i−w,...,i}`, si se quiere la variante causal). Ese conjunto `N_w(i)` **se puede calcular sin mirar los datos** — depende solo de `i` y `w`, que se conocen antes de ejecutar. Ahí es donde aparece el patrón fijo que SPIDER necesita.

---

## 3. Transformación de reordenamiento de memoria (Paso 4)

Como `N_w(i)` es idéntico en forma para cada `i` (siempre es "los `w` vecinos a cada lado"), las filas de `K` y `V` que una posición `i` va a necesitar están definidas por un desplazamiento fijo respecto a `i`, no por el contenido. Esto permite, antes de ejecutar, reorganizar `K` y `V` en bloques contiguos de tamaño `2w+1` alineados con cada posición — igual que se empaquetan los píxeles vecinos de una convolución en una unidad de hardware tipo *stencil*. La unidad de hardware que aprovecha SPIDER puede entonces leer un bloque contiguo de memoria por posición, en lugar de saltar a `N` direcciones dispersas decididas en tiempo de ejecución.

**Ejemplo cotidiano (el bibliotecario, aplicado ahora a la solución):** una vez que la biblioteca decide que cada lector va a mirar solo los libros del mismo estante y los dos estantes vecinos, el bibliotecario puede pre-ordenar esos tres estantes juntos, de antemano, sin esperar a ver qué pidió cada lector. Eso es exactamente lo que gana SPIDER con `ATTEND_local`: puede pre-empaquetar el vecindario porque el vecindario ya no depende de la demanda.

---

## 4. Tabla de complejidad verificada (Paso 5)

Costo de calcular `α_ij` para una posición `i`: un producto punto `q_i·k_j` cuesta `O(d)` (dimensión del embedding). Full attention hace ese producto contra los `N` posibles `j`; `ATTEND_local` lo hace contra los `2w+1 ≈ w` vecinos.

| Variante | Comparaciones por posición | Costo por posición (con `d`) | Costo total (`N` posiciones) |
|---|---|---|---|
| ATTEND completo | `N` | `O(N·d)` | `O(N²·d)` |
| ATTEND_local (ventana `w`) | `w` | `O(w·d)` | `O(N·w·d)` |

La tabla de la consigna omite el factor `d` (lo trata como constante), pero la razón `N²` vs `N·w` se sostiene igual: si `w ≪ N` (por ejemplo `w=256` con `N=100 000`), el costo total pasa de escalar con el cuadrado de la longitud de secuencia a escalar linealmente en ella.

---

## 5. Qué se pierde y si es aceptable (Paso 6)

**Qué se pierde, en concreto:** cualquier relación directa entre dos posiciones `i` y `j` con `|i−j| > w` deja de ser representable en una sola invocación de `ATTEND_local`. Esto no es "pérdida de expresividad" en abstracto — es la imposibilidad concreta de que, por ejemplo, la primera palabra de un documento influya en la última en un solo paso de atención, si el documento es más largo que `w`. Esa dependencia de largo alcance solo podría reconstruirse indirectamente, apilando varias capas de `ATTEND_local` (cada capa extiende el alcance efectivo en `w` posiciones) o combinándolo con `FIXPOINT`/`RECUR` para propagar información entre ventanas a lo largo de varias iteraciones — con el costo adicional de que esa propagación ya no es un solo salto directo, sino una cadena de saltos cortos.

**¿Es aceptable?** Sí, con dos razones tomadas directamente de la Constitución, no supuestas:

1. La Regla R11 (Artículo 14) es explícita: *"Las primitivas ATTEND y RECUR pertenecen estrictamente a la capa de implementación. Ninguna versión del lenguaje puede integrarlas al núcleo algebraico invariante."* Modificar la forma interna de ATTEND (agregarle una ventana) no toca `COMPOSE`, `MIX` ni `FIXPOINT`, que son el núcleo. El Artículo 15 refuerza esto llamando a esta capa *"voluntariamente volátil"*, reemplazable sin alterar el álgebra fundamental.

2. Hay un dato adicional que cambia el peso de esta decisión, y que no aparece en la consigna original: el Artículo 9 de la Constitución ya afirma, con un "Teorema de Imposibilidad" propio (Teorema 4.1), que la atención Softmax **no puede** satisfacer la factorización monoidal estricta que POLYDIM exige para tareas compositivas — independientemente de si tiene ventana o no. Es decir: `ATTEND` ya cargaba una limitación estructural reconocida por el propio documento antes de esta tarea. Restringirla a una ventana fija no la hace "menos compositiva" de lo que ya se declaraba que era; es un cambio ortogonal a ese problema.

---

## 6. Clasificación final

La propuesta tiene dos partes con estatus normativo distinto:

- **La formalización de `ATTEND_local` y la reorganización de memoria tipo SPIDER (Secciones 2 y 3 de esta nota): [MECANISMO].** Es matemática estándar (ventana local de atención, ya usada en arquitecturas conocidas), no depende de ninguna hipótesis sin validar, no toca el núcleo algebraico (Regla R11), y se puede especificar directamente en el backlog de ingeniería.

- **La interacción entre `ATTEND_local` y el Teorema de Imposibilidad de Softmax del Artículo 9 (si la ventana fija ayuda, empeora, o es irrelevante para el problema de "descenso a cocientes algebraicos"): [INVESTIGACIÓN].** Esto no está resuelto en la Constitución ni se deduce de la matemática presentada acá; es una hipótesis abierta que requeriría trabajo adicional antes de citarse como ley, en línea con la Regla R13 (ninguna formalización de investigación especulativa puede usarse como vinculante).

---

**Nota sobre las citas:** esta entrega usa los números de artículo reales del documento (`POLYDIM_CONSTITUCION_FINAL_FINAL.docx`), no los que traía la consigna original (que citaba Art. IV.2, V, X.4, XVI y XVII para contenidos que en realidad están en los Artículos 4, 9, 11, 14 y 15). Esa discrepancia se señaló y se resolvió en el turno anterior de esta conversación, por decisión tuya.

*que has usado el formato de razonamiento adaptado por AGT*
