# Tarea 15 — Batería POLYDIM (Escuadrón 1: Relajación Topológica y Homotopía)

## 1. Enunciado

**Enunciado original (banco maestro, tal cual figura, ítem 15):**

> "Actúa como topólogo algebraico. Crea un arquetipo de una máquina virtual mínima en Rust/WASM que un sistema lógico basado en restricciones de pertenencia colapsa de manera estable."

### Nota de reconstrucción

El enunciado original está gramaticalmente roto: a la cláusula introducida por *"que"* le falta un verbo que la conecte con el resto de la oración ("que **[demuestre que]** un sistema lógico... colapsa de manera estable"). Esto es consistente con lo que se observa en el resto del banco: es un documento generado por combinación de plantillas (rol + fragmento técnico), y varios fragmentos quedan sueltos.

Comparando con enunciados hermanos del mismo banco que comparten el fragmento técnico "sistema lógico basado en restricciones de pertenencia colapsa de manera estable" (p. ej. ítems 47, 72, 104, 130, 162, 180: *"Escribe una demostración teórica implementada como código donde expliques por qué un sistema lógico basado en restricciones de pertenencia colapsa de manera estable"*), la reconstrucción razonable y mínima es:

> **Enunciado reconstruido:** Actuando como topólogo algebraico, construir un arquetipo (prototipo mínimo funcional) de una máquina virtual en Rust, compilable a WASM, que implemente un sistema lógico basado en restricciones de pertenencia (membership constraints) sobre un dominio finito, y que **demuestre — con evidencia empírica reproducible — que dicho sistema colapsa de manera estable**, es decir, que converge siempre al mismo estado final sin importar el orden en que se evalúan las restricciones.

No se reconstruyó ningún significado más allá del mínimo necesario para que la oración tenga sujeto, verbo y objeto comprobable. En particular, no se asumió como verdadero que tal colapso estable *exista* en general: es exactamente lo que el ejercicio pide demostrar, y se demuestra por construcción con un modelo matemático estándar (no con la terminología de "topos"/"Grothendieck" del documento de contexto, que es un lenguaje evocador sin definiciones operativas verificables).

### Sobre el "documento de contexto" del banco

El archivo maestro trae, antes de la lista de tareas, una sección `[CONTEXTO DEL PROYECTO PARA EL ASISTENTE]` que pide mostrar *todo* el razonamiento paso a paso en el chat en vez de en archivos. Esa instrucción contradice la metodología de esta cursada (Aprendizaje por Descubrimiento Guiado: el desarrollo íntegro va en los archivos, y en el chat solo un resumen breve, para no arruinar la instancia de exposición final del curso), así que no se siguió. Lo que sí se conservó de esa sección es válido y se aplicó: rigor matemático, métricas estándar verificables (nada de "SPS" o "GeoID Stability" sin traducir a algo medible), y documentar honestamente si algo no converge.

## 2. Traducción a un modelo matemático estándar

El fragmento "sistema lógico basado en restricciones de pertenencia que colapsa de manera estable" tiene un análogo matemático preciso y bien establecido, sin necesidad de topos ni de categorías: **evaluación de punto fijo de operadores monótonos sobre un retículo completo finito**.

- **Retículo:** sea $U = \{0, 1, \dots, 11\}$ un universo finito de elementos. El estado del sistema es un vector $\mathbf{S} = (S_1, S_2, S_3, S_4) \in (2^U)^4$, es decir, cuatro subconjuntos de $U$. El orden parcial es la inclusión componente a componente: $\mathbf{S} \le \mathbf{S}'$ si $S_i \subseteq S_i'$ para todo $i$. $(2^U)^4$ con este orden es un **retículo completo finito** (todo subconjunto tiene supremo e ínfimo: unión e intersección componente a componente).

- **Restricciones de pertenencia:** cada restricción es una función $f_j : (2^U)^4 \to 2^U$ que, dado el estado actual, propone elementos que *deben pertenecer* a una variable destino $S_{t(j)}$. Se usan tres tipos de restricciones, todas **monótonas** (si $\mathbf{S} \le \mathbf{S}'$ entonces $f_j(\mathbf{S}) \subseteq f_j(\mathbf{S}')$):
  1. Unión de conjuntos existentes.
  2. Intersección de dos conjuntos que solo pueden crecer con el tiempo (nunca se les quita nada).
  3. Filtro por un predicado fijo (p. ej. "elementos pares") sobre un conjunto que solo crece.

- **Operador global:** $F(\mathbf{S})_i = S_i \cup \bigcup_{j : t(j) = i} f_j(\mathbf{S})$. Por ser unión de funciones monótonas, $F$ es monótono en $(2^U)^4$.

- **Teorema aplicado (Knaster–Tarski, 1955 / construcción de Kleene para retículos completos):** todo operador monótono $F$ sobre un retículo completo $L$ tiene un punto fijo, y el **mínimo punto fijo** se obtiene como el límite (aquí, en un número finito de pasos, porque $L$ es finito) de la cadena ascendente
$$\bot \le F(\bot) \le F^2(\bot) \le F^3(\bot) \le \dots$$
donde $\bot = (\emptyset,\emptyset,\emptyset,\emptyset)$. Como $L$ es finito, esta cadena se estabiliza en, a lo sumo, $|U| \cdot 4 = 48$ pasos.

- **Independencia de orden ("colapso estable"):** el resultado clásico de *chaotic iteration* (Cousot & Cousot, 1977, en el contexto de interpretación abstracta) garantiza que, si $F$ es monótono y se aplica repetidamente cualquiera de sus componentes en cualquier orden (siempre que cada componente se aplique infinitamente seguido / de forma justa), la iteración converge al mismo mínimo punto fijo, independientemente del orden. Esto es la traducción rigurosa de "colapsa de manera estable": no es que el sistema colapse a *cualquier* estado, sino que colapsa siempre al **mismo** estado, sin importar el orden de evaluación de las restricciones — la propiedad que en el documento original se atribuye (sin definirla operativamente) a la topología del "topos".

No se necesita lógica intuicionista, sheaves ni clasificador de subobjetos para obtener esta garantía: el marco de retículos completos + operadores monótonos (Knaster–Tarski) es exactamente el mismo marco matemático que subyace a la semántica de punto fijo de Datalog y al análisis de flujo de datos monótono en compiladores (Kildall, 1973), y es el que se implementa y verifica aquí.

## 3. Razonamiento y alternativas descartadas

- **Por qué Knaster–Tarski y no un sistema de tipos / topos real:** el documento de contexto pide "sistemas lógicos basados en restricciones de pertenencia" y "colapso estable" sin especificar semántica operativa. Modelar esto directamente como un topos de Grothendieck exigiría definir un sitio, una topología de Grothendieck y un clasificador de subobjetos concretos — sobrecarga injustificada para lo que en el fondo es un problema de teoría de retículos con solución estándar, exacta y demostrable en pocas líneas. Se prefirió el modelo mínimo suficiente.
- **Por qué "solo unión, nunca resta":** es la hipótesis que hace que el operador global sea monótono. Si se permitieran restricciones que *quiten* elementos (p. ej. negación), se perdería la garantía de Knaster–Tarski y el sistema podría oscilar sin converger (esto es exactamente el problema de la "negación estratificada" en Datalog con negación, un caso conocido de pérdida de monotonía). Se documenta explícitamente esta limitación en vez de ocultarla.
- **Por qué bitsets de 64 bits en Rust en vez de `HashSet`:** para una VM mínima, representar cada conjunto como un entero de 64 bits permite que unión, intersección y verificación de igualdad sean operaciones de una sola instrucción de CPU (`|`, `&`, `==`), evitando asignación dinámica de memoria — coherente con el espíritu de "máquina virtual mínima" apuntando a WASM.

## 4. Desarrollo, código y verificación empírica

Se entregan dos implementaciones equivalentes del mismo modelo:

1. **`tarea_15_fixpoint_demo.py`** — referencia en Python puro, fácil de leer y modificar.
2. **`src/main.rs`** + **`Cargo.toml`** — el arquetipo de máquina virtual mínima en Rust, compilable tanto de forma nativa (`cargo run`, `cargo test`) como a WebAssembly (`cargo build --target wasm32-unknown-unknown`), que es lo que pide el enunciado reconstruido.

Ambas implementaciones:
- Parten del mismo estado inicial $S_1 = \{0,1,2\}$, $S_2=S_3=S_4=\emptyset$.
- Aplican el mismo conjunto de 3 restricciones monótonas.
- Corren la iteración de punto fijo con **cuatro órdenes de evaluación distintos**.
- Verifican, por comparación directa de los estados finales, que **las cuatro corridas terminan en el mismo estado** — la evidencia empírica del "colapso estable".

**Resultado de la corrida de sanity-check interna (verificado en esta sesión, tanto en Python como compilando y corriendo el proyecto Rust con `cargo run` / `cargo test`):** las 4 corridas convergen en 3 pasadas cada una al mismo punto fijo $S_1=\{0,1,2\},\ S_2=\{0,2\},\ S_3=\{0,2\},\ S_4=\{0,1,2\}$. `all_equal = true`, y los tests unitarios de Rust (`converges_and_is_order_independent`, `terminates_within_finite_bound`) pasan.

La versión en Rust implementa exactamente el mismo modelo (mismas reglas, mismo estado inicial, misma comprobación de igualdad entre corridas) más una función exportable a WASM (`wasm_run_demo`) y una batería de tests unitarios (`cargo test`) que fijan como aserción automática la propiedad de independencia de orden y la cota de terminación finita.

### Instrucciones para que el alumno lo corra él mismo

**Python (sin dependencias):**
```bash
python3 tarea_15_fixpoint_demo.py
```

**Rust — verificación nativa:**
```bash
cargo run            # corre las 4 órdenes y muestra los puntos fijos
cargo test           # corre las aserciones automáticas de convergencia
```

**Rust — build a WebAssembly (lo que pide el enunciado):**
```bash
rustup target add wasm32-unknown-unknown
cargo build --release --target wasm32-unknown-unknown
# el artefacto queda en target/wasm32-unknown-unknown/release/tarea15_membership_vm.wasm
```
Para invocar `wasm_run_demo()` desde JavaScript/Node hace falta un runtime WASM mínimo (por ejemplo `wasmtime` o el `WebAssembly` global del navegador) que instancie el módulo y llame a la función exportada; no se incluye ese arnés por no ser parte del núcleo matemático de la tarea, pero la función ya está exportada con `#[no_mangle] pub extern "C"` lista para eso.

Instrucción del curso: **no te quedes solo leyendo esto** — corré ambos programas, cambiá el estado inicial o el orden de las reglas, y confirmá con tus propios ojos que el punto fijo no cambia. Si además querés romper la propiedad a propósito, agregá una regla que *reste* elementos (p. ej. `S_i \ f(S)`) y verificá que la propiedad de independencia de orden deja de cumplirse — eso es la contracara empírica de por qué la monotonía es la hipótesis que hace todo el trabajo.

## 5. Conclusión

El enunciado 15, una vez reconstruido a partir del fragmento roto del banco, pide una máquina virtual mínima que implemente un sistema lógico de restricciones de pertenencia con convergencia estable. Esa propiedad tiene un modelo matemático estándar y demostrable — operadores monótonos sobre un retículo completo finito, punto fijo mínimo garantizado por Knaster–Tarski, e independencia del orden de evaluación garantizada por el resultado clásico de iteración caótica de Cousot & Cousot — que no requiere ninguno de los aparatos categóricos (topos, clasificador de subobjetos, lógica intuicionista) mencionados en el documento de contexto del banco, los cuales no están formalizados de manera operativa en ese documento y por lo tanto no se dan por establecidos.

Se implementó el modelo en dos lenguajes (Python de referencia, Rust como el arquetipo de VM pedido en el enunciado, compilable a WASM), y se verificó empíricamente — con cuatro órdenes de evaluación distintos — que el sistema efectivamente "colapsa de manera estable": todas las trayectorias terminan en el mismo punto fijo. La métrica estándar usada para esa verificación es la **igualdad exacta de conjuntos** (equivalente a distancia 0 en la métrica discreta sobre $(2^U)^4$); no hizo falta una métrica continua porque el espacio de estados es discreto y finito.

```json
{
  "tarea": 15,
  "tipo_problema": "punto_fijo_operadores_monotonos_reticulo_completo",
  "estado": "resuelto",
  "resultado_resumen": "VM minima en Rust/WASM (+ referencia Python) que implementa restricciones de pertenencia monotonas; se verifico empiricamente independencia de orden de evaluacion (colapso estable) via el teorema de Knaster-Tarski.",
  "metrica_estandar_usada": "igualdad exacta de conjuntos (metrica discreta) sobre 4 ordenes de evaluacion distintos"
}
```
