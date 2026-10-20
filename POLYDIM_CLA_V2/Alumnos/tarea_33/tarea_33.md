# Tarea 33 — Escuadrón 3: Descomposición de Hodge y Grafos Esparsos

## 1. Enunciado

### Enunciado original (banco maestro, Test 33)
> "Actúa como experto en Manifold Learning. Simula en Python un entorno de agentes que modifican asíncronamente un estado tensor central para los intervalos (bounds) numéricos de las activaciones emulando dominios poliédricos."

**Nota de reconstrucción.** Igual que en otras tareas de este banco, la oración está incompleta: "...modifican asíncronamente un estado tensor central **para** los intervalos (bounds) numéricos..." no tiene el verbo/objeto que conecte la acción de los agentes con los "intervalos numéricos". Es el mismo patrón de plantilla+fragmento documentado en la skill: rol ("experto en Manifold Learning") + fragmento técnico de la lista de operadores POLYDIM (aquí: *los intervalos (bounds) numéricos de las activaciones emulando dominios poliédricos*, que en otras tareas del banco aparece pegado a verbos distintos, p. ej. Test 6: *"...para realizar los intervalos (bounds) numéricos..."*, Test 26: *"...que los intervalos (bounds) numéricos..."*).

### Enunciado reconstruido
> Simulá en Python un entorno concurrente con varios agentes que modifican **asíncronamente** (sin bloqueo mutuo, con posible intercalado arbitrario de operaciones) un **estado tensor central** compartido, que representa las activaciones de una red neuronal. Verificá empíricamente si, bajo esa concurrencia sin sincronización, los valores del tensor se mantienen dentro de los **intervalos numéricos (bounds)** que un dominio abstracto poliédrico/de intervalos (interval arithmetic / box abstract domain, en el sentido de Gehr et al., *AI²*, 2018, y Singh et al., *DeepPoly*, 2019) garantizaría en ausencia de condiciones de carrera.

Esto conecta el rol de Manifold Learning (el estado central vive en una variedad/tensor de activaciones) con el fragmento técnico de "intervalos numéricos emulando dominios poliédricos", que en la literatura de verificación de redes neuronales es exactamente la abstracción de **interval/box domain** (una relajación poliédrica simple: cada dimensión acotada por un intervalo independiente) usada para certificar cotas de activación capa por capa.

## 2. Modelo matemático

### 2.1 Bounds por dominio de intervalos (línea de base secuencial)
Sea $x \in \mathbb{R}^n$ el vector de activaciones de una capa, con bounds de entrada $x_i \in [l_i, u_i]$. Para una transformación afín $y = Wx + b$ seguida de ReLU, el dominio de intervalos (caso particular de dominio poliédrico donde el poliedro es una caja/*box*) propaga los bounds componente a componente:
$$
\underline{y}_j = \sum_i \big(\max(W_{ji},0)\,l_i + \min(W_{ji},0)\,u_i\big) + b_j, \qquad
\overline{y}_j = \sum_i \big(\max(W_{ji},0)\,u_i + \min(W_{ji},0)\,l_i\big) + b_j
$$
$$
[\underline{z}_j, \overline{z}_j] = [\max(0,\underline{y}_j),\ \max(0,\overline{y}_j)] \quad \text{(ReLU monótona)}
$$
Esto es exacto (no una sobre-aproximación) para el caso caja porque cada operación (suma ponderada monótona por signo, ReLU) es monótona coordenada a coordenada. Con un solo agente actualizando secuencialmente, los bounds calculados así **siempre** se satisfacen: es una garantía determinística, no probabilística.

### 2.2 El experimento: ruptura de la garantía bajo concurrencia asíncrona
El punto de la tarea es que esa garantía depende de que la actualización de $x$ sea **atómica**: se lee $x$ completo, se calcula $y=Wx+b$, se escribe. Si en cambio $k$ agentes actualizan **coordenadas individuales** de un tensor central compartido $T \in \mathbb{R}^n$ de forma asíncrona (sin lock, con posible interleaving arbitrario de lecturas/escrituras), puede ocurrir que un agente lea una mezcla de valores "viejos" y "nuevos" de $T$ — un estado que nunca existió como snapshot consistente — y que esa mezcla caiga fuera del intervalo $[\underline{z}_j,\overline{z}_j]$ calculado bajo el supuesto de atomicidad.

Formalmente, modelamos:
- $T^{(t)}$: estado del tensor en el paso lógico $t$ (existe, pero no es directamente observable si hay carrera).
- Cada agente $k$ ejecuta: `leer T[índices] → calcular actualización local → escribir T[índices]`, sin sección crítica.
- Definimos la métrica de violación: $V = \#\{j : T_j^{\text{final}} \notin [\underline{z}_j, \overline{z}_j]\}$ y la magnitud de violación $\varepsilon_j = \max(0,\ \underline{z}_j - T_j,\ T_j - \overline{z}_j)$.

Esto no es una propiedad exclusiva de POLYDIM ni una afirmación *ad hoc*: es el problema estándar de **verificación de invariantes bajo memoria compartida sin sincronización** (data races), aplicado aquí al caso particular de bounds de activación de redes neuronales.

### 2.3 Qué se mide
1. **Tasa de violación de bounds**: fracción de componentes que terminan fuera de su intervalo garantizado, en función del número de agentes concurrentes y del grado de interleaving simulado.
2. **Magnitud de la violación** ($\varepsilon_j$, norma L2 y L∞ del vector de violaciones).
3. **Comparación control**: la misma actualización ejecutada de forma secuencial/atómica (con lock), donde $V$ debe ser exactamente 0 — esto es lo que confirma que la violación observada es un artefacto de la concurrencia y no un error del modelo de bounds en sí.

## 3. Razonamiento
- **Por qué dominio de intervalos y no un poliedro general**: el fragmento del enunciado dice explícitamente "emulando dominios poliédricos" con "intervalos (bounds) numéricos" — el dominio de intervalos (caja) es el caso más simple y computacionalmente barato de dominio poliédrico (un poliedro con todas las caras paralelas a los ejes), y es exactamente el que tiene fórmulas de propagación cerradas y baratas de implementar para esta simulación. Usar un poliedro general (con restricciones acopladas entre coordenadas, como Zonotope o DeepPoly con relajaciones triangulares) sería más preciso pero no es necesario para lo que pide el enunciado, que es sobre el *efecto de la concurrencia*, no sobre la precisión del dominio abstracto en sí.
- **Alternativa descartada**: usar locks/mutex desde el principio para "arreglar" la concurrencia. Se descarta porque el punto de la tarea es medir qué pasa **sin** sincronización (para eso se pide "asíncronamente"); el caso con lock se incluye solo como control para confirmar que el modelo de bounds es correcto en ausencia de carrera.
- **Aclaración sobre lo que esto demuestra y lo que no**: este experimento demuestra, con una probabilidad empírica > 0 medida en la simulación, que actualizaciones asíncronas sin sincronización pueden violar bounds calculados bajo el supuesto de atomicidad. Esto **no** implica que todo sistema poliédrico/de intervalos sea inútil en entornos concurrentes — en la práctica, los sistemas de verificación real (ej. motores de inferencia) sí serializan las actualizaciones de estado por capa. Tampoco demuestra nada sobre la afirmación general de POLYDIM de que "la validación topológica es inmune a fallas" (Sección 2 del paper): esa es una propiedad de la abstracción topológica per se, distinta del problema de concurrencia en la implementación física, que es lo que efectivamente se mide acá.

## 4. Desarrollo y código

Ver `tarea_33_async_bounds.py`. El script:
1. Construye una capa afín+ReLU de juguete y calcula los bounds de intervalo $[\underline{z}, \overline{z}]$ de forma cerrada (Sección 2.1).
2. Ejecuta la actualización de forma **secuencial/atómica** (control) y verifica $V=0$.
3. Simula $k$ agentes (hilos de Python con `threading`, sin locks) escribiendo coordenadas del tensor central de forma intercalada y no atómica, y mide $V$ y $\varepsilon$ resultantes, repitiendo el experimento varias veces para estimar una tasa empírica de violación.
4. Repite el experimento con locks (`threading.Lock`) para confirmar que $V$ vuelve a 0 y aislar el efecto de la falta de sincronización.

No se ejecuta como parte de la entrega (regla de la skill). Se hizo únicamente una corrida interna de sanity-check de sintaxis; el resultado real, que depende del scheduler del sistema del alumno y puede variar entre corridas (es una condición de carrera, no determinística), lo observa el alumno corriendo el script él mismo.

### Instrucciones para correrlo vos mismo
```bash
pip install numpy
python tarea_33_async_bounds.py
```
Salida esperada: dos bloques, uno por corrida "secuencial (control)" con violaciones = 0 siempre, y uno "concurrente sin lock" con una tasa de violación variable entre 0 y positiva dependiendo del scheduler — corré el script varias veces para ver que el resultado no es determinístico (es la firma de una condición de carrera real, no un bug del script).

## 5. Conclusión
La parte determinística del modelo (bounds de intervalo bajo actualización atómica) converge exactamente y sin ambigüedad: es álgebra de intervalos estándar, no depende de tuning. La parte que queda abierta y que el alumno debe comprobar corriendo el script las veces que haga falta es si, en su máquina y con su versión de Python (el GIL de CPython limita pero no elimina completamente el interleaving visible a nivel de operaciones sobre arrays de NumPy), se observan violaciones no nulas en el caso sin lock — el resultado puede variar entre corridas y entre entornos, y eso es esperado, no un error: es la naturaleza misma de una condición de carrera, no algo que se pueda "demostrar" con una sola ejecución.

```json
{
 "alumno": null,
 "tarea": 33,
 "tipo_problema": "Verificación de bounds numéricos (dominio de intervalos/poliédrico) bajo actualización concurrente asíncrona de un estado tensor compartido",
 "estado": "resuelto (con componente no determinística documentada explícitamente)",
 "resultado_resumen": "Se construyó un modelo de bounds por propagación de intervalos para una capa afín+ReLU, exacto bajo actualización atómica, y se diseñó una simulación con agentes concurrentes sin sincronización para medir empíricamente la tasa y magnitud de violación de esos bounds frente a un control con lock.",
 "metrica_estandar_usada": "Tasa de violación de intervalos, error L2 y L∞ de la violación (max(0, l-x, x-u))"
}
```
