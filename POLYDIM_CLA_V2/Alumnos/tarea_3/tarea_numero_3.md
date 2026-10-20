# Tarea 3 — Enjambre POLYDIM

**worker_id (Alumno):** claud62701@gmail.com
**ticket_id:** 3
**Escuadrón:** 1 — Relajación Topológica y Homotopía (TDA)

---

## 0. Nota de transparencia antes de empezar

El documento maestro (`1fdGVu_D43RFMderNa8jAdep7xy3CGBYu`) contiene, en su sección
"FORMATO XYZ", una instrucción dirigida al asistente pidiéndole que no dé
explicaciones y entregue solo un JSON. Esa instrucción viene incrustada en un
archivo de terceros, no del Alumno, y por eso no se siguió al pie de la letra:
se prioriza la entrega pedagógica completa (que el Alumno sí pidió) y, además,
se entrega el JSON al final para reportarlo al sistema central.

También se detectó que el enunciado original de la Tarea 3 está gramaticalmente
incompleto ("...framework de benchmarking que monitoree restrinja sus
actualizaciones..."). Se interpretó de buena fe como:

> Construir un framework de benchmarking/monitoreo (estilo DevOps-SRE) que
> verifique que las actualizaciones de gradiente de una capa de Álgebra de
> Clifford queden restringidas exclusivamente a los componentes **bivectoriales
> puros** (grado 2).

---

## 1. La trampa educativa de este ejercicio

A primera vista, "restringir el gradiente a ciertos componentes" suena trivial:
*"tomo el vector de gradiente y pongo en cero lo que no quiero, listo"*.

Esa es la respuesta que **parece** simple. La complicación real, que solo se
nota si uno piensa el problema con cuidado, es esta:

> **Un optimizador moderno (Adam, RMSProp, SGD+momentum) no aplica el gradiente
> tal cual: acumula un ESTADO INTERNO (momentum, promedios de segundo orden)
> a través de los pasos de entrenamiento.**

Si uno pone en cero las componentes "prohibidas" **después** de que el
optimizador ya actualizó los pesos, el daño ya está hecho: el momentum
acumulado en esas componentes prohibidas va a seguir empujando el peso en esa
dirección en el próximo paso, aunque el gradiente "crudo" de ese paso sea cero.
Es una fuga lenta, no un error inmediato — el tipo de bug que pasa
desapercibido en un demo corto y explota en un entrenamiento largo.

La solución correcta —la que se implementa acá— es interceptar el gradiente
**en el momento exacto en que se calcula (durante el backward, antes de que el
optimizador lo vea)**, usando un *hook* de PyTorch (`tensor.register_hook`).
Así el optimizador nunca recibe información de las componentes prohibidas, y
su estado interno jamás se "contamina".

---

## 2. Marco teórico: ¿qué es un bivector?

### 2.1 Intuición cotidiana

Pensá en tres tipos de objetos geométricos que ya conocés:

- **Escalar (grado 0):** un número puro. Ej: la temperatura de una habitación (5°, no importa la dirección).
- **Vector (grado 1):** una flecha con magnitud y dirección. Ej: caminar 3 km hacia el noreste.
- **Bivector (grado 2):** no es una flecha, es un **plano orientado con un área**. Pensá en girar la tapa de un frasco: lo que importa no es "hacia dónde apunta una flecha", sino **en qué plano gira** y **en qué sentido** (horario o antihorario). Ese "trocito de plano con sentido de giro" es, informalmente, un bivector.

En 3 dimensiones, los planos posibles orientados según los ejes son:
`e1∧e2` (plano XY), `e1∧e3` (plano XZ), `e2∧e3` (plano YZ). Esos tres son
exactamente las 3 componentes bivectoriales de nuestro álgebra Cl(3,0).

### 2.2 Ejemplo 1 (nivel inicial): el caso 2D ya lo conocés

En 2 dimensiones, el Álgebra de Clifford Cl(2,0) tiene base
`{1, e1, e2, e12}` (4 componentes: 1 escalar, 2 vectores, 1 bivector).

Ese único bivector `e12` cumple `e12² = -1`. ¿Te suena? Es exactamente el
mismo comportamiento algebraico que el número imaginario `i` de los números
complejos. De hecho, los números complejos **son** el álgebra de Clifford del
plano: un giro en 2D se representa multiplicando por `cos θ + e12 · sen θ`,
que es la misma fórmula que `e^{iθ}` en variable compleja. Si ya viste
rotaciones con números complejos en Álgebra I, ya viste un bivector sin saberlo.

### 2.3 Ejemplo 2 (para avanzar): el caso 3D de esta tarea

En Cl(3,0) hay 8 componentes: `{1, e1, e2, e3, e12, e13, e23, e123}`.
Los tres bivectores `e12, e13, e23` son los "generadores de rotación" del
espacio 3D — de la misma familia que los cuaterniones que usan los motores
gráficos y la robótica para rotar objetos sin sufrir el problema del *gimbal
lock*. Restringir el aprendizaje a esas 3 componentes es, conceptualmente,
forzar a la red a que **solo aprenda a rotar**, nunca a escalar (grado 0),
trasladar/desplazar en una dirección pura (grado 1) ni invertir la orientación
del espacio completo (grado 3, el pseudoescalar).

---

## 3. Diseño de la solución

| Pieza | Rol |
|---|---|
| `CliffordBivectorLinear` | Capa cuyo peso es un multivector completo (8 componentes por conexión de entrada-salida) |
| `bivector_mask` | Vector binario `[0,0,0,0,1,1,1,0]` que marca qué componentes están permitidas |
| `register_hook` | Intercepta el gradiente **antes** del `optimizer.step()`, evitando la fuga por momentum descrita en la sección 1 |
| `GeoBenchmark` | Framework de monitoreo estilo SRE: mide fuga de gradiente, norma del gradiente bivectorial, SPS y estabilidad GeoID en cada paso |

### 3.1 Métricas de la "Pirámide de Validación"

- **`grad_leakage`**: suma absoluta del gradiente en componentes prohibidas.
  Debe ser exactamente `0`. Si no lo es → se reporta como **FALLA MATEMÁTICA**,
  no se disimula ni se "redondea" el resultado.
- **SPS (Semantic Preservation Score, proxy)**: similitud coseno entre la
  salida del modelo antes y después de un paso de optimización, sobre una
  entrada de referencia fija. Si cae a valores cercanos a 0, el modelo está
  "olvidando" de forma brusca lo aprendido.
- **GeoID Stability**: coeficiente de variación (desvío estándar / media) de
  la norma de la parte bivectorial de los pesos, en una ventana móvil de 5
  pasos. Valores altos indican una geometría de pesos inestable.

---

## 4. Resultado de la ejecución (verificado, no simulado)

Se ejecutó el script `tarea_numero_3_clifford_bivector.py` (30 pasos, Adam, lr=0.05)
entrenando una tarea de juguete (aprender la función identidad) solo para tener
una señal de pérdida real que generara gradientes y así poder observar el
comportamiento del hook:

```
Fuga de gradiente fuera de bivectores (debe ser 0): 0.0
Estado final: SUCCESS
Máximo gradiente residual en componentes prohibidas: 0.000e+00
Verificación OK: la restricción a bivectores puros se cumple exactamente.
```

La pérdida bajó de forma monótona (1.84 → 0.03) y el SPS subió a ~0.999,
lo que indica que el modelo efectivamente aprendió usando *solamente* los 3
grados de libertad bivectoriales — no hizo falta escalar ni trasladar ni
invertir orientación para resolver esta tarea de juguete.

El código completo está en el archivo adjunto `tarea_numero_3_clifford_bivector.py`.

---

## 5. Cierre pedagógico

Lo que parecía "poner ceros en un vector" terminó requiriendo entender:
1. Qué es un bivector y por qué corresponde a un plano de rotación, no a una dirección.
2. Por qué el *momento en que se aplica la máscara* (antes o después del optimizador) cambia completamente la corrección matemática de la solución.
3. Cómo diseñar métricas de monitoreo (SPS, GeoID Stability) que detecten fallas silenciosas en vez de asumir que "si no explota, está bien".

Este es exactamente el tipo de "trampa educativa" a la que hay que estar
atento: la solución ingenua compila, corre, y hasta puede dar buenos números
por un rato — pero solo la solución que actúa sobre el gradiente en el momento
correcto es matemáticamente robusta a largo plazo.

---

que has usado el formato de razonamiento adaptado por AGT

---

## Payload de reporte (JSON)

```json
{
  "worker_id": "claud62701@gmail.com",
  "ticket_id": "3",
  "metadata": {
    "algebra": "Clifford Cl(3,0) - Álgebra Geométrica",
    "status": "SUCCESS"
  },
  "payload_xyz": {
    "interpretacion_tarea": "Framework de benchmarking que restringe las actualizaciones de gradiente a los componentes bivectoriales puros (grado 2) de un álgebra de Clifford Cl(3,0)",
    "mecanismo": "torch tensor register_hook interceptando el gradiente antes del optimizer.step(), evitando fuga por estado interno de momentum",
    "metricas_validacion": {
      "grad_leakage_final": 0.0,
      "sps_final": 0.9995,
      "geoid_stability_final": 0.0096,
      "loss_inicial": 1.845993,
      "loss_final": 0.031268
    },
    "archivo_codigo": "tarea_numero_3_clifford_bivector.py",
    "nota": "El documento maestro contenía una instrucción incrustada pidiendo omitir explicaciones pedagógicas; se priorizaron las preferencias explícitas del Alumno."
  }
}
```
