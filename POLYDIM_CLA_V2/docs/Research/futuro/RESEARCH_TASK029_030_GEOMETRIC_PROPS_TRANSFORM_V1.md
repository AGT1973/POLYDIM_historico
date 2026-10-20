# POLYDIM_DEST
# destination: polydim/research/
# filename: RESEARCH_TASK029_030_GEOMETRIC_PROPS_TRANSFORM_V1.md
# author: claude-sonnet-4-6 (curso05.mithril@gmail.com)

# RESEARCH — TASK_029 + TASK_030: Props como geometría + Transformaciones sin nombre

**Fecha:** 2026-06-20
**Autorización:** docente (mensaje directo) — "COMO DOCENTE AUTORIO DESBLOQUEO"
**Fuente:** KF_003_antipatrones.md (AP_002 y AP_003)

---

## 1. AP_002: Props como geometría (TASK_029)

### El problema (según KF_003)

```python
obj.add("DIM_SQL", {"tabla": "usuarios", "n": 5000}, w=1.0)
# Internamente: _bind(sub("DIM_SQL"), _enc({"tabla": "usuarios"}))
# _enc hace: sym("tabla") * sym("usuarios") — strings humanos
```

`{"tabla": "usuarios"}` es JSON. El encoding lo convierte a vectores, pero el
origen sigue siendo un par clave-valor textual definido por un humano.

### La alternativa correcta (según KF_003)

Las propiedades deberían ser **posiciones relativas** dentro del subespacio.
"tabla=usuarios" sería el vector resultante de proyectar "usuarios" en el
subespacio DIM_SQL — una posición geométrica, no un par clave-valor.

### Diseño de GeometricProps

```python
class GeometricProps:
    """
    Representa propiedades como posiciones en el subespacio, no como JSON.

    En lugar de:
        obj.add("DIM_SQL", {"tabla": "usuarios"}, w=1.0)

    Con GeometricProps:
        gp = GeometricProps(space, "DIM_SQL")
        gp.position("usuarios")          # "usuarios" proyectado en DIM_SQL
        gp.blend("usuarios", "pedidos")  # superposición de posiciones
        obj.add_gp(gp, w=1.0)
    """
```

### Lo que NO cambia (retrocompatibilidad)

- `obj.add("DIM_SQL", {"tabla": "usuarios"})` sigue funcionando (props dict legacy)
- GeometricProps es una alternativa, no un reemplazo obligatorio
- El hipervector resultante de `add_gp(gp)` debe ser compatible con `receive()`

### Operaciones geométricas de las props

```
position(value) → proj(encode(value), sub(dim))
                = escalar de activación de "value" en el subespacio dim
                → usado como peso dinámico en _sup

blend(*values)  → _sup(encode(v1), encode(v2), ...) en el subespacio
                = superposición de posiciones — "usuarios o pedidos"

focus(value)    → _bind(sub(dim), encode(value))
                = la misma semántica que add() actual pero explícita
```

### Métrica de verificación

Con GeometricProps, la similitud entre dos objetos con la misma "posición semántica"
debe ser mayor que con propiedades JSON equivalentes:
- obj_gp_usuarios vs obj_gp_pedidos < obj_gp_usuarios vs obj_gp_usuarios2
- (distintos valores) < (mismo valor, distintas instancias)

---

## 2. AP_003: Transformaciones sin nombre (TASK_030)

### El problema (según KF_003)

```python
VERB_LEER = sp.sym("VERB_LEER")
instruccion = _bind(VERB_LEER, DIM_SQL)  # "leer datos SQL"
```

VERB_LEER es la palabra humana "leer" mapeada a un vector. Sigue siendo
Python con peluca vectorial.

### La alternativa correcta (según KF_003)

> "Un ObjectND cuyo hipervector tiene alta proyección sobre el subespacio de
> 'datos tabulares' y alta proyección sobre el subespacio de 'extracción/lectura'
> (donde estas regiones emergieron del embedding, no de un humano que escribió
> VERB_LEER) YA ES la instrucción 'leer datos tabulares' — sin nombrarla."

### Lo que esto significa en código

Una "transformación" en POLYDIM es un ObjectND con:
- Activaciones altas en los subespacios que definen QUÉ operar
- Activaciones altas en los subespacios que definen CÓMO operar
- Sin nombre, sin verbo, sin string humano — solo geometría

```python
class Transform:
    """
    Una transformación POLYDIM = un ObjectND que modifica otro ObjectND.
    
    No hay VERB_LEER ni VERB_CREAR. Hay regiones del espacio con alta
    activación en las dimensiones relevantes.
    
    apply(source_obj) → result_obj:
        Proyecta source_obj sobre las dims activas del transform,
        aplica la "dirección" geométrica del transform,
        retorna el ObjectND resultante.
    
    compose(other_transform) → transform:
        T2 ∘ T1: el resultado de aplicar T1 y luego T2.
        En geometría: _sup de los hipervectores con la secuencia codificada.
    """
```

### Cómo se "ejecuta" una transformación

```
apply(source, transform):
    # El transform define una "dirección" en el espacio
    # La aplicación es una proyección + reorientación
    direction = transform._hv()
    source_hv = source._hv()
    
    # Componente del source en la dirección del transform
    alignment = dot(source_hv, direction)
    
    # Resultado: source reorientado hacia la dirección del transform
    result_hv = _sup(source_hv, direction, ws=[1 - alignment, alignment])
    return ObjectND_from_hv(result_hv)
```

### Composición (sin nombres)

```
compose(T1, T2) → T_composed:
    # T2 ∘ T1 = aplicar T1 primero, T2 después
    # En geometría: superposición con peso temporal
    hv = _bind(T1._hv(), T2._hv())  # codifica la secuencia
    return Transform_from_hv(hv)
```

---

## 3. DECISIONES DE DISEÑO

### ¿Módulo separado o integrado al runtime?

**Separado.** Igual que SemanticSpace, estos módulos son extensiones opcionales.
El runtime core (polydim_runtime_v07.py) no cambia.

Archivos:
- `polydim_geometric_props.py` — TASK_029
- `polydim_transform.py` — TASK_030

### ¿Qué pasa con el AP_004 (to_symbolic)?

AP_004 dice que to_symbolic() no debería estar en el core. Lo reconocemos pero
no lo resolvemos en estas tareas — eso es TASK_031 (PolydimDebugger).
Por ahora to_symbolic() se mantiene para retrocompatibilidad.

### ¿Cómo se relaciona con TASK_032?

TASK_032 (compilación como proyección) depende de TASK_030.
Con Transform implementado, project_to_executor() será un caso especial de apply():
    apply(source, executor_transform) donde executor_transform tiene
    alta activación en DIM_PYTHON o DIM_RUST.

---

## 4. CRITERIOS DE ÉXITO

**TASK_029 (GeometricProps):**
- `gp.position("usuarios")` retorna un escalar de activación en DIM_SQL
- Dos GPs con el mismo valor tienen similitud coseno > dos GPs con valores distintos
- `obj.add_gp(gp)` produce dims_activas() equivalentes a `obj.add(dim, props_dict)`

**TASK_030 (Transform):**
- `t.apply(obj)` retorna un ObjectND con geo_id distinto al original
- `t1.compose(t2).apply(obj)` aproxima `t2.apply(t1.apply(obj))`
- Un Transform con alta activación en DIM_SQL "atrae" objetos hacia ese subespacio

---

RESEARCH_TASK029_030_GEOMETRIC_PROPS_TRANSFORM_V1.md
2026-06-20 — curso05.mithril@gmail.com — AUTORIZADO POR DOCENTE
