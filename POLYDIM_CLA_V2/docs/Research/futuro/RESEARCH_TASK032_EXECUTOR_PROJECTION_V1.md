# POLYDIM_DEST
# destination: polydim/research/
# filename: RESEARCH_TASK032_EXECUTOR_PROJECTION_V1.md
# author: claude-sonnet-4-6 (curso05.mithril@gmail.com)

# RESEARCH — TASK_032: Compilación como proyección geométrica nativa

**Fecha:** 2026-06-20
**Autorización:** docente — "desbloquea y continua"
**Prerrequisitos:** TASK_028 ✓ (SemanticSpace) + TASK_030 ✓ (Transform)
**Referencia:** KF_002 sección compilar, KF_003 AP_004

---

## 1. EL PRINCIPIO (de KF_002 y la evaluación Antigravity)

> "Compilar = proyectar sobre el subespacio ejecutor."

Hoy no hay compilación en POLYDIM. Hay transfers entre IAs que detectan
qué dims están activas. "Ejecutar" un ObjectND es una ficción.

Lo que TASK_032 propone:

```
project_to_executor(obj, executor="DIM_PYTHON") =
    Transform(space).activate("DIM_PYTHON", 1.0).apply(obj, strength)
    → ObjectND con alta activación en DIM_PYTHON
    → ese ObjectND "es" código Python (en el sentido de que vive
       en la región semántica donde viven los conceptos Python)
```

Esto cierra la brecha: el resultado de proyectar sobre DIM_PYTHON
no ES código Python válido todavía, pero ES la representación geométrica
de "este objeto expresado en términos Python". Un LLM puede entonces
"colapsar" ese hipervector a código real.

---

## 2. DISEÑO DE polydim_executor.py

### Clase ExecutorSpace

```python
class ExecutorSpace:
    """
    Un Space cuyos subespacios NATIVE son los lenguajes ejecutores.
    La "compilación" es project_to_executor() sobre este espacio.
    """
    EXECUTORS = {
        "DIM_PYTHON":   "python code function class module",
        "DIM_RUST":     "rust struct impl trait unsafe",
        "DIM_FLUTTER":  "flutter widget stateless stateful build",
        "DIM_SQL":      "sql select insert update delete table",
    }
```

### Función project_to_executor

```python
def project_to_executor(
    obj: ObjectND,
    executor: str,          # "DIM_PYTHON", "DIM_RUST", etc.
    strength: float = 0.8,  # qué tan fuerte es la proyección
    space: Space = None
) -> ObjectND:
    """
    Proyecta obj hacia el subespacio del executor.
    "Compila" al lenguaje target sin abandonar la geometría.
    """
    sp = space or obj._sp
    t  = Transform(sp).activate(executor, 1.0)
    return t.apply(obj, strength=strength)
```

### Lo que NO hace (honestidad sobre el estado actual)

- NO genera código Python/Rust real → eso requiere un LLM que "colapsa" el hv
- NO verifica sintaxis → la proyección es semántica, no sintáctica
- NO reemplaza un compilador real → es el paso previo: orientación geométrica

### Lo que SÍ hace

- Orienta el ObjectND hacia la región del espacio donde "viven" los conceptos Python/Rust/SQL
- La activación en DIM_PYTHON después de project_to_executor(obj, "DIM_PYTHON") > antes
- Compatible con SemanticSpace: con MiniLMBackend, DIM_PYTHON tiene semántica real

### Pipeline completo (visión)

```
ObjectND           →  project_to_executor("DIM_PYTHON")  →  ObjectND orientado
(concepto abstracto)  (compilación como proyección)         (en región Python)
                                                             ↓
                                                        LLM.collapse(hv) → código Python
```

---

## 3. INVARIANTES A VERIFICAR

1. `proj(result, sub("DIM_PYTHON")) > proj(source, sub("DIM_PYTHON"))` — la proyección acerca
2. Ejecutar dos veces es idempotente: `project(project(obj)) ≈ project(obj)` (strength≈1.0)
3. `polydim_connect(sp, sp2).transfer(result)` → DIM_PYTHON sigue activo en receptor
4. Cross-executor: `project(project(obj, "DIM_SQL"), "DIM_PYTHON")` → ambas activas o Python domina

---

RESEARCH_TASK032_EXECUTOR_PROJECTION_V1.md — 2026-06-20 — curso05.mithril@gmail.com
