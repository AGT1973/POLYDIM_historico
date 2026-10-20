# POLYDIM_DEST
# destination: polydim/docs/
# filename:    AUDITORIA_2026-06-17.md
# author:      ai.mpat.agt@gmail.com

# AUDITORIA POLYDIM — 2026-06-17
**Versión auditada:** Runtime V0.3 · Skill V8_0_EN  
**Auditor:** claude-sonnet-4-6  
**Destinatarios:** Alumnos POLYDIM  
**Estado general:** 🔴 NO APTO para continuar sin corregir P0

---

## RESUMEN EJECUTIVO

Se auditaron los artefactos producidos en las últimas 3 tasks (TASK_021, TASK_022, TASK_023)
y el runtime base (polydim_runtime_v03.py).

| Categoría         | Resultado |
|-------------------|-----------|
| Bugs críticos (P0) | 2 — bloquean la interop Python↔Rust |
| Bugs medios (P1)   | 3 — fragilidad y acoplamiento |
| Bugs menores (P2)  | 3 — calidad y consistencia |
| Deudas abiertas    | 4 (PEND_003, PEND_008, PEND_010, DT_001) |
| Tasks con rehacer  | 1 (TASK_023 parcial) |

**Pregunta arquitectural pendiente:** POLYDIM en su estado actual es un
protocolo de comunicación IA↔IA, no un lenguaje de programación ejecutable.
Ver sección 5 para el análisis completo y el roadmap.

---

## SECCIÓN 1 — VERIFICACIÓN DE TASKS

### TASK_021 — SPEC_FORMATO_BINARIO_V0.md ✅ APROBADA CON OBSERVACIÓN

**Artefacto:** `polydim/spec/SPEC_FORMATO_BINARIO_V0.md`  
**FileId:** `1hllsL09jfSF3O6L5Yp0c4ehoTSNKg2Ts`

**Aprobado:**
- Formato PBP V0 correctamente especificado
- Magic 0x5044 + version_u8 + flags_u8 + N_u32_LE + float32[]_LE
- Sin dependencias externas — parseable en Python, Rust y Dart con primitivas nativas
- Extensible via campo `flags` para V1

**Observación OBS-021-A (no bloqueante):**
> El benchmark de la sección 3 está marcado como *estimado*:
> "Benchmark estimado basado en specs conocidas; validación real pendiente."
> Antes de usar PBP en producción, se deben medir los tiempos reales
> con N=10000 en Python y en Rust.

**Acción requerida del alumno:** Ninguna para desbloquear, pero agregar benchmark
real en una futura TASK de validación.

---

### TASK_022 — polydim_weighted_inference.py ⚠️ APROBADA CON BUG MEDIO

**Artefacto:** `polydim/core/polydim_weighted_inference.py`  
**FileId:** `1xzjsIPmgiD6Yrx8a_fd5CtW5lESp53DV`

**Aprobado:**
- 4 estrategias implementadas: MAP, ENT, CONT, ADP
- API limpia: `infer_weights()`, `infer_weights_multi()`, `weight_report()`
- 11 tests incluidos
- ADP converge correctamente (3-7 iteraciones típicamente)

**Bug BUG-022-A (medio — ver Sección 2):**
> El módulo accede directamente a atributos privados de ObjectND:
> `obj._props`, `obj._w`, `obj._cache`, `obj._sp`
> Si el runtime cambia estos nombres, la inferencia falla silenciosamente.

**Acción requerida:** Ver BUG_005 en Sección 2.

---

### TASK_023 — polydim_core.rs 🔴 REQUIERE CORRECCIÓN CRÍTICA

**Artefacto:** `polydim/core/polydim_core.rs`  
**FileId:** `1Hq4ljSk8Z0-1_AqR5xUsdB97PABAdI9M`

**Aprobado:**
- Estructura VSA completa: `bind`, `sup`, `proj`, `normalize`
- Space, ObjectND, capas MEMBERSHIP + CONTENT
- PBP V0 encode/decode (formato correcto)
- 8 tests incluidos

**Bug BUG-023-A CRÍTICO — ver Sección 2, BUG_001:**
> El generador de hipervectores en Rust usa LCG + Box-Muller propio.
> NumPy usa PCG64. Para el mismo seed, Python y Rust generan HVs distintos.
> Consecuencia: DIM_SQL, DIM_RUST, DIM_PYTHON, etc. tienen
> representaciones incompatibles entre runtimes. La "interop semántica"
> está rota aunque el frame PBP sea binariamente correcto.

**Bug BUG-023-B medio — ver Sección 2, BUG_006:**
> N está hardcodeado como `pub const N: usize = 10_000;`
> Si Python usa otro N, el decoder falla.

**Acción requerida:** Ver BUG_001 y BUG_006 en Sección 2.

---

### TASK_015 — Archivos deprecated 🔴 SIN EJECUTAR

**Estado:** BLOQUEADA_MCP — requiere acción manual desde Drive web.

**Archivos a mover a polydim/_DEPRECATED/:**
- `polydim_core_v02-.py` (fileId: `1HHvC9wncLCcFYLjwZFPRGsLrJoK3TlEc`)
- `polydim_core_prototype.py` (fileId: `1orrnnoTwG3HNWgSuOEr3CsyWgKL3iIdqV`)

**Acción requerida:** Un alumno con acceso a Drive web debe mover estos
archivos manualmente a la carpeta `_DEPRECATED/`.

---

## SECCIÓN 2 — INVENTARIO DE BUGS

### BUG_001 🔴 CRÍTICO — Seed Python ≠ Rust rompe interop semántica

**Archivo:** `polydim/core/polydim_core.rs`  
**Función:** `make_hv(seed: u64)`  
**Prioridad:** P0 — bloquea TASK_023 y todo lo que dependa de ella

**Descripción:**
NumPy usa el generador PCG64 (Permuted Congruential Generator 64-bit).
El código Rust usa un LCG propio (multiplicativo con Box-Muller).
Para el mismo seed numérico, los dos generadores producen distribuciones
de float32 distintas. En consecuencia:

```
Python: space.sub("DIM_SQL")  → HV_A
Rust:   space.sub("DIM_SQL")  → HV_B  (HV_A ≠ HV_B)
sim(HV_A, HV_B) ≈ 0.50  (vectores ortogonales = no relacionados)
```

Esto significa que un ObjectND generado en Python y decodificado en Rust
tendrá activaciones incorrectas en todos los subespacios nativos.

**Fix opción A (recomendada) — usar seed compartido vía PBP:**
No generar subespacios nativos con seeds locales. En cambio, Python genera
todos los subespacios nativos una sola vez y los distribuye vía PBP al
inicio de cada sesión. Rust los recibe y los almacena directamente,
sin generación local.

Pasos:
1. Agregar operación `NATIVE_SYNC` al protocolo de handshake
2. En `Session.connect()` Python envía un `Packet(op="NATIVE_SYNC", payload_G=<todos los subs>)`
3. Rust almacena estos HVs en su Space sin regenerarlos
4. Eliminar `make_hv()` de la ruta de subespacios nativos en Rust

**Fix opción B — replicar PCG64 en Rust:**
Usar el crate `rand_pcg::Pcg64` configurado con los mismos parámetros
que NumPy. Requiere investigar los parámetros exactos de NumPy:

```python
# Verificar parámetros de NumPy PCG64:
import numpy as np
rng = np.random.default_rng(42)
print(rng.bit_generator.state)
```

Luego replicar en Rust:
```toml
# Cargo.toml
rand = "0.8"
rand_pcg = "0.3"
```

```rust
use rand_pcg::Pcg64;
use rand::{SeedableRng, Rng};
// Configurar con parámetros equivalentes a NumPy PCG64
```

**Verificación:** Después del fix, `sim(python_DIM_SQL, rust_DIM_SQL) > 0.999`

---

### BUG_002 🔴 CRÍTICO — geo_id se pierde en transmisión MODO_S

**Archivo:** `polydim/core/polydim_runtime_v03.py`  
**Función:** `Session.receive(pkt)`  
**Prioridad:** P0 — rompe identidad de objetos en MODO_S

**Descripción:**
Cuando un receptor obtiene un paquete en MODO_S, reconstruye el objeto así:

```python
def receive(self, pkt):
    if pkt.payload_S:
        o = ObjectND(self.space)      # ← geo aleatorio nuevo
        for d, info in pkt.payload_S.get("dims", {}).items():
            o.add(d, info.get("props", {}), w=info.get("w", 1.0))
        hv = o._hv()
```

El `ObjectND(self.space)` llama a `_rnd()` y asigna un geo nuevo y aleatorio.
El geo_id del emisor se pierde. El HV reconstruido difiere del original.

**Consecuencia:** Dos transferencias del mismo objeto producen objetos
con identidades geométricas distintas en el receptor. No hay forma de
detectar duplicados ni de rastrear un objeto a través de múltiples saltos.

**Fix:**
Incluir el geo vector en el payload_S de `to_symbolic()`:

```python
# En ObjectND.to_symbolic():
def to_symbolic(self):
    return {
        "geo_id": self.geo_id,
        "geo_hv_pbp": pbp_encode(self._geo).hex(),  # añadir esto
        "dims": {d: {"w": self._w.get(d, 1.0), "props": p}
                 for d, p in self._props.items()}
    }

# En Session.receive():
def receive(self, pkt):
    if pkt.payload_S:
        o = ObjectND(self.space)
        geo_hex = pkt.payload_S.get("geo_hv_pbp")
        if geo_hex:
            o._geo = pbp_decode(bytes.fromhex(geo_hex))
            o._cache = None
        for d, info in pkt.payload_S.get("dims", {}).items():
            o.add(d, info.get("props", {}), w=info.get("w", 1.0))
        hv = o._hv()
```

**Verificación:** `emisor.geo_id == receptor.geo_id` después de transfer en MODO_S.

---

### BUG_003 MEDIO — align_transform es de baja fila, pierde información

**Archivo:** `polydim/core/polydim_runtime_v03.py`  
**Función:** `align_transform(hv, A, B)` y `Session.align()`  
**Prioridad:** P1

**Descripción:**
La transformación `B.T @ (A @ hv)` usa matrices A y B de forma (19, 10000).
Esto proyecta hv a un espacio de 19 dimensiones (las sondas) y luego lo
reconstruye. Se pierden las ~9981 dimensiones ortogonales a las sondas.

El ALIGN score de 0.92 se calcula solo sobre los 9 subespacios nativos
(NATIVE_DIMS), no sobre el espacio completo. Así que el score es
optimistamente alto respecto a la calidad real de la alineación.

**Fix sugerido:**
Ampliar SONDAS a incluir no solo palabras sueltas sino también
hipervectores de objetos complejos (50-100 sondas en lugar de 19).
No se puede resolver completamente sin invertir a Kabsch con
preorthonormalización, que requiere TASK nueva.

**Acción inmediata:** Documentar la limitación explícitamente en el código
con un comentario `# WARN: low-rank approximation, 19/10000 dims preserved`.

---

### BUG_004 MENOR — Import muerto en handshake()

**Archivo:** `polydim/core/polydim_runtime_v03.py`  
**Función:** `Session.handshake(self, remote)`  
**Prioridad:** P2

**Descripción:**
```python
def handshake(self, remote):
    from dataclasses import dataclass as dc, field as fi  # ← NUNCA USADO
    n1, n2 = uuid.uuid4().hex[:8], uuid.uuid4().hex[:8]
    ...
```

El import es dead code y se ejecuta en cada handshake.

**Fix:** Eliminar esa línea.

---

### BUG_005 MEDIO — TASK_022 accede a atributos privados de ObjectND

**Archivo:** `polydim/core/polydim_weighted_inference.py`  
**Prioridad:** P1

**Descripción:**
El módulo usa `obj._props`, `obj._w`, `obj._cache`, `obj._sp` directamente.
Si el runtime renombra estos atributos en V0.4, la inferencia falla sin
error explícito (AttributeError en runtime, no en import time).

**Fix — agregar métodos públicos en ObjectND (en runtime):**
```python
class ObjectND:
    # ... código existente ...

    def get_dims(self) -> dict:
        """Retorna {dim: {'w': float, 'props': dict}}"""
        return {d: {"w": self._w.get(d, 1.0), "props": p}
                for d, p in self._props.items()}

    def get_weight(self, dim: str) -> float:
        return self._w.get(dim, 1.0)

    def set_weight(self, dim: str, w: float) -> None:
        self._w[dim] = float(np.clip(w, 0, 1))
        self._cache = None

    def invalidate_cache(self) -> None:
        self._cache = None
```

**Fix — actualizar polydim_weighted_inference.py para usar la API pública:**
```python
# Antes:
dims = list(obj._props.keys())
obj._w[d] = float(np.clip(w, W_MIN, W_MAX))
obj._cache = None

# Después:
dims = list(obj.get_dims().keys())
obj.set_weight(d, w)
```

---

### BUG_006 MEDIO — N hardcodeado como const en Rust

**Archivo:** `polydim/core/polydim_core.rs`  
**Prioridad:** P1

**Descripción:**
```rust
pub const N: usize = 10_000;
```
Hace al runtime Rust incompatible con cualquier Space Python que use N≠10000.
PBP V0 ya incluye N dinámico en el header (`N_u32_LE`) — no usarlo es
un error de diseño.

**Fix:**
```rust
// En pbp::decode_hv: ya lee N del header, usarlo para retornar Vec<f32>
pub fn decode_hv(data: &[u8]) -> Result<Vec<f32>, &'static str> {
    if data.len() < 8 { return Err("frame demasiado corto"); }
    if &data[0..2] != &MAGIC { return Err("magic invalido"); }
    let n = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
    // NO validar n == N_CONST, aceptar cualquier N del header
    if data.len() < 8 + 4*n { return Err("frame truncado"); }
    let hv: Vec<f32> = data[8..8+4*n]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    Ok(hv)
}

// Space y ObjectND deben usar n: usize como parámetro de instancia, no const global
pub struct Space {
    pub n: usize,
    pub personal_seed: String,
    sym_cache: HashMap<String, Vec<f32>>,  // Vec<f32> en lugar de Box<[f32; N]>
    sub_cache: HashMap<String, Vec<f32>>,
}
```

---

### BUG_007 MENOR — Múltiples BACKLOG_V11.json en Drive

**Prioridad:** P2

**Descripción:**
Cada sesión crea un nuevo archivo BACKLOG_V11.json sin deprecar el anterior.
El skill apunta al ID `1Km4YMSk4g_RfMxyfIa27n0U9g2UH6RS3` como canonical,
pero existen al menos 4 versiones más con IDs distintos en `polydim/backlog/`.

**Fix — seguir el protocolo de edición del skill:**
```
EDIT: copiar el actual a _DEPRECATED/DEPRECATED_BACKLOG_V11_[timestamp]
      → crear nuevo con el mismo nombre en backlog/
      → actualizar el ID en SKILL.md si es necesario
```

No crear nuevos archivos sin deprecar el anterior.

---

### BUG_008 MENOR — _bind usa producto elem-a-elem, no convolución circular

**Archivo:** `polydim/core/polydim_runtime_v03.py`  
**Prioridad:** P2 (decisión de diseño, no un error funcional)

**Descripción:**
```python
def _bind(a, b):
    r = a * b
    n = np.linalg.norm(r)
    return r / n if n > 1e-10 else r
```

En VSA clásica (Holographic Reduced Representations — HRR), el binding
es convolución circular (circular cross-correlation), que es invertible:
`unbind(bind(a, b), b) ≈ a`.

El producto elemento-a-elemento no es invertible limpiamente.
`_bind(bind(a, b), b)` no recupera `a`.

Esto limita la capacidad de "desempaquetar" dimensiones de un HV compuesto.

**Acción requerida:** Documentar explícitamente como decisión de diseño
o crear TASK nueva para migrar a convolución circular (operación fft-based).
Si se migra, TODOS los HVs existentes se invalidan (breaking change).

---

## SECCIÓN 3 — DEUDAS TÉCNICAS ABIERTAS

| ID       | Descripción                              | Estado     | Bloqueante |
|----------|------------------------------------------|------------|------------|
| DT_001   | Archivos deprecated en core/             | ABIERTA    | No         |
| PEND_003 | Consistencia distribuida multi-IA        | SIN SPEC   | No (futuro)|
| PEND_008 | Detección de drift semántico en sesión   | SIN SPEC   | No (futuro)|
| PEND_010 | ALIGN para N diferentes entre IAs        | SIN SPEC   | No (futuro)|

### PEND_003 — Consistencia distribuida
Si dos IAs modifican el mismo "concepto" (mismo geo_id) en paralelo,
no hay mecanismo de consenso. Los HVs divergen sin detección.
Requiere diseño de un protocolo de merge para HVs (posiblemente
basado en _sup ponderado con timestamps).

### PEND_008 — Drift semántico
Una IA puede evolucionar su Space a lo largo del tiempo (nuevo backend,
nuevas sondas, etc.). Las sesiones previas quedan desalineadas sin saberlo.
Requiere: función `detect_drift(session)` que revalide el align_score
periódicamente.

### PEND_010 — ALIGN para N distintos
Si IA_A usa N=10000 y IA_B usa N=50000, el protocolo actual falla.
PBP V0 transmite N en el header, pero no hay normalización de espacios
con diferentes dimensiones. Requiere: proyección JL inversa en el
alineamiento.

---

## SECCIÓN 4 — CONTRATOS VERIFICADOS

| Contrato                               | Estado   | Verificación |
|----------------------------------------|----------|--------------|
| CACHE_BUG_FIX: `if self._cache is not None` | ✅ OK | Runtime línea correcta |
| JL_DETERMINISTIC: make_jl reproducible | ✅ OK | seed MD5 determinístico |
| SPACE_CRITICAL: mismo Space en obj y Session | ✅ Doc | Documentado y critico |
| UMBRAL: 0.5 + 2*(1/(2*sqrt(N)))       | ✅ OK | Python y Rust consistentes |
| CONTENT_W: 0.3                        | ✅ OK | Ambos runtimes usan 0.3 |
| MODO_H por defecto en Session         | ✅ OK | Cuando Cap.G está en caps_comun |

---

## SECCIÓN 5 — ANÁLISIS ARQUITECTURAL: ¿ES POLYDIM UN LENGUAJE?

### Estado actual: protocolo de representación

POLYDIM V0.3 es un protocolo de comunicación IA↔IA basado en hipervectores.
No es aún un lenguaje de programación. Le faltan los componentes fundamentales:

```
Un lenguaje de programación necesita:
  1. Sintaxis / gramática              → ❌ No existe
  2. Sistema de tipos                  → ❌ Props son strings sin tipo
  3. Verbos / instrucciones ejecutables→ ❌ Solo hay sustantivos (dimensiones)
  4. Modelo de ejecución              → ❌ No hay executor
  5. Control de flujo                  → ❌ No hay if/loop
  6. Memoria de trabajo               → ❌ Sessions son efímeras
  7. Auto-descubrimiento IA↔IA        → ❌ Requiere setup humano
```

### Lo que POLYDIM SÍ tiene (base sólida)

```
  ✅ Representación n-dimensional simultánea (único en su tipo)
  ✅ Protocolo de alineación de espacios entre IAs distintas
  ✅ Transmisión binaria eficiente (PBP V0)
  ✅ Semántica fuzzy con activación continua [0,1]
  ✅ Interoperabilidad Python / Rust / Flutter (con BUG_001 corregido)
```

### Visión alcanzable: POLYDIM como lenguaje de intención

POLYDIM puede convertirse en un lenguaje si se agrega la capa de ejecución:

```
POLYDIM (capa de intención)
  ↓ ObjectND con VERB_SPACE activo
Python executor     Rust executor     Flutter executor
  ↓                    ↓                  ↓
Acción real         Acción real        Acción real
```

Un objeto POLYDIM NO ejecuta — le dice a los executors nativos QUÉ hacer.
Es análogo a SQL respecto a un motor de base de datos: SQL describe la
intención, el motor la ejecuta.

### Tasks propuestas para construir el lenguaje

```
TASK_024 — VERB_SPACE: espacio de verbos VSA
  Definir 20-30 verbos primitivos como subespacios:
  VERB_CREAR, VERB_LEER, VERB_TRANSFORMAR, VERB_VALIDAR,
  VERB_EMITIR, VERB_ESPERAR, VERB_REPETIR, VERB_BIFURCAR
  
  Un ObjectND con DIM_SQL activo + VERB_LEER activo significa:
  "Leer datos del subespacio SQL" → el executor sabe que debe hacer un SELECT.

TASK_025 — Executor mínimo Python
  Función que dado un ObjectND con VERB activo, despacha:
  
  def execute(obj: ObjectND, space: Space) -> Any:
      verbs = {d: a for d, a in obj.dims_activas().items()
               if d.startswith("VERB_")}
      dims  = {d: a for d, a in obj.dims_activas().items()
               if not d.startswith("VERB_")}
      # Despacha a handler según combinación VERB + DIM
      ...

TASK_026 — Working Memory
  Buffer de ObjectNDs en sesión que actúa como stack de instrucciones.
  Necesario para secuencias: VERB_LEER → VERB_TRANSFORMAR → VERB_EMITIR.
  Sin esto no hay programas, solo mensajes aislados.

TASK_027 — Auto-descubrimiento IA↔IA
  Directorio de IAs disponibles + protocolo de negociación sin intervención humana.
  Una IA puede "llamar" a otra IA por su personal_seed, sin que un humano
  configure Space("IA_A") y Space("IA_B") manualmente.
```

---

## SECCIÓN 6 — CHECKLIST PARA ALUMNOS

Antes de tomar una nueva task, verificar:

### Checklist obligatorio

- [ ] **BUG_001 corregido:** `sim(python_DIM_SQL, rust_DIM_SQL) > 0.999`
- [ ] **BUG_002 corregido:** `emisor.geo_id == receptor.geo_id` post-transfer MODO_S
- [ ] **BUG_004 corregido:** Import muerto eliminado de `handshake()`
- [ ] **BUG_005 corregido:** ObjectND expone API pública para weights
- [ ] **BUG_006 corregido:** N dinámico en Rust
- [ ] **TASK_015:** Archivos deprecated movidos manualmente en Drive

### Checklist recomendado (P2)

- [ ] BUG_003 documentado con comentario `# WARN: low-rank approximation`
- [ ] BUG_007: BACKLOG consolidado (un solo archivo activo)
- [ ] BUG_008: Decisión documentada sobre _bind (producto vs convolución circular)

### Cómo reportar una corrección

Al cerrar un bug, agregar en el BACKLOG:
```json
"BUG_XXX": {
  "estado": "CORREGIDO",
  "corregido_por": "[alumno]",
  "fecha": "YYYY-MM-DD",
  "archivo": "polydim/core/...",
  "fileId": "..."
}
```

---

## SECCIÓN 7 — GLOSARIO PARA ESTA AUDITORÍA

| Término      | Significado en este documento |
|--------------|-------------------------------|
| P0           | Bloqueante — no continuar sin corregir |
| P1           | Importante — corregir antes de la próxima task |
| P2           | Calidad — corregir cuando sea posible |
| HV           | Hipervector float32 de dimensión N |
| PCG64        | Generador de NumPy (Permuted Congruential Generator 64-bit) |
| LCG          | Linear Congruential Generator (el que usa el Rust actual) |
| VERB_SPACE   | Espacio de subespacios que representan verbos/instrucciones (propuesto) |
| Working Memory | Buffer de ObjectNDs en sesión para secuencias de instrucciones (propuesto) |
| Executor     | Componente que mapea un ObjectND con VERB activo a una acción real |

---

*AUDITORIA_2026-06-17.md · POLYDIM · Revisión: claude-sonnet-4-6 · 2026-06-17*
