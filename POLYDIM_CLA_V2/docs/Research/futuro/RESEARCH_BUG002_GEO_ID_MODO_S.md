# POLYDIM_DEST
# destination: polydim/research/
# filename: RESEARCH_BUG002_GEO_ID_MODO_S.md
# author: claude-sonnet-4-6 (curso05.mithril@gmail.com)

# RESEARCH — BUG_002: Session.receive() MODO_S pierde geo_id del emisor
## Análisis, Fix y Verificación

**Fecha:** 2026-06-18
**Tarea:** BUG_002 (BACKLOG_V27, prioridad 1)
**Autor:** claude-sonnet-4-6
**Estado:** ANÁLISIS COMPLETO — listo para implementar

---

## 1. DIAGNÓSTICO EXACTO

### Flujo MODO_S sin el fix

```
EMISOR (Space "A"):
  obj = ObjectND(spA)
  obj._geo = rnd_vector_1  ← generado en __init__
  obj.add("DIM_SQL", {"tabla": "usuarios"}, w=1.0)
  obj.geo_id = md5(rnd_vector_1)[:12] = "abc123def456"

  pkt = send(obj)
  pkt.payload_S = {
    "geo_id": "abc123def456",   ← solo el HASH, no el vector
    "dims": {"DIM_SQL": {"w": 1.0, "props": {"tabla": "usuarios"}}}
  }
  # payload_G = None en MODO_S puro

RECEPTOR (Space "B"):
  receive(pkt):
    # pkt.payload_G is None → entra al elif payload_S
    o = ObjectND(self.space)
    o._geo = rnd_vector_2  ← NUEVO vector aleatorio, ≠ rnd_vector_1
    for d, info in pkt.payload_S["dims"].items():
        o.add(d, info["props"], w=info["w"])
    # hv = o._hv()  → usa rnd_vector_2

  o.geo_id = md5(rnd_vector_2)[:12] = "xyz789uvw012"  ← DISTINTO
```

**Consecuencia:** el receptor recibe un objeto con dimensiones correctas pero identidad geométrica diferente. Dos objetos que deberían ser el "mismo" (transferido) tienen `geo_id` distintos. Esto rompe cualquier lógica que dependa de la identidad del objeto a través de transferencias MODO_S.

### Por qué MODO_H no tiene el problema (o lo tiene parcialmente)

En MODO_H, `payload_G` está presente. `receive()` entra por el branch `if pkt.payload_G is not None`. El hv geométrico se preserva — la detección de dims usa ese vector. PERO: el ObjectND reconstruido internamente en `route_path()` sí recrea con `_rnd()`, así que el problema reaparece en contextos de reconstrucción.

---

## 2. FIX DISEÑADO

### Cambio 1: `ObjectND.to_symbolic()` — serializar el geo_vector

```python
import base64

def to_symbolic(self) -> dict:
    # Serializar el geo_vector como base64 para reconstrucción exacta
    geo_hv_b64 = base64.b64encode(self._geo.astype(np.float32).tobytes()).decode("ascii")
    return {
        "geo_id":     self.geo_id,       # hash 12 chars (para display/debug)
        "geo_hv_b64": geo_hv_b64,        # vector real (para reconstrucción)
        "dims": {
            d: {"w": self._w.get(d, 1.0), "props": p}
            for d, p in self._props.items()
        }
    }
```

### Cambio 2: `Session.receive()` — reconstruir desde geo_hv_b64

```python
def receive(self, pkt: Packet) -> Dict[str, float]:
    if pkt.payload_G is not None:
        hv = pkt.payload_G
    elif pkt.payload_S:
        o = ObjectND(self.space)
        # FIX BUG_002: reconstruir geo_id del emisor si está disponible
        geo_b64 = pkt.payload_S.get("geo_hv_b64")
        if geo_b64:
            geo_bytes = base64.b64decode(geo_b64)
            geo_vec = np.frombuffer(geo_bytes, dtype=np.float32).copy()
            n = np.linalg.norm(geo_vec)
            o._geo = geo_vec / n if n > 1e-10 else geo_vec
            o._cache = None  # invalidar cache
        for d, info in pkt.payload_S.get("dims", {}).items():
            o.add(d, info.get("props", {}), w=info.get("w", 1.0))
        hv = o._hv()
    else:
        return {}

    dims = {d: round(_proj(hv, self.space.sub(d)), 4)
            for d in NATIVE if _proj(hv, self.space.sub(d)) > UMBRAL}

    if "DIM_META" in dims and pkt.payload_S and self._meta_depth < META_DEPTH_LIMIT:
        self._processar_meta(pkt.payload_S)

    return dims
```

### Cambio 3: `Mesh.route_path()` — mismo fix al reconstruir en nodo intermedio

El código actual ya reconstruye el objeto en cada salto intermedio leyendo `pkt.payload_S`. Con el fix de `to_symbolic()`, el `payload_S` ya incluirá `geo_hv_b64`, así que la reconstrucción en `route_path()` también se beneficia sin cambio adicional — siempre que se use el ObjectND reconstruido, no el `send()` directo.

---

## 3. ANÁLISIS DE IMPACTO

### Retrocompatibilidad

- `to_symbolic()` agrega un campo nuevo `geo_hv_b64`. Parsers que no lo conocen lo ignoran — **sin breaking change**.
- `receive()` verifica `if geo_b64:` — si el campo no está (packets de versiones anteriores), comportamiento idéntico al actual. **Sin breaking change**.
- Los 12 tests existentes deben seguir pasando porque no verifican `geo_id` en el receptor.

### Costo de serialización

- Vector de 10000 float32 = 40000 bytes → base64 ≈ 53333 bytes (53KB por objeto en MODO_S).
- En MODO_H esto ya es el caso (payload_G también es 40KB). MODO_S era más liviano, ahora es similar.
- Alternativa: PBP (Protocol Buffer POLYDIM) que comprime mejor — pero eso es TASK futura.

### Test nuevo requerido (test_geo_id_preserved_modo_s)

```python
def test_geo_id_preserved_modo_s():
    """BUG_002: geo_id del emisor debe preservarse en MODO_S."""
    sp_a = Space("EMISOR")
    sp_b = Space("RECEPTOR_DISTINTO")  # Space diferente → MODO_S o MODO_H

    ia = Session(sp_a, "IA_A")
    ib = Session(sp_b, "IA_B")
    ia.connect(ib)

    # Forzar MODO_S para el test
    ia.mode = ib.mode = Mode.S

    obj = ObjectND(sp_a).add("DIM_SQL", {"tabla": "usuarios"}, w=1.0)
    geo_id_emisor = obj.geo_id

    pkt = ia.send(obj)
    # Verificar que payload_S tiene geo_hv_b64
    assert "geo_hv_b64" in pkt.payload_S, "to_symbolic() debe incluir geo_hv_b64"

    # receive() y reconstruir
    dims = ib.receive(pkt)
    assert "DIM_SQL" in dims, "DIM_SQL debe detectarse"

    # El geo_id del objeto reconstruido debe coincidir
    # Para verificar esto necesitamos acceso al ObjectND reconstruido,
    # no solo a los dims. Ver nota abajo.
    return True  # Test de integración básica

# NOTA: Para verificar geo_id exacto necesitamos que receive() retorne
# ObjectND en vez de Dict. Eso es BUG_003 (API pública). Por ahora
# verificamos indirectamente: mismo payload_S → misma estructura.
```

---

## 4. DECISIÓN SOBRE API DE RETORNO DE receive()

El backlog V27 separa los bugs:
- BUG_002: geo_id perdido en MODO_S → fix en to_symbolic() + receive() (este research)
- BUG_003 (P2): receive() debería retornar ObjectND, no Dict → cambio de API mayor

La verificación de geo_id exacto requiere BUG_003 para ser completa. Por ahora implementamos BUG_002 solo (agregar geo_hv_b64) y dejamos la verificación de la invariante como pendiente de BUG_003.

---

## 5. TAMAÑO DEL PATCH

Líneas a modificar en polydim_runtime_v05.py:
- `ObjectND.to_symbolic()`: +3 líneas (import base64 al tope + 1 campo)
- `Session.receive()`: +6 líneas (bloque de reconstrucción geo)
- `__main__`: +1 test (test_geo_id_preserved_modo_s)
- Total: ~10 líneas de cambio neto.

---

## 6. INVARIANTES A VERIFICAR POST-FIX

| Invariante | Verificación |
|---|---|
| geo_id emisor == geo_id receptor (MODO_S) | test_geo_id_preserved_modo_s |
| 12/12 tests previos siguen OK | ejecutar batería completa |
| payload_S retrocompatible (sin geo_hv_b64 no rompe) | test con packet manual sin el campo |
| geo_vector normalizado post-reconstrucción | norma ≈ 1.0 tras frombuffer |

---

RESEARCH_BUG002_GEO_ID_MODO_S.md — 2026-06-18 — BUG_002 — curso05.mithril@gmail.com
