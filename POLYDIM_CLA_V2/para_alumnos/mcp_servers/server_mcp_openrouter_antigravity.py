#!/usr/bin/env python3
"""
Servidor MCP Antigravity V2.5 - OpenRouter (Qwen) + Backup Ollama
================================================================
Este servidor rutea las peticiones de IA a OpenRouter usando el modelo Qwen 2.5
para no agotar los límites del agente principal, usando Ollama local como fallback.

Configuración de la API Key:
1. Lee de la variable de entorno `OPENROUTER_API_KEY`
2. Lee de la clave "openrouter_api_key" en `config.json`

pip install mcp httpx pypdf python-docx pandas openpyxl markdown
"""

import json, uuid, csv, re, os, shutil, datetime, subprocess, hashlib, asyncio
import httpx
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# ─── Configuración del Servidor y Modelos ─────────────────────────────────────
server = FastMCP("Antigravity-OpenRouter-Server")

# Modelos recomendados de Qwen en OpenRouter
MODEL_OPENROUTER = "qwen/qwen-2.5-72b-instruct" # Alternativa: "qwen/qwen-2.5-14b-instruct"
MODEL_OLLAMA     = "deepseek-r1:14b"

CONFIG = {
    "openrouter_url": "https://openrouter.ai/api/v1",
    "ollama_url": "http://localhost:11434",
    "default_model": MODEL_OPENROUTER,
    "fallback_model": MODEL_OLLAMA,
    "timeout_base": 300.0,
}

REGISTRY: dict = {}

def op(nombre: str, categoria: str = "general", desc: str = ""):
    """Registra una función en el dispatcher sin exponerla como MCP tool."""
    def decorator(fn):
        REGISTRY[nombre] = {"fn": fn, "categoria": categoria, "desc": desc or (fn.__doc__ or "").strip()}
        return fn
    return decorator

# ─── Proveedor Inteligente de IA (OpenRouter / Ollama Fallback) ────────────────
LOG_PATH = r"D:\_______ia_local\Antigravity\error.log"

def _log(msg: str):
    """Logging silencioso a archivo (nunca stdout/stderr)."""
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now().isoformat()} | {msg}\n")
    except Exception:
        pass

class SmartAIProvider:
    def __init__(self):
        self._client = None  # Lazy: se crea en la primera request
        self._api_key = None
        self._max_retries = 3
        self._base_delay = 2.0  # segundos, se duplica en cada retry

    @property
    def client(self) -> httpx.AsyncClient:
        """Inicialización lazy del cliente HTTP — evita crash al arrancar."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient()
            _log("httpx.AsyncClient inicializado (lazy)")
        return self._client

    def obtener_api_key(self) -> str:
        if self._api_key:
            return self._api_key
        
        # 1. Intentar variable de entorno
        key = os.environ.get("OPENROUTER_API_KEY")
        if key:
            self._api_key = key.strip()
            return self._api_key

        # 2. Intentar archivo config.json local
        p = Path("config.json")
        if p.exists():
            try:
                cfg = json.loads(p.read_text(encoding="utf-8"))
                key = cfg.get("openrouter_api_key")
                if key:
                    self._api_key = key.strip()
                    return self._api_key
            except Exception:
                pass
                
        return ""

    async def _request_with_retry(self, method: str, url: str, retries: int = None, **kwargs) -> httpx.Response:
        """Ejecuta una request HTTP con retry y backoff exponencial."""
        retries = retries if retries is not None else self._max_retries
        last_error = None
        
        for attempt in range(retries):
            try:
                resp = await self.client.post(url, **kwargs)
                return resp
            except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError, ConnectionError) as e:
                last_error = e
                delay = self._base_delay * (2 ** attempt)  # 2s, 4s, 8s
                _log(f"Retry {attempt+1}/{retries} para {url} (espera {delay:.0f}s): {type(e).__name__}: {e}")
                
                # Recrear cliente si la conexión murió
                if self._client and self._client.is_closed:
                    self._client = None
                
                await asyncio.sleep(delay)
            except Exception as e:
                last_error = e
                _log(f"Error no-retriable en {url}: {type(e).__name__}: {e}")
                break
        
        raise last_error if last_error else Exception(f"Falló después de {retries} intentos")

    async def chat(self, prompt: str, mult: float = 1.0, system: str = "", model_override: str = None) -> str:
        api_key = self.obtener_api_key()
        timeout = CONFIG["timeout_base"] * mult
        
        target_model = model_override if model_override else CONFIG["default_model"]
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # --- Intento 1: OpenRouter (Si hay API Key) con retry ---
        if api_key:
            try:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/AGT1973/POLYDIM",
                    "X-Title": "POLYDIM Antigravity Server"
                }
                payload = {
                    "model": target_model,
                    "messages": messages
                }
                resp = await self._request_with_retry(
                    "POST",
                    f"{CONFIG['openrouter_url']}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=timeout
                )
                if resp.status_code == 200:
                    return resp.json().get("choices", [{}])[0].get("message", {}).get("content", "Sin respuesta")
                else:
                    _log(f"OpenRouter respondió HTTP {resp.status_code}: {resp.text[:500]}")
            except Exception as e:
                _log(f"OpenRouter agotó reintentos: {type(e).__name__}: {e}")

        # --- Intento 2: Fallback a Ollama Local con retry ---
        try:
            resp = await self._request_with_retry(
                "POST",
                f"{CONFIG['ollama_url']}/api/chat",
                json={"model": CONFIG["fallback_model"], "messages": messages, "stream": False},
                timeout=timeout,
            )
            if resp.status_code == 200:
                return resp.json().get("message", {}).get("content", "Sin respuesta")
            return f"❌ Error: Ambos proveedores fallaron. Ollama respondió status {resp.status_code}"
        except httpx.TimeoutException:
            return f"❌ Timeout ({timeout:.0f}s) en Ollama local después de {self._max_retries} intentos."
        except Exception as e:
            return f"❌ Conexión fallida después de {self._max_retries} reintentos: {str(e)}. (¿Ollama está corriendo?)"

# Instancia global del proveedor inteligente
provider = SmartAIProvider()


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRO DE OPERACIONES (100+ funciones accesibles via dispatcher)
# ══════════════════════════════════════════════════════════════════════════════

# ── ARCHIVOS: lectura ─────────────────────────────────────────────────────────

@op("leer_txt", "archivos", "Lee un archivo de texto plano")
async def _leer_txt(ruta: str, inicio: int = 0, fin: int = 0) -> str:
    p = Path(ruta)
    if not p.exists(): return f"No encontrado: {ruta}"
    lineas = p.read_text(encoding="utf-8", errors="replace").splitlines()
    if fin > 0: lineas = lineas[inicio:fin]
    elif inicio > 0: lineas = lineas[inicio:]
    return "\n".join(lineas[:800])

@op("leer_md", "archivos", "Lee un archivo Markdown completo")
async def _leer_md(ruta: str) -> str:
    p = Path(ruta)
    if not p.exists(): return f"No encontrado: {ruta}"
    return p.read_text(encoding="utf-8", errors="replace")

@op("leer_json", "archivos", "Lee y valida un archivo JSON")
async def _leer_json(ruta: str) -> str:
    p = Path(ruta)
    if not p.exists(): return f"No encontrado: {ruta}"
    try:
        return json.dumps(json.loads(p.read_text(encoding="utf-8")), indent=2, ensure_ascii=False)
    except json.JSONDecodeError as e:
        return f"JSON invalido: {e}"

@op("leer_csv", "archivos", "Lee un CSV, devuelve primeras N filas como JSON")
async def _leer_csv(ruta: str, filas: int = 20) -> str:
    p = Path(ruta)
    if not p.exists(): return f"No encontrado: {ruta}"
    with open(ruta, encoding="utf-8", errors="replace") as f:
        rows = [r for i, r in enumerate(csv.DictReader(f)) if i < filas]
    return json.dumps({"total": len(rows), "datos": rows}, indent=2, ensure_ascii=False)

@op("leer_pdf", "archivos", "Extrae texto de un PDF")
async def _leer_pdf(ruta: str, paginas_max: int = 20) -> str:
    p = Path(ruta)
    if not p.exists(): return f"No encontrado: {ruta}"
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(p))
        partes = [f"--- Pag {i+1} ---\n{(page.extract_text() or '').strip()}"
                  for i, page in enumerate(reader.pages[:paginas_max])]
        return f"PDF: {p.name} ({len(reader.pages)} pags)\n\n" + "\n\n".join(partes)
    except ImportError: return "pip install pypdf"
    except Exception as e: return f"Error: {e}"

@op("leer_docx", "archivos", "Extrae texto de un Word .docx")
async def _leer_docx(ruta: str) -> str:
    p = Path(ruta)
    if not p.exists(): return f"No encontrado: {ruta}"
    try:
        from docx import Document
        doc = Document(str(p))
        parrafos = [para.text for para in doc.paragraphs if para.text.strip()]
        return f"DOCX: {p.name} ({len(parrafos)} parrafos)\n\n" + "\n\n".join(parrafos)
    except ImportError: return "pip install python-docx"
    except Exception as e: return f"Error: {e}"

@op("leer_xlsx", "archivos", "Lee un Excel .xlsx")
async def _leer_xlsx(ruta: str, hoja: str = "", filas: int = 20) -> str:
    p = Path(ruta)
    if not p.exists(): return f"No encontrado: {ruta}"
    try:
        import openpyxl
        wb = openpyxl.load_workbook(str(p), read_only=True, data_only=True)
        ws = wb[hoja] if hoja else wb.active
        rows = [[str(c) if c is not None else "" for c in row]
                for i, row in enumerate(ws.iter_rows(values_only=True)) if i < filas]
        return json.dumps({"hoja": ws.title, "datos": rows}, indent=2, ensure_ascii=False)
    except ImportError: return "pip install openpyxl"
    except Exception as e: return f"Error: {e}"

@op("leer_lineas", "archivos", "Lee lineas especificas de un archivo (inicio a fin)")
async def _leer_lineas(ruta: str, inicio: int = 1, fin: int = 50) -> str:
    p = Path(ruta)
    if not p.exists(): return f"No encontrado: {ruta}"
    lineas = p.read_text(encoding="utf-8", errors="replace").splitlines()
    sel = lineas[inicio-1:fin]
    return "\n".join(f"{inicio+i}: {l}" for i, l in enumerate(sel))

@op("contar_lineas_archivo", "archivos", "Cuenta lineas y caracteres de un archivo")
async def _contar_lineas_archivo(ruta: str) -> str:
    p = Path(ruta)
    if not p.exists(): return f"No encontrado: {ruta}"
    txt = p.read_text(encoding="utf-8", errors="replace")
    return json.dumps({"archivo": p.name, "lineas": len(txt.splitlines()), "chars": len(txt)}, indent=2)

# ── ARCHIVOS: escritura ───────────────────────────────────────────────────────

@op("guardar_archivo", "archivos", "Guarda texto en un archivo")
async def _guardar_archivo(ruta: str, contenido: str) -> str:
    p = Path(ruta); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(contenido, encoding="utf-8")
    return f"Guardado: {ruta} ({len(contenido):,} chars)"

@op("guardar_json", "archivos", "Guarda un objeto como JSON formateado")
async def _guardar_json(ruta: str, datos: str) -> str:
    try: obj = json.loads(datos)
    except json.JSONDecodeError as e: return f"JSON invalido: {e}"
    p = Path(ruta); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return f"JSON guardado: {ruta}"

@op("guardar_csv", "archivos", "Guarda una lista de dicts como CSV")
async def _guardar_csv(ruta: str, datos: str) -> str:
    try: rows = json.loads(datos)
    except Exception as e: return f"datos debe ser JSON array: {e}"
    if not rows: return "Lista vacia"
    p = Path(ruta); p.parent.mkdir(parents=True, exist_ok=True)
    with open(str(p), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    return f"CSV guardado: {ruta} ({len(rows)} filas)"

@op("agregar_linea", "archivos", "Agrega texto al final de un archivo")
async def _agregar_linea(ruta: str, texto: str) -> str:
    p = Path(ruta); p.parent.mkdir(parents=True, exist_ok=True)
    with open(str(p), "a", encoding="utf-8") as f:
        f.write(texto + "\n")
    return f"Linea agregada a: {ruta}"

@op("reemplazar_texto", "archivos", "Reemplaza texto en un archivo")
async def _reemplazar_texto(ruta: str, buscar: str, reemplazar: str) -> str:
    p = Path(ruta)
    if not p.exists(): return f"No encontrado: {ruta}"
    original = p.read_text(encoding="utf-8", errors="replace")
    nuevo = original.replace(buscar, reemplazar)
    p.write_text(nuevo, encoding="utf-8")
    return f"Reemplazadas {original.count(buscar)} ocurrencias en {ruta}"

# ── ARCHIVOS: operaciones ─────────────────────────────────────────────────────

@op("eliminar_archivo", "archivos", "Elimina un archivo")
async def _eliminar_archivo(ruta: str) -> str:
    p = Path(ruta)
    if not p.exists(): return f"No existe: {ruta}"
    p.unlink(); return f"Eliminado: {ruta}"

@op("mover_archivo", "archivos", "Mueve o renombra un archivo")
async def _mover_archivo(origen: str, destino: str) -> str:
    o = Path(origen)
    if not o.exists(): return f"Origen no existe: {origen}"
    Path(destino).parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(o), destino)
    return f"Movido: {origen} -> {destino}"

@op("copiar_archivo", "archivos", "Copia un archivo")
async def _copiar_archivo(origen: str, destino: str) -> str:
    o = Path(origen)
    if not o.exists(): return f"Origen no existe: {origen}"
    Path(destino).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(o), destino)
    return f"Copiado: {origen} -> {destino}"

@op("info_archivo", "archivos", "Devuelve metadatos del archivo")
async def _info_archivo(ruta: str) -> str:
    p = Path(ruta)
    if not p.exists(): return f"No encontrado: {ruta}"
    s = p.stat()
    return json.dumps({
        "nombre": p.name, "tamano_kb": round(s.st_size/1024, 2),
        "creado": str(datetime.datetime.fromtimestamp(s.st_ctime)),
        "modificado": str(datetime.datetime.fromtimestamp(s.st_mtime)),
    }, indent=2)

# ── DIRECTORIOS ───────────────────────────────────────────────────────────────

@op("listar_archivos", "directorios", "Lista archivos con patron glob")
async def _listar_archivos(directorio: str, patron: str = "*", recursivo: bool = False) -> str:
    d = Path(directorio)
    if not d.exists(): return f"No encontrado: {directorio}"
    fn = d.rglob if recursivo else d.glob
    items = sorted([f for f in fn(patron) if f.is_file()])
    res = [{"nombre": f.name, "ruta": str(f), "kb": round(f.stat().st_size/1024, 1)} for f in items[:100]]
    return json.dumps({"total": len(items), "archivos": res}, indent=2, ensure_ascii=False)

@op("listar_directorio", "directorios", "Arbol de directorios")
async def _listar_directorio(directorio: str, profundidad: int = 2) -> str:
    d = Path(directorio)
    if not d.exists(): return f"No encontrado: {directorio}"
    profundidad = min(max(1, profundidad), 5)
    lineas = [f"[DIR] {d.name}/"]
    def _walk(path, nivel):
        if nivel > profundidad: return
        try: items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        except PermissionError: return
        for item in items[:40]:
            pre = "  " * nivel
            if item.is_dir():
                lineas.append(f"{pre}[D] {item.name}/"); _walk(item, nivel+1)
            else:
                lineas.append(f"{pre}[F] {item.name} ({round(item.stat().st_size/1024,1)} KB)")
    _walk(d, 1)
    return "\n".join(lineas)

@op("crear_directorio", "directorios", "Crea un directorio")
async def _crear_directorio(ruta: str) -> str:
    Path(ruta).mkdir(parents=True, exist_ok=True)
    return f"Creado: {ruta}"

@op("eliminar_directorio", "directorios", "Elimina un directorio")
async def _eliminar_directorio(ruta: str) -> str:
    p = Path(ruta)
    if not p.exists(): return f"No existe: {ruta}"
    shutil.rmtree(str(p))
    return f"Eliminado: {ruta}"

# ── BUSQUEDA ──────────────────────────────────────────────────────────────────

@op("buscar_texto", "busqueda", "Busca texto en archivos de un directorio")
async def _buscar_texto(directorio: str, texto: str, extension: str = "*") -> str:
    d = Path(directorio)
    if not d.exists(): return f"No encontrado: {directorio}"
    patron = f"*.{extension}" if extension != "*" else "*"
    resultados = []
    for f in d.rglob(patron):
        if not f.is_file(): continue
        try:
            for i, linea in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if texto.lower() in linea.lower():
                    resultados.append({"archivo": str(f), "linea": i, "contenido": linea.strip()})
                    if len(resultados) >= 50: break
        except Exception: continue
        if len(resultados) >= 50: break
    return json.dumps({"resultados": resultados}, indent=2, ensure_ascii=False)

@op("buscar_clase", "busqueda", "Extrae el codigo de una clase Python")
async def _buscar_clase(ruta: str, nombre_clase: str) -> str:
    p = Path(ruta)
    if not p.exists(): return f"No encontrado: {ruta}"
    contenido = p.read_text(encoding="utf-8", errors="replace")
    match = re.search(fr"(class\s+{re.escape(nombre_clase)}[\s\S]*?)(?=\nclass\s|\Z)", contenido)
    return match.group(1).strip() if match else f"Clase '{nombre_clase}' no encontrada"

# ── GIT ───────────────────────────────────────────────────────────────────────

def _git(args: list, cwd: str) -> str:
    try:
        r = subprocess.run(["git", "-C", cwd] + args, capture_output=True,
                           text=True, timeout=30, encoding="utf-8", errors="replace")
        return r.stdout.strip() or r.stderr.strip() or "(sin salida)"
    except Exception as e: return f"Error git: {e}"

@op("git_status", "git", "Estado del repositorio git")
async def _git_status(directorio: str) -> str:
    rama = _git(["branch", "--show-current"], directorio)
    estado = _git(["status", "--short"], directorio)
    commits = _git(["log", "--oneline", "-5"], directorio)
    return f"Rama: {rama}\nCambios:\n{estado or '(limpio)'}\n\nUltimos commits:\n{commits}"

@op("git_log", "git", "Historial de commits")
async def _git_log(directorio: str, n: int = 15) -> str:
    return _git(["log", f"-{min(n,50)}", "--format=%h %ad %an: %s", "--date=short"], directorio)

@op("git_diff", "git", "Diferencias no commiteadas")
async def _git_diff(directorio: str, archivo: str = "") -> str:
    return _git(["diff"] + ([archivo] if archivo else []), directorio) or "Sin cambios"

@op("git_commit_push", "git", "Add + Commit + Push de los archivos")
async def _git_commit_push_op(directorio: str, mensaje: str, rama: str = "main") -> str:
    a = _git(["add", "."], directorio)
    c = _git(["commit", "-m", mensaje], directorio)
    p = _git(["push", "origin", rama], directorio)
    return f"Add: {a}\nCommit: {c}\nPush: {p}"

# ── SISTEMA ───────────────────────────────────────────────────────────────────

@op("ejecutar_cmd", "sistema", "Ejecuta un comando de sistema (timeout 60s)")
async def _ejecutar_cmd(comando: str, directorio: str = ".") -> str:
    try:
        r = subprocess.run(comando, shell=True, cwd=directorio, capture_output=True,
                           text=True, timeout=60, encoding="utf-8", errors="replace")
        return json.dumps({"rc": r.returncode, "stdout": r.stdout[:3000], "stderr": r.stderr[:500]}, indent=2)
    except Exception as e: return f"Error: {e}"

@op("tiempo_actual", "sistema", "Devuelve fecha y hora actual")
async def _tiempo_actual() -> str:
    return json.dumps({"iso": datetime.datetime.now().isoformat()}, indent=2)

@op("uuid_gen", "sistema", "Genera un UUID4")
async def _uuid_gen_op() -> str:
    return str(uuid.uuid4())

# ── CONFIG ────────────────────────────────────────────────────────────────────

@op("leer_config", "config", "Lee una clave de config.json")
async def _leer_config(clave: str = "") -> str:
    p = Path("config.json")
    config = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return json.dumps(config.get(clave) if clave else config, indent=2, ensure_ascii=False)

@op("escribir_config", "config", "Guarda una clave-valor en config.json")
async def _escribir_config(clave: str, valor: str) -> str:
    p = Path("config.json")
    config = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    config[clave] = valor
    p.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return f"config[{clave}] = {valor}"

# ── OLLAMA & OPENROUTER / IA ──────────────────────────────────────────────────

@op("ollama_chat", "ia", "Consulta de chat estandar")
async def _ollama_chat(mensaje: str, sistema: str = "") -> str:
    return await provider.chat(mensaje, mult=1.0, system=sistema)

@op("ollama_chat_largo", "ia", "Consulta de chat de alta complejidad (900s timeout)")
async def _ollama_chat_largo(mensaje: str, sistema: str = "") -> str:
    return await provider.chat(mensaje, mult=3.0, system=sistema)

# ── POLYDIM / IA ESPECIALIZADA ────────────────────────────────────────────────

@op("polydim_analizar_proyecto", "polydim", "Analisis profundo de la arquitectura de POLYDIM")
async def _polydim_analizar_proyecto(directorio: str) -> str:
    d = Path(directorio)
    if not d.exists(): return f"No encontrado: {directorio}"
    archivos = [str(f.relative_to(d)) for f in d.rglob("*") if f.is_file()][:50]
    return await provider.chat(
        f"Analiza la arquitectura del proyecto en {directorio}.\nArchivos:\n" + "\n".join(archivos), mult=3.0
    )

@op("polydim_comparar", "polydim", "Compara dos fragmentos de codigo")
async def _polydim_comparar(codigo_a: str, codigo_b: str) -> str:
    return await provider.chat(f"Compara legibilidad y eficiencia:\n\nA:\n{codigo_a}\n\nB:\n{codigo_b}")

@op("polydim_generar_tests", "polydim", "Genera tests unitarios")
async def _polydim_generar_tests(codigo: str) -> str:
    return await provider.chat(f"Genera tests unitarios pytest:\n\n{codigo}", mult=2.0)

@op("polydim_optimizar_complejidad", "polydim", "Optimiza la complejidad Big-O")
async def _polydim_optimizar_complejidad(codigo: str) -> str:
    return await provider.chat(f"Optimiza la complejidad temporal de este codigo:\n\n{codigo}", mult=2.0)

@op("polydim_seguridad_scan", "polydim", "Escanea vulnerabilidades comunes")
async def _polydim_seguridad_scan(codigo: str) -> str:
    return await provider.chat(f"Escanea vulnerabilidades en:\n\n{codigo}", mult=1.5)

@op("resumen_codigo", "polydim", "Resume el proposito de un archivo de codigo")
async def _resumen_codigo(ruta: str) -> str:
    p = Path(ruta)
    if not p.exists(): return f"No encontrado: {ruta}"
    return await provider.chat(f"Resume este codigo:\n\n{p.read_text(encoding='utf-8')[:6000]}")


# ══════════════════════════════════════════════════════════════════════════════
# MCP TOOLS DIRECTAS — 12 herramientas expuestas al cliente
# ══════════════════════════════════════════════════════════════════════════════

@server.tool()
async def ejecutar(operacion: str, args: str = "{}") -> str:
    """
    Dispatcher universal — Ejecuta cualquiera de las 90+ operaciones en OpenRouter/Ollama.
    
    Args:
        operacion: nombre de la operacion (ej: leer_md, git_status, leer_pdf, etc.)
        args: argumentos en formato JSON string. Ej: '{"ruta": "C:/doc.md"}'
    """
    try:
        parsed = json.loads(args) if args.strip() else {}
    except json.JSONDecodeError as e:
        return f"args debe ser JSON valido: {e}"

    entry = REGISTRY.get(operacion)
    if not entry:
        return f"Operacion '{operacion}' no encontrada. Usar listar_operaciones() para ver todas."

    try:
        return await entry["fn"](**parsed)
    except Exception as e:
        return f"Error en '{operacion}': {str(e)}"


@server.tool()
async def listar_operaciones(categoria: str = "") -> str:
    """Lista todas las operaciones del dispatcher perezoso, por categoria."""
    ops = {k: v for k, v in REGISTRY.items() if not categoria or v["categoria"] == categoria}
    por_cat: dict = {}
    for nombre, entry in sorted(ops.items()):
        cat = entry["categoria"]
        por_cat.setdefault(cat, []).append(f"  - {nombre}: {entry['desc'][:70]}")
    
    lineas = [f"{len(ops)} operaciones registradas:"]
    for cat, items in sorted(por_cat.items()):
        lineas.append(f"\n[{cat.upper()}]")
        lineas.extend(items)
    return "\n".join(lineas)


@server.tool()
async def ollama_chat(mensaje: str, sistema: str = "") -> str:
    """Chat directo con Qwen OpenRouter / Ollama (alias rápido)."""
    return await provider.chat(mensaje, mult=1.0, system=sistema)

@server.tool()
async def ask_kimi(mensaje: str, sistema: str = "Eres Kimi, el lector experto. Recibes un texto o pregunta y respondes directamente basandote en tu inmenso contexto.") -> str:
    """Delega una tarea de lectura profunda, extracción o resumen a Kimi (moonshotai/moonshot-v1-32k)."""
    return await provider.chat(mensaje, mult=2.0, system=sistema, model_override="moonshotai/moonshot-v1-32k")

@server.tool()
async def ask_zia(mensaje: str, sistema: str = "Eres Z.ia (GLM), el estructurador. Analiza los datos y formatea la salida de forma estricta y rápida.") -> str:
    """Delega una tarea rápida de formateo, análisis o estructura de datos a Z.ia (zhipu/glm-4)."""
    return await provider.chat(mensaje, mult=1.0, system=sistema, model_override="zhipu/glm-4")

@server.tool()
async def ask_math(mensaje: str, sistema: str = "Eres Qwen Math, un validador simbólico estricto. Resuelve, demuestra o valida el problema topológico o matemático.") -> str:
    """Delega una validación matemática simbólica o problema de topología a Qwen (qwen/qwen-2.5-72b-instruct)."""
    return await provider.chat(mensaje, mult=2.5, system=sistema, model_override="qwen/qwen-2.5-72b-instruct")


@server.tool()
async def ask_gemma(mensaje: str, sistema: str = "Eres Gemma 2, experta en redacción creativa y síntesis. Tu objetivo es resumir y redactar con claridad.") -> str:
    """Delega una tarea de redacción o síntesis a Gemma (google/gemma-2-27b-it)."""
    return await provider.chat(mensaje, mult=1.0, system=sistema, model_override="google/gemma-2-27b-it")

@server.tool()
async def ask_chatgpt(mensaje: str, sistema: str = "Eres ChatGPT (GPT-4o-mini), asistente general versátil. Resuelve dudas generales y asiste en código genérico.") -> str:
    """Delega una tarea general o de consulta versátil a ChatGPT (openai/gpt-4o-mini)."""
    return await provider.chat(mensaje, mult=1.5, system=sistema, model_override="openai/gpt-4o-mini")

@server.tool()
async def ask_deepseek(mensaje: str, sistema: str = "Eres DeepSeek R1, un razonador profundo especializado en pensamiento paso a paso, código complejo y análisis exhaustivo. Muestra tu cadena de razonamiento antes de la respuesta final.") -> str:
    """Delega razonamiento profundo, código complejo o análisis paso a paso a DeepSeek (deepseek/deepseek-r1:free)."""
    return await provider.chat(mensaje, mult=3.0, system=sistema, model_override="deepseek/deepseek-r1:free")


@server.tool()
async def ask_claude(mensaje: str, sistema: str = "Eres Claude (Sonnet 4), experto en arquitectura de software, revisión de código, razonamiento complejo y análisis detallado. Priorizas claridad, corrección y diseño limpio.") -> str:
    """Delega arquitectura, revisión de código o razonamiento complejo a Claude (anthropic/claude-sonnet-4). NOTA: Este modelo tiene costo por token."""
    return await provider.chat(mensaje, mult=2.5, system=sistema, model_override="anthropic/claude-sonnet-4")


@server.tool()
async def mostrar_metodos() -> str:
    """Muestra la lista de herramientas directas del MCP OpenRouter."""
    tools_dict = getattr(server, "_tools", {})
    mcp = sorted(tools_dict.keys()) if tools_dict else []
    return f"Herramientas OpenRouter ({len(mcp)}):\n" + "\n".join(f"  - {t}" for t in mcp)


if __name__ == "__main__":
    _log(f"Servidor OpenRouter iniciado con {len(REGISTRY)} operaciones.")
    server.run()

