#!/usr/bin/env python3
"""
Servidor MCP Antigravity V7 - Consolidado Estable
==================================================
ARQUITECTURA: igual a V6 (tools directas @server.tool(), sin dispatcher)
FUNCIONES:    V6 base + extras de V11 (git, archivos avanzados, sistema, texto, IA)

pip install mcp httpx pypdf python-docx pandas openpyxl markdown
"""

import json
import uuid
import csv
import re
import os
import shutil
import datetime
import subprocess
import hashlib
import httpx
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# ─── Configuracion ────────────────────────────────────────────────────────────
MODEL  = "deepseek-r1:14b"
server = FastMCP("Antigravity-Server-V7")
CONFIG = {
    "ollama_url": "http://localhost:11434",
    "default_model": MODEL,
    "timeout_base": 300.0,
}

class OllamaProvider:
    def __init__(self):
        self.base_timeout = CONFIG["timeout_base"]
        self.client = httpx.AsyncClient()

    async def chat_con_timeout(self, prompt: str, multiplicador: float = 1.0) -> str:
        """Ejecuta una consulta ajustando el timeout dinamicamente"""
        tiempo_final = self.base_timeout * multiplicador
        try:
            resp = await self.client.post(
                f"{CONFIG['ollama_url']}/api/generate",
                json={"model": CONFIG["default_model"], "prompt": prompt, "stream": False},
                timeout=tiempo_final
            )
            return resp.json().get("response", "Sin respuesta")
        except httpx.TimeoutException:
            return f"Timeout ({tiempo_final:.0f}s). Proba un modelo mas pequeno."
        except Exception as e:
            return f"Error tecnico: {str(e)}"

    async def chat(self, prompt: str, multiplicador: float = 1.0, sistema: str = "") -> str:
        """Chat con mensajes (system/user) via /api/chat"""
        tiempo_final = self.base_timeout * multiplicador
        messages = []
        if sistema:
            messages.append({"role": "system", "content": sistema})
        messages.append({"role": "user", "content": prompt})
        try:
            resp = await self.client.post(
                f"{CONFIG['ollama_url']}/api/chat",
                json={"model": CONFIG["default_model"], "messages": messages, "stream": False},
                timeout=tiempo_final,
            )
            return resp.json().get("message", {}).get("content", "Sin respuesta")
        except httpx.TimeoutException:
            return f"Timeout ({tiempo_final:.0f}s)."
        except Exception as e:
            return f"Error: {str(e)}"


# ─── Helper Git (interno, no expuesto como tool) ──────────────────────────────
def _git(args: list, cwd: str) -> str:
    try:
        r = subprocess.run(["git", "-C", cwd] + args, capture_output=True,
                           text=True, timeout=30, encoding="utf-8", errors="replace")
        return r.stdout.strip() or r.stderr.strip() or "(sin salida)"
    except Exception as e:
        return f"Error git: {e}"


# Instancia del proveedor (al final del archivo, antes del if __main__)
provider = OllamaProvider()


# ══════════════════════════════════════════════════════════════════════════════
# GRUPO 0: IDENTIDAD
# ══════════════════════════════════════════════════════════════════════════════

@server.tool()
async def mostrar_metodos() -> str:
    """Muestra todas las herramientas disponibles y sus descripciones"""
    tools_dict = getattr(server, "_tools", {})
    mcp = sorted(tools_dict.keys()) if tools_dict else []
    return f"Servidor Antigravity V7 — {len(mcp)} herramientas:\n" + "\n".join(f"  - {t}" for t in mcp)

@server.tool()
async def uuid_gen() -> str:
    """Genera un identificador unico UUID4"""
    return str(uuid.uuid4())


# ══════════════════════════════════════════════════════════════════════════════
# GRUPO 1: ARCHIVOS — Lectura
# ══════════════════════════════════════════════════════════════════════════════

@server.tool()
async def leer_archivo(ruta: str) -> str:
    """Lee el contenido completo de cualquier archivo de texto"""
    try:
        return Path(ruta).read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error: {str(e)}"

@server.tool()
async def leer_md(ruta: str) -> str:
    """Lee y devuelve el contenido de un archivo Markdown"""
    try:
        return Path(ruta).read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error: {str(e)}"

@server.tool()
async def leer_csv(ruta: str) -> str:
    """Lee un CSV y retorna sus primeras 20 filas en formato JSON"""
    try:
        with open(ruta, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            rows = [row for i, row in enumerate(reader) if i < 20]
        return json.dumps(rows, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"Error: {str(e)}"

@server.tool()
async def leer_lineas(ruta: str, inicio: int = 1, fin: int = 50) -> str:
    """Lee un rango especifico de lineas de un archivo (inicio y fin son 1-indexed)"""
    try:
        lineas = Path(ruta).read_text(encoding="utf-8", errors="replace").splitlines()
        sel = lineas[inicio-1:fin]
        return "\n".join(f"{inicio+i}: {l}" for i, l in enumerate(sel))
    except Exception as e:
        return f"Error: {str(e)}"

@server.tool()
async def leer_json(ruta: str) -> str:
    """Lee y valida un archivo JSON, devuelve formateado"""
    try:
        return json.dumps(json.loads(Path(ruta).read_text(encoding="utf-8")), indent=2, ensure_ascii=False)
    except json.JSONDecodeError as e:
        return f"JSON invalido: {e}"
    except Exception as e:
        return f"Error: {str(e)}"

@server.tool()
async def leer_pdf(ruta: str, paginas_max: int = 20) -> str:
    """Extrae texto de un PDF (requiere pypdf)"""
    p = Path(ruta)
    if not p.exists(): return f"No encontrado: {ruta}"
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(p))
        partes = [f"--- Pag {i+1} ---\n{(page.extract_text() or '').strip()}"
                  for i, page in enumerate(reader.pages[:paginas_max])]
        return f"PDF: {p.name} ({len(reader.pages)} pags)\n\n" + "\n\n".join(partes)
    except ImportError:
        return "pip install pypdf"
    except Exception as e:
        return f"Error: {e}"

@server.tool()
async def leer_docx(ruta: str) -> str:
    """Extrae texto de un archivo Word .docx (requiere python-docx)"""
    p = Path(ruta)
    if not p.exists(): return f"No encontrado: {ruta}"
    try:
        from docx import Document
        doc = Document(str(p))
        parrafos = [para.text for para in doc.paragraphs if para.text.strip()]
        return f"DOCX: {p.name} ({len(parrafos)} parrafos)\n\n" + "\n\n".join(parrafos)
    except ImportError:
        return "pip install python-docx"
    except Exception as e:
        return f"Error: {e}"

@server.tool()
async def leer_xlsx(ruta: str, hoja: str = "", filas: int = 20) -> str:
    """Lee un archivo Excel .xlsx (requiere openpyxl)"""
    p = Path(ruta)
    if not p.exists(): return f"No encontrado: {ruta}"
    try:
        import openpyxl
        wb = openpyxl.load_workbook(str(p), read_only=True, data_only=True)
        ws = wb[hoja] if hoja else wb.active
        rows = [[str(c) if c is not None else "" for c in row]
                for i, row in enumerate(ws.iter_rows(values_only=True)) if i < filas]
        return json.dumps({"hoja": ws.title, "datos": rows}, indent=2, ensure_ascii=False)
    except ImportError:
        return "pip install openpyxl"
    except Exception as e:
        return f"Error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# GRUPO 2: ARCHIVOS — Escritura y Operaciones
# ══════════════════════════════════════════════════════════════════════════════

@server.tool()
async def guardar_archivo(ruta: str, contenido: str) -> str:
    """Guarda texto en un archivo, creando carpetas si no existen"""
    try:
        path = Path(ruta)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contenido, encoding="utf-8")
        return f"Guardado: {ruta} ({len(contenido):,} chars)"
    except Exception as e:
        return f"Error: {str(e)}"

@server.tool()
async def agregar_linea(ruta: str, texto: str) -> str:
    """Agrega una linea al final de un archivo"""
    try:
        p = Path(ruta)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(str(p), "a", encoding="utf-8") as f:
            f.write(texto + "\n")
        return f"Linea agregada a: {ruta}"
    except Exception as e:
        return f"Error: {str(e)}"

@server.tool()
async def reemplazar_texto(ruta: str, buscar: str, reemplazar: str) -> str:
    """Reemplaza todas las ocurrencias de un texto en un archivo"""
    try:
        p = Path(ruta)
        original = p.read_text(encoding="utf-8", errors="replace")
        count = original.count(buscar)
        p.write_text(original.replace(buscar, reemplazar), encoding="utf-8")
        return f"Reemplazadas {count} ocurrencias en {ruta}"
    except Exception as e:
        return f"Error: {str(e)}"

@server.tool()
async def eliminar_archivo(ruta: str) -> str:
    """Elimina un archivo del sistema"""
    try:
        Path(ruta).unlink()
        return f"Eliminado: {ruta}"
    except Exception as e:
        return f"Error: {str(e)}"

@server.tool()
async def mover_archivo(origen: str, destino: str) -> str:
    """Mueve o renombra un archivo"""
    try:
        o = Path(origen)
        if not o.exists(): return f"Origen no existe: {origen}"
        Path(destino).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(o), destino)
        return f"Movido: {origen} -> {destino}"
    except Exception as e:
        return f"Error: {str(e)}"

@server.tool()
async def copiar_archivo(origen: str, destino: str) -> str:
    """Copia un archivo preservando metadatos"""
    try:
        o = Path(origen)
        if not o.exists(): return f"Origen no existe: {origen}"
        Path(destino).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(o), destino)
        return f"Copiado: {origen} -> {destino}"
    except Exception as e:
        return f"Error: {str(e)}"

@server.tool()
async def informacion_archivo(ruta: str) -> str:
    """Devuelve tamano, fecha de creacion y ultima modificacion del archivo"""
    try:
        p = Path(ruta)
        s = p.stat()
        return json.dumps({
            "nombre": p.name,
            "extension": p.suffix,
            "tamano_bytes": s.st_size,
            "tamano_kb": round(s.st_size/1024, 2),
            "creado": str(datetime.datetime.fromtimestamp(s.st_ctime)),
            "modificado": str(datetime.datetime.fromtimestamp(s.st_mtime)),
            "md5": hashlib.md5(p.read_bytes()).hexdigest(),
        }, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@server.tool()
async def hash_archivo(ruta: str) -> str:
    """Genera hash MD5 y SHA256 de un archivo"""
    try:
        data = Path(ruta).read_bytes()
        return json.dumps({
            "md5": hashlib.md5(data).hexdigest(),
            "sha256": hashlib.sha256(data).hexdigest(),
            "tamano_bytes": len(data),
        }, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@server.tool()
async def contar_lineas(ruta: str) -> str:
    """Cuenta lineas, palabras, comentarios y bytes de un archivo"""
    try:
        lineas = Path(ruta).read_text(encoding="utf-8", errors="replace").splitlines()
        vacias = sum(1 for l in lineas if not l.strip())
        comentarios = sum(1 for l in lineas if l.strip().startswith(("#", "//", "/*", "*")))
        return json.dumps({
            "lineas_total": len(lineas),
            "lineas_vacias": vacias,
            "lineas_comentario": comentarios,
            "lineas_codigo": len(lineas) - vacias - comentarios,
        }, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"


# ══════════════════════════════════════════════════════════════════════════════
# GRUPO 3: DIRECTORIOS
# ══════════════════════════════════════════════════════════════════════════════

@server.tool()
async def listar_directorio(directorio: str, profundidad: int = 2) -> str:
    """Muestra el arbol de archivos y carpetas hasta una profundidad indicada"""
    d = Path(directorio)
    if not d.exists(): return f"No encontrado: {directorio}"
    profundidad = min(max(1, profundidad), 5)
    lineas = [f"[DIR] {d.name}/"]
    def _walk(path, nivel):
        if nivel > profundidad: return
        try:
            items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        except PermissionError:
            return
        for item in items[:40]:
            pre = "  " * nivel
            if item.is_dir():
                lineas.append(f"{pre}[D] {item.name}/")
                _walk(item, nivel+1)
            else:
                lineas.append(f"{pre}[F] {item.name} ({round(item.stat().st_size/1024, 1)} KB)")
    _walk(d, 1)
    return "\n".join(lineas)

@server.tool()
async def listar_archivos(directorio: str, patron: str = "*", recursivo: bool = False) -> str:
    """Lista archivos con patron glob. Si recursivo=true busca en subdirectorios"""
    try:
        d = Path(directorio)
        if not d.exists(): return f"No encontrado: {directorio}"
        fn = d.rglob if recursivo else d.glob
        items = sorted([f for f in fn(patron) if f.is_file()])
        res = [{"nombre": f.name, "ruta": str(f), "kb": round(f.stat().st_size/1024, 1)} for f in items[:100]]
        return json.dumps({"total": len(items), "archivos": res}, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"Error: {str(e)}"

@server.tool()
async def archivos_recientes(directorio: str, n: int = 10) -> str:
    """Lista los N archivos modificados mas recientemente en un directorio"""
    try:
        d = Path(directorio)
        if not d.exists(): return f"No encontrado: {directorio}"
        archivos = sorted((f for f in d.rglob("*") if f.is_file()),
                          key=lambda f: f.stat().st_mtime, reverse=True)[:n]
        res = [{"ruta": str(f), "modificado": str(datetime.datetime.fromtimestamp(f.stat().st_mtime))}
               for f in archivos]
        return json.dumps(res, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"Error: {str(e)}"

@server.tool()
async def tamano_directorio(directorio: str) -> str:
    """Calcula el tamano total en MB y cantidad de archivos de un directorio"""
    try:
        d = Path(directorio)
        if not d.exists(): return f"No encontrado: {directorio}"
        total = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        archivos = sum(1 for f in d.rglob("*") if f.is_file())
        return json.dumps({"archivos": archivos, "tamano_mb": round(total/1024/1024, 2)}, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"


# ══════════════════════════════════════════════════════════════════════════════
# GRUPO 4: BUSQUEDA
# ══════════════════════════════════════════════════════════════════════════════

@server.tool()
async def buscar_texto(directorio: str, texto: str, extension: str = "*") -> str:
    """Busca texto literal en archivos de un directorio. extension ej: 'py', 'md'"""
    try:
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
            except Exception:
                continue
            if len(resultados) >= 50: break
        return json.dumps({"total": len(resultados), "resultados": resultados}, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"Error: {str(e)}"

@server.tool()
async def buscar_regex(directorio: str, patron: str) -> str:
    """Busca archivos cuyo NOMBRE coincida con un patron Regex"""
    try:
        archivos = [str(f) for f in Path(directorio).rglob("*") if re.search(patron, f.name)]
        return json.dumps({"total": len(archivos), "archivos": archivos[:50]}, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"Error: {str(e)}"

@server.tool()
async def buscar_funcion(ruta: str, nombre_funcion: str) -> str:
    """Extrae el codigo de una funcion especifica de un archivo Python"""
    try:
        contenido = Path(ruta).read_text(encoding="utf-8", errors="replace")
        match = re.search(
            fr"((?:async\s+)?def\s+{re.escape(nombre_funcion)}[\s\S]*?)(?=\n(?:async\s+)?(?:def|class)\s|\Z)",
            contenido
        )
        return match.group(1).strip() if match else f"Funcion '{nombre_funcion}' no encontrada"
    except Exception as e:
        return f"Error: {str(e)}"

@server.tool()
async def buscar_clase(ruta: str, nombre_clase: str) -> str:
    """Extrae el codigo de una clase entera de un archivo Python"""
    try:
        contenido = Path(ruta).read_text(encoding="utf-8", errors="replace")
        match = re.search(fr"(class\s+{re.escape(nombre_clase)}[\s\S]*?)(?=\nclass\s|\Z)", contenido)
        return match.group(1).strip() if match else f"Clase '{nombre_clase}' no encontrada"
    except Exception as e:
        return f"Error: {str(e)}"

@server.tool()
async def listar_funciones(ruta: str) -> str:
    """Lista todas las funciones definidas en un archivo Python con su numero de linea"""
    try:
        fns = []
        for i, l in enumerate(Path(ruta).read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            m = re.match(r"\s*(async\s+)?def\s+(\w+)\s*\(", l)
            if m: fns.append({"linea": i, "nombre": m.group(2), "async": bool(m.group(1))})
        return json.dumps({"total": len(fns), "funciones": fns}, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"Error: {str(e)}"

@server.tool()
async def buscar_todo(directorio: str) -> str:
    """Encuentra comentarios TODO, FIXME y HACK en archivos Python de un directorio"""
    try:
        d = Path(directorio)
        if not d.exists(): return f"No encontrado: {directorio}"
        patron = re.compile(r"(TODO|FIXME|HACK|XXX|BUG|NOTE)[\s:]+(.+)", re.IGNORECASE)
        resultados = []
        for f in d.rglob("*.py"):
            if not f.is_file(): continue
            try:
                for i, linea in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    m = patron.search(linea)
                    if m:
                        resultados.append({"tipo": m.group(1).upper(), "archivo": str(f),
                                           "linea": i, "texto": m.group(2).strip()})
            except Exception:
                continue
        return json.dumps({"total": len(resultados), "items": resultados}, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"Error: {str(e)}"


# ══════════════════════════════════════════════════════════════════════════════
# GRUPO 5: GIT
# ══════════════════════════════════════════════════════════════════════════════

@server.tool()
async def git_status(directorio: str) -> str:
    """Estado del repositorio git: rama, cambios y ultimos 5 commits"""
    rama = _git(["branch", "--show-current"], directorio)
    estado = _git(["status", "--short"], directorio)
    commits = _git(["log", "--oneline", "-5"], directorio)
    return f"Rama: {rama}\nCambios:\n{estado or '(limpio)'}\n\nUltimos commits:\n{commits}"

@server.tool()
async def git_commit_push(directorio: str, mensaje: str, rama: str = "main") -> str:
    """Hace add + commit + push en un solo paso"""
    add    = _git(["add", "."], directorio)
    commit = _git(["commit", "-m", mensaje], directorio)
    push   = _git(["push", "origin", rama], directorio)
    return f"ADD: {add}\nCOMMIT: {commit}\nPUSH: {push}"

@server.tool()
async def git_log(directorio: str, n: int = 15) -> str:
    """Muestra el historial de commits (hash, fecha, autor, mensaje)"""
    return _git(["log", f"-{min(n, 50)}", "--format=%h %ad %an: %s", "--date=short"], directorio)

@server.tool()
async def git_diff(directorio: str, archivo: str = "") -> str:
    """Muestra las diferencias no commiteadas. Si se pasa archivo, solo ese archivo"""
    args = ["diff"] + ([archivo] if archivo else [])
    return _git(args, directorio) or "Sin cambios"

@server.tool()
async def git_branches(directorio: str) -> str:
    """Lista todas las ramas locales y remotas del repositorio"""
    return _git(["branch", "-a"], directorio)

@server.tool()
async def git_remotes(directorio: str) -> str:
    """Lista los remotos configurados en el repositorio"""
    return _git(["remote", "-v"], directorio)


# ══════════════════════════════════════════════════════════════════════════════
# GRUPO 6: SISTEMA
# ══════════════════════════════════════════════════════════════════════════════

@server.tool()
async def ejecutar_cmd(comando: str, directorio: str = ".") -> str:
    """Ejecuta un comando de sistema y devuelve stdout/stderr (timeout 60s)"""
    try:
        r = subprocess.run(comando, shell=True, cwd=directorio, capture_output=True,
                           text=True, timeout=60, encoding="utf-8", errors="replace")
        return json.dumps({"rc": r.returncode, "stdout": r.stdout[:3000], "stderr": r.stderr[:500]}, indent=2)
    except subprocess.TimeoutExpired:
        return "Timeout (60s)"
    except Exception as e:
        return f"Error: {e}"

@server.tool()
async def ejecutar_python(ruta_script: str, args: str = "") -> str:
    """Ejecuta un script Python y captura su salida (timeout 120s)"""
    try:
        r = subprocess.run(f'python -X utf8 "{ruta_script}" {args}', shell=True,
                           capture_output=True, text=True, timeout=120,
                           encoding="utf-8", errors="replace")
        return json.dumps({"rc": r.returncode, "stdout": r.stdout[:4000], "stderr": r.stderr[:1000]}, indent=2)
    except subprocess.TimeoutExpired:
        return "Timeout (120s)"
    except Exception as e:
        return f"Error: {e}"

@server.tool()
async def tiempo_actual() -> str:
    """Devuelve la fecha y hora actual del sistema"""
    now = datetime.datetime.now()
    return json.dumps({
        "iso": now.isoformat(),
        "fecha": now.strftime("%Y-%m-%d"),
        "hora": now.strftime("%H:%M:%S"),
        "dia_semana": now.strftime("%A"),
    }, indent=2)

@server.tool()
async def info_sistema() -> str:
    """Informacion del sistema operativo, version de Python y directorio actual"""
    import platform, sys
    return json.dumps({
        "os": platform.system(),
        "os_version": platform.version()[:80],
        "python": sys.version[:30],
        "cwd": os.getcwd(),
        "home": str(Path.home()),
    }, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# GRUPO 7: TEXTO
# ══════════════════════════════════════════════════════════════════════════════

@server.tool()
async def contar_palabras(ruta: str) -> str:
    """Cuenta palabras, lineas, caracteres y estima tokens de un archivo"""
    try:
        texto = Path(ruta).read_text(encoding="utf-8", errors="replace")
        return json.dumps({
            "caracteres": len(texto),
            "palabras": len(texto.split()),
            "lineas": len(texto.splitlines()),
            "tokens_estimados": len(texto) // 4,
        }, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@server.tool()
async def extraer_urls(ruta: str) -> str:
    """Extrae todas las URLs de un archivo de texto"""
    try:
        texto = Path(ruta).read_text(encoding="utf-8", errors="replace")
        urls = list(set(re.findall(r"https?://[^\s'\"<>]+", texto)))
        return json.dumps({"total": len(urls), "urls": urls}, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"Error: {str(e)}"

@server.tool()
async def generar_tabla_md(datos: str) -> str:
    """Convierte un JSON array de objetos a tabla Markdown. datos = '[{...}, ...]'"""
    try:
        rows = json.loads(datos)
        if not rows: return "Lista vacia"
        headers = list(rows[0].keys())
        sep = "|" + "|".join(["---"] * len(headers)) + "|"
        head = "|" + "|".join(headers) + "|"
        filas = "\n".join("|" + "|".join(str(r.get(h, "")) for h in headers) + "|" for r in rows)
        return f"{head}\n{sep}\n{filas}"
    except Exception as e:
        return f"Error: {e}"

@server.tool()
async def convertir_md_html(ruta_md: str, ruta_html: str = "") -> str:
    """Convierte un archivo Markdown a HTML (requiere pip install markdown)"""
    p = Path(ruta_md)
    if not p.exists(): return f"No encontrado: {ruta_md}"
    try:
        import markdown as md_lib
        txt = p.read_text(encoding="utf-8", errors="replace")
        body = md_lib.markdown(txt, extensions=["tables", "fenced_code", "toc"])
        html = f'<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8"><title>{p.stem}</title></head><body>{body}</body></html>'
        out = Path(ruta_html) if ruta_html else p.with_suffix(".html")
        out.write_text(html, encoding="utf-8")
        return f"HTML generado: {out} ({len(html):,} bytes)"
    except ImportError:
        return "pip install markdown"
    except Exception as e:
        return f"Error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# GRUPO 8: CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@server.tool()
async def leer_config() -> str:
    """Lee el archivo config.json si existe"""
    return Path("config.json").read_text(encoding="utf-8") if Path("config.json").exists() else "{}"

@server.tool()
async def escribir_config(clave: str, valor: str) -> str:
    """Guarda una clave-valor en config.json"""
    try:
        config = json.loads(Path("config.json").read_text(encoding="utf-8")) if Path("config.json").exists() else {}
        config[clave] = valor
        Path("config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        return f"config[{clave}] = {valor}"
    except Exception as e:
        return f"Error: {str(e)}"


# ══════════════════════════════════════════════════════════════════════════════
# GRUPO 9: OLLAMA — IA
# ══════════════════════════════════════════════════════════════════════════════

@server.tool()
async def ollama_chat(mensaje: str, sistema: str = "") -> str:
    """Chat estandar con el modelo Ollama (timeout 300s)"""
    return await provider.chat(mensaje, multiplicador=1.0, sistema=sistema)

@server.tool()
async def ollama_chat_largo(mensaje: str, sistema: str = "") -> str:
    """Chat extendido para razonamiento complejo (timeout 900s)"""
    return await provider.chat(mensaje, multiplicador=3.0, sistema=sistema)

@server.tool()
async def ollama_modelos() -> str:
    """Lista los modelos descargados en Ollama local"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{CONFIG['ollama_url']}/api/tags")
            modelos = [m["name"] for m in resp.json().get("models", [])]
            return json.dumps({"total": len(modelos), "modelos": modelos, "activo": CONFIG["default_model"]}, indent=2)
    except Exception as e:
        return f"Error: {e}"

@server.tool()
async def ollama_estado() -> str:
    """Verifica si Ollama esta respondiendo en localhost:11434"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{CONFIG['ollama_url']}/api/tags")
            return f"Ollama activo. Modelos: {len(resp.json().get('models', []))}"
    except Exception as e:
        return f"Ollama no responde: {e}"

@server.tool()
async def ollama_pull(modelo: str) -> str:
    """Descarga un modelo de Ollama"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{CONFIG['ollama_url']}/api/pull",
                                     json={"name": modelo, "stream": False})
            return f"Pull '{modelo}': HTTP {resp.status_code}"
    except Exception as e:
        return f"Error: {e}"

@server.tool()
async def ollama_resumir(texto: str, estilo: str = "tecnico") -> str:
    """Resume un texto largo con IA en estilo tecnico, simple o ejecutivo"""
    return await provider.chat(f"Resume en estilo {estilo}, 3-5 puntos clave:\n\n{texto[:6000]}")

@server.tool()
async def ollama_traducir(texto: str, idioma_destino: str = "ingles") -> str:
    """Traduce un texto con IA al idioma indicado"""
    return await provider.chat(f"Traducido al {idioma_destino}:\n\n{texto}")

@server.tool()
async def ollama_generar_codigo(descripcion: str, lenguaje: str = "python") -> str:
    """Genera codigo segun una descripcion. Solo devuelve el codigo"""
    return await provider.chat(
        f"Genera codigo {lenguaje} para: {descripcion}\nSolo el codigo, sin explicaciones.", multiplicador=2.0
    )

@server.tool()
async def ollama_corregir_gramatica(texto: str, idioma: str = "espanol") -> str:
    """Corrige gramatica y ortografia con IA. Solo devuelve el texto corregido"""
    return await provider.chat(f"Corrige gramatica y ortografia en {idioma}. Solo devuelve el texto corregido:\n\n{texto}")


# ══════════════════════════════════════════════════════════════════════════════
# GRUPO 10: POLYDIM — IA Especializada
# ══════════════════════════════════════════════════════════════════════════════

@server.tool()
async def polydim_analizar_proyecto(directorio: str) -> str:
    """Analiza la arquitectura de un proyecto y sugiere mejoras"""
    d = Path(directorio)
    if not d.exists(): return f"No encontrado: {directorio}"
    archivos = [str(f.relative_to(d)) for f in d.rglob("*") if f.is_file()][:60]
    return await provider.chat(
        f"Analiza la arquitectura del proyecto en {directorio}.\n"
        f"Archivos ({len(archivos)}):\n" + "\n".join(archivos) +
        "\n\nEs robusta? Que mejoras sugeris?", multiplicador=3.0
    )

@server.tool()
async def polydim_resumir(texto: str) -> str:
    """Resume un texto largo usando el modelo actual"""
    return await provider.chat(f"Resume esto en 5 puntos clave:\n\n{texto[:5000]}")

@server.tool()
async def polydim_explicar(concepto: str) -> str:
    """Explica un concepto tecnico complejo de forma clara"""
    return await provider.chat(f"Explicame como si fuera experto: {concepto}")

@server.tool()
async def polydim_refactorizar(codigo: str, objetivo: str = "legibilidad") -> str:
    """Refactoriza codigo POLYDIM manteniendo su comportamiento"""
    return await provider.chat(
        f"Refactoriza este codigo para mejorar {objetivo}. Mantene el comportamiento identico. Explica los cambios:\n\n{codigo}",
        multiplicador=2.0
    )

@server.tool()
async def polydim_documentar(codigo: str) -> str:
    """Genera docstrings y comentarios profesionales para el codigo proporcionado"""
    return await provider.chat(f"Agrega docstrings y comentarios a este codigo:\n\n{codigo}", multiplicador=2.0)

@server.tool()
async def polydim_comparar(codigo_a: str, codigo_b: str) -> str:
    """Compara dos fragmentos de codigo y detalla diferencias de calidad"""
    return await provider.chat(
        f"Compara estos dos fragmentos. Cual es mejor y por que? Evalua legibilidad, eficiencia, correctitud.\nA: {codigo_a}\nB: {codigo_b}",
        multiplicador=1.5
    )

@server.tool()
async def polydim_generar_tests(codigo: str) -> str:
    """Genera tests unitarios pytest para un codigo dado"""
    return await provider.chat(
        f"Genera tests unitarios pytest. Incluye casos normales, borde y nulos:\n\n{codigo}",
        multiplicador=2.0
    )

@server.tool()
async def polydim_optimizar_complejidad(codigo: str) -> str:
    """Analiza la complejidad Big O y propone version optimizada"""
    return await provider.chat(
        f"Analiza la complejidad Big O de este codigo y propone version optimizada:\n\n{codigo}",
        multiplicador=2.0
    )

@server.tool()
async def polydim_seguridad_scan(codigo: str) -> str:
    """Analiza codigo buscando vulnerabilidades de seguridad"""
    return await provider.chat(
        f"Escanea este codigo buscando vulnerabilidades. Reporta ALTA/MEDIA/BAJA + fix:\n\n{codigo}",
        multiplicador=1.5
    )

@server.tool()
async def polydim_doc_gen(codigo: str) -> str:
    """Genera documentacion tecnica completa (docstrings + README) para el codigo"""
    return await provider.chat(
        f"Genera: 1) docstrings completos, 2) README con proposito, uso, ejemplo:\n\n{codigo}",
        multiplicador=2.0
    )

@server.tool()
async def resumen_codigo(ruta: str) -> str:
    """Resume que hace un archivo de codigo: proposito, mecanismo y caso de uso"""
    try:
        p = Path(ruta)
        if not p.exists(): return f"No encontrado: {ruta}"
        codigo = p.read_text(encoding="utf-8", errors="replace")[:6000]
        return await provider.chat(
            f"Resume en 3 secciones: 1) Que hace? 2) Como? 3) Para que?\nArchivo: {p.name}\n\n{codigo}",
            multiplicador=1.5
        )
    except Exception as e:
        return f"Error: {str(e)}"

@server.tool()
async def polydim_generar_spec(idea: str, nivel: str = "detallado") -> str:
    """Genera una especificacion formal POLYDIM para una idea o requerimiento"""
    return await provider.chat(
        f"Genera especificacion formal POLYDIM nivel {nivel} para: {idea}\n"
        f"Incluye: proposito, primitivas usadas (COMPOSE/MIX/FIXPOINT/PROJECT), invariantes, ejemplo Python.",
        multiplicador=2.5
    )

@server.tool()
async def polydim_generar_abstract(descripcion: str, palabras: int = 150) -> str:
    """Genera un abstract academico en ingles para una descripcion de POLYDIM"""
    return await provider.chat(
        f"Escribe un abstract academico de {palabras} palabras en ingles para un paper sobre:\n{descripcion}\n"
        f"Formato: Background, Problem, Solution, Results, Conclusion.",
        multiplicador=2.0
    )


# ─── Entrada principal ────────────────────────────────────────────────────────
if __name__ == "__main__":
    server.run()
