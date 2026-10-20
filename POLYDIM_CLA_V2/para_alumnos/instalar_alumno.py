"""
=======================================================
AGY - INSTALADOR PERSONALIZADO PARA ALUMNO
=======================================================
Llamado por el agente Antigravity durante el proceso
de instalación guiado por INSTALL.md.

Uso:
  python instalar_alumno.py
         --nombre "Juan Perez"
         --username "jperez"
         --workspace "C:\\MisProyectos\\AGY"
         --groq "gsk_tukey..."
         --openrouter "sk-or-tukey..."
         --gemini "AIzaSy..."
"""

import argparse
import json
import os
import shutil
import re
from pathlib import Path
from datetime import datetime

# ─── Directorio donde vive este script (el kit del alumno) ───────────────────
KIT_DIR = Path(__file__).parent

# ─── Archivos a copiar al workspace ─────────────────────────────────────────
MCP_SCRIPTS = [
    "mcp_servers/server_mcp_ollama_antigravity_7.py",
    "mcp_servers/server_mcp_openrouter_antigravity.py",
    "mcp_servers/server_mcp_multicloud_antigravity.py",
]

AGENT_FILES = [
    "agentes/SOUL_AGY_ORCHESTRATOR.md",
    "agentes/AGENTS.md",
]

def reemplazar_nombre(content: str, nombre: str, username: str, nombre_agente: str = "") -> str:
    """Reemplaza referencias al creador original por el nombre del alumno."""
    replacements = {
        "Ariel Luithardt": nombre,
        "Ariel":           nombre,
        "eluithi":         username,
        "Aladino":         nombre,
        "POPL 2027":       "mis proyectos del curso",
        "labor docente":   f"curso de {nombre}",
    }
    # Reemplazar el nombre del agente SOLO si se proporciona uno
    if nombre_agente:
        replacements["AGY (El Genio)"] = nombre_agente
        replacements["AGY"]            = nombre_agente
        replacements["El Genio"]       = nombre_agente

    for old, new in replacements.items():
        content = content.replace(old, new)
    return content

def generar_soul_personalizado(nombre: str, username: str, nombre_agente: str = "") -> str:
    """Lee la plantilla del SOUL y la personaliza para el alumno."""
    soul_src = KIT_DIR / "agentes" / "SOUL_AGY_ORCHESTRATOR.md"
    content = soul_src.read_text(encoding="utf-8")
    content = reemplazar_nombre(content, nombre, username, nombre_agente)
    agente = nombre_agente or "Tu Agente"
    content = content.replace(
        "## 8. Misión Terminal",
        f"## 8. Misión Terminal\n*Alumno: {nombre} | Agente: {agente} | Configurado: {datetime.now().strftime('%Y-%m-%d')}*"
    )
    # Personalizar el encabezado del SOUL
    content = content.replace(
        "# SOUL OF THE CRITICAL ORCHESTRATOR",
        f"# SOUL OF {agente.upper()} — ASISTENTE PERSONAL DE {nombre.upper()}"
    )
    return content

def generar_agents_personalizado(nombre: str, username: str, nombre_agente: str = "") -> str:
    """Personaliza el AGENTS.md para el alumno."""
    agents_src = KIT_DIR / "agentes" / "AGENTS.md"
    content = agents_src.read_text(encoding="utf-8")
    content = reemplazar_nombre(content, nombre, username, nombre_agente)
    content = content.replace(
        "9. **Inicialización de Proyectos Nuevos",
        "# 9. (Regla de docente - no aplica)"
    )
    return content

def generar_config(groq="", openrouter="", gemini="", hf="", cerebras="", sambanova="") -> dict:
    """Genera config.json con las keys del alumno (o placeholders si no las tiene)."""
    
    # Auto-detección desde variables de entorno
    groq = groq or os.environ.get("GROQ_API_KEY") or os.environ.get("GROQ_KEY") or ""
    openrouter = openrouter or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_KEY") or ""
    gemini = gemini or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    hf = hf or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or ""
    cerebras = cerebras or os.environ.get("CEREBRAS_API_KEY") or ""
    sambanova = sambanova or os.environ.get("SAMBANOVA_API_KEY") or ""

    def key_or_placeholder(key, url):
        return key if key else f"[PENDIENTE - conseguir en {url}]"

    return {
        "_generado": datetime.now().isoformat(),
        "groq_api_key":       key_or_placeholder(groq,       "https://console.groq.com/keys"),
        "openrouter_api_key": key_or_placeholder(openrouter,  "https://openrouter.ai/workspaces/default/keys"),
        "gemini_api_key":     key_or_placeholder(gemini,     "https://aistudio.google.com/app/apikey"),
        "hf_token":           key_or_placeholder(hf,         "https://huggingface.co/settings/tokens"),
        "cerebras_api_key":   key_or_placeholder(cerebras,   "https://cloud.cerebras.ai/api-keys"),
        "sambanova_api_key":  key_or_placeholder(sambanova,  "https://cloud.sambanova.ai/apis"),
        "ollama_url":         "http://localhost:11434",
        "default_model":      "deepseek-r1:7b",
        "provider":           "groq" if groq else "ollama"
    }

def generar_mcp_config(workspace: Path, python_path: str = None) -> dict:
    """Genera mcp_config.json apuntando al workspace del alumno."""
    if not python_path:
        # Detectar Python automáticamente
        import sys
        python_path = sys.executable.replace("\\", "/")

    mcp_dir = str(workspace / "mcp_servers").replace("\\", "/")

    return {
        "mcpServers": {
            "procesador-local": {
                "command": python_path,
                "args": ["-W", "ignore", "-u",
                         f"{mcp_dir}/server_mcp_ollama_antigravity_7.py"],
                "env": {
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONWARNINGS": "ignore"
                }
            },
            "procesador-openrouter": {
                "command": python_path,
                "args": ["-W", "ignore", "-u",
                         f"{mcp_dir}/server_mcp_openrouter_antigravity.py"],
                "env": {
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONWARNINGS": "ignore"
                }
            },
            "procesador-multicloud": {
                "command": python_path,
                "args": ["-W", "ignore", "-u",
                         f"{mcp_dir}/server_mcp_multicloud_antigravity.py"],
                "env": {
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONWARNINGS": "ignore"
                }
            }
        }
    }

def instalar(nombre, username, workspace_str,
             groq="", openrouter="", gemini="", hf="", cerebras="", sambanova="",
             ollama_model="", python_path="", nombre_agente=""):

    workspace = Path(workspace_str)
    print(f"\n{'='*55}")
    print(f" INSTALANDO ENTORNO AGY PARA: {nombre}")
    print(f" Workspace: {workspace}")
    print(f"{'='*55}\n")

    # 1. Crear estructura de carpetas
    print("[1/6] Creando estructura de carpetas...")
    for carpeta in ["mcp_servers", ".agents", "mis_proyectos", "logs"]:
        (workspace / carpeta).mkdir(parents=True, exist_ok=True)
    print("  OK")

    # 2. Copiar servidores MCP
    print("[2/6] Copiando servidores MCP...")
    for rel_path in MCP_SCRIPTS:
        src = KIT_DIR / rel_path
        if src.exists():
            dst = workspace / rel_path
            shutil.copy2(src, dst)
            print(f"  -> {src.name}")
        else:
            print(f"  [!] No encontrado: {rel_path}")

    # 3. Copiar config.json con las keys del alumno
    print("[3/6] Generando config.json personalizado...")
    config = generar_config(groq, openrouter, gemini, hf, cerebras, sambanova)
    (workspace / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    keys_configuradas = sum(1 for v in config.values()
                            if v and not str(v).startswith("["))
    print(f"  OK ({keys_configuradas} keys configuradas)")

    # 4. Generar mcp_config.json con rutas absolutas reales
    print("[4/6] Generando mcp_config.json con rutas absolutas...")
    mcp_config = generar_mcp_config(workspace, python_path or None)
    (workspace / ".agents" / "mcp_config.json").write_text(
        json.dumps(mcp_config, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    # Si el alumno eligió un modelo Ollama, registrarlo en config.json
    if ollama_model:
        config["ollama_model"] = ollama_model
        config["default_model"] = ollama_model
        (workspace / "config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
    print("  OK")

    # 5. Generar SOUL y AGENTS.md personalizados
    print("[5/6] Personalizando archivos de agente con tu nombre...")
    soul_content = generar_soul_personalizado(nombre, username, nombre_agente)
    (workspace / ".agents" / "SOUL_AGY_ORCHESTRATOR.md").write_text(
        soul_content, encoding="utf-8"
    )
    agents_content = generar_agents_personalizado(nombre, username, nombre_agente)
    (workspace / ".agents" / "AGENTS.md").write_text(
        agents_content, encoding="utf-8"
    )
    agente = nombre_agente if nombre_agente else "Mi Asistente"
    print(f"  OK ('{agente}' aplicado como nombre del agente, '{nombre}' como alumno)")

    # 6. Generar resumen de instalación
    print("[6/6] Generando resumen...")
    resumen = f"""# Instalacion Completada - {nombre}
Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Tu Configuracion
- **Nombre:** {nombre}
- **Tu Agente:** {nombre_agente if nombre_agente else 'Sin nombre aun'}
- **Username:** {username}
- **Workspace:** {workspace}

## Estado de API Keys
"""
    for key, val in config.items():
        if key.startswith("_"):
            continue
        estado = "OK" if not str(val).startswith("[") else "PENDIENTE"
        resumen += f"- {key}: {estado}\n"

    resumen += f"""
## Modelos Configurados
- Ollama local: {ollama_model if ollama_model else 'No instalado'}
- Proveedor principal: {'groq' if groq else 'openrouter' if openrouter else 'ollama'}

## Siguiente Paso
Copia el archivo `.agents/mcp_config.json` a:
  `C:\\Users\\{username}\\.gemini\\config\\mcp_config.json`

Despues reinicia Antigravity y escribi: "hola, quien soy?"
"""
    (workspace / "INSTALACION_COMPLETADA.md").write_text(
        resumen, encoding="utf-8"
    )

    print(f"\n{'='*55}")
    print(f" LISTO! Entorno de {nombre} configurado.")
    print(f" Workspace: {workspace}")
    print(f"\n SIGUIENTE PASO MANUAL:")
    print(f"   Copia .agents/mcp_config.json a:")
    print(f"   C:\\Users\\{username}\\.gemini\\config\\mcp_config.json")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Instalador AGY personalizado para alumno"
    )
    parser.add_argument("--nombre",     required=True,  help="Nombre completo del alumno")
    parser.add_argument("--username",   required=True,  help="Username de Windows del alumno")
    parser.add_argument("--workspace",  required=True,  help="Ruta donde instalar el workspace")
    parser.add_argument("--groq",       default="",     help="API key de Groq")
    parser.add_argument("--openrouter", default="",     help="API key de OpenRouter")
    parser.add_argument("--gemini",     default="",     help="API key de Gemini")
    parser.add_argument("--hf",         default="",     help="Token de HuggingFace")
    parser.add_argument("--cerebras",   default="",     help="API key de Cerebras")
    parser.add_argument("--sambanova",  default="",     help="API key de SambaNova")

    parser.add_argument("--ollama_model",  default="",  help="Modelo Ollama elegido")
    parser.add_argument("--python_path",   default="",  help="Ruta al ejecutable Python")
    parser.add_argument("--nombre_agente", default="",  help="Nombre del agente personal del alumno")

    args = parser.parse_args()

    instalar(
        nombre=args.nombre,
        username=args.username,
        workspace_str=args.workspace,
        groq=args.groq,
        openrouter=args.openrouter,
        gemini=args.gemini,
        hf=args.hf,
        cerebras=args.cerebras,
        sambanova=args.sambanova,
        ollama_model=args.ollama_model,
        python_path=args.python_path,
        nombre_agente=args.nombre_agente,
    )
