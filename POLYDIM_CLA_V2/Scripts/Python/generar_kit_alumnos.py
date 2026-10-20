"""
=======================================================
AGY - KIT PARA ALUMNOS: Generador Automático
=======================================================
Crea la carpeta 'para_alumnos' con:
  - Estructura completa de Antigravity
  - Scripts MCP copiados (sin API keys hardcodeadas)
  - config.json con placeholders de API keys
  - AGENTS.md y SOUL template
  - README de instalación paso a paso

Ejecutar desde cualquier lugar:
  python generar_kit_alumnos.py
"""

import os
import shutil
import re
import json
from pathlib import Path

# ─── Destino del kit ─────────────────────────────────────────────────────────
DESTINO = Path(r"G:\Mi unidad\POLYDIM_v1_1\para_alumnos")

# ─── Fuentes originales ───────────────────────────────────────────────────────
MCP_SRC   = Path(r"D:\_______ia_local\Antigravity")
CFG_SRC   = Path(r"C:\Users\eluithi\.gemini\config")
AGENTS_SRC = Path(r"G:\Mi unidad\POLYDIM_v1_1\.agents")

# ─── Patrones de API keys reales a sanitizar ─────────────────────────────────
# Si alguna key está hardcodeada (aunque no debería), se reemplaza
SANITIZE_PATTERNS = [
    # groq keys: gsk_XXXX
    (r'(groq_api_key["\s:=]+)["\']gsk_[A-Za-z0-9_]+["\']',
     r'\1"[AQUI_TU_API_KEY de https://console.groq.com/keys]"'),
    # openrouter keys: sk-or-XXXX
    (r'(openrouter_api_key["\s:=]+)["\']sk-or-[A-Za-z0-9_-]+["\']',
     r'\1"[AQUI_TU_API_KEY de https://openrouter.ai/workspaces/default/keys]"'),
    # gemini/google keys: AIzaSy
    (r'(gemini_api_key|google_api_key["\s:=]+)["\']AIza[A-Za-z0-9_-]+["\']',
     r'\1"[AQUI_TU_API_KEY de https://aistudio.google.com/app/apikey]"'),
    # huggingface tokens: hf_XXXX
    (r'(hf_token|HF_TOKEN["\s:=]+)["\']hf_[A-Za-z0-9]+["\']',
     r'\1"[AQUI_TU_API_KEY de https://huggingface.co/settings/tokens]"'),
    # cerebras keys
    (r'(cerebras_api_key["\s:=]+)["\']csk-[A-Za-z0-9]+["\']',
     r'\1"[AQUI_TU_API_KEY de https://cloud.cerebras.ai/api-keys]"'),
    # sambanova keys
    (r'(sambanova_api_key["\s:=]+)["\'][A-Za-z0-9_-]{20,}["\']',
     r'\1"[AQUI_TU_API_KEY de https://cloud.sambanova.ai/apis]"'),
]

def sanitize_content(content: str) -> str:
    """Reemplaza cualquier API key real con placeholder descriptivo."""
    for pattern, replacement in SANITIZE_PATTERNS:
        content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
    return content

def copiar_archivo(src: Path, dst: Path, sanitize=False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if sanitize and src.suffix in ['.py', '.json', '.md', '.txt']:
        content = src.read_text(encoding='utf-8', errors='replace')
        content = sanitize_content(content)
        dst.write_text(content, encoding='utf-8')
    else:
        shutil.copy2(src, dst)

def copiar_directorio(src: Path, dst: Path, sanitize=False, 
                       incluir_ext=None, excluir=None):
    excluir = excluir or ['__pycache__', '.git', 'node_modules', '*.pyc']
    for item in src.rglob('*'):
        if item.is_file():
            # Filtrar extensiones si se especifican
            if incluir_ext and item.suffix not in incluir_ext:
                continue
            # Excluir patrones
            skip = any(item.match(ex) for ex in excluir)
            if skip:
                continue
            rel = item.relative_to(src)
            copiar_archivo(item, dst / rel, sanitize=sanitize)

def crear_config_alumno():
    """Genera el config.json con placeholders en lugar de keys reales."""
    config = {
        "_instrucciones": "Completa cada campo con tu API key del proveedor indicado. Las keys gratuitas son suficientes.",
        "groq_api_key": "[AQUI_TU_API_KEY de https://console.groq.com/keys]",
        "openrouter_api_key": "[AQUI_TU_API_KEY de https://openrouter.ai/workspaces/default/keys]",
        "gemini_api_key": "[AQUI_TU_API_KEY de https://aistudio.google.com/app/apikey]",
        "hf_token": "[AQUI_TU_API_KEY de https://huggingface.co/settings/tokens]",
        "cerebras_api_key": "[AQUI_TU_API_KEY de https://cloud.cerebras.ai/api-keys]",
        "sambanova_api_key": "[AQUI_TU_API_KEY de https://cloud.sambanova.ai/apis]",
        "ollama_url": "http://localhost:11434",
        "default_model": "deepseek-r1:7b",
        "provider": "groq"
    }
    return config

def crear_readme():
    return """# 🧠 KIT ANTIGRAVITY PARA ALUMNOS
## Configuración del Agente Inteligente AGY

---

## ¿Qué es esto?
Este kit te da acceso a un agente de IA avanzado (AGY) que puede:
- Programar, analizar código y resolver problemas complejos
- Conectarse a múltiples modelos de IA gratuitos (Groq, Gemini, OpenRouter)
- Leer y escribir archivos en tu computadora
- Hacer investigación autónoma con múltiples agentes en paralelo

---

## Paso 1: Instalá las dependencias
Necesitás Python 3.10 o superior.

```bash
pip install mcp httpx fastmcp
```

---

## Paso 2: Conseguí tus API Keys GRATUITAS
Abrí `config.json` y completá cada campo con tu key personal:

| Proveedor | Link para registrarte | Plan Gratuito |
|-----------|----------------------|---------------|
| **Groq** ⚡ (más rápido) | https://console.groq.com/keys | Sí, generoso |
| **OpenRouter** | https://openrouter.ai/workspaces/default/keys | Sí, crédito inicial |
| **Gemini** | https://aistudio.google.com/app/apikey | Sí |
| **HuggingFace** | https://huggingface.co/settings/tokens | Sí |
| **Cerebras** | https://cloud.cerebras.ai/api-keys | Sí |
| **SambaNova** | https://cloud.sambanova.ai/apis | Sí |

> ⚠️ Con tener solo **Groq** ya tenés acceso a Llama 3.3 70B gratis.
> El resto son opcionales como backup.

---

## Paso 3: Configurá las rutas en `mcp_config.json`
Abrí el archivo y reemplazá las rutas con las de tu máquina:

```json
{
  "mcpServers": {
    "procesador-local": {
      "command": "C:/Python312/python.exe",       <- tu ruta de Python
      "args": ["-u", "C:/ruta/donde/guardaste/server_mcp_ollama_antigravity_7.py"]
    }
  }
}
```

---

## Paso 4: Copiá `mcp_config.json` a tu carpeta de Antigravity
La ubicación depende de tu sistema operativo:
- **Windows:** `C:\\Users\\TU_USUARIO\\.gemini\\config\\mcp_config.json`

---

## Paso 5: Copiá `AGENTS.md` y `SOUL_AGY_ORCHESTRATOR.md`
Estos dos archivos van en `.agents/` dentro de tu proyecto:
```
mi_proyecto/
└── .agents/
    ├── AGENTS.md
    └── SOUL_AGY_ORCHESTRATOR.md
```

---

## Archivos incluidos en este kit

| Archivo | Qué hace |
|---------|----------|
| `config.json` | Tus API keys (¡completar!) |
| `mcp_config.json` | Configuración de los servidores MCP |
| `server_mcp_ollama_antigravity_7.py` | Servidor local (procesador-local) |
| `server_mcp_openrouter_antigravity.py` | Servidor OpenRouter |
| `server_mcp_multicloud_antigravity.py` | Servidor multicloud (Groq/Gemini/HF) |
| `AGENTS.md` | Reglas de comportamiento del agente |
| `SOUL_AGY_ORCHESTRATOR.md` | Personalidad y protocolos del agente |

---

## ¿Problemas?
1. Verificá que Python esté en el PATH: `python --version`
2. Verificá que mcp esté instalado: `pip show mcp`
3. Revisá el archivo `error.log` que genera el servidor MCP

---
*Kit generado por AGY (El Genio) · Proyecto POLYDIM-CLA · 2026*
"""

def main():
    print("=" * 55)
    print(" GENERANDO KIT PARA ALUMNOS...")
    print("=" * 55)

    if DESTINO.exists():
        shutil.rmtree(DESTINO)
    DESTINO.mkdir(parents=True)

    # 1. Scripts MCP (los 3 servidores principales + sanitizar)
    print("[1/6] Copiando servidores MCP...")
    mcp_dst = DESTINO / "mcp_servers"
    scripts_a_copiar = [
        "server_mcp_ollama_antigravity_7.py",
        "server_mcp_openrouter_antigravity.py",
        "server_mcp_multicloud_antigravity.py",
    ]
    for script in scripts_a_copiar:
        src = MCP_SRC / script
        if src.exists():
            copiar_archivo(src, mcp_dst / script, sanitize=True)
            print(f"  -> {script} (sanitizado)")
        else:
            print(f"  [!] No encontrado: {script}")

    # 2. Config.json con placeholders
    print("[2/6] Generando config.json con placeholders...")
    config_data = crear_config_alumno()
    (DESTINO / "config.json").write_text(
        json.dumps(config_data, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )

    # 3. mcp_config.json con rutas relativas de ejemplo
    print("[3/6] Generando mcp_config.json con rutas de ejemplo...")
    mcp_config = {
        "mcpServers": {
            "procesador-local": {
                "command": "C:/Python312/python.exe",
                "_NOTA": "Cambia la ruta de python y del script a las tuyas",
                "args": ["-W", "ignore", "-u",
                         "./mcp_servers/server_mcp_ollama_antigravity_7.py"],
                "env": {
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONWARNINGS": "ignore"
                }
            },
            "procesador-openrouter": {
                "command": "C:/Python312/python.exe",
                "args": ["-W", "ignore", "-u",
                         "./mcp_servers/server_mcp_openrouter_antigravity.py"],
                "env": {
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONWARNINGS": "ignore"
                }
            },
            "procesador-multicloud": {
                "command": "C:/Python312/python.exe",
                "args": ["-W", "ignore", "-u",
                         "./mcp_servers/server_mcp_multicloud_antigravity.py"],
                "env": {
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONWARNINGS": "ignore"
                }
            }
        }
    }
    (DESTINO / "mcp_config.json").write_text(
        json.dumps(mcp_config, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )

    # 4. AGENTS.md y SOUL template
    print("[4/6] Copiando archivos de configuracion del agente...")
    agentes_dst = DESTINO / "agentes"
    agentes_dst.mkdir(exist_ok=True)
    
    soul_src = CFG_SRC / "SOUL_AGY_ORCHESTRATOR_TEMPLATE.md"
    if soul_src.exists():
        copiar_archivo(soul_src, agentes_dst / "SOUL_AGY_ORCHESTRATOR.md")
        print("  -> SOUL_AGY_ORCHESTRATOR.md")

    agents_src = CFG_SRC / "AGENTS.md"
    if agents_src.exists():
        copiar_archivo(agents_src, agentes_dst / "AGENTS.md")
        print("  -> AGENTS.md")

    # 5. README de instalación
    print("[5/6] Generando README de instalacion...")
    (DESTINO / "README.md").write_text(crear_readme(), encoding='utf-8')

    # 6. Script de instalación rápida para Windows
    print("[6/6] Generando instalador rapido...")
    installer = """@echo off
echo ================================================
echo  INSTALADOR AGY - KIT PARA ALUMNOS
echo ================================================

echo Instalando dependencias Python...
pip install mcp httpx fastmcp

echo.
echo SIGUIENTE PASO: Abre config.json y completa tus API keys
echo Despues copia mcp_config.json a:
echo   C:\\Users\\%USERNAME%\\.gemini\\config\\mcp_config.json
echo.
echo Listo! Abre Antigravity IDE y empieza a usar AGY.
pause
"""
    (DESTINO / "instalar.bat").write_text(installer, encoding='utf-8')

    print()
    print("=" * 55)
    print(f"[OK] Kit generado en: {DESTINO}")
    print()
    print("Contenido:")
    for item in sorted(DESTINO.rglob('*')):
        rel = item.relative_to(DESTINO)
        if item.is_file():
            size = item.stat().st_size
            print(f"  {rel}  ({size:,} bytes)")
    print("=" * 55)

if __name__ == "__main__":
    main()
