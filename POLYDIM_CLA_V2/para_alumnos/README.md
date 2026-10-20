# 🧠 KIT ANTIGRAVITY PARA ALUMNOS
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
- **Windows:** `C:\Users\TU_USUARIO\.gemini\config\mcp_config.json`

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
