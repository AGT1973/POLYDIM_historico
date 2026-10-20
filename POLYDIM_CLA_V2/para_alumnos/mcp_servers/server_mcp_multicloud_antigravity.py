#!/usr/bin/env python3
"""
Servidor MCP Antigravity - MultiCloud (Groq + Gemini + HuggingFace)
===================================================================
Proveedores gratuitos de alta velocidad con fallback a Ollama local.
Cadena de fallback: Groq (⚡) → Gemini → HuggingFace → Ollama

pip install mcp httpx
"""

import json, os, datetime, asyncio
import httpx
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# ─── Configuración ────────────────────────────────────────────────────────────
server = FastMCP("Antigravity-MultiCloud-Server")

CONFIG_PATH = Path(r"D:\_______ia_local\Antigravity\config.json")
LOG_PATH = r"D:\_______ia_local\Antigravity\error.log"

GROQ_MODELS = {
    "llama":    "llama-3.3-70b-versatile",
    "llama8b":  "llama-3.1-8b-instant",
    "gemma":    "gemma2-9b-it",
    "mixtral":  "mixtral-8x7b-32768",
}

GEMINI_MODELS = {
    "flash": "gemini-2.0-flash",
    "pro":   "gemini-1.5-pro",
}

HF_MODELS = {
    "llama":   "meta-llama/Llama-3.3-70B-Instruct",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "qwen":    "Qwen/Qwen2.5-72B-Instruct",
    "phi":     "microsoft/Phi-3-medium-4k-instruct",
}

CEREBRAS_MODELS = {
    "llama":    "llama-3.3-70b",
    "llama8b":  "llama3.1-8b",
}

SAMBANOVA_MODELS = {
    "llama":    "Meta-Llama-3.3-70B-Instruct",
    "deepseek": "DeepSeek-R1",
    "qwq":      "QwQ-32B",
}

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "deepseek-r1:14b"


# ─── Logging silencioso (nunca stdout/stderr) ─────────────────────────────────
def _log(msg: str):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now().isoformat()} | MULTICLOUD | {msg}\n")
    except Exception:
        pass


def _load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ─── Proveedor MultiCloud con retry y fallback ───────────────────────────────
class MultiCloudProvider:
    def __init__(self):
        self._client = None
        self._config = None
        self._max_retries = 3
        self._base_delay = 2.0

    @property
    def client(self) -> httpx.AsyncClient:
        """Lazy init — no bloquea el arranque del MCP."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient()
            _log("httpx.AsyncClient inicializado (lazy)")
        return self._client

    @property
    def config(self) -> dict:
        if self._config is None:
            self._config = _load_config()
        return self._config

    def reload_config(self):
        """Fuerza recarga del config.json (por si se agrega una key nueva)."""
        self._config = _load_config()

    async def _retry_post(self, url: str, retries: int = None, **kwargs) -> httpx.Response:
        """POST con retry y backoff exponencial."""
        retries = retries or self._max_retries
        last_error = None
        for attempt in range(retries):
            try:
                return await self.client.post(url, **kwargs)
            except (httpx.ConnectError, httpx.TimeoutException,
                    httpx.RemoteProtocolError, ConnectionError) as e:
                last_error = e
                delay = self._base_delay * (2 ** attempt)
                _log(f"Retry {attempt+1}/{retries} {url}: {type(e).__name__}")
                if self._client and self._client.is_closed:
                    self._client = None
                await asyncio.sleep(delay)
            except Exception as e:
                last_error = e
                _log(f"Error no-retriable {url}: {type(e).__name__}: {e}")
                break
        raise last_error if last_error else Exception(f"Falló tras {retries} intentos")

    @staticmethod
    def _msgs(prompt: str, system: str = "") -> list:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    # ── Groq ──────────────────────────────────────────────────────────────────
    async def groq(self, prompt: str, model: str = None,
                   system: str = "", timeout: float = 120.0) -> str:
        key = self.config.get("groq_api_key", "")
        if not key:
            raise Exception("Sin groq_api_key en config.json")
        model = model or GROQ_MODELS["llama"]
        resp = await self._retry_post(
            "https://api.groq.com/openai/v1/chat/completions",
            json={"model": model, "messages": self._msgs(prompt, system)},
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
            timeout=timeout,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        raise Exception(f"Groq HTTP {resp.status_code}: {resp.text[:300]}")

    # ── Gemini ────────────────────────────────────────────────────────────────
    async def gemini(self, prompt: str, model: str = None,
                     system: str = "", timeout: float = 120.0) -> str:
        key = self.config.get("gemini_api_key", "")
        if not key:
            raise Exception("Sin gemini_api_key en config.json")
        model = model or GEMINI_MODELS["flash"]
        # Endpoint OpenAI-compatible de Google AI Studio
        resp = await self._retry_post(
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            json={"model": model, "messages": self._msgs(prompt, system)},
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
            timeout=timeout,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        raise Exception(f"Gemini HTTP {resp.status_code}: {resp.text[:300]}")

    # ── HuggingFace ───────────────────────────────────────────────────────────
    async def huggingface(self, prompt: str, model: str = None,
                          system: str = "", timeout: float = 180.0) -> str:
        key = self.config.get("hf_token", "")
        if not key:
            raise Exception("Sin hf_token en config.json")
        model = model or HF_MODELS["llama"]
        resp = await self._retry_post(
            "https://api-inference.huggingface.co/v1/chat/completions",
            json={"model": model,
                  "messages": self._msgs(prompt, system),
                  "max_tokens": 2048},
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
            timeout=timeout,
        )
        if resp.status_code == 200:
            data = resp.json()
            if "choices" in data:
                return data["choices"][0]["message"]["content"]
            return data.get("generated_text", json.dumps(data, indent=2))
        raise Exception(f"HuggingFace HTTP {resp.status_code}: {resp.text[:300]}")

    # ── Cerebras (⚡ Ultra rápido, gratis) ─────────────────────────────────────
    async def cerebras(self, prompt: str, model: str = None,
                       system: str = "", timeout: float = 120.0) -> str:
        key = self.config.get("cerebras_api_key", "")
        if not key:
            raise Exception("Sin cerebras_api_key en config.json")
        model = model or CEREBRAS_MODELS["llama"]
        resp = await self._retry_post(
            "https://api.cerebras.ai/v1/chat/completions",
            json={"model": model, "messages": self._msgs(prompt, system)},
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
            timeout=timeout,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        raise Exception(f"Cerebras HTTP {resp.status_code}: {resp.text[:300]}")

    # ── SambaNova (gratis, DeepSeek R1 y QwQ) ─────────────────────────────────
    async def sambanova(self, prompt: str, model: str = None,
                        system: str = "", timeout: float = 180.0) -> str:
        key = self.config.get("sambanova_api_key", "")
        if not key:
            raise Exception("Sin sambanova_api_key en config.json")
        model = model or SAMBANOVA_MODELS["llama"]
        resp = await self._retry_post(
            "https://api.sambanova.ai/v1/chat/completions",
            json={"model": model, "messages": self._msgs(prompt, system)},
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
            timeout=timeout,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        raise Exception(f"SambaNova HTTP {resp.status_code}: {resp.text[:300]}")

    # ── Ollama (fallback final) ───────────────────────────────────────────────
    async def ollama(self, prompt: str, system: str = "",
                     timeout: float = 300.0) -> str:
        resp = await self._retry_post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": OLLAMA_MODEL,
                  "messages": self._msgs(prompt, system),
                  "stream": False},
            timeout=timeout,
        )
        if resp.status_code == 200:
            return resp.json().get("message", {}).get("content", "Sin respuesta")
        raise Exception(f"Ollama HTTP {resp.status_code}")

    # ── Smart routing con fallback automático ─────────────────────────────────
    async def smart_chat(self, prompt: str, system: str = "",
                         preferred: str = "groq") -> str:
        """Intenta proveedores en cascada hasta que uno responda."""
        all_providers = [
            ("Groq", self.groq), ("Cerebras", self.cerebras),
            ("Gemini", self.gemini), ("SambaNova", self.sambanova),
            ("HuggingFace", self.huggingface), ("Ollama", self.ollama),
        ]
        # Mueve el preferido al inicio
        def _reorder(pref_name):
            pref = [(n, f) for n, f in all_providers if n.lower() == pref_name]
            rest = [(n, f) for n, f in all_providers if n.lower() != pref_name]
            return pref + rest
        chains = {
            "groq":     _reorder("groq"),
            "cerebras": _reorder("cerebras"),
            "gemini":   _reorder("gemini"),
            "sambanova": _reorder("sambanova"),
            "hf":       _reorder("huggingface"),
        }
        chain = chains.get(preferred, chains["groq"])
        errors = []

        for name, fn in chain:
            try:
                result = await fn(prompt, system=system)
                _log(f"smart_chat OK via {name}")
                return f"[{name}] {result}"
            except Exception as e:
                errors.append(f"{name}: {e}")
                _log(f"smart_chat: {name} falló — {e}")

        return "❌ Todos los proveedores fallaron:\n" + "\n".join(errors)


# Instancia global
provider = MultiCloudProvider()


# ══════════════════════════════════════════════════════════════════════════════
# MCP TOOLS — Herramientas expuestas a Antigravity
# ══════════════════════════════════════════════════════════════════════════════

# ── Groq (⚡ Ultra rápido, gratis) ───────────────────────────────────────────

@server.tool()
async def ask_groq(mensaje: str, sistema: str = "Eres un asistente experto. Respondes de forma directa, concisa y precisa.") -> str:
    """Consulta ultra-rápida via Groq LPU con Llama 3.3 70B (gratis, ~500 tok/s)."""
    return await provider.groq(mensaje, model=GROQ_MODELS["llama"], system=sistema)


@server.tool()
async def ask_groq_fast(mensaje: str, sistema: str = "Eres un asistente ultra-rápido. Respondes en la menor cantidad de palabras posible sin perder precisión.") -> str:
    """Consulta instantánea via Groq con Llama 3.1 8B (el más rápido, ideal para tareas simples)."""
    return await provider.groq(mensaje, model=GROQ_MODELS["llama8b"], system=sistema)


@server.tool()
async def ask_groq_mixtral(mensaje: str, sistema: str = "Eres Mixtral, un modelo MoE experto en código y razonamiento multi-idioma. Respondes con precisión técnica.") -> str:
    """Consulta a Mixtral 8x7B via Groq (bueno en código y multilingüe)."""
    return await provider.groq(mensaje, model=GROQ_MODELS["mixtral"], system=sistema)


# ── Google Gemini (gratis, multimodal) ────────────────────────────────────────

@server.tool()
async def ask_gemini(mensaje: str, sistema: str = "Eres Gemini, el modelo de Google. Respondes con precisión, usando datos actualizados y razonamiento claro.") -> str:
    """Consulta a Gemini 2.0 Flash via Google AI Studio (gratis, rápido, multimodal)."""
    return await provider.gemini(mensaje, model=GEMINI_MODELS["flash"], system=sistema)


@server.tool()
async def ask_gemini_pro(mensaje: str, sistema: str = "Eres Gemini Pro, el modelo avanzado de Google. Priorizas profundidad de análisis y razonamiento complejo.") -> str:
    """Consulta a Gemini 1.5 Pro via Google AI Studio (gratis, más profundo)."""
    return await provider.gemini(mensaje, model=GEMINI_MODELS["pro"], system=sistema)


# ── HuggingFace (gratis, modelos open-source) ────────────────────────────────

@server.tool()
async def ask_hf(mensaje: str, modelo: str = "", sistema: str = "Eres un asistente basado en modelos open-source de HuggingFace. Respondes con claridad.") -> str:
    """Consulta via HuggingFace Inference API. Modelos: llama, mixtral, qwen, phi. Por defecto usa Llama 3.3 70B."""
    model = HF_MODELS.get(modelo, modelo if modelo else None)
    return await provider.huggingface(mensaje, model=model, system=sistema)


# ── Cerebras (⚡ Ultra rápido, gratis) ────────────────────────────────────────

@server.tool()
async def ask_cerebras(mensaje: str, sistema: str = "Eres un asistente experto corriendo en hardware Cerebras Wafer-Scale. Respondes con velocidad y precisión.") -> str:
    """Consulta ultra-rápida via Cerebras con Llama 3.3 70B (gratis, velocidad extrema)."""
    return await provider.cerebras(mensaje, model=CEREBRAS_MODELS["llama"], system=sistema)


@server.tool()
async def ask_cerebras_fast(mensaje: str, sistema: str = "Eres un asistente instantáneo. Respondes en la menor cantidad de palabras posible.") -> str:
    """Consulta instantánea via Cerebras con Llama 3.1 8B (el más rápido para tareas simples)."""
    return await provider.cerebras(mensaje, model=CEREBRAS_MODELS["llama8b"], system=sistema)


# ── SambaNova (gratis, DeepSeek y QwQ) ────────────────────────────────────────

@server.tool()
async def ask_sambanova(mensaje: str, sistema: str = "Eres un asistente experto corriendo en hardware SambaNova RDU. Respondes con precisión y profundidad.") -> str:
    """Consulta a Llama 3.3 70B via SambaNova (gratis, rápido)."""
    return await provider.sambanova(mensaje, model=SAMBANOVA_MODELS["llama"], system=sistema)


@server.tool()
async def ask_sambanova_deepseek(mensaje: str, sistema: str = "Eres DeepSeek R1, un razonador profundo. Muestra tu cadena de pensamiento paso a paso antes de la respuesta final.") -> str:
    """Consulta a DeepSeek R1 via SambaNova (gratis, razonamiento profundo)."""
    return await provider.sambanova(mensaje, model=SAMBANOVA_MODELS["deepseek"], system=sistema)


@server.tool()
async def ask_sambanova_qwq(mensaje: str, sistema: str = "Eres QwQ, un modelo especializado en razonamiento lógico y matemático. Piensas paso a paso con rigor.") -> str:
    """Consulta a QwQ-32B via SambaNova (gratis, razonamiento lógico/matemático)."""
    return await provider.sambanova(mensaje, model=SAMBANOVA_MODELS["qwq"], system=sistema)


# ── Smart routing (fallback automático) ───────────────────────────────────────

@server.tool()
async def multicloud_chat(mensaje: str, preferido: str = "groq", sistema: str = "") -> str:
    """Chat inteligente con fallback: Groq → Cerebras → Gemini → SambaNova → HF → Ollama. Opciones: groq, cerebras, gemini, sambanova, hf."""
    return await provider.smart_chat(mensaje, system=sistema, preferred=preferido)


# ── Utilidades ────────────────────────────────────────────────────────────────

@server.tool()
async def multicloud_modelos() -> str:
    """Lista todos los modelos disponibles en cada proveedor del MultiCloud."""
    cfg = provider.config
    status = {
        "Groq": "✅ Configurado" if cfg.get("groq_api_key") else "❌ Sin API key",
        "Gemini": "✅ Configurado" if cfg.get("gemini_api_key") else "❌ Sin API key",
        "HuggingFace": "✅ Configurado" if cfg.get("hf_token") else "❌ Sin API key",
        "Cerebras": "✅ Configurado" if cfg.get("cerebras_api_key") else "❌ Sin API key",
        "SambaNova": "✅ Configurado" if cfg.get("sambanova_api_key") else "❌ Sin API key",
        "Ollama": "✅ Fallback local",
    }
    lines = ["═══ MultiCloud - Modelos Disponibles ═══\n"]
    lines.append("[GROQ - ⚡ Ultra rápido, gratis]")
    lines.append(f"  Estado: {status['Groq']}")
    for alias, model in GROQ_MODELS.items():
        lines.append(f"  - {alias}: {model}")
    lines.append(f"\n[CEREBRAS - ⚡ Ultra rápido, gratis]")
    lines.append(f"  Estado: {status['Cerebras']}")
    for alias, model in CEREBRAS_MODELS.items():
        lines.append(f"  - {alias}: {model}")
    lines.append(f"\n[GEMINI - Google AI Studio, gratis]")
    lines.append(f"  Estado: {status['Gemini']}")
    for alias, model in GEMINI_MODELS.items():
        lines.append(f"  - {alias}: {model}")
    lines.append(f"\n[SAMBANOVA - Gratis, RDU]")
    lines.append(f"  Estado: {status['SambaNova']}")
    for alias, model in SAMBANOVA_MODELS.items():
        lines.append(f"  - {alias}: {model}")
    lines.append(f"\n[HUGGINGFACE - Open-source, gratis]")
    lines.append(f"  Estado: {status['HuggingFace']}")
    for alias, model in HF_MODELS.items():
        lines.append(f"  - {alias}: {model}")
    lines.append(f"\n[OLLAMA - Local, offline]")
    lines.append(f"  Estado: {status['Ollama']}")
    lines.append(f"  - fallback: {OLLAMA_MODEL}")
    return "\n".join(lines)


@server.tool()
async def multicloud_health() -> str:
    """Test de conectividad rápido a cada proveedor (no consume tokens)."""
    results = []
    cfg = provider.config

    # Groq
    try:
        if cfg.get("groq_api_key"):
            async with httpx.AsyncClient(timeout=10.0) as c:
                r = await c.get("https://api.groq.com/openai/v1/models",
                                headers={"Authorization": f"Bearer {cfg['groq_api_key']}"})
                results.append(f"Groq: {'✅ OK' if r.status_code == 200 else f'❌ HTTP {r.status_code}'}")
        else:
            results.append("Groq: ⚠️ Sin API key")
    except Exception as e:
        results.append(f"Groq: ❌ {type(e).__name__}")

    # Gemini
    try:
        if cfg.get("gemini_api_key"):
            async with httpx.AsyncClient(timeout=10.0) as c:
                r = await c.get("https://generativelanguage.googleapis.com/v1beta/models",
                                headers={"Authorization": f"Bearer {cfg['gemini_api_key']}"})
                results.append(f"Gemini: {'✅ OK' if r.status_code == 200 else f'❌ HTTP {r.status_code}'}")
        else:
            results.append("Gemini: ⚠️ Sin API key")
    except Exception as e:
        results.append(f"Gemini: ❌ {type(e).__name__}")

    # HuggingFace
    try:
        if cfg.get("hf_token"):
            async with httpx.AsyncClient(timeout=10.0) as c:
                r = await c.get("https://huggingface.co/api/whoami-v2",
                                headers={"Authorization": f"Bearer {cfg['hf_token']}"})
                results.append(f"HuggingFace: {'✅ OK' if r.status_code == 200 else f'❌ HTTP {r.status_code}'}")
        else:
            results.append("HuggingFace: ⚠️ Sin API key")
    except Exception as e:
        results.append(f"HuggingFace: ❌ {type(e).__name__}")

    # Cerebras
    try:
        if cfg.get("cerebras_api_key"):
            async with httpx.AsyncClient(timeout=10.0) as c:
                r = await c.get("https://api.cerebras.ai/v1/models",
                                headers={"Authorization": f"Bearer {cfg['cerebras_api_key']}"})
                results.append(f"Cerebras: {'✅ OK' if r.status_code == 200 else f'❌ HTTP {r.status_code}'}")
        else:
            results.append("Cerebras: ⚠️ Sin API key")
    except Exception as e:
        results.append(f"Cerebras: ❌ {type(e).__name__}")

    # SambaNova
    try:
        if cfg.get("sambanova_api_key"):
            async with httpx.AsyncClient(timeout=10.0) as c:
                r = await c.get("https://api.sambanova.ai/v1/models",
                                headers={"Authorization": f"Bearer {cfg['sambanova_api_key']}"})
                results.append(f"SambaNova: {'✅ OK' if r.status_code == 200 else f'❌ HTTP {r.status_code}'}")
        else:
            results.append("SambaNova: ⚠️ Sin API key")
    except Exception as e:
        results.append(f"SambaNova: ❌ {type(e).__name__}")

    # Ollama
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{OLLAMA_URL}/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
            results.append(f"Ollama: ✅ OK ({len(models)} modelos)")
    except Exception as e:
        results.append(f"Ollama: ❌ {type(e).__name__}")

    return "\n".join(results)


@server.tool()
async def multicloud_reload_config() -> str:
    """Recarga config.json sin reiniciar el servidor (útil si se agrega una API key nueva)."""
    provider.reload_config()
    cfg = provider.config
    keys = [k for k in ["groq_api_key", "gemini_api_key", "hf_token", "openrouter_api_key"]
            if cfg.get(k)]
    return f"Config recargado. Keys activas: {', '.join(keys)}"


@server.tool()
async def mostrar_metodos() -> str:
    """Lista las herramientas directas de este servidor MCP."""
    tools_dict = getattr(server, "_tools", {})
    mcp = sorted(tools_dict.keys()) if tools_dict else []
    return f"Herramientas MultiCloud ({len(mcp)}):\n" + "\n".join(f"  - {t}" for t in mcp)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    _log(f"Servidor MultiCloud iniciado")
    server.run()
