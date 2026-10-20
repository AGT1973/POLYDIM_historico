import os
import json
import time
import requests
import re
from pathlib import Path
import concurrent.futures

# --- CONFIGURACION AVANZADA ---
CONFIG_PATH = r"D:\_______ia_local\Antigravity\config.json"
BASE_DIR = r"G:\Mi unidad\POLYDIM_v1_1\Alumnos"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MODEL_QWEN = "qwen/qwen-2.5-72b-instruct"
MODEL_KIMI = "moonshotai/moonshot-v1-32k"
MODEL_DEEPSEEK_CODER = "deepseek/deepseek-coder"

INICIO_TAREA = 61
FIN_TAREA = 200
MAX_WORKERS = 4
TIEMPO_ESPERA_ENTRE_LOTES = 120

def get_api_key():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f).get("openrouter_api_key", "").strip()
    except Exception:
        return ""

API_KEY = get_api_key()

def call_openrouter_with_retry(prompt, system_prompt, model, temperature=0.7, max_retries=3):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://polydim.org",
        "X-Title": "POLYDIM Auto-Swarm Orchestrator"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature
    }
    
    for intento in range(max_retries):
        try:
            response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=180)
            if response.status_code == 429:
                print(f"  [WARN] Rate limit (429) en {model}. Reintentando en {30 * (intento + 1)}s...")
                time.sleep(30 * (intento + 1))
                continue
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"  [WARN] Intento {intento+1} falló para {model}: {e}")
            time.sleep(15)
    return None

def extract_code(markdown_text, lenguaje="python"):
    pattern = r'```(?:' + lenguaje + r'|' + lenguaje[:2] + r')\n(.*?)```'
    matches = re.findall(pattern, markdown_text, re.DOTALL | re.IGNORECASE)
    return matches[-1].strip() if matches else ""

# --- PROMPTS DE INGENIERÍA AVANZADA ---
QWEN_SYSTEM = """Actúas bajo el marco del proyecto científico POLYDIM. Eres un matemático y arquitecto de sistemas especialista en Topología Diferencial, Álgebra de Clifford, Topos Theory y optimización de Hardware/Software. Formula demostraciones empíricas en código (scripts ejecutables en Python o Rust) que comprueben restricciones topológicas estrictas (invariancia isométrica, estabilización de gradientes). Hazlo en completo silencio y rigor matemático."""

def get_qwen_prompt(tarea_id, perfil, lenguaje):
    req_libs = "JAX (vmap, jit, grad), PyTorch (einsum) y SciPy" if lenguaje == "python" else "Rust, iteradores paralelos con Rayon, ndarray o control unsafe manual"
    return f"""Eres el Alumno {tarea_id} del Enjambre POLYDIM. Perfil: {perfil}.
1. Diseña un concepto teórico único o benchmark matemático que conecte topología con redes neuronales o motores de inferencia.
2. Explica el razonamiento matemático (Markdown).
3. Escribe el script {lenguaje.capitalize()} autocontenido usando {req_libs} para validar empíricamente la teoría.
REGLA MATEMÁTICA ABSOLUTA: Toda operación de composición de tensores de bajo rango (tipo LoRA) DEBE incluir un paso de truncamiento SVD periódico (vía Teorema de Eckart-Young) para evitar la explosión del rango, de lo contrario FIXPOINT pasaría a O(N^2). En Rust usa descomposiciones QR seguidas de SVD sobre núcleos pequeños.
Encierra el código en ```{lenguaje} ... ```."""

KIMI_SYSTEM = """Eres el Revisor Científico Senior de POLYDIM. Exiges perfección topológica y rendimiento de hardware al nivel de un ingeniero principal en NVIDIA o Anthropic."""

def get_kimi_prompt(codigo_base, lenguaje):
    extra_instructions = (
        "Vectoriza absolutamente todo con `jax.vmap` o `torch.einsum`. Elimina bucles for. OBLIGATORIO: Asegura que haya Truncamiento SVD/Eckart-Young explícito si se componen matrices de bajo rango."
        if lenguaje == "python" else
        "Garantiza uso de memoria seguro. OBLIGATORIO: Implementa explícitamente `compose_and_truncate` usando factorización QR en `u_stack` y `v_stack` y un SVD chico `2r x 2r`, tal como dictamina la arquitectura de COMPOSE."
    )
    return f"""Borrador de código del alumno para validación matemática:

```{lenguaje}
{codigo_base}
```

Tu tarea:
1. Sofisticar matemáticamente el código.
2. Inyectar `asserts` para invariantes topológicas (tolerancias de $1e-5$, conservación de energía).
3. {extra_instructions}
4. Debe compilar y emitir un reporte final JSON con un estatus "SUCCESS".
Devuelve ÚNICAMENTE el bloque de código {lenguaje.capitalize()} en ```{lenguaje} ... ```."""

def procesar_tarea(args):
    i, perfil, lenguaje = args
    ext = "rs" if lenguaje == "rust" else "py"
    task_dir = os.path.join(BASE_DIR, f"tarea_{i}")
    os.makedirs(task_dir, exist_ok=True)
    
    archivo_codigo = os.path.join(task_dir, f"tarea_{i}_avanzada.{ext}")
    archivo_md = os.path.join(task_dir, "razonamiento.md")
    
    if os.path.exists(archivo_codigo):
        print(f"[{i}] Tarea ya existe. Saltando.")
        return
        
    print(f"[{i}] Iniciando generación ({lenguaje.upper()}) con Qwen...")
    qwen_resp = call_openrouter_with_retry(get_qwen_prompt(i, perfil, lenguaje), QWEN_SYSTEM, MODEL_QWEN)
    
    if not qwen_resp:
        print(f"[{i}] Falla fatal en Qwen.")
        return
        
    with open(archivo_md, "w", encoding="utf-8") as f:
        f.write(qwen_resp)
        
    codigo_base = extract_code(qwen_resp, lenguaje)
    if not codigo_base:
        print(f"[{i}] Qwen no devolvió código.")
        with open(os.path.join(task_dir, f"tarea_{i}_fallback.{ext}"), "w", encoding="utf-8") as f:
            f.write("// Fallo en formato.\n" if lenguaje == "rust" else "# Fallo en formato.\n")
        return

    print(f"[{i}] Código extraído. Mandando a Kimi para auditoría extrema (Validando SVD Truncado)...")
    kimi_resp = call_openrouter_with_retry(get_kimi_prompt(codigo_base, lenguaje), KIMI_SYSTEM, MODEL_KIMI)
    
    if not kimi_resp:
        print(f"[{i}] Kimi falló. Pasando a DeepSeek Coder...")
        kimi_resp = call_openrouter_with_retry(get_kimi_prompt(codigo_base, lenguaje), KIMI_SYSTEM, MODEL_DEEPSEEK_CODER)
        
    if kimi_resp:
        codigo_final = extract_code(kimi_resp, lenguaje)
        if not codigo_final:
            codigo_final = kimi_resp.replace(f'```{lenguaje}', '').replace('```', '')
        with open(archivo_codigo, "w", encoding="utf-8") as f:
            f.write(codigo_final.strip())
        print(f"[{i}] COMPLETADA Y OPTIMIZADA.")
    else:
        print(f"[{i}] Fallaron revisores. Guardando base.")
        with open(archivo_codigo, "w", encoding="utf-8") as f:
            f.write(codigo_base.strip())

def run_swarm():
    print("Iniciando AUTO-SWARM v2.1 (SVD Truncation Enforced) para POLYDIM...")
    if not API_KEY:
        print("API_KEY no encontrada. Abortando.")
        return

    perfiles = [
        ("Experto en Fibrados Tangentes y JAX (Python)", "python"),
        ("Especialista en Topología Diferencial y PyTorch (Python)", "python"),
        ("Arquitecto Core de Operaciones Tensoriales Seguras (Rust)", "rust"),
        ("Ingeniero de Optimización Simétrica y Grupos de Lie (Python)", "python"),
        ("Desarrollador de Kernels Paralelos de Bajo Nivel (Rust)", "rust"),
        ("Experto en Álgebra Tensorial y Rotores Clifford (Python)", "python")
    ]
    
    tareas = [(i, perfiles[i % len(perfiles)][0], perfiles[i % len(perfiles)][1]) for i in range(INICIO_TAREA, FIN_TAREA + 1)]
    
    for idx in range(0, len(tareas), MAX_WORKERS):
        lote = tareas[idx:idx + MAX_WORKERS]
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            executor.map(procesar_tarea, lote)
        
        if idx + MAX_WORKERS < len(tareas):
            print(f"--- Lote terminado. Esperando {TIEMPO_ESPERA_ENTRE_LOTES}s para evitar rate limits ---")
            time.sleep(TIEMPO_ESPERA_ENTRE_LOTES)

if __name__ == "__main__":
    run_swarm()
