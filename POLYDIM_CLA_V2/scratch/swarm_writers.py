import os
import asyncio
import httpx
from pathlib import Path

# Configuración de Rutas
BASE_DIR = Path(r"G:\Mi unidad\POLYDIM_v1_1")
PAPER_V7_PATH = BASE_DIR / "docs" / "paper" / "POLYDIM_CLA_PAPER_V7_FINAL.md"
CONST_V3_PATH = BASE_DIR / "docs" / "constitution" / "POLYDIM_CONSTITUCION_V3_FORMAL.md"

# Modelos en OpenRouter
MODELS = {
    "qwen": "qwen/qwen-2.5-72b-instruct",
    "kimi": "moonshotai/kimi-k2",
    "zia": "z-ai/glm-5",
    "deepseek": "deepseek/deepseek-chat" # DeepSeek V3
}

API_KEY = os.environ.get("OPENROUTER_API_KEY", "REDACTED-OPENROUTER-KEY")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/AGT1973/POLYDIM",
    "X-Title": "POLYDIM Swarm Writers"
}

async def call_openrouter(client, model_id, system_prompt, user_prompt):
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2
    }
    # Intentar hasta 3 veces por si hay rate limit o timeouts
    for attempt in range(3):
        try:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=300.0
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            else:
                print(f"[ERROR] {model_id} (Intento {attempt+1}): {resp.status_code} - {resp.text}")
                await asyncio.sleep(5)
        except Exception as e:
            print(f"[EXCEPCION] {model_id} (Intento {attempt+1}): {e}")
            await asyncio.sleep(5)
    return f"ERROR: Falló la generación para el modelo {model_id}"

async def generate_swarm():
    # Leer insumos
    paper_content = PAPER_V7_PATH.read_text(encoding="utf-8")
    const_content = CONST_V3_PATH.read_text(encoding="utf-8")

    async with httpx.AsyncClient() as client:
        # ==========================================
        # FASE 1: Generar la Constitución para cada modelo
        # ==========================================
        print("--- INICIANDO FASE 1: GENERACION DE CONSTITUCIONES ---")
        
        system_const = (
            "Eres un matemático y teórico del Deep Learning experto en representaciones latentes y topología. "
            "Tu tarea es reescribir la Constitución de POLYDIM en un formato riguroso y denso."
        )
        
        tasks_const = {}
        for name, model_id in MODELS.items():
            prompt_const = (
                f"Basándote en la última versión del Paper de POLYDIM-CLA:\n\n{paper_content}\n\n"
                f"Y en la versión previa de la Constitución:\n\n{const_content}\n\n"
                f"Escribe tu versión definitiva, formal y matemáticamente rigurosa de la CONSTITUCIÓN DE POLYDIM. "
                f"Utiliza LaTeX ($$) para las ecuaciones y formulas (Hodge Laplacian, Clifford Algebra, etc.). "
                f"Define los axiomas y transformaciones geométricas de los agentes. Responde directamente con el documento en Markdown."
            )
            tasks_const[name] = call_openrouter(client, model_id, system_const, prompt_const)
            
        results_const = await asyncio.gather(*tasks_const.values())
        constructions = dict(zip(tasks_const.keys(), results_const))
        
        # Guardar constituciones
        for name, content in constructions.items():
            out_path = BASE_DIR / "docs" / "constitution" / f"POLYDIM_CONSTITUCION_{name.upper()}.md"
            out_path.write_text(content, encoding="utf-8")
            print(f"[OK] Guardada Constitucion: {out_path.name}")

        # ==========================================
        # FASE 2: Generar el Paper/Abstracción para cada modelo
        # ==========================================
        print("\n--- INICIANDO FASE 2: GENERACION DE PAPERS ---")
        
        system_paper = (
            "Eres un escritor académico experto. Tu objetivo es generar una síntesis formal "
            "y abstracta del paper de POLYDIM-CLA, incorporando los hallazgos empíricos del benchmark de hardware."
        )
        
        tasks_paper = {}
        for name, model_id in MODELS.items():
            my_const = constructions[name]
            prompt_paper = (
                f"Basándote en la Constitución que acabas de escribir:\n\n{my_const}\n\n"
                f"Y en la última versión base del Paper (que contiene los benchmarks reales de TPU/GPU):\n\n{paper_content}\n\n"
                f"Escribe tu versión consolidada, formal e hiper-rigurosa del PAPER DE POLYDIM-CLA. "
                f"Especialmente enfócate en formalizar los resultados empíricos, la conjetura de colapso cognitivo y el bypass de Woodbury. "
                f"Utiliza un tono adecuado para una conferencia top de Ciencias de la Computación (ej. POPL). Responde directamente con el documento en Markdown."
            )
            tasks_paper[name] = call_openrouter(client, model_id, system_paper, prompt_paper)
            
        results_paper = await asyncio.gather(*tasks_paper.values())
        papers = dict(zip(tasks_paper.keys(), results_paper))
        
        # Guardar papers
        for name, content in papers.items():
            out_path = BASE_DIR / "docs" / "paper" / f"POLYDIM_PAPER_{name.upper()}.md"
            out_path.write_text(content, encoding="utf-8")
            print(f"[OK] Guardado Paper: {out_path.name}")

if __name__ == "__main__":
    asyncio.run(generate_swarm())
