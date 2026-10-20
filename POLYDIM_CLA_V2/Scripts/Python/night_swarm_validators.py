import os
import asyncio
from pathlib import Path
try:
    from openai import AsyncOpenAI
except ImportError:
    pass

SOTA_DIR = Path(r"G:\Mi unidad\POLYDIM_v1_1\docs\SOTA_NIGHT_MINING")
VALIDATORS_DIR = Path(r"G:\Mi unidad\POLYDIM_v1_1\docs\SOTA_NIGHT_VALIDATORS")
VALIDATORS_DIR.mkdir(parents=True, exist_ok=True)

# Usamos Llama-3.1 70B o Claude 3.5 Sonnet para escribir el codigo
MODEL_CODER = "meta-llama/llama-3.1-70b-instruct" 

async def generate_validator(report_path, client):
    with open(report_path, 'r', encoding='utf-8') as f:
        report_content = f.read()

    mission_id = report_path.stem.replace("_REPORT", "")
    print(f"[{mission_id}] Analizando hallazgos para proponer validadores empiricos en Python/Rust...")

    prompt = f"""
Eres el Ingeniero de Validacion Empirica del Enjambre POLYDIM-CLA.
Se te ha entregado el siguiente reporte teorico de un Investigador:

{report_content[:5000]}

TU MISION:
Transformar la abstraccion matematica de este reporte en un experimento empírico ejecutable.
1. Define un experimento o test de micro-benchmark.
2. Escribe una Prueba de Concepto (PoC) en Python (JAX, PyTorch o Numpy) o en Rust (ndarray/tch-rs) que valide computacionalmente si las ideas presentadas pueden optimizar POLYDIM-CLA (recordando que usamos Espinores de Clifford y la Identidad de Woodbury en O(N)).
3. Si el reporte teorico era un callejon sin salida, genera un script que lo demuestre empiricamente (ej. mostrando colapso de gradiente o O(N^3)).

Devuelve tu respuesta en formato Markdown con bloques de codigo comentados listos para ejecutar.
"""

    try:
        response = await client.chat.completions.create(
            model=MODEL_CODER,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=3000
        )
        validator_code = response.choices[0].message.content
    except Exception as e:
        validator_code = f"Error generando validador para {mission_id}:\n{e}"
        print(f"[{mission_id}] Error en la generacion.")

    outfile = VALIDATORS_DIR / f"{mission_id}_VALIDATOR.md"
    with open(outfile, 'w', encoding='utf-8') as f:
        f.write(f"# Validador Empírico: {mission_id}\n\n")
        f.write(validator_code)

    print(f"[{mission_id}] Codigo validador generado en {outfile.name}")

async def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    reports = list(SOTA_DIR.glob("*_REPORT.md"))
    if not reports:
        print("No se encontraron reportes teoricos en el directorio de SOTA.")
        return

    print("==================================================")
    print(f"INICIANDO ENJAMBRE VALIDADOR (Detectados {len(reports)} reportes)")
    print(f"Directorio de volcados (Codigo): {VALIDATORS_DIR}")
    print("==================================================")

    tasks = [generate_validator(report, client) for report in reports]
    await asyncio.gather(*tasks)

    print("==================================================")
    print("ENJAMBRE VALIDADOR FINALIZADO. POOL DE PROGRAMAS CREADO.")
    print("==================================================")

if __name__ == "__main__":
    asyncio.run(main())
