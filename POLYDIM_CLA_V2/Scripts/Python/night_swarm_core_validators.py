import os
import asyncio
from pathlib import Path
try:
    from openai import AsyncOpenAI
except ImportError:
    pass

MANIFESTO_PATH = Path(r"G:\Mi unidad\POLYDIM_v1_1\docs\WOODBURY_CORE_ARGUMENT.md")
VALIDATORS_DIR = Path(r"G:\Mi unidad\POLYDIM_v1_1\docs\CORE_NIGHT_VALIDATORS")
VALIDATORS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CODER = "meta-llama/llama-3.1-70b-instruct"

MISSIONS = [
    {
        "id": "JAX_WOODBURY_STABILITY",
        "lang": "Python (JAX)",
        "task": "Escribe un benchmark empirico en JAX BF16 comparando la inversion matricial densa O(N^3) frente a la Identidad de Woodbury en O(NK^2). Muestra como Woodbury evita colapsar en NaNs para N=8192."
    },
    {
        "id": "JAX_CAYLEY_GENERATOR",
        "lang": "Python (JAX/Numpy)",
        "task": "Escribe un validador que tome una matriz antisimetrica (Lambda = Lambda - Lambda.T) y aplique la Transformada de Cayley combinada con Woodbury para generar una matriz ortogonal pura, validando la isometria (Q^T Q = I)."
    },
    {
        "id": "RUST_CLIFFORD_ROTORS",
        "lang": "Rust (ndarray o tch-rs)",
        "task": "Implementa el sandwich espinorial de Clifford (x' = R x R_reversed) usando un enfoque de subespacio Low-Rank, validando el Teorema de Cartan-Dieudonne. Escribe el codigo en Rust compilable."
    },
    {
        "id": "PYTHON_TOPOS_SIEVE",
        "lang": "Python",
        "task": "Escribe una Prueba de Concepto que emule el 'Topos Coverage Sieve' y la logica intuicionista con el clasificador Omega. El sistema debe demostrar un 'Colapso Elegante' a la fibra vacia cuando ocurre un error geometrico, sin crashear."
    }
]

async def generate_core_validator(mission, manifesto_content, client):
    print(f"[{mission['id']}] Diseñando validador empirico en {mission['lang']}...")
    
    prompt = f"""
Eres el Ingeniero de Validacion Empirica del Enjambre POLYDIM-CLA.
Se te ha entregado el MANIFIESTO CORE de la arquitectura:

---
{manifesto_content}
---

TU MISION:
{mission['task']}

Requisitos:
1. Lenguaje: {mission['lang']}.
2. Basate EXCLUSIVAMENTE en las matematicas y conceptos del manifiesto.
3. El codigo debe estar altamente optimizado y demostrar empiricamente las propiedades teoricas.
4. Devuelve el resultado en formato Markdown con bloques de codigo comentados, listos para correr o compilar. No incluyas emojis.
"""

    try:
        response = await client.chat.completions.create(
            model=MODEL_CODER,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=3500
        )
        validator_code = response.choices[0].message.content
    except Exception as e:
        validator_code = f"Error generando validador para {mission['id']}:\n{e}"
        print(f"[{mission['id']}] Error en la generacion.")

    outfile = VALIDATORS_DIR / f"{mission['id']}_CORE_VALIDATOR.md"
    with open(outfile, 'w', encoding='utf-8') as f:
        f.write(f"# Validador Empírico (Core): {mission['id']}\n\n")
        f.write(validator_code)

    print(f"[{mission['id']}] Codigo validador generado en {outfile.name}")

async def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Variable de entorno OPENROUTER_API_KEY no encontrada.")
        return
        
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    if not MANIFESTO_PATH.exists():
        print(f"ERROR: No se encontro el manifiesto en {MANIFESTO_PATH}")
        return

    with open(MANIFESTO_PATH, 'r', encoding='utf-8') as f:
        manifesto_content = f.read()

    print("==================================================")
    print(f"INICIANDO ENJAMBRE VALIDADOR (CORE MANIFESTO)")
    print(f"Directorio de volcados: {VALIDATORS_DIR}")
    print("==================================================")

    tasks = [generate_core_validator(mission, manifesto_content, client) for mission in MISSIONS]
    await asyncio.gather(*tasks)

    print("==================================================")
    print("ENJAMBRE VALIDADOR FINALIZADO. POOL DE PROGRAMAS CREADO.")
    print("==================================================")

if __name__ == "__main__":
    asyncio.run(main())
