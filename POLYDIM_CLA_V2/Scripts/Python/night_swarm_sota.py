import os
import json
import asyncio
import urllib.request
import urllib.parse
from pathlib import Path
from datetime import datetime
try:
    from openai import AsyncOpenAI
except ImportError:
    print("Instalando dependencia openai requerida...")
    os.system("pip install openai")
    from openai import AsyncOpenAI

# Directorio destino absoluto
DEST_DIR = Path(r"G:\Mi unidad\POLYDIM_v1_1\docs\SOTA_NIGHT_MINING")
DEST_DIR.mkdir(parents=True, exist_ok=True)

# 6 Agentes Especializados (Capitanes de Investigación)
MISSIONS = [
    {
        "id": "01_AGY_CATEGORICAL",
        "model": "anthropic/claude-3.5-sonnet", # Claude para razonamiento lógico fuerte
        "query": "Categorical Deep Learning parametric functors natural transformations",
        "role": "Eres el Agente de Topología Categórica. Analiza estos papers e identifica puentes hacia arquitecturas functoriales (como Para) y Lentes Paramétricos. Busca referencias históricas que justifiquen la equivalencia entre arquitecturas de red y funtores."
    },
    {
        "id": "02_AGY_CLIFFORD",
        "model": "meta-llama/llama-3.1-70b-instruct", # Llama para matemáticas puras
        "query": "Clifford algebra spinors equivariant neural networks",
        "role": "Eres el Agente de Álgebra Geométrica. Analiza la literatura sobre redes neuronales que usan álgebra de Clifford, rotores y espinores. Identifica el 'Estado del Arte' en representaciones latentes ortogonales."
    },
    {
        "id": "03_AGY_WOODBURY",
        "model": "google/gemini-1.5-pro", # Gemini Pro para análisis algorítmico y complejidad
        "query": "Woodbury matrix identity optimization machine learning inverse O(N)",
        "role": "Eres el Arquitecto de Complejidad. Mapea la historia y el uso actual de la identidad matricial de Woodbury para eludir inversiones densas O(N^3) en Machine Learning (específicamente en atención, kernels o matrices de covarianza)."
    },
    {
        "id": "04_AGY_GROMOV",
        "model": "anthropic/claude-3-haiku", # Rápido y efectivo para papers
        "query": "Gromov-Wasserstein alignment latent spaces optimal transport",
        "role": "Eres el Agente de Transporte Óptimo. Revisa el SOTA en alineación Gromov-Wasserstein de espacios latentes y arquitecturas sin anclajes (anchorless). Busca los problemas de complejidad y cómo los resuelven hoy."
    },
    {
        "id": "05_AGY_PHYSICS",
        "model": "deepseek/deepseek-chat", # Deepseek para arqueología científica
        "query": "Polydimensional Relativity Pezzaglia C-spaces Carlos Castro",
        "role": "Eres el Físico Teórico. Explora trabajos oscuros sobre Relatividad Polidimensional (Pezzaglia) y Espacios-C (Castro). Haz un mapeo formal (homomorfismo) entre estas teorías del espacio-tiempo y los espacios latentes de redes neuronales."
    },
    {
        "id": "06_AGY_SYNTHESIS",
        "model": "anthropic/claude-3.5-sonnet", 
        "query": "category theory geometric algebra physics formal isomorphism",
        "role": "Eres el Agente Sintetizador (La Mente Enjambre). Encuentra puentes formales históricos entre la teoría de topos, el álgebra geométrica y los espacios no-conmutativos para solidificar los cimientos matemáticos de una arquitectura cognitiva artificial."
    }
]

def search_openalex(query, max_results=12):
    """Busca papers usando la API gratuita de OpenAlex."""
    # Usamos sort by cited_by_count para agarrar papers de peso histórico, 
    # y por relevance (default) para papers recientes específicos.
    url = f"https://api.openalex.org/works?search={urllib.parse.quote(query)}&per-page={max_results}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'POLYDIM_Swarm_Bot/1.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            papers = []
            for work in data.get('results', []):
                title = work.get('title', 'Sin título')
                pub_year = work.get('publication_year', 'Desconocido')
                doi = work.get('doi', 'Sin DOI')
                abstract = work.get('abstract_inverted_index', None)
                
                abs_text = "Sin abstract disponible."
                if abstract:
                    # OpenAlex devuelve abstracts invertidos, reconstruir:
                    words = sorted([(int(pos), word) for word, positions in abstract.items() for pos in positions])
                    abs_text = " ".join([w[1] for w in words])
                
                papers.append(f"Title: {title}\nYear: {pub_year}\nDOI: {doi}\nAbstract: {abs_text}\n")
            return "\n".join(papers)
    except Exception as e:
        return f"Error buscando en OpenAlex: {e}"

async def run_mission(mission, client):
    print(f"[{mission['id']}] Iniciando busqueda de papers para: '{mission['query']}'...")
    papers_text = search_openalex(mission['query'])
    
    prompt = f"""
{mission['role']}

A continuación te presento los abstracts de literatura altamente relevante extraída de bases de datos científicas:

{papers_text}

Misión Estricta para POPL 2027:
1. Extrae los conceptos matemáticos, teóricos y arquitectónicos clave de estos papers.
2. Explica detalladamente cómo estos conceptos respaldan (o contrastan con) nuestra arquitectura "POLYDIM-CLA": un framework donde los tokens son Espinores de Clifford, el ruteo se hace vía Lentes Paramétricos (Functoriales), y el cuello de botella de inversión se elude con la identidad de Woodbury en O(N).
3. Redacta un reporte técnico de arqueología matemática de nivel académico (formato Markdown), con citas precisas de los autores proporcionados, diseñado para integrarse directamente en la sección "Related Work" y "Theoretical Foundations". NO uses adulaciones ni lenguaje comercial. Sed riguroso.
"""
    
    print(f"[{mission['id']}] Despertando a {mission['model']} para la sintesis cognitiva...")
    
    try:
        response = await client.chat.completions.create(
            model=mission['model'],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, # Baja temperatura para rigor matemático
            max_tokens=3000
        )
        report = response.choices[0].message.content
    except Exception as e:
        report = f"Error crítico al contactar al modelo {mission['model']}:\n{e}"
        print(f"[{mission['id']}] Fallo la sintesis.")

    outfile = DEST_DIR / f"{mission['id']}_REPORT.md"
    with open(outfile, 'w', encoding='utf-8') as f:
        f.write(f"# Misión de Arqueología SOTA: {mission['id']}\n")
        f.write(f"**Fecha y Hora:** {datetime.now().isoformat()}\n")
        f.write(f"**Query OpenAlex:** `{mission['query']}`\n")
        f.write(f"**Agente de Computación:** `{mission['model']}`\n\n")
        f.write("---\n\n")
        f.write(report)
        f.write("\n\n---\n## Anexo: Raw Data (OpenAlex)\n\n<details><summary>Ver Papers Extraídos</summary>\n\n")
        f.write(papers_text)
        f.write("\n</details>\n")
        
    print(f"[{mission['id']}] Reporte completado en: {outfile.name}")

async def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ADVERTENCIA: No se detecto OPENROUTER_API_KEY en variables de entorno.")
        print("El enjambre fallará en la fase de síntesis LLM (aunque bajará los papers).")
        print("Asegúrate de ejecutar set OPENROUTER_API_KEY=tu_llave o similar.")
        
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key, 
    )
    
    print("==================================================")
    print("INICIANDO ENJAMBRE NOCTURNO POLYDIM (6 Agentes)")
    print(f"Directorio de volcado: {DEST_DIR}")
    print("==================================================")
    
    # Lanzamos el enjambre de forma concurrente
    tasks = [run_mission(m, client) for m in MISSIONS]
    await asyncio.gather(*tasks)
    
    print("\n==================================================")
    print("EL ENJAMBRE HA TERMINADO. MISION COMPLETADA.")
    print("==================================================")

if __name__ == "__main__":
    asyncio.run(main())
