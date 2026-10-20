import os
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime
import ssl

# Desactivar verificación estricta de SSL si da error en entorno local de Windows
ssl._create_default_https_context = ssl._create_unverified_context

# =================================================================
# POLYDIM: SOTA OVERNIGHT SCOUT (Python Autonomous Fallback)
# =================================================================
# Este script actúa como red de seguridad por si los agentes nativos
# sufren un timeout en la plataforma. Scrapea SOTA y genera markdowns.

SAVE_DIR = r"H:\Mi unidad\POLYDIM_CLA\docs\Research\SOTA_GLOBAL"
os.makedirs(SAVE_DIR, exist_ok=True)

# Consultas específicas de interés para la V7
QUERIES = {
    "HARDWARE": 'all:"Categorical Deep Learning" AND all:hardware',
    "SPIN_ROTORS": 'all:"Spin" AND all:"Neural Network" OR all:"Clifford"',
    "MEASURE_THEORY": 'all:"measure theory" AND all:"deep learning"',
    "SINKHORN": 'all:"Sinkhorn" AND all:"Optimal Transport" AND all:"Transformer"',
    "TOPOLOGY_FRANCE_GERMANY": '(all:"Grothendieck Topos" OR all:"Sheaf Theory" OR all:"Leray") AND all:"Neural Network"',
    "MATH_RUSSIA": '(all:"Kolmogorov" OR all:"Gelfand" OR all:"V.I. Arnold") AND all:"Deep Learning"'
}

def query_arxiv(query, max_results=10):
    url = f'http://export.arxiv.org/api/query?search_query={urllib.parse.quote(query)}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=desc'
    try:
        response = urllib.request.urlopen(url)
        return ET.fromstring(response.read())
    except Exception as e:
        print(f"Error consultando ArXiv: {e}")
        return None

def run_scout():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Iniciando escaneo SOTA nocturno...")
    for topic, q in QUERIES.items():
        print(f"Buscando tópico: {topic}...")
        root = query_arxiv(q)
        if root is None:
            continue
        
        filename = os.path.join(SAVE_DIR, f"SOTA_PYTHON_{topic}_{datetime.now().strftime('%Y%m%d')}.md")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# SOTA Overnight Report: {topic}\n\n")
            f.write(f"**Query:** `{q}`\n\n")
            
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                title = entry.find('{http://www.w3.org/2005/Atom}title').text.replace('\n', ' ').strip()
                published = entry.find('{http://www.w3.org/2005/Atom}published').text[:10]
                summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.replace('\n', ' ').strip()
                link = entry.find('{http://www.w3.org/2005/Atom}id').text
                
                f.write(f"### {title} ({published})\n")
                f.write(f"- **Link:** {link}\n")
                f.write(f"- **Abstract:** {summary}\n\n")
        
        print(f"-> Guardado: {filename}")
        time.sleep(3) # Respetar rate limits de la API

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Escaneo finalizado. Esperando al siguiente ciclo...")

if __name__ == '__main__':
    # Bucle infinito (cada 1 hora)
    while True:
        run_scout()
        # Duerme 1 hora (3600 segundos)
        time.sleep(3600)
