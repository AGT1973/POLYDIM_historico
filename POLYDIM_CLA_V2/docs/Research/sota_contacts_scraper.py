import os
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import json
from datetime import datetime
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

SAVE_DIR = r"H:\Mi unidad\POLYDIM_CLA\docs\Research\SOTA_GLOBAL"
os.makedirs(SAVE_DIR, exist_ok=True)
FILE_PATH = os.path.join(SAVE_DIR, "AI_RESEARCHERS_DIRECTORY.md")

# Buscamos a los verdaderos autores de SOTA en las instituciones clave
QUERIES = [
    ('Rusia (Skoltech, Yandex, MIPT)', 'all:"Tensor Train" AND (all:"Skolkovo" OR all:"Skoltech" OR all:"Yandex" OR all:"MIPT")'),
    ('Europa (INRIA, ELLIS, DeepMind)', 'all:"Categorical Deep Learning" AND (all:"INRIA" OR all:"Max Planck" OR all:"DeepMind" OR all:"ELLIS")'),
    ('Spin Rotors Global', 'all:"Clifford" AND all:"Neural Network"'),
    ('Latent MAS', 'all:"Latent" AND all:"Multi-Agent" AND all:"Transformer"')
]

def scrape_authors():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Iniciando recolector de contactos SOTA...")
    
    with open(FILE_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n## Escaneo Automático: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    
    for region, q in QUERIES:
        print(f"Scrapeando {region}...")
        url = f'http://export.arxiv.org/api/query?search_query={urllib.parse.quote(q)}&start=0&max_results=50&sortBy=submittedDate&sortOrder=desc'
        
        try:
            response = urllib.request.urlopen(url)
            root = ET.fromstring(response.read())
            
            with open(FILE_PATH, "a", encoding="utf-8") as f:
                f.write(f"### Región / Tópico: {region}\n")
                
                for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                    title = entry.find('{http://www.w3.org/2005/Atom}title').text.replace('\n', ' ').strip()
                    link = entry.find('{http://www.w3.org/2005/Atom}id').text
                    
                    authors = []
                    for author in entry.findall('{http://www.w3.org/2005/Atom}author'):
                        name = author.find('{http://www.w3.org/2005/Atom}name').text
                        authors.append(name)
                    
                    f.write(f"- **Paper:** [{title}]({link})\n")
                    f.write(f"  - **Autores:** {', '.join(authors)}\n")
                    f.write(f"  - **Búsqueda manual sugerida (X/LinkedIn/GitHub):** Buscar nombres junto a la institución o tema.\n\n")
                    
        except Exception as e:
            print(f"Error en {region}: {e}")
            
        time.sleep(4) # Rate limit

if __name__ == '__main__':
    while True:
        scrape_authors()
        print("Ciclo terminado. Durmiendo 2 horas...")
        time.sleep(7200)
