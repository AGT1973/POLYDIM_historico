import urllib.request
import urllib.parse
import json
import xml.etree.ElementTree as ET
import re
import time
import os

OUTPUT_FILE = r"G:\Mi unidad\POLYDIM_v1_1\docs\GLOBAL_SOTA_MINING\AUTORES_CONTACTOS_SOTA.md"
KEYWORDS = [
    "Clifford Algebra Neural Network",
    "Geometric Algebra Transformer",
    "Stiefel Manifold Optimization",
    "Cayley Transform Orthogonal",
    "Woodbury Identity Deep Learning",
    "Grothendieck Topos Neural",
    "Sheaf Laplacian Neural",
    "Spinor Neural Networks",
    "Continuous Representation AI"
]

def mine_arxiv(query, max_results=30):
    url = f"http://export.arxiv.org/api/query?search_query=all:{urllib.parse.quote(query)}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        xml_data = response.read()
        root = ET.fromstring(xml_data)
        
        results = []
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns).text.replace('\n', ' ')
            summary = entry.find('atom:summary', ns).text.replace('\n', ' ')
            authors = [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)]
            link = entry.find('atom:id', ns).text
            results.append({
                'title': title,
                'authors': authors,
                'summary': summary,
                'link': link
            })
        return results
    except Exception as e:
        print(f"[!] Error ArXiv para {query}: {e}")
        return []

def mine_semantic_scholar(query, limit=10):
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={urllib.parse.quote(query)}&limit={limit}&fields=title,authors,abstract,url,year"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        data = json.loads(response.read().decode('utf-8'))
        results = []
        for p in data.get('data', []):
            authors = [a.get('name', '') for a in p.get('authors', [])]
            results.append({
                'title': p.get('title', ''),
                'authors': authors,
                'summary': p.get('abstract', '') or '',
                'link': p.get('url', '')
            })
        return results
    except Exception as e:
        print(f"[!] Error Semantic Scholar para {query}: {e}")
        return []

def extract_contacts(text):
    # Buscar emails y menciones tipicas
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    emails = list(set(re.findall(email_pattern, text)))
    return emails

def run_swarm():
    print("==================================================")
    print(" INICIANDO ENJAMBRE SABUESO (MODO BUCLE INFINITO) ")
    print("==================================================")
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write("\n\n# =========================================\n")
        f.write(f"# NUEVA SESIÓN DE MINERÍA PROFUNDA: {time.ctime()}\n")
        f.write("# Objetivo: Extracción Masiva de Contactos y SOTA\n")
        f.write("# =========================================\n\n")

    iteration = 1
    seen_links = set()

    while True:
        print(f"\n--- [ITERACIÓN {iteration}] ---")
        for kw in KEYWORDS:
            print(f"> Sabueso investigando: '{kw}'...")
            
            papers_arxiv = mine_arxiv(kw)
            papers_ss = mine_semantic_scholar(kw)
            all_papers = papers_arxiv + papers_ss
            
            new_finds = 0
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                for p in all_papers:
                    if p['link'] in seen_links or not p['link']:
                        continue
                    
                    seen_links.add(p['link'])
                    new_finds += 1
                    
                    # Generar inferencia de dominio de email de los autores basados en universidades típicas (heurística si no hay email claro)
                    emails = extract_contacts(p['summary'])
                    
                    f.write(f"### {p['title']}\n")
                    f.write(f"- **Autores / Investigadores Clave:** {', '.join(p['authors'])}\n")
                    f.write(f"- **Link / DOI:** {p['link']}\n")
                    if emails:
                        f.write(f"- **CONTACTO (EMAILS DETECTADOS):** {', '.join(emails)}\n")
                    f.write(f"- **Resumen:** {p['summary'][:400]}...\n\n")
            
            print(f"  -> {new_finds} nuevos papers/contactos minados. Escribiendo a disco...")
            time.sleep(15) # Pausa por rate limit de APIs
            
        print("\n[+] Ciclo de palabras clave completado.")
        print("[zZz] Durmiendo el enjambre 30 minutos para evitar IP bans y resetear buffers...")
        iteration += 1
        time.sleep(1800) # Duerme 30 min y vuelve a barrer con distintas heurísticas de tiempo

if __name__ == "__main__":
    try:
        run_swarm()
    except KeyboardInterrupt:
        print("Enjambre detenido manualmente.")
