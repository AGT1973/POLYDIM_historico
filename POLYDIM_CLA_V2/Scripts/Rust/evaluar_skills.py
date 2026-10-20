"""
evaluar_skills.py — POLYDIM V1
Evalúa cada skill en la carpeta toolbox contra SKILL_V1_MINI y marca obsoletos.
Corre en local Windows donde Drive Desktop sincroniza la carpeta.

Uso:
    python evaluar_skills.py
    python evaluar_skills.py --ejecutar   # renombra realmente (default: dry-run)

Autor: ai.mpat.agt@gmail.com · 2026-06-28
"""

import os, re, sys
from pathlib import Path
from datetime import datetime

# ─── CONFIG ───────────────────────────────────────────────────────────────────
TOOLBOX = Path(r"G:\Mi unidad\POLYDIM_v1\toolbox")
DRY_RUN = "--ejecutar" not in sys.argv

# ─── CRITERIOS DE EVALUACIÓN ──────────────────────────────────────────────────
# Cada entrada: (patron_nombre, veredicto, razon)
# Veredicto: BORRAR | WARN | KEEP

CRITERIOS = [

    # ── Bootstraps viejos — supersedidos por SKILL_V1_MINI ────────────────────
    (r"skill_polydim_v1_V1_6\.md$",
     "BORRAR",
     "Bootstrap V1_6 — supersedido por SKILL_V1_MINI"),

    (r"SKILL_V1_8_mpat_final\.md$",
     "BORRAR",
     "Bootstrap V1_8 — supersedido por SKILL_V1_MINI"),

    (r"^skill_polydim_v1\.md$",
     "BORRAR",
     "Bootstrap V1_x (produccion antigua) — supersedido por SKILL_V1_MINI"),

    # ── Duplicados con sufijo (1) ─────────────────────────────────────────────
    (r"\(1\)\.md$",
     "BORRAR",
     "Duplicado Drive con sufijo (1) — artefacto de sincronizacion"),

    (r"\(2\)\.md$",
     "BORRAR",
     "Duplicado Drive con sufijo (2) — artefacto de sincronizacion"),

    # ── Skills que usuario quiere eliminar (MCP nativo) ───────────────────────
    # WARN en vez de BORRAR porque tienen lógica de dedup no en MINI.
    # Si el usuario acepta perder el dedup pre-flight, cambiar a BORRAR.
    (r"^create-file\.md$",
     "WARN",
     "USUARIO: quiere eliminar (MCP nativo). "
     "ATENCION: contiene dedup pre-flight + post-write + DROPZONE fallback "
     "que NO esta en SKILL_V1_MINI. Si se borra se pierde proteccion anti-duplicados. "
     "Alternativa: mover logica dedup a MINI y luego borrar."),

    (r"^create-folder\.md$",
     "WARN",
     "USUARIO: quiere eliminar (MCP nativo). "
     "ATENCION: verificar si contiene dedup o logica de CARPETAS_POLYDIM.json "
     "antes de borrar."),

    # ── Skills con lógica única — KEEP siempre ────────────────────────────────
    (r"^SKILL_V1_MINI\.md$",      "KEEP", "Nuevo bootstrap activo"),
    (r"^task-manager\.md$",       "KEEP", "Logica CLAIMED_TASKS + 6 fases + lock race"),
    (r"^relay\.md$",              "KEEP", "Formatos SHORT/FULL + section definitions"),
    (r"^close-relay\.md$",        "KEEP", "Protocolo cierre + SHORT default"),
    (r"^backlog\.md$",            "KEEP", "BACKLOG_VIGENTE.md via registry (no guessing)"),
    (r"^status\.md$",             "KEEP", "READ_ONLY snapshot + stale lock check"),
    (r"^search\.md$",             "KEEP", "Ranked search + READ_ONLY tier"),
    (r"^orphan\.md$",             "KEEP", "RELAY_ARTIFACT_SET + ghost references"),
    (r"^dedup\.md$",              "KEEP", "Delega a reconcile.md + olds/ protocol"),
    (r"^update-file\.md$",        "KEEP", "READ obligatorio + guardrail branch + MIME fix"),
    (r"^reconcile\.md$",          "KEEP", "Resolucion canonica + registry-first"),
    (r"^audit\.md$",              "KEEP", "Registry integrity check + 5 secciones"),
    (r"^lock-protocol\.md$",      "KEEP", "Lock file format + CLAIMED_TASKS"),
    (r"^unlock-protocol\.md$",    "KEEP", "Release lock + BACKLOG patch"),
    (r"^update-folder\.md$",      "KEEP", "Actualiza CARPETAS_POLYDIM.json"),
    (r"^move-file\.md$",          "KEEP", "Dedup destino + guardrail + BORRAR_"),
    (r"^deprecate-file\.md$",     "KEEP", "Lock check + guardrail branch + BORRAR_"),
    (r"^local-ops\.md$",          "KEEP", "LOCAL_OPS JSON queue para ops sin MCP"),
    (r"^polydim-core\.md$",       "KEEP", "Spec del lenguaje"),
]

# ─── MOTOR ────────────────────────────────────────────────────────────────────

def evaluar(nombre):
    for patron, veredicto, razon in CRITERIOS:
        if re.search(patron, nombre, re.IGNORECASE):
            return veredicto, razon
    return "KEEP", "Sin criterio aplicable — mantener por defecto"

def main():
    if not TOOLBOX.exists():
        print(f"ERROR: {TOOLBOX} no encontrada.")
        sys.exit(1)

    archivos = sorted(p for p in TOOLBOX.iterdir()
                      if p.is_file() and p.suffix == ".md"
                      and not p.name.startswith("BORRAR_"))

    print(f"POLYDIM SKILL EVALUATOR — {'DRY RUN' if DRY_RUN else 'EJECUTANDO'}")
    print(f"Carpeta: {TOOLBOX}")
    print(f"Archivos .md encontrados: {len(archivos)}")
    print("─" * 60)

    resultados = {"BORRAR": [], "WARN": [], "KEEP": [], "SIN_CRITERIO": []}

    for archivo in archivos:
        veredicto, razon = evaluar(archivo.name)
        resultados[veredicto].append((archivo, razon))

    # ── Mostrar resumen ────────────────────────────────────────────────────────
    for v, items in resultados.items():
        if not items:
            continue
        print(f"\n[{v}] {len(items)} archivo(s)")
        for arch, razon in items:
            print(f"  {arch.name}")
            if v != "KEEP":
                print(f"    → {razon[:100]}")

    # ── Ejecutar renombrados ───────────────────────────────────────────────────
    borrar   = resultados["BORRAR"]
    advertir = resultados["WARN"]

    print(f"\n{'─'*60}")
    print(f"BORRAR: {len(borrar)}  |  WARN: {len(advertir)}  |  KEEP: {len(resultados['KEEP'])}")

    if DRY_RUN:
        print("\nMODO DRY-RUN — ningun archivo modificado.")
        print("Para ejecutar: python evaluar_skills.py --ejecutar")
        return

    # Renombrar BORRAR → BORRAR_*
    renombrados = 0
    for arch, razon in borrar:
        nuevo = arch.parent / f"BORRAR_{arch.name}"
        if nuevo.exists():
            print(f"  YA EXISTE: {nuevo.name} — skip")
            continue
        arch.rename(nuevo)
        print(f"  OK  {arch.name}  →  BORRAR_{arch.name}")
        renombrados += 1

    # Mostrar WARNs sin renombrar
    if advertir:
        print("\nWARN — requieren decision manual:")
        for arch, razon in advertir:
            print(f"  ??  {arch.name}")
            print(f"      {razon[:120]}")

    print(f"\nRenombrados: {renombrados}  |  Warnings: {len(advertir)}  |  Errores: 0")
    print("Siguiente: seleccionar BORRAR_* en Drive y eliminar.")

if __name__ == "__main__":
    main()
