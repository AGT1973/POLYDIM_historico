"""
AGY BACKUP DAEMON v1.0
Genera y mantiene un backup sincronizado de toda la configuracion de Antigravity
en el disco de destino especificado.

Ejecutar una vez para backup inicial. Luego correr periodicamente o ante cada cambio.
"""
import os
import shutil
import time
from pathlib import Path
from datetime import datetime

# ============================================================
# CONFIGURACION DE RUTAS
# ============================================================

# Disco destino 
BACKUP_DRIVE = "H:\\Mi unidad"
BACKUP_ROOT = os.path.join(BACKUP_DRIVE, "BACKUP_ANTIGRAVITY")

# Fuentes a respaldar
SOURCES = [
    {
        "label": "01_GLOBAL_CONFIG",
        "src": r"C:\Users\eluithi\.gemini\config",
        "desc": "Configuracion global: AGENTS.md, SOUL template, MCP configs"
    },
    {
        "label": "02_ANTIGRAVITY_APP",
        "src": r"C:\Users\eluithi\.gemini\antigravity",
        "desc": "App data: brain, plugins, skills globales"
    },
    {
        "label": "03_POLYDIM_AGENTS",
        "src": r"G:\Mi unidad\POLYDIM_v1_1\.agents",
        "desc": "Config del proyecto POLYDIM: SOUL, ORCHESTRATOR, skills"
    },
    {
        "label": "04_POLYDIM_DOCS",
        "src": r"G:\Mi unidad\POLYDIM_v1_1\docs",
        "desc": "Documentacion SOTA, papers, validadores"
    },
    {
        "label": "05_POLYDIM_SCRIPTS",
        "src": r"G:\Mi unidad\POLYDIM_v1_1\Scripts",
        "desc": "Scripts Python: validadores JAX, swarm miner, etc."
    },
]

LOG_FILE = os.path.join(BACKUP_ROOT, "_BACKUP_LOG.md")

# ============================================================
# FUNCIONES
# ============================================================

def log(msg, file=None):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if file:
        file.write(line + "\n")

def check_drive(drive):
    if not os.path.exists(drive):
        print(f"[ERROR] El disco {drive} no existe o no esta conectado.")
        print("Discos disponibles:")
        for d in "CDEFGHIJ":
            if os.path.exists(d + ":\\"):
                print(f"  {d}:\\")
        return False
    return True

def backup_source(src_info, backup_root, logfile):
    src = src_info["src"]
    label = src_info["label"]
    desc = src_info["desc"]
    dst = os.path.join(backup_root, label)

    if not os.path.exists(src):
        log(f"[SKIP] Fuente no encontrada: {src}", logfile)
        return 0

    log(f"[SYNC] {label}: {src} -> {dst}", logfile)
    log(f"       ({desc})", logfile)

    copied = 0
    skipped = 0
    errors = 0

    for root, dirs, files in os.walk(src):
        # Excluir carpetas que no interesan en el backup
        dirs[:] = [d for d in dirs if d not in [
            '__pycache__', '.git', 'node_modules',
            'temp_user_data', '.system_generated'
        ]]

        rel_root = os.path.relpath(root, src)
        dst_root = os.path.join(dst, rel_root)
        os.makedirs(dst_root, exist_ok=True)

        for fname in files:
            src_file = os.path.join(root, fname)
            dst_file = os.path.join(dst_root, fname)

            try:
                # Solo copiar si cambio o no existe
                if not os.path.exists(dst_file):
                    shutil.copy2(src_file, dst_file)
                    copied += 1
                else:
                    src_mtime = os.path.getmtime(src_file)
                    dst_mtime = os.path.getmtime(dst_file)
                    if src_mtime > dst_mtime:
                        shutil.copy2(src_file, dst_file)
                        copied += 1
                    else:
                        skipped += 1
            except Exception as e:
                log(f"  [!] Error copiando {fname}: {e}", logfile)
                errors += 1

    log(f"  -> Copiados: {copied} | Sin cambios: {skipped} | Errores: {errors}", logfile)
    return copied

def run_backup(drive=BACKUP_DRIVE):
    print("=" * 60)
    print(" AGY BACKUP DAEMON v1.0")
    print(f" Destino: {drive}")
    print("=" * 60)

    if not check_drive(drive):
        return

    os.makedirs(BACKUP_ROOT, exist_ok=True)

    with open(LOG_FILE, "a", encoding="utf-8") as logfile:
        logfile.write(f"\n\n## Backup {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        log(f"Iniciando backup en: {BACKUP_ROOT}", logfile)

        total_copied = 0
        for source in SOURCES:
            total_copied += backup_source(source, BACKUP_ROOT, logfile)

        # Guardar snapshot de la estructura del backup
        manifest = os.path.join(BACKUP_ROOT, "_MANIFEST.txt")
        with open(manifest, "w", encoding="utf-8") as mf:
            mf.write(f"Backup generado: {datetime.now()}\n")
            mf.write(f"Disco destino: {drive}\n\n")
            for s in SOURCES:
                mf.write(f"- {s['label']}: {s['src']}\n")
                mf.write(f"  {s['desc']}\n\n")

        log(f"\n[OK] Backup completado. Total archivos copiados/actualizados: {total_copied}", logfile)
        log(f"[OK] Log guardado en: {LOG_FILE}", logfile)
        print(f"\nBackup en: {BACKUP_ROOT}")

if __name__ == "__main__":
    run_backup()
