"""
Empaqueta el kit para alumnos en un ZIP listo para distribuir.
"""
import zipfile
import os
from pathlib import Path
from datetime import datetime

KIT_DIR = Path(r"G:\Mi unidad\POLYDIM_v1_1\para_alumnos")
FECHA = datetime.now().strftime("%Y%m%d")
ZIP_OUT = Path(r"G:\Mi unidad\POLYDIM_v1_1") / f"AGY_KIT_ALUMNOS_{FECHA}.zip"

def main():
    print(f"Empaquetando kit en: {ZIP_OUT}")
    with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in KIT_DIR.rglob("*"):
            if file.is_file() and "__pycache__" not in str(file):
                arcname = "AGY_KIT_ALUMNOS/" + str(file.relative_to(KIT_DIR))
                zf.write(file, arcname)
                print(f"  + {arcname}")
    size_mb = ZIP_OUT.stat().st_size / (1024*1024)
    print(f"\nZIP listo: {ZIP_OUT}  ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()
