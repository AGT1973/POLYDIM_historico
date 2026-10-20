import os
import glob
import time
import re

def monitor_progreso():
    base_dir = r"G:\Mi unidad\POLYDIM_v1_1\Alumnos"
    print("🐝 Monitoreando el progreso del Enjambre POLYDIM...")
    print("Presiona Ctrl+C para detener.\n")
    
    total_tareas = 200
    
    try:
        while True:
            # Buscar carpetas tarea_1, tarea_2, etc. (ignorando extensiones)
            carpetas = os.listdir(base_dir)
            carpetas_tareas = []
            for c in carpetas:
                if os.path.isdir(os.path.join(base_dir, c)) and re.match(r'^tarea_\d+$', c):
                    carpetas_tareas.append(os.path.join(base_dir, c))
            
            tareas_completadas = 0
            detalles_completadas = []
            
            for carpeta in carpetas_tareas:
                # Verificamos si hay al menos un archivo .md o .json dentro de la carpeta
                archivos_md = glob.glob(os.path.join(carpeta, "*.md"))
                archivos_json = glob.glob(os.path.join(carpeta, "*.json"))
                
                if archivos_md or archivos_json:
                    tareas_completadas += 1
                    num_tarea = os.path.basename(carpeta).replace("tarea_", "")
                    detalles_completadas.append(num_tarea)
            
            porcentaje = (tareas_completadas / total_tareas) * 100
            
            # Limpiar consola (funciona en Windows)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print("="*50)
            print("📊 ESTADO DEL ENJAMBRE POLYDIM")
            print("="*50)
            print(f"Progreso Total: {tareas_completadas}/{total_tareas} tareas completadas ({porcentaje:.1f}%)")
            print("-" * 50)
            
            # Mostrar las últimas 10 tareas completadas
            if detalles_completadas:
                print("Últimas tareas detectadas como listas:")
                # Ordenar numéricamente si es posible
                try:
                    detalles_completadas.sort(key=int)
                except ValueError:
                    detalles_completadas.sort()
                    
                print(", ".join(detalles_completadas[-20:]))
                
            print("\nEsperando nuevos resultados... (Actualizando cada 5 segundos)")
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\nMonitoreo detenido por el usuario.")

if __name__ == "__main__":
    monitor_progreso()
