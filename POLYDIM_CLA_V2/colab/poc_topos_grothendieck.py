import numpy as np

def main():
    print("--- PRUEBA DE CONCEPTO: TOPOS DE GROTHENDIECK ---")
    print("Demostración de tolerancia a fallos mediante geometría continua (Cero IFs)\n")

    # 1. DEFINICIÓN DEL ESPACIO SEMÁNTICO (R^4)
    # Ejes semánticos: [Es_Carbohidrato, Es_Largo, Es_Grano, Es_Líquido]
    FIDEOS = np.array([1.0, 1.0, 0.0, 0.0])
    ARROZ  = np.array([1.0, 0.0, 1.0, 0.0])
    AGUA   = np.array([0.0, 0.0, 0.0, 1.0])

    # Nombres para visualización
    nombres = {
        tuple(FIDEOS): "FIDEOS",
        tuple(ARROZ): "ARROZ",
        tuple(AGUA): "AGUA"
    }

    # 2. EL AGENTE Y EL ENTORNO
    # El Agente tiene la "Skill" que requiere Fideos.
    intencion_agente = FIDEOS
    
    # El Entorno (La Realidad). Solo tenemos Arroz y Agua.
    # Matemáticamente, el Entorno es una Variedad generada por los vectores disponibles.
    entorno_real = np.column_stack([ARROZ, AGUA])

    # 3. EL CLASIFICADOR DE SUBOBJETOS (Ω)
    # En la computación clásica haríamos:
    # if not "FIDEOS" in entorno: raise Exception("NullReference")
    #
    # En GCGT (POLYDIM), el entorno actúa como un Funtor de Proyección Ortogonal.
    # Matriz de Proyección P = A (A^T A)^-1 A^T
    # Toda intención es "juzgada" por esta matriz.
    P_Omega = entorno_real @ np.linalg.pinv(entorno_real)

    # 4. EJECUCIÓN (EL COLAPSO TOPOLÓGICO)
    # La intención interactúa con la realidad multiplicándose por el Clasificador.
    # Los componentes inválidos (Es_Largo) sufren un pullback a la fibra vacía (Ø = 0.0).
    # Los componentes válidos (Es_Carbohidrato) sobreviven.
    resultado_ejecucion = P_Omega @ intencion_agente

    print(f"Vector Intención (FIDEOS): {intencion_agente}")
    print(f"Vector Realidad Obtenida:  {resultado_ejecucion}\n")

    # 5. VERIFICACIÓN GEOMÉTRICA (INTERPRETACIÓN)
    # Analizamos qué terminó cocinando el Agente calculando distancias
    dist_a_fideos = np.linalg.norm(resultado_ejecucion - FIDEOS)
    dist_a_arroz  = np.linalg.norm(resultado_ejecucion - ARROZ)
    dist_a_agua   = np.linalg.norm(resultado_ejecucion - AGUA)

    print("Distancias Geodésicas post-colapso:")
    print(f"Distancia al estado 'Cocinar Fideos': {dist_a_fideos:.4f}")
    print(f"Distancia al estado 'Cocinar Arroz':  {dist_a_arroz:.4f}")
    print(f"Distancia al estado 'Cocinar Agua':   {dist_a_agua:.4f}\n")

    # El resultado más cercano determina la acción final en el espacio físico.
    # Matemáticamente, dist_a_arroz es menor que dist_a_fideos.
    # El sistema se ha adaptado y ha decidido hacer arroz, puramente por álgebra lineal,
    # SIN evaluar ni un solo condicional (IF).

    print("CONCLUSIÓN: El sistema asimiló la falta de Fideos y colapsó hacia el")
    print("reemplazo semántico más cercano (Arroz) conservando el rasgo [Carbohidrato].")
    print("No se generó ninguna excepción. Causalidad topológica preservada.")

if __name__ == "__main__":
    main()
