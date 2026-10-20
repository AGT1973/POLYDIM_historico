"""
================================================================================
POLYDIM - Enjambre de Validacion - Escuadron 1
Tarea 2 / Test 2
Rol asignado: Tester de robustez de IA
Objetivo: Simular la superposicion (composicion) de rotores y verificar
          empiricamente que se conserva el volumen determinante (det = 1)
          incluso bajo estres numerico (cadenas largas, alta dimension,
          angulos extremos).

que has usado el formato de razonamiento adaptado por AGT
================================================================================
"""

import numpy as np


def generar_bivector_aleatorio(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Genera una matriz antisimetrica aleatoria (bivector) de tamano n x n.

    Por que antisimetrica: la exponencial de una matriz antisimetrica es,
    por construccion algebraica garantizada (no por ajuste posterior), una
    matriz ortogonal con determinante +1. Es la forma "limpia" de generar
    un rotor: no generamos una matriz cualquiera y la corregimos, evitamos
    el error desde el origen.
    """
    A = rng.normal(size=(n, n))
    return (A - A.T) / 2.0  # parte antisimetrica pura


def generar_rotor(n: int, rng: np.random.Generator, escala: float = 1.0) -> np.ndarray:
    """
    Construye un rotor (matriz de rotacion) R = exp(escala * B),
    donde B es un bivector aleatorio.

    'escala' permite forzar angulos de rotacion extremos para el
    stress-test (robustez ante casos limite).
    """
    B = generar_bivector_aleatorio(n, rng) * escala
    # Exponencial de matriz vía la serie de Taylor con precision numerica
    # (scipy.linalg.expm seria la opcion de produccion; aqui usamos la
    # formulacion explicita para que el razonamiento quede transparente).
    from scipy.linalg import expm
    return expm(B)


def superponer_rotores(rotores: list[np.ndarray]) -> np.ndarray:
    """
    'Superposicion' = composicion sucesiva de rotores.
    R_total = R_k @ R_(k-1) @ ... @ R_1

    Esto es exactamente lo que ocurre cuando rotas un objeto varias veces
    seguidas: cada rotacion se aplica sobre el resultado de la anterior.
    """
    n = rotores[0].shape[0]
    R_total = np.eye(n)
    for R in rotores:
        R_total = R @ R_total
    return R_total


def test_conservacion_volumen(
    n: int,
    num_rotores_por_cadena: int,
    num_cadenas: int,
    escala_angulo: float = 1.0,
    tolerancia: float = 1e-9,
    seed: int = 42,
) -> dict:
    """
    Framework de robustez: genera 'num_cadenas' cadenas independientes de
    'num_rotores_por_cadena' rotores en dimension n, las superpone, y mide
    cuanto se desvia el determinante resultante de 1.0.

    Devuelve un reporte con estadisticas de robustez, no solo un booleano,
    porque un tester serio no dice "paso/no paso": cuantifica el margen.
    """
    rng = np.random.default_rng(seed)
    desviaciones = []
    fallos = []

    for i in range(num_cadenas):
        cadena = [generar_rotor(n, rng, escala_angulo) for _ in range(num_rotores_por_cadena)]
        R_total = superponer_rotores(cadena)

        det = np.linalg.det(R_total)
        desviacion = abs(det - 1.0)
        desviaciones.append(desviacion)

        # Chequeo adicional de robustez: ortogonalidad (R^T R = I).
        # Un determinante correcto no garantiza por si solo que la matriz
        # sea una rotacion valida (podria "colarse" un error compensado).
        error_ortogonalidad = np.max(np.abs(R_total.T @ R_total - np.eye(n)))

        if desviacion > tolerancia or error_ortogonalidad > tolerancia:
            fallos.append({
                "cadena": i,
                "desviacion_determinante": float(desviacion),
                "error_ortogonalidad": float(error_ortogonalidad),
            })

    desviaciones = np.array(desviaciones)

    reporte = {
        "dimension": n,
        "rotores_por_cadena": num_rotores_por_cadena,
        "cadenas_evaluadas": num_cadenas,
        "escala_angulo": escala_angulo,
        "tolerancia": tolerancia,
        "desviacion_promedio": float(desviaciones.mean()),
        "desviacion_maxima": float(desviaciones.max()),
        "desviacion_std": float(desviaciones.std()),
        "cadenas_fuera_de_tolerancia": len(fallos),
        "porcentaje_robustez": round(100 * (1 - len(fallos) / num_cadenas), 4),
        "detalle_fallos": fallos[:5],  # solo se muestran los primeros 5
    }
    return reporte


def ejecutar_bateria_de_estres():
    """
    Corre tres escenarios de dificultad creciente, como haria un tester
    de robustez real: caso base, caso de alta dimension, y caso de
    cadena larga (para observar acumulacion de error de punto flotante).
    """
    print("=" * 70)
    print("BATERIA DE ESTRES - CONSERVACION DE VOLUMEN EN SUPERPOSICION DE ROTORES")
    print("=" * 70)

    escenarios = [
        {"nombre": "Caso base (n=3, cadena corta)", "n": 3, "k": 10, "cadenas": 500, "escala": 1.0},
        {"nombre": "Alta dimension (n=12)", "n": 12, "k": 10, "cadenas": 300, "escala": 1.0},
        {"nombre": "Cadena larga (posible deriva numerica)", "n": 3, "k": 5000, "cadenas": 50, "escala": 1.0},
        {"nombre": "Angulos extremos (escala=50)", "n": 3, "k": 10, "cadenas": 300, "escala": 50.0},
    ]

    resultados = {}
    for esc in escenarios:
        reporte = test_conservacion_volumen(
            n=esc["n"],
            num_rotores_por_cadena=esc["k"],
            num_cadenas=esc["cadenas"],
            escala_angulo=esc["escala"],
        )
        resultados[esc["nombre"]] = reporte
        print(f"\n--- {esc['nombre']} ---")
        print(f"  Desviacion promedio del determinante: {reporte['desviacion_promedio']:.3e}")
        print(f"  Desviacion maxima: {reporte['desviacion_maxima']:.3e}")
        print(f"  Cadenas fuera de tolerancia: {reporte['cadenas_fuera_de_tolerancia']} / {esc['cadenas']}")
        print(f"  Robustez: {reporte['porcentaje_robustez']}%")

    return resultados


if __name__ == "__main__":
    ejecutar_bateria_de_estres()
