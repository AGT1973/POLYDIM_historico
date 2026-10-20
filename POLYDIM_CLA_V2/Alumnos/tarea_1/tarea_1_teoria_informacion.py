"""
Tarea 1 - Enjambre POLYDIM
Alumno: ai.mpat.mimi02@gmail.com
Rol aplicado: Ingeniero de Teoria de la Informacion

Nota pedagogica:
La consigna original ("Escribe codigo que simule aplique multiplicaciones
vectoriales usando librerias matematicas compatibles") estaba gramaticalmente
incompleta en el documento fuente. Se interpreto como: implementar y verificar
operaciones de multiplicacion vectorial (producto punto y similitud coseno)
usando NumPy, incluyendo chequeos automaticos antes de la entrega.

Las metricas "SPS" y "GeoID Stability" pedidas por el formato original del
enjambre no tienen definicion matematica en el documento fuente. Fueron
reemplazadas por error_L2 (norma euclidiana), una metrica estandar y
verificable en algebra lineal / machine learning, por decision explicita
del alumno.
"""

import numpy as np


def producto_punto(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calcula el producto escalar (dot product) entre dos vectores.
    Analogia cotidiana: total de un carrito de compras = precios . cantidades.
    """
    assert v1.shape == v2.shape, "Los vectores deben tener la misma dimension"
    return float(np.dot(v1, v2))


def similitud_coseno(v1: np.ndarray, v2: np.ndarray) -> float:
    """Mide el 'angulo' entre dos vectores de informacion (0=distintos, 1=identicos).
    Se usa para comparar embeddings semanticos en IA.
    """
    num = np.dot(v1, v2)
    den = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(num / den) if den != 0 else 0.0


def error_l2(v_calculado: np.ndarray, v_referencia: np.ndarray) -> float:
    """Distancia euclidiana entre el vector calculado y uno de referencia.
    Metrica sustituta de 'SPS'/'GeoID Stability' (no definidas en el documento fuente).
    Cuanto mas cerca de 0, mas preciso es el calculo.
    """
    return float(np.linalg.norm(v_calculado - v_referencia))


if __name__ == "__main__":
    # --- Ejemplo 1: carrito de compras ---
    precios = np.array([10.0, 20.0, 5.0])
    cantidades = np.array([2.0, 1.0, 3.0])
    total = producto_punto(precios, cantidades)
    assert total == 55.0, "Fallo en ejemplo del carrito"
    print(f"Total del carrito (producto punto): {total}")

    # --- Ejemplo 2: similitud entre vectores de gustos (recomendador) ---
    usuario = np.array([0.9, 0.1, 0.3])
    pelicula = np.array([0.8, 0.0, 0.4])
    sim = similitud_coseno(usuario, pelicula)
    print(f"Similitud coseno usuario-pelicula: {sim:.4f}")

    # Casos limite de verificacion
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    assert abs(similitud_coseno(a, b) - 1.0) < 1e-9, "Vectores identicos deben dar similitud 1"

    c = np.array([1.0, 0.0])
    d = np.array([0.0, 1.0])
    assert abs(similitud_coseno(c, d) - 0.0) < 1e-9, "Vectores ortogonales deben dar similitud 0"

    # --- Metrica sustituta de validacion ---
    ref = np.array([55.0])
    calc = np.array([total])
    error = error_l2(calc, ref)
    assert error < 1e-9, "El error deberia ser practicamente cero"
    print(f"Error L2 respecto al valor de referencia: {error}")

    print("Todos los chequeos pasaron correctamente.")
