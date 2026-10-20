import numpy as np

def producto_punto(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calcula el producto escalar (dot product) entre dos vectores.
    Análogo cotidiano: total de un carrito = precios · cantidades.
    """
    assert v1.shape == v2.shape, "Los vectores deben tener la misma dimensión"
    return float(np.dot(v1, v2))

def similitud_coseno(v1: np.ndarray, v2: np.ndarray) -> float:
    """Mide el 'ángulo' entre dos vectores de información (0=distintos, 1=idénticos).
    Se usa para comparar embeddings semánticos en IA.
    """
    num = np.dot(v1, v2)
    den = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(num / den) if den != 0 else 0.0

def error_l2(v_calculado: np.ndarray, v_referencia: np.ndarray) -> float:
    """Distancia euclidiana entre el vector calculado y uno de referencia.
    Cuanto más cerca de 0, más preciso es el cálculo.
    Ejemplo cotidiano: es como medir en metros qué tan lejos cayó
    tu tiro respecto al centro del arco.
    """
    return float(np.linalg.norm(v_calculado - v_referencia))

if __name__ == "__main__":
    # --- Verificación (chequeo antes de entregar) ---
    precios = np.array([10.0, 20.0, 5.0])
    cantidades = np.array([2.0, 1.0, 3.0])
    assert producto_punto(precios, cantidades) == 55.0, "Fallo en ejemplo del carrito"

    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    assert abs(similitud_coseno(a, b) - 1.0) < 1e-9, "Vectores idénticos deben dar similitud 1"

    c = np.array([1.0, 0.0])
    d = np.array([0.0, 1.0])
    assert abs(similitud_coseno(c, d) - 0.0) < 1e-9, "Vectores ortogonales deben dar similitud 0"

    # Verificación de L2
    ref = np.array([55.0])          # resultado esperado del producto punto (carrito)
    calc = np.array([producto_punto(precios, cantidades)])
    error = error_l2(calc, ref)
    assert error < 1e-9, "El error debería ser prácticamente cero"
    
    print("Todos los chequeos pasaron correctamente.")
    print(f"Error L2 respecto al valor de referencia: {error}")
