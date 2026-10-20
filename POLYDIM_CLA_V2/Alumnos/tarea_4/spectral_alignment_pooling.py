"""
POLYDIM SWARM - Tarea 4 / Test 4
Alumno: ai.mpat.mimi01@gmail.com

Interpretacion tecnica adoptada (justificada en el .md adjunto):
  1. Pooling promedio temporal (compresion)   -> operador de reduccion P
  2. Reconstruccion optima lineal (descompresion) -> pseudo-inversa de P (min. cuadrados)
  3. Metrica de "Alineacion Espectral" (validacion) -> angulos principales
     entre el subespacio de PCA de la señal original y el de la reconstruida.

Si el proceso degrada demasiado la señal, se reporta como FALLA MATEMATICA
en vez de simular un exito falso (regla de la Piramide de Validacion).
"""

import numpy as np

RNG = np.random.default_rng(42)


def generar_señal(n_pasos: int, n_canales: int, n_componentes: int = 3) -> np.ndarray:
    """Genera una señal multivariada sintetica como combinacion de pocas
    componentes senoidales de baja frecuencia (esto la hace compresible,
    igual que una señal real de sensores)."""
    t = np.linspace(0, 4 * np.pi, n_pasos)
    base = np.stack(
        [np.sin((k + 1) * t + k) for k in range(n_componentes)], axis=1
    )  # (n_pasos, n_componentes)
    mezcla = RNG.normal(size=(n_componentes, n_canales))  # combina en n_canales
    señal = base @ mezcla
    ruido = RNG.normal(scale=0.05, size=señal.shape)
    return señal + ruido  # (n_pasos, n_canales)


def operador_pooling(n_pasos: int, ventana: int) -> np.ndarray:
    """Construye la matriz P (n_bloques x n_pasos) que hace average pooling.
    Cada fila de P promedia 'ventana' pasos consecutivos.
    señal_pooled = P @ señal
    """
    assert n_pasos % ventana == 0, "n_pasos debe ser multiplo de ventana"
    n_bloques = n_pasos // ventana
    P = np.zeros((n_bloques, n_pasos))
    for b in range(n_bloques):
        P[b, b * ventana:(b + 1) * ventana] = 1.0 / ventana
    return P


def reconstruccion_optima_lineal(P: np.ndarray, señal_pooled: np.ndarray) -> np.ndarray:
    """Reconstruye la señal original a partir de la version 'pooleada'
    usando la pseudo-inversa de P (solucion de minimos cuadrados de
    P @ X = señal_pooled). Es la mejor reconstruccion lineal posible
    dado que P no es invertible (pierde informacion)."""
    P_pinv = np.linalg.pinv(P)          # (n_pasos x n_bloques)
    return P_pinv @ señal_pooled        # (n_pasos, n_canales)


def subespacio_pca(X: np.ndarray, k: int) -> np.ndarray:
    """Devuelve los k autovectores principales (columnas ortonormales)
    de la matriz de covarianza de X."""
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = Xc.T @ Xc
    autovalores, autovectores = np.linalg.eigh(cov)
    orden = np.argsort(autovalores)[::-1]
    return autovectores[:, orden[:k]]


def alineacion_espectral(U: np.ndarray, V: np.ndarray) -> dict:
    """Calcula los angulos principales entre dos subespacios U y V
    (ambos con columnas ortonormales) via la descomposicion SVD de U^T V.
    Devuelve el SPS (Semantic Preservation Score) = coseno promedio de
    los angulos principales, en [0, 1]. 1.0 = subespacios identicos."""
    _, valores_singulares, _ = np.linalg.svd(U.T @ V)
    cosenos = np.clip(valores_singulares, -1.0, 1.0)
    angulos_grados = np.degrees(np.arccos(cosenos))
    sps = float(np.mean(cosenos))
    return {"cosenos_principales": cosenos.tolist(),
            "angulos_grados": angulos_grados.tolist(),
            "SPS": sps}


def ejecutar_pipeline(n_pasos=240, n_canales=8, ventana=6, k_componentes=3,
                       umbral_falla=0.85):
    señal = generar_señal(n_pasos, n_canales, k_componentes)
    P = operador_pooling(n_pasos, ventana)

    señal_pooled = P @ señal
    señal_reconstruida = reconstruccion_optima_lineal(P, señal_pooled)

    error_reconstruccion = float(
        np.linalg.norm(señal - señal_reconstruida) / np.linalg.norm(señal)
    )

    U = subespacio_pca(señal, k_componentes)
    V = subespacio_pca(señal_reconstruida, k_componentes)
    metrica = alineacion_espectral(U, V)

    resultado = {
        "n_pasos": n_pasos,
        "n_canales": n_canales,
        "ventana_pooling": ventana,
        "factor_compresion": f"{n_pasos}:{n_pasos // ventana}",
        "error_relativo_reconstruccion": error_reconstruccion,
        "alineacion_espectral": metrica,
    }

    if metrica["SPS"] < umbral_falla:
        resultado["status"] = "ERROR"
        resultado["motivo_falla"] = (
            f"SPS={metrica['SPS']:.4f} por debajo del umbral {umbral_falla}. "
            "El pooling temporal destruyo demasiada estructura espectral; "
            "no se reporta como exito para evitar alucinar una solucion falsa."
        )
    else:
        resultado["status"] = "SUCCESS"

    return resultado


if __name__ == "__main__":
    import json
    resultado = ejecutar_pipeline()
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
