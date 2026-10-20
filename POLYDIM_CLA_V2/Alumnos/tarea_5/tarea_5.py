"""
Tarea 5 - Banco POLYDIM (Escuadron 1: Relajacion Topologica y Homotopia)
=========================================================================

Pregunta reconstruida (ver tarea_5.md para la nota de reconstruccion completa):

    Dados dos adaptadores tipo-LoRA de bajo rango, implementados como flujos
    de un grupo de Lie (exponencial matricial) en vez de la suma plana
    W0 + B1 A1^T + B2 A2^T de LoRA estandar, ¿bajo que condicion la
    composicion de ambos adaptadores es independiente del orden en que se
    aplican ("trayectorias que conmutan")? ¿Se cumple esa condicion para
    adaptadores de bajo rango genericos, o hace falta estructura adicional?

Modelo:
    G_i = B_i A_i^T / sqrt(r)         (generador de bajo rango, r << d)
    Camino A: W_A = exp(theta2 G2) @ exp(theta1 G1) @ W0
    Camino B: W_B = exp(theta1 G1) @ exp(theta2 G2) @ W0
    Divergencia de camino: ||W_A - W_B||_F / ||W0||_F

Prediccion teorica (formula de Baker-Campbell-Hausdorff, orden dominante):
    exp(t G2) exp(t G1) = exp(t G1) exp(t G2) + t^2 [G2, G1] + O(t^3)
    => si [G1, G2] != 0, la divergencia de camino escala ~ t^2 para t chico.
    => si [G1, G2] == 0 (generadores conmutan), la divergencia es ~0
       (solo error de punto flotante), para cualquier t.

Este script no asume que la propiedad de conmutatividad se cumpla: la
mide empiricamente en dos escenarios (generadores genericos vs. generadores
construidos para conmutar) y reporta el resultado, sin alucinar una
conclusion favorable a priori.
"""

import numpy as np
from scipy.linalg import expm


def make_low_rank_generator(d: int, r: int, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    """Genera un generador de bajo rango G = B A^T / sqrt(r), G en R^{d x d}, rank(G) <= r."""
    B = rng.standard_normal((d, r))
    A = rng.standard_normal((d, r))
    return scale * (B @ A.T) / np.sqrt(r)


def make_commuting_pair(d: int, r: int, rng: np.random.Generator, scale: float = 1.0):
    """
    Genera dos generadores de bajo rango que SI conmutan por construccion:
    comparten la misma base ortonormal de autovectores (dos matrices
    simetricas diagonalizadas en la misma base conmutan siempre).
    """
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    lam1 = np.zeros(d)
    lam1[:r] = scale * rng.standard_normal(r)
    lam2 = np.zeros(d)
    lam2[:r] = scale * rng.standard_normal(r)
    G1 = Q @ np.diag(lam1) @ Q.T
    G2 = Q @ np.diag(lam2) @ Q.T
    return G1, G2


def path_divergence(W0: np.ndarray, G1: np.ndarray, G2: np.ndarray, theta1: float, theta2: float) -> float:
    """
    Divergencia relativa de Frobenius entre las dos ordenes posibles de
    componer dos flujos exponenciales de bajo rango sobre W0.
    """
    path_A = expm(theta2 * G2) @ expm(theta1 * G1) @ W0
    path_B = expm(theta1 * G1) @ expm(theta2 * G2) @ W0
    diff = np.linalg.norm(path_A - path_B, ord="fro")
    norm = np.linalg.norm(W0, ord="fro") + 1e-12
    return diff / norm


def commutator_norm(G1: np.ndarray, G2: np.ndarray) -> float:
    """Norma de Frobenius del conmutador [G1,G2], normalizada por ||G1||*||G2||."""
    comm = G1 @ G2 - G2 @ G1
    denom = (np.linalg.norm(G1, ord="fro") * np.linalg.norm(G2, ord="fro")) + 1e-12
    return np.linalg.norm(comm, ord="fro") / denom


def run_experiment(d: int = 32, r: int = 4, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    W0 = rng.standard_normal((d, d)) / np.sqrt(d)

    print("=== Caso A: generadores de bajo rango genericos (sin forzar conmutatividad) ===")
    G1 = make_low_rank_generator(d, r, rng)
    G2 = make_low_rank_generator(d, r, rng)
    c = commutator_norm(G1, G2)
    print(f"Norma de conmutador normalizada ||[G1,G2]||/(||G1|| ||G2||) = {c:.6f}")
    for theta in (1e-3, 1e-2, 1e-1, 1.0):
        div = path_divergence(W0, G1, G2, theta, theta)
        print(f"  theta={theta:>7.4f}  divergencia relativa de camino = {div:.6e}")

    print()
    print("=== Caso B: generadores de bajo rango CONSTRUIDOS para conmutar (misma base propia) ===")
    G1c, G2c = make_commuting_pair(d, r, rng)
    c2 = commutator_norm(G1c, G2c)
    print(f"Norma de conmutador normalizada ||[G1,G2]||/(||G1|| ||G2||) = {c2:.6e}")
    for theta in (1e-3, 1e-2, 1e-1, 1.0):
        div = path_divergence(W0, G1c, G2c, theta, theta)
        print(f"  theta={theta:>7.4f}  divergencia relativa de camino = {div:.6e}")

    print()
    print("=== Verificacion de la prediccion BCH (exponente de escalado) ===")
    print("Prediccion: para G1,G2 genericos, divergencia ~ theta^2 cuando theta -> 0.")
    thetas = np.array([1e-3, 2e-3, 4e-3, 8e-3])
    divs = np.array([path_divergence(W0, G1, G2, t, t) for t in thetas])
    slope = np.polyfit(np.log(thetas), np.log(divs + 1e-300), 1)[0]
    print(f"Pendiente estimada (log-log) divergencia vs theta: {slope:.3f}  (prediccion teorica ~= 2.0)")

    print()
    print("=== Repeticion con distintas semillas (robustez del resultado del Caso A) ===")
    generic_commutators = []
    for s in range(1, 11):
        rr = np.random.default_rng(s)
        Wt = rr.standard_normal((d, d)) / np.sqrt(d)
        Ga = make_low_rank_generator(d, r, rr)
        Gb = make_low_rank_generator(d, r, rr)
        generic_commutators.append(commutator_norm(Ga, Gb))
    generic_commutators = np.array(generic_commutators)
    print(f"Conmutador normalizado sobre 10 semillas: media={generic_commutators.mean():.4f}, "
          f"min={generic_commutators.min():.4f}, max={generic_commutators.max():.4f}")
    frac_near_zero = np.mean(generic_commutators < 1e-6)
    print(f"Fraccion de semillas con conmutador ~0 (por azar): {frac_near_zero:.2%}")


if __name__ == "__main__":
    run_experiment()
