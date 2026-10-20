"""
POLYDIM - Tarea 54 (Test 54)
Rol: Tester de robustez de IA

Enunciado (verbatim de la bateria de 200 tareas):
"Actua como tester de robustez de IA. Disena una operacion encadenada que
demuestre a nivel de volcado intermedio de grafo que un pooling promedio en
tiempo seguido de una reconstruccion optima lineal."

Nota de interpretacion: la frase, tal como esta en el banco de tareas, queda
sin verbo principal despues de "que" (es una plantilla combinatoria: rol +
accion + fragmento-concepto, y el fragmento "un pooling promedio en tiempo
seguido de una reconstruccion optima lineal" no es una oracion completa).
Se interpreta razonablemente como:

    Disenar la operacion encadenada  R o P  (pooling promedio temporal P,
    seguido de una reconstruccion lineal R optima en sentido de minimos
    cuadrados), exponer su grafo computacional intermedio (jaxpr), y
    someterla a un analisis de robustez: error de reconstruccion, numero
    de condicion, y amplificacion de ruido bajo distintas configuraciones
    de pooling.

Este archivo NO usa nada de "topos" ni "criba de cobertura" (esta tarea del
banco no lo pide); es un ejercicio de algebra lineal + trazado de grafo.
"""

from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 1. Modelo matematico como operadores lineales explicitos
# ---------------------------------------------------------------------------

def build_pooling_matrix(T: int, T_pool: int, overlap: int = 0) -> np.ndarray:
    """
    Construye P in R^{T_pool x T}, el operador lineal de "average pooling"
    sobre el eje temporal.

    overlap = 0  -> ventanas contiguas, no solapadas (caso estandar).
    overlap > 0  -> ventanas solapadas en `overlap` pasos (hace a P mas
                    rank-deficiente / peor condicionada: util para el
                    analisis de robustez).

    Cada fila de P es un promedio (pesos que suman 1) sobre una ventana de
    ancho w = T / T_pool + overlap (recortada a los bordes del arreglo).
    """
    if T_pool <= 0 or T_pool > T:
        raise ValueError("T_pool debe cumplir 0 < T_pool <= T")

    base_w = T / T_pool
    P = np.zeros((T_pool, T), dtype=np.float64)
    for i in range(T_pool):
        start = int(np.floor(i * base_w)) - overlap
        end = int(np.ceil((i + 1) * base_w)) + overlap
        start = max(start, 0)
        end = min(end, T)
        if end <= start:
            end = start + 1
        P[i, start:end] = 1.0 / (end - start)
    return P


def optimal_linear_reconstruction(P: np.ndarray) -> np.ndarray:
    """
    R = pseudo-inversa de Moore-Penrose de P.

    Justificacion (ver reporte, seccion "Razonamiento"): dado Y = P X y sin
    informacion previa adicional sobre X, la solucion de norma minima al
    sistema (en general subdeterminado) P X_hat = Y es X_hat = P^+ Y, y
    P^+ es tambien el reconstructor lineal que minimiza el error cuadratico
    medio de reconstruccion sobre el subespacio alcanzable por P (es la
    mejor inversa lineal por minimos cuadrados de un operador rectangular).
    """
    return np.linalg.pinv(P)


# ---------------------------------------------------------------------------
# 2. Operacion encadenada en JAX (para exponer el grafo intermedio - jaxpr)
# ---------------------------------------------------------------------------

def make_chain(P: jnp.ndarray, R: jnp.ndarray):
    """Devuelve una funcion pura X -> R @ (P @ X), trazable por JAX."""

    def chain(X):
        Y = P @ X          # (1) pooling promedio temporal (proyeccion lineal)
        X_hat = R @ Y       # (2) reconstruccion lineal optima (pseudo-inversa)
        return X_hat, Y

    return jax.jit(chain)


def dump_intermediate_graph(chain_fn, X_example: jnp.ndarray) -> str:
    """
    Vuelca el grafo computacional intermedio de la operacion encadenada
    usando jax.make_jaxpr (equivalente, a nivel de IR, a un 'volcado
    intermedio de grafo' de la composicion pooling -> reconstruccion).
    """
    jaxpr = jax.make_jaxpr(chain_fn)(X_example)
    return str(jaxpr)


# ---------------------------------------------------------------------------
# 3. Bateria de robustez (rol: tester de robustez de IA)
# ---------------------------------------------------------------------------

@dataclass
class RobustnessResult:
    T: int
    T_pool: int
    overlap: int
    cond_P: float
    cond_R: float
    recon_error_clean: float
    recon_error_noisy: dict = field(default_factory=dict)
    amplification_factor: float = 0.0


def run_robustness_battery(
    T: int = 64,
    pool_sizes=(32, 16, 8),
    overlaps=(0, 2, 6),
    noise_stds=(1e-4, 1e-2, 1e-1),
    seed: int = 0,
) -> list[RobustnessResult]:
    rng = np.random.default_rng(seed)
    D = 5  # canales / features por paso temporal
    X = rng.standard_normal((T, D))

    results: list[RobustnessResult] = []
    for T_pool in pool_sizes:
        for overlap in overlaps:
            P = build_pooling_matrix(T, T_pool, overlap=overlap)
            R = optimal_linear_reconstruction(P)

            # Condicion numerica de cada operador (sigma_max / sigma_min de
            # los valores singulares no nulos). cond(P) alto == ventanas
            # de pooling casi linealmente dependientes == mala noticia para
            # la reconstruccion.
            sv_P = np.linalg.svd(P, compute_uv=False)
            sv_P_nonzero = sv_P[sv_P > 1e-12]
            cond_P = float(sv_P_nonzero.max() / sv_P_nonzero.min())

            sv_R = np.linalg.svd(R, compute_uv=False)
            sv_R_nonzero = sv_R[sv_R > 1e-12]
            cond_R = float(sv_R_nonzero.max() / sv_R_nonzero.min())

            chain = make_chain(jnp.array(P), jnp.array(R))
            X_hat, Y = chain(jnp.array(X))
            X_hat = np.array(X_hat)

            err_clean = float(np.linalg.norm(X - X_hat) / np.linalg.norm(X))

            err_noisy = {}
            for std in noise_stds:
                noise = rng.normal(scale=std, size=Y.shape)
                Y_noisy = np.array(Y) + noise
                X_hat_noisy = R @ Y_noisy
                err_noisy[std] = float(
                    np.linalg.norm(X - X_hat_noisy) / np.linalg.norm(X)
                )

            # Factor de amplificacion empirico: cuanto ruido relativo en Y
            # se traduce en ruido relativo en X_hat, comparado contra el
            # ruido clean-baseline. Deberia acercarse a cond(R) en el peor
            # caso (definicion estandar de sensibilidad de sistemas lineales
            # mal condicionados, Golub & Van Loan).
            std_ref = noise_stds[-1]
            amp = (err_noisy[std_ref] - err_clean) / (
                std_ref * np.linalg.norm(Y) / np.linalg.norm(X) + 1e-12
            )

            results.append(
                RobustnessResult(
                    T=T,
                    T_pool=T_pool,
                    overlap=overlap,
                    cond_P=cond_P,
                    cond_R=cond_R,
                    recon_error_clean=err_clean,
                    recon_error_noisy=err_noisy,
                    amplification_factor=float(amp),
                )
            )
    return results


# ---------------------------------------------------------------------------
# 4. Demostracion ejecutable
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    T, T_pool, overlap = 16, 8, 0
    P_np = build_pooling_matrix(T, T_pool, overlap=overlap)
    R_np = optimal_linear_reconstruction(P_np)

    chain = make_chain(jnp.array(P_np), jnp.array(R_np))
    X_example = jnp.array(np.random.default_rng(1).standard_normal((T, 3)))

    print("=" * 70)
    print("VOLCADO INTERMEDIO DE GRAFO (jaxpr) de la operacion encadenada")
    print("=" * 70)
    print(dump_intermediate_graph(chain, X_example))

    print("\n" + "=" * 70)
    print("BATERIA DE ROBUSTEZ")
    print("=" * 70)
    for r in run_robustness_battery():
        print(
            f"T_pool={r.T_pool:>3} overlap={r.overlap:>2} | "
            f"cond(P)={r.cond_P:8.3f} cond(R)={r.cond_R:8.3f} | "
            f"err_clean={r.recon_error_clean:7.4f} | "
            f"err_noisy(std=0.1)={r.recon_error_noisy[0.1]:7.4f} | "
            f"amp~{r.amplification_factor:6.2f}"
        )
