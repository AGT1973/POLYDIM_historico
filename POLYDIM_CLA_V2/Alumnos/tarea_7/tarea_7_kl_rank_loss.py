"""
Tarea 7 - Enjambre POLYDIM
Rol: Ingeniero de Teoria de la Informacion
Objetivo: Medir la perdida semantica (Divergencia KL) generada al reemplazar
          una multiplicacion vectorial densa (W x) por una aproximacion de
          bajo rango (constraint de la "Constitucion V2": preferir tensores
          de bajo rango sobre matrices densas), usando numpy como libreria
          matematica compatible.

Autor (worker_id): ariel.garcia.traba@gmail.com
Ticket: TAREA_7
"""

import numpy as np

RNG_SEED = 7  # reproducibilidad -> ligado al numero de tarea
EPS = 1e-12   # estabilidad numerica en logaritmos


def softmax(logits: np.ndarray) -> np.ndarray:
    """Convierte un vector real arbitrario en una distribucion de probabilidad.

    Este paso NO es opcional: la divergencia KL solo esta definida entre
    distribuciones de probabilidad (vectores no negativos que suman 1).
    Un vector de salida crudo de una multiplicacion vectorial (W x) NO es
    una distribucion de probabilidad por si mismo.
    """
    z = logits - logits.max(axis=-1, keepdims=True)  # estabilidad numerica
    ez = np.exp(z)
    return ez / ez.sum(axis=-1, keepdims=True)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """KL(P || Q) por fila. p, q: (batch, dim), ambas distribuciones validas."""
    p = np.clip(p, EPS, 1.0)
    q = np.clip(q, EPS, 1.0)
    return np.sum(p * np.log(p / q), axis=-1)


def low_rank_approx(W: np.ndarray, r: int):
    """Aproximacion optima de rango r de W via SVD truncada (Eckart-Young).

    Devuelve W_r y el residual de Frobenius teorico ||W - W_r||_F,
    que es la cota de error puramente algebraica (NO informacional).
    """
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    r = min(r, S.shape[0])
    W_r = (U[:, :r] * S[:r]) @ Vt[:r, :]
    frob_residual = np.sqrt(np.sum(S[r:] ** 2))
    return W_r, frob_residual


def experimento(d_in=64, d_out=32, batch=2000, scale=1.0, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((d_out, d_in)) / np.sqrt(d_in)  # init tipo Xavier
    X = rng.standard_normal((batch, d_in)) * scale

    y_exact = X @ W.T
    p_exact = softmax(y_exact)

    resultados = []
    max_rank = min(d_in, d_out)
    for r in range(1, max_rank + 1):
        W_r, frob_residual = low_rank_approx(W, r)
        y_approx = X @ W_r.T
        q_approx = softmax(y_approx)

        kl_por_muestra = kl_divergence(p_exact, q_approx)
        resultados.append({
            "rank": r,
            "kl_mean": float(np.mean(kl_por_muestra)),
            "kl_max": float(np.max(kl_por_muestra)),
            "frob_residual": float(frob_residual),
        })
    return resultados


def validar_invariantes(resultados):
    """Chequeos previos a la entrega (no se reporta nada sin verificar)."""
    errores = []
    kls = [r["kl_mean"] for r in resultados]
    if any(k < -1e-9 for k in kls):
        errores.append("KL negativa detectada (viola no-negatividad de Gibbs).")
    if kls[-1] > 1e-6:
        errores.append("KL en rango completo deberia ser ~0 y no lo es.")
    # monotonicidad NO estricta esperada globalmente (ver seccion trampa educativa),
    # pero el tramo final (ranks altos) si debe decrecer hacia 0.
    if kls[0] < kls[-1]:
        errores.append("KL en rango 1 no es mayor que en rango completo.")
    return errores


if __name__ == "__main__":
    print("=== Experimento A: escala de entrada normal (scale=1.0) ===")
    res_a = experimento(scale=1.0)
    errores_a = validar_invariantes(res_a)
    print("Errores de validacion:", errores_a if errores_a else "ninguno")
    for r in [1, 2, 4, 8, 16, 32]:
        row = res_a[r - 1]
        print(f"rank={row['rank']:>3} | KL_mean={row['kl_mean']:.6f} | "
              f"KL_max={row['kl_max']:.6f} | Frob_residual={row['frob_residual']:.4f}")

    print()
    print("=== Experimento B: escala de entrada alta (scale=5.0) -> saturacion softmax ===")
    res_b = experimento(scale=5.0)
    errores_b = validar_invariantes(res_b)
    print("Errores de validacion:", errores_b if errores_b else "ninguno")
    for r in [1, 2, 4, 8, 16, 32]:
        row = res_b[r - 1]
        print(f"rank={row['rank']:>3} | KL_mean={row['kl_mean']:.6f} | "
              f"KL_max={row['kl_max']:.6f} | Frob_residual={row['frob_residual']:.4f}")

    # Correlacion entre error algebraico (Frobenius) y perdida semantica (KL)
    frob_a = np.array([r["frob_residual"] for r in res_a])
    kl_a = np.array([r["kl_mean"] for r in res_a])
    corr_pearson = float(np.corrcoef(frob_a, kl_a)[0, 1])
    print()
    print(f"Correlacion de Pearson (Frobenius residual vs KL medio), escala 1.0: "
          f"{corr_pearson:.4f}")
