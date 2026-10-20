"""
Tarea 6 - Propagacion de Cotas de Intervalo (Interval Bound Propagation)
emulando dominios poliedricos para activaciones de una red neuronal.

Fundamento real (sin la retorica POLYDIM): esto es Interval Bound
Propagation (IBP), usado en verificacion de robustez de redes
neuronales (Gowal et al. 2018). El dominio "poliedrico" completo
(zonotopos / DeepPoly, Singh et al. 2019) es una cota mas ajustada
que el intervalo simple, pero el intervalo (caja/box domain) es el
caso base correcto y es lo que implementamos y verificamos aqui.
"""
import numpy as np

# ---------------------------------------------------------------
# 1) Aritmetica de intervalos para una capa AFIN: y = W x + b
# ---------------------------------------------------------------
def affine_bounds(W: np.ndarray, b: np.ndarray, l: np.ndarray, u: np.ndarray):
    """
    Dado x in [l, u] (elemento a elemento), calcula cotas exactas
    (no aproximadas) para y = Wx + b.

    Regla (interval arithmetic para forma lineal):
      W_pos = max(W, 0), W_neg = min(W, 0)
      y_l = W_pos @ l + W_neg @ u + b
      y_u = W_pos @ u + W_neg @ l + b

    Por que funciona: cada fila de W define una combinacion lineal
    sum_j w_j x_j. Si w_j >= 0, el minimo de w_j*x_j se alcanza en
    x_j = l_j; si w_j < 0, se alcanza en x_j = u_j. Sumando termino
    a termino se obtiene la cota EXACTA del hiperplano (no es una
    aproximacion, es el optimo real de una funcion lineal sobre una caja).
    """
    W_pos = np.maximum(W, 0.0)
    W_neg = np.minimum(W, 0.0)
    y_l = W_pos @ l + W_neg @ u + b
    y_u = W_pos @ u + W_neg @ l + b
    return y_l, y_u


# ---------------------------------------------------------------
# 2) Propagacion a traves de ReLU (no lineal, monotona)
# ---------------------------------------------------------------
def relu_bounds(l: np.ndarray, u: np.ndarray):
    """
    ReLU es monotona no decreciente, asi que la cota de intervalo
    tambien es EXACTA (no hay perdida de precision extra en este paso,
    a diferencia de dominios poliedricos donde ReLU si introduce
    relajacion cuando l<0<u).
    """
    return np.maximum(l, 0.0), np.maximum(u, 0.0)


# ---------------------------------------------------------------
# 3) Kernel estilo Triton (bosquejo ilustrativo)
# ---------------------------------------------------------------
TRITON_KERNEL_SKETCH = r'''
import triton
import triton.language as tl

@triton.jit
def affine_interval_kernel(
    W_ptr, b_ptr, l_ptr, u_ptr, out_l_ptr, out_u_ptr,
    M, N,                     # W es (M, N): M salidas, N entradas
    stride_wm, stride_wn,
    BLOCK_N: tl.constexpr,
):
    """Un programa por fila de salida (una neurona)."""
    row = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    mask = offs_n < N

    w = tl.load(W_ptr + row * stride_wm + offs_n * stride_wn, mask=mask, other=0.0)
    l = tl.load(l_ptr + offs_n, mask=mask, other=0.0)
    u = tl.load(u_ptr + offs_n, mask=mask, other=0.0)

    w_pos = tl.maximum(w, 0.0)
    w_neg = tl.minimum(w, 0.0)

    # reduccion: y_l = sum(w_pos*l + w_neg*u), y_u = sum(w_pos*u + w_neg*l)
    y_l = tl.sum(w_pos * l + w_neg * u, axis=0)
    y_u = tl.sum(w_pos * u + w_neg * l, axis=0)

    b = tl.load(b_ptr + row)
    tl.store(out_l_ptr + row, y_l + b)
    tl.store(out_u_ptr + row, y_u + b)
'''
# NOTA HONESTA: este kernel Triton compila solo con GPU CUDA/ROCm.
# Este sandbox no tiene GPU, por lo que NO fue ejecutado; es codigo
# de referencia correcto en su logica (misma formula que affine_bounds),
# no una demostracion "ejecutada". La version numpy de abajo SI se
# ejecuta y verifica numericamente.


# ---------------------------------------------------------------
# 4) Verificacion empirica: comparar cota exacta vs. muestreo aleatorio
# ---------------------------------------------------------------
def verify(seed=0, n_in=6, n_out=4, n_samples=200_000):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(size=(n_out, n_in))
    b1 = rng.normal(size=n_out)

    l = rng.uniform(-1.0, -0.1, size=n_in)
    u = l + rng.uniform(0.1, 2.0, size=n_in)

    # Cota analitica (la que reporta la tarea)
    y_l, y_u = affine_bounds(W1, b1, l, u)
    r_l, r_u = relu_bounds(y_l, y_u)

    # Cota empirica por fuerza bruta (muestreo denso de la caja)
    X = rng.uniform(l, u, size=(n_samples, n_in))
    Y = X @ W1.T + b1
    R = np.maximum(Y, 0.0)

    emp_y_l, emp_y_u = Y.min(axis=0), Y.max(axis=0)
    emp_r_l, emp_r_u = R.min(axis=0), R.max(axis=0)

    # La cota analitica debe SIEMPRE contener a la empirica (nunca al reves)
    ok_affine = np.all(y_l <= emp_y_l + 1e-9) and np.all(y_u >= emp_y_u - 1e-9)
    ok_relu = np.all(r_l <= emp_r_l + 1e-9) and np.all(r_u >= emp_r_u - 1e-9)

    print("== Verificacion Tarea 6: Interval Bound Propagation ==")
    print(f"Capa afin  -> cota analitica contiene a la empirica: {ok_affine}")
    print(f"  y_l analitico: {np.round(y_l,4)}")
    print(f"  y_l empirico : {np.round(emp_y_l,4)}  (debe ser >= analitico)")
    print(f"  y_u analitico: {np.round(y_u,4)}")
    print(f"  y_u empirico : {np.round(emp_y_u,4)}  (debe ser <= analitico)")
    print(f"ReLU       -> cota analitica contiene a la empirica: {ok_relu}")

    # "Tightness" (que tan ajustada esta la caja, metrica honesta,
    # analoga en espiritu al SPS que pide el documento pero sin
    # inventar magnitudes: es simplemente el ancho relativo del error).
    gap = (y_u - y_l) - (emp_y_u - emp_y_l)
    print(f"Holgura promedio de la cota (ancho_analitico - ancho_empirico): {gap.mean():.4f}")

    status = "SUCCESS" if (ok_affine and ok_relu) else "ERROR"
    return status, {
        "y_l": y_l.tolist(), "y_u": y_u.tolist(),
        "relu_l": r_l.tolist(), "relu_u": r_u.tolist(),
        "mean_slack": float(gap.mean()),
    }


if __name__ == "__main__":
    status, result = verify()
    print("\nStatus final:", status)
