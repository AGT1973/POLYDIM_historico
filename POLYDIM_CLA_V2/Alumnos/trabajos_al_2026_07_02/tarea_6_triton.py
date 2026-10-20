"""
Test 6 - Enjambre POLYDIM
Rol: Matematico numerico
Tarea: Kernel Triton ultra-optimizado para calcular los intervalos (bounds)
numericos de las activaciones, emulando dominios poliedricos (caja / interval domain
en el sentido de Abstract Interpretation, Cousot & Cousot 1977).

NOTA HONESTA (chequeo de entrega):
Este sandbox no tiene GPU/CUDA, por lo tanto el kernel Triton no puede
ejecutarse aqui. Lo que SI se verifico y se ejecuto es la matematica
subyacente (ver ibp_afin_referencia), que es la parte que puede fallar
silenciosamente. El kernel Triton es una traduccion directa, kernel a kernel,
de esa matematica ya validada.
"""

import triton
import triton.language as tl
import torch


@triton.jit
def ibp_affine_relu_kernel(
    # Punteros de entrada
    W_ptr, b_ptr, l_ptr, u_ptr,
    # Punteros de salida (cotas ya con ReLU aplicado)
    out_l_ptr, out_u_ptr,
    # Dimensiones
    IN_FEATURES: tl.constexpr,
    OUT_FEATURES: tl.constexpr,
    BLOCK_IN: tl.constexpr,
):
    """
    Cada 'programa' (bloque de Triton) calcula UNA fila de salida,
    es decir, las cotas [y_l, y_u] de UNA neurona, para una sola
    muestra del batch. El eje de paralelismo es la dimension de salida.

    Formulas:
        y_l = W_pos @ l + W_neg @ u + b
        y_u = W_pos @ u + W_neg @ l + b
    donde W_pos = max(W, 0), W_neg = min(W, 0).
    """
    pid = tl.program_id(axis=0)  # indice de la neurona de salida (fila de W)

    offs = tl.arange(0, BLOCK_IN)
    mask = offs < IN_FEATURES

    # W tiene forma (OUT_FEATURES, IN_FEATURES), fila pid
    w_row = tl.load(W_ptr + pid * IN_FEATURES + offs, mask=mask, other=0.0)
    l_vec = tl.load(l_ptr + offs, mask=mask, other=0.0)
    u_vec = tl.load(u_ptr + offs, mask=mask, other=0.0)
    bias = tl.load(b_ptr + pid)

    # Separacion de signos: el paso critico que evita la 'trampa educativa'
    w_pos = tl.maximum(w_row, 0.0)
    w_neg = tl.minimum(w_row, 0.0)

    y_l = tl.sum(w_pos * l_vec + w_neg * u_vec, axis=0) + bias
    y_u = tl.sum(w_pos * u_vec + w_neg * l_vec, axis=0) + bias

    # ReLU es monotona creciente -> se puede aplicar directo a cada cota
    r_l = tl.maximum(y_l, 0.0)
    r_u = tl.maximum(y_u, 0.0)

    tl.store(out_l_ptr + pid, r_l)
    tl.store(out_u_ptr + pid, r_u)


def ibp_affine_relu(W: torch.Tensor, b: torch.Tensor,
                     l: torch.Tensor, u: torch.Tensor):
    """Wrapper en Python que lanza el kernel. Requiere GPU CUDA."""
    assert W.is_cuda and b.is_cuda and l.is_cuda and u.is_cuda
    out_features, in_features = W.shape
    out_l = torch.empty(out_features, device=W.device, dtype=W.dtype)
    out_u = torch.empty(out_features, device=W.device, dtype=W.dtype)

    BLOCK_IN = triton.next_power_of_2(in_features)
    grid = (out_features,)

    ibp_affine_relu_kernel[grid](
        W, b, l, u,
        out_l, out_u,
        IN_FEATURES=in_features,
        OUT_FEATURES=out_features,
        BLOCK_IN=BLOCK_IN,
    )
    return out_l, out_u


import numpy as np

def ibp_afin_referencia(W, b, l, u):
    """Referencia matematica pura (numpy) de Interval Bound Propagation
    para una capa afin y = W x + b, dado x in [l, u] elemento a elemento."""
    W_pos = np.maximum(W, 0)
    W_neg = np.minimum(W, 0)
    y_l = W_pos @ l + W_neg @ u + b
    y_u = W_pos @ u + W_neg @ l + b
    return y_l, y_u

def relu_bounds(y_l, y_u):
    return np.maximum(y_l, 0), np.maximum(y_u, 0)

def trampa_ingenua(W, b, l, u):
    """La version 'ingenua' que un alumno haria al primer vistazo."""
    y_l_mal = W @ l + b
    y_u_mal = W @ u + b
    return y_l_mal, y_u_mal

if __name__ == "__main__":
    # Caso de prueba: 3 entradas, 2 salidas, con pesos MIXTOS
    np.random.seed(0)
    W = np.array([[ 2.0, -3.0,  1.0],
                  [-1.0,  0.5,  4.0]])
    b = np.array([0.1, -0.2])
    l = np.array([-1.0, 0.0, 2.0])   # cota inferior de la entrada
    u = np.array([ 1.0, 2.0, 3.0])   # cota superior de la entrada

    y_l, y_u = ibp_afin_referencia(W, b, l, u)
    y_l_mal, y_u_mal = trampa_ingenua(W, b, l, u)

    print("=== Cotas CORRECTAS (separando signos) ===")
    print("y_l =", y_l)
    print("y_u =", y_u)

    print("\n=== Cotas de la version INGENUA (trampa educativa) ===")
    print("y_l_mal =", y_l_mal)
    print("y_u_mal =", y_u_mal)

    # Verificacion empirica por muestreo aleatorio
    n_muestras = 200000
    X = np.random.uniform(l, u, size=(n_muestras, 3))
    Y = X @ W.T + b

    viol_correcta = np.any((Y < y_l - 1e-9) | (Y > y_u + 1e-9))
    viol_ingenua  = np.any((Y < y_l_mal - 1e-9) | (Y > y_u_mal + 1e-9))

    print("\n¿La cota CORRECTA es violada por algun muestreo?", viol_correcta)
    print("¿La cota INGENUA es violada por algun muestreo?   ", viol_ingenua)

    r_l, r_u = relu_bounds(y_l, y_u)
    print("\nCotas tras ReLU:")
    print("r_l =", r_l)
    print("r_u =", r_u)
