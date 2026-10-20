"""
REFERENCIA - NO EJECUTADO EN ESTE ENTORNO
==========================================
Este archivo NO fue corrido. El sandbox de esta sesion no tiene GPU ni el
paquete `triton` instalado (requiere CUDA). Se entrega como referencia de
diseno para mostrar como se llevaria la iteracion de potencias (Paso A de
spectral_fixpoint.py) a un kernel fusionado, tal como lo pide el enunciado
original ("kernel en Triton ultra-optimizado").

Si el alumno tiene acceso a una GPU NVIDIA con `pip install triton`, puede
correr esto para comparar contra la version NumPy (spectral_fixpoint.py) y
verificar que ambas dan el mismo sigma_max.

Idea de diseno (justificacion, no solo codigo):
- Un paso de iteracion de potencias necesita dos productos matriz-vector
  (W @ v y W.T @ (W @ v)) mas una normalizacion. Hacerlo naive en
  PyTorch/JAX lanza varios kernels separados (matmul, matmul, norm,
  division), cada uno con su propio round-trip a memoria global.
- Fusionar mat-vec + normalizacion en un unico kernel Triton evita
  materializar resultados intermedios en memoria global entre esos pasos,
  que es el cuello de botella real (memory-bound, no compute-bound) para
  matrices de tamano moderado como las de una capa GNN.
"""

import numpy as np  # solo para el placeholder de validacion, no para el kernel

try:
    import triton
    import triton.language as tl
    TRITON_DISPONIBLE = True
except ImportError:
    TRITON_DISPONIBLE = False


if TRITON_DISPONIBLE:

    @triton.jit
    def matvec_kernel(W_ptr, v_ptr, out_ptr, n_rows, n_cols,
                       stride_wr, stride_wc, BLOCK_SIZE: tl.constexpr):
        """
        Un bloque de threads calcula una fila de out = W @ v.
        (Kernel simplificado con fines didacticos; una version de
        produccion tilearia tambien sobre las columnas y usaria
        acumulacion en registros/shared memory.)
        """
        row_idx = tl.program_id(0)
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols

        w_row_ptr = W_ptr + row_idx * stride_wr + offs * stride_wc
        w_row = tl.load(w_row_ptr, mask=mask, other=0.0)
        v_vals = tl.load(v_ptr + offs, mask=mask, other=0.0)

        acc = tl.sum(w_row * v_vals, axis=0)
        tl.store(out_ptr + row_idx, acc)

    def power_iteration_triton(W: np.ndarray, n_iters: int = 50):
        """
        Orquesta n_iters pasos de iteracion de potencias llamando al
        kernel `matvec_kernel` dos veces por paso (W@v y W.T@(W@v)),
        con la normalizacion hecha en el host (o en un segundo kernel
        de reduccion, en una version mas optimizada).
        NOTA: no ejecutado; requiere GPU + `triton` instalado.
        """
        raise NotImplementedError(
            "Esqueleto de referencia. Requiere GPU NVIDIA + paquete "
            "'triton' para ejecutarse; no disponible en este sandbox."
        )

else:
    def power_iteration_triton(W: np.ndarray, n_iters: int = 50):
        raise ImportError(
            "El paquete 'triton' no esta instalado (o no hay GPU NVIDIA "
            "disponible). Este archivo es material de referencia; la "
            "comprobacion empirica real de la tarea se hizo en "
            "spectral_fixpoint.py con NumPy puro."
        )


if __name__ == "__main__":
    print("TRITON_DISPONIBLE =", TRITON_DISPONIBLE)
    print("Este script es de referencia y no forma parte de la verificacion")
    print("empirica de la tarea. Ver spectral_fixpoint.py para el codigo")
    print("funcional que si fue ejecutado y validado en este sandbox.")
