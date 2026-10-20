import jax
import jax.numpy as jnp
from jax import vmap, jit, grad
import numpy as np
from scipy.linalg import svd
import torch

# Función para realizar la descomposición SVD y truncar a rango k
def truncated_svd(A, k):
    U, S, Vt = svd(A, full_matrices=False)
    return U[:, :k] @ jnp.diag(S[:k]) @ Vt[:k, :]

# Función para componer tensores de bajo rango con truncamiento SVD
def compose_low_rank_tensors(A, B, k):
    # Realizar la composición de tensores
    C = jnp.einsum('ij,jk->ik', A, B)
    # Aplicar truncamiento SVD
    return truncated_svd(C, k)

# Función para calcular la norma Frobenius de una matriz
def frobenius_norm(A):
    return jnp.sqrt(jnp.sum(A**2))

# Función para calcular el gradiente de la norma Frobenius
def norm_grad(A):
    return grad(frobenius_norm)(A)

# Parámetros
n = 100  # Dimensiones de las matrices
k = 10   # Rango truncado

# Generar matrices aleatorias
A = jnp.array(np.random.randn(n, n))
B = jnp.array(np.random.randn(n, n))

# Componer tensores de bajo rango con truncamiento SVD
C = compose_low_rank_tensors(A, B, k)

# Calcular la norma Frobenius de la matriz resultante
norm_C = frobenius_norm(C)

# Calcular el gradiente de la norma Frobenius
grad_C = norm_grad(C)

# Imprimir resultados
print(f"Norma Frobenius de C: {norm_C}")
print(f"Gradiente de la norma Frobenius de C:\n{grad_C}")

# Validación empírica: Comprobar que la norma Frobenius y el gradiente son estables
def validate_stability(A, B, k, num_iterations=100):
    norms = []
    for _ in range(num_iterations):
        C = compose_low_rank_tensors(A, B, k)
        norms.append(frobenius_norm(C))
    return norms

norms = validate_stability(A, B, k)
print(f"Normas Frobenius a lo largo de {num_iterations} iteraciones: {norms}")