import jax
import jax.numpy as jnp
from typing import Tuple

@jax.jit
def woodbury_invert(A_inv: jnp.ndarray, U: jnp.ndarray, C: jnp.ndarray, V: jnp.ndarray) -> jnp.ndarray:
    """
    Identidad de Woodbury en JAX para inversión optimizada XLA O(NK^2).
    A_inv: Inversa de la matriz base (N, N)
    U: Matriz de actualización izquierda (N, K)
    C: Matriz de capacidad (K, K)
    V: Matriz de actualización derecha (K, N)
    """
    # (C^-1 + V A^-1 U)^-1
    C_inv = jnp.linalg.inv(C)
    inner = C_inv + V @ A_inv @ U
    inner_inv = jnp.linalg.inv(inner)
    
    # A^-1 - A^-1 U (inner_inv) V A^-1
    return A_inv - A_inv @ U @ inner_inv @ V @ A_inv

def apply_stiefel_rotor(state: jnp.ndarray, v1: jnp.ndarray, v2: jnp.ndarray) -> jnp.ndarray:
    """
    Aplica una rotación isométrica sobre el manifold de Stiefel usando 
    el Teorema de Cartan-Dieudonné (composición de dos reflexiones).
    """
    # Normalizar vectores
    v1 = v1 / jnp.linalg.norm(v1)
    v2 = v2 / jnp.linalg.norm(v2)
    
    # Reflexión sobre v1
    proj_v1 = jnp.dot(state, v1)
    state_ref1 = state - 2 * jnp.outer(proj_v1, v1)
    
    # Reflexión sobre v2
    proj_v2 = jnp.dot(state_ref1, v2)
    state_ref2 = state_ref1 - 2 * jnp.outer(proj_v2, v2)
    
    return state_ref2
