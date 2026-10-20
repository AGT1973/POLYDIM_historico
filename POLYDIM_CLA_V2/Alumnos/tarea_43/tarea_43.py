"""
Tarea 43 - Banco POLYDIM
Recurrencia lineal generada por bivectores puros del algebra de Clifford Cl(n,0)
(generadores antisimetricos B en so(n), R = exp(B) en SO(n)).

Verifica:
  1) que R es ortogonal a precision de maquina (sin penalizacion espectral explicita)
  2) que la recurrencia pura h_{t+1} = R h_t NO converge a un punto fijo (norma conservada)
  3) que la recurrencia amortiguada h_{t+1} = (1-alpha) h_t + alpha R h_t SI converge a 0

Ejecutar: python tarea_43.py
Requisitos: pip install jax jaxlib
"""

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm


def build_bivector_generator(key, n):
    """Construye un generador antisimetrico B (bivector puro de Cl(n,0)) de forma aleatoria."""
    A = jax.random.normal(key, (n, n))
    B = A - A.T  # proyeccion a la parte antisimetrica -> subalgebra bivectorial pura
    return B


def rotor_from_bivector(B):
    """R = exp(B). Si B es antisimetrico, R es ortogonal (SO(n)) por construccion."""
    return expm(B)


def orthogonality_error(R):
    n = R.shape[0]
    return jnp.linalg.norm(R.T @ R - jnp.eye(n))


def pure_recurrence_step(h, R):
    return R @ h


def damped_recurrence_step(h, R, alpha):
    return (1 - alpha) * h + alpha * (R @ h)


def run_while_loop(h0, R, alpha, max_iter=500, tol=1e-8):
    """
    Itera la recurrencia con jax.lax.while_loop hasta que ||h_{t+1}-h_t|| < tol
    o se alcance max_iter. alpha=1.0 -> recurrencia pura (rotor sin amortiguar).
    Devuelve: (h_final, num_iter, ultima_diferencia, historial_de_normas)
    """

    def step(h):
        return (1 - alpha) * h + alpha * (R @ h)

    def cond_fn(state):
        h, h_prev, i, diff = state
        return jnp.logical_and(i < max_iter, diff > tol)

    def body_fn(state):
        h, h_prev, i, _ = state
        h_new = step(h)
        diff = jnp.linalg.norm(h_new - h)
        return (h_new, h, i + 1, diff)

    init_state = (h0, h0, 0, jnp.inf)
    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
    h_final, _, num_iter, last_diff = final_state
    return h_final, num_iter, last_diff


def norm_trajectory(h0, R, alpha, steps=30):
    """Historial de ||h_t|| para inspeccionar convergencia (fuera del while_loop, solo para reportar)."""
    norms = [float(jnp.linalg.norm(h0))]
    h = h0
    for _ in range(steps):
        h = (1 - alpha) * h + alpha * (R @ h)
        norms.append(float(jnp.linalg.norm(h)))
    return norms


def main():
    n = 6
    key = jax.random.PRNGKey(0)
    key, subkey1, subkey2 = jax.random.split(key, 3)

    B = build_bivector_generator(subkey1, n)
    R = rotor_from_bivector(B)

    # 1) Verificacion de ortogonalidad exacta (sin penalizacion espectral explicita)
    err = orthogonality_error(R)
    svals = jnp.linalg.svd(R, compute_uv=False)
    print(f"[Ortogonalidad] ||R^T R - I|| = {err:.3e}")
    print(f"[Valores singulares de R] {svals}")
    print(f"[det(R)] {jnp.linalg.det(R):.6f}  (esperado ~ 1.0)")
    print()

    h0 = jax.random.normal(subkey2, (n,))

    # 2) Recurrencia pura (alpha = 1.0): NO deberia converger a un punto fijo
    h_final_pure, iters_pure, diff_pure = run_while_loop(h0, R, alpha=1.0, max_iter=500, tol=1e-8)
    norms_pure = norm_trajectory(h0, R, alpha=1.0, steps=30)
    print("[Recurrencia pura h_{t+1} = R h_t]")
    print(f"  iteraciones hasta max_iter (no convergio): {iters_pure}")
    print(f"  ||h_final - h_prev|| final: {diff_pure:.3e}  (se espera que NO sea pequeno)")
    print(f"  ||h_t|| en pasos 0..10: {[round(v, 5) for v in norms_pure[:11]]}")
    print("  -> conclusion: la norma se conserva (no decae), consistente con R ortogonal.")
    print()

    # 3) Recurrencia amortiguada (alpha = 0.3): SI deberia converger a 0
    alpha = 0.3
    h_final_damped, iters_damped, diff_damped = run_while_loop(h0, R, alpha=alpha, max_iter=2000, tol=1e-8)
    norms_damped = norm_trajectory(h0, R, alpha=alpha, steps=30)
    print(f"[Recurrencia amortiguada h_(t+1) = (1-a)h_t + a R h_t, alpha={alpha}]")
    print(f"  iteraciones hasta convergencia: {iters_damped}")
    print(f"  ||h_final - h_prev|| final: {diff_damped:.3e}  (se espera muy pequeno)")
    print(f"  ||h_t|| en pasos 0..10: {[round(v, 6) for v in norms_damped[:11]]}")
    print(f"  ||h_final||: {float(jnp.linalg.norm(h_final_damped)):.3e}  (se espera ~ 0)")


if __name__ == "__main__":
    main()
