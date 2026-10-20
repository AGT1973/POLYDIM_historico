"""
Tarea 33 - Verificacion de bounds numericos (dominio de intervalos / poliedrico)
bajo actualizacion concurrente asincrona de un tensor central compartido.

Idea:
  1. Se calculan bounds [lo, hi] para una capa afin + ReLU via propagacion de
     intervalos (dominio "caja", caso particular de dominio poliedrico).
  2. Control: se aplica la actualizacion de forma secuencial/atomica -> 0 violaciones.
  3. Experimento: k agentes (threads) escriben coordenadas del tensor central de
     forma asincrona SIN lock -> se mide la tasa de violacion de bounds.
  4. Repeticion con lock, para confirmar que el lock elimina la violacion.

No se ejecuta como parte de la entrega formal (ver skill). Correr manualmente:
  pip install numpy
  python tarea_33_async_bounds.py
"""

import threading
import numpy as np

RNG = np.random.default_rng(7)


def interval_bounds_affine_relu(W: np.ndarray, b: np.ndarray, lo: np.ndarray, hi: np.ndarray):
    """Propagacion de intervalos (dominio caja) para y = ReLU(W x + b), x in [lo, hi]."""
    W_pos = np.maximum(W, 0.0)
    W_neg = np.minimum(W, 0.0)
    y_lo = W_pos @ lo + W_neg @ hi + b
    y_hi = W_pos @ hi + W_neg @ lo + b
    z_lo = np.maximum(0.0, y_lo)
    z_hi = np.maximum(0.0, y_hi)
    return z_lo, z_hi


def violation(x: np.ndarray, lo: np.ndarray, hi: np.ndarray):
    """Devuelve (count_violaciones, vector_epsilon) con epsilon_j = max(0, lo-x, x-hi)."""
    eps = np.maximum(0.0, np.maximum(lo - x, x - hi))
    count = int(np.sum(eps > 0))
    return count, eps


def sequential_update(W, b, x0):
    """Actualizacion atomica de referencia: lee x0 completo, calcula, escribe."""
    return np.maximum(0.0, W @ x0 + b)


def concurrent_update_no_lock(W, b, x0, n_agents: int):
    """
    Simula agentes que escriben coordenadas de un tensor central 'shared_x'
    de forma asincrona, leyendo el estado que sea que este disponible en ese
    instante (posiblemente una mezcla inconsistente de x0 viejo y valores ya
    actualizados por otros agentes), sin ninguna seccion critica.
    """
    n = x0.shape[0]
    shared_x = x0.copy()  # estado central compartido, mutado in-place sin proteccion
    result = np.zeros(n)

    def agent_job(indices):
        for j in indices:
            # lectura no atomica del estado central completo (puede estar a medio actualizar)
            local_read = shared_x.copy()
            val = max(0.0, float(W[j, :] @ local_read + b[j]))
            result[j] = val
            # escritura no atomica de vuelta al estado compartido
            shared_x[j] = val

    idx_splits = np.array_split(np.arange(n), n_agents)
    threads = [threading.Thread(target=agent_job, args=(idx,)) for idx in idx_splits]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return result


def concurrent_update_with_lock(W, b, x0, n_agents: int):
    n = x0.shape[0]
    shared_x = x0.copy()
    result = np.zeros(n)
    lock = threading.Lock()

    def agent_job(indices):
        for j in indices:
            with lock:
                local_read = shared_x.copy()
                val = max(0.0, float(W[j, :] @ local_read + b[j]))
                result[j] = val
                shared_x[j] = val

    idx_splits = np.array_split(np.arange(n), n_agents)
    threads = [threading.Thread(target=agent_job, args=(idx,)) for idx in idx_splits]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return result


def run_experiment(n=200, n_agents=8, n_trials=20):
    W = RNG.normal(0, 1.0, size=(n, n))
    b = RNG.normal(0, 0.1, size=n)
    x0 = RNG.uniform(-1, 1, size=n)

    # bounds "de referencia": intervalo trivial alrededor de x0 mismo
    # (representa la incertidumbre de entrada tolerada por el dominio caja)
    lo = x0 - 1e-6
    hi = x0 + 1e-6
    z_lo, z_hi = interval_bounds_affine_relu(W, b, lo, hi)

    print("=== Control: actualizacion secuencial/atomica ===")
    x_seq = sequential_update(W, b, x0)
    v_count, eps = violation(x_seq, z_lo, z_hi)
    print(f"violaciones = {v_count} / {n}  (esperado: 0)")
    print(f"||eps||_2 = {np.linalg.norm(eps):.3e}  ||eps||_inf = {np.max(eps):.3e}")

    print()
    print(f"=== Experimento: {n_agents} agentes concurrentes SIN lock, {n_trials} corridas ===")
    rates = []
    for trial in range(n_trials):
        x_async = concurrent_update_no_lock(W, b, x0, n_agents)
        v_count, eps = violation(x_async, z_lo, z_hi)
        rates.append(v_count / n)
    rates = np.array(rates)
    print(f"tasa de violacion promedio = {rates.mean():.4f} (min={rates.min():.4f}, max={rates.max():.4f})")
    print("Nota: esta tasa depende del scheduler del sistema y puede variar entre corridas y entre maquinas.")

    print()
    print(f"=== Control: {n_agents} agentes concurrentes CON lock, {n_trials} corridas ===")
    rates_lock = []
    for trial in range(n_trials):
        x_locked = concurrent_update_with_lock(W, b, x0, n_agents)
        v_count, eps = violation(x_locked, z_lo, z_hi)
        rates_lock.append(v_count / n)
    rates_lock = np.array(rates_lock)
    print(f"tasa de violacion promedio = {rates_lock.mean():.4f} (esperado: 0.0000)")


def _sanity_check():
    """Chequeo minimo de sintaxis/ejecucion sobre una instancia chica,
    NO es la comprobacion final (eso lo hace el alumno)."""
    n = 10
    W = RNG.normal(0, 1.0, size=(n, n))
    b = RNG.normal(0, 0.1, size=n)
    x0 = RNG.uniform(-1, 1, size=n)
    lo, hi = x0 - 0.01, x0 + 0.01
    z_lo, z_hi = interval_bounds_affine_relu(W, b, lo, hi)
    assert z_lo.shape == (n,) and z_hi.shape == (n,)
    x_seq = sequential_update(W, b, x0)
    assert x_seq.shape == (n,)


if __name__ == "__main__":
    _sanity_check()
    run_experiment()
