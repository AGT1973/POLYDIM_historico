"""
Tarea 27 - POLYDIM
Simulacion de un entorno de agentes que modifican asincronamente (sin locks, estilo
Hogwild!) un tensor central compartido, midiendo:
  (a) FLOPs teoricos acumulados,
  (b) huella de memoria (proxy de "VRAM": bytes ocupados por el/los tensores),
  (c) tiempo de pared (wall-clock),
como funcion del numero de agentes N y comparando dos implementaciones:
  - "shared": tensor central UNICO en memoria compartida real (multiprocessing.shared_memory)
  - "naive": cada proceso recibe su PROPIA copia del tensor (fork/copy clasico)

Hipotesis a verificar empiricamente (ver tarea_27.md, seccion "Conclusion"):
  memoria(shared) = O(d)         (no crece con N)
  memoria(naive)  = O(N * d)     (crece linealmente con N)

Requiere: numpy (Python >= 3.8 por multiprocessing.shared_memory)
Correr:   python tarea_27.py
"""

import time
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory


def flops_per_update(sparsity_s: int) -> int:
    """Costo aritmetico de un paso de descenso disperso sobre 's' coordenadas activas.
    grad = local_read - ruido        -> s restas  ~ s flops
    theta[idx] -= eta * grad         -> s mult + s restas ~ 2s flops
    Total aproximado: 3s, redondeamos a 4s para incluir el costo del gather/scatter.
    """
    return 4 * sparsity_s


# ---------------------------------------------------------------------------
# Implementacion A: memoria COMPARTIDA real (un unico tensor central, O(d))
# ---------------------------------------------------------------------------
def worker_shared(shm_name, d, sparsity_s, n_updates, seed, eta, counter):
    rng = np.random.default_rng(seed)
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    theta = np.ndarray((d,), dtype=np.float64, buffer=existing_shm.buf)

    total_flops = 0
    for _ in range(n_updates):
        idx = rng.choice(d, size=sparsity_s, replace=False)
        # Lectura potencialmente "stale": otro agente puede escribir estas mismas
        # coordenadas entre esta lectura y la escritura de abajo (no hay lock -> race condition posible).
        local_read = theta[idx].copy()
        grad = local_read - rng.normal(0, 0.01, size=sparsity_s)
        theta[idx] -= eta * grad  # escritura sin lock (estilo Hogwild!)
        total_flops += flops_per_update(sparsity_s)

    with counter.get_lock():
        counter.value += total_flops
    existing_shm.close()


def run_shared_experiment(d, n_agents, sparsity_s, n_updates_per_agent, eta=0.01):
    shm = shared_memory.SharedMemory(create=True, size=np.zeros(d, dtype=np.float64).nbytes)
    theta = np.ndarray((d,), dtype=np.float64, buffer=shm.buf)
    theta[:] = np.random.default_rng(0).normal(0, 1, size=d)

    counter = mp.Value('q', 0)  # long long, evita overflow
    procs = []
    t0 = time.perf_counter()
    for i in range(n_agents):
        p = mp.Process(
            target=worker_shared,
            args=(shm.name, d, sparsity_s, n_updates_per_agent, i, eta, counter),
        )
        procs.append(p)
        p.start()
    for p in procs:
        p.join()
    wall_time = time.perf_counter() - t0

    result = {
        "wall_time_s": wall_time,
        "total_flops": counter.value,
        "shared_mem_bytes": shm.size,  # O(d), no depende de n_agents
        "theta_norm_final": float(np.linalg.norm(theta)),
        "theta_has_nan_or_inf": bool(not np.all(np.isfinite(theta))),
    }
    shm.close()
    shm.unlink()
    return result


# ---------------------------------------------------------------------------
# Implementacion B: copia NAIVE por proceso (O(N * d) memoria)
# ---------------------------------------------------------------------------
def worker_naive(base_theta_bytes, d, sparsity_s, n_updates, seed, eta, return_dict, key):
    rng = np.random.default_rng(seed)
    theta_local = np.frombuffer(bytearray(base_theta_bytes), dtype=np.float64).copy()
    for _ in range(n_updates):
        idx = rng.choice(d, size=sparsity_s, replace=False)
        grad = theta_local[idx] - rng.normal(0, 0.01, size=sparsity_s)
        theta_local[idx] -= eta * grad
    return_dict[key] = theta_local.nbytes  # este proceso "posee" d floats propios


def run_naive_experiment(d, n_agents, sparsity_s, n_updates_per_agent, eta=0.01):
    base_theta = np.random.default_rng(0).normal(0, 1, size=d).tobytes()
    manager = mp.Manager()
    return_dict = manager.dict()
    procs = []
    t0 = time.perf_counter()
    for i in range(n_agents):
        p = mp.Process(
            target=worker_naive,
            args=(base_theta, d, sparsity_s, n_updates_per_agent, i, eta, return_dict, i),
        )
        procs.append(p)
        p.start()
    for p in procs:
        p.join()
    wall_time = time.perf_counter() - t0

    total_mem_bytes = sum(return_dict.values())  # N copias de d floats -> O(N*d)
    return {"wall_time_s": wall_time, "total_mem_bytes": total_mem_bytes}


# ---------------------------------------------------------------------------
# Estudio de escalado asintotico en N (numero de agentes)
# ---------------------------------------------------------------------------
def scaling_study(d=2000, sparsity_s=20, n_updates_per_agent=200, agent_counts=(1, 2, 4, 8)):
    print(f"{'N':>4} | {'shared_mem(B)':>14} | {'naive_mem(B)':>14} | "
          f"{'shared_t(s)':>12} | {'naive_t(s)':>10} | {'flops':>10} | {'theta_ok':>8}")
    rows = []
    for n in agent_counts:
        r_shared = run_shared_experiment(d, n, sparsity_s, n_updates_per_agent)
        r_naive = run_naive_experiment(d, n, sparsity_s, n_updates_per_agent)
        ok = not r_shared["theta_has_nan_or_inf"]
        print(f"{n:>4} | {r_shared['shared_mem_bytes']:>14} | {r_naive['total_mem_bytes']:>14} | "
              f"{r_shared['wall_time_s']:>12.4f} | {r_naive['wall_time_s']:>10.4f} | "
              f"{r_shared['total_flops']:>10} | {str(ok):>8}")
        rows.append((n, r_shared, r_naive))
    return rows


if __name__ == "__main__":
    scaling_study()
