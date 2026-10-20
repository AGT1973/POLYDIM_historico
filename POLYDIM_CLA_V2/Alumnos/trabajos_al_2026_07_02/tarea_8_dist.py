import numpy as np

N_SUB = 5000
rng = np.random.default_rng(42)
v1 = rng.uniform(-1, 1, N_SUB).astype(np.float32)
v2 = rng.uniform(-1, 1, N_SUB).astype(np.float32)

# Método A: NumPy vectorizado (BLAS interno, orden de reducción "optimizado")
def dot_numpy(a, b):
    return np.dot(a, b)

# Método B: bucle secuencial en orden fijo (simula un nodo "naive")
def dot_sequential(a, b):
    total = np.float32(0.0)
    for x, y in zip(a, b):
        total += x * y
    return total

# Método C: reducción por pares (divide y vencerás, otra librería podría usar esto)
def dot_pairwise(a, b):
    prod = a * b
    def pairwise_sum(arr):
        if len(arr) <= 1:
            return arr[0] if len(arr) == 1 else np.float32(0.0)
        mid = len(arr) // 2
        return pairwise_sum(arr[:mid]) + pairwise_sum(arr[mid:])
    return pairwise_sum(prod)

if __name__ == "__main__":
    r_numpy = dot_numpy(v1, v2)
    r_seq   = dot_sequential(v1, v2)
    r_pair  = dot_pairwise(v1, v2)

    print("Método\tResultado (float32)")
    print(f"NumPy (BLAS vectorizado)\t{r_numpy}")
    print(f"Bucle secuencial (orden fijo)\t{r_seq}")
    print(f"Reducción por pares\t{r_pair}")
    print(f"Diferencia NumPy vs secuencial: {abs(r_numpy - r_seq):.2e}")
    print(f"Diferencia NumPy vs pares: {abs(r_numpy - r_pair):.2e}")
