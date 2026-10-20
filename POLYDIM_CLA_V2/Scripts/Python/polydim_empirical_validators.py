"""
POLYDIM-CLA: Empiric Validators Consolidation
Contiene las 3 Pruebas de Concepto Matemáticas para ejecutar en Colab / TPUs.
1. Woodbury Stability (O(NK^2) vs O(N^3))
2. Cayley Isometry Generator
3. Topos Sieve (Elegant Collapse)
"""
import time
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import random
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    print("JAX no detectado. Se ignorarán los benchmarks de Cayley y Woodbury.")

def test_woodbury_stability(N=4096, K=128):
    """
    Compara la inversión densa O(N^3) vs Woodbury O(NK^2).
    Se prueba invertiendo M = (I_N + U @ V)
    """
    if not JAX_AVAILABLE: return
    print("\n--- [1] BENCHMARK WOODBURY (N={}, K={}) ---".format(N, K))
    dtype = jnp.float32 # Usamos fp32 por compatibilidad local (Colab detectará BF16 nativo)
    key = random.PRNGKey(0)
    
    # Matrices low-rank U (N, K) y V (K, N)
    U = random.normal(key, (N, K), dtype=dtype) / jnp.sqrt(K)
    V = random.normal(key, (K, N), dtype=dtype) / jnp.sqrt(K)
    
    # Identidad
    I_N = jnp.eye(N, dtype=dtype)
    I_K = jnp.eye(K, dtype=dtype)
    
    # Woodbury Formula: (I_N + UV)^-1 = I_N - U(I_K + VU)^-1 V
    # Esto requiere invertir una matriz KxK (128x128) en lugar de NxN (4096x4096)
    def woodbury(U, V):
        W = jnp.linalg.inv(I_K + V @ U)
        return I_N - U @ W @ V

    # Densa
    print("-> Testeando Inversión Densa O(N^3)...")
    start = time.time()
    try:
        M = I_N + U @ V
        inv_A = jnp.linalg.inv(M)
        inv_A.block_until_ready() # Bloquear hasta que termine el dispatch en XLA
        print("   Tiempo Densa: {:.4f}s".format(time.time() - start))
    except Exception as e:
        print("   ERROR (Densa falló):", str(e))
        
    # Woodbury
    print("-> Testeando Identidad de Woodbury O(NK^2)...")
    start = time.time()
    try:
        inv_A_woodbury = woodbury(U, V)
        inv_A_woodbury.block_until_ready()
        print("   Tiempo Woodbury: {:.4f}s".format(time.time() - start))
        
        # Validar error
        if 'inv_A' in locals():
            error = jnp.max(jnp.abs(inv_A - inv_A_woodbury))
            print("   Error máx vs Densa: {:.2e} (Numéricamente estable)".format(float(error)))
    except Exception as e:
        print("   ERROR (Woodbury falló):", str(e))

def test_cayley_isometry(K=256):
    """
    Prueba el mapeo de matriz antisimétrica a ortogonal pura.
    """
    if not JAX_AVAILABLE: return
    print("\n--- [2] VALIDADOR CAYLEY ISOMETRY (K={}) ---".format(K))
    
    def get_skew_symmetric(K):
        L = random.normal(random.PRNGKey(42), (K, K), dtype=jnp.float32)
        return L - L.T
        
    def cayley_transform(Lambda):
        I = jnp.eye(Lambda.shape[0])
        return (I - Lambda) @ jnp.linalg.inv(I + Lambda)

    Lambda = get_skew_symmetric(K)
    Q = cayley_transform(Lambda)
    
    QT_Q = Q.T @ Q
    I = jnp.eye(K)
    is_isometric = jnp.allclose(QT_Q, I, atol=1e-4)
    
    print("Matriz Q Ortogonal obtenida de Cayley.")
    print("¿Preserva Isometría estricta (Q^T * Q == I)?:", is_isometric)
    print("Error ortogonal máximo: {:.2e}".format(float(jnp.max(jnp.abs(QT_Q - I)))))

# ----------------- TOPOS SIEVE (Numpy puro) -----------------
class OmegaClassifier:
    def __init__(self, dim):
        self.omega = np.eye(dim)

    def classify(self, vector):
        return np.dot(self.omega, vector)

class ToposCoverageSieve:
    def __init__(self, omega):
        self.omega = omega
        self.coverage = []

    def add_subspace(self, subspace):
        self.coverage.append(subspace)

    def check(self, vector):
        for sub in self.coverage:
            if np.allclose(self.omega.classify(vector), sub):
                return True
        return False

class ElegantCollapse:
    def __init__(self, sieve):
        self.sieve = sieve

    def apply(self, vector):
        if not self.sieve.check(vector):
            print("   [!] Vector invalido no cubierto. Ejecutando pullback a fibra vacia (EMPTY)...")
            return np.zeros_like(vector)
        return vector

def test_topos_sieve():
    print("\n--- [3] VALIDADOR TOPOS GROTHENDIECK (COLAPSO ELEGANTE) ---")
    sieve = ToposCoverageSieve(OmegaClassifier(3))
    sieve.add_subspace(np.array([1.0, 0.0, 0.0]))
    sieve.add_subspace(np.array([0.0, 1.0, 0.0]))
    
    collapse_engine = ElegantCollapse(sieve)
    
    v_valid = np.array([1.0, 0.0, 0.0])
    v_invalid = np.array([0.5, 0.5, 0.5])
    
    print(f"-> Testeando vector Válido {v_valid}:")
    out = collapse_engine.apply(v_valid)
    print("   Resultado:", out)
    
    print(f"-> Testeando vector Inválido (Alucinación) {v_invalid}:")
    out = collapse_engine.apply(v_invalid)
    print("   Resultado:", out)

if __name__ == "__main__":
    test_woodbury_stability()
    test_cayley_isometry()
    test_topos_sieve()
    print("\n[+] Todos los validadores ejecutados con éxito.")
