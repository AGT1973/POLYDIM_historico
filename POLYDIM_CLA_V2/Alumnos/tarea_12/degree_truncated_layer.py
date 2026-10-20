"""
POLYDIM Swarm - Tarea 12 (Escuadron 1)
Rol: experto en Categorical Deep Learning
Afirmacion a validar (reconstruida de la Tarea 9, reutilizada en la 12):
    "una capa que reemplaza a nn.Linear, ignorando interacciones de orden
     superior (>2), ahorra computo sin perjudicar la generalizacion
     composicional."

Nota de infraestructura: no hay PyTorch instalado en este entorno (no se
instala por no ejecutar la comprobacion final aqui, ver SKILL.md paso 8).
Implementamos una clase con la MISMA interfaz que nn.Linear
(forward(x), parameters()) en NumPy puro. Es matematicamente equivalente:
la capa es una combinacion lineal de "features" (monomios), exactamente
lo que nn.Linear + feature expansion manual hace en PyTorch.

Metodologia (analisis de Fourier de funciones booleanas, O'Donnell 2014):
Toda funcion f: {-1,1}^n -> R tiene una unica expansion exacta
    f(x) = sum_{S subset [n]} f_hat(S) * prod_{i in S} x_i
Los "monomios" de grado |S| son ortonormales bajo la medida uniforme del
cubo. Una capa que trunca en grado k solo puede representar la parte de
f cuyo soporte de Fourier tiene grado <= k. Si f tiene TODO su peso en un
monomio de grado > k (ortogonal a cualquier combinacion de grado <= k),
ninguna cantidad de entrenamiento puede recuperarlo: no es una limitacion
empirica, es una imposibilidad algebraica exacta.

Target elegido: PARIDAD de 3 bits, f(x1,x2,x3) = x1*x2*x3 (en codificacion
+-1). Es el ejemplo canonico de una funcion cuyo ÚNICO coeficiente de
Fourier no nulo es el de grado 3 (f_hat({1,2,3}) = 1, todo el resto = 0).
Es "generalizacion composicional" en su forma mas pura: para acertar en
la tripleta hay que capturar la interaccion de orden 3 completa, ninguna
combinacion de orden <=2 correlaciona con ella (ortogonalidad exacta).
"""
import itertools
import numpy as np


# ---------------------------------------------------------------------
# 1. Universo de datos: las 8 esquinas del cubo {-1,1}^3, target = paridad
# ---------------------------------------------------------------------
def full_cube(n):
    return np.array(list(itertools.product([-1, 1], repeat=n)), dtype=float)


def parity_target(X):
    # x1 * x2 * x3 ... * xn  (paridad multiplicativa completa)
    return np.prod(X, axis=1)


# ---------------------------------------------------------------------
# 2. Capa "nativa" (interfaz estilo nn.Linear) con grado de truncamiento k
#    Expande features = todos los monomios de tamaño 0..k, y ajusta un
#    peso lineal sobre esas features (esto ES lo que hace una red que
#    "ignora homotopias de orden > k": no puede construir ningun monomio
#    de grado mayor a k con una sola capa lineal sobre esas features).
# ---------------------------------------------------------------------
class DegreeKLinear:
    """Reemplazo de nn.Linear que opera sobre monomios de grado <= k.
    k = 1 equivale a un nn.Linear estandar (solo terminos lineales).
    k = 2 agrega interacciones de a pares (bilineal).
    k = n es la base completa (representa CUALQUIER funcion en {-1,1}^n).
    """

    def __init__(self, n_inputs: int, degree_k: int):
        self.n = n_inputs
        self.k = degree_k
        self.monomial_indices = []
        for d in range(0, degree_k + 1):
            for combo in itertools.combinations(range(n_inputs), d):
                self.monomial_indices.append(combo)
        self.weight = np.zeros(len(self.monomial_indices))  # "parameters()"

    def _features(self, X):
        feats = np.ones((X.shape[0], len(self.monomial_indices)))
        for j, combo in enumerate(self.monomial_indices):
            if len(combo) == 0:
                continue
            feats[:, j] = np.prod(X[:, combo], axis=1)
        return feats

    def forward(self, X):
        return self._features(X) @ self.weight

    def fit_least_squares(self, X, y):
        """Ajuste exacto (no hace falta gradiente: hay <=8 puntos y
        <=8 features como maximo). Sanity-check interno, no es la
        comprobacion final del alumno."""
        F = self._features(X)
        self.weight, *_ = np.linalg.lstsq(F, y, rcond=None)

    def parameters(self):
        return self.weight

    def flop_estimate(self, n_samples=1):
        # 1 multiplicacion por feature no-constante + 1 producto escalar
        return n_samples * (len(self.monomial_indices))


# ---------------------------------------------------------------------
# 3. Experimento: comparar degree_k=2 (trunca homotopias de orden >2,
#    la tesis de la Tarea 12) contra degree_k=3 (base completa)
# ---------------------------------------------------------------------
def run_experiment():
    n = 3
    X = full_cube(n)
    y = parity_target(X)

    print("=== Tarea 12: DegreeKLinear como reemplazo de nn.Linear ===")
    print(f"Target: paridad de {n} bits (x1*x2*x3), {len(X)} puntos exactos.\n")

    results = {}
    for k in [1, 2, 3]:
        layer = DegreeKLinear(n_inputs=n, degree_k=k)
        layer.fit_least_squares(X, y)
        pred = layer.forward(X)
        residual = y - pred
        ss_res = float(np.sum(residual ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        n_params = len(layer.weight)
        flops = layer.flop_estimate(n_samples=len(X))
        results[k] = dict(r2=r2, n_params=n_params, flops=flops,
                           max_abs_err=float(np.max(np.abs(residual))))
        print(f"grado k={k}: n_params={n_params:2d}  FLOPs(8 pts)={flops:3d}  "
              f"R^2={r2:+.6f}  error_max={results[k]['max_abs_err']:.6f}")

    print("\nInterpretacion:")
    print("- k=1 (equivalente a nn.Linear puro, 'ignora TODA interaccion'):")
    print("  R^2 = 0.0 exacto -> cero correlacion con la paridad. Ahorra")
    print("  computo (4 params) pero destruye por completo la tarea.")
    print("- k=2 ('ignora homotopias de orden >2', la tesis del enunciado):")
    print("  R^2 = 0.0 exacto tambien -> el termino de paridad de 3 bits es")
    print("  ORTOGONAL a cualquier combinacion de grado <=2. Ahorra 1 de 8")
    print("  parametros (12.5%) pero SIGUE siendo un fracaso total, no una")
    print("  degradacion parcial: la funcion objetivo es indetectable.")
    print("- k=3 (base completa, sin truncar):")
    print("  R^2 = 1.0 exacto -> unica combinacion capaz de representar la")
    print("  paridad, al costo de 8/8 parametros (sin ahorro).")
    print()
    print("Conclusion matematica (no solo empirica): para CUALQUIER funcion")
    print("cuyo soporte de Fourier vive enteramente en grado > k, truncar en")
    print("grado k no es un compromiso ('ahorra computo, pierde un poco de")
    print("generalizacion'): es una imposibilidad algebraica total, sin")
    print("relacion con cuanto se entrene el modelo. La afirmacion del")
    print("enunciado ('ahorra computo SIN perjudicar') es FALSA en general;")
    print("es correcta unicamente para el subconjunto de funciones cuyo peso")
    print("de Fourier ya vive en grado <= k (funciones 'casi lineales'), y")
    print("ese subconjunto no se puede asumir sin conocer la tarea real.")

    return results


if __name__ == "__main__":
    run_experiment()
