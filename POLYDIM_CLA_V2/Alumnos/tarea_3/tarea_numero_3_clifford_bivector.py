"""
TAREA 3 - Enjambre POLYDIM (interpretación pedagógica)
=======================================================

Consigna interpretada:
Construir un framework de benchmarking (estilo DevOps/SRE) que monitoree
el entrenamiento de una capa neuronal basada en Álgebra de Clifford Cl(3,0),
verificando que las actualizaciones de gradiente queden restringidas
EXCLUSIVAMENTE a los componentes bivectoriales (grado 2): e12, e13, e23.

Autor: Alumno (worker_id: claud62701@gmail.com) - Enjambre POLYDIM
Ticket: 3
"""

import torch
import torch.nn as nn
import math

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# 1. EL ÁLGEBRA: definimos la base de Cl(3,0)
# ---------------------------------------------------------------------------
# Orden de la base (8 componentes = 2^3, para 3 dimensiones):
#   índice 0 -> 1        (grado 0, escalar)
#   índice 1 -> e1       (grado 1, vector)
#   índice 2 -> e2       (grado 1, vector)
#   índice 3 -> e3       (grado 1, vector)
#   índice 4 -> e12      (grado 2, bivector)
#   índice 5 -> e13      (grado 2, bivector)
#   índice 6 -> e23      (grado 2, bivector)
#   índice 7 -> e123     (grado 3, trivector/pseudoescalar)

GRADE0_IDX = [0]
GRADE1_IDX = [1, 2, 3]
GRADE2_IDX = [4, 5, 6]      # <- los únicos que tienen permiso de actualizarse
GRADE3_IDX = [7]

N_COMPONENTS = 8


def grade_mask(allowed_indices, n=N_COMPONENTS, device=None, dtype=torch.float32):
    """Devuelve un vector binario que vale 1 en los índices permitidos."""
    m = torch.zeros(n, device=device, dtype=dtype)
    m[allowed_indices] = 1.0
    return m


# ---------------------------------------------------------------------------
# 2. LA CAPA: un "GeometricLinear" cuyo peso es un multivector completo
# ---------------------------------------------------------------------------
class CliffordBivectorLinear(nn.Module):
    """
    Capa lineal simplificada donde el peso es un multivector de Cl(3,0)
    (8 componentes). La salida se calcula proyectando la entrada (tratada
    como vector grado-1) contra el peso multivector mediante una
    combinación lineal simple (no el producto geométrico completo, para
    mantener el ejemplo pedagógico tratable) pero el PUNTO CLAVE de la
    tarea no es la operación en sí, sino el control del gradiente.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # Un multivector de pesos por cada par (entrada, salida)
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, N_COMPONENTS) * 0.1
        )
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Máscara de grado 2 (bivectores). No es un parámetro entrenable,
        # es una constante estructural del álgebra.
        self.register_buffer("bivector_mask", grade_mask(GRADE2_IDX))

        # Este hook es EL MECANISMO REAL que impone la restricción:
        # se ejecuta durante el backward, ANTES de que el optimizador
        # (y su estado interno, ej. momentum de Adam) vea el gradiente.
        self.weight.register_hook(self._restrict_to_bivectors)

    def _restrict_to_bivectors(self, grad: torch.Tensor) -> torch.Tensor:
        # grad tiene shape (out_features, in_features, 8).
        # Multiplicamos por la máscara para anular grados 0, 1 y 3.
        return grad * self.bivector_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_features). Usamos solo la "energía total" del
        # multivector de peso (suma de sus 8 componentes ponderadas) como
        # una proyección escalar simple para tener una red que efectivamente
        # aprenda algo con estos pesos.
        w_effective = self.weight.sum(dim=-1)          # (out, in)
        return x @ w_effective.T + self.bias


# ---------------------------------------------------------------------------
# 3. FRAMEWORK DE BENCHMARKING / MONITOREO (estilo SRE)
# ---------------------------------------------------------------------------
class GeoBenchmark:
    """
    Monitorea, en cada paso de entrenamiento:
      - grad_leakage: cuánta "fuga" de gradiente hay fuera de los bivectores.
        Debe ser ~0 si el hook funciona. Si no lo es, es una FALLA MATEMÁTICA.
      - bivector_grad_norm: energía del gradiente efectivamente usado.
      - sps (Semantic Preservation Score, proxy): similitud coseno entre
        la salida del modelo antes y después del paso de optimización,
        para la MISMA entrada de referencia. Un SPS que colapsa a 0 indica
        que el modelo "olvidó" lo que sabía (mala señal).
      - geoid_stability: estabilidad de la norma del subespacio bivectorial
        de los pesos a lo largo del entrenamiento (desviación estándar
        relativa sobre una ventana móvil). Baja variación = entrenamiento
        estable; picos = inestabilidad geométrica.
    """

    def __init__(self, model: CliffordBivectorLinear, window: int = 10, eps: float = 1e-6):
        self.model = model
        self.window = window
        self.eps = eps
        self.history_geoid = []
        self.log = []

    def _forbidden_grade_leakage(self) -> float:
        mask_forbidden = 1.0 - self.model.bivector_mask
        grad = self.model.weight.grad
        if grad is None:
            return 0.0
        leaked = (grad * mask_forbidden).abs().sum().item()
        return leaked

    def _bivector_grad_norm(self) -> float:
        grad = self.model.weight.grad
        if grad is None:
            return 0.0
        return (grad * self.model.bivector_mask).norm().item()

    def _geoid_norm(self) -> float:
        # Norma de la parte bivectorial de TODO el tensor de pesos.
        w = self.model.weight
        return (w * self.model.bivector_mask).norm().item()

    def _sps(self, ref_input: torch.Tensor, out_before: torch.Tensor) -> float:
        with torch.no_grad():
            out_after = self.model(ref_input)
        cos = torch.nn.functional.cosine_similarity(
            out_before.flatten(), out_after.flatten(), dim=0, eps=self.eps
        )
        return cos.item()

    def step(self, ref_input: torch.Tensor, out_before: torch.Tensor, step_idx: int):
        leakage = self._forbidden_grade_leakage()
        biv_grad_norm = self._bivector_grad_norm()
        geoid = self._geoid_norm()
        sps = self._sps(ref_input, out_before)

        self.history_geoid.append(geoid)
        window_vals = self.history_geoid[-self.window:]
        geoid_stability = (
            float(torch.tensor(window_vals).std()) / (float(torch.tensor(window_vals).mean()) + self.eps)
            if len(window_vals) > 1 else 0.0
        )

        status = "SUCCESS"
        reasons = []
        if leakage > 1e-5:
            status = "MATH_FAILURE"
            reasons.append(f"grad_leakage={leakage:.3e} > 0 (la restricción de grado falló)")
        if math.isnan(sps) or math.isnan(geoid_stability):
            status = "MATH_FAILURE"
            reasons.append("NaN detectado en métricas (posible divergencia tipo Jacobiano)")

        record = {
            "step": step_idx,
            "grad_leakage": leakage,
            "bivector_grad_norm": biv_grad_norm,
            "sps": sps,
            "geoid_stability": geoid_stability,
            "status": status,
            "reasons": reasons,
        }
        self.log.append(record)
        return record


# ---------------------------------------------------------------------------
# 4. EXPERIMENTO: entrenar un paso de "tarea de juguete" y verificar
# ---------------------------------------------------------------------------
def run_experiment(n_steps: int = 30, lr: float = 5e-2):
    torch.manual_seed(42)
    model = CliffordBivectorLinear(in_features=4, out_features=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bench = GeoBenchmark(model, window=5)

    # Tarea de juguete: aprender la identidad (regresión simple),
    # el objetivo NO es lo interesante, es solo un vehículo de entrenamiento
    # para poder observar el comportamiento del gradiente restringido.
    x_ref = torch.randn(8, 4)
    y_ref = x_ref.clone()

    for step in range(n_steps):
        optimizer.zero_grad()
        out_before = model(x_ref)              # salida ANTES del update (para SPS)
        loss = torch.nn.functional.mse_loss(out_before, y_ref)
        loss.backward()                         # el hook actúa acá mismo
        optimizer.step()                        # Adam solo ve gradiente ya filtrado

        record = bench.step(x_ref, out_before.detach(), step)
        record["loss"] = loss.item()

    return model, bench


if __name__ == "__main__":
    model, bench = run_experiment()

    print(f"{'step':>4} | {'loss':>10} | {'grad_leakage':>13} | {'biv_grad_norm':>13} | {'sps':>7} | {'geoid_stab':>10} | status")
    for r in bench.log:
        print(f"{r['step']:>4} | {r['loss']:>10.6f} | {r['grad_leakage']:>13.3e} | "
              f"{r['bivector_grad_norm']:>13.6f} | {r['sps']:>7.4f} | {r['geoid_stability']:>10.6f} | {r['status']}")

    final = bench.log[-1]
    print("\nResumen final:")
    print(f"  Fuga de gradiente fuera de bivectores (debe ser 0): {final['grad_leakage']}")
    print(f"  Estado final: {final['status']}")

    # Verificación estructural extra: confirmamos que las componentes
    # grado 0, 1 y 3 de los PESOS nunca se movieron de su valor inicial
    # relativo (más allá de un pequeño init aleatorio, ya que solo la
    # actualización de gradiente está restringida, no el valor inicial).
    forbidden = GRADE0_IDX + GRADE1_IDX + GRADE3_IDX
    grad_final = model.weight.grad
    max_forbidden_grad = grad_final[:, :, forbidden].abs().max().item()
    print(f"  Máximo gradiente residual en componentes prohibidas: {max_forbidden_grad:.3e}")
    assert max_forbidden_grad == 0.0, "FALLA MATEMÁTICA: el hook no está bloqueando correctamente."
    print("  Verificación OK: la restricción a bivectores puros se cumple exactamente.")
