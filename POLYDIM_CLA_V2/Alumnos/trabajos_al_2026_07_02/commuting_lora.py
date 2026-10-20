"""
Tarea 5 — POLYDIM
Experto: PEFT (Parameter-Efficient Fine-Tuning)

Objetivo real del enunciado (no la lectura ingenua):
No se pide "invertir el orden de A y B" en LoRA estándar. Se pide construir
una FAMILIA de adaptadores de bajo rango cuyas actualizaciones de peso
ΔW_i conmuten entre sí como matrices — y que esa propiedad sea verificable
en tres niveles distintos:

  (1) Álgebra estática:      [ΔW_i, ΔW_j] = ΔW_i ΔW_j - ΔW_j ΔW_i ≈ 0
  (2) Dinámica (flujos Lie): exp(t·ΔW_i) y exp(s·ΔW_j) conmutan como
                              transformaciones continuas (Baker-Campbell-Hausdorff:
                              si [X,Y]=0 entonces exp(X)exp(Y)=exp(X+Y)=exp(Y)exp(X))
  (3) Categórica/conceptual: aplicar adaptador_1 luego adaptador_2 sobre una
                              entrada da EXACTAMENTE el mismo resultado que
                              adaptador_2 luego adaptador_1 (diagrama conmutativo)

Construcción clave — "conmutar por diseño, no por suerte":
Todos los adaptadores comparten una base propia FIJA U (no entrenable).
Cada adaptador i solo aprende una matriz diagonal D_i en esa base:

        ΔW_i = U · diag(d_i) · U⁻¹

Las matrices diagonales SIEMPRE conmutan entre sí, sin importar sus valores.
Esto convierte una propiedad matemática difícil de imponer en una consecuencia
automática de la arquitectura. El rango bajo r se logra dejando en cero todas
las entradas de d_i salvo r de ellas.

Supuesto explícito (limitación pedagógica que hay que declarar):
Esta construcción exige que la matriz de pesos sea CUADRADA (d_in == d_out),
porque la conmutación de matrices solo está definida para matrices cuadradas
que actúan sobre el mismo espacio. Es el caso típico de las proyecciones
Q, K, V en atención (d_model x d_model).
"""

import torch
import torch.nn as nn
import math


# ---------------------------------------------------------------------------
# 1. Módulo principal: LoRA con adaptadores que conmutan por construcción
# ---------------------------------------------------------------------------

class CommutingLoRALinear(nn.Module):
    """
    Capa lineal con un adaptador LoRA de rango r, restringido a una base
    propia compartida U, de forma que su ΔW conmuta con el de cualquier
    otro CommutingLoRALinear que use la MISMA base U.

    Parámetros
    ----------
    dim : int
        Dimensión de la matriz de pesos (cuadrada, dim x dim).
    r : int
        Rango del adaptador (0 <= r <= dim). r=0 -> adaptador nulo (ΔW=0).
    shared_basis : torch.Tensor
        Matriz U (dim x dim), invertible, COMPARTIDA entre todos los
        adaptadores que deban conmutar entre sí. Se registra como buffer
        (no se entrena): la base es la estructura fija que garantiza
        la conmutatividad; lo único entrenable es la diagonal.
    """

    def __init__(self, dim: int, r: int, shared_basis: torch.Tensor):
        super().__init__()
        assert shared_basis.shape == (dim, dim), "La base compartida debe ser cuadrada dim x dim"
        assert 0 <= r <= dim, "El rango r debe estar entre 0 y dim"

        self.dim = dim
        self.r = r

        # Peso base "congelado" (simula un peso pre-entrenado W0)
        self.register_buffer("W0", torch.eye(dim))

        # Base propia compartida: FIJA, no entrenable
        self.register_buffer("U", shared_basis)
        self.register_buffer("U_inv", torch.linalg.inv(shared_basis))

        # Máscara de rango bajo: solo r posiciones de la diagonal son libres
        mask = torch.zeros(dim)
        mask[:r] = 1.0
        self.register_buffer("rank_mask", mask)

        # Único parámetro entrenable: la diagonal (ya enmascarada a rango r)
        self.d = nn.Parameter(torch.randn(dim) * 0.01)

    def delta_w(self) -> torch.Tensor:
        """Devuelve ΔW_i = U · diag(d_i ⊙ mask) · U⁻¹"""
        d_masked = self.d * self.rank_mask
        return self.U @ torch.diag(d_masked) @ self.U_inv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_eff = self.W0 + self.delta_w()
        return x @ W_eff.T


def make_shared_orthogonal_basis(dim: int, seed: int = 0) -> torch.Tensor:
    """
    Genera una base U compartida, ortogonal (bien condicionada: U⁻¹ = Uᵀ),
    vía descomposición QR de una matriz aleatoria. Ortogonal no es
    obligatorio para la construcción, pero evita problemas numéricos
    de U⁻¹ y es la práctica recomendada (equivalente a SOTA en
    parametrizaciones ortogonales tipo OFT/BOFT en PEFT).
    """
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(dim, dim, generator=g)
    Q, _ = torch.linalg.qr(A)
    return Q


# ---------------------------------------------------------------------------
# 2. Nivel de verificación 1 — Álgebra estática (conmutador numérico)
# ---------------------------------------------------------------------------

def check_static_commutator(dW1: torch.Tensor, dW2: torch.Tensor, tol: float = 1e-5) -> float:
    """Devuelve ||ΔW1 ΔW2 - ΔW2 ΔW1||_F y valida que sea ≈ 0."""
    commutator = dW1 @ dW2 - dW2 @ dW1
    norm = torch.linalg.norm(commutator).item()
    assert norm < tol, f"Los adaptadores NO conmutan estáticamente: norma={norm}"
    return norm


# ---------------------------------------------------------------------------
# 3. Nivel de verificación 2 — Flujos de Lie (exponenciales de matrices)
# ---------------------------------------------------------------------------

def check_lie_flow_commutation(dW1: torch.Tensor, dW2: torch.Tensor,
                                t: float = 0.7, s: float = 1.3,
                                tol: float = 1e-4) -> float:
    """
    Si [ΔW1, ΔW2] = 0, entonces por Baker-Campbell-Hausdorff:
        exp(t·ΔW1) · exp(s·ΔW2) == exp(s·ΔW2) · exp(t·ΔW1)
    para cualquier t, s. Esto es más fuerte que el chequeo estático:
    valida que la propiedad se mantiene bajo evolución continua
    (relevante si estos adaptadores se interpretan como flujos de
    actualización, no solo como saltos discretos).
    """
    exp1 = torch.matrix_exp(t * dW1)
    exp2 = torch.matrix_exp(s * dW2)
    lhs = exp1 @ exp2
    rhs = exp2 @ exp1
    norm = torch.linalg.norm(lhs - rhs).item()
    assert norm < tol, f"Los flujos de Lie NO conmutan: norma={norm}"
    return norm


# ---------------------------------------------------------------------------
# 4. Nivel de verificación 3 — Diagrama conmutativo (categórico/conceptual)
# ---------------------------------------------------------------------------

def check_categorical_diagram(layer1: CommutingLoRALinear, layer2: CommutingLoRALinear,
                               x: torch.Tensor, tol: float = 1e-4) -> float:
    """
    Emula la verificación de un diagrama conmutativo en teoría de
    categorías: dos caminos (funtores) desde el mismo objeto de partida
    x hasta el objeto final deben coincidir.

        camino A:  x --layer1--> h --layer2--> y_A
        camino B:  x --layer2--> h --layer1--> y_B

    Si y_A == y_B para toda x, el diagrama conmuta.
    """
    y_A = layer2(layer1(x))
    y_B = layer1(layer2(x))
    norm = torch.linalg.norm(y_A - y_B).item()
    assert norm < tol, f"El diagrama NO conmuta: norma={norm}"
    return norm


# ---------------------------------------------------------------------------
# 5. Verificación simbólica ligera (control de sanidad matemática, sympy)
#    Caso de juguete 2x2 para "ver" la conmutatividad en forma cerrada,
#    no solo numérica.
# ---------------------------------------------------------------------------

def symbolic_sanity_check():
    import sympy as sp

    a1, a2, u = sp.symbols("a1 a2 u", real=True)
    # Base propia compartida simbólica (rotación por ángulo u), 2x2
    U = sp.Matrix([[sp.cos(u), -sp.sin(u)], [sp.sin(u), sp.cos(u)]])
    U_inv = U.T  # rotación: ortogonal, U^-1 = U^T

    D1 = sp.diag(a1, 0)   # rango 1
    D2 = sp.diag(0, a2)   # rango 1, distinto subespacio propio

    dW1 = sp.simplify(U * D1 * U_inv)
    dW2 = sp.simplify(U * D2 * U_inv)

    commutator = sp.simplify(dW1 * dW2 - dW2 * dW1)
    return commutator  # debe simplificar a la matriz cero simbólicamente
