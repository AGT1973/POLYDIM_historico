"""
Tarea 25 - POLYDIM V2 (POPL 2027 Amendment, Seccion 9.4)
=========================================================

Framework de benchmarking (PEFT-oriented) que monitorea el MAYOR VALOR SINGULAR
del Jacobiano de un operador FIXPOINT T: R^n -> R^n, para verificar empiricamente
la condicion suficiente de contraccion del Teorema de Punto Fijo de Banach:

    ||J_T(x)||_2 <= lambda < 1   para todo x en el dominio
    =========================================================>  T tiene un unico
    punto fijo x*, y la iteracion x_{k+1} = T(x_k) converge a x* con tasa
    geometrica lambda^k.

Metodo de estimacion (sin materializar el Jacobiano completo):
----------------------------------------------------------------
 Para redes de gran dimension, formar J_T(x) explicitamente cuesta O(n^2) memoria
y obtener su SVD cuesta O(n^3). En su lugar se usa **iteracion de potencia** sobre
J^T J, obteniendo el mayor valor singular sigma_max(J) = sqrt(lambda_max(J^T J))
usando unicamente productos Jacobiano-vector (JVP) y vector-Jacobiano (VJP), cada
uno con el costo de un forward/backward pass (O(n) en la práctica, via autograd).
Esta es exactamente la tecnica de "Spectral Normalization" (Miyato et al., 2018),
aqui aplicada como PENALIZACION / MONITOR en vez de como normalizacion forzada de pesos.

Algoritmo (por paso de benchmarking):
    v <- vector unitario aleatorio (o el v final del paso anterior, "power iteration
         amortizada" -- 1 sola iteracion por step, tal como en Spectral Norm GAN)
    u <- J v            (JVP, forward-mode autodiff)
    sigma <- ||u||_2
    u <- u / sigma
    v <- J^T u           (VJP, reverse-mode autodiff)
    v <- v / ||v||_2
    devolver sigma como estimador de sigma_max(J_T(x))

Validacion empirica del criterio (lo que efectivamente se comprueba, no se da por
hecho): se corren trayectorias de la iteracion de punto fijo x_{k+1} = T(x_k) desde
multiples puntos iniciales aleatorios, en dos regimenes:
    (a) SIN restriccion espectral  -> se espera sigma_max >= 1 en muchos puntos y
        posible NO convergencia (o convergencia solo si el operador es contractivo
        "por casualidad" en esa region).
    (b) CON penalizacion espectral (reescalado de pesos para forzar sigma_max < lambda)
        -> se espera sigma_max < lambda de forma consistente y convergencia empirica
        geometrica de ||x_{k+1} - x_k|| en TODAS las trayectorias.

Nada se da por sentado: el script mide y reporta ambos regimenes.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Callable, List

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Operador FIXPOINT de juguete: una capa afin + no linealidad Lipschitz-1
#    (tanh), que es la forma generica de un bloque de punto fijo tipo Deep
#    Equilibrium Model / capa de "FIXPOINT" mencionada en el paper POLYDIM.
# ---------------------------------------------------------------------------
class FixpointOperator(nn.Module):
    """T(x) = tanh(W x + b). tanh es 1-Lipschitz, asi que la cota de
    contraccion de T depende enteramente de la norma espectral de W:
        ||J_T(x)||_2 <= ||W||_2  (cota superior valida siempre, por la regla
        de la cadena y Lipschitz(tanh) = 1).
    Esto nos da un caso de prueba controlable: escalar W nos permite forzar
    o romper la condicion de contraccion a voluntad, para validar el monitor.
    """

    def __init__(self, dim: int, spectral_scale: float = 1.5, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        W = torch.randn(dim, dim, generator=g)
        # Normalizamos W a norma espectral 1 y luego reescalamos: asi
        # `spectral_scale` es (aprox) la norma espectral real de W.
        with torch.no_grad():
            s = torch.linalg.matrix_norm(W, ord=2)
            W = W / s * spectral_scale
        self.W = nn.Parameter(W)
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x @ self.W.T + self.b)

    @torch.no_grad()
    def apply_spectral_penalty(self, lambda_target: float = 0.9) -> None:
        """Implementa la 'Spectral Lipschitz Penalty' del paper (Sec. 9.4):
        reescala W para que su norma espectral quede por debajo de
        lambda_target, garantizando ||J_T||_2 <= lambda_target < 1."""
        s = torch.linalg.matrix_norm(self.W, ord=2)
        if s > lambda_target:
            self.W.mul_(lambda_target / s)


# ---------------------------------------------------------------------------
# 2. Estimador del mayor valor singular del Jacobiano via iteracion de
#    potencia + JVP/VJP (sin formar el Jacobiano explicito).
# ---------------------------------------------------------------------------
def estimate_top_singular_value(
    T: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    v0: torch.Tensor | None = None,
    n_power_iters: int = 1,
) -> tuple[float, torch.Tensor]:
    """Estima sigma_max(J_T(x)) mediante `n_power_iters` pasos de iteracion de
    potencia sobre J^T J, usando JVP (forward-mode) y VJP (reverse-mode).

    Devuelve (sigma_estimado, v_final) para permitir "amortizar" la iteracion
    de potencia entre pasos de benchmarking consecutivos (tal como se hace en
    Spectral Normalization: 1 iteracion por step es suficiente si se reusa v).
    """
    dim = x.shape[-1]
    if v0 is None:
        v = torch.randn(dim)
        v = v / v.norm()
    else:
        v = v0

    x = x.detach().requires_grad_(True)

    sigma = torch.tensor(0.0)
    for _ in range(n_power_iters):
        # JVP: u = J v  (forward-mode autodiff, sin construir J)
        _, u = torch.autograd.functional.jvp(T, (x,), (v,), create_graph=False)
        sigma = u.norm()
        if sigma.item() < 1e-12:
            break
        u = u / sigma

        # VJP: w = J^T u (reverse-mode autodiff, sin construir J)
        _, vjp_fn = torch.autograd.functional.vjp(T, x, u, create_graph=False)
        w = vjp_fn
        w_norm = w.norm()
        if w_norm.item() < 1e-12:
            break
        v = w / w_norm

    return sigma.item(), v.detach()


# ---------------------------------------------------------------------------
# 3. Framework de benchmarking
# ---------------------------------------------------------------------------
@dataclass
class FixpointBenchmarkReport:
    regime: str
    lambda_threshold: float
    sigma_history: List[float] = field(default_factory=list)
    residual_history: List[float] = field(default_factory=list)  # ||x_{k+1}-x_k||
    converged_flags: List[bool] = field(default_factory=list)
    violations: int = 0

    def summary(self) -> str:
        max_sigma = max(self.sigma_history) if self.sigma_history else float("nan")
        n_conv = sum(self.converged_flags)
        n_total = len(self.converged_flags)
        return (
            f"[{self.regime}] lambda_umbral={self.lambda_threshold:.3f} | "
            f"sigma_max observado en toda la corrida={max_sigma:.4f} | "
            f"violaciones del umbral (sigma>=lambda)={self.violations} | "
            f"trayectorias convergentes={n_conv}/{n_total}"
        )


def benchmark_fixpoint_operator(
    T: FixpointOperator,
    dim: int,
    n_trajectories: sint = 20,
    n_steps: int = 60,
    lambda_threshold: float = 1.0,
    residual_tol: float = 1e-6,
    regime_name: str = "regimen",
    seed: int = 42,
) -> FixpointBenchmarkReport:
    """Corre `n_trajectories` iteraciones de punto fijo x_{k+1}=T(x_k) desde
    puntos iniciales aleatorios, monitoreando en cada paso el mayor valor
    singular estimado del Jacobiano de T (via power iteration + JVP/VJP), y
    registrando si la trayectoria efectivamente converge (criterio de Cauchy
    sobre el residuo ||x_{k+1}-x_k||)."""

    report = FixpointBenchmarkReport(regime=regime_name, lambda_threshold=lambda_threshold)
    g = torch.Generator().manual_seed(seed)

    for traj in range(n_trajectories):
        x = torch.randn(dim, generator=g)
        v = None  # se amortiza la iteracion de potencia dentro de cada trayectoria
        converged = False

        for step in range(n_steps):
            sigma, v = estimate_top_singular_value(T, x, v0=v, n_power_iters=1)
            report.sigma_history.append(sigma)
            if sigma >= lambda_threshold:
                report.violations += 1

            with torch.no_grad():
                x_next = T(x)
                residual = (x_next - x).norm().item()
            report.residual_history.append(residual)

            if residual < residual_tol:
                converged = True
                x = x_next
                break
            x = x_next.detach()

        report.converged_flags.append(converged)

    return report


# ---------------------------------------------------------------------------
# 4. Demostracion empirica: regimen SIN penalidad espectral vs CON ella.
# ---------------------------------------------------------------------------
def run_demo(dim: int = 32) -> None:
    lambda_target = 0.9

    print("=" * 78)
    print("REGIMEN (a): SIN restriccion espectral (spectral_scale=1.5 > 1)")
    print("=" * 78)
    T_unconstrained = FixpointOperator(dim=dim, spectral_scale=1.5, seed=0)
    report_a = benchmark_fixpoint_operator(
        T_unconstrained,
        dim=dim,
        n_trajectories=20,
        n_steps=80,
        lambda_threshold=1.0,
        regime_name="sin_restriccion",
    )
    print(report_a.summary())

    print()
    print("=" * 78)
    print(f"REGIMEN (b): CON Spectral Lipschitz Penalty (lambda={lambda_target})")
    print("=" * 78)
    T_constrained = FixpointOperator(dim=dim, spectral_scale=1.5, seed=0)
    T_constrained.apply_spectral_penalty(lambda_target=lambda_target)
    report_b = benchmark_fixpoint_operator(
        T_constrained,
        dim=dim,
        n_trajectories=20,
        n_steps=80,
        lambda_threshold=lambda_target,
        regime_name="con_penalidad_espectral",
    )
    print(report_b.summary())

    print()
    print("=" * 78)
    print("CONCLUSION EMPIRICA")
    print("=" * 78)
    print(
        "Si el regimen (b) muestra sigma_max < lambda de forma consistente "
        "(violaciones=0) Y una tasa de convergencia de trayectorias claramente "
        "mayor que en (a), el resultado es CONSISTENTE con el Teorema de Punto "
        "Fijo de Banach: la cota espectral del Jacobiano SI predice la garantia "
        "de convergencia del operador FIXPOINT. Si no se observa esa diferencia, "
        "documentarlo tambien es un resultado valido (no forzar la conclusion)."
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    run_demo(dim=32)
