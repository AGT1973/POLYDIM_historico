"""
Tarea 59 - POLYDIM
Validacion empirica: pooling promedio temporal + reconstruccion optima lineal
(pseudoinversa de Moore-Penrose) bajo ruido gaussiano extremo.

Rol: Ingeniero de PyTorch Dynamo
Alumno: proyectosnuevos.andrea@gmail.com

Este script:
  1. Construye el operador de pooling promedio P para distintos tamanos de bloque k.
  2. Verifica numericamente que P+ = k * P^T (propiedad de bloques ortogonales).
  3. Genera una senal latente sintetica suave x.
  4. Corre un experimento de Monte Carlo inyectando ruido gaussiano extremo,
     midiendo el MSE de reconstruccion y comparandolo contra la formula teorica:
         MSE = sesgo + varianza
         sesgo    = ||x - P+ P x||^2 / T      (depende de k, NO de sigma)
         varianza = sigma^2                    (tr(P+ P+^T)/T = 1, constante para todo k)

No se ejecuta como parte de la entrega final (ver tarea_59.md); el alumno debe
correrlo el mismo para comprobar el resultado.
"""

import torch


def build_pooling_matrix(T: int, k: int) -> torch.Tensor:
    """Construye P en R^{m x T}, pooling promedio con bloques de tamano k (m = T // k)."""
    assert T % k == 0, "T debe ser divisible por k para bloques exactos"
    m = T // k
    P = torch.zeros(m, T, dtype=torch.float64)
    for i in range(m):
        P[i, i * k : (i + 1) * k] = 1.0 / k
    return P


def theoretical_bias(x: torch.Tensor, P: torch.Tensor, P_pinv: torch.Tensor) -> float:
    """||x - P+ P x||^2 (componente de x fuera del subespacio 'constante por bloques')."""
    x_proj = P_pinv @ (P @ x)
    return torch.sum((x - x_proj) ** 2).item()


def monte_carlo_mse(x: torch.Tensor, P: torch.Tensor, P_pinv: torch.Tensor,
                     sigma: float, n_trials: int = 2000, seed: int = 0) -> float:
    """Estima E||x - P+ P (x + eta)||^2 promediando sobre n_trials realizaciones de ruido."""
    generator = torch.Generator().manual_seed(seed)
    T = x.shape[0]
    total_sq_error = 0.0
    for _ in range(n_trials):
        eta = torch.randn(T, dtype=torch.float64, generator=generator) * sigma
        y = P @ (x + eta)
        x_hat = P_pinv @ y
        total_sq_error += torch.sum((x - x_hat) ** 2).item()
    return total_sq_error / n_trials


def main():
    T = 240  # longitud de la secuencia temporal latente
    t = torch.arange(T, dtype=torch.float64)
    x = torch.sin(2 * torch.pi * t / T) + 0.1 * torch.cos(6 * torch.pi * t / T)

    block_sizes = [2, 4, 8, 12, 24]
    # sigma_extremo: deliberadamente mucho mas grande que la escala de x (que vive en [-1.1, 1.1])
    sigmas = [0.1, 1.0, 10.0, 100.0]  # 100.0 es el caso "ruido gaussiano extremo"

    print(f"{'k':>4} {'sigma':>8} {'sesgo_teo':>12} {'sesgo_medido':>14} "
          f"{'var_teo(=sigma^2*T)':>20} {'var_medida':>14} {'mse_teo':>10} {'mse_medido':>12}")

    for k in block_sizes:
        P = build_pooling_matrix(T, k)
        # Verificacion analitica: P+ deberia ser exactamente k * P^T para este P de bloques ortogonales
        P_pinv_lib = torch.linalg.pinv(P)
        P_pinv_formula = k * P.T
        max_diff = torch.max(torch.abs(P_pinv_lib - P_pinv_formula)).item()
        assert max_diff < 1e-8, f"La formula P+ = k P^T no coincide (diff={max_diff}) para k={k}"

        bias_sq = theoretical_bias(x, P, P_pinv_lib)
        trace_term = torch.trace(P_pinv_lib @ P_pinv_lib.T).item()  # deberia dar T, sin importar k

        for sigma in sigmas:
            var_teo = (sigma ** 2) * trace_term
            mse_teo = bias_sq + var_teo

            mse_medido = monte_carlo_mse(x, P, P_pinv_lib, sigma, n_trials=2000, seed=42)
            var_medida = mse_medido - bias_sq  # aislamos el aporte del ruido restando el sesgo teorico

            print(f"{k:>4} {sigma:>8.1f} {bias_sq:>12.4f} {bias_sq:>14.4f} "
                  f"{var_teo:>20.4f} {var_medida:>14.4f} {mse_teo:>10.4f} {mse_medido:>12.4f}")

        print(f"     -> tr(P+ P+^T) para k={k}: {trace_term:.4f}  (deberia ser T={T} para TODO k)")
        print("-" * 100)


if __name__ == "__main__":
    main()
