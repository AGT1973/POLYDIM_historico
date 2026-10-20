"""
Tarea 55 - POLYDIM
CodebookLinear: reemplazo de nn.Linear via cuantizacion vectorial de pesos
("Anclaje de Codebook"), con evaluacion empirica de si permite recuperar
la identidad original de cada fila de pesos ("GEO_ID").

Ver tarea_55.md para el desarrollo matematico completo.

Dependencias:
    pip install torch numpy scikit-learn --break-system-packages
"""

import numpy as np
import torch
import torch.nn as nn

try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def simple_kmeans(X, k, n_iter=100, seed=0):
    """K-means minimo en numpy, usado solo si sklearn no esta disponible."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.choice(n, size=min(k, n), replace=False)
    centers = X[idx].copy()
    for _ in range(n_iter):
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        assign = np.argmin(dists, axis=1)
        new_centers = centers.copy()
        for c in range(centers.shape[0]):
            mask = assign == c
            if mask.any():
                new_centers[c] = X[mask].mean(axis=0)
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return centers, assign


class CodebookLinear(nn.Module):
    """Reemplazo de nn.Linear que almacena los pesos como un codebook
    (K vectores) + un indice de anclaje por fila de salida, en vez de la
    matriz de pesos completa. Ver tarea_55.md, Seccion 3.1."""

    def __init__(self, in_features, out_features, num_codewords, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_codewords = num_codewords

        self.codebook = nn.Parameter(torch.randn(num_codewords, in_features) * 0.02)
        self.register_buffer("assignments", torch.zeros(out_features, dtype=torch.long))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear_layer: nn.Linear, num_codewords: int, seed: int = 0):
        """Comprime una capa nn.Linear ya entrenada via k-means sobre sus filas."""
        W = linear_layer.weight.detach().cpu().numpy()  # (out_features, in_features)
        out_features, in_features = W.shape
        num_codewords = min(num_codewords, out_features)

        if HAS_SKLEARN:
            km = KMeans(n_clusters=num_codewords, n_init=10, random_state=seed).fit(W)
            centers, assign = km.cluster_centers_, km.labels_
        else:
            centers, assign = simple_kmeans(W, num_codewords, seed=seed)

        module = cls(in_features, out_features, num_codewords,
                      bias=linear_layer.bias is not None)
        with torch.no_grad():
            module.codebook.copy_(torch.tensor(centers, dtype=torch.float32))
            module.assignments.copy_(torch.tensor(assign, dtype=torch.long))
            if linear_layer.bias is not None:
                module.bias.copy_(linear_layer.bias.detach())
        return module

    def reconstruct_weight(self):
        """Devuelve W_hat = C[a], la matriz de pesos dequantizada."""
        return self.codebook[self.assignments]  # (out_features, in_features)

    def forward(self, x):
        W_hat = self.reconstruct_weight()
        return torch.nn.functional.linear(x, W_hat, self.bias)


def evaluar_compresion(original: nn.Linear, comprimida: CodebookLinear, n_test=256):
    """Calcula las 3 metricas estandar de la Seccion 3.2 del .md."""
    W = original.weight.detach().cpu().numpy()
    W_hat = comprimida.reconstruct_weight().detach().cpu().numpy()

    # 1. Error de reconstruccion L2 relativo
    eps_recon = np.linalg.norm(W - W_hat) / (np.linalg.norm(W) + 1e-12)

    # 2. Fidelidad funcional sobre entradas de prueba aleatorias
    x_test = torch.randn(n_test, original.in_features)
    with torch.no_grad():
        y_orig = original(x_test)
        y_comp = comprimida(x_test)
    eps_func = (torch.norm(y_orig - y_comp, dim=1) /
                (torch.norm(y_orig, dim=1) + 1e-12)).mean().item()

    # 3. Identificabilidad: para cada codeword usado, el vecino mas cercano
    # entre TODAS las filas originales, es la fila que genero ese codeword?
    assignments = comprimida.assignments.numpy()
    correct = 0
    for i in range(W.shape[0]):
        c = W_hat[i]  # codeword asignado a la fila i
        dists = np.linalg.norm(W - c[None, :], axis=1)
        nearest = np.argmin(dists)
        if nearest == i:
            correct += 1
    acc_ident = correct / W.shape[0]

    return eps_recon, eps_func, acc_ident


def main():
    torch.manual_seed(0)
    d_in, d_out = 64, 128

    original = nn.Linear(d_in, d_out)

    print(f"{'K':>6} | {'eps_recon':>10} | {'eps_func':>10} | {'acc_ident':>10}")
    print("-" * 46)
    for K in [4, 8, 16, 32, 64, 96, 128]:
        comprimida = CodebookLinear.from_linear(original, num_codewords=K)
        eps_recon, eps_func, acc_ident = evaluar_compresion(original, comprimida)
        print(f"{K:>6} | {eps_recon:>10.4f} | {eps_func:>10.4f} | {acc_ident:>10.4f}")

    print("\n=== Conclusion ===")
    print("Se espera (y hay que verificar corriendo esto) que acc_ident < 1 para")
    print(f"K < d_out={d_out}, y que acc_ident -> 1 solo cuando K se acerca a d_out")
    print("(es decir, cuando casi no hay compresion). Esto confirma que 'recuperar")
    print("la identidad original' NO es una propiedad automatica del anclaje por")
    print("codebook: es un mapa muchos-a-uno y la informacion se pierde por diseno")
    print("cuando K << d_out (principio del palomar).")


if __name__ == "__main__":
    main()
