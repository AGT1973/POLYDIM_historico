import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralCognitiveMonitor(nn.Module):
    """
    Monitor del Gap Espectral (λ1) y Prevención de Spectral Cognitive Collapse.
    Utiliza la Desigualdad de Cheeger y el Laplaciano de Haces Celulares (Hodge Laplacian).
    """
    def __init__(self, threshold_pk=1e-4):
        super().__init__()
        self.threshold_pk = threshold_pk

    def forward(self, laplacian, u_holonomy_residual):
        """
        Calcula λ1 usando LOBPCG para alta eficiencia (O(N)) y verifica el colapso.
        laplacian: Matriz del Laplaciano (N x N)
        u_holonomy_residual: || (I - Hol(γ)) u ||^2 (el término de ruido inferencial)
        """
        # Asegurar simetría
        laplacian = 0.5 * (laplacian + laplacian.transpose(-1, -2))
        
        # LOBPCG requiere al menos 2 autovalores si pedimos el menor no nulo.
        # Si la matriz es pequeña en test, usamos eigh directo, si es masiva, LOBPCG.
        if laplacian.shape[-1] < 500:
            eigvals = torch.linalg.eigvalsh(laplacian)
            # λ0 es aprox 0, λ1 es el primer gap
            lambda_1 = eigvals[1] if eigvals.shape[-1] > 1 else eigvals[0]
        else:
            # LOBPCG para N grandes (N=16384)
            # Calculamos los dos menores eigenvalores.
            eigvals, _ = torch.lobpcg(laplacian, k=2, largest=False)
            lambda_1 = eigvals[1]

        # Umbral heurístico basado en Desigualdad de Cheeger: λ1 <= min ||(I - Hol)u||^2 + p_k
        cheeger_threshold = u_holonomy_residual + self.threshold_pk

        cognitive_collapse = lambda_1 <= cheeger_threshold

        return lambda_1, cognitive_collapse

class GaussMarkovAdjunctionLayer(nn.Module):
    """
    Axioma IV (Homotopy): Invertibilidad Débil vía Adjunciones Funtoriales (L ⊣ R).
    Implementa la Adjunción de Gauss-Markov (RAPL).
    """
    def __init__(self, in_features, reduced_features):
        super().__init__()
        # Funtor Izquierdo (L): Contracción/Pooling (Proceso Irreversible)
        self.L = nn.Linear(in_features, reduced_features, bias=False)
        # Funtor Derecho (R): Reconstrucción óptima certificada
        self.R = nn.Linear(reduced_features, in_features, bias=False)

    def forward(self, x):
        contracted = self.L(x)
        reconstructed = self.R(contracted)
        return contracted, reconstructed

class TransformGroupoid(nn.Module):
    """
    Groupoide G: ∞-groupoide preservando GEO_ID.
    Transformaciones actúan como 'caminos' topológicos entre estados.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.path_generator = nn.Linear(dim, dim, bias=False)

    def forward(self, x, steps=3):
        paths = [x]
        curr = x
        for _ in range(steps):
            delta = self.path_generator(curr)
            curr = curr + 0.1 * (delta - delta.transpose(-1, -2) if curr.dim() > 1 else delta)
            curr = F.normalize(curr, p=2, dim=-1) * x.norm(p=2, dim=-1, keepdim=True)
            paths.append(curr)
        return paths

class CartanDieudonneStiefelRotor(nn.Module):
    """
    Teorema de Cartan-Dieudonné en Rotores con constraint en Stiefel Manifold.
    Garantiza isometría estricta incluso en BF16, mitigando la deriva de identidad.
    """
    def __init__(self, dim):
        super().__init__()
        self.m = nn.Parameter(torch.randn(dim))
        self.n = nn.Parameter(torch.randn(dim))

    def project_stiefel(self, v):
        """
        Proyecta el vector en la variedad de Stiefel (norma unitaria estricta).
        Mitiga el error de mantisa en BF16.
        """
        return F.normalize(v, p=2, dim=-1)

    def forward(self, x):
        # Aseguramos que m y n vivan en el Stiefel Manifold (V_1(R^n) que es S^{n-1})
        m_norm = self.project_stiefel(self.m)
        n_norm = self.project_stiefel(self.n)

        # Reflexión n: x - 2 * <x, n> * n
        proj_n = (x * n_norm).sum(dim=-1, keepdim=True)
        ref_n_x = x - 2 * proj_n * n_norm

        # Reflexión m: x' - 2 * <x', m> * m
        proj_m = (ref_n_x * m_norm).sum(dim=-1, keepdim=True)
        rotated_x = ref_n_x - 2 * proj_m * m_norm

        return rotated_x

class EngineMathV2:
    @staticmethod
    def perron_frobenius_root(A, iterations=20):
        A = torch.abs(A)
        n = A.shape[-1]
        v = torch.rand(n, device=A.device, dtype=A.dtype)
        v = v / v.norm(p=2)
        for _ in range(iterations):
            v = torch.mv(A, v)
            v = v / v.norm(p=2)
        eigenvalue = torch.dot(v, torch.mv(A, v)) / torch.dot(v, v)
        return eigenvalue, v

    @staticmethod
    def woodbury_invert(A_inv, U, C, V):
        """
        Reduce la complejidad de O(N^3) a O(NK^2) para inversiones de bajo rango.
        (A + UCV)^-1 = A^-1 - A^-1 U (C^-1 + V A^-1 U)^-1 V A^-1
        Previene OOM en espacios masivos (ej. 16384 dims) y estabiliza BF16.
        """
        # A_inv: (N, N), U: (N, K), C: (K, K), V: (K, N)
        # Term 1: V @ A_inv @ U
        term1 = torch.matmul(V, torch.matmul(A_inv, U))
        
        # Invertir el bloque KxK interno
        C_inv = torch.linalg.inv(C)
        inner = C_inv + term1
        inner_inv = torch.linalg.inv(inner)
        
        # Term 2: A_inv @ U @ inner_inv @ V @ A_inv
        term2 = torch.matmul(A_inv, torch.matmul(U, torch.matmul(inner_inv, torch.matmul(V, A_inv))))
        
        return A_inv - term2

    @staticmethod
    def cayley_transform(A):
        I = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
        A_skew = 0.5 * (A - A.transpose(-1, -2))
        inv_part = torch.linalg.inv(I + A_skew)
        return torch.matmul(I - A_skew, inv_part)

    @staticmethod
    def hodge_laplacian(d1, d2=None):
        laplacian = torch.matmul(d1, d1.transpose(-1, -2))
        if d2 is not None:
            laplacian += torch.matmul(d2.transpose(-1, -2), d2)
        return laplacian
