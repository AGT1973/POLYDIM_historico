import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentALIGN(nn.Module):
    """
    Protocolo ALIGN: Comunicación latente-a-latente sin tokens.
    Utiliza el Algoritmo de Procrustes Ortogonal para mapear la geometría 
    del espacio latente del Emisor al Receptor usando el Codebook de Anclas.
    """
    def __init__(self, anchors_A, anchors_B):
        super().__init__()
        # Resolvemos Procrustes Ortogonal: min ||M * anchors_A - anchors_B||^2
        # M = U * V^T donde U, S, V = SVD(anchors_B^T * anchors_A)
        # Asumimos que anchors_A y anchors_B tienen forma (N_anchors, Dim)
        cross_cov = torch.matmul(anchors_B.transpose(-1, -2), anchors_A)
        U, _, Vh = torch.linalg.svd(cross_cov, full_matrices=False)
        # Matriz de proyección isométrica
        self.M = torch.matmul(U, Vh)
        self.M_inv = self.M.transpose(-1, -2) # Es ortogonal, así que M^T = M^-1

    def forward(self, T_emisor):
        """
        Integra la transformación externa en el receptor:
        T_aplicada = M * T_emisor * M_inv
        """
        return torch.matmul(self.M, torch.matmul(T_emisor, self.M_inv))

class DirichletConsensus(nn.Module):
    """
    Minimiza la Energía de Dirichlet (E_F) mediante la Ecuación del Calor
    para alcanzar el consenso global (clase H^0) en el enjambre.
    """
    def __init__(self, dt=0.01):
        super().__init__()
        self.dt = dt

    def forward(self, X, laplacian, steps=10):
        """
        X: Estados de los agentes (N_agents, Dim)
        laplacian: Laplaciano de Haces Celulares (N_agents, N_agents)
        """
        curr_X = X
        for _ in range(steps):
            # E_F(X) = <X, L X>. Gradiente es L * X
            # Ecuación del calor: dX/dt = -L * X
            dX = -torch.matmul(laplacian, curr_X)
            curr_X = curr_X + self.dt * dX
            
        # Calcular Energía de Dirichlet final: || d X ||^2 = trace(X^T L X)
        dirichlet_energy = torch.trace(torch.matmul(curr_X.transpose(-1, -2), torch.matmul(laplacian, curr_X)))
        return curr_X, dirichlet_energy

class SubobjectClassifier(nn.Module):
    """
    Topos de Grothendieck: Retracción a la fibra vacía (∅).
    Lógica Intuicionista: Si una ruta rompe restricciones lógicas (ej. energía infinita),
    el estado se colapsa al origen (∅) sin lanzar una excepción tradicional.
    """
    def __init__(self, max_norm_threshold=100.0):
        super().__init__()
        self.max_norm_threshold = max_norm_threshold

    def forward(self, x):
        norms = x.norm(p=2, dim=-1, keepdim=True)
        # Máscara booleana (Topos Subobject Classifier Ω)
        # 1 si es válido, 0 si rompe la topología
        is_valid = (norms < self.max_norm_threshold).float()
        # Retracción a la fibra vacía (cero)
        return x * is_valid
