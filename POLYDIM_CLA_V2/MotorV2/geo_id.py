import torch
import torch.nn as nn

class GeoID_HIT(nn.Module):
    """
    Identidad Geométrica (GEO_ID) definida como un Tipo Inductivo Superior (HIT)
    en la Teoría de Tipos Homotópicos (HoTT).
    
    Asegura la Regla R10: El GEO_ID es un punto fijo bajo transformaciones permitidas.
    """
    def __init__(self, content_anchor: torch.Tensor, epsilon: float = 1e-4):
        super().__init__()
        # 0-esqueleto (Contenido 'C'): El hipervector base inmutable anclado al Codebook Universal.
        # Es la esencia del objeto (V en la terna S=(V,D,A))
        self.register_buffer("base_anchor", content_anchor.clone().detach())
        self.epsilon = epsilon
        
    def _0_skeleton(self):
        """Devuelve el ancla base inmutable (Contenido)."""
        return self.base_anchor
        
    def _1_skeleton_path_valid(self, transformation: torch.Tensor) -> bool:
        """
        1-esqueleto (Caminos): Verifica la Igualdad Homotópica.
        Un camino/transformación T es válido si preserva la identidad geométrica (Regla R10).
        dist(T(GEO_ID), GEO_ID) < epsilon
        """
        # Asumiendo T como una matriz de transformación ortogonal
        transformed_anchor = torch.matmul(self.base_anchor, transformation.T)
        distance = torch.norm(transformed_anchor - self.base_anchor, p=2)
        return distance.item() < self.epsilon
        
    def _2_skeleton_surface_consistency(self, T1, T2, T3) -> bool:
        """
        2-esqueleto (Superficies): Garantiza la consistencia de composición.
        T3 debe ser homotópicamente equivalente a (T2 ∘ T1).
        || T3(v) - T2(T1(v)) || < epsilon
        """
        path_a = torch.matmul(torch.matmul(self.base_anchor, T1.T), T2.T)
        path_b = torch.matmul(self.base_anchor, T3.T)
        surface_gap = torch.norm(path_a - path_b, p=2)
        return surface_gap.item() < self.epsilon

    def forward(self, T_applied):
        """
        Intenta aplicar una transformación al objeto.
        Si la transformación no es admitida por el 1-esqueleto (R10 violada),
        se colapsa (lógica intuicionista) o se fuerza una retracción al ancla base.
        """
        if not self._1_skeleton_path_valid(T_applied):
            # Proyección forzada para preservar el GEO_ID
            return self.base_anchor
        
        return torch.matmul(self.base_anchor, T_applied.T)
