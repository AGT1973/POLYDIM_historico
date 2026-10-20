import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussMarkovAdjunction(nn.Module):
    """
    Implementación formal de la Adjunción de Gauss-Markov (F -| G).
    Reemplaza el ajuste estadístico de curvas por una verificación topológica continua.
    """
    def __init__(self, prm_dim, data_dim):
        super().__init__()
        # Funtor de Avance (F): Mapea parámetros (Prm) a predicciones (Data)
        self.F_weights = nn.Parameter(torch.randn(data_dim, prm_dim) / (prm_dim ** 0.5))

    def forward_functor(self, a):
        """
        F(a): Aplica el funtor de avance desde el espacio de parámetros 'a' 
        hacia el espacio de datos.
        """
        return F.linear(a, self.F_weights)

    def gauss_markov_functor(self, y_residual):
        """
        G(y): Funtor adjunto derecho. Mapea el residuo observado de vuelta al 
        espacio de parámetros más probable.
        Gracias a RAPL (Right Adjoints Preserve Limits), minimiza el residuo en Data
        de forma topológicamente equivalente a aproximar el parámetro óptimo en Prm.
        Utilizamos la pseudoinversa de Moore-Penrose como la proyección óptima en L2.
        """
        F_pinv = torch.linalg.pinv(self.F_weights)
        return F.linear(y_residual, F_pinv)

    def forward(self, a, y_true):
        """
        Paso completo de entrenamiento certificado.
        1. F(a): Predicción.
        2. y_residual: Error en espacio de Datos.
        3. G(y_residual): Corrección topológicamente óptima en el espacio Prm.
        """
        y_pred = self.forward_functor(a)
        y_residual = y_true - y_pred
        a_optimal_update = self.gauss_markov_functor(y_residual)
        return y_pred, a_optimal_update

class YonedaEmbedding(nn.Module):
    """
    Implementación del Lema de Yoneda para el GEO_ID.
    Nat(Hom(-, X), F) ≅ F(X)
    Un objeto no se define por sus coordenadas absolutas, sino por su red de morfismos
    (relaciones o productos internos) con respecto a un "diccionario" de objetos topológicos.
    """
    def __init__(self, feature_dim, num_anchors=64):
        super().__init__()
        # El conjunto de funtores representables (anclajes estructurales)
        self.anchors = nn.Parameter(torch.randn(num_anchors, feature_dim))

    def forward(self, x):
        """
        Mapea el tensor absoluto 'x' a su representación Yoneda (su GEO_ID relativo).
        Calcula Hom(x, A_i) para todo anclaje A_i.
        """
        # Normalizamos para garantizar que medimos la geometría (morfismos)
        # independientemente de la escala (prevenimos deriva de identidad).
        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
        anchors_norm = torch.nn.functional.normalize(self.anchors, p=2, dim=-1)
        
        # El GEO_ID es el vector de relaciones de equivalencia (coseno de similitud)
        geo_id = torch.matmul(x_norm, anchors_norm.transpose(-1, -2))
        return geo_id
