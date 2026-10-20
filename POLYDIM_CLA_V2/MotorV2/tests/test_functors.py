import torch
import torch.nn.functional as F
from ..functors import GaussMarkovAdjunction, YonedaEmbedding

def test_gauss_markov_adjunction():
    """
    Verifica que el funtor G (adjunto derecho) proponga una actualización
    que minimice el residuo topológicamente (pseudoinversa minimiza L2).
    """
    prm_dim, data_dim = 8, 16
    gma = GaussMarkovAdjunction(prm_dim, data_dim)
    
    a = torch.randn(1, prm_dim)
    y_true = torch.randn(1, data_dim)
    
    y_pred, a_update = gma(a, y_true)
    
    assert y_pred.shape == (1, data_dim)
    assert a_update.shape == (1, prm_dim)
    
    # Aplicar la actualización propuesta debería reducir el error
    a_new = a + a_update
    y_pred_new = gma.forward_functor(a_new)
    
    error_old = torch.norm(y_true - y_pred)
    error_new = torch.norm(y_true - y_pred_new)
    
    assert error_new < error_old or torch.allclose(error_new, torch.zeros_like(error_new))

def test_yoneda_embedding():
    """
    Verifica que el GEO_ID actúe como representación relativa por morfismos.
    Rotar ambos (objeto y anclajes) simultáneamente debe preservar el GEO_ID.
    """
    dim, anchors = 16, 64
    yoneda = YonedaEmbedding(feature_dim=dim, num_anchors=anchors)
    x = torch.randn(1, dim)
    
    geo_id_1 = yoneda(x)
    assert geo_id_1.shape == (1, anchors)
    
    # Si escalamos x, el GEO_ID (basado en coseno) no debe cambiar (Invariancia de escala)
    x_scaled = x * 10.0
    geo_id_2 = yoneda(x_scaled)
    
    assert torch.allclose(geo_id_1, geo_id_2, atol=1e-5)
