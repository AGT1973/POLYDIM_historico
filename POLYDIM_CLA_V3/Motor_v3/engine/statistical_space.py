import torch
import torch.nn as nn
import torch.nn.functional as F

class StatisticalSpace(nn.Module):
    """
    Fase 2 y 6: El Espacio Euclidiano de Aprendizaje.
    Aquí es donde ocurre la destrucción y reconstrucción de información (Feature Extraction).
    A diferencia de la Fase Topológica, aquí NO se respeta la isometría.
    Utiliza activaciones no lineales (SwiGLU) y Descenso de Gradiente (Autograd).
    """
    def __init__(self, d_model: int, d_hidden: int = None):
        super().__init__()
        self.d_model = d_model
        # SwiGLU expansion factor (típicamente 8/3 o 4x de d_model)
        self.d_hidden = d_hidden if d_hidden else int(d_model * 2.666)
        
        # Proyecciones lineales para SwiGLU
        self.w1 = nn.Linear(d_model, self.d_hidden, bias=False)
        self.w2 = nn.Linear(self.d_hidden, d_model, bias=False)
        self.w3 = nn.Linear(d_model, self.d_hidden, bias=False)
        
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass usando la activación SwiGLU (State of the Art en LLMs).
        SwiGLU(x) = Swish(xW_1) ⊗ (xW_3) W_2
        
        Args:
            x: Tensor de entrada [N, D] o [Batch, N, D]
        Returns:
            Tensor transformado (espacio de características)
        """
        # Normalización estadística
        x_norm = self.layer_norm(x)
        
        # Activación SwiGLU
        hidden_swish = F.silu(self.w1(x_norm))
        hidden_linear = self.w3(x_norm)
        
        out = self.w2(hidden_swish * hidden_linear)
        
        # Conexión residual euclidiana
        return x + out

if __name__ == "__main__":
    print("=== POLYDIM V3: Statistical Space (Prueba de Aprendizaje Euclidiano) ===")
    torch.manual_seed(42)
    
    # 10 tokens en un espacio de 10,000D
    N, D = 10, 10000
    x_in = torch.randn(N, D)
    
    model = StatisticalSpace(d_model=D)
    
    # Simulación de forward pass
    x_out = model(x_in)
    
    norm_in = torch.norm(x_in[0]).item()
    norm_out = torch.norm(x_out[0]).item()
    
    print(f"Norma L2 Token 0 (Entrada): {norm_in:.6f}")
    print(f"Norma L2 Token 0 (Salida):  {norm_out:.6f}")
    print(f"Diferencia absoluta (Error): {abs(norm_in - norm_out):.6e}")
    if abs(norm_in - norm_out) > 1.0:
        print("[OK] Comprobado: La Fase Estadística destruye intencionalmente la isometría para aprender características.")
