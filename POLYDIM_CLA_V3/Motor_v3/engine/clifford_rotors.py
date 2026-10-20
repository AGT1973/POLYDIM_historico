import torch

class CliffordRotors:
    """
    Fase 3: Preservación Geométrica (Álgebra de Clifford).
    Implementa rotaciones estrictas mediante reflexiones de Householder
    para garantizar que las operaciones de atención/routing sean Isometrías (norma constante).
    """
    def __init__(self, d_model: int):
        self.d_model = d_model

    def householder_reflection(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Aplica una reflexión de Householder al vector x a través del plano ortogonal al vector normal v.
        R(x) = x - 2 * (x·v) / (v·v) * v
        
        Args:
            x: Tensor de entrada de shape [N, D]
            v: Tensor normal de shape [N, D]
        Returns:
            Tensor reflejado de shape [N, D] con exactamente la misma norma L2.
        """
        # Calcular v·v para cada vector en el batch (shape [N, 1])
        v_squared_norm = torch.sum(v * v, dim=-1, keepdim=True)
        
        # Evitar división por cero si algún vector normal es nulo
        epsilon = 1e-8
        v_squared_norm = torch.clamp(v_squared_norm, min=epsilon)
        
        # Calcular x·v (shape [N, 1])
        dot_product = torch.sum(x * v, dim=-1, keepdim=True)
        
        # Aplicar la fórmula de reflexión
        reflection = x - 2 * (dot_product / v_squared_norm) * v
        return reflection

    def apply_rotor(self, x: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """
        Un Rotor de Clifford es el producto par de reflexiones.
        Aplicamos dos reflexiones sucesivas definidas por los hiperplanos v1 y v2.
        Esto resulta en una rotación pura en el plano definido por v1 y v2,
        preservando el 100% de la longitud (isometría).
        """
        # Primera reflexión
        x_ref1 = self.householder_reflection(x, v1)
        # Segunda reflexión
        x_rotated = self.householder_reflection(x_ref1, v2)
        return x_rotated

if __name__ == "__main__":
    print("=== POLYDIM V3: Clifford Rotors (Prueba de Isometría) ===")
    torch.manual_seed(42)
    N, D = 10, 10000
    
    # Vector de estado inicial (simulando 10 tokens en 10000D proyectados en la hiperesfera)
    x = torch.randn(N, D)
    x = x / torch.norm(x, dim=-1, keepdim=True)  
    
    # Parámetros "aprendidos" por la capa estadística que dictarán el plano de rotación
    v1 = torch.randn(N, D)
    v2 = torch.randn(N, D)
    
    rotor_engine = CliffordRotors(d_model=D)
    x_out = rotor_engine.apply_rotor(x, v1, v2)
    
    norm_in = torch.norm(x[0]).item()
    norm_out = torch.norm(x_out[0]).item()
    
    print(f"Norma L2 Token 0 (Entrada): {norm_in:.6f}")
    print(f"Norma L2 Token 0 (Salida):  {norm_out:.6f}")
    print(f"Diferencia absoluta (Error): {abs(norm_in - norm_out):.6e}")
    if abs(norm_in - norm_out) < 1e-6:
        print("[OK] Isometría estricta preservada mediante álgebra de Clifford.")
    else:
        print("[ERROR] Falla en la isometría.")
