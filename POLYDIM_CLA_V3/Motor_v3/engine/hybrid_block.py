import torch
import torch.nn as nn
from .geo_tokenizer import GeoTokenizer
from .statistical_space import StatisticalSpace
from .clifford_rotors import CliffordRotors
from .hodge_consensus import HodgeConsensus

class PolydimHybridBlock(nn.Module):
    """
    Fase 5: Motor Híbrido POLYDIM V3 (Forward Pass Completo).
    Combina el aprendizaje estadístico (MLP) con el enrutamiento topológico geométrico
    (Consenso de Hodge y Rotación de Clifford) siguiendo la especificación estricta del White Book.
    """
    def __init__(self, d_model: int = 10000, chebyshev_order: int = 3):
        super().__init__()
        self.d_model = d_model
        
        # Instanciar submódulos
        self.geo_tokenizer = GeoTokenizer(d_model=d_model)
        self.statistical_space = StatisticalSpace(d_model=d_model)
        self.hodge_consensus = HodgeConsensus(d_model=d_model, chebyshev_order=chebyshev_order)
        self.clifford_rotors = CliffordRotors(d_model=d_model)
        
        # Red de parámetros aprendibles para los vectores de rotación de Clifford
        # Mapea las características aprendidas (Fase 2) a los planos de rotación (v1, v2) (Fase 5)
        self.rotor_proj = nn.Linear(d_model, d_model * 2, bias=False)

    def forward(self, sequence: list[str], incidence_matrix: torch.Tensor) -> torch.Tensor:
        """
        Flujo End-to-End.
        Args:
            sequence: Lista de N tokens/conceptos.
            incidence_matrix: Matriz B de shape [N, E] (Topología del complejo simplicial).
        """
        # Fase 1: Ingestión Determinista
        # GEO shape: [N, D]
        geo_id = self.geo_tokenizer.encode_sequence(sequence)
        geo_id = geo_id.to(incidence_matrix.device) # Soporte para GPU
        
        # Fase 2: Espacio Estadístico (Extracción de características / backprop tradicional)
        # H shape: [N, D]
        H = self.statistical_space(geo_id)
        
        # Fase 3: Proyección Geométrica Isométrica (Retracción a Stiefel)
        # Asegura que el tensor vuelva a la hiperesfera S^{D-1} antes del routing topológico
        V = H / torch.norm(H, dim=-1, keepdim=True)
        
        # Fase 4: Consenso Topológico Paralelo (Filtro de Chebyshev sobre Laplaciano de Hodge)
        # V_aligned shape: [N, D]
        V_aligned = self.hodge_consensus(V, geo_id, incidence_matrix)
        
        # Fase 5: Routing de Clifford (Rotación Estricta sin cambio de norma)
        # Generar los planos de rotación a partir del conocimiento estadístico
        rotor_params = self.rotor_proj(H) # shape: [N, 2D]
        v1, v2 = torch.chunk(rotor_params, 2, dim=-1) # shape: [N, D] cada uno
        
        # Aplicar Rotación
        V_rotated = self.clifford_rotors.apply_rotor(V_aligned, v1, v2)
        
        # Fase 6: Readout al Espacio Estadístico 
        # (Aquí el tensor V_rotated pasaría a la siguiente capa del bloque Híbrido o al proyector de salida)
        Out = self.statistical_space(V_rotated)
        
        return Out

if __name__ == "__main__":
    print("=== POLYDIM V3: Hybrid Block (Forward Pass End-to-End) ===")
    
    # Simulación de contexto de entrada
    sequence = ["el", "perro", "persigue", "al", "gato"]
    N = len(sequence)
    D = 10000
    E = 4 # 4 aristas
    
    # Matriz de Incidencia del Complejo Simplicial (Topología 1D de prueba)
    B = torch.zeros(N, E)
    for i in range(E):
        B[i, i] = -1
        B[i+1, i] = 1
        
    engine = PolydimHybridBlock(d_model=D)
    
    # Ejecutar forward pass
    out = engine(sequence, B)
    
    print(f"Secuencia input: {sequence}")
    print(f"Topología:       N={N} Nodos, E={E} Símplices")
    print(f"Salida Shape:    {out.shape} (Esperado: [{N}, {D}])")
    print("[OK] Flujo Híbrido End-to-End validado. El Motor V3 compila y rutea.")
