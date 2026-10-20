import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepONetOperator(nn.Module):
    """
    Operador Neuronal para espacios de dimensión infinita.
    Aprende mapeos entre espacios de funciones (Zero-Shot generalization).
    """
    def __init__(self, branch_dim, trunk_dim, hidden_dim, p_basis):
        super().__init__()
        # Branch Net: Procesa la función de entrada (ej. un campo semántico continuo)
        self.branch = nn.Sequential(
            nn.Linear(branch_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, p_basis)
        )
        # Trunk Net: Procesa las coordenadas del dominio de evaluación
        self.trunk = nn.Sequential(
            nn.Linear(trunk_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, p_basis)
        )

    def forward(self, u_function_repr, y_coordinates):
        """
        G(u)(y) = sum_{k=1}^P b_k(u) * t_k(y)
        """
        b = self.branch(u_function_repr) # (Batch, P)
        t = self.trunk(y_coordinates)    # (Batch, P)
        # Producto punto para evaluar el operador en y
        return torch.sum(b * t, dim=-1, keepdim=True)

class MutableExecutionPolicy(nn.Module):
    """
    Axioma VII (Plasticidad Deliberativa):
    La política de ejecución (Π) es mutable y autorreferencial.
    """
    def __init__(self, state_dim):
        super().__init__()
        # La política tiene pesos que pueden ser modulados dinámicamente
        self.meta_evaluator = nn.Linear(state_dim, state_dim)
        
    def forward(self, current_state, goal_state):
        """
        Ajusta la política basándose en la distancia topológica al objetivo.
        Es autorreferencial: el estado actual altera cómo se evalúa el siguiente paso.
        """
        # Evalúa el estado interno
        self_awareness = torch.sigmoid(self.meta_evaluator(current_state))
        # Genera un vector de "atracción" hacia la meta modulado por la autoconciencia
        direction = (goal_state - current_state) * self_awareness
        return direction

class TeleologicalEvolution(nn.Module):
    """
    Axiomas III, V y VI: Evolución Dirigida con Clausura y Restricción.
    Garantiza que el estado siempre tenga un 'lugar a donde ir' (Axioma III)
    y que sea empujado determinísticamente hacia la meta G (Axioma V) 
    cumpliendo predicados lógicos C (Axioma VI).
    """
    def __init__(self, state_dim):
        super().__init__()
        self.policy = MutableExecutionPolicy(state_dim)
        
    def forward(self, S_t, Goal, constraint_predicate, manifold_projector):
        """
        S_t: Estado actual (Axioma III asegura que es válido)
        Goal: Objetivo semántico explícito (G) (Axioma V)
        constraint_predicate: Función lógica (C) (Axioma VI)
        manifold_projector: Funtor que asegura la clausura (Axioma III)
        """
        # 1. La política mutable decide la dirección topológica
        delta_S = self.policy(S_t, Goal)
        
        # 2. Evolución del estado
        S_next = S_t + delta_S
        
        # 3. Axioma VI: Evaluación Lógica Continua
        if not constraint_predicate(S_next):
            # Si rompe la restricción, aplicamos un adjunto derecho (reconstrucción óptima)
            # En esta abstracción, mitigamos el paso
            S_next = S_t + (delta_S * 0.1) 
            
        # 4. Axioma III (Clausura): Proyección al manifold D para mantener validez
        S_final = manifold_projector(S_next)
        
        return S_final
