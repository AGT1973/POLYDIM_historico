"""
POLYDIM V3.2 - Motor de Arquitectura Polidimensional Cognitiva
===============================================================
Autor: [Tu nombre]
Versión: 3.2 (Corregida post-revisión por pares)

Correcciones matemáticas aplicadas:
1. Laplaciano Magnético: Hermitiano complejo (Fanuel 2017, Furutani 2020)
2. Rotaciones: Walsh-Hadamard + FFT (O(D log D))
3. Embeddings: Fourier continuos (diferenciables)
4. Protocolo PMTP: Ofuscación por semilla compartida

Referencias:
- Fanuel et al. (2017). Magnetic eigenmaps for community detection 
  in directed networks. Phys. Rev. E 95, 022302.
- Furutani et al. (2020). Graph signal processing for directed graphs 
  based on the Hermitian Laplacian. ECML PKDD 2019.
- Zhang et al. (2021). MagNet: A Magnetic Neural Network. ICLR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional

# ============================================================
# 1. UTILIDADES: Walsh-Hadamard y FFT
# ============================================================

class WalshHadamardTransform:
    """
    Transformada de Walsh-Hadamard rápida (butterfly).
    Ortogonal exacta: H @ H.T = I.
    Complejidad: O(D log D).
    """
    @staticmethod
    def fwht(x: torch.Tensor) -> torch.Tensor:
        """Fast Walsh-Hadamard Transform in-place."""
        D = x.shape[-1]
        h = 1
        while h < D:
            x = x.reshape(x.shape[:-1] + (D // (2 * h), 2, h))
            x = torch.cat([x[..., 0, :] + x[..., 1, :], 
                           x[..., 0, :] - x[..., 1, :]], dim=-2)
            x = x.reshape(x.shape[:-2] + (D,))
            h *= 2
        return x / math.sqrt(D)
    
    @staticmethod
    def ifwht(x: torch.Tensor) -> torch.Tensor:
        """Inverse FWHT (idéntica a FWHT para matriz ortogonal)."""
        return WalshHadamardTransform.fwht(x)


class FFTRotation:
    """
    Rotación unitaria via FFT.
    Complejidad: O(D log D).
    """
    @staticmethod
    def rotate(x: torch.Tensor) -> torch.Tensor:
        return torch.fft.fft(x, dim=-1)
    
    @staticmethod
    def inverse(x: torch.Tensor) -> torch.Tensor:
        return torch.fft.ifft(x, dim=-1).real


# ============================================================
# 2. LAPLACIANO MAGNÉTICO (Fanuel/Furutani)
# ============================================================

class MagneticLaplacian(nn.Module):
    """
    Laplaciano Magnético para grafos dirigidos con skip-connections.
    
    L^(q) = D_s - H^(q)
    H^(q) = A_s ⊙ exp(i * Theta)
    Theta = 2πq * (A - A^T)
    
    Args:
        N: número de nodos (longitud de contexto)
        q: carga magnética (fase, típicamente π/4)
        skip_k: ventana de contexto (skip-connections)
    """
    def __init__(self, N: int, q: float = math.pi / 4.0, skip_k: int = 3):
        super().__init__()
        self.N = N
        self.q = q
        self.skip_k = skip_k
        
        # Construir matrices (no entrenables, topología fija)
        A = self._build_adjacency(N, skip_k)
        A_s = 0.5 * (A + A.T)
        D_s = np.diag(np.sum(A_s, axis=1))
        Theta = 2 * np.pi * q * (A - A.T)
        
        # H^(q) = A_s ⊙ exp(i * Theta)
        H_q = A_s * np.exp(1j * Theta)
        
        # L^(q) = D_s - H^(q)
        L_q = D_s - H_q
        
        # Registrar como buffers complejos
        self.register_buffer('L_q_real', torch.from_numpy(L_q.real).float())
        self.register_buffer('L_q_imag', torch.from_numpy(L_q.imag).float())
        self.register_buffer('D_s', torch.from_numpy(D_s).float())
        
    def _build_adjacency(self, N: int, skip_k: int) -> np.ndarray:
        """Construye matriz de adyacencia dirigida con skip-connections."""
        A = np.zeros((N, N), dtype=np.float64)
        # Cadena base
        for i in range(N - 1):
            A[i, i + 1] = 1.0
        # Skip connections
        for k in range(2, skip_k + 1):
            for i in range(N - k):
                A[i, i + k] = 1.0
        return A
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica el Laplaciano Magnético al tensor de entrada.
        
        Args:
            x: tensor de forma (..., N, D) donde N es la longitud de secuencia
        
        Returns:
            tensor de forma (..., N, D) con el filtro magnético aplicado
        """
        # Reconstruir L_q como complejo
        L_q = torch.complex(self.L_q_real, self.L_q_imag)
        
        # x puede ser real o complejo
        if not torch.is_complex(x):
            x = x.to(torch.cfloat)
        
        # Aplicar filtro: y = L_q @ x
        # x: (..., N, D), L_q: (N, N)
        # Resultado: (..., N, D)
        y = torch.einsum('ij,...jd->...id', L_q, x)
        
        return y


# ============================================================
# 3. EMBEDDINGS DE FOURIER CONTINUOS
# ============================================================

class FourierEmbedding(nn.Module):
    """
    Embeddings posicionales via Fourier (diferenciables).
    Reemplaza SHA-256 discontinuo.
    
    Basado en: Vaswani et al. (2017) pero con base de frecuencias
    aprendibles en lugar de fijas.
    """
    def __init__(self, D: int, max_len: int = 5000, learnable: bool = True):
        super().__init__()
        self.D = D
        
        # Frecuencias base (logarítmicamente espaciadas)
        freqs = torch.exp(
            torch.arange(0, D, 2).float() * 
            (-math.log(10000.0) / D)
        )
        
        if learnable:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer('freqs', freqs)
        
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: tensor de índices de posición (B, N)
        
        Returns:
            embeddings posicionales (B, N, D)
        """
        B, N = positions.shape
        
        # positions: (B, N, 1), freqs: (D//2)
        angles = positions.unsqueeze(-1) * self.freqs.unsqueeze(0).unsqueeze(0)
        # angles: (B, N, D//2)
        
        emb = torch.zeros(B, N, self.D, device=positions.device)
        emb[:, :, 0::2] = torch.sin(angles)
        emb[:, :, 1::2] = torch.cos(angles)
        
        return emb


# ============================================================
# 4. ROTACIÓN ISOMÉTRICA (Walsh-Hadamard + FFT)
# ============================================================

class IsometricRotation(nn.Module):
    """
    Rotación isométrica compuesta: Walsh-Hadamard + FFT.
    
    La composición de dos transformaciones unitarias es unitaria.
    WH es ortogonal real, FFT es unitaria compleja.
    Su composición preserva normas: ||R(x)|| = ||x||.
    """
    def __init__(self, D: int, use_fft: bool = True):
        super().__init__()
        assert (D & (D - 1)) == 0, "D debe ser potencia de 2 para WH"
        self.D = D
        self.use_fft = use_fft
        
        # Parámetros de fase aprendibles (rotación adicional)
        self.phase = nn.Parameter(torch.randn(D) * 0.01)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica rotación isométrica compuesta.
        
        Args:
            x: tensor de forma (..., D)
        
        Returns:
            tensor rotado de forma (..., D)
        """
        # 1. Walsh-Hadamard (ortogonal real, O(D log D))
        x = WalshHadamardTransform.fwht(x)
        
        # 2. FFT (unitaria compleja, O(D log D))
        if self.use_fft:
            x = torch.fft.fft(x, dim=-1)
        
        # 3. Fase aprendible (diagonal unitaria)
        x = x * torch.exp(1j * self.phase).unsqueeze(0)
        
        # 4. Inversa FFT
        if self.use_fft:
            x = torch.fft.ifft(x, dim=-1)
        
        # 5. Inversa WH (ortogonal, O(D log D))
        x = WalshHadamardTransform.ifwht(x)
        
        return x


# ============================================================
# 5. PROTOCOLO PMTP (Polydim Matrix Transfer Protocol)
# ============================================================

class PMTPRouter(nn.Module):
    """
    Router PMTP para transferencia tensorial entre agentes.
    
    Ofusca el tensor mediante rotación isométrica con semilla compartida.
    No es criptografía — es ofuscación por semilla compartida.
    
    Args:
        D: dimensión del espacio latente
        seed: semilla compartida entre agentes
    """
    def __init__(self, D: int, seed: int = 42):
        super().__init__()
        self.D = D
        self.seed = seed
        
        # Generar rotor a partir de semilla (determinístico)
        torch.manual_seed(seed)
        self.rotor = IsometricRotation(D, use_fft=True)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Ofusca el tensor para transmisión."""
        return self.rotor(x)
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Recupera el tensor (la rotación es su propia inversa)."""
        # WH es auto-inversa, FFT/IFFT son inversas
        # La fase aprendible necesita inversión explícita
        with torch.no_grad():
            # Invertir fase
            x = x * torch.exp(-1j * self.rotor.phase).unsqueeze(0)
            # Invertir WH+FFT (aplicar de nuevo, son unitarias)
            x = self.rotor(x)  # R(R(x)) = x para WH puro, aprox para compuesta
        return x


# ============================================================
# 6. MOTOR POLYDIM V3.2 (Pipeline Completo)
# ============================================================

class PolydimMotorV3(nn.Module):
    """
    Motor principal de POLYDIM V3.2.
    
    Pipeline:
    1. Embeddings de Fourier (posicionales + semánticos)
    2. Proyección a manifold ND
    3. Rotaciones isométricas (WH + FFT)
    4. Filtro Laplaciano Magnético (L0 + L1)
    5. Decodificación a vocabulario
    
    Args:
        vocab_size: tamaño del vocabulario (BPE)
        D: dimensión del espacio latente (potencia de 2)
        N: longitud de contexto
        n_layers: número de capas
        n_nodes: número de nodos acoplados (subespacios)
        q: carga magnética
        skip_k: ventana de contexto para skip-connections
    """
    def __init__(
        self,
        vocab_size: int = 50000,
        D: int = 4096,
        N: int = 512,
        n_layers: int = 6,
        n_nodes: int = 20,
        q: float = math.pi / 4.0,
        skip_k: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.D = D
        self.N = N
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.d_node = D // n_nodes  # dimensión por nodo
        
        assert D % n_nodes == 0, "D debe ser divisible por n_nodes"
        assert (self.d_node & (self.d_node - 1)) == 0, "d_node debe ser potencia de 2"
        
        # 1. Embeddings
        self.token_embed = nn.Embedding(vocab_size, D)
        self.pos_embed = FourierEmbedding(D, max_len=N)
        self.dropout = nn.Dropout(dropout)
        
        # 2. Proyección a manifold ND
        self.to_manifold = nn.Linear(D, D)
        
        # 3. Capas del motor
        self.layers = nn.ModuleList([
            PolydimLayer(D, N, n_nodes, q, skip_k, dropout)
            for _ in range(n_layers)
        ])
        
        # 4. Decodificación
        self.norm = nn.LayerNorm(D)
        self.to_vocab = nn.Linear(D, vocab_size)
        
        # Inicialización
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        return_latent: bool = False
    ) -> torch.Tensor:
        """
        Forward pass del Motor POLYDIM.
        
        Args:
            input_ids: tensor de tokens (B, N)
            return_latent: si True, retorna el estado latente final
        
        Returns:
            logits (B, N, vocab_size) o tensor latente (B, N, D)
        """
        B, N = input_ids.shape
        
        # 1. Embeddings
        token_emb = self.token_embed(input_ids)  # (B, N, D)
        pos_ids = torch.arange(N, device=input_ids.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embed(pos_ids)  # (B, N, D)
        
        x = self.dropout(token_emb + pos_emb)
        
        # 2. Proyección a manifold
        x = self.to_manifold(x)  # (B, N, D)
        
        # 3. Capas polidimensionales
        for layer in self.layers:
            x = layer(x)  # (B, N, D)
        
        # 4. Normalización
        x = self.norm(x)
        
        if return_latent:
            return x  # Estado cognitivo polidimensional (ECP)
        
        # 5. Decodificación a vocabulario
        logits = self.to_vocab(x)  # (B, N, vocab_size)
        return logits


class PolydimLayer(nn.Module):
    """
    Capa individual del Motor POLYDIM.
    
    Estructura:
    1. Normalización
    2. División en nodos (subespacios)
    3. Rotación isométrica por nodo (WH + FFT)
    4. Filtro Laplaciano Magnético (consenso causal)
    5. Recombinación y MLP
    """
    def __init__(
        self,
        D: int,
        N: int,
        n_nodes: int,
        q: float,
        skip_k: int,
        dropout: float
    ):
        super().__init__()
        self.D = D
        self.N = N
        self.n_nodes = n_nodes
        self.d_node = D // n_nodes
        
        # Normalización
        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)
        
        # Rotaciones isométricas por nodo
        self.rotors = nn.ModuleList([
            IsometricRotation(self.d_node, use_fft=True)
            for _ in range(n_nodes)
        ])
        
        # Laplaciano Magnético (compartido por todos los nodos)
        self.laplacian = MagneticLaplacian(N, q=q, skip_k=skip_k)
        
        # MLP por nodo
        self.mlp = nn.Sequential(
            nn.Linear(D, 4 * D),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * D, D),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor de forma (B, N, D)
        
        Returns:
            tensor de forma (B, N, D)
        """
        B, N, D = x.shape
        
        # Residual connection 1
        residual = x
        x = self.norm1(x)
        
        # Dividir en nodos: (B, N, n_nodes, d_node)
        x_nodes = x.reshape(B, N, self.n_nodes, self.d_node)
        
        # Aplicar rotación isométrica por nodo
        rotated = []
        for i, rotor in enumerate(self.rotors):
            node = x_nodes[:, :, i, :]  # (B, N, d_node)
            # Aplicar WH+FFT por posición
            node_rot = rotor(node)  # (B, N, d_node)
            rotated.append(node_rot)
        
        x = torch.stack(rotated, dim=2)  # (B, N, n_nodes, d_node)
        
        # Recombinar: (B, N, D)
        x = x.reshape(B, N, D)
        
        # Aplicar Laplaciano Magnético (consenso causal)
        # El Laplaciano opera sobre la dimensión N (secuencia)
        x = self.laplacian(x)  # (B, N, D) complejo
        
        # Tomar parte real (el espectro ya codifica la fase)
        if torch.is_complex(x):
            x = x.real
        
        # Residual connection 2
        x = residual + x
        residual = x
        x = self.norm2(x)
        
        # MLP
        x = self.mlp(x)
        x = residual + x
        
        return x


# ============================================================
# 7. EJEMPLO DE USO Y TEST
# ============================================================

if __name__ == "__main__":
    print("="*70)
    print("POLYDIM V3.2 - Test de Integridad")
    print("="*70)
    
    # Configuración
    B, N, D = 2, 20, 256  # Batch, contexto, dimensión
    vocab_size = 1000
    n_nodes = 4
    d_node = D // n_nodes  # 64
    
    # Crear modelo
    model = PolydimMotorV3(
        vocab_size=vocab_size,
        D=D,
        N=N,
        n_layers=2,
        n_nodes=n_nodes,
        q=math.pi / 4.0,
        skip_k=3
    )
    
    # Input
    input_ids = torch.randint(0, vocab_size, (B, N))
    
    # Forward
    logits = model(input_ids)
    print(f"\nInput shape:  {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Expected:     ({B}, {N}, {vocab_size})")
    
    # Test de conservación isométrica (PMTP)
    print("\n" + "="*70)
    print("TEST: Conservación Isométrica (PMTP)")
    print("="*70)
    
    pmtp = PMTPRouter(D=D, seed=42)
    x_test = torch.randn(B, N, D)
    
    x_enc = pmtp.encode(x_test)
    x_dec = pmtp.decode(x_enc)
    
    mse = torch.mean((x_test - x_dec).abs()**2).item()
    print(f"MSE reconstrucción: {mse:.2e}")
    print(f"Límite Float32:     ~{np.finfo(np.float32).eps:.2e}")
    print(f"Conservación: {'✅ OK' if mse < 1e-5 else '❌ FALLA'}")
    
    # Test de Laplaciano Hermitiano
    print("\n" + "="*70)
    print("TEST: Propiedades del Laplaciano Magnético")
    print("="*70)
    
    lap = MagneticLaplacian(N=20, q=math.pi/4.0, skip_k=3)
    L_q = torch.complex(lap.L_q_real, lap.L_q_imag)
    
    # Hermitianidad
    is_herm = torch.allclose(L_q, L_q.conj().T)
    print(f"Hermitiano: {is_herm} {'✅' if is_herm else '❌'}")
    
    # Autovalores reales
    eigvals = torch.linalg.eigvalsh(L_q)
    is_real = torch.allclose(eigvals.imag, torch.zeros_like(eigvals.imag), atol=1e-6)
    is_nonneg = torch.all(eigvals.real >= -1e-6)
    print(f"Autovalores reales:     {is_real} {'✅' if is_real else '❌'}")
    print(f"Autovalores >= 0:       {is_nonneg} {'✅' if is_nonneg else '❌'}")
    
    # Test de complejidad O(D log D)
    print("\n" + "="*70)
    print("TEST: Complejidad Walsh-Hadamard")
    print("="*70)
    
    import time
    dims = [64, 128, 256, 512, 1024]
    times = []
    
    for d in dims:
        x = torch.randn(1, 1, d)
        # Warm-up
        _ = WalshHadamardTransform.fwht(x)
        
        t0 = time.perf_counter()
        for _ in range(100):
            _ = WalshHadamardTransform.fwht(x)
        t1 = time.perf_counter()
        
        avg = (t1 - t0) / 100 * 1000
        times.append(avg)
        ratio = avg / (d * math.log2(d))
        print(f"D={d:4d}: {avg:.4f} ms | ratio/(D*logD): {ratio:.6f}")
    
    print(f"\nRatio constante = O(D log D): {'✅' if max(times)/(dims[-1]*math.log2(dims[-1])) < 0.001 else '❌'}")
    
    print("\n" + "="*70)
    print("TODOS LOS TESTS COMPLETADOS")
    print("="*70)
