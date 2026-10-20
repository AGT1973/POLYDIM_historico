import torch
import math
import torch.nn.functional as F

# 1. INYECCIÓN SEMÁNTICA CONTINUA (Fourier)
def continuous_semantic_embedding(text, d, device='cpu'):
    d_c = d // 2 
    chars = torch.tensor([ord(c) for c in text], dtype=torch.float, device=device)
    positions = torch.arange(len(text), dtype=torch.float, device=device)
    freqs = torch.exp(torch.arange(d_c, dtype=torch.float, device=device) * -0.1)
    phases = chars.unsqueeze(1) * (positions.unsqueeze(1) + 1.0) * freqs.unsqueeze(0)
    v_complex = torch.sum(torch.exp(1j * phases), dim=0).unsqueeze(0)
    v_norm = v_complex / torch.norm(v_complex, p=2)
    return v_norm

# 2. LAPLACIANO MAGNÉTICO HERMÍTICO (Fanuel et al. 2017)
def build_hermitian_magnetic_laplacian(num_nodes, q, device='cpu'):
    A = torch.zeros(num_nodes, num_nodes, dtype=torch.cfloat, device=device)
    Theta = torch.zeros(num_nodes, num_nodes, dtype=torch.float, device=device)
    for i in range(1, num_nodes):
        A[i-1, i] = 1.0
        A[i, i-1] = 1.0
        Theta[i-1, i] = q
        Theta[i, i-1] = -q
    H = A * torch.exp(1j * Theta)
    deg = torch.sum(torch.abs(H), dim=1)
    D = torch.diag(deg).to(torch.cfloat)
    return D - H

# 3. MEZCLA HOLOGRÁFICA FFT O(D log D)
class HolographicUnitaryRotor:
    def __init__(self, d_c, seed=42, device='cpu'):
        self.d_c = d_c
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        phases = torch.rand(d_c, generator=gen, device=device) * 2 * math.pi
        self.phase_diag = torch.exp(1j * phases)
        
    def forward(self, x):
        x_freq = torch.fft.fft(x, norm="ortho")
        x_shifted = x_freq * self.phase_diag
        return torch.fft.ifft(x_shifted, norm="ortho")

    def inverse(self, x):
        x_freq = torch.fft.fft(x, norm="ortho")
        x_shifted = x_freq * torch.conj(self.phase_diag)
        return torch.fft.ifft(x_shifted, norm="ortho")
