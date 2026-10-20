import torch
import pytest
from ..core import (
    GaussMarkovAdjunctionLayer, 
    TransformGroupoid, 
    CartanDieudonneStiefelRotor, 
    EngineMathV2,
    SpectralCognitiveMonitor
)

def test_weak_invertibility_gauss_markov():
    """Test Axiom IV: Weak Invertibility via Functorial Adjunctions (Gauss-Markov)"""
    layer = GaussMarkovAdjunctionLayer(in_features=10, reduced_features=5)
    x = torch.randn(1, 10)
    contracted, reconstructed = layer(x)
    assert contracted.shape == (1, 5)
    assert reconstructed.shape == (1, 10)

def test_transform_groupoid():
    """Test Groupoid G preserves GEO_ID up to scaling"""
    dim = 8
    groupoid = TransformGroupoid(dim)
    x = torch.randn(1, dim)
    
    paths = groupoid(x, steps=5)
    assert len(paths) == 6 # initial + 5 steps
    
    # Check GEO_ID preservation (Norm L2)
    initial_norm = x.norm(p=2, dim=-1)
    for p in paths:
        assert torch.allclose(p.norm(p=2, dim=-1), initial_norm, atol=1e-5)

def test_cartan_dieudonne_stiefel_rotor():
    """Test Cartan-Dieudonné Stiefel Rotor rotation preserves norm (Isometry) in BF16"""
    dim = 16
    rotor = CartanDieudonneStiefelRotor(dim).bfloat16()
    x = torch.randn(1, dim).bfloat16()
    rotated_x = rotor(x)
    
    assert rotated_x.shape == x.shape
    assert torch.allclose(x.norm(p=2, dim=-1), rotated_x.norm(p=2, dim=-1), atol=5e-2)

def test_perron_frobenius():
    """Test Perron-Frobenius root computation"""
    A = torch.tensor([[0.5, 0.2], [0.1, 0.8]], dtype=torch.float32)
    eigval, eigvec = EngineMathV2.perron_frobenius_root(A, iterations=50)
    
    # Verify Av = lambda v
    Av = torch.mv(A, eigvec)
    lambda_v = eigval * eigvec
    assert torch.allclose(Av, lambda_v, atol=1e-4)

def test_cayley_transform():
    """Test Cayley Transform yields orthogonal matrix for skew-symmetric inputs"""
    dim = 4
    A = torch.randn(dim, dim)
    A_skew = 0.5 * (A - A.transpose(-1, -2))
    W = EngineMathV2.cayley_transform(A_skew)
    I = torch.eye(dim)
    WtW = torch.matmul(W.transpose(-1, -2), W)
    assert torch.allclose(WtW, I, atol=1e-4)

def test_spectral_cognitive_monitor_small():
    """Test Spectral Gap λ1 detection (small N, direct eigendecomposition)"""
    monitor = SpectralCognitiveMonitor(threshold_pk=0.01)
    
    # Graph with 2 disconnected components -> lambda_1 should be 0 (Cognitive Collapse)
    laplacian_collapse = torch.tensor([
        [ 1.0, -1.0,  0.0,  0.0],
        [-1.0,  1.0,  0.0,  0.0],
        [ 0.0,  0.0,  1.0, -1.0],
        [ 0.0,  0.0, -1.0,  1.0]
    ])
    
    u_residual = 0.001
    l1_col, collapse = monitor(laplacian_collapse, u_residual)
    assert collapse == True
    assert l1_col < 1e-4

    # Fully connected graph -> lambda_1 > 0
    laplacian_healthy = torch.tensor([
        [ 3.0, -1.0, -1.0, -1.0],
        [-1.0,  3.0, -1.0, -1.0],
        [-1.0, -1.0,  3.0, -1.0],
        [-1.0, -1.0, -1.0,  3.0]
    ])
    
    l1_health, collapse_health = monitor(laplacian_healthy, u_residual)
    assert collapse_health == False
    assert l1_health > 1.0

def test_spectral_cognitive_monitor_large():
    """Test Spectral Gap λ1 detection (large N, LOBPCG O(N) approximation)"""
    monitor = SpectralCognitiveMonitor(threshold_pk=1e-4)
    N = 1000
    
    # Random positive definite matrix with large eigenvalues
    X = torch.randn(N, 100)
    laplacian = torch.matmul(X, X.transpose(-1, -2))
    # Add diagonal to ensure it's strongly positive definite
    laplacian += torch.eye(N) * 5.0
    
    u_residual = 0.1
    # For large matrix, it should use LOBPCG
    l1, collapse = monitor(laplacian, u_residual)
    # L1 should be roughly >= 5.0 due to diagonal shift
    assert l1 > 1.0
    assert collapse == False
