import torch
import unittest
from engine.geo_tokenizer import GeoTokenizer
from engine.clifford_rotors import CliffordRotors
from engine.hybrid_block import PolydimHybridBlock

# Constantes para tests
D_MODEL = 512
TOLERANCE = 1e-5

class TestHybridEngine(unittest.TestCase):
    def setUp(self):
        # 5 nodos, 4 aristas (grafo línea simple)
        N, E = 5, 4
        self.sample_topology = torch.zeros(N, E)
        for i in range(E):
            self.sample_topology[i, i] = -1
            self.sample_topology[i+1, i] = 1

    def test_geo_tokenizer_isometry(self):
        """Testea que la tokenización retorne vectores unitarios (Hiperesfera)."""
        tokenizer = GeoTokenizer(d_model=D_MODEL)
        seq = ["inteligencia", "artificial", "topologica"]
        geo_id = tokenizer.encode_sequence(seq)
        
        norms = torch.norm(geo_id, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=TOLERANCE), "GeoTokenizer no proyecta a la esfera unitaria.")

    def test_clifford_strict_isometry(self):
        """Testea la isometría estricta de las reflexiones de Householder."""
        rotors = CliffordRotors(d_model=D_MODEL)
        N = 10
        x = torch.randn(N, D_MODEL)
        x = x / torch.norm(x, dim=-1, keepdim=True)
        v1 = torch.randn(N, D_MODEL)
        v2 = torch.randn(N, D_MODEL)
        
        x_rotated = rotors.apply_rotor(x, v1, v2)
        norm_in = torch.norm(x, dim=-1)
        norm_out = torch.norm(x_rotated, dim=-1)
        
        self.assertTrue(torch.allclose(norm_in, norm_out, atol=TOLERANCE), "El rotor de Clifford destruyó la isometría.")

    def test_hybrid_block_gradient_flow(self):
        """Testea que el Hybrid Block fluya gradientes."""
        seq = ["el", "agente", "autonomo", "evalua", "tensores"]
        engine = PolydimHybridBlock(d_model=D_MODEL)
        out = engine(seq, self.sample_topology)
        
        target = torch.randn_like(out)
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()
        
        has_gradients = any(param.grad is not None for name, param in engine.named_parameters())
        self.assertTrue(has_gradients, "El bloque híbrido está desconectado del autograd.")

    def test_hybrid_block_output_shape(self):
        """Testea la salida del bloque híbrido."""
        seq = ["nodo_1", "nodo_2", "nodo_3", "nodo_4", "nodo_5"]
        engine = PolydimHybridBlock(d_model=D_MODEL)
        out = engine(seq, self.sample_topology)
        self.assertEqual(out.shape, (5, D_MODEL))

if __name__ == '__main__':
    unittest.main()
