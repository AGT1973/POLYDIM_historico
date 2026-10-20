import torch
import hashlib
from typing import List, Dict

class GeoTokenizer:
    """
    Fase 1: Ingestión Determinista y Hashing Topológico (JL Lemma).
    Proyecta conceptos (tokens/strings) al hiperespacio de forma ortogonal sin depender de un vocabulario BPE finito.
    """
    def __init__(self, d_model: int = 10000, seed: int = 42):
        self.d_model = d_model
        # Generador base (aunque cada concepto usa el suyo derivado de su hash)
        self.base_seed = seed
        self._cache: Dict[str, torch.Tensor] = {}

    def _hash_to_seed(self, text: str) -> int:
        """
        Convierte un string en un entero determinista para anclar la proyección hiperdimensional.
        """
        hash_digest = hashlib.sha256(text.encode('utf-8')).digest()
        # Usar los primeros 8 bytes para un entero de 64 bits (sin desbordar C-long)
        return int.from_bytes(hash_digest[:8], byteorder='big') % (2**31 - 1)

    def encode_concept(self, concept: str) -> torch.Tensor:
        """
        Mapea un concepto a un vector D-dimensional unitario sobre la esfera S^{D-1}.
        Garantiza cuasi-ortogonalidad entre conceptos distintos gracias al Lema de Johnson-Lindenstrauss.
        """
        if concept in self._cache:
            return self._cache[concept]

        concept_seed = self._hash_to_seed(concept)
        gen = torch.Generator().manual_seed(concept_seed)
        
        # Proyección Euclidiana
        vec = torch.randn(self.d_model, generator=gen, dtype=torch.float32)
        
        # Retracción a la Variedad de Stiefel (Hiperesfera L2)
        vec = vec / torch.norm(vec, p=2)
        
        self._cache[concept] = vec
        return vec

    def encode_sequence(self, sequence: List[str]) -> torch.Tensor:
        """
        Codifica una secuencia de N conceptos en un tensor [N, D].
        """
        vectors = [self.encode_concept(c) for c in sequence]
        return torch.stack(vectors) # Shape: [N, D]

if __name__ == "__main__":
    print("=== POLYDIM V3: GeoTokenizer (Prueba de Ingestión) ===")
    tokenizer = GeoTokenizer(d_model=10000)
    
    v_perro = tokenizer.encode_concept("perro")
    v_perro_2 = tokenizer.encode_concept("perro")
    v_gato = tokenizer.encode_concept("gato")
    
    print(f"Norma 'perro': {torch.norm(v_perro).item():.4f} (Esperado: 1.0)")
    print(f"Distancia 'perro' a sí mismo: {torch.norm(v_perro - v_perro_2).item():.4f} (Esperado: 0.0)")
    print(f"Producto escalar 'perro' · 'gato': {torch.dot(v_perro, v_gato).item():.4f} (Esperado: ~0.0 por JL Lemma)")
