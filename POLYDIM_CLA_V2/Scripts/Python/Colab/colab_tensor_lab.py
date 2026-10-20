# ============================================================
# LABORATORY: TRANSFORMATION, COMPRESSION, AND SAFEGUARDING
# OF POLYDIMENSIONAL DATA (TENSORS)
#
# How to run in Google Colab:
#   1. Create a new notebook at colab.research.google.com
#   2. Copy each block marked with "# %%" into a separate cell
#      (or upload this file as a .py via "File > Upload notebook")
#   3. Run the cells in order. Parts A, B, and C are independent
#      but share the sample tensor generated in Part A.
# ============================================================

# %% [1] Install dependencies (run once in Colab)
# !pip install tensorly --quiet

# %% [2] Imports
import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker
import time

np.random.seed(42)

# ============================================================
# PART A: Transformation and Compression via Tucker Decomposition
# ============================================================
# We generate a synthetic order-3 tensor (e.g., Time x Channel x Patient)
# with a true low-rank structure plus noise. When we compress down
# to the true rank, reconstruction error should drop sharply.

# %% [3] Test Tensor Generation
I, J, K = 40, 30, 20
rank_real = 5

factors_real = [np.random.rand(I, rank_real),
                np.random.rand(J, rank_real),
                np.random.rand(K, rank_real)]
core_real = np.random.rand(rank_real, rank_real, rank_real)
tensor_clean = tl.tucker_to_tensor((core_real, factors_real))
noise = 0.01 * np.random.randn(I, J, K)
tensor = tensor_clean + noise

print(f"Original Tensor: shape {tensor.shape}, {tensor.size} elements")
print(f"Hidden true rank in data: {rank_real}\n")

# %% [4] Compress with different ranks and measure error vs. space savings
def compress_and_measure(tensor, rank):
    core, factors = tucker(tensor, rank=rank)
    reconstruction = tl.tucker_to_tensor((core, factors))
    error = np.linalg.norm(tensor - reconstruction) / np.linalg.norm(tensor)
    size_original = tensor.size
    size_compressed = core.size + sum(f.size for f in factors)
    savings = 1 - (size_compressed / size_original)
    return error, savings, size_compressed

print(f"{'Rank':<12}{'Relative Error':<18}{'Space Savings':<18}{'Compressed Size'}")
for r in [2, 3, 5, 8, 12, 20]:
    rank = (r, r, r)
    error, savings, tam = compress_and_measure(tensor, rank)
    print(f"{str(rank):<12}{error:<18.4f}{savings*100:<17.1f}%{tam}")

print("\nMethodology check: observe the behaviour exactly at rank 5 (the true rank).")
print("Below rank 5, relative error climbs steeply (loss of underlying structure).")
print("Above rank 5, error barely decreases (mostly spending space encoding noise).")


# ============================================================
# PART B: Scaling limits of brute-force tensor comparison
# ============================================================
# We simulate a small tensor bank (such as a model repository, like TStore)
# grouped into "families" (simulating fine-tuned variants of a base model),
# comparing brute-force O(n²) similarity checks against an LSH fingerprint.

# %% [5] Generate bank of tensors with simulated families
def generate_tensor_bank(n_tensors, shape=(8, 8, 8), n_families=5):
    bases = [np.random.rand(*shape) for _ in range(n_families)]
    bank = []
    labels = []
    for i in range(n_tensors):
        family = i % n_families
        variant = bases[family] + 0.02 * np.random.randn(*shape)
        bank.append(variant)
        labels.append(family)
    return bank, labels

def brute_force_comparisons(n):
    return n * (n - 1) // 2

def fingerprint_similarity(bank, n_bits=12):
    """Simplified LSH-style fingerprint: random projection + sign.
    Similar tensors fall into the same bucket with high probability
    without pairwise comparisons."""
    shape = bank[0].shape
    dim = int(np.prod(shape))
    random_planes = np.random.randn(n_bits, dim)
    fingerprints = {}
    for idx, tensor in enumerate(bank):
        vector = tensor.flatten()
        bits = tuple((random_planes @ vector) > 0)
        fingerprints.setdefault(bits, []).append(idx)
    return fingerprints

# %% [6] Compare theoretical brute-force growth
print("\nBrute-force comparisons growth:")
for n in [10, 100, 1000, 10000]:
    print(f"  n={n:<8} -> {brute_force_comparisons(n):,} pairwise checks")

# %% [7] Test fingerprint clustering on simulated bank
bank, real_labels = generate_tensor_bank(n_tensors=200, shape=(8, 8, 8), n_families=5)

t0 = time.time()
fingerprints = fingerprint_similarity(bank, n_bits=12)
t1 = time.time()

non_trivial_groups = {h: idxs for h, idxs in fingerprints.items() if len(idxs) > 1}

print(f"\nTensors processed: {len(bank)}")
print(f"Fingerprint clustering execution time: {t1 - t0:.4f} seconds")
print(f"Buckets found with more than one member: {len(non_trivial_groups)}")
for h, idxs in list(non_trivial_groups.items())[:3]:
    families_in_group = [real_labels[i] for i in idxs]
    print(f"  Bucket of {len(idxs)} tensors -> real families: {families_in_group}")

print("\nIf the family numbers within each bucket match, the fingerprint")
print("is correctly grouping related tensors WITHOUT having compared them")
print("pairwise. This breaks the quadratic O(n^2) scaling bottleneck.")


# ============================================================
# PART C: Incremental (Streaming) Decomposition Updates
# ============================================================
# We simulate a new data slice arriving (e.g., a new patient record)
# and measure the computational cost of re-decomposing from scratch.
# This illustrates why streaming tensor methods are needed at scale.

# %% [8] Measure cost of full recalculation on extended tensor
def decomposition_time(given_tensor, rank):
    t0 = time.time()
    tucker(given_tensor, rank=rank)
    return time.time() - t0

tensor_base = tensor[:, :, :15]      # 15 slices
tensor_extended = tensor            # 20 slices (5 new slices)

t_base = decomposition_time(tensor_base, rank=(5, 5, 5))
t_extended = decomposition_time(tensor_extended, rank=(5, 5, 5))

print(f"\nDecomposition time for base tensor (15 slices): {t_base:.4f}s")
print(f"Decomposition time RE-CALCULATING ALL (20 slices): {t_extended:.4f}s")
print("\nValidation note: on this small tensor the time difference is minor.")
print("For massive production tensors, full recalculation is exactly what")
print("streaming methods avoid by updating only the affected factors.")
