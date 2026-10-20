"""
tarea_26.py
-----------
Certified bound propagation for a feed-forward ReLU network, combining:

  (a) Interval Bound Propagation, IBP  (Gowal et al., 2018, arXiv:1810.12715)
      -- the "Box" abstract domain: non-relational, one interval per neuron.

  (b) A DeepPoly-style relational/polyhedral bound propagation
      (Singh, Gehr, Puschel & Vechev, "An Abstract Domain for Certifying
      Neural Networks", POPL 2019 / PACMPL 3(POPL):41) -- bounds are affine
      (linear) functions of the ORIGINAL input, i.e. each neuron's range is
      described by two hyperplanes in input space rather than an independent
      scalar interval. This is what makes it "polyhedral"/relational instead
      of just an axis-aligned box.

Role framing of the exercise ("especialista en NVIDIA Triton"): both passes
below are written as a single batched, branch-free (at the batch level)
JAX routine -- the same computational shape (dense matmuls + elementwise
selects, jax.jit-compiled) that a hand-written Triton/CUDA kernel would
have. An actual Triton kernel needs a CUDA GPU to compile and run, which
this sandbox does not provide, so the *reference, runnable* implementation
here is plain JAX/NumPy (jit-compiled, CPU or GPU alike). A Triton kernel
skeleton is sketched at the bottom of this file, clearly marked as
untested/GPU-only, for a student who wants to take it further on their
own machine.

See tarea_26.md for the full derivation, the soundness argument, and the
discussion of why the *general* convex-polyhedra domain (Cousot & Halbwachs,
POPL 1978) is not what is implemented here.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# 1. ReLU relaxation (the only lossy step -- affine layers are exact)
# ---------------------------------------------------------------------------
def relu_relax(l, u):
    """
    Given CONCRETE pre-activation bounds l <= z <= u for a vector of neurons,
    return (lam_up, mu_up, lam_lo, mu_lo) such that, elementwise,

        lam_lo * z + mu_lo  <=  relu(z)  <=  lam_up * z + mu_up      for all z in [l, u].

    Three cases per neuron (DeepPoly / CROWN / Fast-Lin triangle relaxation):
      * always-off  (u <= 0):  relu(z) = 0            -> exact, lam=mu=0
      * always-on   (l >= 0):  relu(z) = z             -> exact, lam=1, mu=0
      * crossing (l < 0 < u):  upper bound is the unique line through
        (l, 0) and (u, u):  slope = u / (u - l), intercept = -slope * l.
        Lower bound: any line y = c*z with c in [0, 1] is sound; we use the
        area-minimizing choice from DeepPoly: c = 1 if u >= -l else c = 0.
    """
    active = l >= 0.0
    inactive = u <= 0.0
    crossing = jnp.logical_not(jnp.logical_or(active, inactive))

    denom = jnp.where(crossing, u - l, 1.0)  # dummy 1.0 outside crossing branch, avoids 0/0
    slope = jnp.where(crossing, u / denom, 0.0)
    intercept = jnp.where(crossing, -slope * l, 0.0)
    lam_lo_cross = jnp.where(u >= -l, 1.0, 0.0)

    lam_up = jnp.where(active, 1.0, jnp.where(inactive, 0.0, slope))
    mu_up = jnp.where(active, 0.0, jnp.where(inactive, 0.0, intercept))
    lam_lo = jnp.where(active, 1.0, jnp.where(inactive, 0.0, lam_lo_cross))
    mu_lo = jnp.zeros_like(l)
    return lam_up, mu_up, lam_lo, mu_lo


# ---------------------------------------------------------------------------
# 2. Concretization of a symbolic (input-linear) bound over a box
# ---------------------------------------------------------------------------
def concretize(C, d, l0, u0):
    """
    f(x) = C @ x + d  (C: [out, in], d: [out]).  Over the box l0 <= x <= u0,
    standard linear-form interval arithmetic gives the exact min/max of an
    affine function over a box:

        max_x f(x) = C+ @ u0 + C- @ l0 + d
        min_x f(x) = C+ @ l0 + C- @ u0 + d

    with C+ = max(C, 0), C- = min(C, 0). Returns (min_val, max_val), [out].
    """
    C_pos = jnp.maximum(C, 0.0)
    C_neg = jnp.minimum(C, 0.0)
    max_val = C_pos @ u0 + C_neg @ l0 + d
    min_val = C_pos @ l0 + C_neg @ u0 + d
    return min_val, max_val


# ---------------------------------------------------------------------------
# 3. Exact affine-layer composition of symbolic bounds
# ---------------------------------------------------------------------------
def affine_compose(C_lo, d_lo, C_up, d_up, W, b):
    """
    Push relational bounds for the previous layer's activations a(x)
    (lo(x) <= a(x) <= up(x), both affine in x) through z = W @ a + b.
    This step is EXACT (no relaxation): for output neuron j,

      up_j(x) = sum_i [ max(W_ji,0) * up_i(x) + min(W_ji,0) * lo_i(x) ] + b_j
      lo_j(x) = sum_i [ max(W_ji,0) * lo_i(x) + min(W_ji,0) * up_i(x) ] + b_j
    """
    W_pos = jnp.maximum(W, 0.0)
    W_neg = jnp.minimum(W, 0.0)

    C_up_new = W_pos @ C_up + W_neg @ C_lo
    d_up_new = W_pos @ d_up + W_neg @ d_lo + b
    C_lo_new = W_pos @ C_lo + W_neg @ C_up
    d_lo_new = W_pos @ d_lo + W_neg @ d_up + b
    return C_lo_new, d_lo_new, C_up_new, d_up_new


# ---------------------------------------------------------------------------
# 4. Full network pass: DeepPoly-style relational bounds + IBP baseline
# ---------------------------------------------------------------------------
@partial(jax.jit, static_argnames=("relu_layers",))
def certified_bounds(weights, biases, l0, u0, relu_layers):
    """
    weights, biases : tuples of per-layer (W_k, b_k) applied as z = W_k @ a + b_k
    l0, u0          : input box, each shape [n_in]
    relu_layers     : static tuple[bool], True at position k if a ReLU
                       follows the k-th affine layer

    Returns two (lower, upper) pairs on the network's final output:
      dp_bounds  : DeepPoly-style relational/polyhedral bounds
      ibp_bounds : plain Interval Bound Propagation bounds

    NOTE on tightness (see tarea_26.md, Conclusion, for the full discussion
    and the empirical data behind this note): the relational domain usually
    -- but NOT provably always -- gives tighter bounds than IBP, because
    this implementation forward-composes the ReLU relaxation coefficients
    layer after layer without ever re-deriving a neuron's symbolic bound
    from scratch against a nearer layer ("back-substitution", as used by
    CROWN, Zhang et al. 2018, and by DeepPoly's actual algorithm). Without
    back-substitution, relaxation error from early layers can compound
    multiplicatively through the coefficient matrix and occasionally make
    dp_bounds LOOSER than ibp_bounds at moderate depth. Both passes remain
    SOUND regardless (the true output is always inside the reported
    interval) -- what varies is only which one is tighter.
    """
    n_in = l0.shape[0]

    # relational (polyhedral) state: two affine maps of the ORIGINAL input x
    C_lo, d_lo = jnp.eye(n_in), jnp.zeros((n_in,))
    C_up, d_up = jnp.eye(n_in), jnp.zeros((n_in,))

    # non-relational (IBP) state: one concrete box per layer
    ibp_l, ibp_u = l0, u0

    for (W, b), has_relu in zip(zip(weights, biases), relu_layers):
        # --- relational / polyhedral path (exact affine step) ---
        C_lo, d_lo, C_up, d_up = affine_compose(C_lo, d_lo, C_up, d_up, W, b)

        # --- IBP path (exact affine step, box arithmetic) ---
        W_pos, W_neg = jnp.maximum(W, 0.0), jnp.minimum(W, 0.0)
        ibp_u_new = W_pos @ ibp_u + W_neg @ ibp_l + b
        ibp_l_new = W_pos @ ibp_l + W_neg @ ibp_u + b
        ibp_l, ibp_u = ibp_l_new, ibp_u_new

        if has_relu:
            # concretize the current symbolic bounds to know each neuron's
            # concrete [l, u] range, needed to pick the ReLU relaxation case
            l_c, _ = concretize(C_lo, d_lo, l0, u0)
            _, u_c = concretize(C_up, d_up, l0, u0)
            lam_up, mu_up, lam_lo, mu_lo = relu_relax(l_c, u_c)

            C_up = lam_up[:, None] * C_up
            d_up = lam_up * d_up + mu_up
            C_lo = lam_lo[:, None] * C_lo
            d_lo = lam_lo * d_lo + mu_lo

            ibp_l = jnp.maximum(ibp_l, 0.0)
            ibp_u = jnp.maximum(ibp_u, 0.0)

    dp_l, _ = concretize(C_lo, d_lo, l0, u0)
    _, dp_u = concretize(C_up, d_up, l0, u0)
    return (dp_l, dp_u), (ibp_l, ibp_u)


# ---------------------------------------------------------------------------
# 5. Empirical sanity check: random ReLU MLP + Monte-Carlo soundness check
# ---------------------------------------------------------------------------
def run_network(weights, biases, relu_layers, x):
    """Concrete forward pass, used only to sample the ground truth for the
    Monte-Carlo soundness check (never used inside certified_bounds)."""
    a = x
    for (W, b), has_relu in zip(zip(weights, biases), relu_layers):
        a = W @ a + b
        if has_relu:
            a = jnp.maximum(a, 0.0)
    return a


def build_random_network(key, layer_sizes, relu_after_all_but_last=True):
    weights, biases, relu_layers = [], [], []
    keys = jax.random.split(key, len(layer_sizes) - 1)
    for i, k in enumerate(keys):
        n_in, n_out = layer_sizes[i], layer_sizes[i + 1]
        wk, bk = jax.random.split(k)
        W = jax.random.normal(wk, (n_out, n_in)) * jnp.sqrt(2.0 / n_in)
        b = jax.random.normal(bk, (n_out,)) * 0.1
        weights.append(W)
        biases.append(b)
        is_last = i == len(keys) - 1
        relu_layers.append((not is_last) if relu_after_all_but_last else True)
    return tuple(weights), tuple(biases), tuple(relu_layers)


def sanity_check(seed=0, layer_sizes=(4, 8, 8, 3), eps=0.05, n_samples=20000):
    key = jax.random.PRNGKey(seed)
    net_key, x_key, sample_key = jax.random.split(key, 3)

    weights, biases, relu_layers = build_random_network(net_key, layer_sizes)

    x0 = jax.random.normal(x_key, (layer_sizes[0],))
    l0, u0 = x0 - eps, x0 + eps

    (dp_l, dp_u), (ibp_l, ibp_u) = certified_bounds(weights, biases, l0, u0, relu_layers)

    # Monte-Carlo soundness check: sample many x inside the box, evaluate the
    # TRUE network, and confirm every sample's output falls inside the
    # claimed certified bounds (a violation would mean a bug: unsound bounds).
    samples = jax.random.uniform(
        sample_key, (n_samples, layer_sizes[0]), minval=l0, maxval=u0
    )
    outs = jax.vmap(lambda x: run_network(weights, biases, relu_layers, x))(samples)
    emp_min = outs.min(axis=0)
    emp_max = outs.max(axis=0)

    dp_width = dp_u - dp_l
    ibp_width = ibp_u - ibp_l

    report = {
        "dp_lower": np.array(dp_l),
        "dp_upper": np.array(dp_u),
        "ibp_lower": np.array(ibp_l),
        "ibp_upper": np.array(ibp_u),
        "empirical_min_sample": np.array(emp_min),
        "empirical_max_sample": np.array(emp_max),
        "dp_bounds_sound": bool(jnp.all(dp_l <= emp_min + 1e-5) and jnp.all(dp_u >= emp_max - 1e-5)),
        "ibp_bounds_sound": bool(jnp.all(ibp_l <= emp_min + 1e-5) and jnp.all(ibp_u >= emp_max - 1e-5)),
        "dp_at_least_as_tight_as_ibp": bool(jnp.all(dp_width <= ibp_width + 1e-6)),
        "mean_width_reduction_pct": float(100.0 * (1.0 - jnp.mean(dp_width) / jnp.mean(ibp_width))),
    }
    return report


def benchmark_depth_effect(configs=None, n_seeds=12, eps=0.08):
    """
    Reproduces the depth-vs-tightness study described in tarea_26.md:
    for a handful of architectures of increasing depth, run `sanity_check`
    over several random seeds and report (a) the fraction of seeds where
    DeepPoly-style bounds are at least as tight as IBP, and (b) the range
    of the mean per-output width reduction (%). Both dp_bounds_sound and
    ibp_bounds_sound are asserted True for every run (soundness is never
    in question here -- only relative tightness).
    """
    if configs is None:
        configs = [
            (4, 8, 3),
            (4, 8, 8, 3),
            (5, 10, 10, 6, 2),
            (5, 10, 10, 10, 10, 3),
            (6, 12, 12, 12, 12, 12, 3),
        ]
    rows = []
    for sizes in configs:
        wins, reductions = 0, []
        for seed in range(n_seeds):
            r = sanity_check(seed=seed, layer_sizes=sizes, eps=eps)
            assert r["dp_bounds_sound"] and r["ibp_bounds_sound"], "unsound bound found -- bug!"
            wins += int(r["dp_at_least_as_tight_as_ibp"])
            reductions.append(r["mean_width_reduction_pct"])
        rows.append({
            "layer_sizes": sizes,
            "depth": len(sizes) - 1,
            "dp_wins_out_of": f"{wins}/{n_seeds}",
            "width_reduction_pct_min": min(reductions),
            "width_reduction_pct_max": max(reductions),
            "width_reduction_pct_mean": sum(reductions) / len(reductions),
        })
    return rows


if __name__ == "__main__":
    print("--- single-network sanity check ---")
    report = sanity_check()
    for k, v in report.items():
        print(f"{k}: {v}")

    print("\n--- depth-vs-tightness benchmark (see tarea_26.md, Conclusion) ---")
    for row in benchmark_depth_effect():
        print(row)


# ---------------------------------------------------------------------------
# 6. OPTIONAL / NOT RUN HERE: illustrative Triton kernel skeleton
# ---------------------------------------------------------------------------
# The block below sketches how the affine-composition step (the hot loop,
# a dense matmul + elementwise select) could be written as a literal
# NVIDIA Triton kernel. It requires a CUDA GPU + `triton` installed and is
# NOT executed as part of this sanity check (this sandbox has no GPU).
# It is left here only so the student can try it on their own machine.
#
# import triton
# import triton.language as tl
#
# @triton.jit
# def affine_relax_kernel(
#     W_ptr, Clo_ptr, Cup_ptr, dlo_ptr, dup_ptr,
#     out_Clo_ptr, out_Cup_ptr, out_dlo_ptr, out_dup_ptr,
#     n_out, n_in, n_feat, BLOCK: tl.constexpr,
# ):
#     # one program instance per output neuron row; BLOCK columns of the
#     # input-linear coefficient vector processed per step. Left as a
#     # skeleton: indexing/masking omitted for brevity.
#     pid = tl.program_id(axis=0)
#     ...
