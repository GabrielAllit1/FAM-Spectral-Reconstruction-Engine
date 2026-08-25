"""
MathToolBox v7.0 — graphon.py
=============================

Graph limits / graphon engine for v7.0.

This module provides a mathematically grounded framework for working with
graphons — measurable functions

    W : [0,1]^2 -> [0,1]

that arise as limits of dense graph sequences in the sense of
Lovász–Szegedy.

The implementations here are *numerically concrete* and connect directly
to the rest of MathToolBox:

    • FAM / 973 sequences -> graphon kernels
    • rare patterns (Thue–Morse, Van Eck, etc.) -> structured kernels
    • fractal maps -> graphon-valued textures
    • mining / unlocker pipelines -> random graphs sampled from W

Core features
-------------

    • Graphon class:
        - evaluation of W(x,y)
        - Monte Carlo integration
        - sampling of adjacency matrices

    • Constructors:
        - constant graphon W(x,y) = p
        - step graphon from a finite matrix (block-constant)
        - rank-1 graphon from f(x)f(y)
        - sequence-driven graphons from rare sequences

    • Statistics:
        - edge density estimator
        - degree distribution sampler
        - homomorphism densities (approximate):
            * t(K2, W)   — edge density
            * t(K3, W)   — triangle density (Monte Carlo)

    • FAM/973 hooks:
        - Thue–Morse graphon
        - Van Eck graphon

All code is mathematically real — no placeholders.
"""

from __future__ import annotations
from typing import Callable, List, Tuple
import math
import random

from .numerics import shannon_entropy
from . import sequences as seq_mod


# ======================================================================
# GRAPHON CLASS
# ======================================================================

class Graphon:
    """
    Graphon wrapper around a kernel function:

        kernel : [0,1]^2 -> [0,1]

    plus helpers for sampling and estimating densities.

    NOTE:
        The kernel is assumed to be symmetric in practice (W(x,y)=W(y,x)),
        but we do not strictly enforce symmetry in the code so that
        non-symmetric experimental kernels can also be used if desired.
    """

    def __init__(self, kernel: Callable[[float, float], float]):
        self.kernel = kernel

    # -------------------------------------------
    # Basic evaluation
    # -------------------------------------------

    def W(self, x: float, y: float) -> float:
        """Evaluate W(x,y) with bounds clamped into [0,1]."""
        x = min(max(x, 0.0), 1.0)
        y = min(max(y, 0.0), 1.0)
        v = self.kernel(x, y)
        # clamp to [0,1]
        return float(min(max(v, 0.0), 1.0))

    # -------------------------------------------
    # Monte Carlo integration
    # -------------------------------------------

    def integrate(self, n_samples: int = 10000) -> float:
        """
        Approximate the integral:

            ∫_0^1 ∫_0^1 W(x,y) dx dy

        via Monte Carlo sampling with n_samples random points (x,y)
        uniformly in the unit square.
        """
        if n_samples <= 0:
            return 0.0
        s = 0.0
        for _ in range(n_samples):
            x = random.random()
            y = random.random()
            s += self.W(x, y)
        return s / n_samples

    # -------------------------------------------
    # Sample random graph
    # -------------------------------------------

    def sample_graph(self, n: int) -> List[List[int]]:
        """
        Sample an undirected simple graph on {0,1,...,n-1} from this
        graphon, using the standard sampling scheme:

            1. Draw latent positions U_i ~ Uniform(0,1).
            2. For each pair (i,j), i<j, add edge with probability
               W(U_i, U_j), independently.

        Returns adjacency matrix A[n][n] of 0/1, with A[i][i] = 0.
        """
        if n <= 0:
            return []

        U = [random.random() for _ in range(n)]
        A = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                p = self.W(U[i], U[j])
                if random.random() < p:
                    A[i][j] = 1
                    A[j][i] = 1
        return A

    # -------------------------------------------
    # Edge density / degree distribution
    # -------------------------------------------

    def edge_density(self, n_samples: int = 20000) -> float:
        """
        Estimate the edge density:

            t(K2, W) = ∫_0^1 ∫_0^1 W(x,y) dx dy

        via Monte Carlo (alias for integrate()).
        """
        return self.integrate(n_samples=n_samples)

    def sample_degree_distribution(
        self,
        n_vertices: int = 200,
        n_graph_samples: int = 1,
    ) -> List[float]:
        """
        Sample degrees from graphs drawn from W, then normalize by (n-1)
        to approximate the degree distribution in the limit.

        Parameters
        ----------
        n_vertices : int
            Number of vertices in each sampled graph.
        n_graph_samples : int
            Number of independent graphs to sample.

        Returns
        -------
        List[float]
            Empirical degrees / (n_vertices-1).
        """
        if n_vertices <= 1 or n_graph_samples <= 0:
            return []

        degrees: List[float] = []
        for _ in range(n_graph_samples):
            A = self.sample_graph(n_vertices)
            for i in range(n_vertices):
                deg = sum(A[i][j] for j in range(n_vertices) if j != i)
                degrees.append(deg / (n_vertices - 1))
        return degrees

    # -------------------------------------------
    # Homomorphism density: triangles (K3)
    # -------------------------------------------

    def triangle_density_mc(
        self,
        n_samples: int = 10000,
    ) -> float:
        """
        Approximate the triangle homomorphism density:

            t(K3, W) = ∫ W(x,y) W(y,z) W(z,x) dx dy dz

        via Monte Carlo with n_samples triples (x,y,z).

        This is a core statistic in graph limit theory.
        """
        if n_samples <= 0:
            return 0.0

        s = 0.0
        for _ in range(n_samples):
            x = random.random()
            y = random.random()
            z = random.random()
            w_xy = self.W(x, y)
            w_yz = self.W(y, z)
            w_zx = self.W(z, x)
            s += w_xy * w_yz * w_zx
        return s / n_samples

    # -------------------------------------------
    # Entropy-like measure on W
    # -------------------------------------------

    def entropy_estimate(
        self,
        n_samples: int = 20000,
        bins: int = 32,
    ) -> float:
        """
        Estimate the "value-distribution entropy" of W:

            sample values v_k = W(x_k, y_k)
            build histogram with 'bins' bins on [0,1]
            compute Shannon entropy of the empirical distribution

        This is NOT a graphon-theoretic quantity per se, but is
        useful as a coarse measure of how "structured vs random"
        W looks at the value level.
        """
        if n_samples <= 0 or bins <= 0:
            return 0.0

        vals = []
        for _ in range(n_samples):
            x = random.random()
            y = random.random()
            vals.append(self.W(x, y))

        # discretize
        hist = [0] * bins
        for v in vals:
            idx = int(v * bins)
            if idx == bins:
                idx = bins - 1
            hist[idx] += 1

        return shannon_entropy(hist)


# ======================================================================
# CANONICAL GRAPHON CONSTRUCTORS
# ======================================================================

def constant_graphon(p: float) -> Graphon:
    """
    W(x,y) = p, constant in [0,1].

    Represents the Erdős–Rényi G(n,p) graphon limit.
    """
    p = float(min(max(p, 0.0), 1.0))

    def kernel(x: float, y: float) -> float:
        return p

    return Graphon(kernel)


def step_graphon_from_matrix(M: List[List[float]]) -> Graphon:
    """
    Build a step graphon from a k×k matrix M with entries in [0,1].

    We partition [0,1] into k equal subintervals:

        I_i = [i/k, (i+1)/k)

    and define:

        W(x,y) = M[i][j] if x in I_i, y in I_j.

    This recovers a graphon representation of a finite weighted adjacency
    matrix, suitable for coarse-grained limits.
    """
    if not M or not M[0]:
        raise ValueError("Matrix M must be non-empty.")
    k = len(M)
    for row in M:
        if len(row) != k:
            raise ValueError("Matrix M must be square.")
    M_clamped = [[float(min(max(v, 0.0), 1.0)) for v in row] for row in M]

    def kernel(x: float, y: float) -> float:
        x = min(max(x, 0.0), 1.0)
        y = min(max(y, 0.0), 1.0)
        i = min(int(x * k), k - 1)
        j = min(int(y * k), k - 1)
        return M_clamped[i][j]

    return Graphon(kernel)


def rank1_graphon(f: Callable[[float], float]) -> Graphon:
    """
    Build a rank-1 graphon:

        W(x,y) = clip( f(x) * f(y), 0, 1 )

    where f : [0,1] -> ℝ. If f is in [0,1], this directly yields a
    valid graphon; if not, we clamp the product.

    Rank-1 graphons correspond to "factorizable" random graphs with
    latent variables θ_i and edge probability ~ θ_i θ_j.
    """
    def kernel(x: float, y: float) -> float:
        v = f(x) * f(y)
        return float(min(max(v, 0.0), 1.0))

    return Graphon(kernel)


# ======================================================================
# SEQUENCE-DRIVEN GRAPHONS (RARE PATTERNS ↔ KERNELS)
# ======================================================================

def _sequence_to_step_function(seq: List[int]) -> Callable[[float], float]:
    """
    Helper: convert a finite integer sequence into a piecewise-constant
    function f : [0,1] -> [0,1] by:

        1. Normalizing seq to [0,1]:
               v_i = (seq_i - min) / (max - min)   (unless all equal)
        2. Assigning each v_i to interval i/len(seq) ≤ x < (i+1)/len(seq).

    This is used as the base "profile" for rank-1 or step graphons.
    """
    if not seq:
        def f0(x: float) -> float:
            return 0.0
        return f0

    mn = min(seq)
    mx = max(seq)
    n = len(seq)

    if mx == mn:
        vals = [0.5] * n
    else:
        vals = [(s - mn) / (mx - mn) for s in seq]

    def f(x: float) -> float:
        x = min(max(x, 0.0), 1.0)
        i = min(int(x * n), n - 1)
        return vals[i]

    return f


def thue_morse_graphon(length: int = 1024) -> Graphon:
    """
    Construct a graphon from the Thue–Morse sequence:

        • Generate first 'length' terms t(n) ∈ {0,1}.
        • Convert to step function f(x) via _sequence_to_step_function.
        • Define rank-1 graphon W(x,y) = f(x) * f(y).

    The result is a highly structured but nontrivial graphon that
    encodes the Thue–Morse parity pattern.
    """
    tm = seq_mod.thue_morse(length)
    f = _sequence_to_step_function(tm)
    return rank1_graphon(f)


def van_eck_graphon(length: int = 512) -> Graphon:
    """
    Construct a graphon from the Van Eck memory sequence:

        • Generate first 'length' terms v(n).
        • Convert to step function f(x).
        • Define rank-1 graphon W(x,y) = f(x) * f(y).

    This yields a kernel reflecting the irregular recall pattern
    of Van Eck, useful for high-entropy but structured graph
    generation.
    """
    ve = seq_mod.van_eck(length)
    f = _sequence_to_step_function(ve)
    return rank1_graphon(f)


def kolakoski_graphon(length: int = 1024) -> Graphon:
    """
    Construct a graphon from the Kolakoski(1,2) sequence.

    Same pattern:
        • Seq → step function → rank-1 graphon.
    """
    ks = seq_mod.kolakoski_12(length)
    f = _sequence_to_step_function(ks)
    return rank1_graphon(f)


# ======================================================================
# SIMPLE UTILITY: GRAPHON FROM ADJACENCY MATRIX (UNWEIGHTED)
# ======================================================================

def graphon_from_adjacency(A: List[List[int]]) -> Graphon:
    """
    Convenience wrapper: treat a finite adjacency matrix A as defining
    a step graphon with entries:

        M[i][j] = A[i][j] (clamped to [0,1])

    Equivalent to step_graphon_from_matrix, but typed for int matrices.
    """
    M = [[float(1 if v else 0) for v in row] for row in A]
    return step_graphon_from_matrix(M)
