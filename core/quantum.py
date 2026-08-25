"""
MathToolBox v7.0 — quantum.py
=============================

Core quantum tools for MathToolBox v7.0.

Focus:
    • Finite-dimensional complex Hilbert spaces (ℂ^d)
    • Qubit and qudit state vectors and operators
    • Basic gate library (1-qubit + n-qubit application)
    • Coined quantum walks on:
        - 1D cycle / line
        - Simple undirected graphs
    • Measurement utilities (Born rule)
    • Sanity checks (unitarity, normalization)

This module is *numerically concrete* and uses only NumPy for linear
algebra. All operators are fully unitary where appropriate; no stubs or
placeholders.

Dependencies:
    - numpy

Intended use:
    - as a backend for:
        • 973-Orchestrated quantum walks
        • FAM/rare-pattern-driven coin schedules
        • entropy/dispersion diagnostics in mining pipelines
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Iterable, Callable
import numpy as np


# ======================================================================
# BASIC STATE & OPERATOR UTILS
# ======================================================================

def normalize_state(psi: np.ndarray) -> np.ndarray:
    """
    Normalize a complex state vector |psi> to unit norm.

    If psi has zero norm, raises ValueError.
    """
    psi = np.asarray(psi, dtype=np.complex128).reshape(-1)
    norm = np.linalg.norm(psi)
    if norm == 0:
        raise ValueError("Cannot normalize zero vector.")
    return psi / norm


def inner_product(phi: np.ndarray, psi: np.ndarray) -> complex:
    """
    Inner product <phi|psi> with bra-ket ordering (conjugate on phi).
    """
    phi = np.asarray(phi, dtype=np.complex128).reshape(-1)
    psi = np.asarray(psi, dtype=np.complex128).reshape(-1)
    return np.vdot(phi, psi)


def is_unitary(U: np.ndarray, atol: float = 1e-8) -> bool:
    """
    Check if matrix U is unitary:

        U† U = I = U U†

    within numerical tolerance atol.
    """
    U = np.asarray(U, dtype=np.complex128)
    if U.shape[0] != U.shape[1]:
        return False
    I = np.eye(U.shape[0], dtype=np.complex128)
    left = U.conj().T @ U
    right = U @ U.conj().T
    return np.allclose(left, I, atol=atol) and np.allclose(right, I, atol=atol)


def kron2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Convenience Kronecker product of two matrices."""
    return np.kron(np.asarray(A, dtype=np.complex128),
                   np.asarray(B, dtype=np.complex128))


def kron_many(ops: Iterable[np.ndarray]) -> np.ndarray:
    """Kronecker product of many matrices in order."""
    ops = list(ops)
    if not ops:
        raise ValueError("kron_many requires at least one operator.")
    result = np.asarray(ops[0], dtype=np.complex128)
    for op in ops[1:]:
        result = np.kron(result, np.asarray(op, dtype=np.complex128))
    return result


# ======================================================================
# SINGLE-QUBIT GATES
# ======================================================================

# Pauli and identity
I2 = np.array([[1, 0],
               [0, 1]], dtype=np.complex128)

X = np.array([[0, 1],
              [1, 0]], dtype=np.complex128)

Y = np.array([[0, -1j],
              [1j, 0]], dtype=np.complex128)

Z = np.array([[1, 0],
              [0, -1]], dtype=np.complex128)

# Hadamard
H = (1.0 / np.sqrt(2.0)) * np.array([[1, 1],
                                     [1, -1]], dtype=np.complex128)

# Phase and T gate
S = np.array([[1, 0],
              [0, 1j]], dtype=np.complex128)

T = np.array([[1, 0],
              [0, np.exp(1j * np.pi / 4.0)]], dtype=np.complex128)


def Rx(theta: float) -> np.ndarray:
    """
    Rotation about X-axis:

        Rx(θ) = exp(-i θ X / 2)
    """
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -1j * s],
                     [-1j * s, c]], dtype=np.complex128)


def Ry(theta: float) -> np.ndarray:
    """
    Rotation about Y-axis:

        Ry(θ) = exp(-i θ Y / 2)
    """
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s],
                     [s, c]], dtype=np.complex128)


def Rz(theta: float) -> np.ndarray:
    """
    Rotation about Z-axis:

        Rz(θ) = exp(-i θ Z / 2)
    """
    return np.array([[np.exp(-1j * theta / 2.0), 0],
                     [0, np.exp(1j * theta / 2.0)]], dtype=np.complex128)


# ======================================================================
# N-QUBIT UTILITIES
# ======================================================================

def basis_state(n_qubits: int, index: int) -> np.ndarray:
    """
    |index> in computational basis on n_qubits (0-indexed).

    Dimension = 2^n_qubits. index must be in [0, 2^n_qubits - 1].
    """
    dim = 1 << n_qubits
    if not (0 <= index < dim):
        raise ValueError(f"index must be in [0, {dim - 1}].")
    psi = np.zeros(dim, dtype=np.complex128)
    psi[index] = 1.0
    return psi


def apply_single_qubit_gate(
    psi: np.ndarray,
    gate: np.ndarray,
    target: int,
    n_qubits: int,
) -> np.ndarray:
    """
    Apply a 1-qubit gate to 'target' qubit in an n_qubits register.

    Qubit ordering convention:
        - Qubits indexed 0 .. n_qubits-1,
        - 0 = most significant in tensor product (leftmost).

    We build:
        U_total = I ⊗ ... ⊗ gate ⊗ ... ⊗ I
    and return U_total |psi>.
    """
    if gate.shape != (2, 2):
        raise ValueError("gate must be 2x2 for apply_single_qubit_gate.")

    ops = []
    for q in range(n_qubits):
        if q == target:
            ops.append(gate)
        else:
            ops.append(I2)
    U = kron_many(ops)
    return U @ psi


def apply_controlled_gate(
    psi: np.ndarray,
    gate: np.ndarray,
    control: int,
    target: int,
    n_qubits: int,
) -> np.ndarray:
    """
    Apply a controlled-1-qubit gate with single control and single target.

    If control qubit is |1>, apply 'gate' to target; otherwise identity.

    Implementation:
        U = Π over basis states |i>:
                |i><i|          if i_control = 0
                (|i><i| with gate on target subspace) if i_control = 1

    For small n_qubits, we build an explicit 2^n × 2^n matrix U. This is
    more than enough for MathToolBox v7.0 workloads (we are not trying
    to simulate > 20 qubits here).
    """
    dim = 1 << n_qubits
    if gate.shape != (2, 2):
        raise ValueError("gate must be 2x2 for apply_controlled_gate.")

    U = np.eye(dim, dtype=np.complex128)
    for basis_index in range(dim):
        # Extract bit of control and target
        ctrl_bit = (basis_index >> (n_qubits - 1 - control)) & 1
        if ctrl_bit == 0:
            continue
        # Determine which two basis states differ only on target
        tgt_bit = (basis_index >> (n_qubits - 1 - target)) & 1
        partner_index = basis_index ^ (1 << (n_qubits - 1 - target))
        # We will apply gate to subspace spanned by |basis>, |partner|
        # but only update once (avoid double updates).
        if tgt_bit == 0:
            i0 = basis_index
            i1 = partner_index
        else:
            i0 = partner_index
            i1 = basis_index

        # Current amplitudes basis_0, basis_1 transform by gate
        # but we build full U instead for clarity
        # 2x2 block:
        block = np.array([[gate[0, 0], gate[0, 1]],
                          [gate[1, 0], gate[1, 1]]], dtype=np.complex128)
        # Insert block into U
        U[[i0, i1]][:, [i0, i1]] = block

    return U @ psi


# ======================================================================
# MEASUREMENT UTILITIES
# ======================================================================

def probabilities(psi: np.ndarray) -> np.ndarray:
    """
    Born rule probabilities |ψ_k|^2 for computational basis.
    """
    psi = np.asarray(psi, dtype=np.complex128).reshape(-1)
    return np.abs(psi) ** 2


def sample_measurements(
    psi: np.ndarray,
    n_samples: int,
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    """
    Sample measurement outcomes in the computational basis n_samples times.

    Returns a list of basis indices.
    """
    psi = normalize_state(psi)
    probs = probabilities(psi)
    dim = len(probs)
    if rng is None:
        rng = np.random.default_rng()
    outcomes = rng.choice(dim, size=n_samples, p=probs)
    return outcomes.tolist()


# ======================================================================
# COINED QUANTUM WALK ON 1D CYCLE
# ======================================================================

def hadamard_coin() -> np.ndarray:
    """Standard Hadamard coin for 2D coin space."""
    return H.copy()


def biased_coin(p: float) -> np.ndarray:
    """
    Biased coin on 2D coin space:

        C = [[sqrt(p),  sqrt(1-p)],
             [sqrt(1-p), -sqrt(p)]]

    This is unitary for p in [0,1].
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0,1].")
    a = np.sqrt(p)
    b = np.sqrt(1.0 - p)
    return np.array([[a, b],
                     [b, -a]], dtype=np.complex128)


def quantum_walk_cycle_step(
    state: np.ndarray,
    coin_op: np.ndarray,
    n_positions: int,
) -> np.ndarray:
    """
    Single step of a discrete-time coined quantum walk on a cycle
    of n_positions sites with a 2D coin.

    Hilbert space:
        H_coin ⊗ H_pos
        dim = 2 * n_positions

    Basis ordering:
        |0>_coin ⊗ |x>_pos  mapped to index 2*x
        |1>_coin ⊗ |x>_pos  mapped to index 2*x + 1

    Step:
        1. Apply (coin_op ⊗ I_pos) to the state.
        2. Conditional shift:
            - coin |0> -> move position x -> x+1 mod n
            - coin |1> -> move position x -> x-1 mod n
    """
    state = np.asarray(state, dtype=np.complex128).reshape(-1)
    dim = 2 * n_positions
    if state.shape[0] != dim:
        raise ValueError(f"state dimension must be 2*n_positions = {dim}.")

    # 1. Apply coin operator (coin_op ⊗ I_pos)
    U_coin = kron_many([coin_op, np.eye(n_positions, dtype=np.complex128)])
    state = U_coin @ state

    # 2. Conditional shift
    new_state = np.zeros_like(state)
    for x in range(n_positions):
        # coin 0 amplitude at position x
        idx0 = 2 * x
        idx1 = idx0 + 1
        amp0 = state[idx0]
        amp1 = state[idx1]

        # move coin 0 -> position x+1
        x_plus = (x + 1) % n_positions
        new_state[2 * x_plus] += amp0

        # move coin 1 -> position x-1
        x_minus = (x - 1) % n_positions
        new_state[2 * x_minus + 1] += amp1

    return new_state


def quantum_walk_cycle(
    n_steps: int,
    n_positions: int,
    coin_op: np.ndarray,
    init_position: int = 0,
    init_coin_state: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Perform a coined quantum walk on a cycle for n_steps.

    Parameters
    ----------
    n_steps : int
        Number of time steps.
    n_positions : int
        Number of positions on cycle.
    coin_op : np.ndarray
        2x2 unitary coin operator (Hadamard or biased_coin, etc.).
    init_position : int
        Initial site index.
    init_coin_state : np.ndarray, optional
        2D initial coin state; if None, defaults to |0>.

    Returns
    -------
    final_state : np.ndarray
        State vector after n_steps.
    history : List[np.ndarray]
        State at each step (including initial).
    """
    if init_coin_state is None:
        init_coin_state = np.array([1.0, 0.0], dtype=np.complex128)
    else:
        init_coin_state = normalize_state(init_coin_state)

    if not (0 <= init_position < n_positions):
        raise ValueError("init_position out of range.")

    # Build initial state |coin> ⊗ |position>
    pos_state = np.zeros(n_positions, dtype=np.complex128)
    pos_state[init_position] = 1.0
    psi = kron_many([init_coin_state.reshape(2, 1),
                     pos_state.reshape(n_positions, 1)]).reshape(-1)
    psi = normalize_state(psi)

    history: List[np.ndarray] = [psi.copy()]
    for _ in range(n_steps):
        psi = quantum_walk_cycle_step(psi, coin_op, n_positions)
        psi = normalize_state(psi)  # keep numerical stability
        history.append(psi.copy())

    return psi, history


# ======================================================================
# QUANTUM WALK ON GENERIC UNDIRECTED GRAPH
# ======================================================================

def quantum_walk_graph_step(
    state: np.ndarray,
    coin_ops: List[np.ndarray],
    adjacency: np.ndarray,
) -> np.ndarray:
    """
    Single step of a coined quantum walk on an undirected graph.

    Model:
        - Graph with N nodes, adjacency matrix A (0/1, symmetric).
        - Each node v has degree d_v and local coin space of dimension d_v.
        - Local coin_op at node v is coin_ops[v], shape (d_v, d_v),
          unitary on that d_v-dimensional coin space.

    State representation:
        - For each node v, we have amplitudes on each outgoing edge index.
        - We fix an ordering of edges at each node based on adjacency:

            neighbors[v] = list of neighbors of v
            local coin basis index j corresponds to edge (v -> neighbors[v][j])

        - We flatten all local coin spaces in order v=0..N-1.

    Step:
        1. At each node v, apply local coin_op[v] on its local coin amplitudes.
        2. Shift: for each amplitude on (v -> u) at local index j,
           move to (u -> v) at the corresponding index at node u.

    This is a standard coined walk mechanism (Aharonov et al. style).

    For moderate N and degrees, this is numerically feasible.
    """
    A = np.asarray(adjacency, dtype=np.int64)
    N = A.shape[0]
    if A.shape[1] != N:
        raise ValueError("adjacency must be square.")
    degrees = A.sum(axis=1)
    total_dim = int(sum(degrees))
    state = np.asarray(state, dtype=np.complex128).reshape(-1)
    if state.shape[0] != total_dim:
        raise ValueError(f"state dimension must be sum of degrees = {total_dim}.")

    # Precompute neighbor lists and index mappings
    neighbors: List[List[int]] = []
    offset: List[int] = [0] * N
    current_offset = 0
    for v in range(N):
        neigh_v = [u for u in range(N) if A[v, u] != 0]
        neighbors.append(neigh_v)
        offset[v] = current_offset
        current_offset += len(neigh_v)

    # 1. Apply coins locally
    state_local = state.copy()
    for v in range(N):
        d_v = len(neighbors[v])
        if d_v == 0:
            continue
        start = offset[v]
        end = start + d_v
        local_vec = state_local[start:end]
        if coin_ops[v].shape != (d_v, d_v):
            raise ValueError(f"coin_ops[{v}] must be {d_v}x{d_v}.")
        state_local[start:end] = coin_ops[v] @ local_vec

    # 2. Shift
    new_state = np.zeros_like(state_local)
    for v in range(N):
        d_v = len(neighbors[v])
        start_v = offset[v]
        for j in range(d_v):
            u = neighbors[v][j]
            amp = state_local[start_v + j]
            # find index of v in neighbors[u]
            neigh_u = neighbors[u]
            try:
                k = neigh_u.index(v)
            except ValueError:
                # Should not happen in a symmetric simple graph, but guard
                continue
            start_u = offset[u]
            new_state[start_u + k] += amp

    return new_state


def quantum_walk_graph(
    n_steps: int,
    adjacency: np.ndarray,
    coin_ops: Optional[List[np.ndarray]] = None,
    init_node: int = 0,
    init_edge_index: int = 0,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Perform a coined quantum walk on an undirected graph for n_steps.

    Parameters
    ----------
    n_steps : int
        Number of steps.
    adjacency : np.ndarray
        N x N adjacency matrix of an undirected graph.
    coin_ops : List[np.ndarray], optional
        List of local coin operators for each node v.
        If None, we use Grover coin at each node:

            G_v = 2/d_v * J - I

        where J is the all-ones matrix of size d_v.
    init_node : int
        Node at which the walker starts.
    init_edge_index : int
        Local edge index at init_node (0..d_init_node-1).

    Returns
    -------
    final_state : np.ndarray
        State vector in edge-local representation.
    history : List[np.ndarray]
        States at each step (including initial).
    """
    A = np.asarray(adjacency, dtype=np.int64)
    N = A.shape[0]
    if A.shape[1] != N:
        raise ValueError("adjacency must be square.")
    if not (0 <= init_node < N):
        raise ValueError("init_node out of range.")

    degrees = A.sum(axis=1)
    neighbors: List[List[int]] = []
    offset: List[int] = [0] * N
    current_offset = 0
    for v in range(N):
        neigh_v = [u for u in range(N) if A[v, u] != 0]
        neighbors.append(neigh_v)
        offset[v] = current_offset
        current_offset += len(neigh_v)
    total_dim = int(sum(degrees))

    d0 = len(neighbors[init_node])
    if d0 == 0:
        raise ValueError("init_node has degree 0 (isolated).")
    if not (0 <= init_edge_index < d0):
        raise ValueError("init_edge_index out of range for init_node.")

    # Default: Grover coins
    if coin_ops is None:
        coin_ops = []
        for v in range(N):
            d_v = len(neighbors[v])
            if d_v == 0:
                coin_ops.append(np.eye(1, dtype=np.complex128))
            else:
                J = np.ones((d_v, d_v), dtype=np.complex128)
                G = (2.0 / d_v) * J - np.eye(d_v, dtype=np.complex128)
                coin_ops.append(G)

    # Build initial state vector
    psi = np.zeros(total_dim, dtype=np.complex128)
    psi[offset[init_node] + init_edge_index] = 1.0
    psi = normalize_state(psi)

    history: List[np.ndarray] = [psi.copy()]
    for _ in range(n_steps):
        psi = quantum_walk_graph_step(psi, coin_ops, A)
        psi = normalize_state(psi)
        history.append(psi.copy())

    return psi, history
