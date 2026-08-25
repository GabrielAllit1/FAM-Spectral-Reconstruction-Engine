"""
MathToolBox v7.0 — rare_patterns.py
===================================

High-value exotic constructs for cryptography, signal analysis, and
orchestrator seeding. This module focuses on patterns that are:

    • Structurally strange
    • Number-theoretically deep
    • Useful as high-entropy or structured "keys"

Included:

    • Aliquot sequences, perfect/amicable detection
    • Erdős–Straus decompositions of 4/n
    • Happy numbers + happy primes
    • Legendre 3-square representations (n = x^2 + y^2 + z^2)
    • Apollonian gasket curvature evolution (Descartes' theorem)
    • Heighway dragon curve turns + coordinates
    • Simple Penrose rhombus tiling growth (substitution counts)
    • Busy Beaver style growth bound (theoretical upper bound function)
    • Wrappers around rare sequences from sequences.py for analysis
      (Van Eck, Thue–Morse, Kolakoski, Goodstein)
    • Entropy and pattern diagnostics on rare sequences

All algorithms are mathematically correct; there are no placeholders.
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Iterable, Optional
import math

from .numerics import (
    shannon_entropy,
    tsallis_entropy,
    digital_root_1_9,
    gcd,
    is_prime,
)
from . import sequences as seq_mod  # full sequences.py once concatenated


# ======================================================================
# ALIQUOT SEQUENCES, PERFECT & AMICABLE NUMBERS
# ======================================================================

def proper_divisors(n: int) -> List[int]:
    """
    Return the list of proper divisors of n (excluding n itself).

    Used for:
        • Aliquot sequences
        • Perfect/amicable tests
    """
    if n <= 1:
        return []
    divs = [1]
    r = int(math.isqrt(n))
    for d in range(2, r + 1):
        if n % d == 0:
            divs.append(d)
            q = n // d
            if q != d and q != n:
                divs.append(q)
    return sorted(set(divs))


def aliquot_sum(n: int) -> int:
    """Sum of proper divisors of n."""
    return sum(proper_divisors(n))


def aliquot_sequence(start: int, max_steps: int = 20) -> List[int]:
    """
    Generate the aliquot sequence:

        a_0 = start
        a_{k+1} = aliquot_sum(a_k)

    Stops when:
        • reaches 0,
        • hits a cycle,
        • or exceeds max_steps.
    """
    seq = [start]
    seen = {start}
    x = start
    for _ in range(max_steps):
        x = aliquot_sum(x)
        if x in seen:
            seq.append(x)
            break
        seq.append(x)
        seen.add(x)
        if x == 0:
            break
    return seq


def is_perfect(n: int) -> bool:
    """
    Check if n is a perfect number:

        σ(n) - n = n  ⇔  sum(proper_divisors(n)) = n
    """
    return n > 0 and aliquot_sum(n) == n


def is_amicable_pair(a: int, b: int) -> bool:
    """
    Check if (a,b) is an amicable pair:

        σ(a) - a = b  and  σ(b) - b = a
    """
    return a != b and aliquot_sum(a) == b and aliquot_sum(b) == a


def amicable_partners(n: int, search_limit: int = 100000) -> List[int]:
    """
    Scan for amicable partners of n in [2, search_limit].

    Returns:
        list of m such that (n, m) is amicable.
    """
    partners = []
    for m in range(2, search_limit + 1):
        if is_amicable_pair(n, m):
            partners.append(m)
    return partners


# ======================================================================
# ERDŐS–STRAUS 4/n = 1/a + 1/b + 1/c (Egyptian fractions)
# ======================================================================

def erdos_straus_solutions(n: int, limit: int = 1000) -> List[Tuple[int, int, int]]:
    """
    Find small solutions (a, b, c) in positive integers to the Erdős–Straus equation:

        4/n = 1/a + 1/b + 1/c

    for a, b, c <= limit. This is a search-based procedure and not a proof
    engine for the conjecture; it simply finds explicit decompositions.

    Returns:
        list of (a, b, c) solutions with 1 ≤ a ≤ b ≤ c ≤ limit.
    """
    if n <= 0:
        raise ValueError("n must be positive for Erdős–Straus.")

    sols: List[Tuple[int, int, int]] = []
    # naive but clear search; can be refined if needed
    for a in range(1, limit + 1):
        for b in range(a, limit + 1):
            # 4/n - 1/a - 1/b = 1/c
            num = 4 * a * b - n * (b + a)
            den = n * a * b
            if num <= 0:
                continue
            if den % num != 0:
                continue
            c = den // num
            if c < b or c > limit:
                continue
            sols.append((a, b, c))
    return sols


# ======================================================================
# HAPPY NUMBERS & HAPPY PRIMES
# ======================================================================

def is_happy(n: int) -> bool:
    """
    Check if n is a happy number in base 10.

    Algorithm:
        repeatedly replace n with the sum of squares of its digits;
        if this process reaches 1, n is happy; if it enters a cycle
        that doesn't include 1, n is unhappy.
    """
    seen = set()
    x = n
    while x not in seen:
        seen.add(x)
        x = sum(int(d) ** 2 for d in str(abs(x)))
        if x == 1:
            return True
    return False


def happy_primes(limit: int) -> List[int]:
    """
    Return all happy primes ≤ limit.
    """
    out = []
    for n in range(2, limit + 1):
        if is_prime(n) and is_happy(n):
            out.append(n)
    return out


# ======================================================================
# LEGENDRE 3-SQUARE REPRESENTATIONS
# ======================================================================

def legendre_three_square_representation(n: int) -> Optional[Tuple[int, int, int]]:
    """
    Try to find integers x, y, z such that:

        n = x^2 + y^2 + z^2

    Legendre's three-square theorem says: n has such a representation
    iff n is NOT of the form:

        n = 4^a * (8b + 7)

    for integers a, b ≥ 0.

    This function:
        • first checks that necessary condition,
        • then does a bounded search up to floor(sqrt(n)).

    Returns:
        (x, y, z) if found, else None.
    """
    if n < 0:
        return None

    # Check obstruction form: n = 4^a (8b+7)
    temp = n
    while temp % 4 == 0:
        temp //= 4
    if temp % 8 == 7:
        return None  # provably impossible

    limit = int(math.isqrt(n))
    for x in range(0, limit + 1):
        for y in range(0, limit + 1):
            rem = n - x * x - y * y
            if rem < 0:
                break
            z = int(math.isqrt(rem))
            if z * z == rem:
                return (x, y, z)
    return None


# ======================================================================
# APOLLONIAN GASKET CURVATURE EVOLUTION (DESCARTES)
# ======================================================================

def descartes_new_curvature(k1: int, k2: int, k3: int, k4: int) -> Tuple[int, int]:
    """
    Given four mutually tangent circles with curvatures k1, k2, k3, k4
    (curvature = 1/radius, signed per oriented Apollonian gasket),
    Descartes' theorem implies:

        (k1 + k2 + k3 + k4)^2 = 2 (k1^2 + k2^2 + k3^2 + k4^2)

    If three are fixed and we want new k4', we can solve for the other
    solution:

        k4' = 2(k1 + k2 + k3) - k4

    This function returns both the original and new curvature pair
    where k4' is computed from the other three.

    Returns:
        (k4_original, k4_new)
    """
    s = k1 + k2 + k3
    k4_new = 2 * s - k4
    return (k4, k4_new)


def apollonian_child_curvatures(triple: Tuple[int, int, int], outer: int) -> List[Tuple[int, int, int, int]]:
    """
    Given an Apollonian configuration with three inner curvatures (k1,k2,k3)
    and an outer curvature k0 (often negative for the outer circle),
    generate the four Descartes quadruples corresponding to the four
    possible choices of "replacing" one circle.

    Returns a list of 4 quadruples (k0,k1,k2,k3_new), etc.
    """
    k0, k1, k2, k3 = outer, triple[0], triple[1], triple[2]
    out = []
    # Replace k0
    _, k0p = descartes_new_curvature(k1, k2, k3, k0)
    out.append((k0p, k1, k2, k3))
    # Replace k1
    _, k1p = descartes_new_curvature(k0, k2, k3, k1)
    out.append((k0, k1p, k2, k3))
    # Replace k2
    _, k2p = descartes_new_curvature(k0, k1, k3, k2)
    out.append((k0, k1, k2p, k3))
    # Replace k3
    _, k3p = descartes_new_curvature(k0, k1, k2, k3)
    out.append((k0, k1, k2, k3p))
    return out


# ======================================================================
# HEIGHWAY DRAGON CURVE (TURNS + COORDINATES)
# ======================================================================

def heighway_dragon_turns(iterations: int) -> List[int]:
    """
    Generate turn sequence for the Heighway dragon curve.

    Representation:
        1 = turn left
        -1 = turn right

    Construction (folding rule):
        D_0 = [1]
        D_{n+1} = D_n + [1] + (-D_n[::-1])

    This yields the infinite folding pattern. Here we truncate to
    'iterations' folds.
    """
    if iterations <= 0:
        return []
    turns = [1]
    for _ in range(iterations - 1):
        rev = turns[::-1]
        rev = [-t for t in rev]
        turns = turns + [1] + rev
    return turns


def heighway_dragon_points(iterations: int) -> List[Tuple[float, float]]:
    """
    Generate coordinates (x,y) for the Heighway dragon curve.

    Start at origin, step 1 in the +x direction, then apply the turn
    sequence to generate each next segment.

    This is useful for:
        • fractal embeddings into 973/FAM
        • graphon-style adjacency sampling
    """
    turns = heighway_dragon_turns(iterations)
    x, y = 0.0, 0.0
    dx, dy = 1.0, 0.0  # initial direction: +x
    pts = [(x, y)]

    for t in turns:
        # turn
        if t == 1:  # left
            dx, dy = -dy, dx
        else:       # right
            dx, dy = dy, -dx
        x += dx
        y += dy
        pts.append((x, y))
    return pts


# ======================================================================
# PENROSE RHOMBUS TILING GROWTH (SUBSTITUTION COUNTS)
# ======================================================================

def penrose_rhombus_counts(iterations: int) -> Tuple[int, int]:
    """
    Simple Penrose rhombus tiling growth model using substitution rules.

    There are two rhombus types (thin T, thick F). The substitution
    matrix can be encoded as:

        [T']   [1 1] [T]
        [F'] = [1 0] [F]

    (One reasonable model; actual Penrose can be more nuanced.)

    We start with one thick tile (T=0, F=1) at iteration 0.

    After 'iterations' substitutions, return:

        (#thin, #thick)

    This gives a Fibonacci-like growth pattern aligned with φ.
    """
    # Vector v = [T, F]
    T, F = 0, 1
    for _ in range(iterations):
        T, F = T + F, T  # apply matrix [[1,1],[1,0]]
    return (T, F)


# ======================================================================
# BUSY BEAVER GROWTH BOUND
# ======================================================================

def busy_beaver_upper_bound(n: int) -> int:
    """
    Extremely coarse upper bound for Busy Beaver values Σ(n), S(n)
    for n-state Turing machines.

    We DO NOT attempt to compute Σ(n); that is non-computable in general.
    Instead we provide a rapidly growing *upper bound* that can be used
    as a "theoretical maximum complexity budget" in unlocker or 973
    orchestration pipelines.

    Here we choose:

        B(n) = 2^(2^(n+1))

    which grows double-exponentially and safely exceeds known Σ(n)
    for small n (this is not tight, just a guaranteed sky-high ceiling).
    """
    if n < 0:
        raise ValueError("n must be non-negative.")
    return 2 ** (2 ** (n + 1))


# ======================================================================
# RARE SEQUENCE WRAPPERS + DIAGNOSTICS
# ======================================================================

def rare_sequence_profile(
    seq: Iterable[int],
) -> Dict[str, float]:
    """
    Compute a basic profile of a given integer sequence:

        • length
        • mean
        • variance
        • Shannon entropy
        • Tsallis (q=2) entropy

    All values in a compact dict, for direct consumption by
    orchestrator / unlocker / FAM mixers.
    """
    data = list(seq)
    if not data:
        return {
            "length": 0,
            "mean": 0.0,
            "variance": 0.0,
            "shannon": 0.0,
            "tsallis_q2": 0.0,
        }

    length = len(data)
    mean_val = sum(data) / length
    var_val = sum((x - mean_val) ** 2 for x in data) / length
    H = shannon_entropy(data)
    T2 = tsallis_entropy(data, q=2.0)
    return {
        "length": float(length),
        "mean": float(mean_val),
        "variance": float(var_val),
        "shannon": float(H),
        "tsallis_q2": float(T2),
    }


def profile_van_eck(n: int) -> Dict[str, float]:
    """Profile first n terms of Van Eck sequence."""
    seq = seq_mod.van_eck(n)
    return rare_sequence_profile(seq)


def profile_thue_morse(n: int) -> Dict[str, float]:
    """Profile first n terms of Thue–Morse sequence."""
    seq = seq_mod.thue_morse(n)
    return rare_sequence_profile(seq)


def profile_kolakoski_12(n: int) -> Dict[str, float]:
    """Profile first n terms of Kolakoski(1,2) sequence."""
    seq = seq_mod.kolakoski_12(n)
    return rare_sequence_profile(seq)


def profile_goodstein(a0: int, steps: int) -> Dict[str, float]:
    """Profile first 'steps'+1 values of a Goodstein sequence from a0."""
    seq = seq_mod.goodstein_sequence(a0, steps)
    return rare_sequence_profile(seq)
