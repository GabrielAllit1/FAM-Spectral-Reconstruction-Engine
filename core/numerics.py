# mathtoolbox_v7/core/numerics.py
# Complete numerics engine — real math, zero placeholders

from __future__ import annotations
from typing import List, Iterable, Tuple, Dict, Callable
import math
import statistics


# ======================================================================
# DIGITAL ROOT SYSTEM (Exact mod-9 arithmetic)
# ======================================================================

def digital_root_1_9(n: int) -> int:
    if n <= 0:
        n = abs(n)
    return 1 + ((n - 1) % 9)

def digital_root_sequence(seq: Iterable[int]) -> List[int]:
    return [digital_root_1_9(abs(int(x))) for x in seq]


# ======================================================================
# GENERALIZED MOD-X ROOT SYSTEM
# ======================================================================

def mod_root(n: int, modulus: int) -> int:
    if modulus <= 0:
        raise ValueError("modulus must be positive.")
    r = n % modulus
    return modulus if r == 0 else r

def mod_root_sequence(seq: Iterable[int], modulus: int) -> List[int]:
    return [mod_root(int(x), modulus) for x in seq]


# ======================================================================
# p-ADIC VALUATION & NORM
# ======================================================================

def p_adic_valuation(n: int, p: int) -> int:
    if p <= 1:
        raise ValueError("p must be a prime or integer > 1.")
    if n == 0:
        return math.inf
    n = abs(n)
    k = 0
    while n % p == 0:
        n //= p
        k += 1
    return k

def p_adic_norm(n: int, p: int) -> float:
    if n == 0:
        return 0.0
    vp = p_adic_valuation(n, p)
    return p ** (-vp)


# ======================================================================
# ENTROPY SYSTEMS
# ======================================================================

def shannon_entropy(seq: Iterable[int]) -> float:
    data = list(seq)
    if not data:
        return 0.0
    freq: Dict[int, int] = {}
    for x in data:
        freq[x] = freq.get(x, 0) + 1
    total = len(data)
    H = 0.0
    for c in freq.values():
        p = c / total
        H -= p * math.log2(p)
    return H

def renyi_entropy(seq: Iterable[int], alpha: float = 2.0) -> float:
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    if abs(alpha - 1.0) < 1e-12:
        return shannon_entropy(seq)
    data = list(seq)
    freq: Dict[int, int] = {}
    for x in data:
        freq[x] = freq.get(x, 0) + 1
    total = len(data)
    s = 0.0
    for c in freq.values():
        p = c / total
        s += p ** alpha
    return (1.0 / (1.0 - alpha)) * math.log2(s)

def tsallis_entropy(seq: Iterable[int], q: float = 2.0) -> float:
    if abs(q - 1.0) < 1e-12:
        return shannon_entropy(seq)
    data = list(seq)
    freq: Dict[int, int] = {}
    for x in data:
        freq[x] = freq.get(x, 0) + 1
    total = len(data)
    sum_pq = 0.0
    for c in freq.values():
        p = c / total
        sum_pq += p ** q
    return (1 - sum_pq) / (q - 1)


# ======================================================================
# MODULAR ARITHMETIC + NUMBER THEORY
# ======================================================================

def mod_add(a: int, b: int, m: int) -> int:
    return (a + b) % m

def mod_sub(a: int, b: int, m: int) -> int:
    return (a - b) % m

def mod_mul(a: int, b: int, m: int) -> int:
    return (a * b) % m

def mod_pow(base: int, exp: int, m: int) -> int:
    return pow(base, exp, m)

def mod_inv(a: int, m: int) -> int:
    g, x, y = _extended_gcd(a, m)
    if g != 1:
        raise ValueError(f"No modular inverse for a={a} mod m={m}.")
    return x % m

def _extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    if b == 0:
        return (a, 1, 0)
    g, x1, y1 = _extended_gcd(b, a % b)
    return (g, y1, x1 - (a // b) * y1)

def euler_phi(n: int) -> int:
    if n <= 0:
        raise ValueError("n must be positive.")
    result = n
    p = 2
    temp = n
    while p * p <= temp:
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
            result -= result // p
        p += 1
    if temp > 1:
        result -= result // temp
    return result

def mobius_mu(n: int) -> int:
    if n == 0:
        return 0
    n = abs(n)
    prime_factors = 0
    p = 2
    temp = n
    while p * p <= temp:
        if temp % p == 0:
            prime_factors += 1
            temp //= p
            if temp % p == 0:
                return 0
        p += 1
    if temp > 1:
        prime_factors += 1
    return -1 if (prime_factors % 2) else 1

def mobius_weighted_root(n: int) -> int:
    μ = mobius_mu(n)
    if μ == 0:
        return 9
    return ((abs(μ) * abs(n) - 1) % 9) + 1


# ======================================================================
# CONTINUED FRACTIONS, DIOPHANTINE, PELL, LLL
# ======================================================================

def continued_fraction(x: float, max_terms: int = 32) -> List[int]:
    result = []
    for _ in range(max_terms):
        a = int(math.floor(x))
        result.append(a)
        frac = x - a
        if abs(frac) < 1e-15:
            break
        x = 1.0 / frac
    return result

def convergents(cf: List[int]) -> List[Tuple[int, int]]:
    p_minus2, p_minus1 = 0, 1
    q_minus2, q_minus1 = 1, 0
    out = []
    for a in cf:
        p = a * p_minus1 + p_minus2
        q = a * q_minus1 + q_minus2
        out.append((p, q))
        p_minus2, p_minus1 = p_minus1, p
        q_minus2, q_minus1 = q_minus1, q
    return out

def solve_linear_diophantine(a: int, b: int, c: int) -> Tuple[int, int]:
    g, x, y = _extended_gcd(a, b)
    if c % g != 0:
        raise ValueError(f"No integer solutions for {a}x + {b}y = {c}.")
    factor = c // g
    return (x * factor, y * factor)

def pell_solve(D: int) -> Tuple[int, int]:
    if int(math.sqrt(D))**2 == D:
        raise ValueError("D must be a non-square integer.")
    cf = continued_fraction(math.sqrt(D), max_terms=256)
    period = cf[1:]
    if len(period) % 2 == 1:
        cf_expanded = cf[:1] + period * 2
    else:
        cf_expanded = cf
    conv = convergents(cf_expanded)
    x, y = conv[-1]
    if x*x - D*y*y != 1:
        raise RuntimeError("Failed to compute Pell solution.")
    return (x, y)

def lll_reduce(B: List[List[float]], delta: float = 0.75) -> List[List[float]]:
    def dot(u, v):
        return sum(ui * vi for ui, vi in zip(u, v))
    n = len(B)
    B = [vec[:] for vec in B]
    Bstar = [B[0][:]]
    for i in range(1, n):
        Bstar.append(B[i][:])
    k = 1
    while k < n:
        for j in range(k - 1, -1, -1):
            mu_kj = dot(B[k], Bstar[j]) / dot(Bstar[j], Bstar[j])
            if abs(mu_kj) > 0.5:
                correction = [round(mu_kj) * bi for bi in B[j]]
                B[k] = [bi - ci for bi, ci in zip(B[k], correction)]
        left = dot(Bstar[k], Bstar[k])
        right = (delta - mu_kj ** 2) * dot(Bstar[k-1], Bstar[k-1])
        if left >= right:
            k += 1
        else:
            B[k], B[k-1] = B[k-1], B[k]
            k = max(k - 1, 1)
        Bstar = [B[0][:]]
        for i in range(1, n):
            v = B[i][:]
            for j in range(i):
                mu_ij = dot(v, Bstar[j]) / dot(Bstar[j], Bstar[j])
                v = [vi - mu_ij * bsj for vi, bsj in zip(v, Bstar[j])]
            Bstar.append(v[:])
    return B


# ======================================================================
# UTILS
# ======================================================================

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    r = int(math.sqrt(n))
    for i in range(3, r + 1, 2):
        if n % i == 0:
            return False
    return True

def gcd(a: int, b: int) -> int:
    return math.gcd(a, b)

def lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b)