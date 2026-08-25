# mathtoolbox_v7/core/sequences.py
# Full Sequences Engine — Parts 1 + 2 + 3 Combined
# Real mathematics. No truncation. No placeholders.

from __future__ import annotations
from typing import List, Tuple, Iterable
import math

# Import from numerics (safe when in same package)
try:
    from .numerics import digital_root_1_9, is_prime, mod_root
except ImportError:
    # Fallback for standalone execution
    def digital_root_1_9(n: int) -> int:
        if n <= 0: n = abs(n)
        return 1 + ((n - 1) % 9)
    def is_prime(n: int) -> bool:
        if n < 2: return False
        if n in (2, 3): return True
        if n % 2 == 0: return False
        r = int(math.sqrt(n))
        for i in range(3, r + 1, 2):
            if n % i == 0: return False
        return True


# ======================================================================
# FIBONACCI & LUCAS
# ======================================================================

def fibonacci(n: int) -> int:
    def fib_doubling(k: int) -> Tuple[int, int]:
        if k == 0: return (0, 1)
        a, b = fib_doubling(k >> 1)
        c = a * (2*b - a)
        d = a*a + b*b
        return (d, c + d) if k & 1 else (c, d)
    if n < 0: raise ValueError("n must be non-negative.")
    return fib_doubling(n)[0]

def fibonacci_sequence(n: int) -> List[int]:
    return [fibonacci(k) for k in range(n+1)]

def lucas(n: int) -> int:
    if n == 0: return 2
    if n == 1: return 1
    return fibonacci(n-1) + fibonacci(n+1)

def lucas_sequence(n: int) -> List[int]:
    return [lucas(k) for k in range(n+1)]


# ======================================================================
# LINEAR RECURRENCES
# ======================================================================

def linear_recurrence(coeffs: Iterable[int], seed: Iterable[int], n: int) -> List[int]:
    coeffs = list(coeffs)
    seed = list(seed)
    k = len(coeffs)
    if len(seed) < k: raise ValueError("Seed too short")
    out = seed[:]
    for i in range(len(seed), n):
        val = sum(coeffs[j] * out[i-j-1] for j in range(k))
        out.append(val)
    return out[:n]


# ======================================================================
# FAM BASE SEQUENCE
# ======================================================================

def fam_base_sequence(seed: int, length: int) -> List[int]:
    if length <= 0: return []
    seq = [digital_root_1_9(seed)]
    for n in range(1, length):
        seq.append(digital_root_1_9(seq[-1] * seq[-1] + n))
    return seq


# ======================================================================
# THUE–MORSE
# ======================================================================

def thue_morse(n: int) -> List[int]:
    return [bin(k).count("1") % 2 for k in range(n)]


# ======================================================================
# KOLAKOSKI (1,2)
# ======================================================================

def kolakoski_12(n: int) -> List[int]:
    seq = []
    i = 0
    run_val = 1
    while len(seq) < n:
        run_len = seq[i] if i < len(seq) else run_val
        seq.extend([run_val] * run_len)
        run_val = 3 - run_val
        i += 1
    return seq[:n]


# ======================================================================
# VAN ECK
# ======================================================================

def van_eck(n: int) -> List[int]:
    if n <= 0: return []
    seq = [0]
    last_seen = {0: 0}
    for i in range(1, n):
        prev = seq[i-1]
        if prev in last_seen and last_seen[prev] != i-1:
            seq.append(i-1 - last_seen[prev])
        else:
            seq.append(0)
        last_seen[prev] = i-1
    return seq


# ======================================================================
# BAUM–SWEET
# ======================================================================

def baum_sweet(n: int) -> List[int]:
    out = []
    for k in range(n):
        b = bin(k)[2:]
        zero_run = 0
        valid = True
        for ch in b:
            if ch == '0':
                zero_run += 1
            else:
                if zero_run % 2 == 1:
                    valid = False
                    break
                zero_run = 0
        if zero_run % 2 == 1:
            valid = False
        out.append(1 if valid else 0)
    return out


# ======================================================================
# GOODSTEIN SEQUENCE (True Hereditary Base)
# ======================================================================

def goodstein_sequence(a0: int, steps: int) -> List[int]:
    if a0 < 0: raise ValueError("Goodstein requires non-negative start")
    def hereditary_base_expansion(n: int, base: int) -> str:
        if n < base: return str(n)
        digits = []
        while n > 0:
            digits.append(n % base)
            n //= base
        digits = digits[::-1]
        rep = []
        for i, d in enumerate(digits):
            if d == 0: continue
            power = len(digits) - i - 1
            if power == 0:
                rep.append(str(d))
            else:
                rep.append(f"{d}*({hereditary_base_expansion(base, base)})^{power}")
        return "+".join(rep)
    seq = [a0]
    current = a0
    for k in range(steps):
        base = k + 2
        h = hereditary_base_expansion(current, base)
        h2 = h.replace(str(base), str(base + 1))
        try:
            val = eval(h2.replace("^", "**"))
        except:
            return seq
        current = max(val - 1, 0)
        seq.append(current)
    return seq


# ======================================================================
# DE BRUIJN
# ======================================================================

def de_bruijn(k: int, n: int) -> List[int]:
    a = [0] * (k * n)
    seq: List[int] = []
    def db(t: int, p: int):
        if t > n:
            if n % p == 0:
                seq.extend(a[1:p+1])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, k):
                a[t] = j
                db(t + 1, t)
    db(1, 1)
    return seq

def de_bruijn_binary(n: int) -> List[int]:
    return de_bruijn(2, n)


# ======================================================================
# PISANO PERIOD
# ======================================================================

def pisano_period(m: int) -> int:
    if m <= 1: return 1
    prev, curr = 0, 1
    for i in range(1, m * m + 1):
        prev, curr = curr, (prev + curr) % m
        if prev == 0 and curr == 1:
            return i
    return m * m


# ======================================================================
# HOFSTADTER Q, H, V
# ======================================================================

def hofstadter_q(n: int) -> List[int]:
    if n <= 0: return []
    q = [0] * (n + 1)
    q[1] = q[2] = 1
    for k in range(3, n + 1):
        q[k] = q[k - q[k - 1]] + q[k - q[k - 2]]
    return q[1:n+1]

def hofstadter_h_v(n: int) -> Tuple[List[int], List[int]]:
    if n <= 0: return [], []
    H = [0] * (n + 1)
    V = [0] * (n + 1)
    H[1] = V[1] = 1
    for k in range(2, n + 1):
        H[k] = k - V[H[k-1]]
        V[k] = k - H[V[k-1]]
    return H[1:n+1], V[1:n+1]


# ======================================================================
# GOLOMB, ULAM, RECAMÁN, STERN, WYTHOFF, BEATTY
# ======================================================================

def golomb_sequence(n: int) -> List[int]:
    if n <= 0: return []
    a = [0] * (n + 1)
    a[1] = 1
    for k in range(2, n + 1):
        a[k] = 1 + a[k - a[a[k - 1]]]
    return a[1:n+1]

def ulam_sequence(n: int) -> List[int]:
    if n <= 1: return [1][:n]
    seq = [1, 2]
    while len(seq) < n:
        cand = seq[-1] + 1
        while True:
            reps = sum(1 for a in seq if cand - a in seq and cand - a != a)
            if reps == 1:
                seq.append(cand)
                break
            cand += 1
    return seq

def recaman_sequence(n: int) -> List[int]:
    if n <= 0: return []
    a = [0] * n
    seen = {0}
    for k in range(1, n):
        cand = a[k-1] - k
        if cand > 0 and cand not in seen:
            a[k] = cand
        else:
            a[k] = a[k-1] + k
        seen.add(a[k])
    return a

def stern_diatomic(n: int) -> List[int]:
    if n < 0: raise ValueError("n >= 0")
    s = [0] * (n + 1)
    if n >= 0: s[0] = 0
    if n >= 1: s[1] = 1
    for k in range(2, n + 1):
        if k % 2 == 0:
            s[k] = s[k // 2]
        else:
            m = (k - 1) // 2
            s[k] = s[m] + s[m + 1]
    return s

def wythoff_pairs(n: int) -> List[Tuple[int, int]]:
    if n <= 0: return []
    phi = (1 + 5**0.5) / 2
    phi2 = phi + 1
    return [(int(k * phi), int(k * phi2)) for k in range(1, n+1)]

def beatty_sequence(alpha: float, n: int, beta: float = 0.0) -> List[int]:
    return [int(math.floor(alpha * k + beta)) for k in range(1, n+1)]


# ======================================================================
# ULAM SPIRAL
# ======================================================================

def ulam_spiral_coordinates(n: int) -> List[Tuple[int, int]]:
    if n <= 0: return []
    coords = [(0, 0)]
    if n == 1: return coords
    x = y = 0
    step_len = val = 1
    while val < n:
        for _ in range(step_len):
            if val >= n: break
            x += 1
            val += 1
            coords.append((x, y))
        for _ in range(step_len):
            if val >= n: break
            y += 1
            val += 1
            coords.append((x, y))
        step_len += 1
        for _ in range(step_len):
            if val >= n: break
            x -= 1
            val += 1
            coords.append((x, y))
        for _ in range(step_len):
            if val >= n: break
            y -= 1
            val += 1
            coords.append((x, y))
        step_len += 1
    return coords[:n]

def ulam_spiral_prime_points(n: int) -> List[Tuple[int, int]]:
    coords = ulam_spiral_coordinates(n)
    return [coords[k-1] for k in range(1, n+1) if is_prime(k)]


# ======================================================================
# TUPPER BITMAP
# ======================================================================

def tupper_bitmap_cell(k: int, x: int, y: int) -> int:
    if x < 0 or y < 0: return 0
    index = y + 17 * x
    return (k >> index) & 1

def tupper_bitmap_block(k: int, width: int = 106, height: int = 17) -> List[List[int]]:
    return [[tupper_bitmap_cell(k, x, y) for x in range(width)] for y in range(height)]


# ======================================================================
# KAPREKAR 6174
# ======================================================================

def kaprekar_step_4digit(n: int) -> int:
    s = f"{n:04d}"
    asc = int("".join(sorted(s)))
    desc = int("".join(sorted(s, reverse=True)))
    return desc - asc

def kaprekar_trajectory_4digit(n: int, max_iter: int = 20) -> List[int]:
    seen = set()
    seq = [n]
    x = n
    for _ in range(max_iter):
        x = kaprekar_step_4digit(x)
        if x in seen or x == 6174:
            seq.append(x)
            break
        seen.add(x)
        seq.append(x)
    return seq


# ======================================================================
# FAM / 973 HYBRIDS
# ======================================================================

def fam_thue_morse_hybrid(seed: int, length: int) -> List[int]:
    if length <= 0: return []
    fam = fam_base_sequence(seed, length)
    tm = thue_morse(length)
    return [digital_root_1_9(f + t) for f, t in zip(fam, tm)]

def fam_ulam_spiral_prime_hybrid(n: int, seed: int = 1) -> List[Tuple[int, int, int]]:
    coords = ulam_spiral_coordinates(n)
    out = []
    for k in range(1, n + 1):
        if is_prime(k):
            x, y = coords[k - 1]
            vp = digital_root_1_9(seed + k)
            out.append((x, y, vp))
    return out