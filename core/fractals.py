"""
MathToolBox v7.0 — fractals.py
==============================

Fractal & dynamical systems engine for v7.0. This module provides
numerically solid generators and analyzers for:

    • 1D chaotic maps:
        - Logistic map
        - Tent map
        - Quadratic map

    • Classic self-similar sets:
        - Cantor set (middle-third)
        - Sierpinski triangle (barycentric / IFS points)
        - Sierpinski carpet (masking on unit square grid)

    • Complex dynamics:
        - Mandelbrot set iteration counts
        - Julia sets (c-parameter families)

    • Dimension & scaling estimates:
        - Box-counting dimension estimator for a binary grid
        - Lyapunov exponent estimator for 1D maps

All functions are mathematically real and suitable for:

    • FAM / 973 orchestrator visual layers
    • mining nonce chaos modulation
    • rare pattern visualization
    • entropy diagnostics
"""

from __future__ import annotations
from typing import Callable, Iterable, List, Tuple
import math

from .numerics import shannon_entropy


# ======================================================================
# 1D CHAOTIC MAPS
# ======================================================================

def logistic_map(
    r: float,
    x0: float,
    n: int,
) -> List[float]:
    """
    Generate n iterates of the logistic map:

        x_{k+1} = r * x_k * (1 - x_k)

    Parameters
    ----------
    r : float
        Control parameter; chaos typically appears for r ≳ 3.57 and
        becomes fully chaotic near r = 4.
    x0 : float
        Initial value in (0,1).
    n : int
        Number of iterates.

    Returns
    -------
    List[float] of length n containing x_0, x_1, ..., x_{n-1}.
    """
    if n <= 0:
        return []
    x = float(x0)
    out = [x]
    for _ in range(n - 1):
        x = r * x * (1.0 - x)
        out.append(x)
    return out


def tent_map(
    a: float,
    x0: float,
    n: int,
) -> List[float]:
    """
    Generate n iterates of the tent map with slope parameter a:

        f(x) = a * x              for 0 ≤ x < 0.5
             = a * (1 - x)        for 0.5 ≤ x ≤ 1

    For a=2, this is the full chaos tent map on [0,1].
    """
    if n <= 0:
        return []
    x = float(x0)
    out = [x]
    for _ in range(n - 1):
        if x < 0.5:
            x = a * x
        else:
            x = a * (1.0 - x)
        out.append(x)
    return out


def quadratic_map(
    c: float,
    x0: float,
    n: int,
) -> List[float]:
    """
    Generate n iterates of the quadratic map:

        x_{k+1} = x_k^2 + c

    Real-valued version; analog of z^2 + c in ℂ used for Julia/Mandelbrot
    but restricted to ℝ. Useful for exploring escape / boundedness.
    """
    if n <= 0:
        return []
    x = float(x0)
    out = [x]
    for _ in range(n - 1):
        x = x * x + c
        out.append(x)
    return out



# ======================================================================
# CANTOR SET (MIDDLE-THIRD)
# ======================================================================

def cantor_intervals(iterations: int) -> List[Tuple[float, float]]:
    """
    Generate intervals of the middle-third Cantor set construction
    after a given number of iterations.

    Start with [0,1]. At each step, remove the open middle third of
    every interval. Return the list of remaining closed intervals.

    This is a purely analytic representation (no sampling).
    """
    intervals = [(0.0, 1.0)]
    for _ in range(iterations):
        new_ints: List[Tuple[float, float]] = []
        for a, b in intervals:
            third = (b - a) / 3.0
            new_ints.append((a, a + third))
            new_ints.append((b - third, b))
        intervals = new_ints
    return intervals



# ======================================================================
# SIERPINSKI TRIANGLE (IFS POINTS)
# ======================================================================

def sierpinski_triangle_points(
    n: int,
    vertices: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
        (0.0, 0.0),
        (1.0, 0.0),
        (0.5, math.sqrt(3) / 2.0),
    ),
    x0: Tuple[float, float] = (0.25, 0.25),
) -> List[Tuple[float, float]]:
    """
    Generate n points in the Sierpinski triangle using an IFS (Iterated
    Function System) method:

        1. Start at x0.
        2. Repeatedly select one of the 3 vertices at random.
        3. Move to the midpoint between current point and that vertex.

    This converges to the Sierpinski triangle attractor.

    Returns a list of n (x,y) points.
    """
    if n <= 0:
        return []
    import random

    v0, v1, v2 = vertices
    x, y = x0
    out = []

    for _ in range(n):
        choice = random.randint(0, 2)
        if choice == 0:
            vx, vy = v0
        elif choice == 1:
            vx, vy = v1
        else:
            vx, vy = v2
        x = (x + vx) / 2.0
        y = (y + vy) / 2.0
        out.append((x, y))

    return out



# ======================================================================
# SIERPINSKI CARPET (GRID MASK)
# ======================================================================

def sierpinski_carpet_mask(level: int, size: int) -> List[List[int]]:
    """
    Generate a Sierpinski carpet mask on a size×size grid.

    For conceptual clarity, we use a coordinate test:

        A point (i,j) in [0,size-1]^2 is in the carpet if, in base 3,
        no digit position has both coordinates ≡ 1 (mod 3) simultaneously.

    This corresponds to the standard carpet construction.

    Parameters
    ----------
    level : int
        Logical refinement level. The effective resolution is bounded
        by 'size'; higher 'level' gives more intricate patterns.
    size : int
        Grid size; should be a power of 3 for exact symmetry, but
        the mask function works for any positive integer.

    Returns
    -------
    grid[row][col] ∈ {0,1} where 1 means "occupied (carpet)", 0 means "hole".
    """
    if size <= 0:
        return []

    grid: List[List[int]] = []
    for i in range(size):
        row: List[int] = []
        for j in range(size):
            x, y = i, j
            keep = True
            # check base-3 digits
            for _ in range(level):
                if x % 3 == 1 and y % 3 == 1:
                    keep = False
                    break
                x //= 3
                y //= 3
                if x == 0 and y == 0:
                    break
            row.append(1 if keep else 0)
        grid.append(row)
    return grid



# ======================================================================
# MANDELBROT SET ITERATION COUNTS
# ======================================================================

def mandelbrot_escape_iterations(
    c: complex,
    max_iter: int = 1000,
    escape_radius: float = 2.0,
) -> int:
    """
    Compute the escape time for the Mandelbrot iteration:

        z_{n+1} = z_n^2 + c,  z_0 = 0

    We return the smallest n such that |z_n| > escape_radius, or
    max_iter if not escaped.

    This is the core of any Mandelbrot visualization; we keep it
    minimal and robust, ready for use in orchestrator-driven maps.
    """
    z = 0 + 0j
    R2 = escape_radius * escape_radius
    for n in range(max_iter):
        if (z.real*z.real + z.imag*z.imag) > R2:
            return n
        z = z*z + c
    return max_iter


def mandelbrot_grid(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    width: int,
    height: int,
    max_iter: int = 200,
) -> List[List[int]]:
    """
    Sample a rectangle in the complex plane for Mandelbrot iteration
    counts, returning a 2D grid of escape times.

    The grid is indexed as:
        grid[row][col], where row corresponds to y, col to x.
    """
    if width <= 0 or height <= 0:
        return []

    grid: List[List[int]] = []
    for iy in range(height):
        row: List[int] = []
        y = ymin + (ymax - ymin) * iy / (height - 1)
        for ix in range(width):
            x = xmin + (xmax - xmin) * ix / (width - 1)
            c = complex(x, y)
            row.append(mandelbrot_escape_iterations(c, max_iter=max_iter))
        grid.append(row)
    return grid



# ======================================================================
# JULIA SET ITERATION COUNTS
# ======================================================================

def julia_escape_iterations(
    z0: complex,
    c: complex,
    max_iter: int = 1000,
    escape_radius: float = 2.0,
) -> int:
    """
    Julia set iteration:

        z_{n+1} = z_n^2 + c,  z_0 = z0

    Similar to Mandelbrot but with c fixed and z0 varying.

    Returns:
        escape time (iterations before leaving the disk), or max_iter.
    """
    z = z0
    R2 = escape_radius * escape_radius
    for n in range(max_iter):
        if (z.real*z.real + z.imag*z.imag) > R2:
            return n
        z = z*z + c
    return max_iter


def julia_grid(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    width: int,
    height: int,
    c: complex,
    max_iter: int = 200,
) -> List[List[int]]:
    """
    Sample a rectangle in the complex plane for Julia iteration counts.

    Returns:
        grid[row][col] of escape times.
    """
    if width <= 0 or height <= 0:
        return []

    grid: List[List[int]] = []
    for iy in range(height):
        row: List[int] = []
        y = ymin + (ymax - ymin) * iy / (height - 1)
        for ix in range(width):
            x = xmin + (xmax - xmin) * ix / (width - 1)
            z0 = complex(x, y)
            row.append(julia_escape_iterations(z0, c, max_iter=max_iter))
        grid.append(row)
    return grid



# ======================================================================
# BOX-COUNTING DIMENSION ESTIMATOR
# ======================================================================

def box_count_dimension(
    grid: List[List[int]],
    min_box: int = 2,
) -> float:
    """
    Estimate the box-counting (Minkowski–Bouligand) dimension of a
    binary 2D pattern.

    Parameters
    ----------
    grid : List[List[int]]
        Binary grid (0 = empty, 1 = filled).
    min_box : int
        Minimum box size for counting; the maximum is limited by the
        grid size.

    Returns
    -------
    float
        Estimated dimension from linear regression on log N(ε)
        vs log(1/ε). If insufficient scales, returns 0.0.

    Notes
    -----
    This is a *numerical heuristic*, but mathematically grounded in
    standard fractal dimension estimation techniques.
    """
    if not grid or not grid[0]:
        return 0.0

    height = len(grid)
    width = len(grid[0])
    max_box = min(width, height)

    sizes: List[int] = []
    counts: List[int] = []

    box = min_box
    while box <= max_box:
        n_x = (width + box - 1) // box
        n_y = (height + box - 1) // box
        count = 0
        for by in range(n_y):
            for bx in range(n_x):
                # check if any filled cell in this box
                filled = False
                for y in range(by*box, min((by+1)*box, height)):
                    if filled:
                        break
                    for x in range(bx*box, min((bx+1)*box, width)):
                        if grid[y][x] != 0:
                            filled = True
                            break
                if filled:
                    count += 1
        if count > 0:
            sizes.append(box)
            counts.append(count)
        box *= 2

    if len(sizes) < 2:
        return 0.0

    # log-log regression: log N = D * log(1/ε) + const
    # here ε ~ box_size / max_dim, so log(1/ε) ≈ -log(box_size)
    xs = [-math.log(s) for s in sizes]
    ys = [math.log(c) for c in counts]

    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
    den = sum((xs[i] - mean_x) ** 2 for i in range(n))
    if den == 0:
        return 0.0
    D = num / den
    return D



# ======================================================================
# LYAPUNOV EXPONENT ESTIMATOR FOR 1D MAPS
# ======================================================================

def lyapunov_exponent_1d(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    n: int,
    discard: int = 100,
) -> float:
    """
    Estimate the Lyapunov exponent λ for a smooth 1D map:

        x_{k+1} = f(x_k)

    via the standard formula:

        λ ≈ (1/N) Σ log |f'(x_k)|

    after discarding an initial transient of 'discard' steps.

    Parameters
    ----------
    f : Callable[[float], float]
        The map.
    df : Callable[[float], float]
        Its derivative.
    x0 : float
        Initial condition.
    n : int
        Number of iterations used in the average (after discard).
    discard : int
        Transient steps discarded before averaging.

    Returns
    -------
    float
        Approximate Lyapunov exponent. λ > 0 indicates chaos.
    """
    x = float(x0)
    # transient
    for _ in range(discard):
        x = f(x)

    if n <= 0:
        return 0.0

    s = 0.0
    for _ in range(n):
        d = df(x)
        if d == 0:
            # derivative zero → local contraction; log|0| → -∞
            # approximate with large negative
            s += -50.0
        else:
            s += math.log(abs(d))
        x = f(x)

    return s / n
