"""Core Remez exchange implementation.

We approximate a real-valued function ``f`` on ``[a, b]`` by a polynomial of
degree ``n`` (``n + 1`` coefficients) minimising the maximum absolute error.
The exchange iteration alternates between solving a linear system for the
coefficients and the extremal error, and relocating the reference nodes to
the extrema of the current error curve.

Design choices:

* The linear system is solved with Gaussian elimination with partial pivoting
  from the standard library only.  The matrices are small (``n + 2`` square)
  and well conditioned for the reference sets we produce, so this is plenty.
* The error curve is sampled on a dense uniform grid to locate extrema.  A
  grid is used instead of derivative root finding because the error curve is
  only known through evaluations of ``f`` and the polynomial; a dense grid is
  simple, deterministic and robust for the modest degrees this library is
  intended for.
* Convergence is declared when the maximum sampled error stops improving by
  more than ``tol``.  We do not rely on wall-clock time anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple


@dataclass(frozen=True)
class RemezResult:
    """Result of a Remez exchange run.

    Attributes:
        coefficients: Polynomial coefficients in ascending order, i.e.
            ``coefficients[k]`` multiplies ``x**k``.
        max_error: The largest absolute error observed on the final sampling
            grid.
        nodes: The final reference nodes, sorted ascending.
        iterations: Number of exchange iterations performed.
        converged: Whether the iteration met the convergence criterion.
    """

    coefficients: Tuple[float, ...]
    max_error: float
    nodes: Tuple[float, ...]
    iterations: int
    converged: bool


def _solve(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> List[float]:
    """Solve a small dense linear system with Gaussian elimination.

    Partial pivoting is used to keep the elimination stable.  The input is
    copied so the caller's data is not mutated.
    """
    n = len(matrix)
    if n == 0:
        return []
    if len(rhs) != n:
        raise ValueError("matrix and rhs must have the same length")
    a = [list(row) + [rhs[i]] for i, row in enumerate(matrix)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-300:
            raise ValueError("singular system in Remez exchange")
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
        pivot_val = a[col][col]
        for r in range(col + 1, n):
            factor = a[r][col] / pivot_val
            if factor == 0.0:
                continue
            for c in range(col, n + 1):
                a[r][c] -= factor * a[col][c]

    x = [0.0] * n
    for row in range(n - 1, -1, -1):
        total = a[row][n]
        for c in range(row + 1, n):
            total -= a[row][c] * x[c]
        x[row] = total / a[row][row]
    return x


def _poly_eval(coeffs: Sequence[float], x: float) -> float:
    """Evaluate a polynomial given ascending coefficients."""
    result = 0.0
    for c in reversed(coeffs):
        result = result * x + c
    return result


def _initial_nodes(a: float, b: float, n: int) -> List[float]:
    """Return ``n + 2`` nodes including the endpoints.

    The Chebyshev extrema are a better starting point than a uniform grid, but
    a uniform grid is simpler and the exchange converges quickly from either
    for the degrees we support.
    """
    count = n + 2
    step = (b - a) / (count - 1)
    return [a + i * step for i in range(count)]


def _find_extrema(
    f: Callable[[float], float],
    coeffs: Sequence[float],
    a: float,
    b: float,
    samples: int,
) -> List[Tuple[float, float]]:
    """Locate local extrema of the error curve on ``[a, b]``.

    Returns a list of ``(x, signed_error)`` pairs.  The endpoints are always
    candidates; interior points are sampled uniformly and a point counts as
    an extremum when its error magnitude is at least as large as both
    neighbours.  We return signed errors so the caller can preserve the
    alternating sign structure.
    """
    if samples < 2:
        raise ValueError("samples must be at least 2")
    xs = [a + (b - a) * i / (samples - 1) for i in range(samples)]
    errs = [f(x) - _poly_eval(coeffs, x) for x in xs]

    extrema: List[Tuple[float, float]] = []
    for i, (x, e) in enumerate(zip(xs, errs)):
        if i == 0 or i == samples - 1:
            extrema.append((x, e))
            continue
        left = abs(errs[i - 1])
        right = abs(errs[i + 1])
        if abs(e) >= left and abs(e) >= right:
            extrema.append((x, e))

    # Merge adjacent extrema with the same sign; keep the larger magnitude.
    merged: List[Tuple[float, float]] = []
    for x, e in extrema:
        if merged and (e >= 0) == (merged[-1][1] >= 0):
            if abs(e) > abs(merged[-1][1]):
                merged[-1] = (x, e)
        else:
            merged.append((x, e))
    return merged


def _select_nodes(
    extrema: List[Tuple[float, float]], count: int, a: float, b: float
) -> List[float]:
    """Pick ``count`` reference nodes from the extrema list.

    The endpoints are forced in when they are present, because the equioscillation
    property requires the error to alternate across the whole interval.  If we
    have more extrema than needed we drop the ones with the smallest magnitude,
    keeping the sign alternation intact by working on the sorted list.
    """
    if len(extrema) < count:
        # Fall back to a uniform grid if the sampler missed extrema.
        return _initial_nodes(a, b, count - 2)

    # Sort by position so the alternation is preserved.
    ordered = sorted(extrema, key=lambda pair: pair[0])
    if len(ordered) == count:
        return [x for x, _ in ordered]

    # Keep the largest-magnitude extrema while preserving order.
    indexed = list(enumerate(ordered))
    indexed.sort(key=lambda item: abs(item[1][1]), reverse=True)
    keep = sorted(idx for idx, _ in indexed[:count])
    return [ordered[i][0] for i in keep]


def remez(
    f: Callable[[float], float],
    a: float,
    b: float,
    degree: int,
    *,
    max_iterations: int = 100,
    tolerance: float = 1e-12,
    samples: int = 2001,
) -> RemezResult:
    """Approximate ``f`` on ``[a, b]`` by a polynomial of the given degree.

    Parameters:
        f: The function to approximate.  Must be callable on ``[a, b]``.
        a: Left endpoint of the interval.
        b: Right endpoint of the interval.  Must be strictly greater than ``a``.
        degree: Degree of the approximating polynomial.  Must be non-negative.
        max_iterations: Maximum number of exchange iterations.
        tolerance: Convergence threshold on the improvement of the maximum error.
        samples: Number of points used to sample the error curve when locating
            extrema.  Larger values give a finer search.

    Returns:
        A :class:`RemezResult` with the coefficients, final nodes, observed
        maximum error, iteration count and convergence flag.

    Raises:
        ValueError: If ``b <= a``, ``degree < 0``, ``samples < 2`` or
            ``max_iterations < 1``.
    """
    if b <= a:
        raise ValueError("b must be greater than a")
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if samples < 2:
        raise ValueError("samples must be at least 2")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    n = degree
    count = n + 2
    nodes = _initial_nodes(a, b, n)
    coeffs: List[float] = []
    max_error = float("inf")
    converged = False
    iterations = 0

    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        # Build the (n + 2) x (n + 2) system:
        # sum_k c_k * x_i^k + (-1)^i * E = f(x_i)
        matrix: List[List[float]] = []
        rhs: List[float] = []
        for i, x in enumerate(nodes):
            row = [x ** k for k in range(n + 1)]
            row.append((-1.0) ** i)
            matrix.append(row)
            rhs.append(f(x))

        solution = _solve(matrix, rhs)
        coeffs = solution[: n + 1]
        new_error = abs(solution[n + 1])

        # Evaluate the error on the sampling grid to measure true error.
        grid = [a + (b - a) * i / (samples - 1) for i in range(samples)]
        observed = max(abs(f(x) - _poly_eval(coeffs, x)) for x in grid)

        if abs(observed - max_error) <= tolerance * max(1.0, observed):
            max_error = observed
            converged = True
            break
        max_error = observed

        extrema = _find_extrema(f, coeffs, a, b, samples)
        new_nodes = _select_nodes(extrema, count, a, b)

        # If the node set did not change we cannot make progress.
        if new_nodes == nodes:
            break
        nodes = new_nodes

    return RemezResult(
        coefficients=tuple(coeffs),
        max_error=max_error,
        nodes=tuple(nodes),
        iterations=iterations,
        converged=converged,
    )
