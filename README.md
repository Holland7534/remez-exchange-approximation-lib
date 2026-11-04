# remez-exchange-approximation-lib

Best uniform polynomial approximation on an interval using the multipoint Remez exchange iteration.

```python
import math
from remez_exchange_approximation_lib import remez

result = remez(math.sin, -1.0, 1.0, 5)
print(result.coefficients)  # ascending-order polynomial coefficients
print(result.max_error)     # largest absolute error on the sampling grid
print(result.nodes)         # final reference nodes
print(result.iterations)    # number of exchange iterations
print(result.converged)     # whether the convergence criterion was met
```

## Why this library exists

The Remez exchange algorithm finds the polynomial of a given degree that
minimises the maximum absolute error on an interval.  It is the standard way to
build minimax approximations for special functions, but most reference
implementations either depend on a numerical linear algebra package or are
written for a fixed set of functions.

This library keeps the whole thing to the Python standard library.  The linear
system is solved with a small Gaussian elimination routine, and the error curve
is sampled on a dense uniform grid to locate extrema.  That trade-off makes the
code easy to audit and dependency-free at the cost of some speed on very high
degrees; for the degrees where minimax approximations are usually useful this is
not a practical limitation.

## What you get

The public API is two names:

* `remez(f, a, b, degree, *, max_iterations=100, tolerance=1e-12, samples=2001)`
  returns a `RemezResult`.
* `RemezResult` is a frozen dataclass with fields `coefficients`, `max_error`,
  `nodes`, `iterations` and `converged`.

Coefficients are in ascending order, so `coefficients[0]` is the constant term.

## The awkward edge

The iteration stops when the sampled maximum error stops improving by more than
`tolerance`.  If you set `tolerance=0.0` the loop can run until `max_iterations`
is reached; the result is still returned, but `converged` will be `False`.  The
same is true when the reference nodes stop moving, which can happen when the
sampling grid is too coarse to separate adjacent extrema.  Increase `samples`
if you suspect that is happening.

## Running the tests

```
PYTHONPATH=src python -m unittest discover -s tests
```
