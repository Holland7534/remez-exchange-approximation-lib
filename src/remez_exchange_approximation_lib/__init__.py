"""Best uniform polynomial approximation via the Remez exchange algorithm.

Exports :func:`remez` and the :class:`RemezResult` container.
"""
from .core import RemezResult, remez

__all__ = ["RemezResult", "remez"]
