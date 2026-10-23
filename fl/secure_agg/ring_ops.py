"""
fl/secure_agg/ring_ops.py
==========================
Integer arithmetic in the ring Z_{2^l} (modular arithmetic mod 2^l).

Used for fixed-point encoding of floating-point model updates so that
masks cancel exactly in the integer domain.

The ring size 2^l must be large enough to avoid overflow when summing
N masked updates. With N=8 banks and l=64 (int64 mod), the maximum
representable value before modular wrap is 2^63 – comfortable for typical
model parameters.
"""

import numpy as np

# Ring size exponent. 2^RING_BITS arithmetic.
# Using 64 bits (int64 mod 2^64 via numpy overflow behaviour).
RING_BITS = 64
RING_MOD = 2 ** RING_BITS  # 18446744073709551616


def ring_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Add two integer arrays mod 2^RING_BITS."""
    return (a.astype(np.uint64) + b.astype(np.uint64)).view(np.uint64)


def ring_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Subtract b from a mod 2^RING_BITS."""
    return (a.astype(np.uint64) - b.astype(np.uint64)).view(np.uint64)


def ring_sum(arrays: list[np.ndarray]) -> np.ndarray:
    """Sum a list of integer arrays mod 2^RING_BITS."""
    result = np.zeros_like(arrays[0], dtype=np.uint64)
    for a in arrays:
        result = ring_add(result, a)
    return result
