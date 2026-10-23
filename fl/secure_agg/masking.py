"""
fl/secure_agg/masking.py
=========================
Pairwise-mask generation and fixed-point encoding/decoding.

Protocol:
  For bank i with pairwise seeds {(i,j): seed_{ij} for all j ≠ i}:

    mask_i = Σ_{j > i}  PRG(seed_{ij})     (add for bank with smaller id)
           - Σ_{j < i}  PRG(seed_{ji})     (subtract for bank with larger id)

    When summed over all banks:
      Σ_i mask_i = 0  (every seed appears once + and once -)

  Masked update:  m_i = encode(w_i) + mask_i  (mod 2^l)

  Aggregate:  Σ_i m_i = Σ_i encode(w_i)  (mod 2^l, masks cancel)

  Decoded:  decode(Σ_i m_i) ≈ Σ_i w_i  (small quantisation error)

Fixed-point encoding:
  float → round(x * SCALE) as uint64
  Decoding: uint64 → signed int64 → float / SCALE / N

  SCALE = 2^20 gives ~6 decimal digits of precision.
  Works for model parameters typically in [-10, 10] range.
"""

import hashlib
import struct

import numpy as np

from .ring_ops import ring_add, ring_sub, ring_sum, RING_BITS

# Fixed-point scale factor: x_int = round(x * SCALE)
# 2^20 ≈ 1e6; sufficient for model weights in [-100, 100]
FIXED_POINT_SCALE = 2 ** 20


def encode_fixed_point(weights: np.ndarray, scale: int = FIXED_POINT_SCALE) -> np.ndarray:
    """
    Encode a float32 weight array to uint64 fixed-point.

    Parameters
    ----------
    weights : np.ndarray, dtype float32 or float64
    scale : int

    Returns
    -------
    encoded : np.ndarray, dtype uint64
    """
    scaled = np.round(weights.astype(np.float64) * scale).astype(np.int64)
    return scaled.view(np.uint64)


def decode_fixed_point(
    encoded: np.ndarray,
    n_participants: int,
    scale: int = FIXED_POINT_SCALE,
) -> np.ndarray:
    """
    Decode a uint64 fixed-point aggregate to float32.

    Parameters
    ----------
    encoded : np.ndarray, dtype uint64
        The sum of N encoded (masked) weight vectors.
    n_participants : int
        Number of banks whose weights were summed.
    scale : int

    Returns
    -------
    weights : np.ndarray, dtype float32 (average over participants)
    """
    # Reinterpret as signed int64 to handle negative values (two's complement)
    signed = encoded.view(np.int64)
    # Divide by scale to recover the sum, then by N to get the average
    return (signed.astype(np.float64) / scale / n_participants).astype(np.float32)


def prg_expand(seed: bytes, n_elements: int) -> np.ndarray:
    """
    Pseudorandom Generator: expand a 32-byte seed into n_elements uint64 values.

    Uses SHA-256 in counter mode (simple, deterministic, cryptographically
    sufficient for the masking protocol).

    Parameters
    ----------
    seed : bytes, 32 bytes
    n_elements : int

    Returns
    -------
    mask : np.ndarray, dtype uint64, shape (n_elements,)
    """
    mask = np.empty(n_elements, dtype=np.uint64)
    n_uint64_per_block = 32 // 8  # 4 uint64 per 32-byte SHA-256 output

    counter = 0
    idx = 0
    while idx < n_elements:
        # Hash: seed || counter (8 bytes big-endian)
        block_input = seed + counter.to_bytes(8, "big")
        block = hashlib.sha256(block_input).digest()  # 32 bytes = 4 × uint64

        block_ints = np.frombuffer(block, dtype=np.uint64)
        take = min(n_uint64_per_block, n_elements - idx)
        mask[idx: idx + take] = block_ints[:take]

        idx += take
        counter += 1

    return mask


def generate_mask(
    my_bank_id: int,
    pairwise_seeds: dict[tuple[int, int], bytes],
    n_elements: int,
) -> np.ndarray:
    """
    Generate the additive mask for bank `my_bank_id`.

    mask_i = Σ_{j > i} PRG(seed_{i,j}) - Σ_{j < i} PRG(seed_{j,i})  mod 2^64

    Parameters
    ----------
    my_bank_id : int
    pairwise_seeds : dict {(i, j): seed} with i < j always
    n_elements : int
        Number of elements in the weight vector.

    Returns
    -------
    mask : np.ndarray, dtype uint64, shape (n_elements,)
    """
    mask = np.zeros(n_elements, dtype=np.uint64)

    for (i, j), seed in pairwise_seeds.items():
        assert i < j, f"Seed key invariant violated: ({i}, {j})"
        prg_values = prg_expand(seed, n_elements)

        if my_bank_id == i:
            # Add PRG(seed_{i,j}) for j > i
            mask = (mask.astype(np.uint64) + prg_values).astype(np.uint64)
        elif my_bank_id == j:
            # Subtract PRG(seed_{i,j}) for j > i (i.e., i < j = my_bank_id)
            mask = (mask.astype(np.uint64) - prg_values).astype(np.uint64)
        # else: this seed doesn't involve my_bank_id, skip

    return mask


def mask_weights(
    weights: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Apply the mask to an encoded weight vector.

    masked = encode(weights) + mask  mod 2^64

    Parameters
    ----------
    weights : np.ndarray, dtype float32
    mask : np.ndarray, dtype uint64

    Returns
    -------
    masked : np.ndarray, dtype uint64
    """
    encoded = encode_fixed_point(weights)
    return (encoded.astype(np.uint64) + mask.astype(np.uint64)).astype(np.uint64)


def unmask_aggregate(
    masked_sum: np.ndarray,
    n_participants: int,
) -> np.ndarray:
    """
    Decode the aggregate sum (masks already cancelled) to float32 average.

    Parameters
    ----------
    masked_sum : np.ndarray, dtype uint64
        Sum of all masked encoded updates.
    n_participants : int

    Returns
    -------
    average_weights : np.ndarray, dtype float32
    """
    return decode_fixed_point(masked_sum, n_participants)
