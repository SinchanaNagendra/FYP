"""Secure aggregation package (pairwise-mask, X25519 + PRG)."""
from .keys import KeyPair, derive_shared_seed
from .masking import generate_mask, encode_fixed_point, decode_fixed_point
from .aggregator import SecureAggregator

__all__ = [
    "KeyPair",
    "derive_shared_seed",
    "generate_mask",
    "encode_fixed_point",
    "decode_fixed_point",
    "SecureAggregator",
]
