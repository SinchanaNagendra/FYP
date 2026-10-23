"""
fl/secure_agg/keys.py
======================
X25519 key agreement for pairwise-mask secure aggregation.

Protocol:
  1. Each bank generates an X25519 keypair.
  2. Banks exchange public keys (via on-chain or broadcast channel).
  3. For each pair (i, j) with i < j, both banks compute:
       shared_secret = X25519(sk_i, pk_j) = X25519(sk_j, pk_i)  [DH property]
  4. From shared_secret, derive a PRG seed via HKDF-SHA256.

This module handles steps 1–4.
"""

import hashlib
from dataclasses import dataclass

import numpy as np
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
    PrivateFormat,
    NoEncryption,
)
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes


@dataclass
class KeyPair:
    """An X25519 keypair for one bank in one FL round."""
    private_key: X25519PrivateKey
    public_key: X25519PublicKey
    public_bytes: bytes  # 32 bytes, for transmission

    @classmethod
    def generate(cls) -> "KeyPair":
        """Generate a fresh X25519 keypair."""
        sk = X25519PrivateKey.generate()
        pk = sk.public_key()
        pk_bytes = pk.public_bytes(Encoding.Raw, PublicFormat.Raw)
        return cls(private_key=sk, public_key=pk, public_bytes=pk_bytes)

    @classmethod
    def from_private_bytes(cls, private_bytes: bytes) -> "KeyPair":
        """Reconstruct from stored private key bytes (deterministic from seed)."""
        from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
        sk = X25519PrivateKey.from_private_bytes(private_bytes)
        pk = sk.public_key()
        pk_bytes = pk.public_bytes(Encoding.Raw, PublicFormat.Raw)
        return cls(private_key=sk, public_key=pk, public_bytes=pk_bytes)


def derive_shared_seed(
    my_keypair: KeyPair,
    peer_public_bytes: bytes,
    bank_i: int,
    bank_j: int,
    round_id: str,
) -> bytes:
    """
    Derive a 32-byte PRG seed from the shared DH secret between two banks.

    Parameters
    ----------
    my_keypair : KeyPair
        This bank's keypair.
    peer_public_bytes : bytes
        The other bank's public key (32 bytes, raw X25519).
    bank_i, bank_j : int
        Canonical ordering (i < j) for reproducible seed derivation.
        Both parties must use the same (i, j) ordering.
    round_id : str
        Round identifier (prevents seed reuse across rounds).

    Returns
    -------
    seed : bytes (32 bytes)
    """
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PublicKey

    peer_pk = X25519PublicKey.from_public_bytes(peer_public_bytes)
    shared_secret = my_keypair.private_key.exchange(peer_pk)  # 32 bytes

    # HKDF: info encodes the pair identity and round to prevent reuse
    info = f"fraudfl:secagg:bank{min(bank_i, bank_j)}_bank{max(bank_i, bank_j)}:round{round_id}".encode()

    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=info,
    )
    seed = hkdf.derive(shared_secret)
    return seed


def generate_pairwise_seeds(
    my_bank_id: int,
    my_keypair: KeyPair,
    peer_public_keys: dict[int, bytes],  # {bank_id: pk_bytes}
    round_id: str,
) -> dict[tuple[int, int], bytes]:
    """
    Generate all pairwise seeds for this bank.

    Returns
    -------
    dict mapping (i, j) → seed (32 bytes), where i < j always.
    """
    seeds = {}
    for peer_id, peer_pk_bytes in peer_public_keys.items():
        if peer_id == my_bank_id:
            continue
        i = min(my_bank_id, peer_id)
        j = max(my_bank_id, peer_id)
        seed = derive_shared_seed(my_keypair, peer_pk_bytes, i, j, round_id)
        seeds[(i, j)] = seed
    return seeds
