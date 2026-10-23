"""
fl/bank_node/ipfs_client.py
============================
IPFS client for encrypted artifact upload/download.

Security properties:
  - Artifacts are AES-256-GCM encrypted before upload.
  - The encryption key is NOT stored in IPFS.
  - SHA-256 of the ciphertext is verified after download.
  - In the real system, the encryption key is derived from the bank's identity
    and the round ID, and shared via the secure aggregation key exchange.

For the prototype, a round-specific symmetric key is distributed out-of-band
(over the Fabric channel, encrypted to each recipient's public key).
This is a simplification; full key management is future work.
"""

import hashlib
import io
import logging
import os
import pickle
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

# AES-256-GCM key size
AES_KEY_BYTES = 32


class IPFSClient:
    """
    Handles encrypted artifact upload and download from a private IPFS node.

    Parameters
    ----------
    ipfs_api_url : str
        URL of the local IPFS API (e.g., 'http://localhost:5001').
    encryption_key : bytes, optional
        32-byte AES-256-GCM key. If None, a random key is generated per instance
        (use only for testing — in production the key must be shared with peers).
    """

    def __init__(
        self,
        ipfs_api_url: str = "http://localhost:5001",
        encryption_key: bytes = None,
    ):
        self.ipfs_api_url = ipfs_api_url
        self._key = encryption_key or os.urandom(AES_KEY_BYTES)
        self._client = None  # lazy init

    def _get_client(self):
        if self._client is None:
            try:
                import ipfshttpclient
                self._client = ipfshttpclient.connect(self.ipfs_api_url)
            except ImportError:
                raise RuntimeError(
                    "ipfshttpclient is not installed. "
                    "Install with: pip install ipfshttpclient\n"
                    "NOTE: If you are running without a real IPFS node, use "
                    "--no-blockchain mode, which skips IPFS entirely."
                )
            except Exception as e:
                raise RuntimeError(
                    f"Cannot connect to IPFS at {self.ipfs_api_url}: {e}\n"
                    "Ensure the IPFS node is running, or use --no-blockchain mode."
                ) from e
        return self._client

    def upload_artifact(
        self,
        payload: bytes,
        name: str = "update.bin",
    ) -> Tuple[str, str]:
        """
        Encrypt and upload a binary artifact to IPFS.

        Parameters
        ----------
        payload : bytes
            Raw artifact bytes (e.g., serialised model weights).
        name : str
            Filename hint (not secret; just for IPFS metadata).

        Returns
        -------
        cid : str
            IPFS content identifier (hash of the ciphertext).
        sha256 : str
            SHA-256 hex digest of the ciphertext (for on-chain anchoring).
        """
        ciphertext = self._encrypt(payload)
        sha256 = hashlib.sha256(ciphertext).hexdigest()

        client = self._get_client()
        result = client.add(io.BytesIO(ciphertext), wrap_with_directory=False)
        cid = result["Hash"]

        logger.info(
            f"[IPFS] Uploaded artifact: CID={cid}, sha256={sha256[:16]}...")
        return cid, sha256

    def download_artifact(
        self,
        cid: str,
        expected_sha256: str,
    ) -> bytes:
        """
        Download and verify an artifact from IPFS.

        Parameters
        ----------
        cid : str
            IPFS content identifier.
        expected_sha256 : str
            Expected SHA-256 of the ciphertext (from on-chain record).

        Returns
        -------
        payload : bytes
            Decrypted artifact bytes.

        Raises
        ------
        ValueError
            If the downloaded content's SHA-256 does not match expected_sha256.
            This indicates tampering or corruption.
        """
        client = self._get_client()
        ciphertext = client.cat(cid)

        actual_sha256 = hashlib.sha256(ciphertext).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"SHA-256 mismatch for CID {cid}!\n"
                f"  Expected: {expected_sha256}\n"
                f"  Actual  : {actual_sha256}\n"
                "Possible tampering or corruption. Rejecting artifact."
            )

        payload = self._decrypt(ciphertext)
        logger.info(f"[IPFS] Downloaded and verified: CID={cid}")
        return payload

    def _encrypt(self, plaintext: bytes) -> bytes:
        """AES-256-GCM encryption."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        nonce = os.urandom(12)  # 96-bit nonce for GCM
        aesgcm = AESGCM(self._key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data=None)
        return nonce + ciphertext  # prepend nonce

    def _decrypt(self, ciphertext_with_nonce: bytes) -> bytes:
        """AES-256-GCM decryption."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        nonce = ciphertext_with_nonce[:12]
        ciphertext = ciphertext_with_nonce[12:]
        aesgcm = AESGCM(self._key)
        return aesgcm.decrypt(nonce, ciphertext, associated_data=None)

    # ------------------------------------------------------------------
    # Serialisation helpers for model parameters
    # ------------------------------------------------------------------

    @staticmethod
    def serialize_parameters(params: list) -> bytes:
        """Serialize a list of numpy arrays to bytes."""
        buf = io.BytesIO()
        pickle.dump(params, buf)
        return buf.getvalue()

    @staticmethod
    def deserialize_parameters(data: bytes) -> list:
        """Deserialize bytes to a list of numpy arrays."""
        buf = io.BytesIO(data)
        return pickle.load(buf)

    @staticmethod
    def serialize_masked_update(masked: np.ndarray) -> bytes:
        """Serialize a masked (uint64) update array to bytes."""
        return masked.tobytes()

    @staticmethod
    def deserialize_masked_update(data: bytes, n_elements: int) -> np.ndarray:
        """Deserialize bytes to a uint64 masked update array."""
        return np.frombuffer(data, dtype=np.uint64)[:n_elements].copy()
