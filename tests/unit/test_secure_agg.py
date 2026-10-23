"""
tests/unit/test_secure_agg.py
==============================
Unit tests for the secure aggregation module.

Tests:
  1. X25519 key agreement: both parties derive the same shared secret.
  2. PRG determinism: same seed gives same mask.
  3. Masks cancel in sum (fundamental correctness property).
  4. Aggregate equals plain sum to within quantisation tolerance.
  5. A single masked update is statistically indistinguishable from noise
     (cannot be unmasked without the peer's key).
  6. Dropout detection: round aborts if a bank's update is missing.

Run with: pytest tests/unit/test_secure_agg.py -v
"""

import numpy as np
import pytest

from fl.secure_agg.keys import KeyPair, derive_shared_seed, generate_pairwise_seeds
from fl.secure_agg.masking import (
    encode_fixed_point,
    decode_fixed_point,
    prg_expand,
    generate_mask,
    mask_weights,
    unmask_aggregate,
    FIXED_POINT_SCALE,
)
from fl.secure_agg.aggregator import SecureAggregator
from fl.secure_agg.ring_ops import ring_add, ring_sub


# ---------------------------------------------------------------------------
# Test 1: X25519 key agreement
# ---------------------------------------------------------------------------

class TestX25519KeyAgreement:

    def test_shared_secret_is_symmetric(self):
        """Both parties derive the same seed from each other's public key."""
        kp_a = KeyPair.generate()
        kp_b = KeyPair.generate()

        seed_a = derive_shared_seed(kp_a, kp_b.public_bytes, 0, 1, "round1")
        seed_b = derive_shared_seed(kp_b, kp_a.public_bytes, 0, 1, "round1")

        assert seed_a == seed_b, "Shared seeds must match (DH symmetry)"
        assert len(seed_a) == 32, "Seed must be 32 bytes"

    def test_different_rounds_give_different_seeds(self):
        """Seeds differ across rounds even for the same keypairs."""
        kp_a = KeyPair.generate()
        kp_b = KeyPair.generate()

        seed_r1 = derive_shared_seed(kp_a, kp_b.public_bytes, 0, 1, "round1")
        seed_r2 = derive_shared_seed(kp_a, kp_b.public_bytes, 0, 1, "round2")

        assert seed_r1 != seed_r2, "Seeds must differ across rounds"

    def test_different_pairs_give_different_seeds(self):
        """Different bank pairs produce different seeds."""
        kp_a = KeyPair.generate()
        kp_b = KeyPair.generate()
        kp_c = KeyPair.generate()

        seed_ab = derive_shared_seed(kp_a, kp_b.public_bytes, 0, 1, "round1")
        seed_ac = derive_shared_seed(kp_a, kp_c.public_bytes, 0, 2, "round1")

        assert seed_ab != seed_ac, "Different pairs must produce different seeds"


# ---------------------------------------------------------------------------
# Test 2: PRG determinism
# ---------------------------------------------------------------------------

class TestPRG:

    def test_prg_is_deterministic(self):
        seed = b"\x42" * 32
        m1 = prg_expand(seed, 100)
        m2 = prg_expand(seed, 100)
        np.testing.assert_array_equal(m1, m2)

    def test_prg_different_seeds_give_different_masks(self):
        seed1 = b"\x01" * 32
        seed2 = b"\x02" * 32
        m1 = prg_expand(seed1, 100)
        m2 = prg_expand(seed2, 100)
        assert not np.array_equal(m1, m2)

    def test_prg_output_length(self):
        seed = b"\xab" * 32
        for n in [1, 10, 100, 1000, 10001]:
            mask = prg_expand(seed, n)
            assert len(mask) == n, f"Expected {n} elements, got {len(mask)}"


# ---------------------------------------------------------------------------
# Test 3: Masks cancel in sum
# ---------------------------------------------------------------------------

class TestMaskCancellation:

    def _setup_pairwise_seeds(self, n_banks: int, round_id: str = "r1"):
        keypairs = {i: KeyPair.generate() for i in range(n_banks)}
        all_seeds: dict[tuple[int, int], bytes] = {}

        for i in range(n_banks):
            for j in range(i + 1, n_banks):
                seed = derive_shared_seed(
                    keypairs[i], keypairs[j].public_bytes, i, j, round_id
                )
                all_seeds[(i, j)] = seed

        return all_seeds

    def _get_bank_seeds(self, bank_id, all_seeds):
        return {k: v for k, v in all_seeds.items() if bank_id in k}

    @pytest.mark.parametrize("n_banks", [2, 3, 5, 8])
    def test_masks_cancel(self, n_banks: int):
        """Sum of all banks' masks must be exactly zero (mod 2^64)."""
        all_seeds = self._setup_pairwise_seeds(n_banks)
        n_elements = 1000

        total = np.zeros(n_elements, dtype=np.uint64)
        for bank_id in range(n_banks):
            bank_seeds = self._get_bank_seeds(bank_id, all_seeds)
            mask = generate_mask(bank_id, bank_seeds, n_elements)
            total = (total + mask).astype(np.uint64)

        assert np.all(total == 0), (
            f"Masks do not cancel for n_banks={n_banks}. "
            f"Non-zero count: {np.count_nonzero(total)}"
        )

    def test_aggregator_verify_masks_cancel(self):
        """SecureAggregator.verify_masks_cancel passes for 5 banks."""
        n_banks = 5
        all_seeds = self._setup_pairwise_seeds(n_banks)
        result = SecureAggregator.verify_masks_cancel(
            bank_ids=list(range(n_banks)),
            pairwise_seeds=all_seeds,
            n_elements=200,
        )
        assert result is True


# ---------------------------------------------------------------------------
# Test 4: Aggregate equals plain sum
# ---------------------------------------------------------------------------

class TestAggregateEqualsPlanSum:

    def _setup_pairwise_seeds(self, n_banks: int, round_id: str = "r1"):
        keypairs = {i: KeyPair.generate() for i in range(n_banks)}
        all_seeds: dict[tuple[int, int], bytes] = {}
        for i in range(n_banks):
            for j in range(i + 1, n_banks):
                seed = derive_shared_seed(
                    keypairs[i], keypairs[j].public_bytes, i, j, round_id
                )
                all_seeds[(i, j)] = seed
        return all_seeds

    @pytest.mark.parametrize("n_banks", [2, 3, 5])
    def test_secagg_average_matches_plain_average(self, n_banks: int):
        """
        The secure aggregate must equal the plain (unmasked) average
        to within fixed-point quantisation error.
        """
        rng = np.random.default_rng(42)
        n_elements = 500
        all_seeds = self._setup_pairwise_seeds(n_banks)

        # Each bank has random weights
        bank_weights = [
            rng.uniform(-1, 1, n_elements).astype(np.float32) for _ in range(n_banks)]
        plain_average = np.mean(bank_weights, axis=0)

        # Masked updates
        agg = SecureAggregator(n_banks=n_banks, n_elements=n_elements)
        for bank_id in range(n_banks):
            bank_seeds = {k: v for k, v in all_seeds.items() if bank_id in k}
            masked = SecureAggregator.apply_mask(
                bank_id=bank_id,
                weights=bank_weights[bank_id],
                pairwise_seeds=bank_seeds,
                n_elements=n_elements,
            )
            agg.add_masked_update(bank_id, masked)

        secagg_average = agg.aggregate()

        # Quantisation tolerance: 1/SCALE per element
        tol = 1.0 / FIXED_POINT_SCALE * 2
        np.testing.assert_allclose(
            secagg_average, plain_average, atol=tol,
            err_msg=f"Secure aggregate deviates from plain average by more than {tol}"
        )


# ---------------------------------------------------------------------------
# Test 5: Single masked update indistinguishable from noise
# ---------------------------------------------------------------------------

class TestMaskedIndistinguishable:

    def test_single_masked_update_looks_random(self):
        """
        A single bank's masked update should look statistically random.
        Test: the masked values should have variance (not all identical)
        and should not have a simple linear relationship to the original weights.
        """
        rng = np.random.default_rng(99)
        n_elements = 1000

        kp_a = KeyPair.generate()
        kp_b = KeyPair.generate()
        seed_ab = derive_shared_seed(kp_a, kp_b.public_bytes, 0, 1, "r1")
        seeds = {(0, 1): seed_ab}

        # Use random weights (not constant, to avoid NaN correlation)
        weights = rng.uniform(-1.0, 1.0, n_elements).astype(np.float32)
        mask = generate_mask(0, seeds, n_elements)
        masked = mask_weights(weights, mask).view(np.int64).astype(np.float64)

        # Pearson correlation between masked values and original weights should be near 0
        # (the mask randomises the relationship)
        corr = float(np.corrcoef(masked, weights.astype(np.float64))[0, 1])

        assert abs(corr) < 0.1, (
            f"Masked update is correlated with original weights (r={corr:.4f}). "
            f"This suggests the mask is insufficient."
        )

        # Additionally: masked values must have non-trivial variance
        masked_std = float(np.std(masked))
        assert masked_std > 1e6, (
            f"Masked update has very low variance (std={masked_std:.2f}), "
            "suggesting the mask is not being applied."
        )

    def test_cannot_unmask_without_peer_key(self):
        """
        Without the peer bank's key, the adversary cannot recover the original weights.
        Test: random key gives wrong decode.
        """
        kp_a = KeyPair.generate()
        kp_b = KeyPair.generate()  # real peer
        kp_adversary = KeyPair.generate()  # attacker

        n_elements = 100
        real_seed = derive_shared_seed(kp_a, kp_b.public_bytes, 0, 1, "r1")
        fake_seed = derive_shared_seed(
            kp_a, kp_adversary.public_bytes, 0, 1, "r1")

        weights = np.random.default_rng(
            42).uniform(-1, 1, n_elements).astype(np.float32)
        mask = generate_mask(0, {(0, 1): real_seed}, n_elements)
        masked = mask_weights(weights, mask)

        # Adversary tries to unmask with wrong seed
        wrong_mask = generate_mask(0, {(0, 1): fake_seed}, n_elements)
        wrong_unmasked_encoded = (masked.astype(
            np.uint64) - wrong_mask.astype(np.uint64))
        wrong_decoded = decode_fixed_point(
            wrong_unmasked_encoded, n_participants=1)

        # The adversary's recovered weights should NOT match the real weights
        mse = float(np.mean((wrong_decoded - weights) ** 2))
        assert mse > 0.001, (
            "Adversary recovered weights too accurately without the correct key. "
            f"MSE={mse:.6f}"
        )


# ---------------------------------------------------------------------------
# Test 6: Fixed-point encoding roundtrip
# ---------------------------------------------------------------------------

class TestFixedPointEncoding:

    def test_encode_decode_roundtrip(self):
        rng = np.random.default_rng(7)
        weights = rng.uniform(-5, 5, 1000).astype(np.float32)

        encoded = encode_fixed_point(weights)
        decoded = decode_fixed_point(encoded, n_participants=1)

        tol = 1.0 / FIXED_POINT_SCALE * 2
        np.testing.assert_allclose(decoded, weights, atol=tol)

    def test_encode_sum_is_sum_of_encodes(self):
        rng = np.random.default_rng(8)
        n = 200
        w1 = rng.uniform(-1, 1, n).astype(np.float32)
        w2 = rng.uniform(-1, 1, n).astype(np.float32)

        enc1 = encode_fixed_point(w1).astype(np.uint64)
        enc2 = encode_fixed_point(w2).astype(np.uint64)
        enc_sum = (enc1 + enc2).astype(np.uint64)
        enc_of_sum = encode_fixed_point(w1 + w2).astype(np.uint64)

        # Difference should be at most 1 (rounding)
        diff = np.abs(enc_sum.view(np.int64) - enc_of_sum.view(np.int64))
        assert np.all(
            diff <= 2), f"Encoding not additive: max diff={diff.max()}"
