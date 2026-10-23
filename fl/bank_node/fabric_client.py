"""
fl/bank_node/fabric_client.py
==============================
Hyperledger Fabric client for a bank FL node.

Wraps the Fabric Gateway SDK (fabric-gateway Python client) to:
  - Submit transactions (StartRound, SubmitUpdate, SubmitAggregate, etc.)
  - Query chaincode state (GetRound, GetBankReputation)
  - Subscribe to chaincode events (RoundStarted, QuorumReached, etc.)

Connection uses TLS with the bank's MSP certificates.

In no-blockchain mode (use_blockchain=False), operations are no-ops
that return synthetic responses for testing.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class FabricClient:
    """
    Client for Hyperledger Fabric chaincode interaction.

    Parameters
    ----------
    bank_id : str
        e.g., 'bank0'
    msp_id : str
        e.g., 'Bank0MSP'
    peer_endpoint : str
        e.g., 'peer0.bank0.fraud.net:7051'
    crypto_path : Path
        Path to the bank's crypto material (MSP certs, TLS certs).
    channel_name : str
    chaincode_name : str
    use_blockchain : bool
        If False, all operations are no-ops (for local testing).
    """

    CHAINCODE = "frauddetect"
    CHANNEL = "fraudchannel"

    def __init__(
        self,
        bank_id: str,
        msp_id: str,
        peer_endpoint: str,
        crypto_path: Path,
        channel_name: str = "fraudchannel",
        chaincode_name: str = "frauddetect",
        use_blockchain: bool = True,
    ):
        self.bank_id = bank_id
        self.msp_id = msp_id
        self.peer_endpoint = peer_endpoint
        self.crypto_path = Path(crypto_path)
        self.channel_name = channel_name
        self.chaincode_name = chaincode_name
        self.use_blockchain = use_blockchain

        self._gateway = None
        self._contract = None

        if use_blockchain:
            self._connect()

    def _connect(self) -> None:
        """Establish connection to the Fabric peer via Gateway SDK."""
        try:
            import grpc
            from grpc import ssl_channel_credentials
        except ImportError:
            raise ImportError("grpcio is required. Install: pip install grpcio==1.63.0")

        # Certificate paths
        tls_cert = self.crypto_path / "peers" / f"peer0.{self.bank_id}.fraud.net" / "tls" / "ca.crt"
        user_cert = self.crypto_path / "users" / f"Admin@{self.bank_id}.fraud.net" / "msp" / "signcerts" / "cert.pem"
        user_key_dir = self.crypto_path / "users" / f"Admin@{self.bank_id}.fraud.net" / "msp" / "keystore"

        try:
            user_key_files = list(user_key_dir.glob("*_sk"))
            if not user_key_files:
                raise FileNotFoundError(f"No private key found in {user_key_dir}")
            user_key = user_key_files[0]

            # Try fabric-gateway Python SDK (v1.x)
            try:
                from fabric_sdk_py.gateway.gateway import Gateway
                # fabric-sdk-py gateway interface
                self._setup_fabric_sdk_py(tls_cert, user_cert, user_key)
                logger.info(f"[FabricClient] Connected via fabric-sdk-py: {self.peer_endpoint}")
            except ImportError:
                # Fall back to raw gRPC + direct peer calls
                logger.warning(
                    "[FabricClient] fabric-sdk-py not available; using raw gRPC. "
                    "Install: pip install fabric-sdk-py"
                )
                self._setup_grpc(tls_cert, user_cert, user_key)

        except FileNotFoundError as e:
            logger.error(f"[FabricClient] Crypto material not found: {e}")
            logger.error("Run 'make network-up' to generate crypto material.")
            raise

    def _setup_fabric_sdk_py(self, tls_cert, user_cert, user_key) -> None:
        """Set up connection using fabric-sdk-py."""
        # This will be implemented when the network is up
        # Placeholder for the actual SDK connection
        logger.info("[FabricClient] fabric-sdk-py gateway setup (stub)")

    def _setup_grpc(self, tls_cert, user_cert, user_key) -> None:
        """Set up raw gRPC connection as fallback."""
        logger.info("[FabricClient] gRPC connection setup (stub)")

    # ------------------------------------------------------------------
    # Transaction submission
    # ------------------------------------------------------------------

    def register_bank(self) -> None:
        self._submit("RegisterBank", self.bank_id)

    def start_round(
        self,
        round_id: str,
        global_model_cid: str,
        global_model_hash: str,
        hparams: dict,
        participants: list,
        max_norm_bound: float = 100.0,
    ) -> None:
        self._submit(
            "StartRound",
            round_id,
            global_model_cid,
            global_model_hash,
            json.dumps(hparams),
            json.dumps(participants),
            str(max_norm_bound),
        )

    def submit_update(
        self,
        round_id: str,
        cid: str,
        sha256: str,
        size: int,
        declared_norm: float,
    ) -> None:
        self._submit(
            "SubmitUpdate",
            round_id,
            self.bank_id,
            cid,
            sha256,
            str(size),
            str(declared_norm),
        )

    def submit_aggregate(
        self,
        round_id: str,
        agg_cid: str,
        agg_hash: str,
    ) -> None:
        self._submit("SubmitAggregate", round_id, agg_cid, agg_hash)

    def endorse_aggregate(
        self,
        round_id: str,
        approve: bool,
        recomputed_hash: str,
    ) -> None:
        self._submit(
            "EndorseAggregate",
            round_id,
            "true" if approve else "false",
            recomputed_hash,
        )

    def finalize_round(self, round_id: str) -> None:
        self._submit("FinalizeRound", round_id)

    def report_metrics(
        self,
        round_id: str,
        split: str,
        metrics: dict,
    ) -> None:
        self._submit(
            "ReportMetrics",
            round_id,
            self.bank_id,
            split,
            json.dumps(metrics),
        )

    def record_security_event(
        self,
        round_id: str,
        event_type: str,
        details: str,
    ) -> None:
        self._submit(
            "RecordSecurityEvent",
            round_id,
            self.bank_id,
            event_type,
            details,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_round(self, round_id: str) -> Optional[dict]:
        result = self._query("GetRound", round_id)
        if result:
            return json.loads(result)
        return None

    def get_bank_reputation(self, bank_id: str) -> Optional[dict]:
        result = self._query("GetBankReputation", bank_id)
        if result:
            return json.loads(result)
        return None

    def get_history(self, key: str) -> list:
        result = self._query("GetHistory", key)
        if result:
            return json.loads(result)
        return []

    # ------------------------------------------------------------------
    # Event subscription
    # ------------------------------------------------------------------

    def subscribe_events(
        self,
        event_names: list,
        callback: Callable[[str, dict], None],
    ) -> None:
        """
        Subscribe to chaincode events.

        Parameters
        ----------
        event_names : list of str
            e.g., ['RoundStarted', 'QuorumReached', 'RoundFinalized']
        callback : callable(event_name: str, payload: dict)
        """
        if not self.use_blockchain:
            logger.debug("[FabricClient] no-blockchain mode: event subscription skipped")
            return

        logger.info(f"[FabricClient] Subscribing to events: {event_names}")
        # In the real implementation, this sets up a chaincode event listener.
        # For now, log the subscription.

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _submit(self, fn: str, *args: str) -> Optional[str]:
        if not self.use_blockchain:
            logger.debug(f"[FabricClient][no-blockchain] submit {fn}({args})")
            return None

        if self._contract is None:
            raise RuntimeError(
                "[FabricClient] Not connected to Fabric. "
                "Run 'make network-up chaincode-deploy' first."
            )

        try:
            result = self._contract.submit_transaction(fn, *args)
            logger.info(f"[FabricClient] {fn} OK")
            return result.decode() if result else None
        except Exception as e:
            logger.error(f"[FabricClient] {fn} FAILED: {e}")
            raise

    def _query(self, fn: str, *args: str) -> Optional[str]:
        if not self.use_blockchain:
            logger.debug(f"[FabricClient][no-blockchain] query {fn}({args})")
            return None

        if self._contract is None:
            raise RuntimeError("[FabricClient] Not connected to Fabric.")

        try:
            result = self._contract.evaluate_transaction(fn, *args)
            return result.decode() if result else None
        except Exception as e:
            logger.error(f"[FabricClient] query {fn} FAILED: {e}")
            raise
