"""
fl/bank_node/node.py
=====================
BankNode: the main entry point for a bank's FL process.

Manages the full round lifecycle:
  1. Receive global model (from IPFS, hash-verified via Fabric)
  2. Train locally (with balancing + optional DP-SGD)
  3. Secure-aggregation masking
  4. Encrypt + upload artifact to IPFS
  5. Submit update to Fabric chaincode
  6. (If aggregator role) Download all updates, compute aggregate, submit
  7. Endorse aggregate
  8. Evaluate on local val/test set, report metrics

Privacy invariant: raw data NEVER leaves the bank's process.
Only parameters, hashes, and metrics cross boundaries.
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from fl.balancing.methods import BalancingMethod
from fl.bank_node.evaluator import evaluate, metrics_to_summary_str
from fl.bank_node.trainer import LocalTrainer
from fl.preprocessing import FraudPreprocessor
from fl.secure_agg.keys import KeyPair, generate_pairwise_seeds
from fl.secure_agg.aggregator import SecureAggregator

logger = logging.getLogger(__name__)


class BankNode:
    """
    A federated learning node for one bank.

    Parameters
    ----------
    bank_id : int
    n_banks : int
    data_path : Path
        Path to this bank's parquet files (train/val/test).
    model_type : str
    model_kwargs : dict
    balancing : BalancingMethod
    n_epochs : int
    use_dp : bool
    dp_noise_multiplier : float
    use_secure_agg : bool
    use_blockchain : bool
    fabric_config : dict, optional
    ipfs_api_url : str
    vocab_dir : Path
    results_path : Path
    device : str
    random_state : int
    """

    def __init__(
        self,
        bank_id: int,
        n_banks: int,
        data_path: Path,
        model_type: str,
        model_kwargs: dict = None,
        balancing: BalancingMethod = BalancingMethod.NONE,
        n_epochs: int = 5,
        use_dp: bool = False,
        dp_noise_multiplier: float = 1.1,
        dp_max_grad_norm: float = 1.0,
        use_secure_agg: bool = True,
        use_blockchain: bool = False,
        fabric_config: dict = None,
        ipfs_api_url: str = "http://localhost:5001",
        vocab_dir: Path = None,
        results_path: Path = None,
        device: str = "cpu",
        random_state: int = 42,
    ):
        self.bank_id = bank_id
        self.n_banks = n_banks
        self.data_path = Path(data_path)
        self.model_type = model_type.lower()
        self.model_kwargs = model_kwargs or {}
        self.balancing = balancing
        self.n_epochs = n_epochs
        self.use_dp = use_dp
        self.dp_noise_multiplier = dp_noise_multiplier
        self.dp_max_grad_norm = dp_max_grad_norm
        self.use_secure_agg = use_secure_agg
        self.use_blockchain = use_blockchain
        self.ipfs_api_url = ipfs_api_url
        self.results_path = Path(
            results_path) if results_path else Path("results")
        self.device = device
        self.random_state = random_state

        # Preprocessing
        self.preprocessor = FraudPreprocessor(vocab_dir)

        # Lazy-loaded data
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._X_val: Optional[np.ndarray] = None
        self._y_val: Optional[np.ndarray] = None
        self._X_test: Optional[np.ndarray] = None
        self._y_test: Optional[np.ndarray] = None

        # Model
        self.model = self._build_model()
        self.n_features = self.preprocessor.n_features

        # Secure aggregation keypair (refreshed each round)
        self._keypair: Optional[KeyPair] = None

        # Clients (lazy init)
        self._fabric_client = None
        self._ipfs_client = None

        if use_blockchain and fabric_config:
            self._init_fabric_client(fabric_config)

    def _build_model(self):
        """Build the model instance from model_type and kwargs."""
        from fl.models import (
            LogisticRegressionModel, CNNModel, GaussianNBModel,
            DecisionTreeModel, RandomForestModel, KNNModel,
        )

        n_features = self.preprocessor.n_features
        kw = self.model_kwargs

        model_map = {
            "logistic_regression": lambda: LogisticRegressionModel(n_features, **kw),
            "cnn": lambda: CNNModel(n_features, **kw),
            "gaussian_nb": lambda: GaussianNBModel(n_features, **kw),
            "decision_tree": lambda: DecisionTreeModel(n_features, **kw),
            "random_forest": lambda: RandomForestModel(n_features, **kw),
            "knn": lambda: KNNModel(n_features, **kw),
        }
        if self.model_type not in model_map:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        return model_map[self.model_type]()

    def _init_fabric_client(self, config: dict) -> None:
        from fl.bank_node.fabric_client import FabricClient
        self._fabric_client = FabricClient(
            bank_id=f"bank{self.bank_id}",
            msp_id=config.get("msp_id", f"Bank{self.bank_id}MSP"),
            peer_endpoint=config.get(
                "peer_endpoint", f"peer0.bank{self.bank_id}.fraud.net:7051"),
            crypto_path=Path(config.get(
                "crypto_path", "fabric/crypto-config")),
            use_blockchain=self.use_blockchain,
        )

    def load_data(self, split: str = "all") -> None:
        """
        Load and preprocess one or all splits from the bank's data volume.
        Called lazily on first use.

        Privacy: reads from the local read-only data volume only.
        """
        if split in ("train", "all"):
            df_train = pd.read_parquet(self.data_path / "train.parquet")
            self._X_train, self._y_train = self.preprocessor.transform(
                df_train)
            logger.info(f"[Bank{self.bank_id}] Loaded train: {len(df_train):,} rows, "
                        f"fraud={self._y_train.mean():.3%}")

        if split in ("val", "all"):
            df_val = pd.read_parquet(self.data_path / "val.parquet")
            self._X_val, self._y_val = self.preprocessor.transform(df_val)
            logger.info(
                f"[Bank{self.bank_id}] Loaded val: {len(df_val):,} rows")

        if split in ("test", "all"):
            df_test = pd.read_parquet(self.data_path / "test.parquet")
            self._X_test, self._y_test = self.preprocessor.transform(df_test)
            logger.info(
                f"[Bank{self.bank_id}] Loaded test: {len(df_test):,} rows")

    def run_round(
        self,
        round_id: str,
        global_params: Optional[List[np.ndarray]],
        peer_public_keys: Dict[int, bytes],
        encryption_key: bytes,
        config: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Execute a full FL round for this bank.

        Parameters
        ----------
        round_id : str
        global_params : list of np.ndarray or None (None = round 0, random init)
        peer_public_keys : {bank_id: pk_bytes}
        encryption_key : bytes, AES-256-GCM key for artifact encryption
        config : dict (hyperparameters from chaincode)

        Returns
        -------
        result : dict with keys:
            - 'bank_id': int
            - 'round_id': str
            - 'n_samples': int
            - 'train_time': float
            - 'cid': str (IPFS CID of the encrypted update)
            - 'sha256': str
            - 'declared_norm': float
            - 'privacy_spent': dict or None
            - 'val_metrics': dict
        """
        config = config or {}
        t_round_start = time.time()

        # Ensure data is loaded
        if self._X_train is None:
            self.load_data()

        # Apply global model
        if global_params is not None:
            self.model.set_parameters(global_params)

        # Build trainer
        trainer = LocalTrainer(
            model_type=self.model_type,
            model=self.model,
            balancing=self.balancing,
            n_epochs=self.n_epochs,
            device=self.device,
            use_dp=self.use_dp,
            dp_noise_multiplier=self.dp_noise_multiplier,
            dp_max_grad_norm=self.dp_max_grad_norm,
            fedprox_mu=config.get("mu", 0.0),
            random_state=self.random_state,
        )

        # Local training
        train_result = trainer.train(
            self._X_train, self._y_train, global_params)

        # Get model update
        local_params = trainer.get_parameters()
        declared_norm = float(np.linalg.norm(
            np.concatenate([p.ravel().astype(np.float64)
                           for p in local_params])
        ))

        # Secure aggregation masking
        if self.use_secure_agg and peer_public_keys:
            self._keypair = KeyPair.generate()
            pairwise_seeds = generate_pairwise_seeds(
                self.bank_id, self._keypair, peer_public_keys, round_id
            )

            flat = np.concatenate([p.ravel() for p in local_params])
            n_elements = len(flat)
            masked_flat = SecureAggregator.apply_mask(
                self.bank_id, flat, pairwise_seeds, n_elements
            )
            artifact_payload = masked_flat.tobytes()
        else:
            # No secure agg: transmit plain parameters (for robust agg mode)
            import pickle
            import io
            buf = io.BytesIO()
            pickle.dump(local_params, buf)
            artifact_payload = buf.getvalue()

        # Encrypt and upload to IPFS
        from fl.bank_node.ipfs_client import IPFSClient
        ipfs = IPFSClient(self.ipfs_api_url, encryption_key=encryption_key)
        cid, sha256 = ipfs.upload_artifact(artifact_payload)

        # Submit update to Fabric chaincode
        if self.use_blockchain and self._fabric_client:
            self._fabric_client.submit_update(
                round_id, cid, sha256,
                size=len(artifact_payload),
                declared_norm=declared_norm,
            )

        # Evaluate on local validation set
        model_type_for_eval = (
            "torch" if self.model_type in (
                "logistic_regression", "cnn") else "sklearn"
        )
        if self.model_type == "gaussian_nb":
            model_type_for_eval = "gnb"
        elif self.model_type in ("decision_tree", "random_forest"):
            model_type_for_eval = "tree"

        val_metrics = evaluate(
            self.model, self._X_val, self._y_val,
            split_name="val",
            model_type=model_type_for_eval,
            device=self.device,
        )

        round_time = time.time() - t_round_start

        result = {
            "bank_id": self.bank_id,
            "round_id": round_id,
            "n_samples": train_result["n_samples"],
            "train_time": train_result["train_time"],
            "round_time": round_time,
            "cid": cid,
            "sha256": sha256,
            "declared_norm": declared_norm,
            "privacy_spent": train_result["privacy_spent"],
            "val_metrics": val_metrics,
        }

        logger.info(
            f"[Bank{self.bank_id}] Round {round_id} done in {round_time:.1f}s. "
            f"PR-AUC={val_metrics.get('pr_auc', 'N/A')}"
        )
        return result

    def evaluate_val(self) -> Dict[str, Any]:
        """Evaluate the current model on the local validation set."""
        if self._X_val is None:
            self.load_data("val")
        model_type_for_eval = self._get_eval_model_type()
        metrics = evaluate(
            self.model, self._X_val, self._y_val,
            split_name="val",
            model_type=model_type_for_eval,
            device=self.device,
        )
        return metrics

    def evaluate_test(self) -> Dict[str, Any]:
        """Evaluate the current global model on the local test set."""
        if self._X_test is None:
            self.load_data("test")

        model_type_for_eval = (
            "torch" if self.model_type in (
                "logistic_regression", "cnn") else "sklearn"
        )
        metrics = evaluate(
            self.model, self._X_test, self._y_test,
            split_name="test",
            model_type=model_type_for_eval,
            device=self.device,
        )
        logger.info(
            f"[Bank{self.bank_id}] Test metrics:\n{metrics_to_summary_str(metrics)}"
        )
        return metrics

    def _get_eval_model_type(self) -> str:
        """Return the model type string expected by the evaluator."""
        if self.model_type in ("logistic_regression", "cnn"):
            return "torch"
        elif self.model_type == "gaussian_nb":
            return "gnb"
        elif self.model_type in ("decision_tree", "random_forest"):
            return "tree"
        return "sklearn"

    def _train_local_only(
        self,
        global_params: Optional[List[np.ndarray]],
        config: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Train locally without IPFS/blockchain upload.
        Returns {n_samples, train_time, losses, privacy_spent}.
        Used by run_fl.py in no-blockchain mode.
        """
        if self._X_train is None:
            self.load_data()

        if global_params is not None:
            self.model.set_parameters(global_params)

        config = config or {}
        trainer = LocalTrainer(
            model_type=self.model_type,
            model=self.model,
            balancing=self.balancing,
            n_epochs=self.n_epochs,
            device=self.device,
            use_dp=self.use_dp,
            dp_noise_multiplier=self.dp_noise_multiplier,
            dp_max_grad_norm=self.dp_max_grad_norm,
            fedprox_mu=config.get("mu", 0.0),
            random_state=self.random_state,
        )
        return trainer.train(self._X_train, self._y_train, global_params)
