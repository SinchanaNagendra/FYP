"""
fl/strategies/fabric_fedprox.py
=================================
FabricFedProx: FedProx strategy for non-IID scenarios.

FedProx adds a proximal term to the local objective:
  F_i(w) + (μ/2) * ||w - w_global||²

This regularises local updates toward the global model, improving convergence
when bank data distributions are heterogeneous (S2/S3 scenarios).

Reference: Li et al. (2020). Federated Optimization in Heterogeneous Networks. MLSys.

Federation mechanism: same weighted FedAvg aggregation as FabricFedAvg.
The proximal term is applied during LOCAL training, not at aggregation.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .fabric_fedavg import FabricFedAvg

logger = logging.getLogger(__name__)


class FabricFedProx(FabricFedAvg):
    """
    FedProx with proximal regularisation.

    Parameters
    ----------
    mu : float
        Proximal term coefficient. 0.0 = reduces to FedAvg.
        Typical values: 0.001 to 1.0.
    """

    def __init__(self, n_banks: int, mu: float = 0.01, **kwargs):
        super().__init__(n_banks=n_banks, **kwargs)
        self.mu = mu

    def proximal_loss(
        self,
        model: nn.Module,
        global_params: List[np.ndarray],
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Compute the proximal term: (μ/2) * Σ ||w_i - w_global_i||²

        Parameters
        ----------
        model : current local model
        global_params : list of numpy arrays (global model weights)
        device : str

        Returns
        -------
        proximal_term : torch.Tensor (scalar)
        """
        prox = torch.tensor(0.0, device=device)
        for p, g in zip(model.parameters(), global_params):
            g_tensor = torch.tensor(g, dtype=p.dtype, device=device)
            prox += (self.mu / 2.0) * torch.sum((p - g_tensor) ** 2)
        return prox

    def configure_fit(self, round_num: int, parameters: List[np.ndarray]) -> Dict[str, Any]:
        config = super().configure_fit(round_num, parameters)
        config["mu"] = self.mu
        config["use_fedprox"] = True
        return config
