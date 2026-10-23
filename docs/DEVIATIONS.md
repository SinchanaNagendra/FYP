# DEVIATIONS.md

This file documents every deliberate departure from the source paper (Salam et al., 2024) and the master prompt. Each entry states the reason and what was done instead.

---

## DEV-01: KNN not federated

**Paper claim:** KNN listed as one of the six federated algorithms.

**Why deviation:** KNN is a non-parametric, instance-based model. It stores the training data as the model itself. Federating KNN would require transmitting the raw training records to other banks, which directly violates the privacy invariant (Rule 2 of the master prompt).

**What we do instead:** KNN is implemented as a local-only baseline (each bank trains and evaluates on its own data) and as a centralized baseline (all data pooled, for the paper comparison). It is never included in the federated evaluation.

**Verification:** `tests/unit/test_models.py::test_knn_raises_on_federate` asserts that calling `KNNModel.get_parameters()` raises `PrivacyViolationError`.

---

## DEV-02: PySyft legacy FL API unavailable

**Paper claim:** Compares PyTorch/PySyft vs TensorFlow Federated.

**Why deviation:** PySyft 0.8+ is a data-science server platform (Syft). The legacy FL API (RemoteWorker, send/get, Plan) used in the paper has been removed. Installation of `syft==0.8.x` confirms no `FederatedClient`, `BaseWorker`, or `Plan` classes exist. Running `pip install syft==0.2.9` (the legacy version used in the paper era) fails on Python 3.10+ due to incompatible dependencies.

**What we do instead:** The "PyTorch FL" column in the comparison table uses **Flower (`flwr`) + PyTorch**, which is the current standard replacement. The TFF column uses TensorFlow Federated in its own Docker container. This gives a valid framework comparison; we note in all tables that PySyft was unavailable and Flower was substituted.

**Verification:** `docs/DEVIATIONS.md` (this file); noted in README and report.

---

## DEV-03: Formal DP not claimed for tree models

**Paper claim:** Differential privacy applied across all models.

**Why deviation:** DP-SGD (Opacus) requires gradient-based training. Scikit-learn's `DecisionTreeClassifier` and `RandomForestClassifier` are not differentiable and do not support per-sample gradient clipping.

**What we do instead:** Tree models use the **k-leaf rule** (`min_samples_leaf ≥ 20`) so no leaf can expose an individual's record. Leaf class probabilities are transmitted, not counts. This provides some privacy protection but does not meet the formal (ε, δ)-DP definition. The report explicitly states: "Formal differential privacy (via DP-SGD) is applied to Logistic Regression and CNN only. Tree models use the k-leaf rule; no formal DP guarantee is claimed for them."

**Verification:** `tests/unit/test_models.py::test_tree_min_samples_leaf` asserts the constraint is enforced.

---

## DEV-04: Secure aggregation dropout — abort-and-restart only

**Paper scope:** Secure aggregation with dropout recovery.

**Why deviation:** Full dropout recovery using secret sharing (Shamir's) would require a complete implementation of the Bonawitz et al. (2017) protocol including the double-masking scheme for handling surviving/dropped-out sets. This is a significant standalone engineering effort.

**What we do instead:** If a bank fails to submit during the secure aggregation window, the round is aborted and re-run with the surviving set. The keys are regenerated for the new participant set. This is safe (no information leaked) but less efficient than the full recovery protocol.

**Verification:** `tests/unit/test_secure_agg.py::test_dropout_aborts_round`

---

## DEV-05: Banks simulated on one real dataset (not truly multi-institutional data)

**Paper/prompt context:** The system is designed for multiple real banks.

**Reality:** Neither the IEEE-CIS nor ULB dataset contains a real bank identifier. We simulate multi-bank distribution using a pseudo-cardholder key (`uid = hash(card1, card2, card3, card4, card5, addr1)`) and three partition scenarios (S1 IID, S2 feature skew, S3 Dirichlet label+quantity skew).

**Documentation:** All experiment tables, the README, and DATASET.md explicitly state: "Banks are simulated by partitioning a single real-world dataset. This is a simulation of the multi-bank setting, not a deployment with truly distinct institutional data."

---

*Last updated: auto-generated; update with each new deviation as discovered.*
