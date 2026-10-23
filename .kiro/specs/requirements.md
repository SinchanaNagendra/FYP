# Requirements – Blockchain-Secured Federated Learning for Multi-Bank Credit Card Fraud Detection

**Based on:** Salam et al. (2024), "Privacy-Preserving Credit Card Fraud Detection"  
**Extended with:** secure aggregation, differential privacy, real multi-bank non-IID data, attack evaluation, private blockchain layer.

---

## 1. Functional Requirements

### 1.1 Data Requirements
- **REQ-D01** Download and use the IEEE-CIS Fraud Detection dataset (Kaggle `ieee-fraud-detection`). No synthetic data.
- **REQ-D02** Use the ULB Credit Card Fraud dataset for paper-comparison tables only.
- **REQ-D03** Build pseudo-cardholder key `uid = hash(card1, card2, card3, card4, card5, addr1)` ensuring one cardholder's records never span two banks.
- **REQ-D04** Implement three partition scenarios: S1 (IID), S2 (feature/domain skew), S3 (Dirichlet label+quantity skew, α ∈ {0.1, 0.5, 1.0}).
- **REQ-D05** Split each bank's data by `TransactionDT`: 70% train / 15% validation / 15% test. No random splits.
- **REQ-D06** Never resample validation or test sets.
- **REQ-D07** Compute global feature-scaling statistics via one federated secure-aggregation round.
- **REQ-D08** Document per-bank size, fraud rate, and Jensen–Shannon distance from global in `docs/DATASET.md`.

### 1.2 Model Requirements
- **REQ-M01** Implement all six algorithms: Logistic Regression, CNN (1-D), Random Forest, Decision Tree, Gaussian Naive Bayes, KNN.
- **REQ-M02** Implement all seven balancing methods: ROS, RUS, SMOTE, ADASYN, Hybrid (SMOTEENN), Class-Weighted, None.
- **REQ-M03** Apply balancing only to each bank's training split. Never to validation or test.
- **REQ-M04** Federate LR and CNN via FedAvg on weight tensors (PyTorch + Flower).
- **REQ-M05** Federate GNB via exact secure aggregation of per-class sufficient statistics.
- **REQ-M06** Federate Decision Tree via ensemble of bank-local trees (soft vote).
- **REQ-M07** Federate Random Forest via union of bank-local forests (federated bagging).
- **REQ-M08** Run KNN as local-only and centralized baselines ONLY. Do NOT federate KNN. Document in `docs/DEVIATIONS.md`.
- **REQ-M09** Enforce `min_samples_leaf ≥ 20` on all tree models. Transmit leaf class probabilities, not counts.
- **REQ-M10** Implement FedProx as an optional alternative to FedAvg for non-IID scenarios (CNN/LR).
- **REQ-M11** Implement TFF comparison for LR and CNN in a Docker container.

### 1.3 Blockchain Requirements
- **REQ-B01** Deploy a real Hyperledger Fabric 2.5.x network with N orgs (default 5), 3-node Raft orderer, one channel `fraudchannel`, TLS everywhere, Fabric CA per org.
- **REQ-B02** Implement chaincode in Go with functions: `RegisterBank`, `StartRound`, `SubmitUpdate`, `SubmitAggregate`, `EndorseAggregate`, `FinalizeRound`, `ReportMetrics`, `RecordSecurityEvent`, `GetRound`, `GetHistory`, `GetBankReputation`.
- **REQ-B03** Enforce MSP-based access control: a bank may only submit for its own identity.
- **REQ-B04** Endorsement policy: majority of orgs must endorse.
- **REQ-B05** Chaincode validates: round state machine, no duplicate submissions, hash/CID format, declared L2 norm ≤ bound, quorum logic.
- **REQ-B06** Store encrypted weight artifacts in a private IPFS swarm (shared `swarm.key`). Anchor (CID, SHA-256, size, roundId, bankId) on-chain.
- **REQ-B07** Clients must verify SHA-256 after IPFS download; reject mismatches.
- **REQ-B08** Measure and report: tx latency, throughput, block size, per-round on-chain time, ledger growth.

### 1.4 Privacy Requirements
- **REQ-P01** Raw transaction rows never leave a bank's trust boundary. Proven by privacy audit tests.
- **REQ-P02** Implement pairwise-mask secure aggregation: X25519 key agreement, PRG mask expansion, fixed-point encoding in a finite ring.
- **REQ-P03** Handle bank dropout by aborting and re-running the round with the surviving set.
- **REQ-P04** Implement DP-SGD with Opacus for CNN and LR: per-sample gradient clipping, Gaussian noise, RDP accountant.
- **REQ-P05** Report ε, δ per run and cumulative per-round ε.
- **REQ-P06** Generate privacy–utility curve sweeping noise multiplier (ε ≈ {1, 3, 8, ∞}).
- **REQ-P07** For tree models, document k-leaf rule and state no formal DP is claimed unless implemented.
- **REQ-P08** All submissions signed via Fabric identities; SHA-256 anchored on-chain.

### 1.5 Security Evaluation Requirements
- **REQ-S01** Implement label-flipping poisoning for m ∈ {1, 2} malicious banks.
- **REQ-S02** Implement model poisoning: scaled/sign-flipped updates and backdoor-style update.
- **REQ-S03** Implement membership inference attack (loss-threshold and shadow-model) with and without DP.
- **REQ-S04** Implement free-rider/malicious participant (random or stale updates).
- **REQ-S05** Implement Mode A defences: norm clipping + on-chain norm/reputation checks + secure aggregation.
- **REQ-S06** Implement Mode B defences: robust aggregation (coordinate-wise median, trimmed mean, Krum) without secure aggregation.
- **REQ-S07** Report Mode A vs Mode B trade-off with numbers. Never claim both simultaneously.

### 1.6 Evaluation Requirements
- **REQ-E01** Report per model, per balancing, per framework, per scenario: accuracy, precision, recall, F1, PR-AUC, ROC-AUC, recall@1%FPR, log-loss, training time, round time, comm bytes, blockchain latency, ε.
- **REQ-E02** Run ≥ 3 seeds; report mean ± std.
- **REQ-E03** Baselines: (a) centralized pooled, (b) local-only per bank, (c) federated.
- **REQ-E04** Generate all tables and plots by script. All numbers in README/report trace to `results/`.
- **REQ-E05** Print majority-class baseline accuracy alongside reported accuracy.

### 1.7 Dashboard Requirements
- **REQ-UI01** FastAPI backend + React frontend.
- **REQ-UI02** Show: live rounds, per-bank status, ledger/block explorer, model metrics, comm cost, privacy budget (ε), attack results.
- **REQ-UI03** Data-boundary panel: bytes sent per bank, confirmed zero raw-record transfer.
- **REQ-UI04** Subscribe to chaincode events for real-time updates.

---

## 2. Non-Functional Requirements

- **NFR-01 Reproducibility:** fixed seeds, pinned dependencies (lock files), one command per stage, results written with config + git hash.
- **NFR-02 Privacy invariant:** proven by test suite, not asserted.
- **NFR-03 Honesty:** no tuning toward claimed paper metrics; report as obtained.
- **NFR-04 Traceability:** `docs/TRACEABILITY.md` maps every feature to implementing files and verifying test/command.
- **NFR-05 Deployability:** `make demo` runs the full system; supports N=3, 5, 8.
- **NFR-06 Environment:** Python 3.10/3.11, Docker + Docker Compose v2, WSL2 on Windows, ≥16 GB RAM.
- **NFR-07 Documentation:** DATASET.md, DEVIATIONS.md, TRACEABILITY.md, ARCHITECTURE.md, README with stated limitations.

---

## 3. Explicitly Out of Scope

- PySyft legacy FL API (unavailable in PySyft 0.8+). Deviation documented.
- Full secret-sharing recovery for secure aggregation dropout (abort-and-restart instead).
- Federated KNN (privacy invariant violation). Local/centralized baselines only.
- Formal DP for tree models (k-leaf rule applied; no formal claim).
