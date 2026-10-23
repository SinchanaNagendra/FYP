# Design – Blockchain-Secured Federated Learning for Multi-Bank Credit Card Fraud Detection

---

## 1. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Hyperledger Fabric Network                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  (fraudchannel)           │
│  │ Orderer0 │  │ Orderer1 │  │ Orderer2 │  ← Raft consensus         │
│  └──────────┘  └──────────┘  └──────────┘                           │
│                                                                        │
│  ┌────────────┐ ┌────────────┐ ... ┌────────────┐                   │
│  │ Bank0 Peer │ │ Bank1 Peer │     │ BankN Peer │  ← MSP per org    │
│  └────────────┘ └────────────┘     └────────────┘                   │
└──────────────────────────────────────────────────────────────────────┘
         │ chaincode events (CID, hash, metadata only)
         ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     Private IPFS Swarm (swarm.key)                    │
│  ┌──────────┐ ┌──────────┐ ... ┌──────────┐                         │
│  │ IPFS-0   │ │ IPFS-1   │     │ IPFS-N   │  ← encrypted artifacts  │
│  └──────────┘ └──────────┘     └──────────┘                         │
└──────────────────────────────────────────────────────────────────────┘
         │ encrypted model updates only
         ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      FL Node Layer (per bank)                         │
│                                                                        │
│  Bank 0 FL Node                Bank 1 FL Node  ...                   │
│  ┌──────────────────────┐     ┌──────────────────────┐               │
│  │ local data volume    │     │ local data volume    │               │
│  │ (bank0_data, RO)     │     │ (bank1_data, RO)     │               │
│  │ ─────────────────    │     │ ─────────────────    │               │
│  │ BankNode             │     │ BankNode             │               │
│  │  ├ DataLoader        │     │  ├ DataLoader        │               │
│  │  ├ LocalTrainer      │     │  ├ LocalTrainer      │               │
│  │  ├ SecureAggClient   │     │  ├ SecureAggClient   │               │
│  │  ├ DPTrainer (Opacus)│     │  ├ DPTrainer (Opacus)│               │
│  │  ├ FabricClient      │     │  ├ FabricClient      │               │
│  │  └ IPFSClient        │     │  └ IPFSClient        │               │
│  └──────────────────────┘     └──────────────────────┘               │
└──────────────────────────────────────────────────────────────────────┘
         │ metrics, hashes only
         ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      Dashboard (FastAPI + React)                       │
│  Rounds panel │ Block explorer │ Metrics │ ε budget │ Data boundary  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Design

### 2.1 Data Pipeline (`data/`)

**`data/scripts/download.py`**
- Uses `kaggle competitions download -c ieee-fraud-detection`
- Downloads ULB dataset from Kaggle
- Verifies checksums
- Stores raw CSVs in `data/raw/` (gitignored)

**`data/scripts/partition.py`**
- Joins `train_transaction.csv` + `train_identity.csv` on `TransactionID`
- Builds `uid = SHA-256(card1 || card2 || card3 || card4 || card5 || addr1)` (hex, first 16 chars for stability)
- Three scenarios:
  - **S1 IID:** shuffle uids, assign round-robin to N banks
  - **S2 Feature skew:** cluster uids by modal `ProductCD`, `card4`, `card6`; assign clusters to banks
  - **S3 Dirichlet:** sample uid→bank assignment from Dirichlet(α) per label; α ∈ {0.1, 0.5, 1.0}
- Temporal split per bank: earliest 70% → `train`, next 15% → `val`, last 15% → `test`
- Saves per-bank Parquet files to `data/partitions/N{n}/{scenario}/bank{i}/`
- Computes and saves `data/partitions/N{n}/{scenario}/stats.json`

**`data/scripts/verify_splits.py`**
- Reads `stats.json`; prints per-bank table
- Asserts: no uid appears in two banks; val/test contain no training records; temporal ordering holds
- Computes Jensen–Shannon distance per bank vs global distribution
- Exits 0 only if all checks pass

### 2.2 Preprocessing Schema (`fl/preprocessing.py`)

Fixed public schema applied identically at each bank:
- Frequency-bucket encoding for high-cardinality categoricals (bucket counts fixed from a public vocabulary file generated once and committed)
- Missing-value indicator columns
- Global scaling: `(x - μ) / σ` where μ, σ come from the federated secure-sum round

### 2.3 FL Models (`fl/models/`)

```
fl/models/
  __init__.py
  logistic_regression.py   # nn.Linear(n_features, 1) → FedAvg on weights
  cnn.py                   # 1-D CNN over feature vector → FedAvg on weights
  gaussian_nb.py           # Tracks per-class (n, sum, sum_sq) → exact agg
  decision_tree.py         # sklearn tree → leaf-probability ensemble
  random_forest.py         # sklearn forest → federated bagging union
  knn.py                   # local/centralized ONLY; raises if federation attempted
```

### 2.4 Balancing (`fl/balancing/`)

```
fl/balancing/
  __init__.py
  methods.py   # Enum + factory: NONE, WEIGHTED, ROS, RUS, SMOTE, ADASYN, HYBRID_SMOTEENN
```

Applied in `BankNode.train()` before each local epoch, on training split only.

### 2.5 Secure Aggregation (`fl/secure_agg/`)

Protocol (Bonawitz-style pairwise masking):

1. **KeyGen:** each bank generates an X25519 keypair; broadcasts public key on-chain (or via peer-to-peer channel).
2. **Shared secrets:** for each pair (i, j), compute `shared = X25519(sk_i, pk_j)`; derive a PRG seed via HKDF-SHA256.
3. **Mask generation:** for each bank i, mask = Σ_{j>i} PRG(seed_{ij}) − Σ_{j<i} PRG(seed_{ji}), quantised to fixed-point ring Z_{2^l}.
4. **Masked update:** `m_i = encode(w_i) + mask_i mod 2^l`
5. **Aggregate:** `Σ m_i mod 2^l = Σ encode(w_i)` (masks cancel exactly)
6. **Decode:** decode the aggregate back to float.

Dropout handling: if a bank fails to submit, the surviving banks re-run without it (masks regenerated). Full secret-sharing recovery is out of scope (see DEVIATIONS.md).

Files:
```
fl/secure_agg/
  __init__.py
  keys.py        # X25519 key generation, HKDF seed derivation
  masking.py     # PRG, fixed-point encoding/decoding, mask computation
  aggregator.py  # Coordinates the secure sum across banks
  ring_ops.py    # Arithmetic in Z_{2^l}
```

### 2.6 Differential Privacy (`fl/dp/`)

```
fl/dp/
  __init__.py
  dp_trainer.py   # Wraps Opacus PrivacyEngine; per-sample clipping + Gaussian noise
  accountant.py   # RDP → (ε, δ) conversion; cumulative ε tracking
```

### 2.7 Federated Strategy (`fl/strategies/`)

```
fl/strategies/
  __init__.py
  fabric_fedavg.py    # FabricFedAvg: replaces Flower's central server
                       # Drives round lifecycle via FabricClient
  fabric_fedprox.py   # FabricFedProx: adds proximal term for non-IID
  gnb_strategy.py     # GNB sufficient-stat aggregation
  tree_strategy.py    # Tree ensemble aggregation
```

### 2.8 Bank Node (`fl/bank_node/`)

```
fl/bank_node/
  __init__.py
  node.py         # BankNode: main entry point per bank container
  fabric_client.py # Wraps Fabric Gateway SDK calls
  ipfs_client.py   # IPFS upload/download + encrypt/decrypt + hash verify
  trainer.py       # Local training loop (standard + DP-SGD)
  evaluator.py     # Local evaluation → metrics dict
```

### 2.9 Chaincode (`fabric/chaincode/`)

State objects (JSON in state DB):
```go
type Round struct {
    RoundID        string
    Status         string   // OPEN | COLLECTING | AGGREGATING | FINALIZED
    GlobalModelCID string
    GlobalModelHash string
    HParams        map[string]interface{}
    Participants   []string
    Updates        map[string]UpdateRecord
    AggCID         string
    AggHash        string
    Endorsements   map[string]bool
    StartTime      int64
    FinalizeTime   int64
}

type UpdateRecord struct {
    BankID       string
    CID          string
    SHA256       string
    Size         int64
    DeclaredNorm float64
    Timestamp    int64
}

type BankReputation struct {
    BankID         string
    TotalRounds    int
    GoodRounds     int
    FlaggedRounds  int
    Score          float64
}
```

State machine transitions:
```
OPEN → COLLECTING (banks submitting updates)
COLLECTING → AGGREGATING (quorum reached)
AGGREGATING → FINALIZED (majority endorsement)
```

### 2.10 Dashboard (`dashboard/`)

**Backend (`dashboard/backend/`):**
- FastAPI app
- Endpoints: `/rounds`, `/rounds/{id}`, `/banks`, `/metrics`, `/blocks`, `/privacy`, `/attacks`, `/boundary`
- WebSocket endpoint `/ws/events` for chaincode event streaming

**Frontend (`dashboard/frontend/`):**
- React + TypeScript + Tailwind CSS
- Panels: RoundsView, BlockExplorer, MetricsChart, PrivacyBudget, AttackResults, DataBoundary
- Uses WebSocket for live updates

---

## 3. Round Flow Sequence

```
Bank_i                    Chaincode               Bank_j (any bank)
  │                           │                        │
  │── StartRound(params) ────►│                        │
  │◄─ roundId, globalCID ─────│                        │
  │                           │                        │
  │ [fetch global model from IPFS, verify hash]       │
  │ [train locally with DP-SGD + balancing]           │
  │ [secure-agg: KeyGen → broadcast pk]               │
  │ [secure-agg: compute masked update]               │
  │ [encrypt artifact, upload to IPFS → CID]          │
  │── SubmitUpdate(roundId, bankId, CID, hash, meta)─►│
  │                           │                        │
  :  (N banks submit)         :                        :
  │                           │                        │
  │                    [quorum reached]                │
  │◄─ chaincode event: QUORUM_REACHED ─────────────────┤
  │                           │                        │
  │ [download all masked updates from IPFS]           │
  │ [verify all hashes]                               │
  │ [compute secure aggregate: sum → decode]          │
  │ [encrypt aggregate, upload to IPFS → aggCID]      │
  │── SubmitAggregate(roundId, aggCID, aggHash) ─────►│
  │                           │                        │
  │                           │◄── EndorseAggregate ───│
  │                           │    (majority)          │
  │◄─ FinalizeRound ──────────│                        │
  │                           │                        │
  │ [evaluate on local val set]                       │
  │── ReportMetrics(roundId, bankId, metrics) ───────►│
```

---

## 4. Privacy Audit Design

Five tests in `tests/privacy_audit/`:

1. **`test_canary.py`** – injects canary rows (uniquely identifiable strings in unused columns), runs a full round, captures all inter-container traffic and ledger contents, asserts no canary appears outside the bank's volume.
2. **`test_boundary.py`** – intercepts all outbound payloads from each FL container; validates schema (tensors + hashes + metadata only); fails on any row-shaped data.
3. **`test_isolation.py`** – from bank1 container, attempts to read bank0's data volume; asserts permission denied.
4. **`test_secure_agg.py`** – records single masked update; attempts to unmask without peer keys; asserts statistical indistinguishability.
5. **`test_tamper.py`** – modifies a IPFS artifact; confirms hash mismatch detection, rejection, and chaincode security event.

---

## 5. Dependency Versions (pinned)

```
# Python FL stack
torch==2.2.2
torchvision==0.17.2
flwr==1.8.0
opacus==1.4.0
scikit-learn==1.4.2
imbalanced-learn==0.12.2
pandas==2.2.2
numpy==1.26.4
pyarrow==16.0.0
scipy==1.13.0
cryptography==42.0.7
ipfshttpclient==0.8.0a2
grpcio==1.63.0
protobuf==4.25.3
kaggle==1.6.12
fastapi==0.111.0
uvicorn==0.29.0
websockets==12.0
pytest==8.2.0

# Fabric
hyperledger/fabric-peer:2.5.9
hyperledger/fabric-orderer:2.5.9
hyperledger/fabric-ca:1.5.12
ipfs/kubo:v0.26.0

# Go chaincode
go 1.21
github.com/hyperledger/fabric-contract-api-go v1.2.2
```
