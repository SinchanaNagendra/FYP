# ARCHITECTURE.md

## System Architecture: Blockchain-Secured Federated Learning for Multi-Bank Credit Card Fraud Detection

---

## 1. High-Level Architecture

The system consists of five layers:

```
┌────────────────────────────────────────────────────────────────┐
│  Layer 5: Dashboard (FastAPI + React)                           │
│  Live rounds, block explorer, metrics, ε budget, data boundary │
└────────────────────────────────────────────────────────────────┘
                          │ REST / WebSocket
┌────────────────────────────────────────────────────────────────┐
│  Layer 4: Coordination + Trust (Hyperledger Fabric 2.5.x)      │
│  fraudchannel | Raft orderer ×3 | 5 org MSPs                   │
│  Chaincode: StartRound, SubmitUpdate, …, FinalizeRound         │
└────────────────────────────────────────────────────────────────┘
           │ CID + hash anchoring       │ chaincode events
┌──────────────────┐        ┌───────────────────────────────────┐
│  Layer 3: Storage│        │  Layer 2: Federated Learning       │
│  Private IPFS    │        │  Bank FL Nodes (N=3/5/8)           │
│  (swarm.key)     │        │  Flower/PyTorch  |  Opacus DP-SGD  │
│  Encrypted       │        │  Secure Aggregation (X25519+PRG)   │
│  artifacts only  │        │  Local training, never shares data │
└──────────────────┘        └───────────────────────────────────┘
                                        │ reads (read-only)
┌────────────────────────────────────────────────────────────────┐
│  Layer 1: Data (per-bank isolated volumes)                      │
│  bank0_data | bank1_data | … | bankN_data                      │
│  IEEE-CIS partitioned by pseudo-cardholder uid                 │
│  Access: only that bank's FL container (read-only mount)       │
└────────────────────────────────────────────────────────────────┘
```

---

## 2. Network Topology

```
  Orderer Org (Raft)
  ┌──────────────────────────────────────┐
  │  orderer0:7050  orderer1:8050  orderer2:9050  │
  └──────────────────────────────────────┘
            │ TLS-encrypted gRPC
  ┌─────────┴─────────┬──────────────────────┐
  │                   │                      │
  peer0.bank0:7051  peer0.bank1:8051  ... peer0.bank4:11051
  (Bank0MSP)        (Bank1MSP)             (Bank4MSP)
  │                   │
  CA:bank0:7054     CA:bank1:8054       ...
```

---

## 3. FL Round Data Flow

Only the following cross the network boundary for each bank:

| What crosses | Format | Direction | Size (approx.) |
|---|---|---|---|
| Global model CID + hash | 64 bytes | Ledger → Banks | Negligible |
| Masked model update | Encrypted binary | Bank → IPFS | ~100 KB–10 MB depending on model |
| CID + SHA-256 + metadata | <200 bytes | Bank → Ledger | Negligible |
| Aggregate model CID + hash | 64 bytes | Ledger → Banks | Negligible |
| Metrics (PR-AUC, F1, etc.) | ~512 bytes JSON | Bank → Ledger | Negligible |

**What NEVER crosses:**
- Raw transaction rows
- Feature matrices
- Labels
- Any record-shaped data

---

## 4. Security Model

**Threat model:** Honest-but-curious banks (each follows the protocol but may try to infer others' data). The blockchain layer handles Byzantine fault tolerance at the consensus level.

**Controls:**
- MSP-based identity: each bank signs all submissions; a bank cannot impersonate another
- Secure aggregation: individual updates are masked; no bank (or coordinator) sees plaintext updates
- DP-SGD: adds calibrated Gaussian noise before masking, providing ε-DP per round for CNN/LR
- Norm bound: chaincode rejects updates with declared norm above threshold (anti-poisoning)
- Reputation scoring: chaincode tracks per-bank behaviour; repeated violations flag a bank
- Encrypted IPFS: artifacts are AES-256-GCM encrypted before upload; key not stored in IPFS
- TLS everywhere: all Fabric communications use mutual TLS

---

## 5. Repository Map

```
FYPKiro/
├── fabric/                     Hyperledger Fabric network + chaincode
│   ├── network/                configtx.yaml, crypto-config.yaml
│   ├── chaincode/frauddetect/  Go chaincode + unit tests
│   ├── scripts/                network.sh, deploy_chaincode.sh, swarm.key
│   └── crypto-config/          generated MSP material (gitignored)
│
├── fl/                         Federated learning stack (PyTorch + Flower)
│   ├── models/                 LR, CNN, GNB, DT, RF, KNN
│   ├── balancing/              ROS, RUS, SMOTE, ADASYN, HYBRID, WEIGHTED, NONE
│   ├── secure_agg/             X25519 masking, ring ops, aggregator
│   ├── dp/                     Opacus DP-SGD, RDP accountant
│   ├── strategies/             FabricFedAvg, FabricFedProx, GNB, Tree
│   ├── bank_node/              BankNode, FabricClient, IPFSClient, Trainer, Evaluator
│   ├── preprocessing.py        Fixed schema, encoding, scaling
│   ├── requirements.txt        Pinned Python deps
│   └── Dockerfile
│
├── fl_tff/                     TFF comparison (separate Docker)
│   ├── run_tff.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── data/
│   ├── scripts/                download.py, partition.py, verify_splits.py
│   ├── raw/                    (gitignored – downloaded data)
│   └── partitions/             (gitignored – generated partitions)
│
├── attacks/
│   ├── poisoning/              label_flip.py, model_poison.py, free_rider.py
│   ├── inference/              membership_inference.py
│   └── defences/               robust_agg.py (median, trimmed mean, Krum)
│
├── dashboard/
│   ├── backend/                FastAPI, fabric_gateway.py
│   └── frontend/               React + TypeScript + Tailwind
│
├── experiments/
│   ├── configs/                YAML experiment configs
│   ├── run_baselines.py
│   ├── run_fl.py
│   ├── run_dp_sweep.py
│   ├── run_attacks.py
│   ├── run_matrix.py
│   └── utils.py                git hash, result writer
│
├── results/                    All generated outputs (gitignored except .gitkeep)
├── tests/
│   ├── unit/
│   ├── integration/
│   └── privacy_audit/
│
├── docs/                       ARCHITECTURE, DATASET, DEVIATIONS, TRACEABILITY, README
├── docker-compose.yml
└── Makefile
```
