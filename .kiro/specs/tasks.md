# Implementation Tasks

## Phase 1 – Data Pipeline

- [ ] 1.1 Write `data/scripts/download.py` – Kaggle API download for IEEE-CIS and ULB
- [ ] 1.2 Write `data/scripts/partition.py` – uid key, S1/S2/S3 scenarios, temporal splits
- [ ] 1.3 Write `data/scripts/verify_splits.py` – gate check: stats table, no leakage, JS distances
- [ ] 1.4 Write `data/scripts/compute_vocab.py` – builds public frequency-bucket vocabulary (committed to repo)
- [ ] 1.5 Write `docs/DATASET.md`
- [ ] 1.6 Gate: run `make data-verify` and show output

## Phase 2 – Centralized + Local Baselines

- [ ] 2.1 Write `fl/preprocessing.py` – fixed schema, encoding, global scaling
- [ ] 2.2 Write `fl/models/` – all six model classes
- [ ] 2.3 Write `fl/balancing/methods.py` – all seven balancing methods
- [ ] 2.4 Write `experiments/run_baselines.py` – centralized and local training loop
- [ ] 2.5 Write `experiments/plot_baselines.py` – generates tables and plots to `results/baselines/`
- [ ] 2.6 Gate: run `make baselines-centralized baselines-local` and show PR-AUC table

## Phase 3 – Federated Learning (no blockchain)

- [ ] 3.1 Write `fl/secure_agg/` – keys, masking, ring ops, aggregator
- [ ] 3.2 Write `fl/strategies/fabric_fedavg.py` and `fabric_fedprox.py`
- [ ] 3.3 Write `fl/strategies/gnb_strategy.py` and `tree_strategy.py`
- [ ] 3.4 Write `fl/bank_node/trainer.py` and `evaluator.py`
- [ ] 3.5 Write `experiments/run_fl.py` – FL runner (no-blockchain mode)
- [ ] 3.6 Write `tests/unit/test_secure_agg.py`
- [ ] 3.7 Gate: run `make fl-train` and `make secagg-test`; show federated ≈ centralized gap table

## Phase 4 – Fabric Network + Chaincode

- [ ] 4.1 Write `fabric/network/configtx.yaml` – channel config, policies, orderer
- [ ] 4.2 Write `fabric/network/crypto-config.yaml` – org definitions
- [ ] 4.3 Write `fabric/scripts/network.sh` – up/down/reset
- [ ] 4.4 Write `fabric/scripts/generate_compose.py` – generates docker-compose for N=3/5/8
- [ ] 4.5 Write `fabric/scripts/swarm.key` – private IPFS swarm key
- [ ] 4.6 Write `fabric/chaincode/frauddetect/` – Go chaincode (all required functions)
- [ ] 4.7 Write `fabric/chaincode/frauddetect/frauddetect_test.go` – unit tests
- [ ] 4.8 Write `fabric/scripts/deploy_chaincode.sh`
- [ ] 4.9 Gate: run `make chaincode-test`; show `go test ./... -v` passing

## Phase 5 – Integrated FL + Fabric + IPFS

- [ ] 5.1 Write `fl/bank_node/fabric_client.py` – Gateway SDK integration
- [ ] 5.2 Write `fl/bank_node/ipfs_client.py` – encrypt/upload/download/verify
- [ ] 5.3 Write `fl/bank_node/node.py` – full round lifecycle
- [ ] 5.4 Write `experiments/run_fl.py` – with-blockchain mode (update existing)
- [ ] 5.5 Write `tests/integration/test_full_round.py`
- [ ] 5.6 Gate: run `make fl-fabric-run` (20 rounds); show ledger history and hash verification

## Phase 6 – Secure Aggregation + DP

- [ ] 6.1 Write `fl/dp/dp_trainer.py` – Opacus PrivacyEngine wrapper
- [ ] 6.2 Write `fl/dp/accountant.py` – RDP accountant, cumulative ε
- [ ] 6.3 Write `experiments/run_dp_sweep.py` – noise multiplier sweep
- [ ] 6.4 Write `experiments/plot_dp_curve.py` – privacy–utility curve
- [ ] 6.5 Write `tests/privacy_audit/` – all 5 audit tests
- [ ] 6.6 Gate: run `make privacy-audit dp-train`; show privacy–utility curve

## Phase 7 – Attacks and Defences

- [ ] 7.1 Write `attacks/poisoning/label_flip.py`
- [ ] 7.2 Write `attacks/poisoning/model_poison.py`
- [ ] 7.3 Write `attacks/poisoning/free_rider.py`
- [ ] 7.4 Write `attacks/inference/membership_inference.py`
- [ ] 7.5 Write `attacks/defences/robust_agg.py` – median, trimmed mean, Krum
- [ ] 7.6 Write `experiments/run_attacks.py` – Mode A and Mode B evaluation
- [ ] 7.7 Write `experiments/plot_attacks.py`
- [ ] 7.8 Gate: run `make attacks-run`; show Mode A vs Mode B table

## Phase 8 – TFF Comparison

- [ ] 8.1 Write `fl_tff/run_tff.py` – TFF LR and CNN
- [ ] 8.2 Write `fl_tff/Dockerfile` and `fl_tff/docker-compose.tff.yml`
- [ ] 8.3 Write `fl_tff/requirements.txt`
- [ ] 8.4 Write `experiments/compare_frameworks.py` – generates comparison table
- [ ] 8.5 Gate: run `make tff-compare`; show comparison table

## Phase 9 – Dashboard

- [ ] 9.1 Write `dashboard/backend/main.py` – FastAPI app with all endpoints
- [ ] 9.2 Write `dashboard/backend/fabric_gateway.py` – chaincode query/event subscription
- [ ] 9.3 Write `dashboard/backend/Dockerfile`
- [ ] 9.4 Write `dashboard/frontend/src/` – React components (Rounds, Blocks, Metrics, Privacy, Attacks, DataBoundary)
- [ ] 9.5 Write `dashboard/frontend/Dockerfile`
- [ ] 9.6 Gate: run `make dashboard-up`; demo live round with data-boundary panel

## Phase 10 – Full Matrix + Report Assets

- [ ] 10.1 Write `experiments/run_matrix.py` – full grid (3 seeds × 3 N-banks × 3 scenarios)
- [ ] 10.2 Write `experiments/generate_report_tables.py` – all LaTeX/CSV tables
- [ ] 10.3 Write `experiments/generate_report_figures.py` – all plots
- [ ] 10.4 Complete `docs/TRACEABILITY.md`
- [ ] 10.5 Complete `docs/DEVIATIONS.md`
- [ ] 10.6 Complete `docs/ARCHITECTURE.md`
- [ ] 10.7 Complete `README.md` with stated limitations
- [ ] 10.8 Gate: run `make experiments-all`; verify all results trace to files
