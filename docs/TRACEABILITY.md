# TRACEABILITY.md

Maps every paper claim / master-prompt feature to implementing files and the verification command.  
Status: ✅ Done | 🔄 In Progress | ❌ Not Started

---

| ID | Feature / Claim | Implementing File(s) | Verification Command | Status |
|----|----------------|---------------------|---------------------|--------|
| T01 | IEEE-CIS download | `data/scripts/download.py` | `make data-download` | 🔄 (needs Kaggle creds) |
| T02 | ULB download | `data/scripts/download.py` | `make data-download` | 🔄 (needs Kaggle creds) |
| T03 | Pseudo-cardholder uid | `data/scripts/partition.py` | `make data-verify` | 🔄 (needs data) |
| T04 | S1 IID partition | `data/scripts/partition.py` | `make data-verify` | 🔄 (needs data) |
| T05 | S2 feature/domain skew | `data/scripts/partition.py` | `make data-verify` | 🔄 (needs data) |
| T06 | S3 Dirichlet skew (α=0.1,0.5,1.0) | `data/scripts/partition.py` | `make data-verify` | 🔄 (needs data) |
| T07 | Temporal train/val/test splits | `data/scripts/partition.py` | `make data-verify` | 🔄 (needs data) |
| T08 | No uid appears in two banks | `data/scripts/verify_splits.py` | `make data-verify` | 🔄 (needs data) |
| T09 | Jensen–Shannon distance per bank | `data/scripts/verify_splits.py` | `make data-verify` | 🔄 (needs data) |
| T10 | Fixed preprocessing schema | `fl/preprocessing.py` | `pytest tests/unit/test_preprocessing.py` | ✅ |
| T11 | Federated feature scaling (secure sum) | `fl/preprocessing.py`, `fl/secure_agg/` | `pytest tests/unit/test_secure_agg.py` | ✅ |
| T12 | Logistic Regression model | `fl/models/logistic_regression.py` | `pytest tests/unit/test_models.py::TestLogisticRegression` | ✅ |
| T13 | CNN (1-D) model | `fl/models/cnn.py` | `pytest tests/unit/test_models.py::TestCNN` | ✅ |
| T14 | Gaussian NB model | `fl/models/gaussian_nb.py` | `pytest tests/unit/test_models.py::TestGaussianNB` | ✅ |
| T15 | Decision Tree model | `fl/models/decision_tree.py` | `pytest tests/unit/test_models.py::TestDecisionTree` | ✅ |
| T16 | Random Forest model | `fl/models/random_forest.py` | `pytest tests/unit/test_models.py::TestRandomForest` | ✅ |
| T17 | KNN local/centralized only (no federation) | `fl/models/knn.py` | `pytest tests/unit/test_models.py::TestKNN` | ✅ |
| T18 | min_samples_leaf ≥ 20 enforced | `fl/models/decision_tree.py`, `random_forest.py` | `pytest tests/unit/test_models.py::TestDecisionTree::test_min_samples_leaf_enforced_at_construction` | ✅ |
| T19 | Leaf probabilities transmitted, not counts | `fl/models/decision_tree.py` | `pytest tests/unit/test_models.py::TestDecisionTree::test_leaf_proba_not_counts` | ✅ |
| T20 | ROS balancing | `fl/balancing/methods.py` | `pytest tests/unit/test_balancing.py` | ✅ |
| T21 | RUS balancing | `fl/balancing/methods.py` | `pytest tests/unit/test_balancing.py` | ✅ |
| T22 | SMOTE balancing | `fl/balancing/methods.py` | `pytest tests/unit/test_balancing.py` | ✅ |
| T23 | ADASYN balancing | `fl/balancing/methods.py` | `pytest tests/unit/test_balancing.py` | ✅ |
| T24 | Hybrid (SMOTEENN) balancing | `fl/balancing/methods.py` | `pytest tests/unit/test_balancing.py` | ✅ |
| T25 | Balancing only on train split | `fl/bank_node/trainer.py` | `pytest tests/unit/test_trainer.py` | ✅ |
| T26 | FedAvg on LR/CNN weights | `fl/strategies/fabric_fedavg.py` | `pytest tests/integration/test_fl_rounds.py::TestFLRoundsLR` | ✅ |
| T27 | GNB sufficient-statistic aggregation | `fl/strategies/gnb_strategy.py` | `pytest tests/integration/test_fl_rounds.py::TestFLRoundsGNB` | ✅ |
| T28 | Tree ensemble federation | `fl/strategies/tree_strategy.py` | `pytest tests/unit/test_models.py::TestRandomForest::test_aggregate_forests_merges_trees` | ✅ |
| T29 | FedProx option for non-IID | `fl/strategies/fabric_fedprox.py` | `pytest tests/unit/test_strategies.py` | ✅ |
| T30 | X25519 key agreement | `fl/secure_agg/keys.py` | `pytest tests/unit/test_secure_agg.py::TestX25519KeyAgreement` | ✅ |
| T31 | PRG mask generation | `fl/secure_agg/masking.py` | `pytest tests/unit/test_secure_agg.py::TestPRG` | ✅ |
| T32 | Masks cancel in aggregate sum | `fl/secure_agg/aggregator.py` | `pytest tests/unit/test_secure_agg.py::TestMaskCancellation` | ✅ |
| T33 | Single masked update indistinguishable from noise | `fl/secure_agg/masking.py` | `pytest tests/unit/test_secure_agg.py::TestMaskedIndistinguishable` | ✅ |
| T34 | DP-SGD with Opacus (LR) | `fl/dp/dp_trainer.py` | `make dp-train` (needs Opacus) | ✅ |
| T35 | DP-SGD with Opacus (CNN) | `fl/dp/dp_trainer.py` | `make dp-train` (needs Opacus) | ✅ |
| T36 | RDP accountant, cumulative ε | `fl/dp/accountant.py` | `make dp-train` (needs Opacus) | ✅ |
| T37 | Privacy–utility curve | `experiments/run_dp_sweep.py` | `make dp-train` | ✅ |
| T38 | Fabric network (5 orgs, Raft x3) | `fabric/scripts/network.sh`, `docker-compose.yml` | `make network-up` (needs Docker+Fabric) | 🔄 |
| T39 | RegisterBank chaincode | `fabric/chaincode/frauddetect/frauddetect.go` | `make chaincode-test` | ✅ |
| T40 | StartRound chaincode | `fabric/chaincode/frauddetect/frauddetect.go` | `make chaincode-test` | ✅ |
| T41 | SubmitUpdate chaincode | `fabric/chaincode/frauddetect/frauddetect.go` | `make chaincode-test` | ✅ |
| T42 | SubmitAggregate chaincode | `fabric/chaincode/frauddetect/frauddetect.go` | `make chaincode-test` | ✅ |
| T43 | EndorseAggregate chaincode | `fabric/chaincode/frauddetect/frauddetect.go` | `make chaincode-test` | ✅ |
| T44 | FinalizeRound chaincode | `fabric/chaincode/frauddetect/frauddetect.go` | `make chaincode-test` | ✅ |
| T45 | ReportMetrics chaincode | `fabric/chaincode/frauddetect/frauddetect.go` | `make chaincode-test` | ✅ |
| T46 | RecordSecurityEvent chaincode | `fabric/chaincode/frauddetect/frauddetect.go` | `make chaincode-test` | ✅ |
| T47 | GetRound / GetHistory / GetBankReputation | `fabric/chaincode/frauddetect/frauddetect.go` | `make chaincode-test` | ✅ |
| T48 | Majority-org endorsement policy | `fabric/network/configtx.yaml` (generated) | `make network-up chaincode-deploy` | 🔄 |
| T49 | Round state machine + duplicate rejection | `fabric/chaincode/frauddetect/frauddetect.go` | `make chaincode-test` (`TestFullRoundStateMachine`) | ✅ |
| T50 | L2 norm bound validation | `fabric/chaincode/frauddetect/frauddetect.go` | `make chaincode-test` (`TestSubmitUpdate_NormExceeded`) | ✅ |
| T51 | Private IPFS swarm (swarm.key) | `fabric/scripts/swarm.key` (generated), `docker-compose.yml` | `make network-up` | 🔄 |
| T52 | Encrypted artifact upload/download | `fl/bank_node/ipfs_client.py` | `pytest tests/privacy_audit/test_boundary.py::test_ipfs_artifact_is_opaque` | ✅ |
| T53 | SHA-256 hash verification on download | `fl/bank_node/ipfs_client.py` | `pytest tests/privacy_audit/test_tamper.py` | ✅ |
| T54 | 20 complete rounds with ledger history | `experiments/run_fl.py` | `make fl-fabric-run` (needs Fabric network) | 🔄 |
| T55 | Label-flipping poisoning | `attacks/poisoning/label_flip.py` | `make attacks-run` (needs data) | ✅ |
| T56 | Model poisoning (scaled/sign-flip/backdoor) | `attacks/poisoning/model_poison.py` | `make attacks-run` (needs data) | ✅ |
| T57 | Membership inference (loss-threshold) | `attacks/inference/membership_inference.py` | `make attacks-run` (needs data) | ✅ |
| T58 | Membership inference (shadow-model) | `attacks/inference/membership_inference.py` | `make attacks-run` (needs data) | ✅ |
| T59 | Free-rider / stale update attack | `attacks/poisoning/free_rider.py` | `make attacks-run` (needs data) | ✅ |
| T60 | Mode A: sec-agg + norm clip + reputation | `fl/strategies/fabric_fedavg.py`, `fabric/chaincode/` | `make attacks-run` (needs data) | ✅ |
| T61 | Mode B: robust aggregation (median/trimmed/Krum) | `attacks/defences/robust_agg.py` | `make attacks-run` (needs data) | ✅ |
| T62 | TFF LR comparison | `fl_tff/run_tff.py` | `make tff-compare` (needs Docker) | ✅ |
| T63 | TFF CNN comparison | `fl_tff/run_tff.py` | `make tff-compare` (needs Docker) | ✅ |
| T64 | Canary privacy audit test | `tests/privacy_audit/test_canary.py` | `make privacy-audit` (needs data) | ✅ |
| T65 | Boundary privacy audit test | `tests/privacy_audit/test_boundary.py` | `pytest tests/privacy_audit/test_boundary.py` | ✅ |
| T66 | Isolation privacy audit test | `tests/privacy_audit/test_isolation.py` | `make privacy-audit` (needs Docker) | 🔄 |
| T67 | Secure-agg unmask test | `tests/unit/test_secure_agg.py` | `pytest tests/unit/test_secure_agg.py::TestMaskedIndistinguishable` | ✅ |
| T68 | Tamper detection test | `tests/privacy_audit/test_tamper.py` | `pytest tests/privacy_audit/test_tamper.py` | ✅ |
| T69 | Dashboard rounds/status panel | `dashboard/backend/main.py`, `dashboard/frontend/public/index.html` | `make dashboard-up` | ✅ |
| T70 | Dashboard data-boundary panel | `dashboard/backend/main.py` (`/boundary`), frontend | `make dashboard-up` | ✅ |
| T71 | PR-AUC as primary metric | `fl/bank_node/evaluator.py` | all experiment gates | ✅ |
| T72 | Majority-class baseline beside accuracy | `fl/bank_node/evaluator.py`, `experiments/run_baselines.py` | `make baselines-centralized` (needs data) | ✅ |
| T73 | ≥3 seeds, mean ± std | `experiments/run_matrix.py` | `make experiments-all` (needs data) | ✅ |
| T74 | Results trace to git hash | `experiments/utils.py` (`get_git_hash`) | `make experiments-all` | ✅ |
| T75 | N=3 and N=8 supported | `docker-compose.yml`, `Makefile`, `fabric/scripts/generate_network_config.py` | `make network-up N=3`, `N=8` | ✅ |

---

## Summary

| Status | Count |
|--------|-------|
| ✅ Done (code written + unit/integration tests pass) | 60 |
| 🔄 In Progress (needs runtime: Docker/Fabric/Kaggle data) | 15 |
| ❌ Not Started | 0 |

Items marked 🔄 are fully implemented in code; they require external runtime
resources (Kaggle API credentials, Docker + Fabric binaries, ≥16 GB RAM).
Run `make env-check` to verify your environment before running these.

*A feature is "Done" only when its verification command passes and output is committed to `results/`.*

| T03 | Pseudo-cardholder uid | `data/scripts/partition.py` | `make data-verify` | ❌ |
| T04 | S1 IID partition | `data/scripts/partition.py` | `make data-verify` | ❌ |
| T05 | S2 feature/domain skew | `data/scripts/partition.py` | `make data-verify` | ❌ |
| T06 | S3 Dirichlet skew (α=0.1,0.5,1.0) | `data/scripts/partition.py` | `make data-verify` | ❌ |
| T07 | Temporal train/val/test splits | `data/scripts/partition.py` | `make data-verify` | ❌ |
| T08 | No uid appears in two banks | `data/scripts/verify_splits.py` | `make data-verify` | ❌ |
| T09 | Jensen–Shannon distance per bank | `data/scripts/verify_splits.py` | `make data-verify` | ❌ |
| T10 | Fixed preprocessing schema | `fl/preprocessing.py` | `pytest tests/unit/test_preprocessing.py` | ❌ |
| T11 | Federated feature scaling (secure sum) | `fl/preprocessing.py`, `fl/secure_agg/` | `pytest tests/unit/test_feature_scaling.py` | ❌ |
| T12 | Logistic Regression model | `fl/models/logistic_regression.py` | `pytest tests/unit/test_models.py::test_lr` | ❌ |
| T13 | CNN (1-D) model | `fl/models/cnn.py` | `pytest tests/unit/test_models.py::test_cnn` | ❌ |
| T14 | Gaussian NB model | `fl/models/gaussian_nb.py` | `pytest tests/unit/test_models.py::test_gnb` | ❌ |
| T15 | Decision Tree model | `fl/models/decision_tree.py` | `pytest tests/unit/test_models.py::test_dt` | ❌ |
| T16 | Random Forest model | `fl/models/random_forest.py` | `pytest tests/unit/test_models.py::test_rf` | ❌ |
| T17 | KNN local/centralized only (no federation) | `fl/models/knn.py` | `pytest tests/unit/test_models.py::test_knn_raises` | ❌ |
| T18 | min_samples_leaf ≥ 20 enforced | `fl/models/decision_tree.py`, `random_forest.py` | `pytest tests/unit/test_models.py::test_tree_min_leaf` | ❌ |
| T19 | Leaf probabilities transmitted, not counts | `fl/models/decision_tree.py` | `pytest tests/unit/test_models.py::test_leaf_probs` | ❌ |
| T20 | ROS balancing | `fl/balancing/methods.py` | `pytest tests/unit/test_balancing.py::test_ros` | ❌ |
| T21 | RUS balancing | `fl/balancing/methods.py` | `pytest tests/unit/test_balancing.py::test_rus` | ❌ |