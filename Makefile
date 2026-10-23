# =============================================================================
# Blockchain-Secured Federated Learning – Multi-Bank Fraud Detection
# Master Makefile
# =============================================================================
SHELL := /bin/bash
N ?= 5          # number of banks; override with N=3 or N=8
SEED ?= 42
SCENARIO ?= S1  # S1 | S2 | S3

.PHONY: help env-check \
        data-download data-partition data-verify \
        baselines-centralized baselines-local \
        fl-train \
        network-up network-down network-reset chaincode-deploy chaincode-test \
        fl-fabric-run \
        secagg-test dp-train privacy-audit \
        attacks-run \
        tff-compare \
        dashboard-up dashboard-down \
        experiments-all \
        demo \
        results-clean

help:
	@echo "=== Blockchain-FL Fraud Detection – Makefile targets ==="
	@echo ""
	@echo "  make env-check            Verify all dependencies are present"
	@echo ""
	@echo "  PHASE 1 – DATA"
	@echo "  make data-download        Download IEEE-CIS and ULB datasets via Kaggle API"
	@echo "  make data-partition       Partition into N=$(N) banks, scenarios S1/S2/S3"
	@echo "  make data-verify          Run gate checks: stats table + leak-free splits"
	@echo ""
	@echo "  PHASE 2 – BASELINES"
	@echo "  make baselines-centralized  Train all 6 models x balancing x IID/non-IID centrally"
	@echo "  make baselines-local        Train per-bank local models"
	@echo ""
	@echo "  PHASE 3 – FEDERATED LEARNING (no blockchain)"
	@echo "  make fl-train N=$(N) SCENARIO=$(SCENARIO)"
	@echo ""
	@echo "  PHASE 4 – FABRIC NETWORK"
	@echo "  make network-up N=$(N)    Bring up Fabric network (N orgs, Raft x3)"
	@echo "  make network-down         Tear down network"
	@echo "  make network-reset        Tear down + clean volumes"
	@echo "  make chaincode-deploy     Deploy frauddetect chaincode"
	@echo "  make chaincode-test       Run Go unit tests for chaincode"
	@echo ""
	@echo "  PHASE 5 – INTEGRATED FL + FABRIC + IPFS"
	@echo "  make fl-fabric-run N=$(N) SCENARIO=$(SCENARIO)"
	@echo ""
	@echo "  PHASE 6 – SECURE AGG + DP"
	@echo "  make secagg-test          Unit tests for secure aggregation"
	@echo "  make dp-train             Train with DP-SGD; output privacy budget curve"
	@echo "  make privacy-audit        Run all 5 privacy audit tests"
	@echo ""
	@echo "  PHASE 7 – ATTACKS"
	@echo "  make attacks-run          Full attack/defence matrix"
	@echo ""
	@echo "  PHASE 8 – TFF COMPARISON"
	@echo "  make tff-compare          Run TFF LR+CNN in Docker, output comparison table"
	@echo ""
	@echo "  PHASE 9 – DASHBOARD"
	@echo "  make dashboard-up         Start FastAPI + React dashboard"
	@echo "  make dashboard-down       Stop dashboard"
	@echo ""
	@echo "  PHASE 10 – FULL MATRIX"
	@echo "  make experiments-all      Full experiment matrix (all seeds, scenarios, N)"
	@echo ""
	@echo "  DEMO"
	@echo "  make demo                 Full end-to-end demo (network up + 5 banks + dashboard)"
	@echo ""
	@echo "  UTIL"
	@echo "  make results-clean        Remove generated results (keeps git-tracked files)"

# ---------------------------------------------------------------------------
# Environment check
# ---------------------------------------------------------------------------
env-check:
	@echo "[env-check] Checking dependencies..."
	@python --version 2>&1 | grep -E "3\.(10|11)" || (echo "ERROR: Python 3.10/3.11 required" && exit 1)
	@docker --version || (echo "ERROR: Docker required" && exit 1)
	@docker compose version || (echo "ERROR: Docker Compose v2 required" && exit 1)
	@go version 2>/dev/null || echo "WARNING: Go not found – chaincode won't compile locally"
	@kaggle --version 2>/dev/null || echo "WARNING: kaggle CLI not found – run: pip install kaggle"
	@python -c "import torch" 2>/dev/null || echo "WARNING: torch not installed – run: pip install -r fl/requirements.txt"
	@echo "[env-check] Done."

# ---------------------------------------------------------------------------
# Phase 1 – Data
# ---------------------------------------------------------------------------
data-download:
	@echo "[data] Downloading datasets..."
	python data/scripts/download.py --output data/raw

data-partition:
	@echo "[data] Partitioning into $(N) banks..."
	python data/scripts/partition.py \
		--n-banks $(N) \
		--seed $(SEED) \
		--output data/partitions

data-verify:
	@echo "[data] Running Phase 1 gate verification..."
	python data/scripts/verify_splits.py \
		--partitions data/partitions \
		--n-banks $(N)

# ---------------------------------------------------------------------------
# Phase 2 – Baselines
# ---------------------------------------------------------------------------
baselines-centralized:
	python experiments/run_baselines.py \
		--mode centralized \
		--n-banks $(N) \
		--seed $(SEED) \
		--output results/baselines

baselines-local:
	python experiments/run_baselines.py \
		--mode local \
		--n-banks $(N) \
		--seed $(SEED) \
		--output results/baselines

# ---------------------------------------------------------------------------
# Phase 3 – Federated learning (no blockchain)
# ---------------------------------------------------------------------------
fl-train:
	python experiments/run_fl.py \
		--n-banks $(N) \
		--scenario $(SCENARIO) \
		--seed $(SEED) \
		--no-blockchain \
		--output results/fl

# ---------------------------------------------------------------------------
# Phase 4 – Fabric network
# ---------------------------------------------------------------------------
network-up:
	@echo "[fabric] Starting $(N)-org Fabric network..."
	bash fabric/scripts/network.sh up $(N)

network-down:
	@echo "[fabric] Stopping Fabric network..."
	bash fabric/scripts/network.sh down

network-reset:
	@echo "[fabric] Resetting Fabric network (removes all volumes)..."
	bash fabric/scripts/network.sh reset

chaincode-deploy:
	bash fabric/scripts/deploy_chaincode.sh $(N)

chaincode-test:
	cd fabric/chaincode && go test ./... -v

# ---------------------------------------------------------------------------
# Phase 5 – Integrated FL + Fabric + IPFS
# ---------------------------------------------------------------------------
fl-fabric-run:
	python experiments/run_fl.py \
		--n-banks $(N) \
		--scenario $(SCENARIO) \
		--seed $(SEED) \
		--with-blockchain \
		--rounds 20 \
		--output results/fl_fabric

# ---------------------------------------------------------------------------
# Phase 6 – Secure aggregation + DP
# ---------------------------------------------------------------------------
secagg-test:
	python -m pytest tests/unit/test_secure_agg.py -v

dp-train:
	python experiments/run_dp_sweep.py \
		--n-banks $(N) \
		--seed $(SEED) \
		--output results/dp_sweep

privacy-audit:
	python -m pytest tests/privacy_audit/ -v --tb=short

# ---------------------------------------------------------------------------
# Phase 7 – Attacks
# ---------------------------------------------------------------------------
attacks-run:
	python experiments/run_attacks.py \
		--n-banks $(N) \
		--seed $(SEED) \
		--output results/attacks

# ---------------------------------------------------------------------------
# Phase 8 – TFF comparison
# ---------------------------------------------------------------------------
tff-compare:
	docker compose -f fl_tff/docker-compose.tff.yml run --rm tff-runner \
		python run_tff.py \
		--n-banks $(N) \
		--seed $(SEED) \
		--output /results/tff
	@echo "[tff] Comparison table written to results/tff/"

# ---------------------------------------------------------------------------
# Phase 9 – Dashboard
# ---------------------------------------------------------------------------
dashboard-up:
	docker compose up -d dashboard-api dashboard-ui

dashboard-down:
	docker compose stop dashboard-api dashboard-ui

# ---------------------------------------------------------------------------
# Phase 10 – Full matrix
# ---------------------------------------------------------------------------
experiments-all:
	python experiments/run_matrix.py \
		--seeds 42 123 777 \
		--n-banks 3 5 8 \
		--scenarios S1 S2 S3 \
		--output results/full_matrix

# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
demo: network-up chaincode-deploy
	@echo "[demo] Starting 5-bank federated round with dashboard..."
	python experiments/run_fl.py \
		--n-banks 5 \
		--scenario S1 \
		--seed 42 \
		--with-blockchain \
		--rounds 5 \
		--output results/demo &
	$(MAKE) dashboard-up
	@echo "[demo] Dashboard available at http://localhost:3000"
	@echo "[demo] Fabric explorer at http://localhost:8080"

# ---------------------------------------------------------------------------
# Util
# ---------------------------------------------------------------------------
results-clean:
	find results/ -type f ! -name ".gitkeep" -delete
	@echo "[util] results/ cleared (git-tracked stubs preserved)"
