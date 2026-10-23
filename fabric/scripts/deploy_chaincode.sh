#!/bin/bash
# fabric/scripts/deploy_chaincode.sh
# ====================================
# Package, install, approve, and commit the frauddetect chaincode.
#
# Usage: bash fabric/scripts/deploy_chaincode.sh [N_BANKS]
#
# Requires: peer CLI binary on PATH, FABRIC_CFG_PATH set, network up.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CHAINCODE_DIR="${PROJECT_ROOT}/fabric/chaincode/frauddetect"
CHANNEL_NAME="fraudchannel"
CC_NAME="frauddetect"
CC_VERSION="1.0"
CC_SEQUENCE="1"
N_BANKS="${1:-5}"

export FABRIC_CFG_PATH="${PROJECT_ROOT}/fabric/network"
export CORE_PEER_TLS_ENABLED=true

echo ""
echo "================================================================"
echo "  Deploying chaincode: ${CC_NAME} v${CC_VERSION}"
echo "  Channel: ${CHANNEL_NAME}, N_BANKS: ${N_BANKS}"
echo "================================================================"
echo ""

# ------------------------------------------------------------------
# 1. Package chaincode
# ------------------------------------------------------------------
echo "[1/5] Packaging chaincode..."
cd "${CHAINCODE_DIR}"
GO111MODULE=on go mod vendor
cd "${PROJECT_ROOT}"

peer lifecycle chaincode package "${CC_NAME}.tar.gz" \
    --path "${CHAINCODE_DIR}" \
    --lang golang \
    --label "${CC_NAME}_${CC_VERSION}"

echo "[1/5] Packaged: ${CC_NAME}.tar.gz"

# ------------------------------------------------------------------
# 2. Install on all bank peers
# ------------------------------------------------------------------
for i in $(seq 0 $((N_BANKS - 1))); do
    PORT=$((7051 + i * 1000))
    echo "[2/5] Installing on bank${i} (peer0.bank${i}.fraud.net:${PORT})..."

    export CORE_PEER_LOCALMSPID="Bank${i}MSP"
    export CORE_PEER_ADDRESS="peer0.bank${i}.fraud.net:${PORT}"
    export CORE_PEER_TLS_ROOTCERT_FILE="${PROJECT_ROOT}/fabric/crypto-config/peerOrganizations/bank${i}.fraud.net/peers/peer0.bank${i}.fraud.net/tls/ca.crt"
    export CORE_PEER_MSPCONFIGPATH="${PROJECT_ROOT}/fabric/crypto-config/peerOrganizations/bank${i}.fraud.net/users/Admin@bank${i}.fraud.net/msp"

    peer lifecycle chaincode install "${CC_NAME}.tar.gz"
    echo "  [OK] Installed on bank${i}"
done

# ------------------------------------------------------------------
# 3. Get package ID
# ------------------------------------------------------------------
echo "[3/5] Getting package ID..."
export CORE_PEER_LOCALMSPID="Bank0MSP"
export CORE_PEER_ADDRESS="peer0.bank0.fraud.net:7051"
export CORE_PEER_TLS_ROOTCERT_FILE="${PROJECT_ROOT}/fabric/crypto-config/peerOrganizations/bank0.fraud.net/peers/peer0.bank0.fraud.net/tls/ca.crt"
export CORE_PEER_MSPCONFIGPATH="${PROJECT_ROOT}/fabric/crypto-config/peerOrganizations/bank0.fraud.net/users/Admin@bank0.fraud.net/msp"

PACKAGE_ID=$(peer lifecycle chaincode queryinstalled | \
    grep "${CC_NAME}_${CC_VERSION}" | \
    sed -n 's/.*Package ID: \(.*\), Label.*/\1/p')
echo "  Package ID: ${PACKAGE_ID}"

# ------------------------------------------------------------------
# 4. Approve for all orgs
# ------------------------------------------------------------------
for i in $(seq 0 $((N_BANKS - 1))); do
    PORT=$((7051 + i * 1000))
    echo "[4/5] Approving for Bank${i}MSP..."

    export CORE_PEER_LOCALMSPID="Bank${i}MSP"
    export CORE_PEER_ADDRESS="peer0.bank${i}.fraud.net:${PORT}"
    export CORE_PEER_TLS_ROOTCERT_FILE="${PROJECT_ROOT}/fabric/crypto-config/peerOrganizations/bank${i}.fraud.net/peers/peer0.bank${i}.fraud.net/tls/ca.crt"
    export CORE_PEER_MSPCONFIGPATH="${PROJECT_ROOT}/fabric/crypto-config/peerOrganizations/bank${i}.fraud.net/users/Admin@bank${i}.fraud.net/msp"

    peer lifecycle chaincode approveformyorg \
        -o "orderer0.orderer.fraud.net:7050" \
        --ordererTLSHostnameOverride orderer0.orderer.fraud.net \
        --tls \
        --cafile "${PROJECT_ROOT}/fabric/crypto-config/ordererOrganizations/orderer.fraud.net/orderers/orderer0.orderer.fraud.net/tls/ca.crt" \
        --channelID "${CHANNEL_NAME}" \
        --name "${CC_NAME}" \
        --version "${CC_VERSION}" \
        --package-id "${PACKAGE_ID}" \
        --sequence "${CC_SEQUENCE}"

    echo "  [OK] Approved for Bank${i}MSP"
done

# ------------------------------------------------------------------
# 5. Commit chaincode
# ------------------------------------------------------------------
echo "[5/5] Committing chaincode..."
export CORE_PEER_LOCALMSPID="Bank0MSP"
export CORE_PEER_ADDRESS="peer0.bank0.fraud.net:7051"
export CORE_PEER_TLS_ROOTCERT_FILE="${PROJECT_ROOT}/fabric/crypto-config/peerOrganizations/bank0.fraud.net/peers/peer0.bank0.fraud.net/tls/ca.crt"
export CORE_PEER_MSPCONFIGPATH="${PROJECT_ROOT}/fabric/crypto-config/peerOrganizations/bank0.fraud.net/users/Admin@bank0.fraud.net/msp"

# Build peer address args for all banks
PEER_ARGS=""
for i in $(seq 0 $((N_BANKS - 1))); do
    PORT=$((7051 + i * 1000))
    TLS_CERT="${PROJECT_ROOT}/fabric/crypto-config/peerOrganizations/bank${i}.fraud.net/peers/peer0.bank${i}.fraud.net/tls/ca.crt"
    PEER_ARGS="${PEER_ARGS} --peerAddresses peer0.bank${i}.fraud.net:${PORT} --tlsRootCertFiles ${TLS_CERT}"
done

peer lifecycle chaincode commit \
    -o "orderer0.orderer.fraud.net:7050" \
    --ordererTLSHostnameOverride orderer0.orderer.fraud.net \
    --tls \
    --cafile "${PROJECT_ROOT}/fabric/crypto-config/ordererOrganizations/orderer.fraud.net/orderers/orderer0.orderer.fraud.net/tls/ca.crt" \
    --channelID "${CHANNEL_NAME}" \
    --name "${CC_NAME}" \
    --version "${CC_VERSION}" \
    --sequence "${CC_SEQUENCE}" \
    ${PEER_ARGS}

echo ""
echo "✅ Chaincode '${CC_NAME}' deployed successfully on channel '${CHANNEL_NAME}'"
echo "   Run 'make chaincode-test' to verify with unit tests."
