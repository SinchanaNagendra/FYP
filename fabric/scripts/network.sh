#!/bin/bash
# fabric/scripts/network.sh
# =========================
# Bring up / down / reset the Hyperledger Fabric network.
#
# Usage:
#   bash fabric/scripts/network.sh up [N_BANKS]
#   bash fabric/scripts/network.sh down
#   bash fabric/scripts/network.sh reset
#
# Requirements:
#   - Docker + Docker Compose v2
#   - Hyperledger Fabric binaries (cryptogen, configtxgen) on PATH
#   - FABRIC_CFG_PATH set to fabric/network/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FABRIC_DIR="${PROJECT_ROOT}/fabric"
NETWORK_DIR="${FABRIC_DIR}/network"
CRYPTO_DIR="${FABRIC_DIR}/crypto-config"
CHANNEL_ARTIFACTS="${FABRIC_DIR}/channel-artifacts"
CHANNEL_NAME="fraudchannel"

N_BANKS="${2:-5}"

export FABRIC_CFG_PATH="${NETWORK_DIR}"
export COMPOSE_PROJECT_NAME="fraudfl"

print_header() {
    echo ""
    echo "================================================================"
    echo "  Hyperledger Fabric Network – $1"
    echo "  N_BANKS=${N_BANKS}, CHANNEL=${CHANNEL_NAME}"
    echo "================================================================"
    echo ""
}

check_binaries() {
    for bin in cryptogen configtxgen docker; do
        if ! command -v ${bin} &> /dev/null; then
            echo "[ERROR] ${bin} not found on PATH."
            if [[ "${bin}" == "cryptogen" || "${bin}" == "configtxgen" ]]; then
                echo "  Download Fabric binaries: https://hyperledger-fabric.readthedocs.io/en/latest/install.html"
                echo "  Or: curl -sSL https://bit.ly/2ysbOFE | bash -s -- 2.5.9 1.5.12"
            fi
            exit 1
        fi
    done
}

generate_crypto() {
    echo "[crypto] Generating MSP material for ${N_BANKS} banks..."
    mkdir -p "${CRYPTO_DIR}" "${CHANNEL_ARTIFACTS}"

    # Generate crypto-config.yaml for N banks
    python3 "${SCRIPT_DIR}/generate_network_config.py" \
        --n-banks "${N_BANKS}" \
        --output "${NETWORK_DIR}"

    cryptogen generate \
        --config="${NETWORK_DIR}/crypto-config.yaml" \
        --output="${CRYPTO_DIR}"

    echo "[crypto] Done."
}

generate_genesis() {
    echo "[configtx] Generating genesis block..."
    configtxgen \
        -profile FraudOrdererGenesis \
        -channelID system-channel \
        -outputBlock "${CHANNEL_ARTIFACTS}/genesis.block"

    echo "[configtx] Generating channel transaction..."
    configtxgen \
        -profile FraudChannel \
        -channelID "${CHANNEL_NAME}" \
        -outputCreateChannelTx "${CHANNEL_ARTIFACTS}/${CHANNEL_NAME}.tx"

    echo "[configtx] Generating anchor peer transactions..."
    for i in $(seq 0 $((N_BANKS - 1))); do
        configtxgen \
            -profile FraudChannel \
            -channelID "${CHANNEL_NAME}" \
            -outputAnchorPeersUpdate "${CHANNEL_ARTIFACTS}/Bank${i}MSPanchors.tx" \
            -asOrg "Bank${i}MSP"
    done
    echo "[configtx] Done."
}

generate_swarm_key() {
    if [[ ! -f "${SCRIPT_DIR}/swarm.key" ]]; then
        echo "[ipfs] Generating IPFS swarm key..."
        # Generate a random swarm key for the private IPFS network
        printf '/key/swarm/psk/1.0.0/\n/base16/\n' > "${SCRIPT_DIR}/swarm.key"
        python3 -c "import secrets; print(secrets.token_hex(32))" >> "${SCRIPT_DIR}/swarm.key"
        echo "[ipfs] swarm.key generated at ${SCRIPT_DIR}/swarm.key"
    else
        echo "[ipfs] swarm.key already exists, skipping."
    fi
}

network_up() {
    print_header "STARTING"
    check_binaries
    generate_crypto
    generate_genesis
    generate_swarm_key

    echo "[network] Starting containers with N=${N_BANKS} banks..."

    # Generate docker-compose for N banks if not the default 5
    if [[ "${N_BANKS}" != "5" ]]; then
        python3 "${SCRIPT_DIR}/generate_compose.py" --n-banks "${N_BANKS}" \
            --output "${PROJECT_ROOT}/docker-compose.generated.yml"
        COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.generated.yml"
    else
        COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.yml"
    fi

    docker compose -f "${COMPOSE_FILE}" up -d \
        orderer0 orderer1 orderer2 \
        $(for i in $(seq 0 $((N_BANKS-1))); do echo "peer0.bank${i}.fraud.net"; done)

    sleep 5
    echo "[network] Waiting for peers to be ready..."
    for i in $(seq 0 $((N_BANKS-1))); do
        docker exec peer0.bank${i}.fraud.net peer version || true
    done

    echo "[channel] Creating and joining channel ${CHANNEL_NAME}..."
    bash "${SCRIPT_DIR}/create_channel.sh" "${N_BANKS}" "${CHANNEL_NAME}"

    echo ""
    echo "[network] ✅ Network is UP with ${N_BANKS} banks."
    echo "  Channel: ${CHANNEL_NAME}"
    echo "  Orderers: orderer0:7050, orderer1:8050, orderer2:9050"
    for i in $(seq 0 $((N_BANKS-1))); do
        PORT=$((7051 + i * 1000))
        echo "  Bank${i}: peer0.bank${i}.fraud.net:${PORT}"
    done
}

network_down() {
    print_header "STOPPING"
    docker compose -p "${COMPOSE_PROJECT_NAME}" stop
    echo "[network] Network stopped."
}

network_reset() {
    print_header "RESETTING"
    docker compose -p "${COMPOSE_PROJECT_NAME}" down --volumes --remove-orphans
    rm -rf "${CRYPTO_DIR}" "${CHANNEL_ARTIFACTS}"
    echo "[network] Network reset. All volumes and crypto material removed."
    echo "  Run 'make network-up' to recreate."
}

case "${1:-up}" in
    up)     network_up ;;
    down)   network_down ;;
    reset)  network_reset ;;
    *)
        echo "Usage: $0 {up|down|reset} [N_BANKS]"
        exit 1
        ;;
esac
