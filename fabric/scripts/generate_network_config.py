"""
fabric/scripts/generate_network_config.py
==========================================
Generates crypto-config.yaml and configtx.yaml for N banks.

Supports N = 3, 5, 8 (or any value ≥ 2).

Usage:
    python fabric/scripts/generate_network_config.py --n-banks 5 --output fabric/network/
"""

import argparse
from pathlib import Path


def generate_crypto_config(n_banks: int) -> str:
    lines = ["OrdererOrgs:"]
    lines += [
        "  - Name: Orderer",
        "    Domain: orderer.fraud.net",
        "    EnableNodeOUs: true",
        "    Specs:",
        "      - Hostname: orderer0",
        "      - Hostname: orderer1",
        "      - Hostname: orderer2",
        "",
        "PeerOrgs:",
    ]
    for i in range(n_banks):
        lines += [
            f"  - Name: Bank{i}",
            f"    Domain: bank{i}.fraud.net",
            "    EnableNodeOUs: true",
            "    Template:",
            "      Count: 1",
            "    Users:",
            "      Count: 1",
        ]
    return "\n".join(lines)


def generate_configtx(n_banks: int) -> str:
    # Organizations section
    orgs = ["Organizations:"]
    orgs += [
        "  - &OrdererOrg",
        "    Name: OrdererMSP",
        "    ID: OrdererMSP",
        "    MSPDir: ../crypto-config/ordererOrganizations/orderer.fraud.net/msp",
        "    Policies:",
        "      Readers:",
        "        Type: Signature",
        "        Rule: \"OR('OrdererMSP.member')\"",
        "      Writers:",
        "        Type: Signature",
        "        Rule: \"OR('OrdererMSP.member')\"",
        "      Admins:",
        "        Type: Signature",
        "        Rule: \"OR('OrdererMSP.admin')\"",
    ]
    for i in range(n_banks):
        anchor_port = 7051 + i * 1000
        orgs += [
            f"  - &Bank{i}Org",
            f"    Name: Bank{i}MSP",
            f"    ID: Bank{i}MSP",
            f"    MSPDir: ../crypto-config/peerOrganizations/bank{i}.fraud.net/msp",
            "    Policies:",
            "      Readers:",
            "        Type: Signature",
            f"        Rule: \"OR('Bank{i}MSP.admin', 'Bank{i}MSP.peer', 'Bank{i}MSP.client')\"",
            "      Writers:",
            "        Type: Signature",
            f"        Rule: \"OR('Bank{i}MSP.admin', 'Bank{i}MSP.client')\"",
            "      Admins:",
            "        Type: Signature",
            f"        Rule: \"OR('Bank{i}MSP.admin')\"",
            "      Endorsement:",
            "        Type: Signature",
            f"        Rule: \"OR('Bank{i}MSP.peer')\"",
            "    AnchorPeers:",
            f"      - Host: peer0.bank{i}.fraud.net",
            f"        Port: {anchor_port}",
        ]

    # Capabilities
    caps = """
Capabilities:
  Channel: &ChannelCapabilities
    V2_0: true
  Orderer: &OrdererCapabilities
    V2_0: true
  Application: &ApplicationCapabilities
    V2_5: true
"""

    # Application section with majority endorsement policy
    member_rules = " ".join([f"'Bank{i}MSP.peer'" for i in range(n_banks)])
    # Majority = floor(n_banks/2) + 1
    majority = n_banks // 2 + 1

    app_section = f"""
Application: &ApplicationDefaults
  Organizations:
  Policies:
    Readers:
      Type: ImplicitMeta
      Rule: "ANY Readers"
    Writers:
      Type: ImplicitMeta
      Rule: "ANY Writers"
    Admins:
      Type: ImplicitMeta
      Rule: "MAJORITY Admins"
    LifecycleEndorsement:
      Type: ImplicitMeta
      Rule: "MAJORITY Endorsement"
    Endorsement:
      Type: ImplicitMeta
      Rule: "MAJORITY Endorsement"
  Capabilities:
    <<: *ApplicationCapabilities
"""

    # Orderer section
    orderer_section = """
Orderer: &OrdererDefaults
  OrdererType: etcdraft
  Addresses:
    - orderer0.orderer.fraud.net:7050
    - orderer1.orderer.fraud.net:8050
    - orderer2.orderer.fraud.net:9050
  EtcdRaft:
    Consenters:
      - Host: orderer0.orderer.fraud.net
        Port: 7050
        ClientTLSCert: ../crypto-config/ordererOrganizations/orderer.fraud.net/orderers/orderer0.orderer.fraud.net/tls/server.crt
        ServerTLSCert: ../crypto-config/ordererOrganizations/orderer.fraud.net/orderers/orderer0.orderer.fraud.net/tls/server.crt
      - Host: orderer1.orderer.fraud.net
        Port: 8050
        ClientTLSCert: ../crypto-config/ordererOrganizations/orderer.fraud.net/orderers/orderer1.orderer.fraud.net/tls/server.crt
        ServerTLSCert: ../crypto-config/ordererOrganizations/orderer.fraud.net/orderers/orderer1.orderer.fraud.net/tls/server.crt
      - Host: orderer2.orderer.fraud.net
        Port: 9050
        ClientTLSCert: ../crypto-config/ordererOrganizations/orderer.fraud.net/orderers/orderer2.orderer.fraud.net/tls/server.crt
        ServerTLSCert: ../crypto-config/ordererOrganizations/orderer.fraud.net/orderers/orderer2.orderer.fraud.net/tls/server.crt
  BatchTimeout: 2s
  BatchSize:
    MaxMessageCount: 10
    AbsoluteMaxBytes: 99 MB
    PreferredMaxBytes: 512 KB
  Organizations:
  Policies:
    Readers:
      Type: ImplicitMeta
      Rule: "ANY Readers"
    Writers:
      Type: ImplicitMeta
      Rule: "ANY Writers"
    Admins:
      Type: ImplicitMeta
      Rule: "MAJORITY Admins"
    BlockValidation:
      Type: ImplicitMeta
      Rule: "ANY Writers"
  Capabilities:
    <<: *OrdererCapabilities
"""

    # Channel section
    channel_section = """
Channel: &ChannelDefaults
  Policies:
    Readers:
      Type: ImplicitMeta
      Rule: "ANY Readers"
    Writers:
      Type: ImplicitMeta
      Rule: "ANY Writers"
    Admins:
      Type: ImplicitMeta
      Rule: "MAJORITY Admins"
  Capabilities:
    <<: *ChannelCapabilities
"""

    # Profiles
    bank_org_refs = "\n".join(
        [f"          - *Bank{i}Org" for i in range(n_banks)]
    )

    profiles = f"""
Profiles:
  FraudOrdererGenesis:
    <<: *ChannelDefaults
    Orderer:
      <<: *OrdererDefaults
      Organizations:
        - *OrdererOrg
    Consortiums:
      FraudConsortium:
        Organizations:
{bank_org_refs}

  FraudChannel:
    Consortium: FraudConsortium
    <<: *ChannelDefaults
    Application:
      <<: *ApplicationDefaults
      Organizations:
{bank_org_refs}
"""

    return "\n".join(orgs) + caps + app_section + orderer_section + channel_section + profiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-banks", type=int, default=5)
    parser.add_argument("--output", default="fabric/network")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Write crypto-config.yaml
    crypto_path = out / "crypto-config.yaml"
    crypto_path.write_text(generate_crypto_config(args.n_banks))
    print(f"[generate] Written {crypto_path}")

    # Write configtx.yaml
    configtx_path = out / "configtx.yaml"
    configtx_path.write_text(generate_configtx(args.n_banks))
    print(f"[generate] Written {configtx_path}")

    print(f"[generate] Network config generated for {args.n_banks} banks.")


if __name__ == "__main__":
    main()
