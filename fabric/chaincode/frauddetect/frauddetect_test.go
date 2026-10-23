// frauddetect_test.go
// Unit tests for the FraudDetect chaincode using mock stubs.
//
// Run with: go test ./... -v

package main

import (
	"encoding/json"
	"testing"

	"github.com/hyperledger/fabric-contract-api-go/contractapi"
	"github.com/hyperledger/fabric-chaincode-go/shim"
	"github.com/hyperledger/fabric-chaincode-go/shimtest"
)

// ============================================================================
// Test helpers
// ============================================================================

// newTestStub creates a mock chaincode stub.
func newTestStub(t *testing.T) *shimtest.MockStub {
	t.Helper()
	cc, err := contractapi.NewChaincode(&FraudDetectContract{})
	if err != nil {
		t.Fatalf("failed to create chaincode: %v", err)
	}
	stub := shimtest.NewMockStub("frauddetect", cc)
	return stub
}

// invoke is a helper that calls the chaincode and checks for success.
func invoke(t *testing.T, stub *shimtest.MockStub, fn string, args ...string) {
	t.Helper()
	allArgs := append([]string{fn}, args...)
	byteArgs := make([][]byte, len(allArgs))
	for i, a := range allArgs {
		byteArgs[i] = []byte(a)
	}
	res := stub.MockInvoke("tx1", byteArgs)
	if res.Status != shim.OK {
		t.Fatalf("[%s] expected OK, got status=%d message=%s", fn, res.Status, res.Message)
	}
}

// invokeExpectError calls the chaincode and asserts an error is returned.
func invokeExpectError(t *testing.T, stub *shimtest.MockStub, fn string, args ...string) {
	t.Helper()
	allArgs := append([]string{fn}, args...)
	byteArgs := make([][]byte, len(allArgs))
	for i, a := range allArgs {
		byteArgs[i] = []byte(a)
	}
	res := stub.MockInvoke("tx2", byteArgs)
	if res.Status == shim.OK {
		t.Fatalf("[%s] expected error, but got OK", fn)
	}
}

// ============================================================================
// Test: RegisterBank
// ============================================================================

func TestRegisterBank(t *testing.T) {
	stub := newTestStub(t)
	stub.Creator = []byte("Bank0MSP") // simulate MSP identity

	invoke(t, stub, "RegisterBank", "bank0")

	// Verify state written
	data := stub.State[bankKey("bank0")]
	if data == nil {
		t.Fatal("bank0 state not written")
	}
	var info BankInfo
	if err := json.Unmarshal(data, &info); err != nil {
		t.Fatal(err)
	}
	if info.BankID != "bank0" {
		t.Errorf("expected bankId=bank0, got %s", info.BankID)
	}
	if !info.Active {
		t.Error("bank should be active")
	}
}

func TestRegisterBank_Duplicate(t *testing.T) {
	stub := newTestStub(t)
	stub.Creator = []byte("Bank0MSP")
	invoke(t, stub, "RegisterBank", "bank0")
	invokeExpectError(t, stub, "RegisterBank", "bank0")
}

// ============================================================================
// Test: StartRound
// ============================================================================

func TestStartRound(t *testing.T) {
	stub := newTestStub(t)
	stub.Creator = []byte("Bank0MSP")
	invoke(t, stub, "RegisterBank", "bank0")
	invoke(t, stub, "RegisterBank", "bank1")

	hparams := `{"lr": 0.01, "epochs": 3}`
	participants := `["bank0", "bank1"]`

	invoke(t, stub, "StartRound",
		"round1",
		"QmTestCID1234567890abcdef", // fake CID (starts with Qm)
		"a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3", // 64 hex
		hparams,
		participants,
		"10.0", // maxNormBound
	)

	data := stub.State[roundKey("round1")]
	if data == nil {
		t.Fatal("round1 state not written")
	}
	var round Round
	if err := json.Unmarshal(data, &round); err != nil {
		t.Fatal(err)
	}
	if round.Status != RoundOpen {
		t.Errorf("expected status=OPEN, got %s", round.Status)
	}
	if round.Quorum != 2 {
		t.Errorf("expected quorum=2, got %d", round.Quorum)
	}
}

func TestStartRound_DuplicateID(t *testing.T) {
	stub := newTestStub(t)
	stub.Creator = []byte("Bank0MSP")
	invoke(t, stub, "RegisterBank", "bank0")
	invoke(t, stub, "RegisterBank", "bank1")
	invoke(t, stub, "StartRound", "round1",
		"QmTestCID", "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
		`{"lr":0.01}`, `["bank0","bank1"]`, "10.0")
	invokeExpectError(t, stub, "StartRound", "round1",
		"QmTestCID", "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
		`{"lr":0.01}`, `["bank0","bank1"]`, "10.0")
}

func TestStartRound_InvalidCID(t *testing.T) {
	stub := newTestStub(t)
	stub.Creator = []byte("Bank0MSP")
	invoke(t, stub, "RegisterBank", "bank0")
	invoke(t, stub, "RegisterBank", "bank1")
	invokeExpectError(t, stub, "StartRound", "round1",
		"BAD_CID", "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
		`{}`, `["bank0","bank1"]`, "10.0")
}

// ============================================================================
// Test: SubmitUpdate
// ============================================================================

func TestSubmitUpdate(t *testing.T) {
	stub := newTestStub(t)
	stub.Creator = []byte("Bank0MSP")
	invoke(t, stub, "RegisterBank", "bank0")
	stub.Creator = []byte("Bank1MSP")
	invoke(t, stub, "RegisterBank", "bank1")

	// Start round as bank0
	stub.Creator = []byte("Bank0MSP")
	invoke(t, stub, "StartRound", "round1",
		"QmTestGlobalModel", "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
		`{}`, `["bank0","bank1"]`, "100.0")

	// bank0 submits update
	invoke(t, stub, "SubmitUpdate", "round1", "bank0",
		"QmBank0UpdateCID", "b94f5a5b37a54890e7c7b8e5d57f6c2d3e4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c",
		"102400", "5.5")

	data := stub.State[roundKey("round1")]
	var round Round
	json.Unmarshal(data, &round)
	if _, ok := round.Updates["bank0"]; !ok {
		t.Fatal("bank0 update not recorded")
	}
	if round.Status != RoundCollecting {
		t.Errorf("expected COLLECTING, got %s", round.Status)
	}
}

func TestSubmitUpdate_NormExceeded(t *testing.T) {
	stub := newTestStub(t)
	stub.Creator = []byte("Bank0MSP")
	invoke(t, stub, "RegisterBank", "bank0")
	stub.Creator = []byte("Bank1MSP")
	invoke(t, stub, "RegisterBank", "bank1")

	stub.Creator = []byte("Bank0MSP")
	invoke(t, stub, "StartRound", "round1",
		"QmTestGlobalModel", "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
		`{}`, `["bank0","bank1"]`, "10.0") // max norm = 10

	// norm 999 > 10 → should fail
	invokeExpectError(t, stub, "SubmitUpdate", "round1", "bank0",
		"QmBank0UpdateCID", "b94f5a5b37a54890e7c7b8e5d57f6c2d3e4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c",
		"102400", "999.0")
}

func TestSubmitUpdate_DuplicateRejected(t *testing.T) {
	stub := newTestStub(t)
	stub.Creator = []byte("Bank0MSP")
	invoke(t, stub, "RegisterBank", "bank0")
	stub.Creator = []byte("Bank1MSP")
	invoke(t, stub, "RegisterBank", "bank1")

	stub.Creator = []byte("Bank0MSP")
	invoke(t, stub, "StartRound", "round1",
		"QmGlobal", "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
		`{}`, `["bank0","bank1"]`, "100.0")
	invoke(t, stub, "SubmitUpdate", "round1", "bank0",
		"QmUpdate1", "b94f5a5b37a54890e7c7b8e5d57f6c2d3e4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c",
		"102400", "5.0")
	// Second submission from same bank → error
	invokeExpectError(t, stub, "SubmitUpdate", "round1", "bank0",
		"QmUpdate2", "b94f5a5b37a54890e7c7b8e5d57f6c2d3e4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c",
		"102400", "5.0")
}

// ============================================================================
// Test: Full round state machine
// ============================================================================

func TestFullRoundStateMachine(t *testing.T) {
	stub := newTestStub(t)

	// Register banks
	stub.Creator = []byte("Bank0MSP")
	invoke(t, stub, "RegisterBank", "bank0")
	stub.Creator = []byte("Bank1MSP")
	invoke(t, stub, "RegisterBank", "bank1")

	modelHash := "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3"
	updateHash := "b94f5a5b37a54890e7c7b8e5d57f6c2d3e4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c"
	aggHash := "c12345678901234567890123456789012345678901234567890123456789abcd"

	// StartRound
	stub.Creator = []byte("Bank0MSP")
	invoke(t, stub, "StartRound", "r1", "QmGlobal", modelHash, `{}`, `["bank0","bank1"]`, "100.0")

	// Both banks submit updates → quorum reached → AGGREGATING
	stub.Creator = []byte("Bank0MSP")
	invoke(t, stub, "SubmitUpdate", "r1", "bank0", "QmUpd0", updateHash, "1000", "3.0")
	stub.Creator = []byte("Bank1MSP")
	invoke(t, stub, "SubmitUpdate", "r1", "bank1", "QmUpd1", updateHash, "1000", "4.0")

	data := stub.State[roundKey("r1")]
	var round Round
	json.Unmarshal(data, &round)
	if round.Status != RoundAggregating {
		t.Fatalf("expected AGGREGATING after quorum, got %s", round.Status)
	}

	// SubmitAggregate
	stub.Creator = []byte("Bank0MSP")
	invoke(t, stub, "SubmitAggregate", "r1", "QmAgg123", aggHash)

	// Both banks endorse
	stub.Creator = []byte("Bank0MSP")
	invoke(t, stub, "EndorseAggregate", "r1", "true", aggHash)
	stub.Creator = []byte("Bank1MSP")
	invoke(t, stub, "EndorseAggregate", "r1", "true", aggHash)

	// FinalizeRound
	stub.Creator = []byte("Bank0MSP")
	invoke(t, stub, "FinalizeRound", "r1")

	data = stub.State[roundKey("r1")]
	json.Unmarshal(data, &round)
	if round.Status != RoundFinalized {
		t.Fatalf("expected FINALIZED, got %s", round.Status)
	}
	if round.FinalizeTime == 0 {
		t.Error("FinalizeTime not set")
	}

	// ReportMetrics
	metrics := `{"pr_auc": 0.85, "roc_auc": 0.92, "f1": 0.73}`
	stub.Creator = []byte("Bank0MSP")
	invoke(t, stub, "ReportMetrics", "r1", "bank0", "val", metrics)

	// GetBankReputation
	data = stub.State[reputationKey("bank0")]
	var rep BankReputation
	json.Unmarshal(data, &rep)
	if rep.TotalRounds < 1 {
		t.Error("reputation not updated")
	}
}

// ============================================================================
// Test: GetBankReputation
// ============================================================================

func TestGetBankReputation(t *testing.T) {
	stub := newTestStub(t)
	stub.Creator = []byte("Bank0MSP")
	invoke(t, stub, "RegisterBank", "bank0")

	data := stub.State[reputationKey("bank0")]
	if data == nil {
		t.Fatal("reputation not initialised on RegisterBank")
	}
	var rep BankReputation
	json.Unmarshal(data, &rep)
	if rep.Score != 1.0 {
		t.Errorf("initial score should be 1.0, got %f", rep.Score)
	}
}

// ============================================================================
// Test: ValidateCID and ValidateSHA256
// ============================================================================

func TestValidateCID(t *testing.T) {
	valid := []string{"QmAbcdef1234567890", "bafk2bzaced", "bafybeigdyr7"}
	for _, cid := range valid {
		if err := validateCID(cid); err != nil {
			t.Errorf("expected valid CID %q to pass: %v", cid, err)
		}
	}

	invalid := []string{"", "SHORT", "http://bad"}
	for _, cid := range invalid {
		if err := validateCID(cid); err == nil {
			t.Errorf("expected invalid CID %q to fail", cid)
		}
	}
}

func TestValidateSHA256(t *testing.T) {
	valid := "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3"
	if err := validateSHA256(valid); err != nil {
		t.Errorf("expected valid hash to pass: %v", err)
	}

	invalid := []string{
		"tooshort",
		"a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae",  // 63 chars
		"a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3x", // 65 chars
		"ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ",    // not hex
	}
	for _, h := range invalid {
		if err := validateSHA256(h); err == nil {
			t.Errorf("expected invalid hash %q to fail", h)
		}
	}
}
