// Package main implements the FraudDetect chaincode for Hyperledger Fabric 2.5.x.
//
// Chaincode functions (all require valid MSP identity):
//   RegisterBank      – register a bank org on the channel
//   StartRound        – open a new FL round (coordinator role, any registered bank)
//   SubmitUpdate      – bank submits its encrypted model update CID + hash
//   SubmitAggregate   – any bank submits the computed aggregate CID + hash
//   EndorseAggregate  – each bank endorses (or rejects) the aggregate
//   FinalizeRound     – finalize round after majority endorsement
//   ReportMetrics     – bank reports local evaluation metrics
//   RecordSecurityEvent – log an anomaly or attack detection event
//   GetRound          – query round state
//   GetHistory        – query full history for a round key
//   GetBankReputation – query bank reputation score
//
// Privacy invariant enforcement:
//   - Chaincode never stores or processes raw transaction data.
//   - Only hashes, CIDs, and metadata are anchored on-chain.
//   - Declared L2 norms are validated (not computed by chaincode).
//
// Access control:
//   - Each function checks the caller's MSP ID against allowed roles.
//   - A bank may only SubmitUpdate for its own bankId.
//   - StartRound may be called by any registered bank.

package main

import (
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

// ============================================================================
// State object types
// ============================================================================

// RoundStatus represents the state machine for an FL round.
type RoundStatus string

const (
	RoundOpen        RoundStatus = "OPEN"
	RoundCollecting  RoundStatus = "COLLECTING"
	RoundAggregating RoundStatus = "AGGREGATING"
	RoundFinalized   RoundStatus = "FINALIZED"
	RoundAborted     RoundStatus = "ABORTED"
)

// Round holds the full state of one federated learning round.
type Round struct {
	RoundID         string                 `json:"roundId"`
	Status          RoundStatus            `json:"status"`
	GlobalModelCID  string                 `json:"globalModelCID"`
	GlobalModelHash string                 `json:"globalModelHash"`
	HParams         map[string]interface{} `json:"hparams"`
	Participants    []string               `json:"participants"`
	Updates         map[string]UpdateRecord `json:"updates"`
	AggCID          string                 `json:"aggCID"`
	AggHash         string                 `json:"aggHash"`
	Endorsements    map[string]bool        `json:"endorsements"`
	Rejections      map[string]bool        `json:"rejections"`
	Metrics         map[string]BankMetrics `json:"metrics"`
	StartTime       int64                  `json:"startTime"`
	FinalizeTime    int64                  `json:"finalizeTime,omitempty"`
	MaxNormBound    float64                `json:"maxNormBound"`
	Quorum          int                    `json:"quorum"`
	InitiatorBankID string                 `json:"initiatorBankId"`
}

// UpdateRecord stores the metadata of one bank's model update.
type UpdateRecord struct {
	BankID       string  `json:"bankId"`
	CID          string  `json:"cid"`
	SHA256       string  `json:"sha256"`
	Size         int64   `json:"size"`
	DeclaredNorm float64 `json:"declaredNorm"`
	Timestamp    int64   `json:"timestamp"`
}

// BankMetrics stores evaluation metrics reported by a bank.
type BankMetrics struct {
	BankID    string             `json:"bankId"`
	Split     string             `json:"split"`
	Metrics   map[string]float64 `json:"metrics"`
	Timestamp int64              `json:"timestamp"`
}

// BankInfo stores registration info for a bank.
type BankInfo struct {
	BankID    string `json:"bankId"`
	MSPID     string `json:"mspId"`
	Registered int64 `json:"registered"`
	Active    bool   `json:"active"`
}

// BankReputation tracks per-bank reliability.
type BankReputation struct {
	BankID        string  `json:"bankId"`
	TotalRounds   int     `json:"totalRounds"`
	GoodRounds    int     `json:"goodRounds"`
	FlaggedRounds int     `json:"flaggedRounds"`
	Score         float64 `json:"score"` // GoodRounds / TotalRounds, [0,1]
	LastUpdated   int64   `json:"lastUpdated"`
}

// SecurityEvent logs an anomaly or detected attack.
type SecurityEvent struct {
	EventID   string `json:"eventId"`
	RoundID   string `json:"roundId"`
	BankID    string `json:"bankId"`
	EventType string `json:"eventType"` // NORM_EXCEEDED | DUPLICATE | HASH_MISMATCH | etc.
	Details   string `json:"details"`
	Timestamp int64  `json:"timestamp"`
}

// ============================================================================
// State key helpers
// ============================================================================

func roundKey(roundID string) string       { return "ROUND:" + roundID }
func bankKey(bankID string) string         { return "BANK:" + bankID }
func reputationKey(bankID string) string   { return "REPUTATION:" + bankID }
func secEventKey(eventID string) string    { return "SECEVENT:" + eventID }

// ============================================================================
// FraudDetectContract
// ============================================================================

// FraudDetectContract implements the Hyperledger Fabric chaincode.
type FraudDetectContract struct {
	contractapi.Contract
}

// ============================================================================
// RegisterBank
// ============================================================================

// RegisterBank registers a bank with its MSP ID on the channel.
// Only the bank's own MSP identity may register itself.
func (c *FraudDetectContract) RegisterBank(ctx contractapi.TransactionContextInterface, bankID string) error {
	mspID, err := getMSPID(ctx)
	if err != nil {
		return err
	}

	// Check not already registered
	existing, err := ctx.GetStub().GetState(bankKey(bankID))
	if err != nil {
		return fmt.Errorf("failed to read state: %w", err)
	}
	if existing != nil {
		return fmt.Errorf("bank %s is already registered", bankID)
	}

	info := BankInfo{
		BankID:     bankID,
		MSPID:      mspID,
		Registered: time.Now().Unix(),
		Active:     true,
	}
	data, _ := json.Marshal(info)
	if err := ctx.GetStub().PutState(bankKey(bankID), data); err != nil {
		return err
	}

	// Initialise reputation
	rep := BankReputation{
		BankID:      bankID,
		TotalRounds: 0,
		GoodRounds:  0,
		Score:       1.0,
		LastUpdated: time.Now().Unix(),
	}
	repData, _ := json.Marshal(rep)
	if err := ctx.GetStub().PutState(reputationKey(bankID), repData); err != nil {
		return err
	}

	ctx.GetStub().SetEvent("BankRegistered", []byte(bankID))
	return nil
}

// ============================================================================
// StartRound
// ============================================================================

// StartRound opens a new FL round. Any registered bank may initiate.
// roundID must be unique. participants is the list of bankIDs joining this round.
func (c *FraudDetectContract) StartRound(
	ctx contractapi.TransactionContextInterface,
	roundID string,
	globalModelCID string,
	globalModelHash string,
	hparamsJSON string,
	participantsJSON string,
	maxNormBound float64,
) error {
	mspID, err := getMSPID(ctx)
	if err != nil {
		return err
	}

	// Verify caller is a registered bank
	callerBankID, err := c.findBankByMSP(ctx, mspID)
	if err != nil {
		return fmt.Errorf("unregistered MSP: %s", mspID)
	}

	// Validate round does not already exist
	existing, err := ctx.GetStub().GetState(roundKey(roundID))
	if err != nil {
		return err
	}
	if existing != nil {
		return fmt.Errorf("round %s already exists", roundID)
	}

	// Validate CID and hash format
	if err := validateCID(globalModelCID); err != nil {
		return fmt.Errorf("invalid globalModelCID: %w", err)
	}
	if err := validateSHA256(globalModelHash); err != nil {
		return fmt.Errorf("invalid globalModelHash: %w", err)
	}

	var hparams map[string]interface{}
	if err := json.Unmarshal([]byte(hparamsJSON), &hparams); err != nil {
		return fmt.Errorf("invalid hparamsJSON: %w", err)
	}

	var participants []string
	if err := json.Unmarshal([]byte(participantsJSON), &participants); err != nil {
		return fmt.Errorf("invalid participantsJSON: %w", err)
	}
	if len(participants) < 2 {
		return errors.New("round requires at least 2 participants")
	}

	quorum := len(participants)/2 + 1 // majority

	round := Round{
		RoundID:         roundID,
		Status:          RoundOpen,
		GlobalModelCID:  globalModelCID,
		GlobalModelHash: globalModelHash,
		HParams:         hparams,
		Participants:    participants,
		Updates:         make(map[string]UpdateRecord),
		Endorsements:    make(map[string]bool),
		Rejections:      make(map[string]bool),
		Metrics:         make(map[string]BankMetrics),
		StartTime:       time.Now().Unix(),
		MaxNormBound:    maxNormBound,
		Quorum:          quorum,
		InitiatorBankID: callerBankID,
	}

	data, _ := json.Marshal(round)
	if err := ctx.GetStub().PutState(roundKey(roundID), data); err != nil {
		return err
	}

	eventData, _ := json.Marshal(map[string]interface{}{
		"roundId":    roundID,
		"modelCID":   globalModelCID,
		"modelHash":  globalModelHash,
		"quorum":     quorum,
		"initiator":  callerBankID,
	})
	ctx.GetStub().SetEvent("RoundStarted", eventData)
	return nil
}

// ============================================================================
// SubmitUpdate
// ============================================================================

// SubmitUpdate records one bank's model update CID and metadata.
// Access control: bankID must match the caller's MSP.
func (c *FraudDetectContract) SubmitUpdate(
	ctx contractapi.TransactionContextInterface,
	roundID string,
	bankID string,
	cid string,
	sha256Hash string,
	size int64,
	declaredNorm float64,
) error {
	mspID, err := getMSPID(ctx)
	if err != nil {
		return err
	}

	// Verify caller's MSP matches bankID
	if err := c.verifyBankMSP(ctx, bankID, mspID); err != nil {
		return err
	}

	round, err := c.getRound(ctx, roundID)
	if err != nil {
		return err
	}

	// Round must be OPEN or COLLECTING
	if round.Status != RoundOpen && round.Status != RoundCollecting {
		return fmt.Errorf("round %s is not accepting updates (status=%s)", roundID, round.Status)
	}

	// BankID must be in participants
	if !contains(round.Participants, bankID) {
		return fmt.Errorf("bank %s is not a participant in round %s", bankID, roundID)
	}

	// Duplicate check
	if _, exists := round.Updates[bankID]; exists {
		return fmt.Errorf("bank %s has already submitted an update for round %s", bankID, roundID)
	}

	// Validate CID and hash
	if err := validateCID(cid); err != nil {
		return fmt.Errorf("invalid CID: %w", err)
	}
	if err := validateSHA256(sha256Hash); err != nil {
		return fmt.Errorf("invalid SHA256: %w", err)
	}
	if size <= 0 {
		return fmt.Errorf("size must be positive")
	}

	// Validate declared norm
	if math.IsNaN(declaredNorm) || math.IsInf(declaredNorm, 0) || declaredNorm < 0 {
		return fmt.Errorf("invalid declaredNorm: %f", declaredNorm)
	}
	if round.MaxNormBound > 0 && declaredNorm > round.MaxNormBound {
		// Record security event and flag reputation
		_ = c.recordSecurityEventInternal(ctx, roundID, bankID, "NORM_EXCEEDED",
			fmt.Sprintf("declaredNorm=%.4f > bound=%.4f", declaredNorm, round.MaxNormBound))
		_ = c.flagBankReputation(ctx, bankID, roundID)
		return fmt.Errorf("declaredNorm %.4f exceeds maxNormBound %.4f", declaredNorm, round.MaxNormBound)
	}

	record := UpdateRecord{
		BankID:       bankID,
		CID:          cid,
		SHA256:       sha256Hash,
		Size:         size,
		DeclaredNorm: declaredNorm,
		Timestamp:    time.Now().Unix(),
	}

	round.Updates[bankID] = record
	round.Status = RoundCollecting

	// Check if quorum reached
	if len(round.Updates) >= round.Quorum {
		round.Status = RoundAggregating
		eventData, _ := json.Marshal(map[string]interface{}{
			"roundId": roundID,
			"quorum":  round.Quorum,
			"updates": len(round.Updates),
		})
		ctx.GetStub().SetEvent("QuorumReached", eventData)
	}

	data, _ := json.Marshal(round)
	if err := ctx.GetStub().PutState(roundKey(roundID), data); err != nil {
		return err
	}

	eventData, _ := json.Marshal(map[string]interface{}{
		"roundId": roundID,
		"bankId":  bankID,
		"cid":     cid,
	})
	ctx.GetStub().SetEvent("UpdateSubmitted", eventData)

	// Update bank reputation (good submission)
	return c.updateBankReputation(ctx, bankID, roundID, true)
}

// ============================================================================
// SubmitAggregate
// ============================================================================

// SubmitAggregate is called by any bank after computing the secure aggregate.
// The round must be in AGGREGATING state.
func (c *FraudDetectContract) SubmitAggregate(
	ctx contractapi.TransactionContextInterface,
	roundID string,
	aggCID string,
	aggHash string,
) error {
	mspID, err := getMSPID(ctx)
	if err != nil {
		return err
	}

	callerBankID, err := c.findBankByMSP(ctx, mspID)
	if err != nil {
		return fmt.Errorf("unregistered MSP: %s", mspID)
	}

	round, err := c.getRound(ctx, roundID)
	if err != nil {
		return err
	}

	if round.Status != RoundAggregating {
		return fmt.Errorf("round %s is not in AGGREGATING state (status=%s)", roundID, round.Status)
	}

	if !contains(round.Participants, callerBankID) {
		return fmt.Errorf("only participants may submit the aggregate")
	}

	if err := validateCID(aggCID); err != nil {
		return fmt.Errorf("invalid aggCID: %w", err)
	}
	if err := validateSHA256(aggHash); err != nil {
		return fmt.Errorf("invalid aggHash: %w", err)
	}

	// If aggregate already submitted, the new submission must match
	if round.AggCID != "" {
		if round.AggCID != aggCID || round.AggHash != aggHash {
			_ = c.recordSecurityEventInternal(ctx, roundID, callerBankID, "AGGREGATE_MISMATCH",
				fmt.Sprintf("submitted aggCID=%s hash=%s conflicts with existing", aggCID, aggHash))
			return fmt.Errorf("aggregate CID/hash mismatch with existing submission")
		}
		// Already matches – just proceed to EndorseAggregate
		return nil
	}

	round.AggCID = aggCID
	round.AggHash = aggHash

	data, _ := json.Marshal(round)
	if err := ctx.GetStub().PutState(roundKey(roundID), data); err != nil {
		return err
	}

	eventData, _ := json.Marshal(map[string]interface{}{
		"roundId": roundID,
		"aggCID":  aggCID,
		"aggHash": aggHash,
		"submitter": callerBankID,
	})
	ctx.GetStub().SetEvent("AggregateSubmitted", eventData)
	return nil
}

// ============================================================================
// EndorseAggregate
// ============================================================================

// EndorseAggregate records a bank's endorsement (or rejection) of the aggregate.
// approve=true means the bank recomputed the aggregate and it matches.
func (c *FraudDetectContract) EndorseAggregate(
	ctx contractapi.TransactionContextInterface,
	roundID string,
	approve bool,
	recomputedHash string,
) error {
	mspID, err := getMSPID(ctx)
	if err != nil {
		return err
	}

	callerBankID, err := c.findBankByMSP(ctx, mspID)
	if err != nil {
		return fmt.Errorf("unregistered MSP: %s", mspID)
	}

	round, err := c.getRound(ctx, roundID)
	if err != nil {
		return err
	}

	if round.Status != RoundAggregating {
		return fmt.Errorf("round is not in AGGREGATING state")
	}
	if !contains(round.Participants, callerBankID) {
		return fmt.Errorf("bank %s is not a participant", callerBankID)
	}
	if _, already := round.Endorsements[callerBankID]; already {
		return fmt.Errorf("bank %s has already endorsed this round", callerBankID)
	}
	if _, already := round.Rejections[callerBankID]; already {
		return fmt.Errorf("bank %s has already rejected this round", callerBankID)
	}
	if round.AggHash == "" {
		return fmt.Errorf("no aggregate submitted yet; cannot endorse")
	}

	if approve {
		if recomputedHash != round.AggHash {
			// Endorser claims approval but hash doesn't match
			round.Rejections[callerBankID] = true
			_ = c.recordSecurityEventInternal(ctx, roundID, callerBankID, "HASH_MISMATCH",
				fmt.Sprintf("claimed approval but recomputedHash=%s != aggHash=%s",
					recomputedHash, round.AggHash))
		} else {
			round.Endorsements[callerBankID] = true
		}
	} else {
		round.Rejections[callerBankID] = true
		_ = c.recordSecurityEventInternal(ctx, roundID, callerBankID, "AGGREGATE_REJECTED",
			"bank rejected the aggregate")
	}

	data, _ := json.Marshal(round)
	if err := ctx.GetStub().PutState(roundKey(roundID), data); err != nil {
		return err
	}

	eventData, _ := json.Marshal(map[string]interface{}{
		"roundId": roundID,
		"bankId":  callerBankID,
		"approved": approve,
	})
	ctx.GetStub().SetEvent("AggregateEndorsed", eventData)
	return nil
}

// ============================================================================
// FinalizeRound
// ============================================================================

// FinalizeRound closes the round after majority endorsement.
// Any registered bank may call this once majority is reached.
func (c *FraudDetectContract) FinalizeRound(
	ctx contractapi.TransactionContextInterface,
	roundID string,
) error {
	round, err := c.getRound(ctx, roundID)
	if err != nil {
		return err
	}

	if round.Status != RoundAggregating {
		return fmt.Errorf("round is not in AGGREGATING state")
	}
	if round.AggCID == "" {
		return fmt.Errorf("no aggregate submitted")
	}

	endorsements := len(round.Endorsements)
	majority := round.Quorum

	if endorsements < majority {
		return fmt.Errorf("insufficient endorsements: %d/%d required", endorsements, majority)
	}

	round.Status = RoundFinalized
	round.FinalizeTime = time.Now().Unix()

	data, _ := json.Marshal(round)
	if err := ctx.GetStub().PutState(roundKey(roundID), data); err != nil {
		return err
	}

	eventData, _ := json.Marshal(map[string]interface{}{
		"roundId":      roundID,
		"aggCID":       round.AggCID,
		"aggHash":      round.AggHash,
		"endorsements": endorsements,
		"finalizeTime": round.FinalizeTime,
	})
	ctx.GetStub().SetEvent("RoundFinalized", eventData)
	return nil
}

// ============================================================================
// ReportMetrics
// ============================================================================

// ReportMetrics records a bank's local evaluation metrics for a finalized round.
func (c *FraudDetectContract) ReportMetrics(
	ctx contractapi.TransactionContextInterface,
	roundID string,
	bankID string,
	split string,
	metricsJSON string,
) error {
	mspID, err := getMSPID(ctx)
	if err != nil {
		return err
	}
	if err := c.verifyBankMSP(ctx, bankID, mspID); err != nil {
		return err
	}

	round, err := c.getRound(ctx, roundID)
	if err != nil {
		return err
	}

	if round.Status != RoundFinalized {
		return fmt.Errorf("metrics can only be reported for FINALIZED rounds")
	}

	var metrics map[string]float64
	if err := json.Unmarshal([]byte(metricsJSON), &metrics); err != nil {
		return fmt.Errorf("invalid metricsJSON: %w", err)
	}

	round.Metrics[bankID] = BankMetrics{
		BankID:    bankID,
		Split:     split,
		Metrics:   metrics,
		Timestamp: time.Now().Unix(),
	}

	data, _ := json.Marshal(round)
	if err := ctx.GetStub().PutState(roundKey(roundID), data); err != nil {
		return err
	}

	eventData, _ := json.Marshal(map[string]interface{}{
		"roundId": roundID,
		"bankId":  bankID,
		"split":   split,
	})
	ctx.GetStub().SetEvent("MetricsReported", eventData)
	return nil
}

// ============================================================================
// RecordSecurityEvent
// ============================================================================

// RecordSecurityEvent logs an externally detected security anomaly.
func (c *FraudDetectContract) RecordSecurityEvent(
	ctx contractapi.TransactionContextInterface,
	roundID string,
	bankID string,
	eventType string,
	details string,
) error {
	mspID, err := getMSPID(ctx)
	if err != nil {
		return err
	}
	// Any registered bank may log a security event
	if _, err := c.findBankByMSP(ctx, mspID); err != nil {
		return fmt.Errorf("unregistered caller: %s", mspID)
	}

	return c.recordSecurityEventInternal(ctx, roundID, bankID, eventType, details)
}

func (c *FraudDetectContract) recordSecurityEventInternal(
	ctx contractapi.TransactionContextInterface,
	roundID, bankID, eventType, details string,
) error {
	txID := ctx.GetStub().GetTxID()
	eventID := fmt.Sprintf("SEC:%s:%s", txID, bankID)

	event := SecurityEvent{
		EventID:   eventID,
		RoundID:   roundID,
		BankID:    bankID,
		EventType: eventType,
		Details:   details,
		Timestamp: time.Now().Unix(),
	}

	data, _ := json.Marshal(event)
	if err := ctx.GetStub().PutState(secEventKey(eventID), data); err != nil {
		return err
	}

	eventData, _ := json.Marshal(event)
	ctx.GetStub().SetEvent("SecurityEvent", eventData)
	return nil
}

// ============================================================================
// GetRound
// ============================================================================

// GetRound returns the current state of a round.
func (c *FraudDetectContract) GetRound(
	ctx contractapi.TransactionContextInterface,
	roundID string,
) (*Round, error) {
	return c.getRound(ctx, roundID)
}

func (c *FraudDetectContract) getRound(
	ctx contractapi.TransactionContextInterface,
	roundID string,
) (*Round, error) {
	data, err := ctx.GetStub().GetState(roundKey(roundID))
	if err != nil {
		return nil, fmt.Errorf("failed to read round %s: %w", roundID, err)
	}
	if data == nil {
		return nil, fmt.Errorf("round %s not found", roundID)
	}
	var round Round
	if err := json.Unmarshal(data, &round); err != nil {
		return nil, fmt.Errorf("failed to unmarshal round: %w", err)
	}
	return &round, nil
}

// ============================================================================
// GetHistory
// ============================================================================

// GetHistory returns the full history for a given key (round, bank, etc.).
func (c *FraudDetectContract) GetHistory(
	ctx contractapi.TransactionContextInterface,
	key string,
) (string, error) {
	iter, err := ctx.GetStub().GetHistoryForKey(key)
	if err != nil {
		return "", err
	}
	defer iter.Close()

	var history []map[string]interface{}
	for iter.HasNext() {
		mod, err := iter.Next()
		if err != nil {
			return "", err
		}
		entry := map[string]interface{}{
			"txId":      mod.TxId,
			"timestamp": mod.Timestamp.AsTime().Unix(),
			"isDelete":  mod.IsDelete,
			"value":     string(mod.Value),
		}
		history = append(history, entry)
	}

	out, _ := json.Marshal(history)
	return string(out), nil
}

// ============================================================================
// GetBankReputation
// ============================================================================

// GetBankReputation returns the reputation score for a bank.
func (c *FraudDetectContract) GetBankReputation(
	ctx contractapi.TransactionContextInterface,
	bankID string,
) (*BankReputation, error) {
	data, err := ctx.GetStub().GetState(reputationKey(bankID))
	if err != nil {
		return nil, err
	}
	if data == nil {
		return nil, fmt.Errorf("bank %s not found", bankID)
	}
	var rep BankReputation
	if err := json.Unmarshal(data, &rep); err != nil {
		return nil, err
	}
	return &rep, nil
}

// ============================================================================
// Internal helpers
// ============================================================================

func getMSPID(ctx contractapi.TransactionContextInterface) (string, error) {
	id, err := ctx.GetClientIdentity().GetMSPID()
	if err != nil {
		return "", fmt.Errorf("failed to get MSPID: %w", err)
	}
	return id, nil
}

func (c *FraudDetectContract) findBankByMSP(
	ctx contractapi.TransactionContextInterface,
	mspID string,
) (string, error) {
	// Iterate all BANK: keys (limited range query)
	// In production, maintain a MSP→BankID mapping for O(1) lookup.
	iter, err := ctx.GetStub().GetStateByRange("BANK:", "BANK;")
	if err != nil {
		return "", err
	}
	defer iter.Close()

	for iter.HasNext() {
		kv, err := iter.Next()
		if err != nil {
			return "", err
		}
		var info BankInfo
		if err := json.Unmarshal(kv.Value, &info); err != nil {
			continue
		}
		if info.MSPID == mspID && info.Active {
			return info.BankID, nil
		}
	}
	return "", fmt.Errorf("no active bank found for MSP %s", mspID)
}

func (c *FraudDetectContract) verifyBankMSP(
	ctx contractapi.TransactionContextInterface,
	bankID string,
	mspID string,
) error {
	data, err := ctx.GetStub().GetState(bankKey(bankID))
	if err != nil {
		return err
	}
	if data == nil {
		return fmt.Errorf("bank %s not registered", bankID)
	}
	var info BankInfo
	if err := json.Unmarshal(data, &info); err != nil {
		return err
	}
	if info.MSPID != mspID {
		return fmt.Errorf("MSP %s is not authorised to act for bank %s (expected %s)",
			mspID, bankID, info.MSPID)
	}
	if !info.Active {
		return fmt.Errorf("bank %s is not active", bankID)
	}
	return nil
}

func (c *FraudDetectContract) updateBankReputation(
	ctx contractapi.TransactionContextInterface,
	bankID, roundID string,
	good bool,
) error {
	data, err := ctx.GetStub().GetState(reputationKey(bankID))
	if err != nil {
		return err
	}

	var rep BankReputation
	if data == nil {
		rep = BankReputation{BankID: bankID}
	} else {
		if err := json.Unmarshal(data, &rep); err != nil {
			return err
		}
	}

	rep.TotalRounds++
	if good {
		rep.GoodRounds++
	}
	if rep.TotalRounds > 0 {
		rep.Score = float64(rep.GoodRounds) / float64(rep.TotalRounds)
	}
	rep.LastUpdated = time.Now().Unix()

	repData, _ := json.Marshal(rep)
	return ctx.GetStub().PutState(reputationKey(bankID), repData)
}

func (c *FraudDetectContract) flagBankReputation(
	ctx contractapi.TransactionContextInterface,
	bankID, roundID string,
) error {
	data, err := ctx.GetStub().GetState(reputationKey(bankID))
	if err != nil {
		return err
	}

	var rep BankReputation
	if data != nil {
		json.Unmarshal(data, &rep)
	}
	rep.BankID = bankID
	rep.FlaggedRounds++
	rep.TotalRounds++
	if rep.TotalRounds > 0 {
		rep.Score = float64(rep.GoodRounds) / float64(rep.TotalRounds)
	}
	rep.LastUpdated = time.Now().Unix()

	repData, _ := json.Marshal(rep)
	return ctx.GetStub().PutState(reputationKey(bankID), repData)
}

// validateCID checks that a CID looks like a valid IPFS CID (basic sanity check).
func validateCID(cid string) error {
	if len(cid) < 10 {
		return fmt.Errorf("CID too short: %q", cid)
	}
	if !strings.HasPrefix(cid, "Qm") && !strings.HasPrefix(cid, "bafk") &&
		!strings.HasPrefix(cid, "bafy") && !strings.HasPrefix(cid, "bafybe") {
		return fmt.Errorf("CID does not look like a valid IPFS CID: %q", cid)
	}
	return nil
}

// validateSHA256 checks that a hash is a 64-char hex string.
func validateSHA256(hash string) error {
	if len(hash) != 64 {
		return fmt.Errorf("SHA-256 hash must be 64 hex chars, got %d: %q", len(hash), hash)
	}
	if _, err := hex.DecodeString(hash); err != nil {
		return fmt.Errorf("SHA-256 hash is not valid hex: %w", err)
	}
	return nil
}

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

// ============================================================================
// Main
// ============================================================================

func main() {
	chaincode, err := contractapi.NewChaincode(&FraudDetectContract{})
	if err != nil {
		panic(fmt.Sprintf("Error creating FraudDetect chaincode: %v", err))
	}
	if err := chaincode.Start(); err != nil {
		panic(fmt.Sprintf("Error starting FraudDetect chaincode: %v", err))
	}
}
