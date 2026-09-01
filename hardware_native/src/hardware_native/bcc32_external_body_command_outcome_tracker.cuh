#pragma once

#include "bcc32_external_body_command_ack_verifier.hpp"
#include "bcc32_external_body_heartbeat_verifier.hpp"
#include "bcc32_external_body_command_outbox.cuh"

#include <cstdint>

#if defined(__CUDACC__)
#define BCC32_COMMAND_OUTCOME_HD __host__ __device__
#else
#define BCC32_COMMAND_OUTCOME_HD
#endif

namespace bcc32 {

enum class ExternalBodyCommandOutcome : std::uint8_t {
    kNone = 0,
    kApplied = 1,
    kRejected = 2,
    kExpired = 3,
};

struct ExternalBodyCommandOutcomeReceipt {
    bool complete = false;
    bool capture_may_open = false;
    bool physical_application_proven = false;
    ExternalBodyCommandOutcome outcome = ExternalBodyCommandOutcome::kNone;
    std::uint64_t device_instance = 0;
    std::uint64_t session_epoch = 0;
    std::uint64_t route_digest = 0;
    std::uint64_t device_key_fingerprint = 0;
    std::uint64_t route_sequence = 0;
    std::uint64_t outbox_ticket = 0;
    std::uint64_t attempt_id = 0;
    std::uint64_t action_envelope = 0;
    std::uint64_t command_digest = 0;
    std::uint64_t deadline_device_tick = 0;
    std::uint64_t terminal_device_tick = 0;
    std::uint64_t structural_commitment = 0;
};

class ExternalBodyCommandOutcomeTracker {
public:
    BCC32_COMMAND_OUTCOME_HD bool open(
        std::uint64_t device_instance,
        std::uint64_t session_epoch,
        std::uint64_t route_digest,
        std::uint64_t device_key_fingerprint,
        std::uint64_t deadline_device_tick,
        const ExternalBodyTransportWriteReceipt& transport) {
        if (pending_ || device_instance == 0 || session_epoch == 0 ||
            route_digest == 0 || device_key_fingerprint == 0 ||
            deadline_device_tick == 0 ||
            !transport.complete || !transport.transport_written_only ||
            transport.device_applied || transport.outbox_ticket == 0 ||
            transport.route_sequence <= last_route_sequence_ ||
            transport.attempt_id <= last_attempt_id_ ||
            transport.action_envelope == 0 || transport.command_digest == 0 ||
            transport.frame_digest == 0 ||
            deadline_device_tick <= last_device_tick_) {
            return false;
        }
        device_instance_ = device_instance;
        session_epoch_ = session_epoch;
        route_digest_ = route_digest;
        device_key_fingerprint_ = device_key_fingerprint;
        deadline_device_tick_ = deadline_device_tick;
        transport_ = transport;
        pending_ = true;
        return true;
    }

    BCC32_COMMAND_OUTCOME_HD bool observe_verified_ack(
        const ExternalBodyAckVerificationReceipt& verified,
        ExternalBodyCommandOutcomeReceipt* receipt) {
        if (receipt == nullptr) {
            return false;
        }
        *receipt = ExternalBodyCommandOutcomeReceipt{};
        const ExternalBodyCommandAckReceipt& ack = verified.command_ack;
        if (!pending_ || !verified.complete || !verified.signature_verified ||
            verified.physical_application_proven ||
            verified.device_key_fingerprint != device_key_fingerprint_ ||
            verified.signature_commitment == 0 || !ack.complete ||
            !ack.transport_and_declared_ack_only ||
            ack.physical_application_proven || ack.structural_commitment == 0 ||
            ack.device_key_fingerprint != verified.device_key_fingerprint ||
            ack.signature_commitment != verified.signature_commitment ||
            !matches_ack(ack) || ack.device_tick <= last_device_tick_ ||
            ack.device_tick > deadline_device_tick_) {
            return false;
        }
        const ExternalBodyCommandOutcome outcome =
            ack.status == ExternalBodyCommandAckStatus::kApplied
                ? ExternalBodyCommandOutcome::kApplied
                : ack.status == ExternalBodyCommandAckStatus::kRejected
                      ? ExternalBodyCommandOutcome::kRejected
                      : ExternalBodyCommandOutcome::kNone;
        if (outcome == ExternalBodyCommandOutcome::kNone) {
            return false;
        }
        finish(outcome, ack.device_tick, receipt);
        return true;
    }

    BCC32_COMMAND_OUTCOME_HD bool observe_verified_heartbeat(
        const ExternalBodyHeartbeatVerificationReceipt& verified,
        ExternalBodyCommandOutcomeReceipt* receipt) {
        if (receipt == nullptr) {
            return false;
        }
        *receipt = ExternalBodyCommandOutcomeReceipt{};
        const ExternalBodyHeartbeatEvent& heartbeat = verified.heartbeat;
        if (!pending_ || !verified.complete || !verified.signature_verified ||
            verified.challenge_nonce == 0 ||
            verified.signature_commitment == 0 ||
            heartbeat.device_instance != device_instance_ ||
            heartbeat.session_epoch != session_epoch_ ||
            heartbeat.route_digest != route_digest_ ||
            heartbeat.device_key_fingerprint != device_key_fingerprint_ ||
            heartbeat.challenge_nonce != verified.challenge_nonce ||
            heartbeat.signature_commitment != verified.signature_commitment ||
            heartbeat.device_tick <= last_device_tick_ ||
            heartbeat.heartbeat_sequence <= last_heartbeat_sequence_) {
            return false;
        }
        last_device_tick_ = heartbeat.device_tick;
        last_heartbeat_sequence_ = heartbeat.heartbeat_sequence;
        if (heartbeat.device_tick <= deadline_device_tick_) {
            return false;
        }
        finish(ExternalBodyCommandOutcome::kExpired,
               heartbeat.device_tick, receipt);
        return true;
    }

    BCC32_COMMAND_OUTCOME_HD bool pending() const { return pending_; }
    BCC32_COMMAND_OUTCOME_HD std::uint64_t last_device_tick() const {
        return last_device_tick_;
    }

private:
    BCC32_COMMAND_OUTCOME_HD bool matches_ack(
        const ExternalBodyCommandAckReceipt& ack) const {
        return ack.device_instance == device_instance_ &&
               ack.session_epoch == session_epoch_ &&
               ack.route_digest == route_digest_ &&
               ack.route_sequence == transport_.route_sequence &&
               ack.outbox_ticket == transport_.outbox_ticket &&
               ack.command_frame_digest == transport_.frame_digest &&
               ack.attempt_id == transport_.attempt_id &&
               ack.action_envelope == transport_.action_envelope &&
               ack.command_digest == transport_.command_digest;
    }

    BCC32_COMMAND_OUTCOME_HD static std::uint64_t mix(std::uint64_t state,
                                                      std::uint64_t value) {
        state ^= value + 0x9e3779b97f4a7c15ULL + (state << 6U) + (state >> 2U);
        return state;
    }

    BCC32_COMMAND_OUTCOME_HD void finish(
        ExternalBodyCommandOutcome outcome,
        std::uint64_t terminal_device_tick,
        ExternalBodyCommandOutcomeReceipt* receipt) {
        ExternalBodyCommandOutcomeReceipt result{};
        result.complete = true;
        result.capture_may_open = outcome == ExternalBodyCommandOutcome::kApplied;
        result.outcome = outcome;
        result.device_instance = device_instance_;
        result.session_epoch = session_epoch_;
        result.route_digest = route_digest_;
        result.device_key_fingerprint = device_key_fingerprint_;
        result.route_sequence = transport_.route_sequence;
        result.outbox_ticket = transport_.outbox_ticket;
        result.attempt_id = transport_.attempt_id;
        result.action_envelope = transport_.action_envelope;
        result.command_digest = transport_.command_digest;
        result.deadline_device_tick = deadline_device_tick_;
        result.terminal_device_tick = terminal_device_tick;
        std::uint64_t state = 0x4f5554434f4d4531ULL;
        const std::uint64_t fields[] = {
            static_cast<std::uint64_t>(outcome), device_instance_, session_epoch_,
            route_digest_, device_key_fingerprint_, transport_.route_sequence,
            transport_.outbox_ticket,
            transport_.attempt_id, transport_.action_envelope,
            transport_.command_digest, deadline_device_tick_, terminal_device_tick,
        };
        for (const std::uint64_t field : fields) {
            state = mix(state, field);
        }
        result.structural_commitment = state == 0 ? 1 : state;
        *receipt = result;
        last_route_sequence_ = transport_.route_sequence;
        last_attempt_id_ = transport_.attempt_id;
        if (terminal_device_tick > last_device_tick_) {
            last_device_tick_ = terminal_device_tick;
        }
        transport_ = ExternalBodyTransportWriteReceipt{};
        deadline_device_tick_ = 0;
        pending_ = false;
    }

    ExternalBodyTransportWriteReceipt transport_{};
    std::uint64_t device_instance_ = 0;
    std::uint64_t session_epoch_ = 0;
    std::uint64_t route_digest_ = 0;
    std::uint64_t device_key_fingerprint_ = 0;
    std::uint64_t deadline_device_tick_ = 0;
    std::uint64_t last_route_sequence_ = 0;
    std::uint64_t last_attempt_id_ = 0;
    std::uint64_t last_device_tick_ = 0;
    std::uint64_t last_heartbeat_sequence_ = 0;
    bool pending_ = false;
};

}  // namespace bcc32

#undef BCC32_COMMAND_OUTCOME_HD
