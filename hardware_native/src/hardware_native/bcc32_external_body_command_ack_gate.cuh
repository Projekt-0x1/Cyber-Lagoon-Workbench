#pragma once

#include "bcc32_external_body_command_ack_wire.cuh"
#include "bcc32_external_body_command_outbox.cuh"

#include <cstdint>

#if defined(__CUDACC__)
#define BCC32_COMMAND_ACK_HD __host__ __device__
#else
#define BCC32_COMMAND_ACK_HD
#endif

namespace bcc32 {

struct ExternalBodyCommandAckReceipt {
    bool complete = false;
    bool transport_and_declared_ack_only = true;
    bool device_identity_authenticated = false;
    bool physical_application_proven = false;
    bool declared_applied = false;
    ExternalBodyCommandAckStatus status = ExternalBodyCommandAckStatus::kRejected;
    std::uint64_t device_instance = 0;
    std::uint64_t session_epoch = 0;
    std::uint64_t route_digest = 0;
    std::uint64_t route_sequence = 0;
    std::uint64_t outbox_ticket = 0;
    std::uint64_t command_frame_digest = 0;
    std::uint64_t attempt_id = 0;
    std::uint64_t action_envelope = 0;
    std::uint64_t command_digest = 0;
    std::uint64_t device_tick = 0;
    std::uint64_t device_key_fingerprint = 0;
    std::uint64_t signature_commitment = 0;
    std::uint64_t device_result_code = 0;
    std::uint64_t structural_commitment = 0;
};

class ExternalBodyCommandAckGate {
public:
    BCC32_COMMAND_ACK_HD bool open(
        std::uint64_t device_instance,
        std::uint64_t session_epoch,
        std::uint64_t route_digest,
        const ExternalBodyTransportWriteReceipt& transport) {
        if (pending_ || device_instance == 0 || session_epoch == 0 ||
            route_digest == 0 || !transport.complete ||
            !transport.transport_written_only || transport.device_applied ||
            transport.outbox_ticket == 0 || transport.route_sequence == 0 ||
            transport.attempt_id == 0 || transport.action_envelope == 0 ||
            transport.command_digest == 0 || transport.frame_digest == 0 ||
            transport.route_sequence <= last_route_sequence_ ||
            transport.attempt_id <= last_attempt_id_) {
            return false;
        }
        device_instance_ = device_instance;
        session_epoch_ = session_epoch;
        route_digest_ = route_digest;
        transport_ = transport;
        pending_ = true;
        return true;
    }

    BCC32_COMMAND_ACK_HD bool accept(
        const ExternalBodyCommandAckEvent& ack,
        ExternalBodyCommandAckReceipt* receipt) {
        const bool known_status =
            ack.status == ExternalBodyCommandAckStatus::kApplied ||
            ack.status == ExternalBodyCommandAckStatus::kRejected;
        const bool result_matches_status =
            ack.status == ExternalBodyCommandAckStatus::kApplied
                ? ack.device_result_code == 0
                : ack.device_result_code != 0;
        if (!pending_ || receipt == nullptr || !known_status ||
            !result_matches_status || ack.device_key_fingerprint == 0 ||
            ack.signature_commitment == 0 ||
            ack.device_instance != device_instance_ ||
            ack.session_epoch != session_epoch_ ||
            ack.route_digest != route_digest_ ||
            ack.route_sequence != transport_.route_sequence ||
            ack.outbox_ticket != transport_.outbox_ticket ||
            ack.command_frame_digest != transport_.frame_digest ||
            ack.attempt_id != transport_.attempt_id ||
            ack.action_envelope != transport_.action_envelope ||
            ack.command_digest != transport_.command_digest ||
            ack.device_tick == 0 || ack.device_tick <= last_device_tick_) {
            return false;
        }
        ExternalBodyCommandAckReceipt result{};
        result.complete = true;
        result.declared_applied =
            ack.status == ExternalBodyCommandAckStatus::kApplied;
        result.status = ack.status;
        result.device_instance = ack.device_instance;
        result.session_epoch = ack.session_epoch;
        result.route_digest = ack.route_digest;
        result.route_sequence = ack.route_sequence;
        result.outbox_ticket = ack.outbox_ticket;
        result.command_frame_digest = ack.command_frame_digest;
        result.attempt_id = ack.attempt_id;
        result.action_envelope = ack.action_envelope;
        result.command_digest = ack.command_digest;
        result.device_tick = ack.device_tick;
        result.device_key_fingerprint = ack.device_key_fingerprint;
        result.signature_commitment = ack.signature_commitment;
        result.device_result_code = ack.device_result_code;
        result.structural_commitment = commitment(result);
        *receipt = result;
        last_route_sequence_ = ack.route_sequence;
        last_attempt_id_ = ack.attempt_id;
        last_device_tick_ = ack.device_tick;
        transport_ = ExternalBodyTransportWriteReceipt{};
        pending_ = false;
        return true;
    }

    BCC32_COMMAND_ACK_HD bool pending() const { return pending_; }
    BCC32_COMMAND_ACK_HD std::uint64_t last_device_tick() const {
        return last_device_tick_;
    }

private:
    BCC32_COMMAND_ACK_HD static std::uint64_t mix(std::uint64_t state,
                                                  std::uint64_t value) {
        state ^= value + 0x9e3779b97f4a7c15ULL + (state << 6U) + (state >> 2U);
        return state;
    }

    BCC32_COMMAND_ACK_HD static std::uint64_t commitment(
        const ExternalBodyCommandAckReceipt& receipt) {
        std::uint64_t state = 0x434f4d4d41434b31ULL;
        const std::uint64_t fields[] = {
            receipt.device_instance, receipt.session_epoch, receipt.route_digest,
            receipt.route_sequence, receipt.outbox_ticket,
            receipt.command_frame_digest, receipt.attempt_id,
            receipt.action_envelope, receipt.command_digest, receipt.device_tick,
            static_cast<std::uint64_t>(receipt.status),
            receipt.device_key_fingerprint, receipt.signature_commitment,
            receipt.device_result_code,
        };
        for (const std::uint64_t field : fields) {
            state = mix(state, field);
        }
        return state == 0 ? 1 : state;
    }

    ExternalBodyTransportWriteReceipt transport_{};
    std::uint64_t device_instance_ = 0;
    std::uint64_t session_epoch_ = 0;
    std::uint64_t route_digest_ = 0;
    std::uint64_t last_route_sequence_ = 0;
    std::uint64_t last_attempt_id_ = 0;
    std::uint64_t last_device_tick_ = 0;
    bool pending_ = false;
};

}  // namespace bcc32

#undef BCC32_COMMAND_ACK_HD
