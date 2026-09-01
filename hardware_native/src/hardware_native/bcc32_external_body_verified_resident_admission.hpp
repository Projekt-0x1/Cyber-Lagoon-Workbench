#pragma once

#include "bcc32_external_body_heartbeat_verifier.hpp"
#include "bcc32_external_body_resident_ticket_gate.cuh"
#include "bcc32_external_body_verified_capture_assembly.hpp"

#include <cstddef>
#include <cstdint>

namespace bcc32 {

enum class ExternalBodyVerifiedResidentAdmissionFailure : std::uint8_t {
    kNone,
    kAlreadyConsumed,
    kTicketNotConsumed,
    kSessionNotCommitted,
    kCaptureNotVerified,
    kSessionMismatch,
    kActionMismatch,
    kAttemptMismatch,
    kCaptureMismatch,
    kPayloadDigestMismatch,
    kHeartbeatNotBound,
    kHeartbeatSessionMismatch,
    kHeartbeatTickMismatch,
    kParentRouteMismatch,
};

struct ExternalBodyVerifiedResidentAdmissionReceipt {
    bool ticket_bound = false;
    bool route_bound = false;
    bool body_heartbeat_bound = false;
    bool parent_route_bound = false;
    std::uint64_t parent_route_digest = 0;
    std::uint64_t attempt_id = 0;
    std::uint64_t capture_id = 0;
    std::uint64_t raw_payload_digest = 0;
    std::size_t raw_payload_bytes = 0;
    bool physical_source_proven = false;
    bool physical_consequence_proven = false;
    bool resident_assimilation_proven = false;
    ExternalBodyVerifiedResidentAdmissionFailure failure =
        ExternalBodyVerifiedResidentAdmissionFailure::kNone;
};

template <std::size_t MaxBytes = 256>
class ExternalBodyVerifiedResidentAdmission final {
public:
    bool admit(const ExternalBodyResidentTicketReceipt& ticket,
               const ExternalBodySessionReceipt& session,
               const ExternalBodyVerifiedRawCapture<MaxBytes>& capture,
               ExternalBodyVerifiedResidentAdmissionReceipt* receipt) {
        if (receipt == nullptr) return false;
        *receipt = {};
        return fail(receipt,
                    ExternalBodyVerifiedResidentAdmissionFailure::kHeartbeatNotBound);
    }

    bool admit(const ExternalBodyResidentTicketReceipt& ticket,
               const ExternalBodySessionReceipt& session,
               const ExternalBodyHeartbeatVerificationReceipt& heartbeat,
               const ExternalBodyVerifiedRawCapture<MaxBytes>& capture,
               ExternalBodyVerifiedResidentAdmissionReceipt* receipt) {
        if (receipt == nullptr) return false;
        *receipt = {};
        if (consumed_) return fail(receipt, ExternalBodyVerifiedResidentAdmissionFailure::kAlreadyConsumed);
        if (!ticket.consumed || ticket.failure != ExternalBodyResidentTicketFailure::kNone ||
            ticket.attempt_id == 0 || ticket.capture_id == 0 || ticket.raw_payload_digest == 0) {
            return fail(receipt, ExternalBodyVerifiedResidentAdmissionFailure::kTicketNotConsumed);
        }
        if (ticket.parent_route_digest == 0) {
            return fail(receipt,
                        ExternalBodyVerifiedResidentAdmissionFailure::kHeartbeatNotBound);
        }
        if (!session.opened || !session.capture_committed ||
            session.failure != ExternalBodySessionFailure::kNone ||
            session.accepted_attempt_id != ticket.attempt_id ||
            session.accepted_action_envelope != ticket.action_envelope ||
            session.accepted_command_digest != ticket.command_digest ||
            session.captured_id != ticket.capture_id ||
            session.captured_raw_digest != ticket.raw_payload_digest) {
            return fail(receipt, ExternalBodyVerifiedResidentAdmissionFailure::kSessionNotCommitted);
        }
        if (!capture.signature_verified || capture.capture_id == 0 ||
            capture.raw_payload_digest == 0 || capture.size == 0 ||
            capture.physical_source_proven || capture.physical_consequence_proven) {
            return fail(receipt, ExternalBodyVerifiedResidentAdmissionFailure::kCaptureNotVerified);
        }
        if (capture.device_instance != session.device_instance ||
            capture.session_epoch != session.session_epoch ||
            capture.route_digest != session.route_digest) {
            return fail(receipt, ExternalBodyVerifiedResidentAdmissionFailure::kSessionMismatch);
        }
        if (capture.action_envelope != ticket.action_envelope ||
            capture.command_digest != ticket.command_digest ||
            capture.capture_device_tick != session.last_device_tick) {
            return fail(receipt, ExternalBodyVerifiedResidentAdmissionFailure::kActionMismatch);
        }
        if (capture.attempt_id != ticket.attempt_id) {
            return fail(receipt, ExternalBodyVerifiedResidentAdmissionFailure::kAttemptMismatch);
        }
        if (capture.capture_id != ticket.capture_id) {
            return fail(receipt, ExternalBodyVerifiedResidentAdmissionFailure::kCaptureMismatch);
        }
        if (capture.raw_payload_digest != ticket.raw_payload_digest) {
            return fail(receipt, ExternalBodyVerifiedResidentAdmissionFailure::kPayloadDigestMismatch);
        }
        const auto& heartbeat_event = heartbeat.heartbeat;
        if (!heartbeat.complete || !heartbeat.signature_verified ||
            heartbeat_event.device_instance != session.device_instance ||
            heartbeat_event.session_epoch != session.session_epoch ||
            heartbeat_event.route_digest != session.route_digest ||
            heartbeat_event.heartbeat_sequence == 0 ||
            heartbeat_event.device_tick == 0 ||
            heartbeat_event.parent_route_digest == 0) {
            return fail(receipt,
                        ExternalBodyVerifiedResidentAdmissionFailure::kHeartbeatNotBound);
        }
        if (heartbeat_event.device_instance != capture.device_instance ||
            heartbeat_event.session_epoch != capture.session_epoch ||
            heartbeat_event.route_digest != capture.route_digest) {
            return fail(receipt,
                        ExternalBodyVerifiedResidentAdmissionFailure::kHeartbeatSessionMismatch);
        }
        if (heartbeat_event.device_tick != capture.capture_device_tick) {
            return fail(receipt,
                        ExternalBodyVerifiedResidentAdmissionFailure::kHeartbeatTickMismatch);
        }
        if (heartbeat_event.parent_route_digest != ticket.parent_route_digest) {
            return fail(receipt,
                        ExternalBodyVerifiedResidentAdmissionFailure::kParentRouteMismatch);
        }
        receipt->ticket_bound = true;
        receipt->route_bound = true;
        receipt->body_heartbeat_bound = true;
        receipt->parent_route_bound = true;
        receipt->parent_route_digest = heartbeat_event.parent_route_digest;
        receipt->attempt_id = capture.attempt_id;
        receipt->capture_id = capture.capture_id;
        receipt->raw_payload_digest = capture.raw_payload_digest;
        receipt->raw_payload_bytes = capture.size;
        consumed_ = true;
        return true;
    }

private:
    static bool fail(ExternalBodyVerifiedResidentAdmissionReceipt* receipt,
                     ExternalBodyVerifiedResidentAdmissionFailure failure) {
        receipt->failure = failure;
        return false;
    }

    bool consumed_ = false;
};

}  // namespace bcc32
