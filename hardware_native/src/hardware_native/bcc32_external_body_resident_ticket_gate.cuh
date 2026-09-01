#pragma once

#include "hardware_native/bcc32_external_body_session_gate.cuh"

#include <cstdint>

namespace bcc32 {

enum class ExternalBodyResidentTicketFailure : std::uint8_t {
    kNone,
    kTicketInvalid,
    kParentRouteMissing,
    kTicketAlreadyOpen,
    kCommandDoesNotMatchTicket,
    kCaptureWithoutTicket,
    kSessionRejected,
};

struct ExternalBodyResidentTicketReceipt {
    bool ticket_open = false;
    bool consumed = false;
    std::uint64_t attempt_id = 0;
    std::uint64_t resident_pre_revision = 0;
    std::uint64_t resident_action_tick = 0;
    std::uint64_t action_envelope = 0;
    std::uint64_t command_digest = 0;
    std::uint64_t parent_route_digest = 0;
    std::uint64_t capture_id = 0;
    std::uint64_t raw_payload_digest = 0;
    ExternalBodyResidentTicketFailure failure = ExternalBodyResidentTicketFailure::kNone;
};

class ExternalBodyResidentTicketGate final {
public:
    bool issue(std::uint64_t attempt_id, std::uint64_t resident_pre_revision,
               std::uint64_t resident_action_tick, std::uint64_t action_envelope,
               std::uint64_t command_digest,
               std::uint64_t parent_route_digest) {
        if (ticket_.ticket_open || ticket_.consumed || attempt_id == 0 ||
            resident_action_tick == 0 || action_envelope == 0 || command_digest == 0) {
            return fail(ExternalBodyResidentTicketFailure::kTicketInvalid);
        }
        if (parent_route_digest == 0) {
            return fail(ExternalBodyResidentTicketFailure::kParentRouteMissing);
        }
        ticket_.ticket_open = true;
        ticket_.attempt_id = attempt_id;
        ticket_.resident_pre_revision = resident_pre_revision;
        ticket_.resident_action_tick = resident_action_tick;
        ticket_.action_envelope = action_envelope;
        ticket_.command_digest = command_digest;
        ticket_.parent_route_digest = parent_route_digest;
        return true;
    }

    bool accept(const ExternalBodyWireEvent& event) {
        if (event.type == ExternalBodyWireType::kCommandAccepted) {
            if (!ticket_.ticket_open || event.attempt_id != ticket_.attempt_id ||
                event.action_envelope != ticket_.action_envelope ||
                event.command_digest != ticket_.command_digest) {
                return fail(ExternalBodyResidentTicketFailure::kCommandDoesNotMatchTicket);
            }
        }
        if (event.type == ExternalBodyWireType::kRawCapture && !ticket_.ticket_open) {
            return fail(ExternalBodyResidentTicketFailure::kCaptureWithoutTicket);
        }
        if (!session_.accept(event)) {
            return fail(ExternalBodyResidentTicketFailure::kSessionRejected);
        }
        if (event.type == ExternalBodyWireType::kRawCapture) {
            ticket_.ticket_open = false;
            ticket_.consumed = true;
            ticket_.capture_id = event.capture_id;
            ticket_.raw_payload_digest = event.raw_payload_digest;
        }
        return true;
    }

    [[nodiscard]] ExternalBodyResidentTicketReceipt receipt() const noexcept { return ticket_; }
    [[nodiscard]] ExternalBodySessionReceipt session_receipt() const noexcept {
        return session_.receipt();
    }

private:
    bool fail(ExternalBodyResidentTicketFailure failure) noexcept {
        ticket_.failure = failure;
        return false;
    }

    ExternalBodySessionGate session_{};
    ExternalBodyResidentTicketReceipt ticket_{};
};

}  // namespace bcc32
