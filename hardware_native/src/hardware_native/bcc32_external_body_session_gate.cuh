#pragma once

#include "hardware_native/bcc32_external_body_wire_protocol.cuh"

#include <cstdint>

namespace bcc32 {

enum class ExternalBodySessionFailure : std::uint8_t {
    kNone,
    kFaultLatched,
    kDecode,
    kExpectedSession,
    kSessionIdentity,
    kRouteSequence,
    kDeviceTick,
    kExpectedCommand,
    kExpectedCapture,
    kActionBinding,
};

struct ExternalBodySessionReceipt {
    bool opened = false;
    bool command_pending = false;
    bool capture_committed = false;
    std::uint64_t device_instance = 0;
    std::uint64_t session_epoch = 0;
    std::uint64_t route_digest = 0;
    std::uint64_t last_route_sequence = 0;
    std::uint64_t last_device_tick = 0;
    std::uint64_t accepted_attempt_id = 0;
    std::uint64_t accepted_action_envelope = 0;
    std::uint64_t accepted_command_digest = 0;
    std::uint64_t captured_raw_digest = 0;
    std::uint64_t captured_id = 0;
    ExternalBodySessionFailure failure = ExternalBodySessionFailure::kNone;
};

class ExternalBodySessionGate final {
public:
    bool accept(const std::uint8_t* bytes, std::size_t byte_count) {
        ExternalBodyWireEvent event{};
        if (!ExternalBodyWireProtocol::decode(bytes, byte_count, &event)) {
            return fail(ExternalBodySessionFailure::kDecode);
        }
        return accept(event);
    }

    bool accept(const ExternalBodyWireEvent& event) {
        if (receipt_.failure != ExternalBodySessionFailure::kNone) {
            return fail(ExternalBodySessionFailure::kFaultLatched);
        }
        if (!receipt_.opened) {
            if (event.type != ExternalBodyWireType::kSessionOpen) {
                return fail(ExternalBodySessionFailure::kExpectedSession);
            }
            receipt_.opened = true;
            receipt_.device_instance = event.device_instance;
            receipt_.session_epoch = event.session_epoch;
            receipt_.route_digest = event.route_digest;
            receipt_.last_route_sequence = event.route_sequence;
            receipt_.last_device_tick = event.device_tick;
            return true;
        }
        if (event.device_instance != receipt_.device_instance ||
            event.session_epoch != receipt_.session_epoch ||
            event.route_digest != receipt_.route_digest) {
            return fail(ExternalBodySessionFailure::kSessionIdentity);
        }
        if (event.route_sequence != receipt_.last_route_sequence + 1u) {
            return fail(ExternalBodySessionFailure::kRouteSequence);
        }
        if (event.device_tick <= receipt_.last_device_tick) {
            return fail(ExternalBodySessionFailure::kDeviceTick);
        }
        if (!receipt_.command_pending) {
            if (event.type != ExternalBodyWireType::kCommandAccepted) {
                return fail(ExternalBodySessionFailure::kExpectedCommand);
            }
            receipt_.command_pending = true;
            receipt_.accepted_attempt_id = event.attempt_id;
            receipt_.accepted_action_envelope = event.action_envelope;
            receipt_.accepted_command_digest = event.command_digest;
        } else {
            if (event.type != ExternalBodyWireType::kRawCapture) {
                return fail(ExternalBodySessionFailure::kExpectedCapture);
            }
            if (event.attempt_id != receipt_.accepted_attempt_id ||
                event.action_envelope != receipt_.accepted_action_envelope ||
                event.command_digest != receipt_.accepted_command_digest) {
                return fail(ExternalBodySessionFailure::kActionBinding);
            }
            receipt_.command_pending = false;
            receipt_.capture_committed = true;
            receipt_.captured_id = event.capture_id;
            receipt_.captured_raw_digest = event.raw_payload_digest;
        }
        receipt_.last_route_sequence = event.route_sequence;
        receipt_.last_device_tick = event.device_tick;
        return true;
    }

    [[nodiscard]] ExternalBodySessionReceipt receipt() const noexcept { return receipt_; }

private:
    bool fail(ExternalBodySessionFailure failure) noexcept {
        receipt_.failure = failure;
        return false;
    }

    ExternalBodySessionReceipt receipt_{};
};

}  // namespace bcc32
