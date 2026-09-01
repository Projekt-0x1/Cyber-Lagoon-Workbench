#pragma once

#include "bcc32_external_body_command_ack_gate.cuh"
#include "bcc32_external_body_command_ack_wire.cuh"

#include <cstddef>
#include <cstdint>

namespace bcc32 {

struct ExternalBodyAckVerificationReceipt {
    bool complete = false;
    bool signature_verified = false;
    bool physical_application_proven = false;
    std::uint64_t device_key_fingerprint = 0;
    std::uint64_t signature_commitment = 0;
    ExternalBodyCommandAckReceipt command_ack{};
};

template <typename SignatureVerifier>
class ExternalBodyCommandAckVerifier {
public:
    explicit ExternalBodyCommandAckVerifier(
        std::uint64_t enrolled_device_key_fingerprint,
        SignatureVerifier verifier)
        : enrolled_key_(enrolled_device_key_fingerprint),
          verifier_(verifier) {}

    bool bind_transport(std::uint64_t device_instance,
                        std::uint64_t session_epoch,
                        std::uint64_t route_digest,
                        const ExternalBodyTransportWriteReceipt& transport) {
        return enrolled_key_ != 0 &&
               gate_.open(device_instance, session_epoch, route_digest, transport);
    }

    bool accept(const std::uint8_t* frame_bytes,
                std::size_t frame_byte_count,
                const std::uint8_t* signature_bytes,
                std::size_t signature_byte_count,
                ExternalBodyAckVerificationReceipt* receipt) {
        if (receipt == nullptr) {
            return false;
        }
        *receipt = ExternalBodyAckVerificationReceipt{};
        if (frame_bytes == nullptr || signature_bytes == nullptr ||
            signature_byte_count == 0 || !gate_.pending()) {
            return false;
        }
        ExternalBodyCommandAckEvent event{};
        if (!ExternalBodyCommandAckWire::decode(
                frame_bytes, frame_byte_count, &event) ||
            event.device_key_fingerprint != enrolled_key_) {
            return false;
        }
        std::uint8_t signed_message[120]{};
        for (std::size_t index = 0; index < sizeof(signed_message); ++index) {
            signed_message[index] = frame_bytes[index];
        }
        for (std::size_t index = 96; index < 104; ++index) {
            signed_message[index] = 0;
        }
        if (!verifier_.verify(enrolled_key_, signed_message,
                              sizeof(signed_message),
                              signature_bytes, signature_byte_count,
                              event.signature_commitment)) {
            return false;
        }

        ExternalBodyCommandAckReceipt command_ack{};
        if (!gate_.accept(event, &command_ack)) {
            return false;
        }
        ExternalBodyAckVerificationReceipt result{};
        result.complete = true;
        result.signature_verified = true;
        result.device_key_fingerprint = event.device_key_fingerprint;
        result.signature_commitment = event.signature_commitment;
        result.command_ack = command_ack;
        *receipt = result;
        return true;
    }

    bool pending() const { return gate_.pending(); }

private:
    std::uint64_t enrolled_key_ = 0;
    SignatureVerifier verifier_;
    ExternalBodyCommandAckGate gate_{};
};

}  // namespace bcc32
