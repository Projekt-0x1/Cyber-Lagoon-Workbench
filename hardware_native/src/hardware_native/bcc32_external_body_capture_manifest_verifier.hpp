#pragma once

#include "bcc32_external_body_capture_manifest_wire.cuh"
#include "bcc32_external_body_command_ack_verifier.hpp"

#include <cstddef>
#include <cstdint>

namespace bcc32 {

struct ExternalBodyCaptureManifestVerificationReceipt {
    bool complete = false;
    bool signature_verified = false;
    bool physical_source_proven = false;
    bool physical_consequence_proven = false;
    ExternalBodyCaptureManifest manifest{};
    std::uint64_t verification_commitment = 0;
};

template <typename SignatureVerifier>
class ExternalBodyCaptureManifestVerifier {
public:
    ExternalBodyCaptureManifestVerifier(std::uint64_t enrolled_key_fingerprint,
                                        SignatureVerifier verifier)
        : enrolled_key_(enrolled_key_fingerprint), verifier_(verifier) {}

    bool verify(const ExternalBodyAckVerificationReceipt& acknowledged,
                const std::uint8_t* frame_bytes,
                std::size_t frame_byte_count,
                const std::uint8_t* signature_bytes,
                std::size_t signature_byte_count,
                ExternalBodyCaptureManifestVerificationReceipt* receipt) {
        if (receipt == nullptr) {
            return false;
        }
        *receipt = ExternalBodyCaptureManifestVerificationReceipt{};
        if (frame_bytes == nullptr || signature_bytes == nullptr ||
            signature_byte_count == 0 || enrolled_key_ == 0 ||
            !acknowledged.complete || !acknowledged.signature_verified ||
            acknowledged.physical_application_proven ||
            acknowledged.device_key_fingerprint != enrolled_key_ ||
            acknowledged.signature_commitment == 0 ||
            !acknowledged.command_ack.complete ||
            !acknowledged.command_ack.declared_applied ||
            acknowledged.command_ack.status != ExternalBodyCommandAckStatus::kApplied ||
            acknowledged.command_ack.device_key_fingerprint != enrolled_key_ ||
            acknowledged.command_ack.signature_commitment !=
                acknowledged.signature_commitment) {
            return false;
        }
        ExternalBodyCaptureManifest manifest{};
        if (!ExternalBodyCaptureManifestWire::decode(
                frame_bytes, frame_byte_count, &manifest) ||
            manifest.device_key_fingerprint != enrolled_key_) {
            return false;
        }
        std::uint8_t signed_message[120]{};
        for (std::size_t index = 0; index < sizeof(signed_message); ++index) {
            signed_message[index] = frame_bytes[index];
        }
        for (std::size_t index = 104; index < 112; ++index) {
            signed_message[index] = 0;
        }
        if (!verifier_.verify(enrolled_key_, signed_message,
                              sizeof(signed_message),
                              signature_bytes, signature_byte_count,
                              manifest.signature_commitment)) {
            return false;
        }
        const auto& ack = acknowledged.command_ack;
        if (manifest.device_instance != ack.device_instance ||
            manifest.session_epoch != ack.session_epoch ||
            manifest.route_digest != ack.route_digest ||
            manifest.attempt_id != ack.attempt_id ||
            manifest.action_envelope != ack.action_envelope ||
            manifest.accepted_command_digest != ack.command_digest ||
            manifest.command_ack_commitment != ack.structural_commitment ||
            manifest.capture_device_tick <= ack.device_tick ||
            manifest.capture_ticket <= last_capture_ticket_ ||
            manifest.attempt_id <= last_attempt_id_) {
            return false;
        }
        ExternalBodyCaptureManifestVerificationReceipt result{};
        result.complete = true;
        result.signature_verified = true;
        result.manifest = manifest;
        result.verification_commitment = commitment(manifest);
        *receipt = result;
        last_capture_ticket_ = manifest.capture_ticket;
        last_attempt_id_ = manifest.attempt_id;
        return true;
    }

private:
    static std::uint64_t mix(std::uint64_t state, std::uint64_t value) {
        state ^= value + 0x9e3779b97f4a7c15ULL + (state << 6U) + (state >> 2U);
        return state;
    }

    static std::uint64_t commitment(const ExternalBodyCaptureManifest& manifest) {
        std::uint64_t state = 0x4341504d414e5631ULL;
        const std::uint64_t fields[] = {
            manifest.device_instance, manifest.session_epoch, manifest.route_digest,
            manifest.capture_ticket, manifest.attempt_id, manifest.action_envelope,
            manifest.accepted_command_digest, manifest.command_ack_commitment,
            manifest.capture_device_tick, manifest.total_payload_bytes,
            manifest.total_raw_words, manifest.full_raw_payload_digest,
            manifest.device_key_fingerprint, manifest.signature_commitment,
        };
        for (const std::uint64_t field : fields) state = mix(state, field);
        return state == 0 ? 1 : state;
    }

    std::uint64_t enrolled_key_ = 0;
    SignatureVerifier verifier_;
    std::uint64_t last_capture_ticket_ = 0;
    std::uint64_t last_attempt_id_ = 0;
};

}  // namespace bcc32
