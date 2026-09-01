#pragma once

#include "bcc32_external_body_heartbeat_wire.cuh"

#include <cstddef>
#include <cstdint>

namespace bcc32 {

struct ExternalBodyHeartbeatVerificationReceipt {
    bool complete = false;
    bool signature_verified = false;
    bool physical_liveness_proven = false;
    std::uint64_t challenge_nonce = 0;
    std::uint64_t signature_commitment = 0;
    ExternalBodyHeartbeatEvent heartbeat{};
};

template <typename SignatureVerifier>
class ExternalBodyHeartbeatVerifier {
public:
    ExternalBodyHeartbeatVerifier(std::uint64_t device_instance,
                                  std::uint64_t session_epoch,
                                  std::uint64_t route_digest,
                                  std::uint64_t enrolled_key_fingerprint,
                                  SignatureVerifier verifier)
        : device_instance_(device_instance),
          session_epoch_(session_epoch),
          route_digest_(route_digest),
          enrolled_key_(enrolled_key_fingerprint),
          verifier_(verifier),
          configured_(device_instance != 0 && session_epoch != 0 &&
                      route_digest != 0 && enrolled_key_fingerprint != 0) {}

    bool issue_challenge(std::uint64_t nonce) {
        if (!configured_ || challenge_pending_ || nonce == 0 ||
            nonce <= last_consumed_nonce_) {
            return false;
        }
        pending_nonce_ = nonce;
        challenge_pending_ = true;
        return true;
    }

    bool accept(const std::uint8_t* frame_bytes,
                std::size_t frame_byte_count,
                const std::uint8_t* signature_bytes,
                std::size_t signature_byte_count,
                ExternalBodyHeartbeatVerificationReceipt* receipt) {
        if (receipt == nullptr) {
            return false;
        }
        *receipt = ExternalBodyHeartbeatVerificationReceipt{};
        if (!challenge_pending_ || frame_bytes == nullptr ||
            signature_bytes == nullptr || signature_byte_count == 0) {
            return false;
        }
        ExternalBodyHeartbeatEvent event{};
        if (!ExternalBodyHeartbeatWire::decode(
                frame_bytes, frame_byte_count, &event) ||
            event.device_instance != device_instance_ ||
            event.session_epoch != session_epoch_ ||
            event.route_digest != route_digest_ ||
            event.device_key_fingerprint != enrolled_key_ ||
            event.challenge_nonce != pending_nonce_ ||
            event.heartbeat_sequence <= last_heartbeat_sequence_ ||
            event.device_tick <= last_device_tick_) {
            return false;
        }
        std::uint8_t signed_message[88]{};
        for (std::size_t index = 0; index < sizeof(signed_message); ++index) {
            signed_message[index] = frame_bytes[index];
        }
        for (std::size_t index = 64; index < 72; ++index) {
            signed_message[index] = 0;
        }
        if (!verifier_.verify(enrolled_key_, signed_message,
                              sizeof(signed_message),
                              signature_bytes, signature_byte_count,
                              event.signature_commitment)) {
            return false;
        }
        ExternalBodyHeartbeatVerificationReceipt result{};
        result.complete = true;
        result.signature_verified = true;
        result.challenge_nonce = event.challenge_nonce;
        result.signature_commitment = event.signature_commitment;
        result.heartbeat = event;
        *receipt = result;
        last_consumed_nonce_ = pending_nonce_;
        pending_nonce_ = 0;
        challenge_pending_ = false;
        last_heartbeat_sequence_ = event.heartbeat_sequence;
        last_device_tick_ = event.device_tick;
        return true;
    }

    bool challenge_pending() const { return challenge_pending_; }
    std::uint64_t last_device_tick() const { return last_device_tick_; }
    std::uint64_t last_heartbeat_sequence() const {
        return last_heartbeat_sequence_;
    }

private:
    std::uint64_t device_instance_ = 0;
    std::uint64_t session_epoch_ = 0;
    std::uint64_t route_digest_ = 0;
    std::uint64_t enrolled_key_ = 0;
    SignatureVerifier verifier_;
    std::uint64_t pending_nonce_ = 0;
    std::uint64_t last_consumed_nonce_ = 0;
    std::uint64_t last_heartbeat_sequence_ = 0;
    std::uint64_t last_device_tick_ = 0;
    bool configured_ = false;
    bool challenge_pending_ = false;
};

}  // namespace bcc32
