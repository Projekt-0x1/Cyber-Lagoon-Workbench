#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define BCC32_HEARTBEAT_WIRE_HD __host__ __device__
#else
#define BCC32_HEARTBEAT_WIRE_HD
#endif

namespace bcc32 {

constexpr std::size_t kExternalBodyHeartbeatFrameBytes = 96;

struct ExternalBodyHeartbeatEvent {
    std::uint64_t device_instance = 0;
    std::uint64_t session_epoch = 0;
    std::uint64_t route_digest = 0;
    std::uint64_t heartbeat_sequence = 0;
    std::uint64_t device_tick = 0;
    std::uint64_t device_key_fingerprint = 0;
    std::uint64_t challenge_nonce = 0;
    std::uint64_t signature_commitment = 0;
    // Optional opaque upstream ancestry for the physical route. This is
    // transport evidence, not a source identity or trust value. Keeping it
    // in the signed/checksummed frame lets the resident adapter bind it to a
    // RawRewriteEvent later without inventing ancestry on the host.
    std::uint64_t parent_route_digest = 0;
};

struct ExternalBodyHeartbeatFrame {
    std::uint8_t bytes[kExternalBodyHeartbeatFrameBytes]{};
};

class ExternalBodyHeartbeatWire {
public:
    BCC32_HEARTBEAT_WIRE_HD static bool encode(
        const ExternalBodyHeartbeatEvent& event,
        ExternalBodyHeartbeatFrame* frame) {
        if (frame == nullptr || !valid(event)) {
            return false;
        }
        for (std::size_t index = 0; index < sizeof(frame->bytes); ++index) {
            frame->bytes[index] = 0;
        }
        frame->bytes[0] = 'B';
        frame->bytes[1] = 'C';
        frame->bytes[2] = 'C';
        frame->bytes[3] = 'H';
        put16(frame->bytes + 4, 1);
        put16(frame->bytes + 6,
              static_cast<std::uint16_t>(kExternalBodyHeartbeatFrameBytes));
        put64(frame->bytes + 8, event.device_instance);
        put64(frame->bytes + 16, event.session_epoch);
        put64(frame->bytes + 24, event.route_digest);
        put64(frame->bytes + 32, event.heartbeat_sequence);
        put64(frame->bytes + 40, event.device_tick);
        put64(frame->bytes + 48, event.device_key_fingerprint);
        put64(frame->bytes + 56, event.challenge_nonce);
        put64(frame->bytes + 64, event.signature_commitment);
        put64(frame->bytes + 72, event.parent_route_digest);
        put64(frame->bytes + 88, checksum(frame->bytes, 88));
        return true;
    }

    BCC32_HEARTBEAT_WIRE_HD static bool decode(
        const std::uint8_t* bytes,
        std::size_t byte_count,
        ExternalBodyHeartbeatEvent* event) {
        if (bytes == nullptr || event == nullptr ||
            byte_count != kExternalBodyHeartbeatFrameBytes || bytes[0] != 'B' ||
            bytes[1] != 'C' || bytes[2] != 'C' || bytes[3] != 'H' ||
            get16(bytes + 4) != 1 ||
            get16(bytes + 6) != kExternalBodyHeartbeatFrameBytes ||
            get64(bytes + 88) != checksum(bytes, 88)) {
            return false;
        }
        ExternalBodyHeartbeatEvent decoded{};
        decoded.device_instance = get64(bytes + 8);
        decoded.session_epoch = get64(bytes + 16);
        decoded.route_digest = get64(bytes + 24);
        decoded.heartbeat_sequence = get64(bytes + 32);
        decoded.device_tick = get64(bytes + 40);
        decoded.device_key_fingerprint = get64(bytes + 48);
        decoded.challenge_nonce = get64(bytes + 56);
        decoded.signature_commitment = get64(bytes + 64);
        decoded.parent_route_digest = get64(bytes + 72);
        if (!valid(decoded)) {
            return false;
        }
        *event = decoded;
        return true;
    }

private:
    BCC32_HEARTBEAT_WIRE_HD static bool valid(
        const ExternalBodyHeartbeatEvent& event) {
        return event.device_instance != 0 && event.session_epoch != 0 &&
               event.route_digest != 0 && event.heartbeat_sequence != 0 &&
               event.device_tick != 0 && event.device_key_fingerprint != 0 &&
               event.challenge_nonce != 0 && event.signature_commitment != 0;
    }

    BCC32_HEARTBEAT_WIRE_HD static void put16(std::uint8_t* out,
                                              std::uint16_t value) {
        out[0] = static_cast<std::uint8_t>(value);
        out[1] = static_cast<std::uint8_t>(value >> 8U);
    }

    BCC32_HEARTBEAT_WIRE_HD static void put64(std::uint8_t* out,
                                              std::uint64_t value) {
        for (unsigned shift = 0; shift < 64U; shift += 8U) {
            out[shift / 8U] = static_cast<std::uint8_t>(value >> shift);
        }
    }

    BCC32_HEARTBEAT_WIRE_HD static std::uint16_t get16(const std::uint8_t* in) {
        return static_cast<std::uint16_t>(in[0]) |
               static_cast<std::uint16_t>(in[1]) << 8U;
    }

    BCC32_HEARTBEAT_WIRE_HD static std::uint64_t get64(const std::uint8_t* in) {
        std::uint64_t value = 0;
        for (unsigned shift = 0; shift < 64U; shift += 8U) {
            value |= static_cast<std::uint64_t>(in[shift / 8U]) << shift;
        }
        return value;
    }

    BCC32_HEARTBEAT_WIRE_HD static std::uint64_t checksum(
        const std::uint8_t* bytes,
        std::size_t byte_count) {
        std::uint64_t state = 1469598103934665603ULL;
        for (std::size_t index = 0; index < byte_count; ++index) {
            state ^= bytes[index];
            state *= 1099511628211ULL;
        }
        return state == 0 ? 1 : state;
    }
};

}  // namespace bcc32

#undef BCC32_HEARTBEAT_WIRE_HD
