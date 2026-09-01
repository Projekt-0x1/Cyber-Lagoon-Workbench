#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define BCC32_ACK_WIRE_HD __host__ __device__
#else
#define BCC32_ACK_WIRE_HD
#endif

namespace bcc32 {

constexpr std::size_t kExternalBodyCommandAckFrameBytes = 128;

enum class ExternalBodyCommandAckStatus : std::uint8_t {
    kApplied = 1,
    kRejected = 2,
};

struct ExternalBodyCommandAckEvent {
    ExternalBodyCommandAckStatus status = ExternalBodyCommandAckStatus::kApplied;
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
};

struct ExternalBodyCommandAckFrame {
    std::uint8_t bytes[kExternalBodyCommandAckFrameBytes]{};
};

class ExternalBodyCommandAckWire {
public:
    BCC32_ACK_WIRE_HD static bool encode(
        const ExternalBodyCommandAckEvent& event,
        ExternalBodyCommandAckFrame* frame) {
        if (frame == nullptr || !valid(event)) {
            return false;
        }
        for (std::size_t index = 0; index < sizeof(frame->bytes); ++index) {
            frame->bytes[index] = 0;
        }
        frame->bytes[0] = 'B';
        frame->bytes[1] = 'C';
        frame->bytes[2] = 'C';
        frame->bytes[3] = 'A';
        put16(frame->bytes + 4, 1);
        frame->bytes[6] = static_cast<std::uint8_t>(event.status);
        put64(frame->bytes + 8, event.device_instance);
        put64(frame->bytes + 16, event.session_epoch);
        put64(frame->bytes + 24, event.route_digest);
        put64(frame->bytes + 32, event.route_sequence);
        put64(frame->bytes + 40, event.outbox_ticket);
        put64(frame->bytes + 48, event.command_frame_digest);
        put64(frame->bytes + 56, event.attempt_id);
        put64(frame->bytes + 64, event.action_envelope);
        put64(frame->bytes + 72, event.command_digest);
        put64(frame->bytes + 80, event.device_tick);
        put64(frame->bytes + 88, event.device_key_fingerprint);
        put64(frame->bytes + 96, event.signature_commitment);
        put64(frame->bytes + 104, event.device_result_code);
        put64(frame->bytes + 120, checksum(frame->bytes, 120));
        return true;
    }

    BCC32_ACK_WIRE_HD static bool decode(const std::uint8_t* bytes,
                                         std::size_t byte_count,
                                         ExternalBodyCommandAckEvent* event) {
        if (bytes == nullptr || event == nullptr ||
            byte_count != kExternalBodyCommandAckFrameBytes || bytes[0] != 'B' ||
            bytes[1] != 'C' || bytes[2] != 'C' || bytes[3] != 'A' ||
            get16(bytes + 4) != 1 || bytes[7] != 0 ||
            get64(bytes + 120) != checksum(bytes, 120)) {
            return false;
        }
        for (std::size_t index = 112; index < 120; ++index) {
            if (bytes[index] != 0) {
                return false;
            }
        }
        ExternalBodyCommandAckEvent decoded{};
        decoded.status = static_cast<ExternalBodyCommandAckStatus>(bytes[6]);
        decoded.device_instance = get64(bytes + 8);
        decoded.session_epoch = get64(bytes + 16);
        decoded.route_digest = get64(bytes + 24);
        decoded.route_sequence = get64(bytes + 32);
        decoded.outbox_ticket = get64(bytes + 40);
        decoded.command_frame_digest = get64(bytes + 48);
        decoded.attempt_id = get64(bytes + 56);
        decoded.action_envelope = get64(bytes + 64);
        decoded.command_digest = get64(bytes + 72);
        decoded.device_tick = get64(bytes + 80);
        decoded.device_key_fingerprint = get64(bytes + 88);
        decoded.signature_commitment = get64(bytes + 96);
        decoded.device_result_code = get64(bytes + 104);
        if (!valid(decoded)) {
            return false;
        }
        *event = decoded;
        return true;
    }

private:
    BCC32_ACK_WIRE_HD static bool valid(
        const ExternalBodyCommandAckEvent& event) {
        const bool known_status = event.status == ExternalBodyCommandAckStatus::kApplied ||
                                  event.status == ExternalBodyCommandAckStatus::kRejected;
        const bool result_matches_status =
            event.status == ExternalBodyCommandAckStatus::kApplied
                ? event.device_result_code == 0
                : event.device_result_code != 0;
        return known_status && result_matches_status && event.device_instance != 0 &&
               event.session_epoch != 0 && event.route_digest != 0 &&
               event.route_sequence != 0 && event.outbox_ticket != 0 &&
               event.command_frame_digest != 0 && event.attempt_id != 0 &&
               event.action_envelope != 0 && event.command_digest != 0 &&
               event.device_tick != 0 && event.device_key_fingerprint != 0 &&
               event.signature_commitment != 0;
    }

    BCC32_ACK_WIRE_HD static void put16(std::uint8_t* out,
                                        std::uint16_t value) {
        out[0] = static_cast<std::uint8_t>(value);
        out[1] = static_cast<std::uint8_t>(value >> 8U);
    }

    BCC32_ACK_WIRE_HD static void put64(std::uint8_t* out,
                                        std::uint64_t value) {
        for (unsigned shift = 0; shift < 64U; shift += 8U) {
            out[shift / 8U] = static_cast<std::uint8_t>(value >> shift);
        }
    }

    BCC32_ACK_WIRE_HD static std::uint16_t get16(const std::uint8_t* in) {
        return static_cast<std::uint16_t>(in[0]) |
               static_cast<std::uint16_t>(in[1]) << 8U;
    }

    BCC32_ACK_WIRE_HD static std::uint64_t get64(const std::uint8_t* in) {
        std::uint64_t value = 0;
        for (unsigned shift = 0; shift < 64U; shift += 8U) {
            value |= static_cast<std::uint64_t>(in[shift / 8U]) << shift;
        }
        return value;
    }

    BCC32_ACK_WIRE_HD static std::uint64_t checksum(
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

#undef BCC32_ACK_WIRE_HD
