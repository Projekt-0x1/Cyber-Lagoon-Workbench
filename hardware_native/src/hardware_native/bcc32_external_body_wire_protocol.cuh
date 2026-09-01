#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define BCC32_BODY_WIRE_HD __host__ __device__
#else
#define BCC32_BODY_WIRE_HD
#endif

namespace bcc32 {

constexpr std::size_t kExternalBodyWireFrameBytes = 128;

enum class ExternalBodyWireType : std::uint8_t {
    kSessionOpen = 1,
    kCommandAccepted = 2,
    kRawCapture = 3,
};

struct ExternalBodyWireEvent {
    ExternalBodyWireType type = ExternalBodyWireType::kSessionOpen;
    bool command_applied = false;
    std::uint64_t device_instance = 0;
    std::uint64_t session_epoch = 0;
    std::uint64_t route_digest = 0;
    std::uint64_t attempt_id = 0;
    std::uint64_t action_envelope = 0;
    std::uint64_t command_digest = 0;
    std::uint64_t capture_id = 0;
    std::uint64_t device_tick = 0;
    std::uint64_t raw_payload_digest = 0;
    std::uint32_t raw_word_count = 0;
    std::uint32_t raw_payload_bytes = 0;
    std::uint64_t route_sequence = 0;
};

struct ExternalBodyWireFrame {
    std::uint8_t bytes[kExternalBodyWireFrameBytes]{};
};

class ExternalBodyWireProtocol {
public:
    BCC32_BODY_WIRE_HD static bool encode(const ExternalBodyWireEvent& event,
                                          ExternalBodyWireFrame* frame) {
        if (frame == nullptr || !valid(event)) {
            return false;
        }
        for (std::size_t index = 0; index < kExternalBodyWireFrameBytes; ++index) {
            frame->bytes[index] = 0;
        }
        frame->bytes[0] = 'B';
        frame->bytes[1] = 'C';
        frame->bytes[2] = 'C';
        frame->bytes[3] = '3';
        put16(frame->bytes + 4, 1);
        frame->bytes[6] = static_cast<std::uint8_t>(event.type);
        frame->bytes[7] = event.command_applied ? 1U : 0U;
        put64(frame->bytes + 8, event.device_instance);
        put64(frame->bytes + 16, event.session_epoch);
        put64(frame->bytes + 24, event.route_digest);
        put64(frame->bytes + 32, event.attempt_id);
        put64(frame->bytes + 40, event.action_envelope);
        put64(frame->bytes + 48, event.command_digest);
        put64(frame->bytes + 56, event.capture_id);
        put64(frame->bytes + 64, event.device_tick);
        put64(frame->bytes + 72, event.raw_payload_digest);
        put32(frame->bytes + 80, event.raw_word_count);
        put32(frame->bytes + 84, event.raw_payload_bytes);
        put64(frame->bytes + 88, event.route_sequence);
        put64(frame->bytes + 120, checksum(frame->bytes, 120));
        return true;
    }

    BCC32_BODY_WIRE_HD static bool decode(const std::uint8_t* bytes,
                                          std::size_t byte_count,
                                          ExternalBodyWireEvent* event) {
        if (bytes == nullptr || event == nullptr ||
            byte_count != kExternalBodyWireFrameBytes || bytes[0] != 'B' ||
            bytes[1] != 'C' || bytes[2] != 'C' || bytes[3] != '3' ||
            get16(bytes + 4) != 1 || bytes[7] > 1 ||
            get64(bytes + 120) != checksum(bytes, 120)) {
            return false;
        }
        for (std::size_t index = 96; index < 120; ++index) {
            if (bytes[index] != 0) {
                return false;
            }
        }

        ExternalBodyWireEvent decoded{};
        decoded.type = static_cast<ExternalBodyWireType>(bytes[6]);
        decoded.command_applied = bytes[7] == 1;
        decoded.device_instance = get64(bytes + 8);
        decoded.session_epoch = get64(bytes + 16);
        decoded.route_digest = get64(bytes + 24);
        decoded.attempt_id = get64(bytes + 32);
        decoded.action_envelope = get64(bytes + 40);
        decoded.command_digest = get64(bytes + 48);
        decoded.capture_id = get64(bytes + 56);
        decoded.device_tick = get64(bytes + 64);
        decoded.raw_payload_digest = get64(bytes + 72);
        decoded.raw_word_count = get32(bytes + 80);
        decoded.raw_payload_bytes = get32(bytes + 84);
        decoded.route_sequence = get64(bytes + 88);
        if (!valid(decoded)) {
            return false;
        }
        *event = decoded;
        return true;
    }

private:
    BCC32_BODY_WIRE_HD static bool valid(const ExternalBodyWireEvent& event) {
        if (event.device_instance == 0 || event.session_epoch == 0 ||
            event.route_digest == 0 || event.device_tick == 0 ||
            event.route_sequence == 0 || event.raw_word_count > (1U << 20U) ||
            event.raw_payload_bytes > (16U << 20U)) {
            return false;
        }
        if (event.type == ExternalBodyWireType::kSessionOpen) {
            return !event.command_applied && event.attempt_id == 0 &&
                   event.action_envelope == 0 && event.command_digest == 0 &&
                   event.capture_id == 0 && event.raw_payload_digest == 0 &&
                   event.raw_word_count == 0 && event.raw_payload_bytes == 0;
        }
        if (event.type == ExternalBodyWireType::kCommandAccepted) {
            return event.command_applied && event.attempt_id != 0 &&
                   event.action_envelope != 0 && event.command_digest != 0 &&
                   event.capture_id == 0 && event.raw_payload_digest == 0 &&
                   event.raw_word_count == 0 && event.raw_payload_bytes == 0;
        }
        if (event.type == ExternalBodyWireType::kRawCapture) {
            return !event.command_applied && event.attempt_id != 0 &&
                   event.action_envelope != 0 && event.command_digest != 0 &&
                   event.capture_id != 0 && event.raw_payload_digest != 0 &&
                   event.raw_word_count != 0 && event.raw_payload_bytes != 0;
        }
        return false;
    }

    BCC32_BODY_WIRE_HD static void put16(std::uint8_t* out, std::uint16_t value) {
        out[0] = static_cast<std::uint8_t>(value);
        out[1] = static_cast<std::uint8_t>(value >> 8U);
    }

    BCC32_BODY_WIRE_HD static void put32(std::uint8_t* out, std::uint32_t value) {
        for (unsigned shift = 0; shift < 32U; shift += 8U) {
            out[shift / 8U] = static_cast<std::uint8_t>(value >> shift);
        }
    }

    BCC32_BODY_WIRE_HD static void put64(std::uint8_t* out, std::uint64_t value) {
        for (unsigned shift = 0; shift < 64U; shift += 8U) {
            out[shift / 8U] = static_cast<std::uint8_t>(value >> shift);
        }
    }

    BCC32_BODY_WIRE_HD static std::uint16_t get16(const std::uint8_t* in) {
        return static_cast<std::uint16_t>(in[0]) |
               static_cast<std::uint16_t>(in[1]) << 8U;
    }

    BCC32_BODY_WIRE_HD static std::uint32_t get32(const std::uint8_t* in) {
        std::uint32_t value = 0;
        for (unsigned shift = 0; shift < 32U; shift += 8U) {
            value |= static_cast<std::uint32_t>(in[shift / 8U]) << shift;
        }
        return value;
    }

    BCC32_BODY_WIRE_HD static std::uint64_t get64(const std::uint8_t* in) {
        std::uint64_t value = 0;
        for (unsigned shift = 0; shift < 64U; shift += 8U) {
            value |= static_cast<std::uint64_t>(in[shift / 8U]) << shift;
        }
        return value;
    }

    BCC32_BODY_WIRE_HD static std::uint64_t checksum(const std::uint8_t* bytes,
                                                     std::size_t byte_count) {
        std::uint64_t hash = 1469598103934665603ULL;
        for (std::size_t index = 0; index < byte_count; ++index) {
            hash ^= bytes[index];
            hash *= 1099511628211ULL;
        }
        return hash == 0 ? 1 : hash;
    }
};

}  // namespace bcc32

#undef BCC32_BODY_WIRE_HD
