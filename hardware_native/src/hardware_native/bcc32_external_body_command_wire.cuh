#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define BCC32_COMMAND_WIRE_HD __host__ __device__
#else
#define BCC32_COMMAND_WIRE_HD
#endif

namespace bcc32 {

constexpr std::size_t kExternalBodyCommandFrameBytes = 128;

enum class ExternalBodyCommandDisposition : std::uint8_t {
    kApply = 1,
    kWithhold = 2,
    kCancelPending = 3,
};

struct ExternalBodyCommandIssue {
    ExternalBodyCommandDisposition disposition =
        ExternalBodyCommandDisposition::kApply;
    std::uint64_t device_instance = 0;
    std::uint64_t session_epoch = 0;
    std::uint64_t route_digest = 0;
    std::uint64_t route_sequence = 0;
    std::uint64_t attempt_id = 0;
    std::uint64_t resident_pre_revision = 0;
    std::uint64_t resident_action_tick = 0;
    std::uint64_t candidate_set_digest = 0;
    std::uint64_t selected_action_digest = 0;
    std::uint64_t action_envelope = 0;
    std::uint64_t command_digest = 0;
    std::uint64_t deadline_device_tick = 0;
    std::uint64_t contrast_id = 0;
    std::uint8_t contrast_arm = 0;
};

struct ExternalBodyCommandFrame {
    std::uint8_t bytes[kExternalBodyCommandFrameBytes]{};
};

class ExternalBodyCommandWire {
public:
    BCC32_COMMAND_WIRE_HD static bool encode(
        const ExternalBodyCommandIssue& issue,
        ExternalBodyCommandFrame* frame) {
        if (frame == nullptr || !valid(issue)) {
            return false;
        }
        for (std::size_t index = 0; index < sizeof(frame->bytes); ++index) {
            frame->bytes[index] = 0;
        }
        frame->bytes[0] = 'B';
        frame->bytes[1] = 'C';
        frame->bytes[2] = 'C';
        frame->bytes[3] = 'O';
        put16(frame->bytes + 4, 1);
        frame->bytes[6] = static_cast<std::uint8_t>(issue.disposition);
        frame->bytes[7] = issue.contrast_arm;
        put64(frame->bytes + 8, issue.device_instance);
        put64(frame->bytes + 16, issue.session_epoch);
        put64(frame->bytes + 24, issue.route_digest);
        put64(frame->bytes + 32, issue.route_sequence);
        put64(frame->bytes + 40, issue.attempt_id);
        put64(frame->bytes + 48, issue.resident_pre_revision);
        put64(frame->bytes + 56, issue.resident_action_tick);
        put64(frame->bytes + 64, issue.candidate_set_digest);
        put64(frame->bytes + 72, issue.selected_action_digest);
        put64(frame->bytes + 80, issue.action_envelope);
        put64(frame->bytes + 88, issue.command_digest);
        put64(frame->bytes + 96, issue.deadline_device_tick);
        put64(frame->bytes + 104, issue.contrast_id);
        put64(frame->bytes + 120, checksum(frame->bytes, 120));
        return true;
    }

    BCC32_COMMAND_WIRE_HD static bool decode(
        const std::uint8_t* bytes,
        std::size_t byte_count,
        ExternalBodyCommandIssue* issue) {
        if (bytes == nullptr || issue == nullptr ||
            byte_count != kExternalBodyCommandFrameBytes || bytes[0] != 'B' ||
            bytes[1] != 'C' || bytes[2] != 'C' || bytes[3] != 'O' ||
            get16(bytes + 4) != 1 ||
            get64(bytes + 120) != checksum(bytes, 120)) {
            return false;
        }
        for (std::size_t index = 112; index < 120; ++index) {
            if (bytes[index] != 0) {
                return false;
            }
        }
        ExternalBodyCommandIssue decoded{};
        decoded.disposition = static_cast<ExternalBodyCommandDisposition>(bytes[6]);
        decoded.contrast_arm = bytes[7];
        decoded.device_instance = get64(bytes + 8);
        decoded.session_epoch = get64(bytes + 16);
        decoded.route_digest = get64(bytes + 24);
        decoded.route_sequence = get64(bytes + 32);
        decoded.attempt_id = get64(bytes + 40);
        decoded.resident_pre_revision = get64(bytes + 48);
        decoded.resident_action_tick = get64(bytes + 56);
        decoded.candidate_set_digest = get64(bytes + 64);
        decoded.selected_action_digest = get64(bytes + 72);
        decoded.action_envelope = get64(bytes + 80);
        decoded.command_digest = get64(bytes + 88);
        decoded.deadline_device_tick = get64(bytes + 96);
        decoded.contrast_id = get64(bytes + 104);
        if (!valid(decoded)) {
            return false;
        }
        *issue = decoded;
        return true;
    }

private:
    BCC32_COMMAND_WIRE_HD static bool valid(
        const ExternalBodyCommandIssue& issue) {
        if (issue.device_instance == 0 || issue.session_epoch == 0 ||
            issue.route_digest == 0 || issue.route_sequence == 0 ||
            issue.attempt_id == 0 || issue.resident_action_tick == 0 ||
            issue.candidate_set_digest == 0 ||
            issue.selected_action_digest == 0 || issue.action_envelope == 0 ||
            issue.deadline_device_tick == 0 || issue.contrast_arm > 3) {
            return false;
        }
        if ((issue.contrast_id == 0) != (issue.contrast_arm == 0)) {
            return false;
        }
        if (issue.disposition == ExternalBodyCommandDisposition::kApply) {
            return issue.command_digest != 0;
        }
        if (issue.disposition == ExternalBodyCommandDisposition::kWithhold) {
            return issue.command_digest == 0 && issue.contrast_id != 0 &&
                   issue.contrast_arm != 0;
        }
        if (issue.disposition == ExternalBodyCommandDisposition::kCancelPending) {
            return issue.command_digest == 0 && issue.contrast_id == 0 &&
                   issue.contrast_arm == 0;
        }
        return false;
    }

    BCC32_COMMAND_WIRE_HD static void put16(std::uint8_t* out,
                                            std::uint16_t value) {
        out[0] = static_cast<std::uint8_t>(value);
        out[1] = static_cast<std::uint8_t>(value >> 8U);
    }

    BCC32_COMMAND_WIRE_HD static void put64(std::uint8_t* out,
                                            std::uint64_t value) {
        for (unsigned shift = 0; shift < 64U; shift += 8U) {
            out[shift / 8U] = static_cast<std::uint8_t>(value >> shift);
        }
    }

    BCC32_COMMAND_WIRE_HD static std::uint16_t get16(const std::uint8_t* in) {
        return static_cast<std::uint16_t>(in[0]) |
               static_cast<std::uint16_t>(in[1]) << 8U;
    }

    BCC32_COMMAND_WIRE_HD static std::uint64_t get64(const std::uint8_t* in) {
        std::uint64_t value = 0;
        for (unsigned shift = 0; shift < 64U; shift += 8U) {
            value |= static_cast<std::uint64_t>(in[shift / 8U]) << shift;
        }
        return value;
    }

    BCC32_COMMAND_WIRE_HD static std::uint64_t checksum(
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

#undef BCC32_COMMAND_WIRE_HD
