#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define BCC32_CAPTURE_MANIFEST_HD __host__ __device__
#else
#define BCC32_CAPTURE_MANIFEST_HD
#endif

namespace bcc32 {

constexpr std::size_t kExternalBodyCaptureManifestFrameBytes = 128;

struct ExternalBodyCaptureManifest {
    std::uint64_t device_instance = 0;
    std::uint64_t session_epoch = 0;
    std::uint64_t route_digest = 0;
    std::uint64_t capture_ticket = 0;
    std::uint64_t attempt_id = 0;
    std::uint64_t action_envelope = 0;
    std::uint64_t accepted_command_digest = 0;
    std::uint64_t command_ack_commitment = 0;
    std::uint64_t capture_device_tick = 0;
    std::uint32_t total_payload_bytes = 0;
    std::uint32_t total_raw_words = 0;
    std::uint64_t full_raw_payload_digest = 0;
    std::uint64_t device_key_fingerprint = 0;
    std::uint64_t signature_commitment = 0;
};

struct ExternalBodyCaptureManifestFrame {
    std::uint8_t bytes[kExternalBodyCaptureManifestFrameBytes]{};
};

class ExternalBodyCaptureManifestWire {
public:
    BCC32_CAPTURE_MANIFEST_HD static bool encode(
        const ExternalBodyCaptureManifest& manifest,
        ExternalBodyCaptureManifestFrame* frame) {
        if (frame == nullptr || !valid(manifest)) {
            return false;
        }
        for (std::size_t index = 0; index < sizeof(frame->bytes); ++index) {
            frame->bytes[index] = 0;
        }
        frame->bytes[0] = 'B';
        frame->bytes[1] = 'C';
        frame->bytes[2] = 'C';
        frame->bytes[3] = 'M';
        put16(frame->bytes + 4, 1);
        put16(frame->bytes + 6,
              static_cast<std::uint16_t>(kExternalBodyCaptureManifestFrameBytes));
        put64(frame->bytes + 8, manifest.device_instance);
        put64(frame->bytes + 16, manifest.session_epoch);
        put64(frame->bytes + 24, manifest.route_digest);
        put64(frame->bytes + 32, manifest.capture_ticket);
        put64(frame->bytes + 40, manifest.attempt_id);
        put64(frame->bytes + 48, manifest.action_envelope);
        put64(frame->bytes + 56, manifest.accepted_command_digest);
        put64(frame->bytes + 64, manifest.command_ack_commitment);
        put64(frame->bytes + 72, manifest.capture_device_tick);
        put32(frame->bytes + 80, manifest.total_payload_bytes);
        put32(frame->bytes + 84, manifest.total_raw_words);
        put64(frame->bytes + 88, manifest.full_raw_payload_digest);
        put64(frame->bytes + 96, manifest.device_key_fingerprint);
        put64(frame->bytes + 104, manifest.signature_commitment);
        put64(frame->bytes + 120, checksum(frame->bytes, 120));
        return true;
    }

    BCC32_CAPTURE_MANIFEST_HD static bool decode(
        const std::uint8_t* bytes,
        std::size_t byte_count,
        ExternalBodyCaptureManifest* manifest) {
        if (bytes == nullptr || manifest == nullptr ||
            byte_count != kExternalBodyCaptureManifestFrameBytes || bytes[0] != 'B' ||
            bytes[1] != 'C' || bytes[2] != 'C' || bytes[3] != 'M' ||
            get16(bytes + 4) != 1 ||
            get16(bytes + 6) != kExternalBodyCaptureManifestFrameBytes ||
            get64(bytes + 120) != checksum(bytes, 120)) {
            return false;
        }
        for (std::size_t index = 112; index < 120; ++index) {
            if (bytes[index] != 0) return false;
        }
        ExternalBodyCaptureManifest decoded{};
        decoded.device_instance = get64(bytes + 8);
        decoded.session_epoch = get64(bytes + 16);
        decoded.route_digest = get64(bytes + 24);
        decoded.capture_ticket = get64(bytes + 32);
        decoded.attempt_id = get64(bytes + 40);
        decoded.action_envelope = get64(bytes + 48);
        decoded.accepted_command_digest = get64(bytes + 56);
        decoded.command_ack_commitment = get64(bytes + 64);
        decoded.capture_device_tick = get64(bytes + 72);
        decoded.total_payload_bytes = get32(bytes + 80);
        decoded.total_raw_words = get32(bytes + 84);
        decoded.full_raw_payload_digest = get64(bytes + 88);
        decoded.device_key_fingerprint = get64(bytes + 96);
        decoded.signature_commitment = get64(bytes + 104);
        if (!valid(decoded)) return false;
        *manifest = decoded;
        return true;
    }

private:
    BCC32_CAPTURE_MANIFEST_HD static bool valid(
        const ExternalBodyCaptureManifest& manifest) {
        if (manifest.device_instance == 0 || manifest.session_epoch == 0 ||
            manifest.route_digest == 0 || manifest.capture_ticket == 0 ||
            manifest.attempt_id == 0 || manifest.action_envelope == 0 ||
            manifest.accepted_command_digest == 0 ||
            manifest.command_ack_commitment == 0 ||
            manifest.capture_device_tick == 0 ||
            manifest.total_payload_bytes == 0 ||
            manifest.total_payload_bytes > (16U << 20U) ||
            manifest.total_raw_words == 0 ||
            manifest.total_raw_words != (manifest.total_payload_bytes + 3U) / 4U ||
            manifest.full_raw_payload_digest == 0 ||
            manifest.device_key_fingerprint == 0 ||
            manifest.signature_commitment == 0) {
            return false;
        }
        return true;
    }

    BCC32_CAPTURE_MANIFEST_HD static void put16(std::uint8_t* out,
                                                std::uint16_t value) {
        out[0] = static_cast<std::uint8_t>(value);
        out[1] = static_cast<std::uint8_t>(value >> 8U);
    }

    BCC32_CAPTURE_MANIFEST_HD static void put32(std::uint8_t* out,
                                                std::uint32_t value) {
        for (unsigned shift = 0; shift < 32U; shift += 8U)
            out[shift / 8U] = static_cast<std::uint8_t>(value >> shift);
    }

    BCC32_CAPTURE_MANIFEST_HD static void put64(std::uint8_t* out,
                                                std::uint64_t value) {
        for (unsigned shift = 0; shift < 64U; shift += 8U)
            out[shift / 8U] = static_cast<std::uint8_t>(value >> shift);
    }

    BCC32_CAPTURE_MANIFEST_HD static std::uint16_t get16(const std::uint8_t* in) {
        return static_cast<std::uint16_t>(in[0]) |
               static_cast<std::uint16_t>(in[1]) << 8U;
    }

    BCC32_CAPTURE_MANIFEST_HD static std::uint32_t get32(const std::uint8_t* in) {
        std::uint32_t value = 0;
        for (unsigned shift = 0; shift < 32U; shift += 8U)
            value |= static_cast<std::uint32_t>(in[shift / 8U]) << shift;
        return value;
    }

    BCC32_CAPTURE_MANIFEST_HD static std::uint64_t get64(const std::uint8_t* in) {
        std::uint64_t value = 0;
        for (unsigned shift = 0; shift < 64U; shift += 8U)
            value |= static_cast<std::uint64_t>(in[shift / 8U]) << shift;
        return value;
    }

    BCC32_CAPTURE_MANIFEST_HD static std::uint64_t checksum(
        const std::uint8_t* bytes, std::size_t byte_count) {
        std::uint64_t state = 1469598103934665603ULL;
        for (std::size_t index = 0; index < byte_count; ++index) {
            state ^= bytes[index];
            state *= 1099511628211ULL;
        }
        return state == 0 ? 1 : state;
    }
};

}  // namespace bcc32

#undef BCC32_CAPTURE_MANIFEST_HD
