#pragma once

#include "bcc32_external_body_capture_manifest_verifier.hpp"
#include "bcc32_external_body_ingress_queue.cuh"

#include <array>
#include <cstddef>
#include <cstdint>

namespace substrate::bcc32::external_body_capture_wire {

struct DeviceSession {
    std::uint64_t device_instance = 0;
    std::uint64_t session_epoch = 0;
};

struct ReturnRoute {
    std::uint64_t producer_instance = 0;
    std::uint64_t source_epoch = 0;
};

struct ChunkFields {
    DeviceSession session{};
    ReturnRoute route{};
    std::uint64_t capture_ticket = 0;
    std::uint64_t chunk_sequence = 0;
    std::uint64_t chunk_offset = 0;
    std::uint32_t total_size = 0;
    std::uint32_t chunk_size = 0;
    std::uint64_t full_digest = 0;
};

template <std::size_t MaxChunkBytes>
struct DecodedChunk {
    ChunkFields fields{};
    std::array<std::uint8_t, MaxChunkBytes> bytes{};
    std::size_t size = 0;
};

struct EncodeResult { bool ok = false; };
struct DecodeResult { bool ok = false; };

class ExternalBodyCaptureWire {
public:
    static constexpr std::size_t kHeaderBytes = 80;

    template <std::size_t FrameBytes>
    static EncodeResult encode(const ChunkFields& fields, const std::uint8_t* bytes,
                               std::size_t count,
                               std::array<std::uint8_t, FrameBytes>* frame,
                               std::size_t* written) {
        if (frame == nullptr || written == nullptr || (count != 0 && bytes == nullptr) ||
            count != fields.chunk_size || count > FrameBytes - kHeaderBytes ||
            !valid(fields)) return {};
        auto& out = *frame;
        put64(out.data() + 0, fields.session.device_instance);
        put64(out.data() + 8, fields.session.session_epoch);
        put64(out.data() + 16, fields.route.producer_instance);
        put64(out.data() + 24, fields.route.source_epoch);
        put64(out.data() + 32, fields.capture_ticket);
        put64(out.data() + 40, fields.chunk_sequence);
        put64(out.data() + 48, fields.chunk_offset);
        put32(out.data() + 56, fields.total_size);
        put32(out.data() + 60, fields.chunk_size);
        put64(out.data() + 64, fields.full_digest);
        for (std::size_t i = 0; i < count; ++i) out[kHeaderBytes + i] = bytes[i];
        put64(out.data() + 72, checksum(out.data(), 72, out.data() + kHeaderBytes, count));
        *written = kHeaderBytes + count;
        return {true};
    }

    template <std::size_t MaxChunkBytes>
    static DecodeResult decode(const std::uint8_t* frame, std::size_t frame_size,
                               DecodedChunk<MaxChunkBytes>* output) {
        if (frame == nullptr || output == nullptr || frame_size < kHeaderBytes) return {};
        ChunkFields fields{};
        fields.session = {get64(frame + 0), get64(frame + 8)};
        fields.route = {get64(frame + 16), get64(frame + 24)};
        fields.capture_ticket = get64(frame + 32);
        fields.chunk_sequence = get64(frame + 40);
        fields.chunk_offset = get64(frame + 48);
        fields.total_size = get32(frame + 56);
        fields.chunk_size = get32(frame + 60);
        fields.full_digest = get64(frame + 64);
        const std::size_t count = fields.chunk_size;
        if (!valid(fields) || count > MaxChunkBytes || frame_size != kHeaderBytes + count ||
            get64(frame + 72) != checksum(frame, 72, frame + kHeaderBytes, count)) return {};
        output->fields = fields;
        output->size = count;
        for (std::size_t i = 0; i < count; ++i) output->bytes[i] = frame[kHeaderBytes + i];
        return {true};
    }

private:
    static bool valid(const ChunkFields& fields) {
        return fields.session.device_instance != 0 && fields.session.session_epoch != 0 &&
               fields.route.producer_instance != 0 && fields.route.source_epoch != 0 &&
               fields.capture_ticket != 0 && fields.chunk_sequence != 0 &&
               fields.full_digest != 0 && fields.chunk_offset <= fields.total_size &&
               fields.chunk_size <= fields.total_size - fields.chunk_offset;
    }
    static void put32(std::uint8_t* out, std::uint32_t value) {
        for (std::size_t i = 0; i < 4; ++i) out[i] = static_cast<std::uint8_t>(value >> (8 * i));
    }
    static void put64(std::uint8_t* out, std::uint64_t value) {
        for (std::size_t i = 0; i < 8; ++i) out[i] = static_cast<std::uint8_t>(value >> (8 * i));
    }
    static std::uint32_t get32(const std::uint8_t* in) {
        std::uint32_t value = 0;
        for (std::size_t i = 0; i < 4; ++i) value |= static_cast<std::uint32_t>(in[i]) << (8 * i);
        return value;
    }
    static std::uint64_t get64(const std::uint8_t* in) {
        std::uint64_t value = 0;
        for (std::size_t i = 0; i < 8; ++i) value |= static_cast<std::uint64_t>(in[i]) << (8 * i);
        return value;
    }
    static std::uint64_t checksum(const std::uint8_t* header, std::size_t header_size,
                                  const std::uint8_t* bytes, std::size_t size) {
        std::uint64_t state = 1469598103934665603ULL;
        for (std::size_t i = 0; i < header_size; ++i) { state ^= header[i]; state *= 1099511628211ULL; }
        for (std::size_t i = 0; i < size; ++i) { state ^= bytes[i]; state *= 1099511628211ULL; }
        return state == 0 ? 1 : state;
    }
};

}  // namespace substrate::bcc32::external_body_capture_wire

namespace substrate::bcc32::external_body_capture_assembler {

struct DeviceSession {
    std::uint64_t device_instance = 0;
    std::uint64_t session_epoch = 0;
};

struct ReturnRoute {
    std::uint64_t producer_instance = 0;
    std::uint64_t source_epoch = 0;
};

template <std::size_t MaxBytes>
struct RawPayload {
    std::array<std::uint8_t, MaxBytes> bytes{};
    std::size_t size = 0;
};

template <std::size_t MaxBytes>
class ExternalBodyCaptureAssembler {
public:
    ExternalBodyCaptureAssembler(DeviceSession session, ReturnRoute route)
        : session_(session), route_(route) {}

    bool begin(DeviceSession session, ReturnRoute route, std::uint64_t ticket,
               std::size_t total_bytes, std::uint64_t digest) {
        if (begun_ || ticket == 0 || total_bytes > MaxBytes || digest == 0 ||
            !same_session(session_, session) || !same_route(route_, route)) {
            return false;
        }
        ticket_ = ticket;
        total_bytes_ = total_bytes;
        digest_ = digest;
        begun_ = true;
        return true;
    }

    bool append(DeviceSession session, ReturnRoute route, std::uint64_t ticket,
                std::uint64_t sequence, std::size_t offset,
                const std::uint8_t* bytes, std::size_t size, bool final) {
        if (!begun_ || complete_ || bytes == nullptr ||
            !same_session(session_, session) || !same_route(route_, route) ||
            ticket != ticket_ || sequence != next_sequence_ || offset != payload_.size ||
            size > total_bytes_ - payload_.size ||
            final != (payload_.size + size == total_bytes_)) {
            return false;
        }
        for (std::size_t i = 0; i < size; ++i) payload_.bytes[payload_.size + i] = bytes[i];
        payload_.size += size;
        ++next_sequence_;
        if (!final) return true;
        if (full_digest(session_, route_, ticket_, payload_.size, payload_.bytes.data()) != digest_) {
            return false;
        }
        complete_ = true;
        return true;
    }

    bool consume(RawPayload<MaxBytes>* result) {
        if (!complete_ || consumed_ || result == nullptr) return false;
        *result = payload_;
        consumed_ = true;
        return true;
    }

    static std::uint64_t full_digest(DeviceSession session, ReturnRoute route,
                                     std::uint64_t ticket, std::size_t size,
                                     const std::uint8_t* bytes) {
        if (session.device_instance == 0 || session.session_epoch == 0 ||
            route.producer_instance == 0 || route.source_epoch == 0 || ticket == 0 ||
            (size != 0 && bytes == nullptr)) return 0;
        std::uint64_t state = 1469598103934665603ULL;
        mix(&state, session.device_instance);
        mix(&state, session.session_epoch);
        mix(&state, route.producer_instance);
        mix(&state, route.source_epoch);
        mix(&state, ticket);
        mix(&state, static_cast<std::uint64_t>(size));
        for (std::size_t i = 0; i < size; ++i) mix(&state, bytes[i]);
        return state == 0 ? 1 : state;
    }

private:
    static bool same_session(DeviceSession left, DeviceSession right) {
        return left.device_instance == right.device_instance &&
               left.session_epoch == right.session_epoch;
    }
    static bool same_route(ReturnRoute left, ReturnRoute right) {
        return left.producer_instance == right.producer_instance &&
               left.source_epoch == right.source_epoch;
    }
    static void mix(std::uint64_t* state, std::uint64_t value) {
        *state ^= value;
        *state *= 1099511628211ULL;
    }

    DeviceSession session_{};
    ReturnRoute route_{};
    RawPayload<MaxBytes> payload_{};
    std::uint64_t ticket_ = 0;
    std::uint64_t digest_ = 0;
    std::size_t total_bytes_ = 0;
    std::uint64_t next_sequence_ = 1;
    bool begun_ = false;
    bool complete_ = false;
    bool consumed_ = false;
};

}  // namespace substrate::bcc32::external_body_capture_assembler

namespace bcc32 {

namespace verified_capture_assembler =
    substrate::bcc32::external_body_capture_assembler;
namespace verified_capture_wire = substrate::bcc32::external_body_capture_wire;

template <std::size_t MaxBytes = 256>
struct ExternalBodyVerifiedRawCapture {
    std::array<std::uint8_t, MaxBytes> bytes{};
    std::size_t size = 0;
    std::uint64_t device_instance = 0;
    std::uint64_t session_epoch = 0;
    std::uint64_t route_digest = 0;
    std::uint64_t capture_id = 0;
    std::uint64_t capture_ticket = 0;
    std::uint64_t attempt_id = 0;
    std::uint64_t action_envelope = 0;
    std::uint64_t command_digest = 0;
    std::uint64_t capture_device_tick = 0;
    std::uint64_t raw_payload_digest = 0;
    std::uint64_t manifest_verification_commitment = 0;
    bool signature_verified = false;
    bool physical_source_proven = false;
    bool physical_consequence_proven = false;
};

template <std::size_t MaxBytes = 256, std::size_t MaxChunkBytes = 256>
class ExternalBodyVerifiedCaptureAssembly {
    using AssemblerNamespace =
        verified_capture_assembler::ExternalBodyCaptureAssembler<MaxBytes>;

public:
    ExternalBodyVerifiedCaptureAssembly(
        const ExternalBodyCaptureManifestVerificationReceipt& verified_manifest,
        std::uint64_t route_producer_instance,
        std::uint64_t route_source_epoch)
        : manifest_(verified_manifest),
          session_{verified_manifest.manifest.device_instance,
                   verified_manifest.manifest.session_epoch},
          route_{route_producer_instance, route_source_epoch},
          assembler_(session_, route_) {
        const auto& manifest = manifest_.manifest;
        valid_ = manifest_.complete && manifest_.signature_verified &&
                 !manifest_.physical_source_proven &&
                 !manifest_.physical_consequence_proven &&
                 manifest_.verification_commitment != 0 &&
                 route_producer_instance != 0 && route_source_epoch != 0 &&
                 manifest.route_digest == route_commitment(
                     route_producer_instance, route_source_epoch) &&
                 manifest.total_payload_bytes <= MaxBytes;
        if (valid_) {
            valid_ = assembler_.begin(session_, route_, manifest.capture_ticket,
                                      manifest.total_payload_bytes,
                                      manifest.full_raw_payload_digest);
        }
    }

    bool append_frame(const std::uint8_t* frame_bytes,
                      std::size_t frame_byte_count) {
        if (!valid_ || complete_) return false;
        verified_capture_wire::DecodedChunk<MaxChunkBytes> chunk{};
        const auto decoded = verified_capture_wire::ExternalBodyCaptureWire::decode(
            frame_bytes, frame_byte_count, &chunk);
        const auto& manifest = manifest_.manifest;
        if (!decoded.ok ||
            chunk.fields.session.device_instance != manifest.device_instance ||
            chunk.fields.session.session_epoch != manifest.session_epoch ||
            chunk.fields.route.producer_instance != route_.producer_instance ||
            chunk.fields.route.source_epoch != route_.source_epoch ||
            chunk.fields.capture_ticket != manifest.capture_ticket ||
            chunk.fields.total_size != manifest.total_payload_bytes ||
            chunk.fields.full_digest != manifest.full_raw_payload_digest) {
            return false;
        }
        const bool final = chunk.fields.chunk_offset + chunk.size ==
                           manifest.total_payload_bytes;
        if (!assembler_.append(session_, route_, manifest.capture_ticket,
                               chunk.fields.chunk_sequence,
                               static_cast<std::size_t>(chunk.fields.chunk_offset),
                               chunk.bytes.data(), chunk.size, final)) {
            return false;
        }
        complete_ = final;
        return true;
    }

    bool consume(ExternalBodyVerifiedRawCapture<MaxBytes>* result) {
        if (!valid_ || !complete_ || consumed_ || result == nullptr) return false;
        verified_capture_assembler::RawPayload<MaxBytes> payload{};
        if (!assembler_.consume(&payload)) return false;
        result->bytes = payload.bytes;
        result->size = payload.size;
        result->device_instance = manifest_.manifest.device_instance;
        result->session_epoch = manifest_.manifest.session_epoch;
        result->route_digest = manifest_.manifest.route_digest;
        result->capture_id = manifest_.manifest.capture_ticket;
        result->capture_ticket = manifest_.manifest.capture_ticket;
        result->attempt_id = manifest_.manifest.attempt_id;
        result->action_envelope = manifest_.manifest.action_envelope;
        result->command_digest = manifest_.manifest.accepted_command_digest;
        result->capture_device_tick = manifest_.manifest.capture_device_tick;
        result->raw_payload_digest =
            ExternalBodyIngressQueue<1, MaxBytes>::raw_payload_digest(
                payload.bytes.data(), payload.size);
        result->manifest_verification_commitment =
            manifest_.verification_commitment;
        result->signature_verified = true;
        consumed_ = true;
        return true;
    }

    bool valid() const { return valid_; }
    bool complete() const { return complete_; }

    static std::uint64_t route_commitment(std::uint64_t producer_instance,
                                          std::uint64_t source_epoch) {
        if (producer_instance == 0 || source_epoch == 0) return 0;
        std::uint64_t state = 1469598103934665603ULL;
        state ^= producer_instance;
        state *= 1099511628211ULL;
        state ^= source_epoch;
        state *= 1099511628211ULL;
        return state == 0 ? 1 : state;
    }

private:
    using Session =
        verified_capture_assembler::DeviceSession;
    using Route =
        verified_capture_assembler::ReturnRoute;

    ExternalBodyCaptureManifestVerificationReceipt manifest_{};
    Session session_{};
    Route route_{};
    AssemblerNamespace assembler_;
    bool valid_ = false;
    bool complete_ = false;
    bool consumed_ = false;
};

}  // namespace bcc32
