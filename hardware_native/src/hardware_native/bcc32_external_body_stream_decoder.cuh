#pragma once

#include "bcc32_external_body_wire_protocol.cuh"

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define BCC32_STREAM_DECODER_HD __host__ __device__
#else
#define BCC32_STREAM_DECODER_HD
#endif

namespace bcc32 {

template <std::size_t MaxPayloadBytes = 4096>
class ExternalBodyStreamDecoder {
    static_assert(MaxPayloadBytes > 0, "payload capacity must be positive");

public:
    struct FeedResult {
        std::size_t accepted_bytes = 0;
        bool message_ready = false;
        bool faulted = false;
    };

    struct MessageView {
        const ExternalBodyWireEvent* event = nullptr;
        const std::uint8_t* frame = nullptr;
        const std::uint8_t* payload = nullptr;
        std::uint32_t payload_bytes = 0;
        std::uint64_t ticket = 0;
        std::uint64_t message_digest = 0;
    };

    BCC32_STREAM_DECODER_HD FeedResult feed(const std::uint8_t* bytes,
                                            std::size_t byte_count) {
        FeedResult result{};
        if (faulted_ || ready_ || (byte_count != 0 && bytes == nullptr)) {
            result.message_ready = ready_;
            result.faulted = faulted_ || (byte_count != 0 && bytes == nullptr);
            return result;
        }

        while (result.accepted_bytes < byte_count && !ready_ && !faulted_) {
            if (frame_bytes_ < kExternalBodyWireFrameBytes) {
                frame_.bytes[frame_bytes_++] = bytes[result.accepted_bytes++];
                if (frame_bytes_ == kExternalBodyWireFrameBytes) {
                    if (!ExternalBodyWireProtocol::decode(
                            frame_.bytes, sizeof(frame_.bytes), &event_) ||
                        event_.route_sequence != last_sequence_ + 1 ||
                        event_.raw_payload_bytes > MaxPayloadBytes) {
                        faulted_ = true;
                        break;
                    }
                    expected_payload_bytes_ = event_.raw_payload_bytes;
                    if (expected_payload_bytes_ == 0) {
                        seal_message();
                    }
                }
                continue;
            }

            if (payload_bytes_ < expected_payload_bytes_) {
                payload_[payload_bytes_++] = bytes[result.accepted_bytes++];
                if (payload_bytes_ == expected_payload_bytes_) {
                    const std::uint64_t observed = payload_digest(
                        payload_, payload_bytes_);
                    const std::uint32_t words =
                        static_cast<std::uint32_t>((payload_bytes_ + 3U) / 4U);
                    if (observed != event_.raw_payload_digest ||
                        words != event_.raw_word_count) {
                        faulted_ = true;
                    } else {
                        seal_message();
                    }
                }
            }
        }
        result.message_ready = ready_;
        result.faulted = faulted_;
        return result;
    }

    BCC32_STREAM_DECODER_HD bool peek(MessageView* view) const {
        if (!ready_ || faulted_ || view == nullptr ||
            message_digest_ != calculate_message_digest()) {
            return false;
        }
        view->event = &event_;
        view->frame = frame_.bytes;
        view->payload = payload_;
        view->payload_bytes = payload_bytes_;
        view->ticket = ticket_;
        view->message_digest = message_digest_;
        return true;
    }

    BCC32_STREAM_DECODER_HD bool consume(std::uint64_t ticket,
                                         std::uint64_t message_digest) {
        if (!ready_ || faulted_ || ticket != ticket_ ||
            message_digest != message_digest_ ||
            message_digest_ != calculate_message_digest()) {
            return false;
        }
        last_sequence_ = event_.route_sequence;
        clear_message();
        return true;
    }

    BCC32_STREAM_DECODER_HD bool reset_fault() {
        if (!faulted_ || ready_) {
            return false;
        }
        clear_message();
        faulted_ = false;
        return true;
    }

    BCC32_STREAM_DECODER_HD bool ready() const { return ready_; }
    BCC32_STREAM_DECODER_HD bool faulted() const { return faulted_; }
    BCC32_STREAM_DECODER_HD std::uint64_t last_sequence() const {
        return last_sequence_;
    }

    BCC32_STREAM_DECODER_HD static std::uint64_t payload_digest(
        const std::uint8_t* bytes, std::size_t byte_count) {
        if (bytes == nullptr || byte_count == 0) {
            return 0;
        }
        std::uint64_t state = 1469598103934665603ULL;
        for (std::size_t index = 0; index < byte_count; ++index) {
            state ^= bytes[index];
            state *= 1099511628211ULL;
        }
        return state == 0 ? 1 : state;
    }

private:
    BCC32_STREAM_DECODER_HD static std::uint64_t mix_bytes(
        std::uint64_t state, const std::uint8_t* bytes, std::size_t byte_count) {
        for (std::size_t index = 0; index < byte_count; ++index) {
            state ^= bytes[index];
            state *= 1099511628211ULL;
        }
        return state;
    }

    BCC32_STREAM_DECODER_HD std::uint64_t calculate_message_digest() const {
        std::uint64_t state = 1469598103934665603ULL;
        state = mix_bytes(state, frame_.bytes, sizeof(frame_.bytes));
        state = mix_bytes(state, payload_, payload_bytes_);
        for (unsigned shift = 0; shift < 64U; shift += 8U) {
            state ^= static_cast<std::uint8_t>(ticket_ >> shift);
            state *= 1099511628211ULL;
        }
        return state == 0 ? 1 : state;
    }

    BCC32_STREAM_DECODER_HD void seal_message() {
        ready_ = true;
        ticket_ = next_ticket_++;
        message_digest_ = calculate_message_digest();
    }

    BCC32_STREAM_DECODER_HD void clear_message() {
        frame_ = ExternalBodyWireFrame{};
        event_ = ExternalBodyWireEvent{};
        for (std::size_t index = 0; index < MaxPayloadBytes; ++index) {
            payload_[index] = 0;
        }
        frame_bytes_ = 0;
        payload_bytes_ = 0;
        expected_payload_bytes_ = 0;
        ticket_ = 0;
        message_digest_ = 0;
        ready_ = false;
    }

    ExternalBodyWireFrame frame_{};
    ExternalBodyWireEvent event_{};
    std::uint8_t payload_[MaxPayloadBytes]{};
    std::size_t frame_bytes_ = 0;
    std::size_t payload_bytes_ = 0;
    std::size_t expected_payload_bytes_ = 0;
    std::uint64_t next_ticket_ = 1;
    std::uint64_t ticket_ = 0;
    std::uint64_t message_digest_ = 0;
    std::uint64_t last_sequence_ = 0;
    bool ready_ = false;
    bool faulted_ = false;
};

}  // namespace bcc32

#undef BCC32_STREAM_DECODER_HD
