#pragma once

#include "bcc32_external_body_wire_protocol.cuh"

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define BCC32_INGRESS_QUEUE_HD __host__ __device__
#else
#define BCC32_INGRESS_QUEUE_HD
#endif

namespace bcc32 {

template <std::size_t Capacity = 8, std::size_t MaxPayloadBytes = 256>
class ExternalBodyIngressQueue {
    static_assert(Capacity > 0, "queue capacity must be positive");
    static_assert(MaxPayloadBytes > 0, "payload capacity must be positive");

public:
    struct View {
        const std::uint8_t* frame = nullptr;
        const std::uint8_t* payload = nullptr;
        std::uint32_t payload_bytes = 0;
        std::uint64_t route_sequence = 0;
        std::uint64_t item_ticket = 0;
        std::uint64_t item_digest = 0;
    };

    BCC32_INGRESS_QUEUE_HD bool publish(
        const ExternalBodyWireFrame& frame,
        const std::uint8_t* payload,
        std::size_t payload_bytes) {
        ExternalBodyWireEvent event{};
        if (count_ == Capacity ||
            !ExternalBodyWireProtocol::decode(
                frame.bytes, sizeof(frame.bytes), &event) ||
            event.route_sequence != last_published_sequence_ + 1 ||
            payload_bytes > MaxPayloadBytes ||
            (payload_bytes != 0 && payload == nullptr) ||
            event.raw_payload_bytes != payload_bytes ||
            (event.type == ExternalBodyWireType::kRawCapture &&
             raw_payload_digest(payload, payload_bytes) != event.raw_payload_digest) ||
            (event.type != ExternalBodyWireType::kRawCapture && payload_bytes != 0)) {
            ++rejected_publications_;
            return false;
        }

        Slot& slot = slots_[tail_];
        for (std::size_t index = 0; index < kExternalBodyWireFrameBytes; ++index) {
            slot.frame.bytes[index] = frame.bytes[index];
        }
        for (std::size_t index = 0; index < payload_bytes; ++index) {
            slot.payload[index] = payload[index];
        }
        slot.payload_bytes = static_cast<std::uint32_t>(payload_bytes);
        slot.route_sequence = event.route_sequence;
        slot.item_ticket = ++last_ticket_;
        slot.item_digest = digest(slot);
        slot.occupied = true;
        tail_ = (tail_ + 1U) % Capacity;
        ++count_;
        last_published_sequence_ = event.route_sequence;
        return true;
    }

    BCC32_INGRESS_QUEUE_HD bool peek(View* view) const {
        if (view == nullptr || count_ == 0 || !slots_[head_].occupied) {
            return false;
        }
        const Slot& slot = slots_[head_];
        if (slot.item_digest != digest(slot)) {
            return false;
        }
        view->frame = slot.frame.bytes;
        view->payload = slot.payload;
        view->payload_bytes = slot.payload_bytes;
        view->route_sequence = slot.route_sequence;
        view->item_ticket = slot.item_ticket;
        view->item_digest = slot.item_digest;
        return true;
    }

    BCC32_INGRESS_QUEUE_HD bool consume(std::uint64_t item_ticket,
                                        std::uint64_t item_digest) {
        if (count_ == 0 || !slots_[head_].occupied) {
            return false;
        }
        Slot& slot = slots_[head_];
        if (slot.item_ticket != item_ticket || slot.item_digest != item_digest ||
            slot.item_digest != digest(slot)) {
            return false;
        }
        last_consumed_sequence_ = slot.route_sequence;
        slot = Slot{};
        head_ = (head_ + 1U) % Capacity;
        --count_;
        return true;
    }

    BCC32_INGRESS_QUEUE_HD std::size_t size() const { return count_; }
    BCC32_INGRESS_QUEUE_HD std::size_t capacity() const { return Capacity; }
    BCC32_INGRESS_QUEUE_HD std::uint64_t last_published_sequence() const {
        return last_published_sequence_;
    }
    BCC32_INGRESS_QUEUE_HD std::uint64_t last_consumed_sequence() const {
        return last_consumed_sequence_;
    }
    BCC32_INGRESS_QUEUE_HD std::uint64_t rejected_publications() const {
        return rejected_publications_;
    }

    BCC32_INGRESS_QUEUE_HD static std::uint64_t raw_payload_digest(
        const std::uint8_t* bytes, std::size_t byte_count) {
        if (bytes == nullptr || byte_count == 0) {
            return 0;
        }
        return mix_bytes(1469598103934665603ULL, bytes, byte_count);
    }

private:
    struct Slot {
        ExternalBodyWireFrame frame{};
        std::uint8_t payload[MaxPayloadBytes]{};
        std::uint32_t payload_bytes = 0;
        std::uint64_t route_sequence = 0;
        std::uint64_t item_ticket = 0;
        std::uint64_t item_digest = 0;
        bool occupied = false;
    };

    BCC32_INGRESS_QUEUE_HD static std::uint64_t mix_bytes(
        std::uint64_t state, const std::uint8_t* bytes, std::size_t byte_count) {
        for (std::size_t index = 0; index < byte_count; ++index) {
            state ^= bytes[index];
            state *= 1099511628211ULL;
        }
        return state;
    }

    BCC32_INGRESS_QUEUE_HD static std::uint64_t digest(const Slot& slot) {
        std::uint64_t state = 1469598103934665603ULL;
        state = mix_bytes(state, slot.frame.bytes, sizeof(slot.frame.bytes));
        state = mix_bytes(state, slot.payload, slot.payload_bytes);
        for (unsigned shift = 0; shift < 64U; shift += 8U) {
            state ^= static_cast<std::uint8_t>(slot.item_ticket >> shift);
            state *= 1099511628211ULL;
            state ^= static_cast<std::uint8_t>(slot.route_sequence >> shift);
            state *= 1099511628211ULL;
        }
        return state == 0 ? 1 : state;
    }

    Slot slots_[Capacity]{};
    std::size_t head_ = 0;
    std::size_t tail_ = 0;
    std::size_t count_ = 0;
    std::uint64_t last_ticket_ = 0;
    std::uint64_t last_published_sequence_ = 0;
    std::uint64_t last_consumed_sequence_ = 0;
    std::uint64_t rejected_publications_ = 0;
};

}  // namespace bcc32

#undef BCC32_INGRESS_QUEUE_HD
