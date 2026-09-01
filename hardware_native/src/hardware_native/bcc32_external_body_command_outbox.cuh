#pragma once

#include "bcc32_external_body_command_wire.cuh"

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define BCC32_COMMAND_OUTBOX_HD __host__ __device__
#else
#define BCC32_COMMAND_OUTBOX_HD
#endif

namespace bcc32 {

struct ExternalBodyWriteResult {
    std::size_t byte_count = 0;
    bool would_block = false;
    bool fault = false;
};

enum class ExternalBodyOutboxStatus : std::uint8_t {
    kProgress = 1,
    kWouldBlock = 2,
    kEmpty = 3,
    kTransportFault = 4,
};

struct ExternalBodyTransportWriteReceipt {
    bool complete = false;
    bool transport_written_only = true;
    bool device_applied = false;
    std::uint64_t outbox_ticket = 0;
    std::uint64_t route_sequence = 0;
    std::uint64_t attempt_id = 0;
    std::uint64_t action_envelope = 0;
    std::uint64_t command_digest = 0;
    std::uint64_t frame_digest = 0;
};

template <std::size_t Capacity = 8>
class ExternalBodyCommandOutbox {
    static_assert(Capacity > 0, "outbox capacity must be positive");

public:
    BCC32_COMMAND_OUTBOX_HD bool enqueue(
        const ExternalBodyCommandFrame& frame) {
        if (faulted_ || count_ == Capacity) {
            ++rejected_enqueues_;
            return false;
        }
        ExternalBodyCommandIssue issue{};
        if (!ExternalBodyCommandWire::decode(
                frame.bytes, sizeof(frame.bytes), &issue) ||
            issue.route_sequence != last_enqueued_sequence_ + 1 ||
            issue.attempt_id <= last_enqueued_attempt_) {
            ++rejected_enqueues_;
            return false;
        }
        Slot& slot = slots_[tail_];
        for (std::size_t index = 0; index < sizeof(frame.bytes); ++index) {
            slot.frame.bytes[index] = frame.bytes[index];
        }
        slot.issue = issue;
        slot.ticket = ++last_ticket_;
        slot.frame_digest = digest(frame.bytes, sizeof(frame.bytes), slot.ticket);
        slot.written_bytes = 0;
        slot.occupied = true;
        tail_ = (tail_ + 1U) % Capacity;
        ++count_;
        last_enqueued_sequence_ = issue.route_sequence;
        last_enqueued_attempt_ = issue.attempt_id;
        return true;
    }

    template <typename Writer>
    BCC32_COMMAND_OUTBOX_HD ExternalBodyOutboxStatus flush_one(
        Writer* writer,
        ExternalBodyTransportWriteReceipt* receipt = nullptr) {
        if (receipt != nullptr) {
            *receipt = ExternalBodyTransportWriteReceipt{};
        }
        if (faulted_ || writer == nullptr) {
            faulted_ = true;
            return ExternalBodyOutboxStatus::kTransportFault;
        }
        if (count_ == 0 || !slots_[head_].occupied) {
            return ExternalBodyOutboxStatus::kEmpty;
        }
        Slot& slot = slots_[head_];
        const std::size_t remaining = sizeof(slot.frame.bytes) - slot.written_bytes;
        const ExternalBodyWriteResult result = writer->write(
            slot.frame.bytes + slot.written_bytes, remaining);
        if (result.fault || result.byte_count > remaining ||
            (result.byte_count != 0 && result.would_block)) {
            faulted_ = true;
            return ExternalBodyOutboxStatus::kTransportFault;
        }
        if (result.byte_count == 0) {
            if (result.would_block) {
                return ExternalBodyOutboxStatus::kWouldBlock;
            }
            faulted_ = true;
            return ExternalBodyOutboxStatus::kTransportFault;
        }
        slot.written_bytes += result.byte_count;
        total_written_bytes_ += result.byte_count;
        if (slot.written_bytes != sizeof(slot.frame.bytes)) {
            return ExternalBodyOutboxStatus::kProgress;
        }
        if (slot.frame_digest != digest(
                slot.frame.bytes, sizeof(slot.frame.bytes), slot.ticket)) {
            faulted_ = true;
            return ExternalBodyOutboxStatus::kTransportFault;
        }
        if (receipt != nullptr) {
            receipt->complete = true;
            receipt->outbox_ticket = slot.ticket;
            receipt->route_sequence = slot.issue.route_sequence;
            receipt->attempt_id = slot.issue.attempt_id;
            receipt->action_envelope = slot.issue.action_envelope;
            receipt->command_digest = slot.issue.command_digest;
            receipt->frame_digest = slot.frame_digest;
        }
        last_written_sequence_ = slot.issue.route_sequence;
        last_written_attempt_ = slot.issue.attempt_id;
        slot = Slot{};
        head_ = (head_ + 1U) % Capacity;
        --count_;
        ++written_frames_;
        return ExternalBodyOutboxStatus::kProgress;
    }

    BCC32_COMMAND_OUTBOX_HD std::size_t size() const { return count_; }
    BCC32_COMMAND_OUTBOX_HD bool faulted() const { return faulted_; }
    BCC32_COMMAND_OUTBOX_HD std::uint64_t written_frames() const {
        return written_frames_;
    }
    BCC32_COMMAND_OUTBOX_HD std::uint64_t total_written_bytes() const {
        return total_written_bytes_;
    }
    BCC32_COMMAND_OUTBOX_HD std::uint64_t last_written_sequence() const {
        return last_written_sequence_;
    }
    BCC32_COMMAND_OUTBOX_HD std::uint64_t rejected_enqueues() const {
        return rejected_enqueues_;
    }

private:
    struct Slot {
        ExternalBodyCommandFrame frame{};
        ExternalBodyCommandIssue issue{};
        std::uint64_t ticket = 0;
        std::uint64_t frame_digest = 0;
        std::size_t written_bytes = 0;
        bool occupied = false;
    };

    BCC32_COMMAND_OUTBOX_HD static std::uint64_t digest(
        const std::uint8_t* bytes,
        std::size_t byte_count,
        std::uint64_t ticket) {
        std::uint64_t state = 1469598103934665603ULL;
        for (std::size_t index = 0; index < byte_count; ++index) {
            state ^= bytes[index];
            state *= 1099511628211ULL;
        }
        for (unsigned shift = 0; shift < 64U; shift += 8U) {
            state ^= static_cast<std::uint8_t>(ticket >> shift);
            state *= 1099511628211ULL;
        }
        return state == 0 ? 1 : state;
    }

    Slot slots_[Capacity]{};
    std::size_t head_ = 0;
    std::size_t tail_ = 0;
    std::size_t count_ = 0;
    std::uint64_t last_ticket_ = 0;
    std::uint64_t last_enqueued_sequence_ = 0;
    std::uint64_t last_enqueued_attempt_ = 0;
    std::uint64_t last_written_sequence_ = 0;
    std::uint64_t last_written_attempt_ = 0;
    std::uint64_t written_frames_ = 0;
    std::uint64_t total_written_bytes_ = 0;
    std::uint64_t rejected_enqueues_ = 0;
    bool faulted_ = false;
};

}  // namespace bcc32

#undef BCC32_COMMAND_OUTBOX_HD
