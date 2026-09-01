#pragma once

#include "bcc32_external_body_ingress_queue.cuh"
#include "bcc32_external_body_stream_decoder.cuh"

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define BCC32_STREAM_PUMP_HD __host__ __device__
#else
#define BCC32_STREAM_PUMP_HD
#endif

namespace bcc32 {

enum class ExternalBodyPumpStatus : std::uint8_t {
    kProgress = 1,
    kWouldBlock = 2,
    kIngressFull = 3,
    kEndOfStream = 4,
    kTransportFault = 5,
    kProtocolFault = 6,
};

struct ExternalBodyReadResult {
    std::size_t byte_count = 0;
    bool would_block = false;
    bool end_of_stream = false;
    bool fault = false;
};

template <typename Reader,
          std::size_t QueueCapacity = 8,
          std::size_t MaxPayloadBytes = 4096,
          std::size_t ReadChunkBytes = 256>
class ExternalBodyStreamPump {
    static_assert(ReadChunkBytes > 0, "read chunk must be positive");

public:
    using Queue = ExternalBodyIngressQueue<QueueCapacity, MaxPayloadBytes>;
    using Decoder = ExternalBodyStreamDecoder<MaxPayloadBytes>;

    BCC32_STREAM_PUMP_HD explicit ExternalBodyStreamPump(Reader reader)
        : reader_(reader) {}

    BCC32_STREAM_PUMP_HD ExternalBodyPumpStatus poll() {
        if (terminal_fault_) {
            return ExternalBodyPumpStatus::kTransportFault;
        }
        if (decoder_.faulted()) {
            return ExternalBodyPumpStatus::kProtocolFault;
        }
        if (!publish_ready_message()) {
            return ExternalBodyPumpStatus::kIngressFull;
        }
        if (queue_.size() == queue_.capacity()) {
            return ExternalBodyPumpStatus::kIngressFull;
        }

        std::uint8_t bytes[ReadChunkBytes]{};
        const ExternalBodyReadResult read = reader_.read(bytes, sizeof(bytes));
        if (read.fault || read.byte_count > sizeof(bytes) ||
            (read.byte_count != 0 && (read.would_block || read.end_of_stream))) {
            terminal_fault_ = true;
            return ExternalBodyPumpStatus::kTransportFault;
        }
        if (read.byte_count == 0) {
            if (read.end_of_stream) {
                return ExternalBodyPumpStatus::kEndOfStream;
            }
            return ExternalBodyPumpStatus::kWouldBlock;
        }

        std::size_t offset = 0;
        while (offset < read.byte_count) {
            const auto result = decoder_.feed(bytes + offset, read.byte_count - offset);
            offset += result.accepted_bytes;
            total_transport_bytes_ += result.accepted_bytes;
            if (result.faulted) {
                return ExternalBodyPumpStatus::kProtocolFault;
            }
            if (result.message_ready) {
                if (!publish_ready_message()) {
                    if (offset != read.byte_count) {
                        terminal_fault_ = true;
                        return ExternalBodyPumpStatus::kTransportFault;
                    }
                    return ExternalBodyPumpStatus::kIngressFull;
                }
            }
            if (result.accepted_bytes == 0) {
                terminal_fault_ = true;
                return ExternalBodyPumpStatus::kTransportFault;
            }
        }
        return ExternalBodyPumpStatus::kProgress;
    }

    BCC32_STREAM_PUMP_HD const Queue& queue() const { return queue_; }
    BCC32_STREAM_PUMP_HD Queue& queue() { return queue_; }
    BCC32_STREAM_PUMP_HD const Decoder& decoder() const { return decoder_; }
    BCC32_STREAM_PUMP_HD std::uint64_t total_transport_bytes() const {
        return total_transport_bytes_;
    }
    BCC32_STREAM_PUMP_HD std::uint64_t published_messages() const {
        return published_messages_;
    }

private:
    BCC32_STREAM_PUMP_HD bool publish_ready_message() {
        if (!decoder_.ready()) {
            return true;
        }
        typename Decoder::MessageView view{};
        if (!decoder_.peek(&view)) {
            return false;
        }
        ExternalBodyWireFrame frame{};
        for (std::size_t index = 0; index < kExternalBodyWireFrameBytes; ++index) {
            frame.bytes[index] = view.frame[index];
        }
        if (!queue_.publish(frame, view.payload, view.payload_bytes)) {
            return false;
        }
        if (!decoder_.consume(view.ticket, view.message_digest)) {
            terminal_fault_ = true;
            return false;
        }
        ++published_messages_;
        return true;
    }

    Reader reader_;
    Decoder decoder_{};
    Queue queue_{};
    std::uint64_t total_transport_bytes_ = 0;
    std::uint64_t published_messages_ = 0;
    bool terminal_fault_ = false;
};

}  // namespace bcc32

#undef BCC32_STREAM_PUMP_HD
