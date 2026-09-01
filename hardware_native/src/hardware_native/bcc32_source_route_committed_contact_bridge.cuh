#pragma once

#include "hardware_native/bcc32_source_route_contact_adapter.cuh"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>

namespace substrate::bcc32::source_route_contact {

enum class CommittedBridgeFailure : std::uint8_t {
    none,
    route_rejected,
    committed_payload_unreadable,
};

// The consumer may be PersistentKernel or another raw-contact boundary with
// the same present_raw signature.  This bridge deliberately accepts no label,
// reward, semantic selector, or host-selected target.
template <typename Consumer>
concept RawContactConsumer = requires(Consumer& consumer,
                                      std::span<const Word> contact) {
    { consumer.present_raw(contact) } -> std::same_as<void>;
};

// A committed-only handoff: fragmented source-route bytes cannot reach the
// raw consumer until SourceRouteContactAdapter has committed the complete,
// sequenced, digest-checked contact.  The adapter remains receipt-shaped and
// caller-identity is still not physical authentication.
template <RawContactConsumer Consumer>
class SourceRouteCommittedContactBridge final {
public:
    explicit SourceRouteCommittedContactBridge(Consumer& consumer) noexcept
        : consumer_(consumer) {}

    [[nodiscard]] bool accept(const SourceRouteFrame& frame) {
        if (!adapter_.accept(frame)) {
            last_failure_ = CommittedBridgeFailure::route_rejected;
            return false;
        }
        if (adapter_.receipt().status != SourceRouteStatus::committed) {
            return true;
        }

        const ConsumerOnlyResult committed = adapter_.consumer_only();
        if (committed.words.empty()) {
            last_failure_ = CommittedBridgeFailure::committed_payload_unreadable;
            return false;
        }
        consumer_.present_raw(std::span<const Word>(committed.words.data(),
                                                     committed.words.size()));
        last_receipt_ = committed.receipt;
        ++delivery_count_;
        last_failure_ = CommittedBridgeFailure::none;
        return true;
    }

    [[nodiscard]] SourceRouteReceipt receipt() const noexcept {
        return adapter_.receipt();
    }

    [[nodiscard]] SourceRouteReceipt last_delivered_receipt() const noexcept {
        return last_receipt_;
    }

    [[nodiscard]] std::size_t delivery_count() const noexcept {
        return delivery_count_;
    }

    [[nodiscard]] CommittedBridgeFailure last_failure() const noexcept {
        return last_failure_;
    }

private:
    Consumer& consumer_;
    SourceRouteContactAdapter adapter_{};
    SourceRouteReceipt last_receipt_{};
    std::size_t delivery_count_ = 0u;
    CommittedBridgeFailure last_failure_ = CommittedBridgeFailure::none;
};

}  // namespace substrate::bcc32::source_route_contact
