#pragma once

#include "hardware_native/bcc32_logical_contact_transport.cuh"

#include <cstdint>
#include <utility>
#include <vector>

namespace substrate::bcc32::source_route_contact {

using Word = logical_contact_transport::Word;
using LogicalContactReceipt = logical_contact_transport::LogicalContactReceipt;

// This is a receipt-shaped adapter boundary.  The producer is expected to
// obtain these identity fields from a body/validator route; this type does not
// authenticate a caller and therefore cannot, by itself, prove physical
// provenance.  It does prevent a consumer from treating an unsequenced,
// source-swapped, or caller-rehashed packet as a committed contact.
struct SourceRouteFrame final {
    std::uint64_t source_epoch = 0u;
    std::uint64_t stream_id = 0u;
    std::uint64_t contact_sequence = 0u;
    std::uint64_t fragment_sequence = 0u;
    std::uint32_t codec_id = 0u;
    std::uint32_t total_words = 0u;
    std::uint32_t offset_words = 0u;
    std::uint32_t word_count = 0u;
    bool is_final_fragment = false;
    // Validator-issued expected digest.  The adapter computes the observed
    // digest from payload bytes through the resident transport assembler.
    std::uint64_t validator_payload_hash = 0u;
    const Word* payload = nullptr;
};

enum class SourceRouteFailure : std::uint8_t {
    none,
    invalid_identity,
    invalid_extent,
    source_identity_mismatch,
    contact_sequence_gap,
    replayed_contact,
    fragment_sequence_gap,
    codec_mismatch,
    validator_digest_mismatch,
    transport_rejected,
    no_committed_contact,
    fault_latched,
};

enum class SourceRouteStatus : std::uint8_t {
    empty,
    assembling,
    committed,
    faulted,
};

struct SourceRouteReceipt final {
    std::uint64_t source_epoch = 0u;
    std::uint64_t stream_id = 0u;
    std::uint64_t contact_sequence = 0u;
    std::uint64_t first_fragment_sequence = 0u;
    std::uint64_t final_fragment_sequence = 0u;
    std::uint32_t codec_id = 0u;
    std::uint64_t validator_payload_hash = 0u;
    LogicalContactReceipt logical_contact{};
    SourceRouteStatus status = SourceRouteStatus::empty;
    SourceRouteFailure failure = SourceRouteFailure::none;
};

struct ConsumerOnlyResult final {
    SourceRouteReceipt receipt{};
    std::vector<Word> words;
};

class SourceRouteContactAdapter final {
public:
    [[nodiscard]] bool accept(const SourceRouteFrame& frame) noexcept {
        if (receipt_.status == SourceRouteStatus::faulted) {
            receipt_.failure = SourceRouteFailure::fault_latched;
            return false;
        }
        if (frame.source_epoch == 0u || frame.stream_id == 0u ||
            frame.contact_sequence == 0u || frame.fragment_sequence == 0u ||
            frame.codec_id == 0u || frame.validator_payload_hash == 0u) {
            return fail(SourceRouteFailure::invalid_identity);
        }
        if (frame.word_count != 0u && frame.payload == nullptr) {
            return fail(SourceRouteFailure::invalid_extent);
        }
        if (frame.total_words == 0u ||
            frame.total_words > logical_contact_transport::kMaximumLogicalContactWords ||
            frame.offset_words > frame.total_words ||
            frame.word_count > frame.total_words - frame.offset_words) {
            return fail(SourceRouteFailure::invalid_extent);
        }

        if (source_epoch_ == 0u) {
            source_epoch_ = frame.source_epoch;
            stream_id_ = frame.stream_id;
        } else if (frame.source_epoch != source_epoch_ || frame.stream_id != stream_id_) {
            return fail(SourceRouteFailure::source_identity_mismatch);
        }
        if (last_fragment_sequence_ != 0u &&
            frame.fragment_sequence != last_fragment_sequence_ + 1u) {
            return fail(frame.fragment_sequence <= last_fragment_sequence_
                            ? SourceRouteFailure::fragment_sequence_gap
                            : SourceRouteFailure::fragment_sequence_gap);
        }

        if (active_contact_sequence_ == 0u) {
            if (frame.contact_sequence < next_contact_sequence_) {
                return fail(SourceRouteFailure::replayed_contact);
            }
            if (frame.contact_sequence != next_contact_sequence_) {
                return fail(SourceRouteFailure::contact_sequence_gap);
            }
            if (frame.offset_words != 0u) {
                return fail(SourceRouteFailure::invalid_extent);
            }
            logical_contact_transport_.reset();
            active_contact_sequence_ = frame.contact_sequence;
            active_codec_id_ = frame.codec_id;
            active_total_words_ = frame.total_words;
            active_validator_payload_hash_ = frame.validator_payload_hash;
            receipt_.status = SourceRouteStatus::assembling;
            receipt_.source_epoch = source_epoch_;
            receipt_.stream_id = stream_id_;
            receipt_.contact_sequence = active_contact_sequence_;
            receipt_.first_fragment_sequence = frame.fragment_sequence;
            receipt_.codec_id = active_codec_id_;
            receipt_.validator_payload_hash = active_validator_payload_hash_;
        } else if (frame.contact_sequence != active_contact_sequence_) {
            return fail(frame.contact_sequence < active_contact_sequence_
                            ? SourceRouteFailure::replayed_contact
                            : SourceRouteFailure::contact_sequence_gap);
        } else if (frame.codec_id != active_codec_id_ ||
                   frame.total_words != active_total_words_) {
            return fail(SourceRouteFailure::codec_mismatch);
        } else if (frame.validator_payload_hash != active_validator_payload_hash_) {
            return fail(SourceRouteFailure::validator_digest_mismatch);
        }

        const logical_contact_transport::LogicalContactFragment fragment{
            active_contact_sequence_,
            active_validator_payload_hash_,
            active_total_words_,
            frame.offset_words,
            frame.word_count,
            frame.is_final_fragment,
        };
        const auto result = logical_contact_transport_.accept(
            fragment, frame.payload, frame.word_count, frame.fragment_sequence);
        if (result == logical_contact_transport::AcceptResult::rejected) {
            return fail(SourceRouteFailure::transport_rejected);
        }
        last_fragment_sequence_ = frame.fragment_sequence;
        receipt_.logical_contact = logical_contact_transport_.logical_receipt();
        if (result == logical_contact_transport::AcceptResult::committed) {
            receipt_.status = SourceRouteStatus::committed;
            receipt_.final_fragment_sequence = frame.fragment_sequence;
            receipt_.failure = SourceRouteFailure::none;
            ++next_contact_sequence_;
            active_contact_sequence_ = 0u;
        }
        return true;
    }

    [[nodiscard]] SourceRouteReceipt receipt() const noexcept { return receipt_; }

    // This is deliberately a passive, consumer-only copy.  It does not expose
    // a writer, semantic label, action target, reward, or resident state view.
    [[nodiscard]] ConsumerOnlyResult consumer_only() const {
        if (receipt_.status != SourceRouteStatus::committed) {
            return ConsumerOnlyResult{receipt_, {}};
        }
        std::vector<Word> words;
        words.reserve(logical_contact_transport_.completed_word_count());
        for (std::uint32_t index = 0u;
             index < logical_contact_transport_.completed_word_count(); ++index) {
            Word word = 0u;
            if (!logical_contact_transport_.completed_word_at(index, word)) {
                return ConsumerOnlyResult{receipt_, {}};
            }
            words.push_back(word);
        }
        return ConsumerOnlyResult{receipt_, std::move(words)};
    }

private:
    [[nodiscard]] bool fail(SourceRouteFailure failure) noexcept {
        receipt_.status = SourceRouteStatus::faulted;
        receipt_.failure = failure;
        return false;
    }

    logical_contact_transport::ResidentLogicalContactAssembler logical_contact_transport_{};
    SourceRouteReceipt receipt_{};
    std::uint64_t source_epoch_ = 0u;
    std::uint64_t stream_id_ = 0u;
    std::uint64_t next_contact_sequence_ = 1u;
    std::uint64_t last_fragment_sequence_ = 0u;
    std::uint64_t active_contact_sequence_ = 0u;
    std::uint32_t active_codec_id_ = 0u;
    std::uint32_t active_total_words_ = 0u;
    std::uint64_t active_validator_payload_hash_ = 0u;
};

}  // namespace substrate::bcc32::source_route_contact
