#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace substrate::bcc32::logical_contact_transport {

// This is a bounded transport boundary, not a source-provenance authority.  Its
// only job is to make one logical contact survive a lossless ordered fragment
// route without letting a malformed route produce a resident completion.
using Word = std::uint32_t;

inline constexpr std::uint32_t kMaximumLogicalContactWords = 1024u;

enum class TransportStatus : std::uint8_t {
    empty,
    assembling,
    committed,
    faulted,
};

enum class TransportFailure : std::uint8_t {
    none,
    null_payload,
    payload_size_mismatch,
    invalid_extent,
    missing_first_fragment,
    identity_mismatch,
    total_extent_mismatch,
    declared_hash_mismatch,
    reordered_or_duplicate_fragment,
    nonmonotonic_fragment_sequence,
    payload_hash_mismatch,
    replayed_contact,
    commit_counter_exhausted,
    duplicate_after_commit,
    fault_latched,
};

enum class AcceptResult : std::uint8_t {
    accepted,
    committed,
    rejected,
};

struct LogicalContactFragment final {
    // Stable across every fragment of one logical contact.  The caller owns
    // the scope in which this identity is unique.
    std::uint64_t logical_contact_id = 0u;
    std::uint64_t declared_payload_hash = 0u;
    std::uint32_t total_words = 0u;
    std::uint32_t offset_words = 0u;
    std::uint32_t word_count = 0u;
    bool is_final_fragment = false;

    [[nodiscard]] constexpr std::uint64_t byte_offset() const noexcept {
        return static_cast<std::uint64_t>(offset_words) * sizeof(Word);
    }

    [[nodiscard]] constexpr std::uint64_t byte_extent() const noexcept {
        return static_cast<std::uint64_t>(word_count) * sizeof(Word);
    }
};

// This receipt is intentionally independent of fragment partitioning and of
// physical ingress sequence.  It is the bounded value a resident consumer can
// retain after successful reassembly; its 64-bit digest is not a collision-free
// substitute for the exact completed words retained by the assembler.
struct LogicalContactReceipt final {
    std::uint64_t logical_contact_id = 0u;
    std::uint64_t declared_payload_hash = 0u;
    std::uint64_t observed_payload_hash = 0u;
    std::uint32_t total_words = 0u;
    std::uint32_t received_words = 0u;
    std::uint64_t commit_ordinal = 0u;
    TransportStatus status = TransportStatus::empty;
};

[[nodiscard]] constexpr bool operator==(const LogicalContactReceipt& left,
                                        const LogicalContactReceipt& right) noexcept {
    return left.logical_contact_id == right.logical_contact_id &&
           left.declared_payload_hash == right.declared_payload_hash &&
           left.observed_payload_hash == right.observed_payload_hash &&
           left.total_words == right.total_words &&
           left.received_words == right.received_words &&
           left.commit_ordinal == right.commit_ordinal && left.status == right.status;
}

// This carries physical route detail for diagnosis.  It is deliberately not
// included in LogicalContactReceipt equality: different legal partitions must
// describe the same completed logical contact.
struct FragmentRouteReceipt final {
    std::uint64_t first_fragment_sequence = 0u;
    std::uint64_t final_fragment_sequence = 0u;
    std::uint64_t last_accepted_fragment_sequence = 0u;
    std::uint64_t last_accepted_byte_offset = 0u;
    std::uint64_t last_accepted_byte_extent = 0u;
    std::uint32_t accepted_fragment_count = 0u;
    bool final_fragment_seen = false;
    TransportFailure failure = TransportFailure::none;
};

[[nodiscard]] constexpr std::uint64_t initial_payload_hash() noexcept {
    return 1469598103934665603ull;
}

[[nodiscard]] constexpr std::uint64_t extend_payload_hash(std::uint64_t hash,
                                                           Word word) noexcept {
    constexpr std::uint64_t kPrime = 1099511628211ull;
    for (std::uint32_t byte = 0u; byte < sizeof(Word); ++byte) {
        hash ^= static_cast<std::uint8_t>((word >> (byte * 8u)) & 0xffu);
        hash *= kPrime;
    }
    return hash;
}

[[nodiscard]] inline std::uint64_t payload_hash(const Word* words,
                                                std::size_t word_count) noexcept {
    std::uint64_t hash = initial_payload_hash();
    for (std::size_t index = 0u; index < word_count; ++index) {
        hash = extend_payload_hash(hash, words[index]);
    }
    return hash;
}

class ResidentLogicalContactAssembler final {
public:
    [[nodiscard]] AcceptResult accept(const LogicalContactFragment& fragment,
                                      const Word* words,
                                      std::size_t supplied_word_count,
                                      std::uint64_t fragment_sequence) noexcept {
        if (logical_receipt_.status == TransportStatus::faulted) {
            last_rejection_ = TransportFailure::fault_latched;
            return AcceptResult::rejected;
        }
        if (logical_receipt_.status == TransportStatus::committed) {
            last_rejection_ = TransportFailure::duplicate_after_commit;
            return AcceptResult::rejected;
        }
        if (fragment_sequence == 0u) {
            return fail(TransportFailure::nonmonotonic_fragment_sequence);
        }
        if (supplied_word_count != static_cast<std::size_t>(fragment.word_count)) {
            return fail(TransportFailure::payload_size_mismatch);
        }
        if (fragment.word_count != 0u && words == nullptr) {
            return fail(TransportFailure::null_payload);
        }
        if (!valid_extent(fragment)) {
            return fail(TransportFailure::invalid_extent);
        }

        if (logical_receipt_.status == TransportStatus::empty) {
            if (fragment.offset_words != 0u) {
                return fail(TransportFailure::missing_first_fragment);
            }
            if (committed_contacts_ != 0u &&
                fragment.logical_contact_id == last_committed_logical_contact_id_) {
                return fail(TransportFailure::replayed_contact);
            }
            start(fragment, fragment_sequence);
        } else if (fragment.logical_contact_id != logical_receipt_.logical_contact_id) {
            return fail(TransportFailure::identity_mismatch);
        } else if (fragment.total_words != logical_receipt_.total_words) {
            return fail(TransportFailure::total_extent_mismatch);
        } else if (fragment.declared_payload_hash != logical_receipt_.declared_payload_hash) {
            return fail(TransportFailure::declared_hash_mismatch);
        } else if (fragment.offset_words != logical_receipt_.received_words) {
            return fail(TransportFailure::reordered_or_duplicate_fragment);
        } else if (fragment_sequence <= route_receipt_.last_accepted_fragment_sequence) {
            return fail(TransportFailure::nonmonotonic_fragment_sequence);
        }

        for (std::uint32_t index = 0u; index < fragment.word_count; ++index) {
            words_[logical_receipt_.received_words + index] = words[index];
            rolling_hash_ = extend_payload_hash(rolling_hash_, words[index]);
        }
        logical_receipt_.received_words += fragment.word_count;
        route_receipt_.accepted_fragment_count += 1u;
        route_receipt_.last_accepted_fragment_sequence = fragment_sequence;
        route_receipt_.last_accepted_byte_offset = fragment.byte_offset();
        route_receipt_.last_accepted_byte_extent = fragment.byte_extent();

        if (!fragment.is_final_fragment) {
            return AcceptResult::accepted;
        }

        route_receipt_.final_fragment_seen = true;
        route_receipt_.final_fragment_sequence = fragment_sequence;
        logical_receipt_.observed_payload_hash = rolling_hash_;
        if (rolling_hash_ != logical_receipt_.declared_payload_hash) {
            return fail(TransportFailure::payload_hash_mismatch);
        }
        if (committed_contacts_ == std::numeric_limits<std::uint64_t>::max()) {
            return fail(TransportFailure::commit_counter_exhausted);
        }

        logical_receipt_.status = TransportStatus::committed;
        logical_receipt_.commit_ordinal = ++committed_contacts_;
        last_committed_logical_contact_id_ = logical_receipt_.logical_contact_id;
        last_rejection_ = TransportFailure::none;
        return AcceptResult::committed;
    }

    // Reset is an explicit lifecycle boundary.  It permits one new logical
    // contact but never turns a previous rejection into a completion.
    void reset() noexcept {
        logical_receipt_ = {};
        route_receipt_ = {};
        rolling_hash_ = initial_payload_hash();
        last_rejection_ = TransportFailure::none;
    }

    [[nodiscard]] const LogicalContactReceipt& logical_receipt() const noexcept {
        return logical_receipt_;
    }

    [[nodiscard]] const FragmentRouteReceipt& route_receipt() const noexcept {
        return route_receipt_;
    }

    [[nodiscard]] TransportFailure last_rejection() const noexcept {
        return last_rejection_;
    }

    [[nodiscard]] std::uint64_t committed_contact_count() const noexcept {
        return committed_contacts_;
    }

    [[nodiscard]] bool completed_word_at(std::uint32_t index, Word& word) const noexcept {
        if (logical_receipt_.status != TransportStatus::committed ||
            index >= logical_receipt_.total_words) {
            return false;
        }
        word = words_[index];
        return true;
    }

    [[nodiscard]] std::uint32_t completed_word_count() const noexcept {
        return logical_receipt_.status == TransportStatus::committed
                   ? logical_receipt_.total_words
                   : 0u;
    }

private:
    [[nodiscard]] static constexpr bool valid_extent(const LogicalContactFragment& fragment) noexcept {
        if (fragment.total_words == 0u ||
            fragment.total_words > kMaximumLogicalContactWords ||
            fragment.word_count > kMaximumLogicalContactWords ||
            fragment.offset_words > fragment.total_words ||
            fragment.word_count > fragment.total_words - fragment.offset_words) {
            return false;
        }
        const std::uint32_t end = fragment.offset_words + fragment.word_count;
        if (fragment.is_final_fragment) {
            return end == fragment.total_words;
        }
        return fragment.word_count != 0u && end < fragment.total_words;
    }

    void start(const LogicalContactFragment& fragment, std::uint64_t fragment_sequence) noexcept {
        logical_receipt_.logical_contact_id = fragment.logical_contact_id;
        logical_receipt_.declared_payload_hash = fragment.declared_payload_hash;
        logical_receipt_.total_words = fragment.total_words;
        logical_receipt_.status = TransportStatus::assembling;
        route_receipt_.first_fragment_sequence = fragment_sequence;
        rolling_hash_ = initial_payload_hash();
    }

    [[nodiscard]] AcceptResult fail(TransportFailure failure) noexcept {
        logical_receipt_.status = TransportStatus::faulted;
        route_receipt_.failure = failure;
        last_rejection_ = failure;
        return AcceptResult::rejected;
    }

    std::array<Word, kMaximumLogicalContactWords> words_{};
    LogicalContactReceipt logical_receipt_{};
    FragmentRouteReceipt route_receipt_{};
    std::uint64_t rolling_hash_ = initial_payload_hash();
    std::uint64_t committed_contacts_ = 0u;
    std::uint64_t last_committed_logical_contact_id_ = 0u;
    TransportFailure last_rejection_ = TransportFailure::none;
};

}  // namespace substrate::bcc32::logical_contact_transport
