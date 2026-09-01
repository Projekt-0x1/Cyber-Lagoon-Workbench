#include "bcc32_provenance.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace substrate::bcc32 {
namespace {

[[nodiscard]] constexpr std::uint32_t rotate_right(std::uint32_t value,
                                                    std::uint32_t amount) {
    return (value >> amount) | (value << (32u - amount));
}

constexpr std::array<std::uint32_t, 64> kRoundConstants = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
    0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
    0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
    0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
    0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
    0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
    0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
    0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
    0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

constexpr std::size_t kMaxSelectionIndexableCount = 1u << 20u;
constexpr std::string_view kSelectionCandidateSetDomain =
    "0x1/bcc32/selection-candidate-set/v1";
constexpr std::string_view kSelectionEvaluationRequestDomain =
    "0x1/bcc32/selection-evaluation-request/v2";
constexpr std::string_view kSelectionTranscriptDomain =
    "0x1/bcc32/evaluator-selection-transcript/v4";

[[nodiscard]] bool all_zero(const Hash256& hash) {
    return std::all_of(hash.begin(), hash.end(), [](std::uint8_t value) {
        return value == 0u;
    });
}

bool fail(std::string* error, const char* message) {
    if (error != nullptr) *error = message;
    return false;
}

[[nodiscard]] bool contains(const std::vector<ContentAddress>& addresses,
                            const ContentAddress& candidate) {
    return std::any_of(addresses.begin(), addresses.end(), [&](const ContentAddress& value) {
        return value == candidate;
    });
}

[[nodiscard]] bool has_duplicate(const std::vector<ContentAddress>& addresses) {
    std::vector<ContentAddress> ordered = addresses;
    std::sort(ordered.begin(), ordered.end(), [](const ContentAddress& left,
                                                  const ContentAddress& right) {
        if (left.digest != right.digest) return left.digest < right.digest;
        return left.byte_count < right.byte_count;
    });
    return std::adjacent_find(ordered.begin(), ordered.end()) != ordered.end();
}

[[nodiscard]] bool strictly_sorted_addresses(
    const std::vector<ContentAddress>& addresses) {
    return std::adjacent_find(
               addresses.begin(), addresses.end(),
               [](const ContentAddress& left, const ContentAddress& right) {
                   if (left.digest != right.digest) return !(left.digest < right.digest);
                   return left.byte_count >= right.byte_count;
               }) == addresses.end();
}

void append_u32(std::vector<std::byte>* bytes, std::uint32_t value) {
    for (std::uint32_t shift = 24u;; shift -= 8u) {
        bytes->push_back(static_cast<std::byte>((value >> shift) & 0xffu));
        if (shift == 0u) return;
    }
}

void append_u64(std::vector<std::byte>* bytes, std::uint64_t value) {
    for (std::uint32_t shift = 56u;; shift -= 8u) {
        bytes->push_back(static_cast<std::byte>((value >> shift) & 0xffu));
        if (shift == 0u) return;
    }
}

void append_address(std::vector<std::byte>* bytes, const ContentAddress& address) {
    for (const std::uint8_t byte : address.digest) bytes->push_back(static_cast<std::byte>(byte));
    append_u64(bytes, address.byte_count);
}

void append_domain(std::vector<std::byte>* bytes, std::string_view domain) {
    append_u64(bytes, domain.size());
    for (const char byte : domain) {
        bytes->push_back(static_cast<std::byte>(static_cast<unsigned char>(byte)));
    }
}

void append_candidate(std::vector<std::byte>* bytes,
                      const SelectionCandidate& candidate) {
    append_address(bytes, candidate.root);
    append_address(bytes, candidate.generator_provenance);
    append_u64(bytes, candidate.parentage.size());
    for (const ContentAddress& parent : candidate.parentage) {
        append_address(bytes, parent);
    }
    append_u64(bytes, candidate.mutation_seeds.size());
    for (const std::uint64_t seed : candidate.mutation_seeds) append_u64(bytes, seed);
}

void append_actions(std::vector<std::byte>* bytes,
                    const std::vector<SelectionAction>& actions) {
    append_u64(bytes, actions.size());
    for (const SelectionAction action : actions) {
        append_u32(bytes, static_cast<std::uint32_t>(action));
    }
}

class CanonicalReader {
  public:
    explicit CanonicalReader(std::span<const std::byte> bytes) : bytes_(bytes) {}

    [[nodiscard]] bool take_u32(std::uint32_t* value) {
        if (value == nullptr || remaining() < sizeof(std::uint32_t)) return false;
        std::uint32_t decoded = 0u;
        for (std::size_t index = 0u; index < sizeof(decoded); ++index) {
            decoded = (decoded << 8u) |
                      std::to_integer<std::uint8_t>(bytes_[cursor_ + index]);
        }
        cursor_ += sizeof(decoded);
        *value = decoded;
        return true;
    }

    [[nodiscard]] bool take_u64(std::uint64_t* value) {
        if (value == nullptr || remaining() < sizeof(std::uint64_t)) return false;
        std::uint64_t decoded = 0u;
        for (std::size_t index = 0u; index < sizeof(decoded); ++index) {
            decoded = (decoded << 8u) |
                      std::to_integer<std::uint8_t>(bytes_[cursor_ + index]);
        }
        cursor_ += sizeof(decoded);
        *value = decoded;
        return true;
    }

    [[nodiscard]] bool take_address(ContentAddress* address) {
        if (address == nullptr || remaining() < address->digest.size() + sizeof(std::uint64_t)) {
            return false;
        }
        for (std::uint8_t& byte : address->digest) {
            byte = std::to_integer<std::uint8_t>(bytes_[cursor_++]);
        }
        return take_u64(&address->byte_count);
    }

    [[nodiscard]] bool take_domain(std::string_view expected) {
        std::uint64_t size = 0u;
        if (!take_u64(&size) || size != expected.size() || remaining() < expected.size()) {
            return false;
        }
        const bool matches = std::equal(
            expected.begin(), expected.end(), bytes_.begin() + static_cast<std::ptrdiff_t>(cursor_),
            [](char expected_byte, std::byte actual_byte) {
                return static_cast<unsigned char>(expected_byte) ==
                       std::to_integer<unsigned char>(actual_byte);
            });
        cursor_ += expected.size();
        return matches;
    }

    [[nodiscard]] std::size_t remaining() const { return bytes_.size() - cursor_; }
    [[nodiscard]] bool done() const { return cursor_ == bytes_.size(); }

  private:
    std::span<const std::byte> bytes_;
    std::size_t cursor_ = 0u;
};

[[nodiscard]] bool take_bounded_count(CanonicalReader* reader,
                                      std::size_t maximum,
                                      std::size_t minimum_item_bytes,
                                      std::size_t* count) {
    std::uint64_t wire_count = 0u;
    if (reader == nullptr || count == nullptr || minimum_item_bytes == 0u ||
        !reader->take_u64(&wire_count) || wire_count > maximum ||
        wire_count > reader->remaining() / minimum_item_bytes) {
        return false;
    }
    *count = static_cast<std::size_t>(wire_count);
    return true;
}

[[nodiscard]] bool take_candidate(CanonicalReader* reader,
                                  SelectionCandidate* candidate) {
    if (reader == nullptr || candidate == nullptr ||
        !reader->take_address(&candidate->root) ||
        !reader->take_address(&candidate->generator_provenance)) {
        return false;
    }
    std::size_t parent_count = 0u;
    if (!take_bounded_count(reader, kMaxSelectionIndexableCount, 40u, &parent_count)) {
        return false;
    }
    candidate->parentage.reserve(parent_count);
    for (std::size_t index = 0u; index < parent_count; ++index) {
        ContentAddress parent{};
        if (!reader->take_address(&parent)) return false;
        candidate->parentage.push_back(parent);
    }
    std::size_t mutation_count = 0u;
    if (!take_bounded_count(reader, kMaxSelectionIndexableCount,
                            sizeof(std::uint64_t), &mutation_count)) {
        return false;
    }
    candidate->mutation_seeds.reserve(mutation_count);
    for (std::size_t index = 0u; index < mutation_count; ++index) {
        std::uint64_t seed = 0u;
        if (!reader->take_u64(&seed)) return false;
        candidate->mutation_seeds.push_back(seed);
    }
    return true;
}

[[nodiscard]] bool canonical_bytes_equal(std::span<const std::byte> encoded,
                                         const std::vector<std::byte>& canonical) {
    return encoded.size() == canonical.size() &&
           std::equal(encoded.begin(), encoded.end(), canonical.begin());
}

template <typename T>
[[nodiscard]] bool contains_value(const std::vector<T>& values, const T& value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

[[nodiscard]] bool valid_action(SelectionAction action) {
    return static_cast<std::uint32_t>(action) <=
           static_cast<std::uint32_t>(SelectionAction::stop);
}

[[nodiscard]] bool valid_stop_reason(SelectionStopReason reason) {
    return static_cast<std::uint32_t>(reason) <=
           static_cast<std::uint32_t>(SelectionStopReason::evaluator_completed);
}

[[nodiscard]] bool valid_compute_semantics(SelectionComputeSemantics semantics) {
    return semantics == SelectionComputeSemantics::production_supersteps;
}

[[nodiscard]] bool valid_artifact_kind(ArtifactKind kind) {
    return kind == ArtifactKind::propagule_capsule ||
           kind == ArtifactKind::adult_continuity_checkpoint ||
           kind == ArtifactKind::cultural_capsule;
}

[[nodiscard]] bool valid_mating_pair(const SelectionMatingPair& pair,
                                     std::size_t candidate_count) {
    return pair.first_candidate_index < pair.second_candidate_index &&
           pair.second_candidate_index < candidate_count;
}

[[nodiscard]] bool valid_evaluator_event_kind(SelectionEvaluatorEventKind kind) {
    return kind == SelectionEvaluatorEventKind::score ||
           kind == SelectionEvaluatorEventKind::comparison;
}

[[nodiscard]] bool valid_entry_event_kind(EntryEventKind kind) {
    return kind == EntryEventKind::genesis_entry ||
           kind == EntryEventKind::evaluator_selection ||
           kind == EntryEventKind::law_continuation;
}

[[nodiscard]] bool is_continuation_action(SelectionAction action) {
    return action == SelectionAction::preserve_candidate ||
           action == SelectionAction::branch_candidate ||
           action == SelectionAction::migrate_intact_candidate;
}

[[nodiscard]] bool is_candidate_action(SelectionAction action) {
    return is_continuation_action(action) ||
           action == SelectionAction::allocate_compute ||
           action == SelectionAction::choose_mates;
}

[[nodiscard]] bool is_zero_address(const ContentAddress& address) {
    return address == ContentAddress{};
}

[[nodiscard]] bool validate_decoded_selection_transcript_shape(
    const SelectionTranscript& transcript,
    std::string* error) {
    if (transcript.schema_version != SelectionTranscript::kSchemaVersion ||
        !is_valid_content_address(transcript.evaluation_request_commitment) ||
        !is_valid_content_address(transcript.evaluator_identity) ||
        !is_valid_content_address(transcript.evaluator_version) ||
        !is_valid_content_address(transcript.evaluator_target_provenance) ||
        !is_valid_content_address(transcript.environment_contact_input) ||
        !is_valid_content_address(transcript.candidate_set_commitment) ||
        transcript.evaluator_transcript.size() > kMaxSelectionIndexableCount ||
        transcript.actions.size() > kMaxSelectionIndexableCount) {
        return fail(error, "BCC-32 decoded selection transcript has an invalid header");
    }
    for (const SelectionEvaluatorEvent& event : transcript.evaluator_transcript) {
        if (!valid_evaluator_event_kind(event.kind) ||
            event.scores.size() > kMaxSelectionIndexableCount) {
            return fail(error, "BCC-32 decoded selection transcript has an invalid evaluator event");
        }
        if (event.kind == SelectionEvaluatorEventKind::score) {
            if (event.compared_candidate_index != 0u || event.scores.empty() ||
                event.result != SelectionComparison::equal) {
                return fail(error, "BCC-32 decoded selection transcript has a noncanonical score event");
            }
        } else if (event.compared_candidate_index == event.candidate_index ||
                   !event.scores.empty() || event.rank != 0u ||
                   (event.result != SelectionComparison::less &&
                    event.result != SelectionComparison::equal &&
                    event.result != SelectionComparison::greater)) {
            return fail(error,
                        "BCC-32 decoded selection transcript has a noncanonical comparison event");
        }
    }
    if (transcript.action_alphabet.empty() ||
        transcript.action_alphabet.size() >
            static_cast<std::size_t>(SelectionAction::stop) + 1u ||
        !valid_stop_reason(transcript.stopping_reason) || transcript.actions.empty()) {
        return fail(error, "BCC-32 decoded selection transcript has an invalid action header");
    }
    for (std::size_t index = 0u; index < transcript.action_alphabet.size(); ++index) {
        const SelectionAction action = transcript.action_alphabet[index];
        if (!valid_action(action) ||
            (index != 0u &&
             static_cast<std::uint32_t>(transcript.action_alphabet[index - 1u]) >=
                 static_cast<std::uint32_t>(action))) {
            return fail(error,
                        "BCC-32 decoded selection transcript has a noncanonical action alphabet");
        }
    }
    std::uint64_t population_slot_count = 0u;
    for (const SelectionActionRecord& action : transcript.actions) {
        if (!valid_action(action.action) ||
            !contains_value(transcript.action_alphabet, action.action) ||
            action.mate_indices.size() > kMaxSelectionIndexableCount) {
            return fail(error, "BCC-32 decoded selection transcript has an invalid action");
        }
        if (is_continuation_action(action.action)) {
            if (action.multiplicity == 0u || action.compute_units != 0u ||
                !action.mate_indices.empty() ||
                action.multiplicity > std::numeric_limits<std::uint64_t>::max() -
                                          population_slot_count) {
                return fail(error,
                            "BCC-32 decoded selection transcript has a noncanonical continuation");
            }
            population_slot_count += action.multiplicity;
            continue;
        }
        if (action.action == SelectionAction::allocate_compute) {
            if (action.multiplicity != 0u || action.compute_units == 0u ||
                !action.mate_indices.empty()) {
                return fail(error,
                            "BCC-32 decoded selection transcript has a noncanonical compute action");
            }
            continue;
        }
        if (action.action == SelectionAction::choose_mates) {
            if (action.multiplicity != 0u || action.compute_units != 0u ||
                action.mate_indices.empty()) {
                return fail(error,
                            "BCC-32 decoded selection transcript has a noncanonical mating action");
            }
            std::uint32_t previous = 0u;
            bool have_previous = false;
            for (const std::uint32_t mate : action.mate_indices) {
                if (mate == action.candidate_index ||
                    (have_previous && mate <= previous)) {
                    return fail(error,
                                "BCC-32 decoded selection transcript has noncanonical mate indices");
                }
                previous = mate;
                have_previous = true;
            }
            continue;
        }
        if (action.candidate_index != 0u || action.multiplicity != 0u ||
            action.compute_units != 0u || !action.mate_indices.empty()) {
            return fail(error,
                        "BCC-32 decoded selection transcript hides irrelevant action bytes");
        }
    }
    return true;
}

[[nodiscard]] bool validate_entry_event_shape(const Provenance& provenance,
                                               std::string* error) {
    const EntryEvent& event = provenance.entry_event;
    if (!valid_entry_event_kind(event.kind)) {
        return fail(error, "BCC-32 provenance entry event has an unknown tag");
    }
    if (event.kind == EntryEventKind::genesis_entry) {
        if (!is_valid_content_address(event.genesis_entry) ||
            event.genesis_entry != provenance.genesis ||
            !is_zero_address(event.evaluator_transcript) ||
            event.next_population_slot != 0u) {
            return fail(error, "BCC-32 provenance has a malformed genesis entry event");
        }
        return true;
    }
    if (event.kind == EntryEventKind::evaluator_selection) {
        if (!is_zero_address(event.genesis_entry) ||
            !is_valid_content_address(event.evaluator_transcript)) {
            return fail(error,
                        "BCC-32 provenance has a malformed evaluator-selection entry event");
        }
        return true;
    }
    if (!is_zero_address(event.genesis_entry) ||
        !is_zero_address(event.evaluator_transcript) ||
        event.next_population_slot != 0u) {
        return fail(error, "BCC-32 provenance has a malformed law-continuation entry event");
    }
    return true;
}

[[nodiscard]] std::uint64_t pair_key(std::uint32_t left, std::uint32_t right) {
    if (right < left) std::swap(left, right);
    return (static_cast<std::uint64_t>(left) << 32u) | right;
}

}  // namespace

ContentHasher::ContentHasher()
    : state_({
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    }) {}

void ContentHasher::compress(const std::uint8_t* block) {
    std::array<std::uint32_t, 64> words{};
    for (std::uint32_t index = 0; index < 16u; ++index) {
        const std::size_t base = index * 4u;
        words[index] = (static_cast<std::uint32_t>(block[base]) << 24u) |
                       (static_cast<std::uint32_t>(block[base + 1u]) << 16u) |
                       (static_cast<std::uint32_t>(block[base + 2u]) << 8u) |
                       static_cast<std::uint32_t>(block[base + 3u]);
    }
    for (std::uint32_t index = 16u; index < 64u; ++index) {
        const std::uint32_t s0 = rotate_right(words[index - 15u], 7u) ^
                                 rotate_right(words[index - 15u], 18u) ^
                                 (words[index - 15u] >> 3u);
        const std::uint32_t s1 = rotate_right(words[index - 2u], 17u) ^
                                 rotate_right(words[index - 2u], 19u) ^
                                 (words[index - 2u] >> 10u);
        words[index] = words[index - 16u] + s0 + words[index - 7u] + s1;
    }

    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];
    for (std::uint32_t index = 0; index < 64u; ++index) {
        const std::uint32_t s1 = rotate_right(e, 6u) ^ rotate_right(e, 11u) ^
                                 rotate_right(e, 25u);
        const std::uint32_t choice = (e & f) ^ ((~e) & g);
        const std::uint32_t temporary_one =
            h + s1 + choice + kRoundConstants[index] + words[index];
        const std::uint32_t s0 = rotate_right(a, 2u) ^ rotate_right(a, 13u) ^
                                 rotate_right(a, 22u);
        const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t temporary_two = s0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temporary_one;
        d = c;
        c = b;
        b = a;
        a = temporary_one + temporary_two;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
}

void ContentHasher::update(std::span<const std::byte> bytes) {
    if (finished_ || bytes.empty()) return;
    total_bytes_ += static_cast<std::uint64_t>(bytes.size());
    const auto* cursor = reinterpret_cast<const std::uint8_t*>(bytes.data());
    std::size_t remaining = bytes.size();
    if (pending_bytes_ != 0u) {
        const std::size_t copied = std::min(remaining, pending_.size() - pending_bytes_);
        std::memcpy(pending_.data() + pending_bytes_, cursor, copied);
        pending_bytes_ += copied;
        cursor += copied;
        remaining -= copied;
        if (pending_bytes_ == pending_.size()) {
            compress(pending_.data());
            pending_bytes_ = 0u;
        }
    }
    while (remaining >= pending_.size()) {
        compress(cursor);
        cursor += pending_.size();
        remaining -= pending_.size();
    }
    if (remaining != 0u) {
        std::memcpy(pending_.data(), cursor, remaining);
        pending_bytes_ = remaining;
    }
}

Hash256 ContentHasher::finish() {
    if (finished_) return digest_;
    const std::uint64_t bit_count = total_bytes_ * 8u;
    pending_[pending_bytes_++] = 0x80u;
    if (pending_bytes_ > 56u) {
        std::fill(pending_.begin() + static_cast<std::ptrdiff_t>(pending_bytes_),
                  pending_.end(),
                  0u);
        compress(pending_.data());
        pending_bytes_ = 0u;
    }
    std::fill(pending_.begin() + static_cast<std::ptrdiff_t>(pending_bytes_),
              pending_.begin() + 56,
              0u);
    for (std::uint32_t byte = 0; byte < 8u; ++byte) {
        pending_[56u + byte] =
            static_cast<std::uint8_t>(bit_count >> (56u - byte * 8u));
    }
    compress(pending_.data());
    for (std::uint32_t index = 0; index < 8u; ++index) {
        for (std::uint32_t byte = 0; byte < 4u; ++byte) {
            digest_[index * 4u + byte] = static_cast<std::uint8_t>(
                state_[index] >> (24u - byte * 8u));
        }
    }
    finished_ = true;
    return digest_;
}

Hash256 content_hash(std::span<const std::byte> bytes) {
    ContentHasher hasher;
    hasher.update(bytes);
    return hasher.finish();
}

ContentAddress content_address(std::span<const std::byte> bytes) {
    return {content_hash(bytes), static_cast<std::uint64_t>(bytes.size())};
}

bool is_valid_content_address(const ContentAddress& address) {
    return !all_zero(address.digest);
}

std::string hash_hex(const Hash256& hash) {
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (const std::uint8_t byte : hash) output << std::setw(2) << +byte;
    return output.str();
}

std::vector<std::byte> canonical_selection_candidate_set(
    const SelectionCandidateSet& candidate_set) {
    std::vector<std::byte> bytes;
    append_domain(&bytes, kSelectionCandidateSetDomain);
    append_u32(&bytes, candidate_set.schema_version);
    append_u64(&bytes, candidate_set.candidates.size());
    for (const SelectionCandidate& candidate : candidate_set.candidates) {
        append_candidate(&bytes, candidate);
    }
    return bytes;
}

ContentAddress selection_candidate_set_address(
    const SelectionCandidateSet& candidate_set) {
    const std::vector<std::byte> bytes =
        canonical_selection_candidate_set(candidate_set);
    return content_address(bytes);
}

bool validate_selection_candidate_set(const SelectionCandidateSet& candidate_set,
                                      std::string* error) {
    if (candidate_set.schema_version != SelectionCandidateSet::kSchemaVersion) {
        return fail(error, "BCC-32 selection candidate set has an unsupported schema");
    }
    if (candidate_set.candidates.empty() ||
        candidate_set.candidates.size() > kMaxSelectionIndexableCount) {
        return fail(error, "BCC-32 selection candidate set exceeds its count bound");
    }
    std::vector<ContentAddress> roots;
    roots.reserve(candidate_set.candidates.size());
    for (const SelectionCandidate& candidate : candidate_set.candidates) {
        if (!is_valid_content_address(candidate.root) ||
            !is_valid_content_address(candidate.generator_provenance)) {
            return fail(error, "BCC-32 selection candidate set has an unaddressed cause");
        }
        if (candidate.parentage.size() > kMaxSelectionIndexableCount ||
            candidate.mutation_seeds.size() > kMaxSelectionIndexableCount) {
            return fail(error, "BCC-32 selection candidate set exceeds a cause-count bound");
        }
        if (!strictly_sorted_addresses(candidate.parentage)) {
            return fail(error,
                        "BCC-32 selection candidate parentage is not sorted and unique");
        }
        for (const ContentAddress& parent : candidate.parentage) {
            if (!is_valid_content_address(parent)) {
                return fail(error, "BCC-32 selection candidate set has an unaddressed parent");
            }
        }
        roots.push_back(candidate.root);
    }
    if (has_duplicate(roots)) {
        return fail(error, "BCC-32 selection candidate set repeats a frozen root");
    }
    return true;
}

bool decode_selection_candidate_set(std::span<const std::byte> bytes,
                                    SelectionCandidateSet* candidate_set,
                                    std::string* error) {
    if (candidate_set == nullptr) {
        return fail(error, "BCC-32 selection candidate-set decode requires an output");
    }
    CanonicalReader reader(bytes);
    SelectionCandidateSet decoded{};
    std::size_t candidate_count = 0u;
    if (!reader.take_domain(kSelectionCandidateSetDomain) ||
        !reader.take_u32(&decoded.schema_version) ||
        !take_bounded_count(&reader, kMaxSelectionIndexableCount, 96u,
                            &candidate_count)) {
        return fail(error, "BCC-32 selection candidate set has malformed canonical bytes");
    }
    decoded.candidates.reserve(candidate_count);
    for (std::size_t index = 0u; index < candidate_count; ++index) {
        SelectionCandidate candidate{};
        if (!take_candidate(&reader, &candidate)) {
            return fail(error, "BCC-32 selection candidate set has a truncated candidate");
        }
        decoded.candidates.push_back(std::move(candidate));
    }
    if (!reader.done()) {
        return fail(error, "BCC-32 selection candidate set has trailing canonical bytes");
    }
    if (!validate_selection_candidate_set(decoded, error)) return false;
    if (!canonical_bytes_equal(bytes, canonical_selection_candidate_set(decoded))) {
        return fail(error, "BCC-32 selection candidate set is not canonically encoded");
    }
    *candidate_set = std::move(decoded);
    return true;
}

std::vector<std::byte> canonical_selection_evaluation_request(
    const SelectionEvaluationRequest& request) {
    std::vector<std::byte> bytes;
    append_domain(&bytes, kSelectionEvaluationRequestDomain);
    append_u32(&bytes, request.schema_version);
    append_address(&bytes, request.operation_predecessor);
    append_u64(&bytes, request.selection_round);
    append_address(&bytes, request.candidate_set_commitment);
    append_u32(&bytes, static_cast<std::uint32_t>(request.candidate_artifact_kind));
    append_address(&bytes, request.evaluator_identity);
    append_address(&bytes, request.evaluator_version);
    append_address(&bytes, request.evaluator_artifact);
    append_address(&bytes, request.evaluator_target_provenance);
    append_u64(&bytes, request.evaluation_inputs.size());
    for (const ContentAddress& input : request.evaluation_inputs) {
        append_address(&bytes, input);
    }
    append_address(&bytes, request.environment_contact_input);
    append_address(&bytes, request.replay_semantics);
    append_address(&bytes, request.slot_materialization_semantics);
    append_address(&bytes, request.scheduler_protocol);
    append_address(&bytes, request.mating_contact_manifest);
    append_u32(&bytes, static_cast<std::uint32_t>(request.compute_semantics));
    append_actions(&bytes, request.action_alphabet);
    append_u64(&bytes, request.allowed_mating_pairs.size());
    for (const SelectionMatingPair& pair : request.allowed_mating_pairs) {
        append_u32(&bytes, pair.first_candidate_index);
        append_u32(&bytes, pair.second_candidate_index);
    }
    append_u64(&bytes, request.limits.maximum_input_bytes);
    append_u64(&bytes, request.limits.maximum_transcript_bytes);
    append_u64(&bytes, request.limits.maximum_wall_milliseconds);
    append_u64(&bytes, request.limits.maximum_cpu_seconds);
    append_u64(&bytes, request.limits.maximum_address_space_bytes);
    append_u64(&bytes, request.limits.maximum_evaluator_events);
    append_u64(&bytes, request.limits.maximum_action_records);
    append_u64(&bytes, request.limits.maximum_score_dimensions);
    append_u64(&bytes, request.limits.minimum_population_slots);
    append_u64(&bytes, request.limits.maximum_population_slots);
    append_u64(&bytes, request.limits.maximum_total_compute_units);
    append_u64(&bytes, request.limits.maximum_mating_edges);
    append_u64(&bytes, request.limits.maximum_retries);
    return bytes;
}

ContentAddress selection_evaluation_request_address(
    const SelectionEvaluationRequest& request) {
    const std::vector<std::byte> bytes =
        canonical_selection_evaluation_request(request);
    return content_address(bytes);
}

bool validate_selection_evaluation_request(
    const SelectionCandidateSet& candidate_set,
    const SelectionEvaluationRequest& request,
    std::string* error) {
    if (request.schema_version != SelectionEvaluationRequest::kSchemaVersion) {
        return fail(error, "BCC-32 selection request has an unsupported schema");
    }
    if (!validate_selection_candidate_set(candidate_set, error) ||
        selection_candidate_set_address(candidate_set) !=
            request.candidate_set_commitment) {
        return fail(error, "BCC-32 selection request does not bind its candidate set");
    }
    if (request.selection_round == 0u ||
        !is_valid_content_address(request.operation_predecessor) ||
        !valid_artifact_kind(request.candidate_artifact_kind) ||
        !is_valid_content_address(request.evaluator_identity) ||
        !is_valid_content_address(request.evaluator_version) ||
        !is_valid_content_address(request.evaluator_artifact) ||
        !is_valid_content_address(request.evaluator_target_provenance) ||
        !is_valid_content_address(request.environment_contact_input) ||
        !is_valid_content_address(request.replay_semantics) ||
        !is_valid_content_address(request.slot_materialization_semantics) ||
        !is_valid_content_address(request.scheduler_protocol) ||
        !valid_compute_semantics(request.compute_semantics)) {
        return fail(error, "BCC-32 selection request has an unaddressed protocol input");
    }
    if (request.evaluation_inputs.empty() ||
        request.evaluation_inputs.size() > kMaxSelectionIndexableCount ||
        request.evaluation_inputs.size() != request.limits.maximum_retries + 1u) {
        return fail(error, "BCC-32 selection request has an invalid retry-input schedule");
    }
    for (const ContentAddress& input : request.evaluation_inputs) {
        if (!is_valid_content_address(input)) {
            return fail(error, "BCC-32 selection request has an unaddressed evaluator input");
        }
    }
    if (request.action_alphabet.empty() ||
        request.action_alphabet.size() >
            static_cast<std::size_t>(SelectionAction::stop) + 1u) {
        return fail(error, "BCC-32 selection request has an invalid action alphabet");
    }
    bool has_continuation = false;
    bool has_compute = false;
    bool has_mating = false;
    bool has_stop = false;
    for (std::size_t index = 0u; index < request.action_alphabet.size(); ++index) {
        const SelectionAction action = request.action_alphabet[index];
        if (!valid_action(action) ||
            (index != 0u &&
             static_cast<std::uint32_t>(request.action_alphabet[index - 1u]) >=
                 static_cast<std::uint32_t>(action))) {
            return fail(error,
                        "BCC-32 selection request action alphabet is not sorted and unique");
        }
        has_continuation = has_continuation || is_continuation_action(action);
        has_compute = has_compute || action == SelectionAction::allocate_compute;
        has_mating = has_mating || action == SelectionAction::choose_mates;
        has_stop = has_stop || action == SelectionAction::stop;
    }
    if (!has_continuation || !has_stop) {
        return fail(error, "BCC-32 selection request cannot produce a stopped population");
    }
    if (request.allowed_mating_pairs.size() > kMaxSelectionIndexableCount) {
        return fail(error, "BCC-32 selection request exceeds its mating-pair bound");
    }
    SelectionMatingPair previous{};
    bool have_previous = false;
    for (const SelectionMatingPair& pair : request.allowed_mating_pairs) {
        if (!valid_mating_pair(pair, candidate_set.candidates.size()) ||
            (have_previous &&
             (pair.first_candidate_index < previous.first_candidate_index ||
              (pair.first_candidate_index == previous.first_candidate_index &&
               pair.second_candidate_index <= previous.second_candidate_index)))) {
            return fail(error,
                        "BCC-32 selection request mating pairs are not sorted and unique");
        }
        previous = pair;
        have_previous = true;
    }
    if (has_mating != !request.allowed_mating_pairs.empty() ||
        (has_mating && !is_valid_content_address(request.mating_contact_manifest)) ||
        (!has_mating && !is_zero_address(request.mating_contact_manifest))) {
        return fail(error, "BCC-32 selection request has an inconsistent mating channel");
    }

    const SelectionEvaluationLimits& limits = request.limits;
    if (limits.maximum_input_bytes == 0u ||
        limits.maximum_input_bytes > (1ull << 30u) ||
        limits.maximum_transcript_bytes == 0u ||
        limits.maximum_transcript_bytes > (1ull << 30u) ||
        limits.maximum_wall_milliseconds == 0u ||
        limits.maximum_wall_milliseconds > 86'400'000u ||
        limits.maximum_cpu_seconds == 0u ||
        limits.maximum_cpu_seconds > 86'400u ||
        limits.maximum_address_space_bytes < (16ull << 20u) ||
        limits.maximum_address_space_bytes > (1ull << 40u) ||
        limits.maximum_evaluator_events == 0u ||
        limits.maximum_evaluator_events > kMaxSelectionIndexableCount ||
        limits.maximum_action_records == 0u ||
        limits.maximum_action_records > kMaxSelectionIndexableCount ||
        limits.maximum_score_dimensions == 0u ||
        limits.maximum_score_dimensions > kMaxSelectionIndexableCount ||
        limits.maximum_population_slots == 0u ||
        limits.maximum_population_slots > kMaxSelectionIndexableCount ||
        limits.minimum_population_slots > limits.maximum_population_slots ||
        limits.maximum_mating_edges > kMaxSelectionIndexableCount ||
        limits.maximum_retries > kMaxSelectionIndexableCount ||
        limits.maximum_retries + 1u != request.evaluation_inputs.size() ||
        (has_compute != (limits.maximum_total_compute_units != 0u)) ||
        (has_mating != (limits.maximum_mating_edges != 0u))) {
        return fail(error, "BCC-32 selection request has inconsistent aggregate limits");
    }
    return true;
}

bool decode_selection_evaluation_request(
    std::span<const std::byte> bytes,
    SelectionEvaluationRequest* request,
    std::string* error) {
    if (request == nullptr) {
        return fail(error, "BCC-32 selection-request decode requires an output");
    }
    CanonicalReader reader(bytes);
    SelectionEvaluationRequest decoded{};
    std::uint32_t artifact_kind = 0u;
    std::uint32_t compute_semantics = 0u;
    if (!reader.take_domain(kSelectionEvaluationRequestDomain) ||
        !reader.take_u32(&decoded.schema_version) ||
        !reader.take_address(&decoded.operation_predecessor) ||
        !reader.take_u64(&decoded.selection_round) ||
        !reader.take_address(&decoded.candidate_set_commitment) ||
        !reader.take_u32(&artifact_kind) ||
        !reader.take_address(&decoded.evaluator_identity) ||
        !reader.take_address(&decoded.evaluator_version) ||
        !reader.take_address(&decoded.evaluator_artifact) ||
        !reader.take_address(&decoded.evaluator_target_provenance)) {
        return fail(error, "BCC-32 selection request has malformed canonical bytes");
    }
    decoded.candidate_artifact_kind = static_cast<ArtifactKind>(artifact_kind);
    std::size_t input_count = 0u;
    if (!take_bounded_count(&reader, kMaxSelectionIndexableCount, 40u,
                            &input_count)) {
        return fail(error, "BCC-32 selection request has an invalid input count");
    }
    decoded.evaluation_inputs.reserve(input_count);
    for (std::size_t index = 0u; index < input_count; ++index) {
        ContentAddress input{};
        if (!reader.take_address(&input)) {
            return fail(error, "BCC-32 selection request has a truncated input");
        }
        decoded.evaluation_inputs.push_back(input);
    }
    if (!reader.take_address(&decoded.environment_contact_input) ||
        !reader.take_address(&decoded.replay_semantics) ||
        !reader.take_address(&decoded.slot_materialization_semantics) ||
        !reader.take_address(&decoded.scheduler_protocol) ||
        !reader.take_address(&decoded.mating_contact_manifest) ||
        !reader.take_u32(&compute_semantics)) {
        return fail(error, "BCC-32 selection request has a truncated protocol header");
    }
    decoded.compute_semantics =
        static_cast<SelectionComputeSemantics>(compute_semantics);
    std::size_t alphabet_count = 0u;
    if (!take_bounded_count(
            &reader, static_cast<std::size_t>(SelectionAction::stop) + 1u,
            sizeof(std::uint32_t), &alphabet_count)) {
        return fail(error, "BCC-32 selection request has an invalid action count");
    }
    decoded.action_alphabet.reserve(alphabet_count);
    for (std::size_t index = 0u; index < alphabet_count; ++index) {
        std::uint32_t action = 0u;
        if (!reader.take_u32(&action)) {
            return fail(error, "BCC-32 selection request has a truncated action");
        }
        decoded.action_alphabet.push_back(static_cast<SelectionAction>(action));
    }
    std::size_t pair_count = 0u;
    if (!take_bounded_count(&reader, kMaxSelectionIndexableCount, 8u,
                            &pair_count)) {
        return fail(error, "BCC-32 selection request has an invalid mating count");
    }
    decoded.allowed_mating_pairs.reserve(pair_count);
    for (std::size_t index = 0u; index < pair_count; ++index) {
        SelectionMatingPair pair{};
        if (!reader.take_u32(&pair.first_candidate_index) ||
            !reader.take_u32(&pair.second_candidate_index)) {
            return fail(error, "BCC-32 selection request has a truncated mating pair");
        }
        decoded.allowed_mating_pairs.push_back(pair);
    }
    SelectionEvaluationLimits& limits = decoded.limits;
    if (!reader.take_u64(&limits.maximum_input_bytes) ||
        !reader.take_u64(&limits.maximum_transcript_bytes) ||
        !reader.take_u64(&limits.maximum_wall_milliseconds) ||
        !reader.take_u64(&limits.maximum_cpu_seconds) ||
        !reader.take_u64(&limits.maximum_address_space_bytes) ||
        !reader.take_u64(&limits.maximum_evaluator_events) ||
        !reader.take_u64(&limits.maximum_action_records) ||
        !reader.take_u64(&limits.maximum_score_dimensions) ||
        !reader.take_u64(&limits.minimum_population_slots) ||
        !reader.take_u64(&limits.maximum_population_slots) ||
        !reader.take_u64(&limits.maximum_total_compute_units) ||
        !reader.take_u64(&limits.maximum_mating_edges) ||
        !reader.take_u64(&limits.maximum_retries) || !reader.done()) {
        return fail(error, "BCC-32 selection request has truncated or trailing bytes");
    }
    if (!canonical_bytes_equal(bytes,
                               canonical_selection_evaluation_request(decoded))) {
        return fail(error, "BCC-32 selection request is not canonically encoded");
    }
    *request = std::move(decoded);
    return true;
}

std::vector<std::byte> canonical_selection_transcript(
    const SelectionTranscript& transcript) {
    std::vector<std::byte> bytes;
    // A length-prefixed semantic domain is the first canonical field. Other
    // artifact encoders cannot collide by merely presenting the same payload.
    append_domain(&bytes, kSelectionTranscriptDomain);
    append_u32(&bytes, transcript.schema_version);
    append_address(&bytes, transcript.evaluation_request_commitment);
    append_address(&bytes, transcript.evaluator_identity);
    append_address(&bytes, transcript.evaluator_version);
    append_address(&bytes, transcript.evaluator_target_provenance);
    append_address(&bytes, transcript.environment_contact_input);
    append_address(&bytes, transcript.candidate_set_commitment);
    append_u64(&bytes, transcript.evaluator_transcript.size());
    for (const SelectionEvaluatorEvent& event : transcript.evaluator_transcript) {
        append_u32(&bytes, static_cast<std::uint32_t>(event.kind));
        append_u32(&bytes, event.candidate_index);
        append_u32(&bytes, event.compared_candidate_index);
        append_u64(&bytes, event.scores.size());
        for (const std::int64_t score : event.scores) {
            append_u64(&bytes, static_cast<std::uint64_t>(score));
        }
        append_u32(&bytes, event.rank);
        append_u32(&bytes, static_cast<std::uint32_t>(event.result));
    }
    append_actions(&bytes, transcript.action_alphabet);
    append_u32(&bytes, static_cast<std::uint32_t>(transcript.stopping_reason));
    append_u64(&bytes, transcript.actions.size());
    for (const SelectionActionRecord& action : transcript.actions) {
        append_u32(&bytes, static_cast<std::uint32_t>(action.action));
        append_u32(&bytes, action.candidate_index);
        append_u64(&bytes, action.multiplicity);
        append_u64(&bytes, action.compute_units);
        append_u64(&bytes, action.mate_indices.size());
        for (const std::uint32_t mate : action.mate_indices) append_u32(&bytes, mate);
    }
    return bytes;
}

ContentAddress selection_transcript_address(const SelectionTranscript& transcript) {
    const std::vector<std::byte> bytes = canonical_selection_transcript(transcript);
    return content_address(bytes);
}

bool validate_selection_transcript(
    const SelectionCandidateSet& candidate_set,
    const SelectionEvaluationRequest& request,
    const SelectionTranscript& transcript,
    std::string* error) {
    if (transcript.schema_version != SelectionTranscript::kSchemaVersion) {
        return fail(error, "BCC-32 selection transcript has an unsupported schema");
    }
    if (!validate_selection_evaluation_request(candidate_set, request, error)) {
        return false;
    }
    if (transcript.evaluation_request_commitment !=
            selection_evaluation_request_address(request) ||
        transcript.evaluator_identity != request.evaluator_identity ||
        transcript.evaluator_version != request.evaluator_version ||
        transcript.evaluator_target_provenance !=
            request.evaluator_target_provenance ||
        transcript.environment_contact_input !=
            request.environment_contact_input ||
        transcript.candidate_set_commitment !=
            request.candidate_set_commitment ||
        transcript.action_alphabet != request.action_alphabet) {
        return fail(error,
                    "BCC-32 selection transcript changes its precommitted request");
    }
    if (!is_valid_content_address(transcript.evaluation_request_commitment) ||
        !is_valid_content_address(transcript.evaluator_identity) ||
        !is_valid_content_address(transcript.evaluator_version) ||
        !is_valid_content_address(transcript.evaluator_target_provenance) ||
        !is_valid_content_address(transcript.environment_contact_input) ||
        !is_valid_content_address(transcript.candidate_set_commitment)) {
        return fail(error, "BCC-32 selection transcript has an unaddressed evaluator channel");
    }
    if (canonical_selection_transcript(transcript).size() >
            request.limits.maximum_transcript_bytes ||
        transcript.evaluator_transcript.size() >
            request.limits.maximum_evaluator_events ||
        transcript.actions.size() > request.limits.maximum_action_records) {
        return fail(error, "BCC-32 selection transcript exceeds its precommitted bound");
    }
    if (!validate_selection_candidate_set(candidate_set, error)) return false;
    if (selection_candidate_set_address(candidate_set) !=
        transcript.candidate_set_commitment) {
        return fail(error,
                    "BCC-32 selection transcript commitment does not match its candidate set");
    }
    const std::size_t candidate_count = candidate_set.candidates.size();

    std::vector<const SelectionEvaluatorEvent*> score_by_candidate(candidate_count, nullptr);
    std::size_t score_width = 0u;
    for (const SelectionEvaluatorEvent& event : transcript.evaluator_transcript) {
        if (!valid_evaluator_event_kind(event.kind) ||
            event.candidate_index >= candidate_count) {
            return fail(error, "BCC-32 selection transcript has an invalid evaluator event");
        }
        if (event.kind == SelectionEvaluatorEventKind::score) {
            if (event.compared_candidate_index != 0u || event.scores.empty() ||
                event.scores.size() > request.limits.maximum_score_dimensions ||
                event.rank >= candidate_count ||
                event.result != SelectionComparison::equal) {
                return fail(error, "BCC-32 selection transcript has a malformed score event");
            }
            if (score_width == 0u) score_width = event.scores.size();
            if (event.scores.size() != score_width) {
                return fail(error, "BCC-32 selection transcript score dimensions disagree");
            }
            score_by_candidate[event.candidate_index] = &event;
            continue;
        }
        if (event.compared_candidate_index >= candidate_count ||
            event.compared_candidate_index == event.candidate_index ||
            !event.scores.empty() || event.rank != 0u ||
            (event.result != SelectionComparison::less &&
             event.result != SelectionComparison::equal &&
             event.result != SelectionComparison::greater)) {
            return fail(error, "BCC-32 selection transcript has a malformed comparison event");
        }
        const SelectionEvaluatorEvent* left = score_by_candidate[event.candidate_index];
        const SelectionEvaluatorEvent* right =
            score_by_candidate[event.compared_candidate_index];
        if (left == nullptr || right == nullptr ||
            (event.result == SelectionComparison::greater && left->rank >= right->rank) ||
            (event.result == SelectionComparison::less && left->rank <= right->rank) ||
            (event.result == SelectionComparison::equal &&
             (left->scores != right->scores || left->rank != right->rank))) {
            return fail(error, "BCC-32 selection transcript comparison contradicts scores or ranks");
        }
    }
    if (std::any_of(score_by_candidate.begin(), score_by_candidate.end(),
                    [](const SelectionEvaluatorEvent* event) { return event == nullptr; })) {
        return fail(error, "BCC-32 selection transcript omits a candidate score event");
    }
    std::vector<bool> observed_rank(candidate_count, false);
    for (const SelectionEvaluatorEvent* score : score_by_candidate) {
        observed_rank[score->rank] = true;
    }
    for (std::size_t rank = 1u; rank < observed_rank.size(); ++rank) {
        if (observed_rank[rank] && !observed_rank[rank - 1u]) {
            return fail(error, "BCC-32 selection transcript final ranks are not dense");
        }
    }
    if (transcript.action_alphabet.empty() ||
        transcript.action_alphabet.size() >
            static_cast<std::size_t>(SelectionAction::stop) + 1u) {
        return fail(error, "BCC-32 selection transcript has an invalid action alphabet bound");
    }
    for (std::size_t index = 0u; index < transcript.action_alphabet.size(); ++index) {
        const SelectionAction action = transcript.action_alphabet[index];
        if (!valid_action(action) ||
            (index != 0u && static_cast<std::uint32_t>(transcript.action_alphabet[index - 1u]) >=
                                static_cast<std::uint32_t>(action))) {
            return fail(error, "BCC-32 selection transcript action alphabet is not sorted and unique");
        }
    }
    const bool has_continuation_channel = std::any_of(
        transcript.action_alphabet.begin(), transcript.action_alphabet.end(),
        [](SelectionAction action) { return is_continuation_action(action); });
    std::uint64_t action_capacity = has_continuation_channel ? candidate_count : 0u;
    for (const SelectionAction action : transcript.action_alphabet) {
        const bool per_candidate_channel =
            action == SelectionAction::allocate_compute ||
            action == SelectionAction::choose_mates;
        const bool retry_channel = action == SelectionAction::retry;
        const bool singleton_channel = action == SelectionAction::stop;
        const std::uint64_t increment =
            per_candidate_channel ? static_cast<std::uint64_t>(candidate_count)
            : (retry_channel ? kMaxSelectionIndexableCount
                             : (singleton_channel ? 1u : 0u));
        if (increment > std::numeric_limits<std::uint64_t>::max() - action_capacity) {
            return fail(error, "BCC-32 selection transcript action capacity overflows");
        }
        action_capacity += increment;
    }
    if (transcript.actions.size() > action_capacity) {
        return fail(error,
                    "BCC-32 selection transcript exceeds its declared action alphabet capacity");
    }
    if (!valid_stop_reason(transcript.stopping_reason) || transcript.actions.empty()) {
        return fail(error, "BCC-32 selection transcript has no declared stopping action");
    }

    std::vector<bool> continued(candidate_count, false);
    std::vector<bool> compute_allocated(candidate_count, false);
    std::vector<bool> mates_chosen(candidate_count, false);
    std::unordered_set<std::uint64_t> mating_edges;
    bool stop_logged = false;
    bool has_continuation = false;
    std::uint64_t next_population_slot_count = 0u;
    std::uint64_t total_compute_units = 0u;
    std::size_t total_mating_edges = 0u;
    std::uint64_t retry_count = 0u;
    bool left_retry_prefix = false;
    for (std::size_t index = 0u; index < transcript.actions.size(); ++index) {
        const SelectionActionRecord& action = transcript.actions[index];
        if (!valid_action(action.action) ||
            !contains_value(transcript.action_alphabet, action.action)) {
            return fail(error, "BCC-32 selection transcript uses an unlogged action channel");
        }
        if (action.mate_indices.size() > kMaxSelectionIndexableCount) {
            return fail(error, "BCC-32 selection transcript exceeds its mate-index bound");
        }
        const bool candidate_action = is_candidate_action(action.action);
        if (candidate_action && action.candidate_index >= candidate_count) {
            return fail(error,
                        "BCC-32 selection transcript allocation does not name a scored candidate");
        }

        if (is_continuation_action(action.action)) {
            left_retry_prefix = true;
            if (continued[action.candidate_index] || action.multiplicity == 0u ||
                action.compute_units != 0u || !action.mate_indices.empty()) {
                return fail(error, "BCC-32 selection transcript has a conflicting continuation");
            }
            if (action.multiplicity >
                std::numeric_limits<std::uint64_t>::max() -
                    next_population_slot_count) {
                return fail(error,
                            "BCC-32 selection transcript population multiplicity overflows its slot space");
            }
            next_population_slot_count += action.multiplicity;
            continued[action.candidate_index] = true;
            has_continuation = true;
            continue;
        }
        if (action.action == SelectionAction::allocate_compute) {
            left_retry_prefix = true;
            if (compute_allocated[action.candidate_index] || action.multiplicity != 0u ||
                action.compute_units == 0u || !action.mate_indices.empty()) {
                return fail(error, "BCC-32 selection transcript has a malformed compute allocation");
            }
            if (action.compute_units >
                std::numeric_limits<std::uint64_t>::max() - total_compute_units) {
                return fail(error,
                            "BCC-32 selection transcript compute allocation overflows");
            }
            total_compute_units += action.compute_units;
            compute_allocated[action.candidate_index] = true;
            continue;
        }
        if (action.action == SelectionAction::choose_mates) {
            left_retry_prefix = true;
            if (mates_chosen[action.candidate_index] || action.multiplicity != 0u ||
                action.compute_units != 0u || action.mate_indices.empty()) {
                return fail(error, "BCC-32 selection transcript has a malformed mating allocation");
            }
            std::uint32_t previous = 0u;
            bool have_previous = false;
            for (const std::uint32_t mate : action.mate_indices) {
                if (total_mating_edges == kMaxSelectionIndexableCount) {
                    return fail(error,
                                "BCC-32 selection transcript mating graph exceeds its aggregate bound");
                }
                if (mate >= candidate_count || mate == action.candidate_index ||
                    (have_previous && mate <= previous) ||
                    !mating_edges.insert(pair_key(action.candidate_index, mate)).second) {
                    return fail(error, "BCC-32 selection transcript has duplicate or invalid mates");
                }
                const SelectionMatingPair requested_pair{
                    .first_candidate_index = std::min(action.candidate_index, mate),
                    .second_candidate_index = std::max(action.candidate_index, mate),
                };
                if (!contains_value(request.allowed_mating_pairs, requested_pair)) {
                    return fail(error,
                                "BCC-32 selection transcript chooses an uncommitted mate");
                }
                ++total_mating_edges;
                previous = mate;
                have_previous = true;
            }
            mates_chosen[action.candidate_index] = true;
            continue;
        }

        const bool canonical_non_candidate = action.candidate_index == 0u &&
            action.multiplicity == 0u && action.compute_units == 0u &&
            action.mate_indices.empty();
        if (!canonical_non_candidate) {
            return fail(error, "BCC-32 selection transcript hides data in irrelevant action fields");
        }
        if (action.action == SelectionAction::retry) {
            if (left_retry_prefix || retry_count == request.limits.maximum_retries) {
                return fail(error,
                            "BCC-32 selection transcript has an invalid retry sequence");
            }
            ++retry_count;
            continue;
        }
        left_retry_prefix = true;
        if (stop_logged || index + 1u != transcript.actions.size()) {
            return fail(error, "BCC-32 selection transcript duplicates or continues after stop");
        }
        stop_logged = true;
    }
    if (!stop_logged || !has_continuation) {
        return fail(error, "BCC-32 selection transcript has inconsistent stop outcomes");
    }
    if (next_population_slot_count < request.limits.minimum_population_slots ||
        next_population_slot_count > request.limits.maximum_population_slots ||
        total_compute_units > request.limits.maximum_total_compute_units ||
        total_mating_edges > request.limits.maximum_mating_edges) {
        return fail(error,
                    "BCC-32 selection transcript exceeds a precommitted aggregate");
    }
    for (std::size_t candidate = 0u; candidate < candidate_count; ++candidate) {
        if ((compute_allocated[candidate] || mates_chosen[candidate]) && !continued[candidate]) {
            return fail(error, "BCC-32 selection transcript allocates an uncontinued candidate");
        }
    }
    for (const SelectionActionRecord& action : transcript.actions) {
        if (action.action != SelectionAction::choose_mates) continue;
        for (const std::uint32_t mate : action.mate_indices) {
            if (!continued[mate]) {
                return fail(error,
                            "BCC-32 selection transcript mates an uncontinued candidate");
            }
        }
    }
    return true;
}

bool decode_selection_transcript(std::span<const std::byte> bytes,
                                 SelectionTranscript* transcript,
                                 std::string* error) {
    if (transcript == nullptr) {
        return fail(error, "BCC-32 selection transcript decode requires an output");
    }
    CanonicalReader reader(bytes);
    SelectionTranscript decoded{};
    if (!reader.take_domain(kSelectionTranscriptDomain) ||
        !reader.take_u32(&decoded.schema_version) ||
        !reader.take_address(&decoded.evaluation_request_commitment) ||
        !reader.take_address(&decoded.evaluator_identity) ||
        !reader.take_address(&decoded.evaluator_version) ||
        !reader.take_address(&decoded.evaluator_target_provenance) ||
        !reader.take_address(&decoded.environment_contact_input) ||
        !reader.take_address(&decoded.candidate_set_commitment)) {
        return fail(error, "BCC-32 selection transcript has malformed canonical bytes");
    }

    std::size_t evaluator_event_count = 0u;
    if (!take_bounded_count(&reader, kMaxSelectionIndexableCount, 28u,
                            &evaluator_event_count)) {
        return fail(error, "BCC-32 selection transcript has an invalid evaluator count");
    }
    decoded.evaluator_transcript.reserve(evaluator_event_count);
    for (std::size_t index = 0u; index < evaluator_event_count; ++index) {
        SelectionEvaluatorEvent event{};
        std::uint32_t kind = 0u;
        if (!reader.take_u32(&kind) || !reader.take_u32(&event.candidate_index) ||
            !reader.take_u32(&event.compared_candidate_index)) {
            return fail(error, "BCC-32 selection transcript has a truncated evaluator event");
        }
        event.kind = static_cast<SelectionEvaluatorEventKind>(kind);
        std::size_t score_count = 0u;
        if (!take_bounded_count(&reader, kMaxSelectionIndexableCount,
                                sizeof(std::uint64_t), &score_count)) {
            return fail(error, "BCC-32 selection transcript has an invalid score count");
        }
        event.scores.reserve(score_count);
        for (std::size_t score_index = 0u; score_index < score_count; ++score_index) {
            std::uint64_t score_bits = 0u;
            if (!reader.take_u64(&score_bits)) {
                return fail(error, "BCC-32 selection transcript has a truncated score");
            }
            event.scores.push_back(std::bit_cast<std::int64_t>(score_bits));
        }
        std::uint32_t result_bits = 0u;
        if (!reader.take_u32(&event.rank) || !reader.take_u32(&result_bits)) {
            return fail(error, "BCC-32 selection transcript has a truncated evaluator result");
        }
        event.result = static_cast<SelectionComparison>(
            std::bit_cast<std::int32_t>(result_bits));
        decoded.evaluator_transcript.push_back(std::move(event));
    }

    std::size_t alphabet_count = 0u;
    if (!take_bounded_count(
            &reader, static_cast<std::size_t>(SelectionAction::stop) + 1u,
            sizeof(std::uint32_t), &alphabet_count)) {
        return fail(error, "BCC-32 selection transcript has an invalid action alphabet count");
    }
    decoded.action_alphabet.reserve(alphabet_count);
    for (std::size_t index = 0u; index < alphabet_count; ++index) {
        std::uint32_t action = 0u;
        if (!reader.take_u32(&action)) {
            return fail(error, "BCC-32 selection transcript has a truncated action alphabet");
        }
        decoded.action_alphabet.push_back(static_cast<SelectionAction>(action));
    }

    std::uint32_t stopping_reason = 0u;
    if (!reader.take_u32(&stopping_reason)) {
        return fail(error, "BCC-32 selection transcript has a truncated outcome");
    }
    decoded.stopping_reason = static_cast<SelectionStopReason>(stopping_reason);

    std::size_t action_count = 0u;
    if (!take_bounded_count(&reader, kMaxSelectionIndexableCount, 32u,
                            &action_count)) {
        return fail(error, "BCC-32 selection transcript has an invalid action count");
    }
    decoded.actions.reserve(action_count);
    for (std::size_t index = 0u; index < action_count; ++index) {
        SelectionActionRecord action{};
        std::uint32_t action_kind = 0u;
        if (!reader.take_u32(&action_kind) ||
            !reader.take_u32(&action.candidate_index) ||
            !reader.take_u64(&action.multiplicity) ||
            !reader.take_u64(&action.compute_units)) {
            return fail(error, "BCC-32 selection transcript has a truncated action");
        }
        action.action = static_cast<SelectionAction>(action_kind);
        std::size_t mate_count = 0u;
        if (!take_bounded_count(&reader, kMaxSelectionIndexableCount,
                                sizeof(std::uint32_t), &mate_count)) {
            return fail(error, "BCC-32 selection transcript has an invalid mate count");
        }
        action.mate_indices.reserve(mate_count);
        for (std::size_t mate_index = 0u; mate_index < mate_count; ++mate_index) {
            std::uint32_t mate = 0u;
            if (!reader.take_u32(&mate)) {
                return fail(error, "BCC-32 selection transcript has a truncated mate index");
            }
            action.mate_indices.push_back(mate);
        }
        decoded.actions.push_back(std::move(action));
    }
    if (!reader.done()) {
        return fail(error, "BCC-32 selection transcript has trailing canonical bytes");
    }
    if (!validate_decoded_selection_transcript_shape(decoded, error)) return false;
    if (!canonical_bytes_equal(bytes, canonical_selection_transcript(decoded))) {
        return fail(error, "BCC-32 selection transcript is not canonically encoded");
    }
    *transcript = std::move(decoded);
    return true;
}

bool replay_selection_transcript(
    const SelectionCandidateSet& candidate_set,
    const SelectionEvaluationRequest& request,
    const SelectionTranscript& transcript,
    SelectionReplay* replay,
    std::string* error) {
    if (replay == nullptr) return fail(error, "BCC-32 selection replay requires an output");
    if (!validate_selection_transcript(candidate_set, request, transcript, error)) {
        return false;
    }
    SelectionReplay derived{};
    derived.stopping_reason = transcript.stopping_reason;
    for (const SelectionActionRecord& action : transcript.actions) {
        const SelectionCandidate* candidate =
            is_candidate_action(action.action)
                ? &candidate_set.candidates[action.candidate_index]
                : nullptr;
        if (is_continuation_action(action.action)) {
            derived.next_population.push_back({
                .action = action.action,
                .candidate_index = action.candidate_index,
                .root = candidate->root,
                .multiplicity = action.multiplicity,
            });
            derived.next_population_slot_count += action.multiplicity;
        } else if (action.action == SelectionAction::allocate_compute) {
            derived.compute_allocations.push_back({
                .candidate_index = action.candidate_index,
                .root = candidate->root,
                .compute_units = action.compute_units,
            });
            derived.total_compute_units += action.compute_units;
        } else if (action.action == SelectionAction::choose_mates) {
            for (const std::uint32_t mate_index : action.mate_indices) {
                derived.mating_graph.push_back({
                    .parent_candidate_index = action.candidate_index,
                    .mate_candidate_index = mate_index,
                    .parent_root = candidate->root,
                    .mate_root = candidate_set.candidates[mate_index].root,
                });
            }
        } else if (action.action == SelectionAction::retry) {
            ++derived.retry_count;
        } else if (action.action == SelectionAction::stop) {
            derived.stopped = true;
        }
    }
    *replay = std::move(derived);
    return true;
}

bool resolve_selection_population_slot(const SelectionReplay& replay,
                                       std::uint64_t slot,
                                       SelectionPopulationSlot* population_slot,
                                       std::string* error) {
    if (population_slot == nullptr) {
        return fail(error, "BCC-32 selection slot resolution requires an output");
    }
    if (slot >= replay.next_population_slot_count) {
        return fail(error, "BCC-32 selection population slot is out of range");
    }
    std::uint64_t actual_slot_count = 0u;
    for (const SelectionContinuation& continuation : replay.next_population) {
        if (continuation.multiplicity == 0u ||
            continuation.multiplicity >
                std::numeric_limits<std::uint64_t>::max() - actual_slot_count) {
            return fail(error,
                        "BCC-32 selection replay has an invalid population multiplicity");
        }
        actual_slot_count += continuation.multiplicity;
    }
    if (actual_slot_count != replay.next_population_slot_count) {
        return fail(error, "BCC-32 selection replay has an inconsistent population slot total");
    }
    std::uint64_t first_slot = 0u;
    for (const SelectionContinuation& continuation : replay.next_population) {
        if (slot - first_slot < continuation.multiplicity) {
            *population_slot = {
                .slot = slot,
                .action = continuation.action,
                .candidate_index = continuation.candidate_index,
                .root = continuation.root,
                .branch_ordinal = slot - first_slot,
            };
            return true;
        }
        first_slot += continuation.multiplicity;
    }
    return fail(error, "BCC-32 selection replay has an inconsistent population slot total");
}

bool validate_provenance(const Provenance& provenance, std::string* error) {
    if (provenance.genesis_class != GenesisClass::G0 &&
        provenance.genesis_class != GenesisClass::G1 &&
        provenance.genesis_class != GenesisClass::G2) {
        return fail(error, "BCC-32 provenance has an invalid genesis class");
    }
    if ((provenance.genesis_class == GenesisClass::G0 ||
         provenance.genesis_class == GenesisClass::G1) &&
        !provenance.biological_parents.empty()) {
        return fail(error, "BCC-32 G0 and G1 are independent parentless origins");
    }
    if (provenance.genesis_class == GenesisClass::G2 &&
        provenance.biological_parents.empty()) {
        return fail(error, "BCC-32 G2 requires an eligible biological parent");
    }
    if (!is_valid_content_address(provenance.law) ||
        !is_valid_content_address(provenance.genesis) ||
        !is_valid_content_address(provenance.environment_contact_manifest) ||
        !is_valid_content_address(provenance.replay_commitment)) {
        return fail(error, "BCC-32 provenance has an unaddressed required input");
    }
    if (!validate_entry_event_shape(provenance, error)) return false;
    for (const ContentAddress& parent : provenance.biological_parents) {
        if (!is_valid_content_address(parent)) {
            return fail(error, "BCC-32 provenance has an unaddressed biological parent");
        }
    }
    for (const ContentAddress& input : provenance.causal_inputs) {
        if (!is_valid_content_address(input)) {
            return fail(error, "BCC-32 provenance has an unaddressed causal input");
        }
    }
    if (has_duplicate(provenance.biological_parents) ||
        has_duplicate(provenance.causal_inputs)) {
        return fail(error, "BCC-32 provenance repeats a causal identity");
    }
    for (const ContentAddress& parent : provenance.biological_parents) {
        if (contains(provenance.causal_inputs, parent)) {
            return fail(error, "BCC-32 biological parent is also labeled as a causal input");
        }
    }
    if (provenance.entry_event.kind == EntryEventKind::evaluator_selection &&
        (contains(provenance.biological_parents,
                  provenance.entry_event.evaluator_transcript) ||
         contains(provenance.causal_inputs,
                  provenance.entry_event.evaluator_transcript))) {
        return fail(error, "BCC-32 evaluator entry is not parentage or a causal artifact");
    }
    return true;
}

bool validate_provenance(const Provenance& provenance,
                         const SelectionCandidateSet& candidate_set,
                         const SelectionEvaluationRequest& request,
                         const SelectionTranscript& selection_transcript,
                         std::string* error) {
    if (!validate_provenance(provenance, error) ||
        !validate_selection_transcript(
            candidate_set, request, selection_transcript, error)) {
        return false;
    }
    if (provenance.entry_event.kind != EntryEventKind::evaluator_selection ||
        provenance.entry_event.evaluator_transcript !=
            selection_transcript_address(selection_transcript)) {
        return fail(error, "BCC-32 provenance selection address does not bind its transcript");
    }
    SelectionReplay replay{};
    if (!replay_selection_transcript(
            candidate_set, request, selection_transcript, &replay, error)) {
        return false;
    }
    SelectionPopulationSlot selected{};
    if (!resolve_selection_population_slot(
            replay, provenance.entry_event.next_population_slot, &selected, error)) {
        return false;
    }
    return true;
}

}  // namespace substrate::bcc32
