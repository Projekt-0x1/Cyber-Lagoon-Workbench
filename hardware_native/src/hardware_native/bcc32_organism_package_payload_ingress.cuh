#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace substrate::bcc32::organism_package_payload_ingress {

// This frame is produced after package validation. Metadata identifies the
// stream and orders the frame; only decoded payload bytes cross the callback.
// The hashes are still declarations at this boundary: a physical body route
// must attest them before any learning claim is possible.
struct ValidatedPackageFrame final {
  std::string_view source_package_sha256;
  std::string_view source_stream_sha256;
  std::uint64_t sequence = 0u;
  std::string_view payload_hex;
  std::uint64_t expected_payload_hash = 0u;
};

struct IngressCursor final {
  std::string source_package_sha256;
  std::string source_stream_sha256;
  std::uint64_t next_sequence = 1u;
};

enum class ProjectionStatus : std::uint8_t {
  resident_assimilated,
  resident_not_assimilated,
  resident_not_observed,
  observation_payload_hash_mismatch,
  source_not_attested,
  source_attestation_mismatch,
  source_not_canonical,
  missing_source_package_hash,
  missing_source_stream_hash,
  package_identity_changed,
  stream_identity_changed,
  invalid_sequence,
  whole_record_or_invalid_payload,
  payload_hash_mismatch,
};

struct IngressReceipt final {
  std::string source_package_sha256;
  std::string source_stream_sha256;
  std::uint64_t sequence = 0u;
  std::string projection = "payload_hex_only";
  std::string ingress_route = "resident_raw_contact_api";
  std::string state_delta = "UNPROVEN";
  ProjectionStatus status = ProjectionStatus::missing_source_package_hash;
  std::string source_authority = "UNPROVEN";
  bool source_attestation_present = false;
  bool canonical_source_attested = false;
  bool transport_accepted = false;
  bool resident_observed = false;
  bool resident_assimilated = false;
  std::uint64_t resident_observation_sequence = 0u;
  std::uint64_t resident_contact_sequence = 0u;

  [[nodiscard]] std::string to_json() const {
    return "{\"source_package_sha256\":\"" + source_package_sha256 +
           "\",\"source_stream_sha256\":\"" + source_stream_sha256 +
           "\",\"sequence\":" + std::to_string(sequence) +
           ",\"projection\":\"" + projection +
           "\",\"ingress_route\":\"" + ingress_route +
           "\",\"state_delta\":\"" + state_delta +
           "\",\"status\":\"" + status_name() +
           "\",\"source_authority\":\"" + source_authority +
           "\",\"source_attestation_present\":" +
           (source_attestation_present ? "true" : "false") +
           ",\"canonical_source_attested\":" +
           (canonical_source_attested ? "true" : "false") +
           ",\"transport_accepted\":" +
           (transport_accepted ? "true" : "false") +
           ",\"resident_observed\":" +
           (resident_observed ? "true" : "false") +
           ",\"resident_assimilated\":" +
           (resident_assimilated ? "true" : "false") +
           ",\"resident_observation_sequence\":" +
           std::to_string(resident_observation_sequence) +
           ",\"resident_contact_sequence\":" +
           std::to_string(resident_contact_sequence) + "}";
  }

 private:
  [[nodiscard]] const char* status_name() const noexcept {
    switch (status) {
      case ProjectionStatus::resident_assimilated:
        return "resident_assimilated";
      case ProjectionStatus::resident_not_assimilated:
        return "resident_not_assimilated";
      case ProjectionStatus::resident_not_observed:
        return "resident_not_observed";
      case ProjectionStatus::observation_payload_hash_mismatch:
        return "observation_payload_hash_mismatch";
      case ProjectionStatus::source_not_attested:
        return "source_not_attested";
      case ProjectionStatus::source_attestation_mismatch:
        return "source_attestation_mismatch";
      case ProjectionStatus::source_not_canonical:
        return "source_not_canonical";
      case ProjectionStatus::missing_source_package_hash:
        return "missing_source_package_hash";
      case ProjectionStatus::missing_source_stream_hash:
        return "missing_source_stream_hash";
      case ProjectionStatus::package_identity_changed:
        return "package_identity_changed";
      case ProjectionStatus::stream_identity_changed:
        return "stream_identity_changed";
      case ProjectionStatus::invalid_sequence: return "invalid_sequence";
      case ProjectionStatus::whole_record_or_invalid_payload:
        return "whole_record_or_invalid_payload";
      case ProjectionStatus::payload_hash_mismatch:
        return "payload_hash_mismatch";
    }
    return "unknown";
  }
};

struct ResidentAssimilationReceipt final {
  bool committed = false;
  std::uint64_t resident_contact_sequence = 0u;
  std::string state_delta = "UNPROVEN";
};

// Observation is deliberately separate from assimilation. A body route may
// observe a raw contact and expose only a bounded receipt to the independent
// source verifier; it must not claim learning until the verified observation
// is committed by the resident.
struct ResidentObservationReceipt final {
  bool observed = false;
  std::uint64_t resident_contact_sequence = 0u;
  std::uint64_t observed_payload_hash = 0u;
  std::string state_delta = "UNPROVEN";
};

class CanonicalBodySourceRoute;

class SourceAttestationReceipt final {
 public:
  SourceAttestationReceipt() = default;

  [[nodiscard]] bool source_attested() const noexcept {
    return source_attested_;
  }
  [[nodiscard]] bool canonical_body() const noexcept {
    return canonical_body_;
  }
  [[nodiscard]] const std::string& source_package_sha256() const noexcept {
    return source_package_sha256_;
  }
  [[nodiscard]] const std::string& source_stream_sha256() const noexcept {
    return source_stream_sha256_;
  }
  [[nodiscard]] std::uint64_t sequence() const noexcept { return sequence_; }
  [[nodiscard]] std::uint64_t payload_hash() const noexcept {
    return payload_hash_;
  }

 private:
  friend class CanonicalBodySourceRoute;
  SourceAttestationReceipt(std::string package_hash, std::string stream_hash,
                           std::uint64_t sequence, std::uint64_t payload_hash)
      : source_attested_(true),
        canonical_body_(true),
        source_package_sha256_(std::move(package_hash)),
        source_stream_sha256_(std::move(stream_hash)),
        sequence_(sequence),
        payload_hash_(payload_hash) {}

  bool source_attested_ = false;
  bool canonical_body_ = false;
  std::string source_package_sha256_;
  std::string source_stream_sha256_;
  std::uint64_t sequence_ = 0u;
  std::uint64_t payload_hash_ = 0u;
};

struct SourceAttestationAuthority {
  virtual ~SourceAttestationAuthority() = default;
  [[nodiscard]] virtual SourceAttestationReceipt attest(
      const ValidatedPackageFrame& frame, std::span<const std::uint8_t> bytes,
      std::uint64_t observed_payload_hash) = 0;
};

struct ResidentRawContactSink {
  virtual ~ResidentRawContactSink() = default;
  [[nodiscard]] virtual ResidentAssimilationReceipt present_raw(
      std::span<const std::uint8_t> bytes) = 0;
};

struct ObservedResidentRawContactSink {
  virtual ~ObservedResidentRawContactSink() = default;
  [[nodiscard]] virtual ResidentObservationReceipt observe_raw(
      std::span<const std::uint8_t> bytes) = 0;
  [[nodiscard]] virtual ResidentAssimilationReceipt commit_observed(
      std::span<const std::uint8_t> bytes,
      const ResidentObservationReceipt& observation) = 0;
};

// This authority cannot be used until the body has returned an observation
// receipt. The older SourceAttestationAuthority path remains for transport
// fixture contracts only and is not a resident-bound cognition claim.
struct ObservedSourceAttestationAuthority : SourceAttestationAuthority {
  ~ObservedSourceAttestationAuthority() override = default;
  [[nodiscard]] virtual SourceAttestationReceipt attest_after_observation(
      const ValidatedPackageFrame& frame, std::span<const std::uint8_t> bytes,
      std::uint64_t observed_payload_hash,
      const ResidentObservationReceipt& observation) = 0;
};

[[nodiscard]] inline bool is_sha256_hex(std::string_view value) noexcept {
  if (value.size() != 64u) return false;
  for (const char digit : value) {
    if (!((digit >= '0' && digit <= '9') ||
          (digit >= 'a' && digit <= 'f')))
      return false;
  }
  return true;
}

[[nodiscard]] inline int hex_digit(char value) noexcept {
  if (value >= '0' && value <= '9') return value - '0';
  if (value >= 'a' && value <= 'f') return value - 'a' + 10;
  return -1;
}

// The package format permits ASCII whitespace between byte pairs. Any other
// non-hex character, including a serialized record, is rejected.
[[nodiscard]] inline bool decode_payload_hex(
    std::string_view payload_hex, std::vector<std::uint8_t>* bytes) {
  if (bytes == nullptr || payload_hex.empty()) return false;
  bytes->clear();
  int high = -1;
  for (const char value : payload_hex) {
    if (value == ' ' || value == '\t' || value == '\n' || value == '\r')
      continue;
    const int nibble = hex_digit(value);
    if (nibble < 0) {
      bytes->clear();
      return false;
    }
    if (high < 0) high = nibble;
    else {
      bytes->push_back(static_cast<std::uint8_t>((high << 4) | nibble));
      high = -1;
    }
  }
  if (high >= 0 || bytes->empty()) {
    bytes->clear();
    return false;
  }
  return true;
}

[[nodiscard]] inline std::uint64_t payload_hash(
    std::span<const std::uint8_t> bytes) noexcept {
  std::uint64_t hash = 1469598103934665603ull;
  for (const std::uint8_t byte : bytes) {
    hash ^= byte;
    hash *= 1099511628211ull;
  }
  return hash;
}

// This remains an admission projection, not a resident learner. Cursor state
// rejects duplicate/replay, gap, package-swap, and stream-swap frames before
// the typed sink. A future body adapter must attest identity and bind this call
// to PersistentKernel before state_delta can be promoted from UNPROVEN.
[[nodiscard]] IngressReceipt project_validated_payload(
    const ValidatedPackageFrame& frame, IngressCursor* cursor,
    SourceAttestationAuthority& source_authority,
    ResidentRawContactSink& resident_raw_contact) {
  IngressReceipt receipt;
  receipt.source_package_sha256 = std::string(frame.source_package_sha256);
  receipt.source_stream_sha256 = std::string(frame.source_stream_sha256);
  receipt.sequence = frame.sequence;
  if (!is_sha256_hex(frame.source_package_sha256)) return receipt;
  if (!is_sha256_hex(frame.source_stream_sha256)) {
    receipt.status = ProjectionStatus::missing_source_stream_hash;
    return receipt;
  }
  if (cursor == nullptr) {
    receipt.status = ProjectionStatus::invalid_sequence;
    return receipt;
  }
  if (!cursor->source_package_sha256.empty() &&
      cursor->source_package_sha256 != frame.source_package_sha256) {
    receipt.status = ProjectionStatus::package_identity_changed;
    return receipt;
  }
  if (!cursor->source_stream_sha256.empty() &&
      cursor->source_stream_sha256 != frame.source_stream_sha256) {
    receipt.status = ProjectionStatus::stream_identity_changed;
    return receipt;
  }
  if (frame.sequence != cursor->next_sequence) {
    receipt.status = ProjectionStatus::invalid_sequence;
    return receipt;
  }
  std::vector<std::uint8_t> bytes;
  if (!decode_payload_hex(frame.payload_hex, &bytes)) {
    receipt.status = ProjectionStatus::whole_record_or_invalid_payload;
    return receipt;
  }
  if (payload_hash(bytes) != frame.expected_payload_hash) {
    receipt.status = ProjectionStatus::payload_hash_mismatch;
    return receipt;
  }
  const SourceAttestationReceipt attestation = source_authority.attest(
      frame, std::span<const std::uint8_t>(bytes.data(), bytes.size()),
      frame.expected_payload_hash);
  receipt.source_authority =
      attestation.canonical_body() ? "canonical_body" : "UNPROVEN";
  receipt.source_attestation_present = attestation.source_attested();
  if (!receipt.source_attestation_present) {
    receipt.status = ProjectionStatus::source_not_attested;
    return receipt;
  }
  if (attestation.source_package_sha256() != frame.source_package_sha256 ||
      attestation.source_stream_sha256() != frame.source_stream_sha256 ||
      attestation.sequence() != frame.sequence ||
      attestation.payload_hash() != frame.expected_payload_hash) {
    receipt.status = ProjectionStatus::source_attestation_mismatch;
    return receipt;
  }
  if (!attestation.canonical_body()) {
    receipt.status = ProjectionStatus::source_not_canonical;
    return receipt;
  }
  receipt.canonical_source_attested = true;
  receipt.transport_accepted = true;
  const ResidentAssimilationReceipt assimilation = resident_raw_contact.present_raw(
      std::span<const std::uint8_t>(bytes.data(), bytes.size()));
  if (!assimilation.committed) {
    receipt.status = ProjectionStatus::resident_not_assimilated;
    return receipt;
  }
  receipt.resident_assimilated = true;
  receipt.resident_contact_sequence = assimilation.resident_contact_sequence;
  receipt.state_delta = assimilation.state_delta;
  if (cursor->source_package_sha256.empty())
    cursor->source_package_sha256 = std::string(frame.source_package_sha256);
  if (cursor->source_stream_sha256.empty())
    cursor->source_stream_sha256 = std::string(frame.source_stream_sha256);
  ++cursor->next_sequence;
  receipt.status = ProjectionStatus::resident_assimilated;
  return receipt;
}

// Production-shaped ordering for a body-bound route:
// raw bytes -> resident observation -> independent source attestation ->
// resident commit -> cursor advancement. No host-side label, answer, or
// semantic target is accepted by this seam. The concrete body adapter still
// has to provide real physical provenance; this function only prevents the
// coordinator from claiming that transport admission was cognition.
[[nodiscard]] IngressReceipt project_validated_payload_after_observation(
    const ValidatedPackageFrame& frame, IngressCursor* cursor,
    ObservedSourceAttestationAuthority& source_authority,
    ObservedResidentRawContactSink& resident_raw_contact) {
  IngressReceipt receipt;
  receipt.projection = "payload_hex_only_after_body_observation";
  receipt.source_package_sha256 = std::string(frame.source_package_sha256);
  receipt.source_stream_sha256 = std::string(frame.source_stream_sha256);
  receipt.sequence = frame.sequence;
  if (!is_sha256_hex(frame.source_package_sha256)) return receipt;
  if (!is_sha256_hex(frame.source_stream_sha256)) {
    receipt.status = ProjectionStatus::missing_source_stream_hash;
    return receipt;
  }
  if (cursor == nullptr) {
    receipt.status = ProjectionStatus::invalid_sequence;
    return receipt;
  }
  if (!cursor->source_package_sha256.empty() &&
      cursor->source_package_sha256 != frame.source_package_sha256) {
    receipt.status = ProjectionStatus::package_identity_changed;
    return receipt;
  }
  if (!cursor->source_stream_sha256.empty() &&
      cursor->source_stream_sha256 != frame.source_stream_sha256) {
    receipt.status = ProjectionStatus::stream_identity_changed;
    return receipt;
  }
  if (frame.sequence != cursor->next_sequence) {
    receipt.status = ProjectionStatus::invalid_sequence;
    return receipt;
  }
  std::vector<std::uint8_t> bytes;
  if (!decode_payload_hex(frame.payload_hex, &bytes)) {
    receipt.status = ProjectionStatus::whole_record_or_invalid_payload;
    return receipt;
  }
  if (payload_hash(bytes) != frame.expected_payload_hash) {
    receipt.status = ProjectionStatus::payload_hash_mismatch;
    return receipt;
  }

  const ResidentObservationReceipt observation = resident_raw_contact.observe_raw(
      std::span<const std::uint8_t>(bytes.data(), bytes.size()));
  receipt.resident_observed = observation.observed;
  receipt.resident_observation_sequence =
      observation.resident_contact_sequence;
  if (!observation.observed) {
    receipt.status = ProjectionStatus::resident_not_observed;
    return receipt;
  }
  if (observation.resident_contact_sequence == 0u ||
      observation.observed_payload_hash != frame.expected_payload_hash) {
    receipt.status = ProjectionStatus::observation_payload_hash_mismatch;
    return receipt;
  }

  const SourceAttestationReceipt attestation =
      source_authority.attest_after_observation(
          frame, std::span<const std::uint8_t>(bytes.data(), bytes.size()),
          frame.expected_payload_hash, observation);
  receipt.source_authority =
      attestation.canonical_body() ? "canonical_body" : "UNPROVEN";
  receipt.source_attestation_present = attestation.source_attested();
  if (!receipt.source_attestation_present) {
    receipt.status = ProjectionStatus::source_not_attested;
    return receipt;
  }
  if (attestation.source_package_sha256() != frame.source_package_sha256 ||
      attestation.source_stream_sha256() != frame.source_stream_sha256 ||
      attestation.sequence() != frame.sequence ||
      attestation.payload_hash() != frame.expected_payload_hash) {
    receipt.status = ProjectionStatus::source_attestation_mismatch;
    return receipt;
  }
  if (!attestation.canonical_body()) {
    receipt.status = ProjectionStatus::source_not_canonical;
    return receipt;
  }
  receipt.canonical_source_attested = true;
  receipt.transport_accepted = true;
  const ResidentAssimilationReceipt assimilation =
      resident_raw_contact.commit_observed(
          std::span<const std::uint8_t>(bytes.data(), bytes.size()),
          observation);
  if (!assimilation.committed) {
    receipt.status = ProjectionStatus::resident_not_assimilated;
    return receipt;
  }
  receipt.resident_assimilated = true;
  receipt.resident_contact_sequence = assimilation.resident_contact_sequence;
  receipt.state_delta = assimilation.state_delta;
  if (cursor->source_package_sha256.empty())
    cursor->source_package_sha256 = std::string(frame.source_package_sha256);
  if (cursor->source_stream_sha256.empty())
    cursor->source_stream_sha256 = std::string(frame.source_stream_sha256);
  ++cursor->next_sequence;
  receipt.status = ProjectionStatus::resident_assimilated;
  return receipt;
}

}  // namespace substrate::bcc32::organism_package_payload_ingress
