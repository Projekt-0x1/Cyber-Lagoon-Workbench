#pragma once

#include "hardware_native/bcc32_organism_package_payload_ingress.cuh"
#include "hardware_native/bcc32_source_route_contact_adapter.cuh"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace substrate::bcc32::package_source_route {
template <typename BodyVerifier>
class ResidentBoundSourceRouteAuthority;
}

namespace substrate::bcc32::organism_package_payload_ingress {

// Only a body-route verifier may manufacture the canonical receipt.  A
// transport digest or a package hash is deliberately insufficient.
class CanonicalBodySourceRoute final {
 private:
  static SourceAttestationReceipt attest(
      const ValidatedPackageFrame& frame, std::uint64_t observed_payload_hash) {
    return SourceAttestationReceipt(
        std::string(frame.source_package_sha256),
        std::string(frame.source_stream_sha256), frame.sequence,
        observed_payload_hash);
  }

  template <typename BodyVerifier>
  friend class substrate::bcc32::package_source_route::ResidentBoundSourceRouteAuthority;
};

}  // namespace substrate::bcc32::organism_package_payload_ingress

namespace substrate::bcc32::package_source_route {

namespace ingress = organism_package_payload_ingress;
namespace route = source_route_contact;

struct SourceRoutePackageBinding final {
  std::uint64_t source_epoch = 0u;
  std::uint64_t stream_id = 0u;
  std::string source_package_sha256;
  std::string source_stream_sha256;
};

// The committed source-route receipt binds ordered route identity and the
// exact observed payload digest. The existing route remains fixture-only:
// its own contract explicitly does not authenticate physical origin.
class CommittedSourceRouteAuthority final
    : public ingress::SourceAttestationAuthority {
 public:
  CommittedSourceRouteAuthority(route::SourceRouteReceipt receipt,
                                SourceRoutePackageBinding binding)
      : receipt_(receipt), binding_(std::move(binding)) {}

  [[nodiscard]] bool matches_transport(
      const ingress::ValidatedPackageFrame& frame,
      std::uint64_t observed_payload_hash) const noexcept {
    return receipt_.status == route::SourceRouteStatus::committed &&
           receipt_.source_epoch == binding_.source_epoch &&
           receipt_.stream_id == binding_.stream_id &&
           frame.source_package_sha256 == binding_.source_package_sha256 &&
           frame.source_stream_sha256 == binding_.source_stream_sha256 &&
           receipt_.contact_sequence == frame.sequence &&
           receipt_.validator_payload_hash == observed_payload_hash;
  }

  [[nodiscard]] ingress::SourceAttestationReceipt attest(
      const ingress::ValidatedPackageFrame& frame,
      std::span<const std::uint8_t> bytes,
      std::uint64_t observed_payload_hash) override {
    (void)frame;
    (void)bytes;
    (void)observed_payload_hash;
    // Transport receipt identity is retained and can be checked, but this
    // public adapter cannot mint a canonical attestation.
    return {};
  }

  [[nodiscard]] const route::SourceRouteReceipt& receipt() const noexcept {
    return receipt_;
  }

 private:
  route::SourceRouteReceipt receipt_{};
  SourceRoutePackageBinding binding_{};
};

// This adapter is the production seam between validated raw bytes and the
// resident boundary.  It waits for the resident's own contact sequence rather
// than treating enqueue success as assimilation.  The wait policy is supplied
// by the coordinator/runtime because PersistentKernel's clock is device-owned.
template <typename Resident, typename WaitForContact>
class PersistentKernelRawContactSink final
    : public ingress::ResidentRawContactSink {
 public:
  PersistentKernelRawContactSink(Resident& resident, WaitForContact wait_for_contact)
      : resident_(resident), wait_for_contact_(std::move(wait_for_contact)) {}

  [[nodiscard]] ingress::ResidentAssimilationReceipt present_raw(
      std::span<const std::uint8_t> bytes) override {
    constexpr std::size_t kPacketWords = Resident::kMaximumRawContactWords;
    static_assert(kPacketWords != 0u);
    if (bytes.empty())
      return {false, 0u, "resident_contact_unconfirmed"};

    const std::uint64_t before = resident_.read_receipt().contact_sequence;
    // One source byte becomes one unlabelled boundary word. This retains every
    // byte, including arbitrary UTF-8 tails, and avoids inventing padding
    // bytes or four-byte-alignment admission rules for natural language.
    std::vector<std::uint32_t> words(bytes.size(), 0u);
    for (std::size_t index = 0u; index < words.size(); ++index) {
      words[index] = static_cast<std::uint32_t>(bytes[index]);
    }

    resident_.present_raw(
        std::span<const std::uint32_t>(words.data(), words.size()));
    const std::uint64_t expected = before +
        static_cast<std::uint64_t>((words.size() + kPacketWords - 1u) /
                                   kPacketWords);
    const std::uint64_t after = wait_for_contact_(resident_, expected);
    if (after < expected)
      return {false, after, "resident_contact_unconfirmed"};
    return {true, after, "resident_contact_stream_delta"};
  }

 private:
  Resident& resident_;
  WaitForContact wait_for_contact_;
};

// A committed transport receipt becomes a canonical source attestation only
// when an independent body verifier confirms the exact source identity,
// sequence, and payload after the body route has observed the contact.  The
// verifier is intentionally a required dependency, not a boolean field on the
// frame and not a hash comparison performed by this adapter.
template <typename BodyVerifier>
class ResidentBoundSourceRouteAuthority final
    : public ingress::ObservedSourceAttestationAuthority {
 public:
  ResidentBoundSourceRouteAuthority(route::SourceRouteReceipt receipt,
                                    SourceRoutePackageBinding binding,
                                    BodyVerifier verifier)
      : receipt_(receipt), binding_(std::move(binding)),
        verifier_(std::move(verifier)) {}

  [[nodiscard]] ingress::SourceAttestationReceipt attest(
      const ingress::ValidatedPackageFrame& frame,
      std::span<const std::uint8_t> bytes,
      std::uint64_t observed_payload_hash) override {
    (void)frame;
    (void)bytes;
    (void)observed_payload_hash;
    // The legacy path authenticates before the resident has observed the
    // bytes. It remains available only as an explicit fixture boundary.
    return {};
  }

  [[nodiscard]] ingress::SourceAttestationReceipt attest_after_observation(
      const ingress::ValidatedPackageFrame& frame,
      std::span<const std::uint8_t> bytes,
      std::uint64_t observed_payload_hash,
      const ingress::ResidentObservationReceipt& observation) override {
    if (!matches_transport(frame, observed_payload_hash) ||
        !observation.observed ||
        observation.observed_payload_hash != observed_payload_hash ||
        !verifier_(frame, bytes, observed_payload_hash, observation))
      return {};
    return ingress::CanonicalBodySourceRoute::attest(frame,
                                                      observed_payload_hash);
  }

 private:
  [[nodiscard]] bool matches_transport(
      const ingress::ValidatedPackageFrame& frame,
      std::uint64_t observed_payload_hash) const noexcept {
    return receipt_.status == route::SourceRouteStatus::committed &&
           receipt_.source_epoch == binding_.source_epoch &&
           receipt_.stream_id == binding_.stream_id &&
           frame.source_package_sha256 == binding_.source_package_sha256 &&
           frame.source_stream_sha256 == binding_.source_stream_sha256 &&
           receipt_.contact_sequence == frame.sequence &&
           receipt_.validator_payload_hash == observed_payload_hash;
  }

  route::SourceRouteReceipt receipt_{};
  SourceRoutePackageBinding binding_{};
  BodyVerifier verifier_;
};

}  // namespace substrate::bcc32::package_source_route
