#ifndef HARDWARE_NATIVE_DIRECT_CONTINUITY_SEAL_CUH
#define HARDWARE_NATIVE_DIRECT_CONTINUITY_SEAL_CUH

#include <cstdint>

#include "hardware_native/direct_content_address.cuh"
#include "hardware_native/direct_network_brain.cuh"

#if defined(__CUDACC__)
#define DIRECT_CONTINUITY_HD __host__ __device__
#else
#define DIRECT_CONTINUITY_HD
#endif

namespace substrate::direct_network {

inline constexpr std::uint32_t kDirectContinuitySealAbiV2 = 2u;

// github #1434 e.long_horizon_continuity: one content-addressed seal over
// everything a continuing subject is -- authored birth identity plus current
// learned state. Equal seals name the same continuing subject across
// transport discontinuities; any lived change moves the learned component,
// so a stale seal can never vouch for a later self. The seal is evidence
// bookkeeping, not cognition: nothing consults it at runtime.
struct DirectContinuitySealV2 {
  std::uint32_t abi_version;
  Root256 birth_identity;
  Root256 learned_state;
  DirectSha256Address seal;
};

// A checkpoint can be restored more than once, producing sibling executions.
// Designation therefore belongs to an explicit one-shot continuation edge,
// not to the copied checkpoint bytes.  This host-side linear authority is
// observer genealogy only: it cannot alter resident state or causal credit.
struct DirectDesignatedContinuationEdgeV1 {
  Root256 birth_identity{};
  DirectSha256Address predecessor_identity{};
  DirectSha256Address parent_checkpoint_digest{};
  DirectSha256Address continuation_identity{};
  std::uint64_t generation = 0u;
  std::uint32_t designated = 0u;
  std::uint32_t reserved = 0u;
};

class DirectDesignatedContinuationAuthorityV1 {
 public:
  DirectDesignatedContinuationAuthorityV1() = default;
  DirectDesignatedContinuationAuthorityV1(
      const DirectDesignatedContinuationAuthorityV1&) = delete;
  DirectDesignatedContinuationAuthorityV1& operator=(
      const DirectDesignatedContinuationAuthorityV1&) = delete;

 private:
  friend inline bool initialize_direct_designated_continuation_authority(
      DirectDesignatedContinuationAuthorityV1*, const DirectContinuitySealV2&,
      const DirectSha256Address&, std::uint64_t);
  friend inline bool claim_direct_designated_continuation(
      DirectDesignatedContinuationAuthorityV1*, const Root256&,
      const DirectSha256Address&, const Root256&,
      DirectDesignatedContinuationEdgeV1*);
  DirectSha256Address predecessor_identity_{};
  DirectSha256Address parent_checkpoint_digest_{};
  Root256 birth_identity_{};
  std::uint64_t parent_generation_ = 0u;
  bool available_ = false;
};

// `learned_root` is the caller's device-side digest of the current learned
// matter (direct_brain_root). The seal binds it to the immutable birth
// identity so a restored copy must reproduce both halves to match.
DIRECT_CONTINUITY_HD inline bool direct_continuity_seal(
    const DirectBirthReceiptV1& birth, const Root256& learned_root,
    std::uint64_t exact_history_slots, DirectContinuitySealV2* out) {
  if (out == nullptr) return false;
  detail::DirectSha256State state{};
  const std::uint32_t tag = kDirectContinuitySealAbiV2;
  state.update(&tag, sizeof(tag));
  state.update(&birth.genome_root, sizeof(birth.genome_root));
  state.update(&birth.body_root, sizeof(birth.body_root));
  state.update(&birth.environment_root, sizeof(birth.environment_root));
  state.update(&birth.birth_root, sizeof(birth.birth_root));
  state.update(&learned_root, sizeof(learned_root));
  state.update(&exact_history_slots, sizeof(exact_history_slots));
  out->abi_version = kDirectContinuitySealAbiV2;
  out->birth_identity = birth.birth_root;
  out->learned_state = learned_root;
  out->seal = state.finish();
  return true;
}

inline bool initialize_direct_designated_continuation_authority(
    DirectDesignatedContinuationAuthorityV1* authority,
    const DirectContinuitySealV2& parent,
    const DirectSha256Address& parent_checkpoint_digest,
    std::uint64_t parent_generation) {
  if (authority == nullptr || parent.abi_version != kDirectContinuitySealAbiV2)
    return false;
  authority->predecessor_identity_ = parent.seal;
  authority->parent_checkpoint_digest_ = parent_checkpoint_digest;
  authority->birth_identity_ = parent.birth_identity;
  authority->parent_generation_ = parent_generation;
  authority->available_ = true;
  return true;
}

inline bool claim_direct_designated_continuation(
    DirectDesignatedContinuationAuthorityV1* authority,
    const Root256& restored_birth_identity,
    const DirectSha256Address& restored_parent_checkpoint_digest,
    const Root256& restored_learned_root,
    DirectDesignatedContinuationEdgeV1* out) {
  if (authority == nullptr || out == nullptr || !authority->available_ ||
      restored_birth_identity != authority->birth_identity_ ||
      restored_parent_checkpoint_digest != authority->parent_checkpoint_digest_)
    return false;
  detail::DirectSha256State state{};
  const std::uint32_t tag = 0x44314531u;  // "D1E1"
  const std::uint64_t generation = authority->parent_generation_ + 1u;
  state.update(&tag, sizeof(tag));
  state.update(&restored_birth_identity, sizeof(restored_birth_identity));
  state.update(&authority->predecessor_identity_,
               sizeof(authority->predecessor_identity_));
  state.update(&restored_parent_checkpoint_digest,
               sizeof(restored_parent_checkpoint_digest));
  state.update(&restored_learned_root, sizeof(restored_learned_root));
  state.update(&generation, sizeof(generation));
  out->birth_identity = restored_birth_identity;
  out->predecessor_identity = authority->predecessor_identity_;
  out->parent_checkpoint_digest = restored_parent_checkpoint_digest;
  out->continuation_identity = state.finish();
  out->generation = generation;
  out->designated = 1u;
  out->reserved = 0u;
  authority->available_ = false;
  return true;
}

}  // namespace substrate::direct_network

#undef DIRECT_CONTINUITY_HD

#endif
