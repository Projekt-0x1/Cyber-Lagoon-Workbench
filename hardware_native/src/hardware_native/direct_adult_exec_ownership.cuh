#ifndef HARDWARE_NATIVE_DIRECT_ADULT_EXEC_OWNERSHIP_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_EXEC_OWNERSHIP_CUH

// gh #1563 -- g.resident_execution_ownership.
//
// One resident adult owns runtime execution and output authority on device.
// The host, the compiler, and any execution backing may be replaced under
// logical equivalence, but none of them may ever step or schedule semantic
// occurrences. These primitives are the ownership record, the resident step
// gate, the output-authority stamp, and the backing-equivalence discriminator
// for that law. They own no semantics themselves; a resident executor binds
// them and every semantic step must pass through the gate.

#if defined(__CUDACC__)
#define DIRECT_EXEC_HD __host__ __device__
#else
#define DIRECT_EXEC_HD
#endif

#include <cstdint>
#include <type_traits>

namespace substrate::direct_adult {

// Origin seal carried by resident work. A step or schedule request that does
// not present both the live owner token and this origin is transport at best
// and a takeover attempt at worst; either way it is refused.
inline constexpr std::uint32_t kDirectResidentExecOrigin = 0x0E51D07u;

struct DirectExecutionOwnership {
  std::uint64_t owner_token;  // 0 = unowned; exactly one live claimant
  unsigned long long resident_steps;
  unsigned long long refused_second_claimants;
  unsigned long long refused_external_steps;
  unsigned long long stamped_outputs;
  unsigned long long refused_output_tags;
  std::uint32_t owner_incarnation;
  std::uint32_t backing_generation;
};
static_assert(sizeof(DirectExecutionOwnership) == 56 &&
              std::is_standard_layout_v<DirectExecutionOwnership> &&
              std::is_trivial_v<DirectExecutionOwnership>);

struct DirectExecutionOutputRecord {
  std::uint64_t ticket_id;
  std::uint64_t owner_token;
  std::uint64_t resident_step;
  std::uint32_t payload_word;
  std::uint32_t provenance;  // checksum binding identity, step and payload
};
static_assert(sizeof(DirectExecutionOutputRecord) == 32 &&
              std::is_standard_layout_v<DirectExecutionOutputRecord> &&
              std::is_trivial_v<DirectExecutionOutputRecord>);

// An execution backing describes how semantic work is mechanically arranged.
// lane_stride and generation are replacement knobs a new backing may choose
// freely. semantics_word is the executed semantic schedule content: two
// backings are logically equivalent only when it survives the swap intact.
struct DirectExecutionBackingPlan {
  std::uint32_t semantics_word;
  std::uint32_t lane_stride;
  std::uint32_t generation;
};
static_assert(sizeof(DirectExecutionBackingPlan) == 12);

DIRECT_EXEC_HD inline bool direct_execution_backings_equivalent(
    const DirectExecutionBackingPlan& a, const DirectExecutionBackingPlan& b) {
  return a.semantics_word == b.semantics_word;
}

DIRECT_EXEC_HD inline std::uint32_t direct_exec_digest_fold(std::uint32_t digest,
                                                            std::uint32_t word) {
  return digest ^ (word + 0x9e3779b9u + (digest << 6u) + (digest >> 2u));
}

DIRECT_EXEC_HD inline bool direct_execution_gate_open(
    const DirectExecutionOwnership& ownership, std::uint64_t presented_token,
    std::uint32_t origin) {
  return ownership.owner_token != 0u && presented_token == ownership.owner_token &&
         origin == kDirectResidentExecOrigin;
}

#ifdef __CUDACC__

__device__ inline bool direct_execution_acquire(DirectExecutionOwnership& ownership,
                                                std::uint64_t token,
                                                std::uint32_t backing_generation) {
  if (token == 0u ||
      atomicCAS(reinterpret_cast<unsigned long long*>(&ownership.owner_token), 0ull,
                token) != 0ull) {
    atomicAdd(&ownership.refused_second_claimants, 1ull);
    return false;
  }
  ownership.backing_generation = backing_generation;
  atomicAdd(&ownership.owner_incarnation, 1u);
  return true;
}

__device__ inline bool direct_execution_release(DirectExecutionOwnership& ownership,
                                                std::uint64_t token) {
  if (token == 0u || token != ownership.owner_token ||
      atomicCAS(reinterpret_cast<unsigned long long*>(&ownership.owner_token),
                static_cast<unsigned long long>(token), 0ull) !=
          static_cast<unsigned long long>(token)) {
    atomicAdd(&ownership.refused_second_claimants, 1ull);
    return false;
  }
  return true;
}

// The single doorway to semantic progression. Refused requests are counted
// and leave all semantic state untouched; the caller's duty is to have made
// the mutation conditional on this result.
__device__ inline bool direct_execution_request_step(DirectExecutionOwnership& ownership,
                                                     std::uint64_t presented_token,
                                                     std::uint32_t origin) {
  if (!direct_execution_gate_open(ownership, presented_token, origin)) {
    atomicAdd(&ownership.refused_external_steps, 1ull);
    return false;
  }
  atomicAdd(&ownership.resident_steps, 1ull);
  return true;
}

DIRECT_EXEC_HD inline std::uint32_t direct_execution_output_checksum(
    const DirectExecutionOutputRecord& record, std::uint32_t incarnation,
    std::uint32_t backing_generation) {
  std::uint32_t digest = 0x1E51D07u;
  digest = direct_exec_digest_fold(digest, static_cast<std::uint32_t>(record.ticket_id));
  digest = direct_exec_digest_fold(digest, static_cast<std::uint32_t>(record.ticket_id >> 32u));
  digest = direct_exec_digest_fold(digest, static_cast<std::uint32_t>(record.owner_token));
  digest = direct_exec_digest_fold(digest, static_cast<std::uint32_t>(record.owner_token >> 32u));
  digest = direct_exec_digest_fold(digest, static_cast<std::uint32_t>(record.resident_step));
  digest = direct_exec_digest_fold(digest, record.payload_word);
  digest = direct_exec_digest_fold(digest, incarnation);
  digest = direct_exec_digest_fold(digest, backing_generation);
  return digest;
}

DIRECT_EXEC_HD inline bool direct_execution_output_authoritative(
    const DirectExecutionOutputRecord& record, const DirectExecutionOwnership& ownership) {
  if (record.owner_token == 0u || record.owner_token != ownership.owner_token)
    return false;
  return record.provenance ==
         direct_execution_output_checksum(record, ownership.owner_incarnation,
                                          ownership.backing_generation);
}

__device__ inline void direct_execution_stamp_output(DirectExecutionOwnership& ownership,
                                                     DirectExecutionOutputRecord* out,
                                                     std::uint64_t ticket_id,
                                                     std::uint32_t payload_word) {
  out->ticket_id = ticket_id;
  out->owner_token = ownership.owner_token;
  out->resident_step = ownership.resident_steps;
  out->payload_word = payload_word;
  out->provenance = direct_execution_output_checksum(*out, ownership.owner_incarnation,
                                                     ownership.backing_generation);
  atomicAdd(&ownership.stamped_outputs, 1ull);
}

__device__ inline bool direct_execution_verify_output(DirectExecutionOwnership& ownership,
                                                      const DirectExecutionOutputRecord& record) {
  if (direct_execution_output_authoritative(record, ownership)) return true;
  atomicAdd(&ownership.refused_output_tags, 1ull);
  return false;
}

#endif  // __CUDACC__

}  // namespace substrate::direct_adult

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_EXEC_OWNERSHIP_CUH
