#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "bcc32_law.cuh"
#include "bcc32_spatial_macro_executor.cuh"

namespace substrate::bcc32 {
namespace {
#include "bcc32_spatial_macro_cuda_detail.cuh"
[[nodiscard]] __device__ bool junction_owner_slot(
    const DeviceChunkMap& chunks, std::uint64_t center,
    const std::uint32_t permutation[4], std::uint32_t outgoing_sign,
    std::uint32_t incoming_sign, std::uint64_t* destination) {
  const auto authority = kEligibilityResidualJunctionAuthority;
  const std::uint32_t outgoing =
      permutation[authority.outgoing_role] + 4u * outgoing_sign;
  const std::uint32_t incoming =
      permutation[authority.incoming_role] + 4u * incoming_sign;
  const std::uint32_t control_a = permutation[authority.control_a_role];
  const std::uint32_t control_b = permutation[authority.control_b_role];
  std::uint64_t current = center;
  if (!junction_walk(chunks, current, outgoing, 2u, &current) ||
      !junction_walk(chunks, current, incoming, 3u, &current) ||
      !junction_walk(chunks, current, control_a, 1u, &current) ||
      !junction_walk(chunks, current, control_b, 1u, &current))
    return false;
  *destination = current;
  return true;
}

[[nodiscard]] __device__ bool junction_decode_code(
    std::uint32_t code, std::uint32_t permutation[4],
    std::uint32_t* outgoing_sign, std::uint32_t* incoming_sign) {
  static_assert(
      kEligibilityResidualJunctionAuthority.outgoing_sign_count == 2u &&
          kEligibilityResidualJunctionAuthority.incoming_sign_count == 2u,
      "junction match-code encoding assumes both incoming/outgoing signs");
  constexpr std::uint32_t kOutgoingSignCount =
      kEligibilityResidualJunctionAuthority.outgoing_sign_count;
  constexpr std::uint32_t kIncomingSignCount =
      kEligibilityResidualJunctionAuthority.incoming_sign_count;
  constexpr std::uint32_t kSignPairCount =
      kOutgoingSignCount * kIncomingSignCount;
  constexpr std::uint32_t kActionCount = 24u * kSignPairCount;
  if (code == 0u || code > 2u * kActionCount) return false;
  if (code > kActionCount) code -= kActionCount;
  const std::uint32_t wanted = (code - 1u) / kSignPairCount;
  const std::uint32_t signs = (code - 1u) % kSignPairCount;
  *outgoing_sign = signs / kIncomingSignCount;
  *incoming_sign = signs % kIncomingSignCount;
  std::uint32_t rank = 0u;
  for (std::uint32_t outgoing = 0u; outgoing < 4u; ++outgoing) {
    for (std::uint32_t control_a = 0u; control_a < 4u; ++control_a) {
      if (control_a == outgoing) continue;
      for (std::uint32_t incoming = 0u; incoming < 4u; ++incoming) {
        if (incoming == outgoing || incoming == control_a) continue;
        if (rank++ != wanted) continue;
        processive_permutation(outgoing, control_a, incoming, permutation);
        return true;
      }
    }
  }
  return false;
}

struct JunctionDescriptorState {
  bool matches = false;
  bool latched = false;
  bool released = false;
  std::uint64_t token_slot = 0u;
  std::uint32_t token_basis = 0u;
  std::uint32_t token_out_basis = 0u;
};

[[nodiscard]] __device__ std::uint8_t junction_lane_lock_phase(
    SiteWord word, std::uint32_t incoming_basis,
    std::uint32_t outgoing_basis) {
  if (word ==
      (kQ | channel_bit(kReactiveShift, incoming_basis))) return 1u;
  if (word ==
      (kQ | channel_bit(kReactiveShift, outgoing_basis))) return 2u;
  return 0u;
}

[[nodiscard]] __device__ JunctionDescriptorState
junction_descriptor_state_with_center_word(
    const SiteWord* words, std::uint64_t site_count,
    const DeviceChunkMap& chunks, std::uint64_t center,
    SiteWord center_word,
    const std::uint32_t permutation[4], std::uint32_t outgoing_sign,
    std::uint32_t incoming_sign) {
  JunctionDescriptorState result{};
  const auto authority = kEligibilityResidualJunctionAuthority;
  const std::uint32_t incoming =
      permutation[authority.incoming_role] + 4u * incoming_sign;
  const std::uint32_t outgoing =
      permutation[authority.outgoing_role] + 4u * outgoing_sign;
  const SiteWord target_mask = carrier_bit(incoming) | carrier_bit(outgoing);
  if ((center_word & ~target_mask) != (kQ & ~target_mask) ||
      (((center_word & carrier_bit(incoming)) != 0u) ==
       ((center_word & carrier_bit(outgoing)) != 0u)))
    return result;
  std::uint64_t control_a = 0u;
  std::uint64_t control_b = 0u;
  if (!(junction_control_slot(chunks, center, permutation, outgoing_sign,
                              incoming_sign, 0u, &control_a) &&
        junction_control_slot(chunks, center, permutation, outgoing_sign,
                              incoming_sign, 1u, &control_b) &&
        control_a < site_count && control_b < site_count))
    return result;
  const std::uint32_t control_a_basis =
      permutation[authority.control_a_role];
  const std::uint32_t control_b_basis =
      permutation[authority.control_b_role];
  const SiteWord control_a_word = words[control_a];
  const SiteWord control_b_word = words[control_b];
  const bool vacancy_a =
      control_a_word == (kQ ^ carrier_bit(control_a_basis));
  const bool vacancy_b =
      control_b_word == (kQ ^ carrier_bit(control_b_basis));
  const bool control_a_present =
      (control_a_word & energy_bit(control_a_basis)) != 0u || vacancy_a;
  const bool control_b_present =
      (control_b_word & energy_bit(control_b_basis)) != 0u || vacancy_b;
  if (!(control_a_present && control_b_present)) return result;
  std::uint64_t owner = 0u;
  if (junction_owner_slot(chunks, center, permutation, outgoing_sign,
                          incoming_sign, &owner) && owner < site_count) {
    const SiteWord owner_word = words[owner];
    const std::uint8_t phase = junction_lane_lock_phase(
        owner_word, incoming & 3u, outgoing & 3u);
    if (phase != 0u) {
    result.matches = true;
    result.latched = true;
    result.released = phase == 2u;
    result.token_slot = owner;
    result.token_basis = incoming & 3u;
    result.token_out_basis = outgoing & 3u;
    return result;
    }
  }
  // Production CUDA follows the same authority boundary as the reference
  // law: centre/control coincidence is insufficient without resident owner
  // phase matter.
  result.matches = false;
  return result;
}

[[nodiscard]] __device__ JunctionDescriptorState junction_descriptor_state(
    const SiteWord* words, std::uint64_t site_count,
    const DeviceChunkMap& chunks, std::uint64_t center,
    const std::uint32_t permutation[4], std::uint32_t outgoing_sign,
    std::uint32_t incoming_sign) {
  return junction_descriptor_state_with_center_word(
      words, site_count, chunks, center, words[center], permutation,
      outgoing_sign, incoming_sign);
}

[[nodiscard]] __device__ bool junction_descriptor_matches(
    const SiteWord* words, std::uint64_t site_count,
    const DeviceChunkMap& chunks, std::uint64_t center,
    const std::uint32_t permutation[4], std::uint32_t outgoing_sign,
    std::uint32_t incoming_sign) {
  return junction_descriptor_state(words, site_count, chunks, center,
                                   permutation, outgoing_sign,
                                   incoming_sign).matches;
}

[[nodiscard]] __device__ bool junction_equivalent_candidate(
    const DeviceChunkMap& chunks, std::uint64_t center,
    std::uint8_t prior_code, const std::uint32_t candidate[4],
    std::uint32_t candidate_outgoing_sign,
    std::uint32_t candidate_incoming_sign) {
  std::uint32_t prior[4]{};
  std::uint32_t prior_outgoing_sign = 0u;
  std::uint32_t prior_incoming_sign = 0u;
  if (!junction_decode_code(prior_code, prior, &prior_outgoing_sign,
                            &prior_incoming_sign))
    return false;
  const auto authority = kEligibilityResidualJunctionAuthority;
  const std::uint32_t prior_incoming =
      prior[authority.incoming_role] + 4u * prior_incoming_sign;
  const std::uint32_t prior_outgoing =
      prior[authority.outgoing_role] + 4u * prior_outgoing_sign;
  const std::uint32_t candidate_incoming =
      candidate[authority.incoming_role] + 4u * candidate_incoming_sign;
  const std::uint32_t candidate_outgoing =
      candidate[authority.outgoing_role] + 4u * candidate_outgoing_sign;
  if (prior_incoming != candidate_incoming ||
      prior_outgoing != candidate_outgoing)
    return false;
  std::uint64_t prior_controls[2]{};
  std::uint64_t candidate_controls[2]{};
  for (std::uint32_t index = 0u; index < 2u; ++index) {
    if (!junction_control_slot(chunks, center, prior, prior_outgoing_sign,
                               prior_incoming_sign, index,
                               &prior_controls[index]) ||
        !junction_control_slot(chunks, center, candidate,
                               candidate_outgoing_sign,
                               candidate_incoming_sign, index,
                               &candidate_controls[index]))
      return false;
  }
  return (prior_controls[0u] == candidate_controls[0u] &&
          prior_controls[1u] == candidate_controls[1u]) ||
         (prior_controls[0u] == candidate_controls[1u] &&
          prior_controls[1u] == candidate_controls[0u]);
}

[[nodiscard]] __device__ std::uint8_t junction_resolve_code(
    const SiteWord* words, std::uint64_t site_count,
    const DeviceChunkMap& chunks, std::uint64_t center,
    SiteWord center_word) {
  constexpr std::uint32_t kActionCount =
      24u * kEligibilityResidualJunctionAuthority.outgoing_sign_count *
      kEligibilityResidualJunctionAuthority.incoming_sign_count;
  std::uint8_t staged = 0u;
  std::uint8_t released = 0u;
  bool staged_latched = false;
  bool staged_ambiguous = false;
  bool released_ambiguous = false;
  bool mixed_phase_ambiguous = false;
  std::uint32_t rank = 0u;
  for (std::uint32_t outgoing = 0u; outgoing < 4u; ++outgoing) {
    for (std::uint32_t control_a = 0u; control_a < 4u; ++control_a) {
      if (control_a == outgoing) continue;
      for (std::uint32_t incoming = 0u; incoming < 4u; ++incoming) {
        if (incoming == outgoing || incoming == control_a) continue;
        std::uint32_t permutation[4]{};
        processive_permutation(outgoing, control_a, incoming, permutation);
        for (std::uint32_t sign = 0u;
             sign < kEligibilityResidualJunctionAuthority.outgoing_sign_count;
             ++sign) {
          for (std::uint32_t incoming_sign = 0u;
               incoming_sign <
                   kEligibilityResidualJunctionAuthority.incoming_sign_count;
               ++incoming_sign) {
            constexpr std::uint32_t kSignPairCount =
                kEligibilityResidualJunctionAuthority.outgoing_sign_count *
                kEligibilityResidualJunctionAuthority.incoming_sign_count;
            const std::uint8_t code = static_cast<std::uint8_t>(
                kSignPairCount * rank +
                sign *
                    kEligibilityResidualJunctionAuthority.incoming_sign_count +
                incoming_sign + 1u);
            const JunctionDescriptorState state =
                junction_descriptor_state_with_center_word(
                    words, site_count, chunks, center, center_word,
                    permutation, sign, incoming_sign);
            if (!state.matches)
              continue;
            if (state.released) {
              if (staged != 0u &&
                  !junction_equivalent_candidate(
                      chunks, center, staged, permutation, sign,
                      incoming_sign))
                mixed_phase_ambiguous = true;
              if (released == 0u) {
                released = code;
              } else if (!junction_equivalent_candidate(
                             chunks, center, released, permutation, sign,
                             incoming_sign)) {
                released_ambiguous = true;
              }
            } else if (released != 0u &&
                       !junction_equivalent_candidate(
                           chunks, center, released, permutation, sign,
                           incoming_sign)) {
              mixed_phase_ambiguous = true;
              if (staged == 0u) {
                staged = code;
                staged_latched = state.latched;
              }
            } else if (staged == 0u) {
              staged = code;
              staged_latched = state.latched;
            } else if (!junction_equivalent_candidate(
                           chunks, center, staged, permutation, sign,
                           incoming_sign)) {
              staged_ambiguous = true;
            } else if (state.latched && !staged_latched) {
              staged = code;
              staged_latched = true;
            }
          }
        }
        ++rank;
      }
    }
  }
  if (mixed_phase_ambiguous) return 0u;
  if (released != 0u)
    return released_ambiguous
        ? 0u
        : static_cast<std::uint8_t>(released + kActionCount);
  return staged_ambiguous ? 0u : staged;
}

[[nodiscard]] __device__ std::uint8_t junction_match_code(
    const SiteWord* words, std::uint64_t site_count,
    const DeviceChunkMap& chunks, std::uint64_t center) {
  const SiteWord center_word = words[center];
  const std::uint8_t code = junction_resolve_code(
      words, site_count, chunks, center, center_word);
  if (code == 0u) return 0u;
  std::uint32_t permutation[4]{};
  std::uint32_t outgoing_sign = 0u;
  std::uint32_t incoming_sign = 0u;
  if (!junction_decode_code(code, permutation, &outgoing_sign,
                            &incoming_sign))
    return 0u;
  const JunctionDescriptorState state = junction_descriptor_state(
      words, site_count, chunks, center, permutation, outgoing_sign,
      incoming_sign);
  if (state.latched) return code;
  const auto authority = kEligibilityResidualJunctionAuthority;
  const std::uint32_t incoming =
      permutation[authority.incoming_role] + 4u * incoming_sign;
  const std::uint32_t outgoing =
      permutation[authority.outgoing_role] + 4u * outgoing_sign;
  const SiteWord post_word =
      center_word ^ carrier_bit(incoming) ^ carrier_bit(outgoing);
  const std::uint8_t post_code = junction_resolve_code(
      words, site_count, chunks, center, post_word);
  std::uint32_t post_permutation[4]{};
  std::uint32_t post_outgoing_sign = 0u;
  std::uint32_t post_incoming_sign = 0u;
  if (!junction_decode_code(post_code, post_permutation,
                            &post_outgoing_sign, &post_incoming_sign))
    return 0u;
  return junction_equivalent_candidate(
             chunks, center, code, post_permutation, post_outgoing_sign,
             post_incoming_sign)
      ? code
      : 0u;
}

[[nodiscard]] __device__ bool junction_footprint_slot(
    const DeviceChunkMap& chunks, std::uint64_t center,
    const std::uint32_t permutation[4], std::uint32_t outgoing_sign,
    std::uint32_t incoming_sign, std::uint32_t index,
    std::uint64_t* destination) {
  if (index == 0u) {
    *destination = center;
    return true;
  }
  if (index == 3u)
    return junction_owner_slot(chunks, center, permutation, outgoing_sign,
                               incoming_sign, destination);
  return junction_control_slot(chunks, center, permutation, outgoing_sign,
                               incoming_sign, index - 1u, destination);
}

[[nodiscard]] __device__ bool junction_inverse_footprint_slot(
    const DeviceChunkMap& chunks, std::uint64_t shared,
    const std::uint32_t permutation[4], std::uint32_t outgoing_sign,
    std::uint32_t incoming_sign, std::uint32_t index,
    std::uint64_t* center) {
  if (index == 0u) {
    *center = shared;
    return true;
  }
  const auto authority = kEligibilityResidualJunctionAuthority;
  if (index == 3u) {
    const std::uint32_t outgoing =
        permutation[authority.outgoing_role] + 4u * outgoing_sign;
    const std::uint32_t incoming =
        permutation[authority.incoming_role] + 4u * incoming_sign;
    std::uint64_t current = shared;
    if (!junction_walk(chunks, current,
                       permutation[authority.control_b_role] ^ 4u, 1u,
                       &current) ||
        !junction_walk(chunks, current,
                       permutation[authority.control_a_role] ^ 4u, 1u,
                       &current) ||
        !junction_walk(chunks, current, incoming ^ 4u, 3u, &current) ||
        !junction_walk(chunks, current, outgoing ^ 4u, 2u, &current))
      return false;
    *center = current;
    return true;
  }
  const std::uint32_t crossed =
      permutation[index == 1u ? authority.control_b_role
                              : authority.control_a_role];
  const std::uint32_t incoming = permutation[authority.incoming_role];
  const std::uint32_t outgoing =
      permutation[authority.outgoing_role] + 4u * outgoing_sign;
  std::uint64_t current = shared;
  if ((incoming_sign == 0u &&
       !junction_walk(
           chunks, current, incoming ^ 4u,
           kEligibilityResidualJunctionAuthority.positive_incoming_control_offset,
           &current)) ||
      !junction_walk(chunks, current, crossed ^ 4u, 1u, &current) ||
      !junction_walk(chunks, current, outgoing ^ 4u,
                     static_cast<std::uint32_t>(authority.outgoing_distance),
                     &current))
    return false;
  *center = current;
  return true;
}

[[nodiscard]] __device__ bool junction_collides(
    const SiteWord* words, std::uint64_t site_count,
    const DeviceChunkMap& chunks, std::uint64_t center, std::uint8_t code) {
  std::uint32_t own_permutation[4]{};
  std::uint32_t own_sign = 0u;
  std::uint32_t own_incoming_sign = 0u;
  if (!junction_decode_code(code, own_permutation, &own_sign,
                            &own_incoming_sign))
    return true;
  const JunctionDescriptorState own_state = junction_descriptor_state(
      words, site_count, chunks, center, own_permutation, own_sign,
      own_incoming_sign);
  const std::uint32_t own_count = own_state.latched ? 4u : 3u;
  for (std::uint32_t own_index = 0u; own_index < own_count; ++own_index) {
    std::uint64_t shared = 0u;
    if (!junction_footprint_slot(chunks, center, own_permutation, own_sign,
                                 own_incoming_sign, own_index, &shared))
      continue;
    for (std::uint32_t other_code = 1u;
         other_code <=
         24u * kEligibilityResidualJunctionAuthority.outgoing_sign_count *
             kEligibilityResidualJunctionAuthority.incoming_sign_count;
         ++other_code) {
      std::uint32_t other_permutation[4]{};
      std::uint32_t other_sign = 0u;
      std::uint32_t other_incoming_sign = 0u;
      if (!junction_decode_code(other_code, other_permutation, &other_sign,
                                &other_incoming_sign))
        continue;
      for (std::uint32_t other_index = 0u; other_index < 4u; ++other_index) {
        std::uint64_t other_center = 0u;
        if (!junction_inverse_footprint_slot(
                chunks, shared, other_permutation, other_sign,
                other_incoming_sign, other_index, &other_center) ||
            other_center == center || other_center >= site_count)
          continue;
        // Test the inverse-derived descriptor itself.  Resolving the other
        // center through junction_match_code would hide all of its real
        // owners when that center has two inequivalent matches, allowing a
        // third overlapping owner to commit despite CPU collision semantics.
        const JunctionDescriptorState other_state = junction_descriptor_state(
            words, site_count, chunks, other_center, other_permutation,
            other_sign, other_incoming_sign);
        if (other_state.matches &&
            other_index < (other_state.latched ? 4u : 3u))
          return true;
      }
    }
  }
  return false;
}

__global__ void junction_match_kernel(
    const SiteWord* words, std::uint8_t* matches, std::uint64_t site_count,
    DeviceChunkMap chunks, const std::uint64_t* active_slots,
    std::uint64_t active_count,
    const std::uint32_t* device_active_count) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (device_active_count != nullptr)
    active_count =
        min(active_count, static_cast<std::uint64_t>(*device_active_count));
  if (index >= active_count) return;
  const std::uint64_t center = active_slots == nullptr ? index : active_slots[index];
  matches[index] = center < site_count
                       ? junction_match_code(words, site_count, chunks, center)
                       : 0u;
}

__global__ void junction_collision_kernel(
    const SiteWord* words, std::uint8_t* matches, std::uint64_t site_count,
    DeviceChunkMap chunks, const std::uint64_t* active_slots,
    std::uint64_t active_count,
    const std::uint32_t* device_active_count) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (device_active_count != nullptr)
    active_count =
        min(active_count, static_cast<std::uint64_t>(*device_active_count));
  if (index >= active_count || matches[index] == 0u) return;
  const std::uint64_t center = active_slots == nullptr ? index : active_slots[index];
  if (junction_collides(words, site_count, chunks, center, matches[index]))
    matches[index] = 0u;
}

__global__ void junction_apply_kernel(
    SiteWord* words, const std::uint8_t* matches,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    std::uint64_t site_count, DeviceChunkMap chunks,
    const std::uint32_t* device_active_count) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (device_active_count != nullptr)
    active_count =
        min(active_count, static_cast<std::uint64_t>(*device_active_count));
  if (index >= active_count || matches[index] == 0u) return;
  std::uint32_t permutation[4]{};
  std::uint32_t sign = 0u;
  std::uint32_t incoming_sign = 0u;
  if (!junction_decode_code(matches[index], permutation, &sign,
                            &incoming_sign))
    return;
  const auto authority = kEligibilityResidualJunctionAuthority;
  const std::uint32_t incoming =
      permutation[authority.incoming_role] + 4u * incoming_sign;
  const std::uint32_t outgoing =
      permutation[authority.outgoing_role] + 4u * sign;
  const std::uint64_t center = active_slots == nullptr ? index : active_slots[index];
  const JunctionDescriptorState state = junction_descriptor_state(
      words, site_count, chunks, center, permutation, sign, incoming_sign);
  constexpr std::uint32_t kActionCount =
      24u * kEligibilityResidualJunctionAuthority.outgoing_sign_count *
      kEligibilityResidualJunctionAuthority.incoming_sign_count;
  const bool encoded_released = matches[index] > kActionCount;
  if (!state.matches ||
      (state.latched ? encoded_released != state.released
                     : encoded_released))
    return;
  SiteWord word = words[center];
  controlled_transpose(word, carrier_bit(incoming), word,
                       carrier_bit(outgoing), true);
  words[center] = word;
  if (state.latched) {
    words[state.token_slot] ^=
        channel_bit(kReactiveShift, state.token_basis) |
        channel_bit(kReactiveShift, state.token_out_basis);
  }
}



}  // namespace

namespace {

void launch_eligibility_residual_junction_impl(
    SiteWord* words, std::uint8_t* scratch, std::uint64_t site_count,
    const DeviceChunkMap& chunks, const std::uint64_t* active_slots,
    std::uint64_t active_count, const std::uint32_t* device_active_count,
    cudaStream_t stream) {
  const std::uint64_t count = active_slots == nullptr ? site_count : active_count;
  const std::uint32_t blocks = launch_blocks(count);
  junction_match_kernel<<<blocks, kThreads, 0, stream>>>(
      words, scratch, site_count, chunks, active_slots, count,
      device_active_count);
  check_cuda(cudaGetLastError(), "launch BCC32 eligibility junction match");
  junction_collision_kernel<<<blocks, kThreads, 0, stream>>>(
      words, scratch, site_count, chunks, active_slots, count,
      device_active_count);
  check_cuda(cudaGetLastError(), "launch BCC32 eligibility junction collision");
  junction_apply_kernel<<<blocks, kThreads, 0, stream>>>(
      words, scratch, active_slots, count, site_count, chunks,
      device_active_count);
  check_cuda(cudaGetLastError(), "launch BCC32 eligibility junction apply");
}

}  // namespace

void launch_eligibility_residual_junction_cuda(
    SiteWord* words, std::uint8_t* scratch, std::uint64_t site_count,
    const DeviceChunkMap& chunks, const std::uint64_t* active_slots,
    std::uint64_t active_count, cudaStream_t stream) {
  launch_eligibility_residual_junction_impl(
      words, scratch, site_count, chunks, active_slots, active_count, nullptr,
      stream);
}

void launch_eligibility_residual_junction_graph_safe_cuda(
    SiteWord* words, std::uint8_t* scratch, std::uint64_t site_count,
    const DeviceChunkMap& chunks, const std::uint64_t* active_slots,
    const std::uint32_t* device_active_count, std::uint32_t capacity,
    cudaStream_t stream) {
  launch_eligibility_residual_junction_impl(
      words, scratch, site_count, chunks, active_slots, capacity,
      device_active_count, stream);
}

}  // namespace substrate::bcc32
