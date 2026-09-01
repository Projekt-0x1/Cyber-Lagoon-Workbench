#ifndef HARDWARE_NATIVE_DIRECT_ADULT_LANGUAGE_ONE_SHOT_ABI_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_LANGUAGE_ONE_SHOT_ABI_CUH

#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_adult_multimodal_grounding_abi.cuh"

namespace substrate::direct_network {

struct DirectExactHistoryRecord;
struct DirectNode;

inline constexpr std::uint32_t kLanguageOneShotCapacity = 24u;
inline constexpr std::uint32_t kLanguageOneShotTicketCapacity = 96u;
inline constexpr std::uint32_t kLanguageOneShotMaximumGap = 24u;

struct DirectLanguageOneShotSite {
  std::uint32_t surface_channel;
  std::uint32_t surface_value;
  std::uint32_t ground_channel;
  std::uint32_t ground_value;
  std::uint32_t motor_node;
  std::uint32_t motor_channel;
  std::uint32_t motor_value;
  std::uint32_t outcome_value;
  std::uint32_t support;
  std::uint32_t contradictions;
  std::uint32_t active;
  std::uint64_t matter_identity;
};

struct DirectLanguageOneShotState {
  DirectMultimodalGroundingTable grounding;
  DirectLanguageOneShotSite sites[kLanguageOneShotCapacity];
  std::uint64_t admitted_motor_identities[kLanguageOneShotTicketCapacity];
  std::uint32_t site_count;
  std::uint32_t admitted_motor_count;
  std::uint32_t cursor;
  std::uint32_t q_contacts;
  std::uint32_t verified_closures;
  std::uint32_t refused_closures;
  std::uint32_t provisional_births;
  std::uint32_t revisions;
  std::uint32_t retractions;
  std::uint32_t matter;
  std::uint32_t work;
  std::uint32_t lesion_events;
  std::uint32_t sham_matter;
  std::uint32_t reacquired_sites;
  std::uint64_t source_hash;
  std::uint64_t revision_identity;
};

struct DirectLanguageOneShotPlan {
  std::uint32_t admitted;
  std::uint32_t motor_node;
  std::uint32_t motor_channel;
  std::uint32_t motor_value;
  std::uint32_t support;
  std::uint32_t provisional;
  std::uint32_t q;
  std::uint32_t N;
  std::uint32_t matter;
  std::uint32_t work;
  std::uint64_t p_next;
};

static_assert(std::is_trivially_copyable_v<DirectLanguageOneShotSite>);
static_assert(std::is_trivially_copyable_v<DirectLanguageOneShotState>);
static_assert(std::is_trivially_copyable_v<DirectLanguageOneShotPlan>);
static_assert(sizeof(DirectLanguageOneShotSite) == 56u);
static_assert(sizeof(DirectLanguageOneShotState) == 12704u);
static_assert(sizeof(DirectLanguageOneShotPlan) == 48u);
static_assert(alignof(DirectLanguageOneShotState) == 8u);

#if defined(__CUDACC__)
__device__ bool language_one_shot_seen(
    const DirectLanguageOneShotState& state, std::uint64_t identity);
__device__ bool language_one_shot_verified_return(
    const DirectExactHistoryRecord* records, std::uint32_t count,
    const DirectExactHistoryRecord& motor,
    DirectExactHistoryRecord* verified_return);
__device__ bool language_one_shot_preceding_coalition(
    const DirectExactHistoryRecord* records, std::uint32_t motor_index,
    DirectExactHistoryRecord* surface, DirectExactHistoryRecord* ground);
__device__ bool language_one_shot_same_cue(
    const DirectLanguageOneShotSite& site,
    const DirectExactHistoryRecord& surface,
    const DirectExactHistoryRecord& ground);
__device__ bool language_one_shot_same_motor(
    const DirectLanguageOneShotSite& site,
    const DirectExactHistoryRecord& motor);
__device__ void language_one_shot_admit(
    DirectLanguageOneShotState* state,
    const DirectExactHistoryRecord& surface,
    const DirectExactHistoryRecord& ground,
    const DirectExactHistoryRecord& motor,
    const DirectExactHistoryRecord& consequence);
__device__ void language_one_shot_assimilate(
    DirectLanguageOneShotState* state,
    const DirectExactHistoryRecord* records, std::uint32_t count);
__device__ DirectLanguageOneShotPlan language_one_shot_plan(
    DirectLanguageOneShotState* state,
    const DirectExactHistoryRecord* records, std::uint32_t begin,
    std::uint32_t count);
__device__ bool language_one_shot_drive(
    const DirectLanguageOneShotPlan& plan, DirectNode* nodes,
    std::uint32_t node_count);
__device__ std::uint32_t language_one_shot_focal_lesion(
    DirectLanguageOneShotState* state);
__device__ std::uint32_t language_one_shot_remote_sham(
    DirectLanguageOneShotState* state, std::uint32_t matter);
#endif

#if defined(__CUDACC__)
#define DIRECT_LANGUAGE_ONE_SHOT_HD __host__ __device__
#else
#define DIRECT_LANGUAGE_ONE_SHOT_HD
#endif
DIRECT_LANGUAGE_ONE_SHOT_HD inline std::uint64_t language_one_shot_fold(
    std::uint64_t hash, std::uint64_t value) {
  return (hash ^ (value + 0x9e3779b97f4a7c15ULL + (hash << 6u) +
                  (hash >> 2u))) *
         1099511628211ULL;
}
#undef DIRECT_LANGUAGE_ONE_SHOT_HD

}  // namespace substrate::direct_network

#endif
