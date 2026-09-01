#ifndef HARDWARE_NATIVE_DIRECT_ADULT_WANTING_LIKING_DISSOCIATION_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_WANTING_LIKING_DISSOCIATION_CUH

#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_adult_affect_body.cuh"

#if defined(__CUDACC__)
#define WANTING_LIKING_HD __host__ __device__
#else
#define WANTING_LIKING_HD
#endif

namespace substrate::direct_network {

inline constexpr std::uint32_t kWantingLikingActionCapacity =
    direct_adult_core::kMaxAsynchronousTickets;
inline constexpr std::uint32_t kWantingLikingRecipeScanCapacity = 256u;

struct alignas(8) ResidentWantingLikingProfileV1 {
  std::int64_t wanting_q16;
  std::int64_t liking_q16;
  std::int64_t positive_recipe_credit_q16;
  std::int32_t vitality_q16;
  std::uint32_t open_pursuits;
  std::uint32_t causal_participants;
  std::uint32_t vitality_channels;
  std::uint32_t credited_recipes;
  std::uint32_t invalid_sources;
  std::uint32_t reserved;
  std::uint32_t reserved2;
};
static_assert(std::is_trivial_v<ResidentWantingLikingProfileV1> &&
              std::is_standard_layout_v<ResidentWantingLikingProfileV1> &&
              std::has_unique_object_representations_v<
                  ResidentWantingLikingProfileV1>);

struct ResidentWantingLikingMotorGateV1 {
  std::int32_t activation_threshold_q16;
  std::uint32_t live_pursuit;
  std::uint32_t settled_liking;
};

// Current authorized causal participation is WANTING and sustains pursuit at
// a lower activation. Positive settled credit is LIKING and raises the
// threshold, so consummation cannot author a new pursuit. This reads only the
// touched resident frontier; it cannot write evidence, settlement, or credit.
WANTING_LIKING_HD inline ResidentWantingLikingMotorGateV1
resident_wanting_liking_motor_gate(
    const direct_adult_core::DirectParticipationDescriptor* contributions,
    std::uint32_t contribution_count, std::uint32_t motor_node,
    const DirectAffectBodyState* affect) {
  using namespace direct_adult_core;
  ResidentWantingLikingMotorGateV1 gate{
      kQ16One / 4, 0u,
      affect != nullptr && affect->vitality_aggregate_q16 > 0 ? 1u : 0u};
  if (contributions != nullptr) {
    for (std::uint32_t i = 0u; i < contribution_count; ++i) {
      const auto& contribution = contributions[i];
      if (contribution.target_node == motor_node &&
          contribution.ticket_id != 0u &&
          contribution.authority != DirectParticipationAuthority::none &&
          contribution.contribution_kind != DirectContributionKind::none &&
          contribution.contribution_kind !=
              DirectContributionKind::ancestry_incomplete) {
        gate.live_pursuit = 1u;
        break;
      }
    }
  }
  if (gate.live_pursuit != 0u) gate.activation_threshold_q16 -= kQ16One / 8;
  if (gate.settled_liking != 0u) gate.activation_threshold_q16 += kQ16One / 8;
  return gate;
}

WANTING_LIKING_HD inline std::int64_t wanting_liking_positive_add(
    std::int64_t total, std::int64_t value) {
  constexpr std::int64_t kMax = 0x7fffffffffffffffLL;
  if (value <= 0) return total;
  return total > kMax - value ? kMax : total + value;
}

// WANTING reads only pending public action occurrences whose frozen ancestry
// is complete and externally authorized. LIKING reads only already-settled
// vitality and positive identified recipe credit. This observer is read-only:
// neither aggregate can create participation, evidence, credit, or settlement.
WANTING_LIKING_HD inline ResidentWantingLikingProfileV1
observe_resident_wanting_liking(
    const direct_adult_core::DirectActionOccurrence* actions,
    std::uint32_t action_count,
    const direct_adult_core::DirectActionParticipationLink* links,
    std::uint32_t link_count, const DirectAffectBodyState* affect,
    const ResidentRecipeCell* recipes, std::uint32_t recipe_count) {
  using namespace direct_adult_core;
  ResidentWantingLikingProfileV1 profile{};

  if (actions != nullptr && links != nullptr &&
      action_count <= kWantingLikingActionCapacity) {
    for (std::uint32_t i = 0u; i < action_count; ++i) {
      const DirectActionOccurrence& action = actions[i];
      if (action.state != kActionOccurrencePending ||
          action.action_ticket_id == 0u)
        continue;
      if (action.occurrence_identity_required == 0u ||
          action.occurrence_identity_complete == 0u ||
          action.participant_count == 0u ||
          action.participant_offset > link_count ||
          action.participant_count > link_count - action.participant_offset) {
        ++profile.invalid_sources;
        continue;
      }
      std::int64_t pursuit_q16 = kQ16One;
      std::uint32_t admitted = 0u;
      for (std::uint32_t j = 0u; j < action.participant_count; ++j) {
        const DirectActionParticipationLink& link =
            links[action.participant_offset + j];
        const bool authorized =
            link.occurrence_identity != 0u &&
            link.participation_identity != 0u &&
            link.authority != DirectParticipationAuthority::none &&
            link.contribution_kind != DirectContributionKind::none &&
            link.contribution_kind !=
                DirectContributionKind::ancestry_incomplete;
        if (!authorized) {
          ++profile.invalid_sources;
          continue;
        }
        ++admitted;
        pursuit_q16 = wanting_liking_positive_add(
            pursuit_q16, link.frozen_eligibility_q16);
      }
      if (admitted == 0u) continue;
      ++profile.open_pursuits;
      profile.causal_participants += admitted;
      profile.wanting_q16 =
          wanting_liking_positive_add(profile.wanting_q16, pursuit_q16);
    }
  } else if (action_count != 0u) {
    ++profile.invalid_sources;
  }

  if (affect != nullptr) {
    if (affect->count <= kAffectBodyTableCapacity) {
      profile.vitality_q16 = affect->vitality_aggregate_q16 > 0
                                 ? affect->vitality_aggregate_q16
                                 : 0;
      profile.liking_q16 = wanting_liking_positive_add(
          profile.liking_q16, profile.vitality_q16);
      for (std::uint32_t i = 0u; i < affect->count; ++i)
        profile.vitality_channels +=
            affect->entries[i].vitality_q16 > 0 ? 1u : 0u;
    } else {
      ++profile.invalid_sources;
    }
  }

  if (recipes != nullptr && recipe_count <= kWantingLikingRecipeScanCapacity) {
    for (std::uint32_t i = 0u; i < recipe_count; ++i) {
      const ResidentRecipeCell& recipe = recipes[i];
      if (recipe.credit_q16 <= 0 || recipe.logical_recipe_id == 0u ||
          recipe.revision_identity == 0u)
        continue;
      ++profile.credited_recipes;
      profile.positive_recipe_credit_q16 = wanting_liking_positive_add(
          profile.positive_recipe_credit_q16, recipe.credit_q16);
    }
    profile.liking_q16 = wanting_liking_positive_add(
        profile.liking_q16, profile.positive_recipe_credit_q16);
  } else if (recipe_count != 0u) {
    ++profile.invalid_sources;
  }
  return profile;
}

}  // namespace substrate::direct_network

#undef WANTING_LIKING_HD

#endif
