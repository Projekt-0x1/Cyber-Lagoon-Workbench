#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "bcc32_geometry.cuh"
#include "bcc32_law.cuh"

namespace substrate::bcc32 {

// One radius-one neighborhood. Roles are discovered from covariant collars;
// no descriptor, route id, byte, or semantic label enters the factor.
struct PredictionResidualNeighborhood {
  SiteWord center{};
  SiteWord positive[4]{};
  SiteWord negative[4]{};
};

struct PredictionResidualRouteToggleReceipt {
  std::uint32_t compare_positive = 0u;
  std::uint32_t compare_negative = 0u;
  std::uint32_t commit_positive = 0u;
  std::uint32_t commit_negative = 0u;
  std::uint32_t collisions = 0u;
  bool action_overflow = false;
  bool target_control_disjoint = true;
  bool material_changed = false;
  // Compact device discovery receipt.  The candidate code is 1..48 and is
  // deliberately independent of any route, byte, or center identity.
  std::uint32_t selected_candidate = 0xffffffffu;
  std::uint32_t selected_kind = 0xffffffffu;
};

struct PredictionResidualRouteToggleResult {
  PredictionResidualNeighborhood after{};
  PredictionResidualRouteToggleReceipt receipt{};
};

namespace prediction_residual_route_toggle_detail {

inline constexpr std::uint32_t kPhysicalSites = 9u;
inline constexpr std::uint32_t kActionCapacity = 8u;

enum class ActionKind : std::uint8_t {
  compare_positive,
  compare_negative,
  commit_positive,
  commit_negative,
};

struct Candidate {
  std::uint32_t q = 0u;
  std::uint32_t r = 0u;
  std::uint32_t c = 0u;
  std::uint32_t u = 0u;
  std::uint32_t probe_direction = 0u;
};

struct Action {
  std::uint8_t site = 0u;
  SiteWord first_bit = 0u;
  SiteWord second_bit = 0u;
  ActionKind kind = ActionKind::compare_positive;
  SiteWord read_masks[kPhysicalSites]{};
  SiteWord predicate_masks[kPhysicalSites]{};
  SiteWord write_masks[kPhysicalSites]{};
  std::uint32_t candidate_rank = 0u;
  bool negative_probe = false;
  bool enabled = false;
};

[[nodiscard]] __host__ __device__ inline std::uint32_t factorial(std::uint32_t value) {
  std::uint32_t result = 1u;
  for (std::uint32_t i = 2u; i <= value; ++i)
    result *= i;
  return result;
}

[[nodiscard]] __host__ __device__ inline Candidate candidate_at(std::uint32_t rank,
                                                                bool negative_probe) {
  Candidate result{};
  std::uint32_t available = 0x0fu;
  std::uint32_t* fields[4] = {&result.q, &result.r, &result.c, &result.u};
  for (std::uint32_t position = 0u; position < 4u; ++position) {
    const std::uint32_t block = factorial(3u - position);
    const std::uint32_t ordinal = rank / block;
    rank %= block;
    std::uint32_t seen = 0u;
    for (std::uint32_t basis = 0u; basis < 4u; ++basis) {
      if ((available & (1u << basis)) == 0u)
        continue;
      if (seen++ != ordinal)
        continue;
      *fields[position] = basis;
      available &= ~(1u << basis);
      break;
    }
  }
  result.probe_direction = result.q + (negative_probe ? 4u : 0u);
  return result;
}

[[nodiscard]] __host__ __device__ inline std::uint32_t positive_site(std::uint32_t basis) {
  return 1u + basis;
}

[[nodiscard]] __host__ __device__ inline std::uint32_t negative_site(std::uint32_t basis) {
  return 5u + basis;
}

[[nodiscard]] __host__ __device__ inline bool envelope_matches(SiteWord observed, SiteWord expected,
                                                               SiteWord ignored) {
  return (observed & ~ignored) == (expected & ~ignored);
}

[[nodiscard]] __host__ __device__ inline SiteWord conformation_marker(std::uint32_t basis) {
  return channel_bit(kConformationShift, basis);
}

[[nodiscard]] __host__ __device__ inline SiteWord h_collar(const Candidate& candidate) {
  return kQ | conformation_marker(candidate.q) | conformation_marker(candidate.r) |
         face_bit(candidate.probe_direction);
}

[[nodiscard]] __host__ __device__ inline SiteWord a_collar(const Candidate& candidate) {
  return kQ | energy_bit(candidate.c) | conformation_marker(candidate.q) |
         conformation_marker(candidate.r);
}

[[nodiscard]] __host__ __device__ inline SiteWord positive_gate_collar(const Candidate& candidate) {
  return kQ | face_bit(candidate.r);
}

[[nodiscard]] __host__ __device__ inline SiteWord negative_gate_collar(const Candidate& candidate) {
  return kQ | face_bit(candidate.r + 4u);
}

[[nodiscard]] __host__ __device__ inline SiteWord owner_collar(const Candidate& candidate) {
  return kQ | energy_bit(candidate.u);
}

[[nodiscard]] __host__ __device__ inline bool exactly_one(bool left, bool right) {
  return left != right;
}

[[nodiscard]] __host__ __device__ inline SiteWord neighborhood_word(
    const PredictionResidualNeighborhood& value, std::uint32_t site) {
  if (site == 0u)
    return value.center;
  if (site <= 4u)
    return value.positive[site - 1u];
  return value.negative[site - 5u];
}

__host__ __device__ inline SiteWord* neighborhood_word(PredictionResidualNeighborhood& value,
                                                       std::uint32_t site) {
  if (site == 0u)
    return &value.center;
  if (site <= 4u)
    return &value.positive[site - 1u];
  return &value.negative[site - 5u];
}

[[nodiscard]] __host__ __device__ inline bool same_action(const Action& left, const Action& right) {
  if (!left.enabled || !right.enabled || left.kind != right.kind ||
      left.site != right.site || left.first_bit != right.first_bit ||
      left.second_bit != right.second_bit)
    return false;
  for (std::uint32_t site = 0u; site < kPhysicalSites; ++site) {
    if (left.read_masks[site] != right.read_masks[site] ||
        left.predicate_masks[site] != right.predicate_masks[site] ||
        left.write_masks[site] != right.write_masks[site])
      return false;
  }
  return true;
}

__host__ __device__ inline void add_control(Action& action, std::uint32_t site, SiteWord mask) {
  action.read_masks[site] |= mask;
  action.predicate_masks[site] |= mask;
}

[[nodiscard]] __host__ __device__ inline Action make_action(std::uint32_t site, SiteWord first_bit,
                                                            SiteWord second_bit, ActionKind kind) {
  Action result{};
  result.site = static_cast<std::uint8_t>(site);
  result.first_bit = first_bit;
  result.second_bit = second_bit;
  result.kind = kind;
  result.enabled = true;
  result.read_masks[site] = first_bit | second_bit;
  result.write_masks[site] = first_bit | second_bit;
  return result;
}

[[nodiscard]] __host__ __device__ inline Action make_selected_action(
    const Candidate& candidate, ActionKind kind) {
  const std::uint32_t a_site = positive_site(candidate.c);
  const std::uint32_t gp_site = positive_site(candidate.r);
  const std::uint32_t gn_site = negative_site(candidate.r);
  const std::uint32_t u_site = positive_site(candidate.u);
  const SiteWord prediction_mask =
      channel_bit(kReactiveShift, candidate.q) |
      channel_bit(kReactiveShift, candidate.r);
  const SiteWord observation_mask =
      carrier_bit(candidate.c) | carrier_bit(candidate.c + 4u);
  const SiteWord owner_target = prediction_mask;
  const SiteWord probe = carrier_bit(candidate.probe_direction);
  const SiteWord positive_residual = carrier_bit(candidate.r);
  const SiteWord negative_residual = carrier_bit(candidate.r + 4u);
  const SiteWord compare_target = probe | positive_residual | negative_residual;
  Action action{};
  if (kind == ActionKind::compare_positive || kind == ActionKind::compare_negative) {
    const bool positive = kind == ActionKind::compare_positive;
    action = make_action(0u, probe, positive ? positive_residual : negative_residual, kind);
    add_control(action, 0u, ~action.write_masks[0u]);
    add_control(action, a_site, 0xffffffffu);
  } else {
    action = make_action(u_site, channel_bit(kReactiveShift, candidate.q),
                         channel_bit(kReactiveShift, candidate.r), kind);
    add_control(action, a_site, ~observation_mask);
    add_control(action, 0u, ~compare_target);
    add_control(action, kind == ActionKind::commit_positive ? gp_site : gn_site,
                0xffffffffu);
    add_control(action, u_site, ~owner_target);
  }
  action.candidate_rank = candidate.q;
  action.negative_probe = candidate.probe_direction >= 4u;
  return action;
}

[[nodiscard]] __host__ __device__ inline std::uint32_t selected_candidate_code(
    std::uint32_t permutation, bool negative_probe) {
  return 1u + permutation + (negative_probe ? 24u : 0u);
}

[[nodiscard]] __host__ __device__ inline ActionKind selected_action_kind(
    std::uint32_t code) {
  return static_cast<ActionKind>(code);
}

[[nodiscard]] __host__ __device__ inline std::uint32_t selected_permutation(
    std::uint32_t code) {
  return (code - 1u) % 24u;
}

[[nodiscard]] __host__ __device__ inline bool selected_negative_probe(
    std::uint32_t code) {
  return code > 24u;
}

__host__ __device__ inline void append_action(Action* actions, std::uint32_t& count,
                                              const Action& action, bool& overflow) {
  for (std::uint32_t i = 0u; i < count; ++i) {
    if (!same_action(actions[i], action))
      continue;
    return;
  }
  if (count < kActionCapacity) {
    actions[count++] = action;
  } else {
    overflow = true;
  }
}

[[nodiscard]] __host__ __device__ inline bool conflicts(const Action& left, const Action& right) {
  if (same_action(left, right))
    return false;
  for (std::uint32_t site = 0u; site < kPhysicalSites; ++site) {
    if ((left.write_masks[site] & (right.read_masks[site] | right.write_masks[site])) != 0u ||
        (right.write_masks[site] & (left.read_masks[site] | left.write_masks[site])) != 0u) {
      return true;
    }
  }
  return false;
}

__host__ __device__ inline void count_action(PredictionResidualRouteToggleReceipt& receipt,
                                             ActionKind kind) {
  if (kind == ActionKind::compare_positive)
    ++receipt.compare_positive;
  if (kind == ActionKind::compare_negative)
    ++receipt.compare_negative;
  if (kind == ActionKind::commit_positive)
    ++receipt.commit_positive;
  if (kind == ActionKind::commit_negative)
    ++receipt.commit_negative;
}

}  // namespace prediction_residual_route_toggle_detail

[[nodiscard]] __host__ __device__ inline PredictionResidualRouteToggleResult
evaluate_prediction_residual_route_toggle(const PredictionResidualNeighborhood& before) {
  using namespace prediction_residual_route_toggle_detail;
  PredictionResidualRouteToggleResult result{};
  result.after = before;
  Action actions[kActionCapacity]{};
  std::uint32_t action_count = 0u;

  for (std::uint32_t permutation = 0u; permutation < 24u; ++permutation) {
    for (std::uint32_t sign = 0u; sign < 2u; ++sign) {
      const Candidate candidate = candidate_at(permutation, sign != 0u);
      const std::uint32_t a_site = positive_site(candidate.c);
      const std::uint32_t gp_site = positive_site(candidate.r);
      const std::uint32_t gn_site = negative_site(candidate.r);
      const std::uint32_t u_site = positive_site(candidate.u);
      const SiteWord prediction_mask =
          channel_bit(kReactiveShift, candidate.q) | channel_bit(kReactiveShift, candidate.r);
      const SiteWord observation_mask = carrier_bit(candidate.c) | carrier_bit(candidate.c + 4u);
      const SiteWord owner_target = prediction_mask;
      const SiteWord probe = carrier_bit(candidate.probe_direction);
      const SiteWord positive_residual = carrier_bit(candidate.r);
      const SiteWord negative_residual = carrier_bit(candidate.r + 4u);
      const SiteWord compare_target = probe | positive_residual | negative_residual;
      const SiteWord& a = before.positive[candidate.c];
      const bool prediction_zero = (a & channel_bit(kReactiveShift, candidate.q)) != 0u;
      const bool prediction_one = (a & channel_bit(kReactiveShift, candidate.r)) != 0u;
      const bool observation_zero = (a & carrier_bit(candidate.c)) == 0u;
      const bool observation_one = (a & carrier_bit(candidate.c + 4u)) == 0u;
      const bool a_roles =
          envelope_matches(a, a_collar(candidate), prediction_mask | observation_mask) &&
          exactly_one(prediction_zero, prediction_one);

      const bool compare_roles =
          a_roles && exactly_one(observation_zero, observation_one) &&
          envelope_matches(before.center, h_collar(candidate), compare_target);
      if (compare_roles && prediction_one != observation_one) {
        const bool positive = prediction_zero;
        Action action =
            make_action(0u, probe, positive ? positive_residual : negative_residual,
                        positive ? ActionKind::compare_positive : ActionKind::compare_negative);
        action.candidate_rank = permutation;
        action.negative_probe = sign != 0u;
        add_control(action, 0u, ~action.write_masks[0u]);
        add_control(action, a_site, 0xffffffffu);
        append_action(actions, action_count, action, result.receipt.action_overflow);
      }

      const bool owner_roles =
          envelope_matches(before.positive[candidate.u], owner_collar(candidate), owner_target);
      const bool comparator_roles =
          envelope_matches(before.center, h_collar(candidate), compare_target);
      if (a_roles && comparator_roles && owner_roles && prediction_zero &&
          envelope_matches(before.positive[candidate.r], positive_gate_collar(candidate),
                           positive_residual) &&
          (before.positive[candidate.r] & positive_residual) == 0u) {
        Action action =
            make_action(u_site, channel_bit(kReactiveShift, candidate.q),
                        channel_bit(kReactiveShift, candidate.r), ActionKind::commit_positive);
        action.candidate_rank = permutation;
        action.negative_probe = sign != 0u;
        add_control(action, a_site, ~observation_mask);
        add_control(action, 0u, ~compare_target);
        add_control(action, gp_site, 0xffffffffu);
        add_control(action, u_site, ~owner_target);
        append_action(actions, action_count, action, result.receipt.action_overflow);
      }
      if (a_roles && comparator_roles && owner_roles && prediction_one &&
          envelope_matches(before.negative[candidate.r], negative_gate_collar(candidate),
                           negative_residual) &&
          (before.negative[candidate.r] & negative_residual) == 0u) {
        Action action =
            make_action(u_site, channel_bit(kReactiveShift, candidate.q),
                        channel_bit(kReactiveShift, candidate.r), ActionKind::commit_negative);
        action.candidate_rank = permutation;
        action.negative_probe = sign != 0u;
        add_control(action, a_site, ~observation_mask);
        add_control(action, 0u, ~compare_target);
        add_control(action, gn_site, 0xffffffffu);
        add_control(action, u_site, ~owner_target);
        append_action(actions, action_count, action, result.receipt.action_overflow);
      }
    }
  }

  // Never execute an enumeration-order-dependent prefix from a malformed
  // neighborhood. Any overflow rejects the complete local action set.
  if (result.receipt.action_overflow)
    return result;

  bool rejected[kActionCapacity]{};
  for (std::uint32_t i = 0u; i < action_count; ++i) {
    for (std::uint32_t site = 0u; site < kPhysicalSites; ++site) {
      result.receipt.target_control_disjoint &=
          (actions[i].predicate_masks[site] & actions[i].write_masks[site]) == 0u;
    }
    for (std::uint32_t j = i + 1u; j < action_count; ++j) {
      if (!conflicts(actions[i], actions[j]))
        continue;
      rejected[i] = true;
      rejected[j] = true;
      ++result.receipt.collisions;
    }
  }

  std::uint32_t accepted_count = 0u;
  std::uint32_t accepted_index = 0u;
  for (std::uint32_t i = 0u; i < action_count; ++i) {
    if (rejected[i])
      continue;
    ++accepted_count;
    accepted_index = i;
  }
  // One center may publish one immutable-snapshot action. Multiple accepted
  // actions are physically ambiguous and therefore abstain without priority.
  if (accepted_count != 1u)
    return result;
  const Action& selected = actions[accepted_index];
  result.receipt.selected_candidate = selected_candidate_code(
      selected.candidate_rank, selected.negative_probe);
  result.receipt.selected_kind = static_cast<std::uint32_t>(selected.kind);
  {
    SiteWord* word = neighborhood_word(result.after, selected.site);
    const SiteWord before_word = *word;
    controlled_transpose(*word, selected.first_bit, *word, selected.second_bit, true);
    result.receipt.material_changed = *word != before_word;
    count_action(result.receipt, selected.kind);
  }
  return result;
}

[[nodiscard]] __host__ __device__ inline PredictionResidualNeighborhood
apply_prediction_residual_route_toggle(const PredictionResidualNeighborhood& before) {
  return evaluate_prediction_residual_route_toggle(before).after;
}

void apply_prediction_residual_route_toggle_batch(
    const PredictionResidualNeighborhood* device_before,
    PredictionResidualNeighborhood* device_after,
    PredictionResidualRouteToggleReceipt* device_receipts, std::uint32_t count,
    cudaStream_t stream = nullptr);

}  // namespace substrate::bcc32
