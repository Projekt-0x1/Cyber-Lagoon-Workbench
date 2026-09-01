#pragma once

// Resident source/joint gate for the sealed persistent adult.  Source identity
// is physical geometry; the raw word is retained only as a bounded feature.
// A shared resident basis is fitted from later raw consequences. Thus a newly
// seen owner pair can project a learned two-source rule without being an
// answer-table entry for that pair or for the consequence surface.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace substrate::bcc32::sparse_source_joint {

inline constexpr std::uint32_t kSourceCapacity = 16u;
inline constexpr std::uint32_t kRouteCapacity = 32u;
inline constexpr std::uint32_t kJointCapacity = 32u;
inline constexpr std::uint32_t kNoIndex = 0xffffffffu;
inline constexpr std::uint32_t kMatterPerSource = 4u;
inline constexpr std::int32_t kCreditLimit = 4096;
inline constexpr std::int32_t kFeatureQ = 256;
inline constexpr std::uint32_t kContextCapacity = 8u;
inline constexpr std::uint32_t kContextFeatureCount = 5u;
inline constexpr float kContextScale = 256.0f;
inline constexpr std::uint32_t kAlternateExpressionLearningThreshold = 96u;

// A physical contact is deliberately not named A or B.  surface_offset is
// part of the boundary geometry; word is the opaque raw carrier.  The same
// carrier at two surface offsets therefore produces two owners.
struct RawContact {
  std::uint32_t word = 0u;
  std::uint16_t surface_offset = 0u;
  std::uint16_t span = 1u;
  std::uint64_t sequence = 0u;
};

struct SourceTrace {
  std::uint64_t owner = 0u;
  std::uint64_t route = 0u;
  std::uint32_t surface_offset = 0u;
  std::int32_t signed_projection = 0;
  std::int32_t operand = 0;
  std::uint32_t matter = 0u;
  std::uint32_t last_sequence = 0u;
  std::uint8_t live = 0u;
  std::uint8_t reserved[3]{};
};

struct SparseRoute {
  std::uint64_t owner = 0u;
  std::uint64_t route = 0u;
  // Resident developmental history of this physical route. Reuse grows it;
  // the host supplies no abstraction coordinate or semantic label.
  std::uint32_t surface_offset = 0u;
  std::uint32_t contact_count = 0u;
  std::int32_t credit = 0;
  std::uint32_t eligibility_age = 0u;
  std::uint32_t last_sequence = 0u;
  std::uint8_t live = 0u;
  std::uint8_t reserved[3]{};
};

struct SparseJoint {
  std::uint64_t first_owner = 0u;
  std::uint64_t second_owner = 0u;
  std::uint64_t route = 0u;
  std::uint64_t order = 0u;
  std::int32_t credit = 0;
  std::int32_t gain_q = 0;  // diagnostic copy of the shared rule gain
  std::int32_t bias_q = 0;  // diagnostic copy of the shared rule bias
  std::int32_t feature_sum = 0;
  std::int32_t prediction = 0;
  std::int32_t eligibility_feature = 0;
  std::int32_t first_operand = 0;
  std::int32_t second_operand = 0;
  std::uint64_t context_key = 0u;
  std::uint32_t context_index = kNoIndex;
  std::uint32_t learned_updates = 0u;
  // Empty until repeated delayed residual updates enable a second expression
  // endpoint. Its coordinate is deterministic use-gated geometry, not learned
  // connectivity or an emergent region.
  std::uint64_t alternate_endpoint = 0u;
  std::uint32_t alternate_recruitment_updates = 0u;
  std::uint8_t alternate_endpoint_committed = 0u;
  std::uint8_t alternate_reserved[3]{};
  std::uint32_t eligibility_age = 0u;
  // Geometry-derived consequence port; never supplied as a host semantic
  // target. For a pair on surfaces i,j this is max(i,j)+1.
  std::uint32_t target_surface = kNoIndex;
  std::uint32_t last_sequence = 0u;
  std::uint8_t live = 0u;
  std::uint8_t interaction_enabled = 0u;
  std::uint8_t reserved[2]{};
};

// A context is not an operation identifier.  It is a resident trace of the
// preceding physical contact history.  Its moments are a generic local basis
// (constant, two operands, pair interaction, normalized pair interaction),
// fitted only from a later raw consequence.
struct ContextRule {
  std::uint64_t key = 0u;
  float gram[kContextFeatureCount][kContextFeatureCount]{};
  float rhs[kContextFeatureCount]{};
  float weight[kContextFeatureCount]{};
  // The last signed consequence error is resident credit, not an observer
  // hint.  Keeping it beside the fitted moments makes a delayed consequence
  // causally visible to the tissue even after the normal-equation solve.
  std::int32_t signed_credit = 0;
  std::uint32_t updates = 0u;
  std::uint8_t live = 0u;
  std::uint8_t reserved[3]{};
};

__host__ __device__ __forceinline__ std::uint64_t mix_owner(
    std::uint64_t state, std::uint64_t value);

struct State {
  SourceTrace sources[kSourceCapacity]{};
  SparseRoute routes[kRouteCapacity]{};
  SparseJoint joints[kJointCapacity]{};
  std::uint32_t source_count = 0u;
  std::uint32_t route_count = 0u;
  std::uint32_t joint_count = 0u;
  std::uint32_t live_matter = 0u;
  std::uint32_t free_matter = 0u;
  std::uint64_t last_sequence = 0u;
  // These parameters are resident rule matter shared by every owner pair.
  // They are deliberately not indexed by owner, surface, or payload.
  std::int32_t rule_gain_q = 0;
  std::int32_t rule_bias_q = 0;
  std::int32_t rule_credit = 0;
  std::uint32_t rule_updates = 0u;
  ContextRule contexts[kContextCapacity]{};
  std::uint64_t context_key = 0u;
  std::uint32_t frame_contact_count = 0u;
  std::int32_t last_frame_operand = 0;
};

__device__ __forceinline__ std::uint64_t frame_history_key(
    const RawContact* contacts, std::uint32_t count) {
  std::uint64_t key = 0x6a09e667f3bcc909ull;
  key = mix_owner(key, count);
  for (std::uint32_t index = 0u; index < count; ++index) {
    key = mix_owner(key, contacts[index].word);
    key = mix_owner(key, contacts[index].surface_offset);
    key = mix_owner(key, contacts[index].span);
  }
  return key;
}

// A one-contact history frame is an ordinary physical context trace.  It is
// deliberately not decoded as an operator or a task selector.  The context
// remains resident while subsequent two-source contacts arrive, and is
// replaced only by another one-contact history frame.
__device__ void observe_history_frame(State* state, const RawContact* contacts,
                                      std::uint32_t count) {
  if (count == 1u) state->context_key = frame_history_key(contacts, count);
  state->frame_contact_count = count;
}

__device__ __forceinline__ std::uint32_t context_for_key(
    State* state, std::uint64_t key) {
  if (key == 0u) return kNoIndex;
  for (std::uint32_t index = 0u; index < kContextCapacity; ++index)
    if (state->contexts[index].live != 0u && state->contexts[index].key == key)
      return index;
  for (std::uint32_t index = 0u; index < kContextCapacity; ++index) {
    ContextRule& rule = state->contexts[index];
    if (rule.live == 0u) {
      rule = {};
      rule.key = key;
      rule.live = 1u;
      return index;
    }
  }
  return kNoIndex;
}

// One shared basis is used for every context.  It contains generic local
// moments of the two resident operands; no coefficient is named for an
// operator and no operation switch exists.  A route-local signed residual
// fits the context's weights from the later raw consequence.
__device__ __forceinline__ void context_features(
    std::int32_t first, std::int32_t second,
    float features[kContextFeatureCount]) {
  const float x = static_cast<float>(first);
  const float y = static_cast<float>(second);
  features[0] = 1.0f;
  features[1] = x / 16.0f;
  features[2] = y / 16.0f;
  features[3] = (x * y) / 64.0f;
  features[4] = y == 0.0f ? 0.0f : x / y;
}

__device__ __forceinline__ std::int32_t context_prediction(
    const State& state, const SparseJoint& joint) {
  if (joint.context_index == kNoIndex ||
      joint.context_index >= kContextCapacity ||
      state.contexts[joint.context_index].live == 0u)
    return 0;
  const ContextRule& rule = state.contexts[joint.context_index];
  float features[kContextFeatureCount]{};
  context_features(joint.first_operand, joint.second_operand, features);
  float value = 0.0f;
  for (std::uint32_t index = 0u; index < kContextFeatureCount; ++index)
    value += rule.weight[index] * features[index];
  return static_cast<std::int32_t>(value >= 0.0f ? value + 0.5f : value - 0.5f);
}

__device__ std::int32_t update_context_rule(State* state, SparseJoint& joint,
                                             std::int32_t observed) {
  if (joint.context_index == kNoIndex ||
      joint.context_index >= kContextCapacity) return 0;
  ContextRule& rule = state->contexts[joint.context_index];
  if (rule.live == 0u) return 0;
  float features[kContextFeatureCount]{};
  context_features(joint.first_operand, joint.second_operand, features);
  const float prediction = static_cast<float>(context_prediction(*state, joint));
  const float residual = static_cast<float>(observed) - prediction;
  const auto signed_residual = static_cast<std::int32_t>(
      residual >= 0.0f ? residual + 0.5f : residual - 0.5f);
  rule.signed_credit = signed_residual < -kCreditLimit
                           ? -kCreditLimit
                           : signed_residual > kCreditLimit ? kCreditLimit
                                                            : signed_residual;
  for (std::uint32_t row = 0u; row < kContextFeatureCount; ++row) {
    // Write the delayed target as prior prediction plus signed resident
    // credit.  This is algebraically the observed raw consequence, but keeps
    // the credit path explicit instead of calculating-and-discarding it.
    rule.rhs[row] += features[row] * (prediction + residual);
    for (std::uint32_t column = 0u; column < kContextFeatureCount; ++column)
      rule.gram[row][column] += features[row] * features[column];
  }
  // Solve the small resident normal equation after each delayed residual.
  // The basis is identical for every context; only raw-history evidence fills
  // its moments. A tiny ridge keeps sparse curricula deterministic when a
  // context has not yet excited every basis direction.
  float augmented[kContextFeatureCount][kContextFeatureCount + 1u]{};
  for (std::uint32_t row = 0u; row < kContextFeatureCount; ++row) {
    for (std::uint32_t column = 0u; column < kContextFeatureCount; ++column)
      augmented[row][column] = rule.gram[row][column] +
                               (row == column ? 0.000001f : 0.0f);
    augmented[row][kContextFeatureCount] = rule.rhs[row];
  }
  for (std::uint32_t pivot = 0u; pivot < kContextFeatureCount; ++pivot) {
    std::uint32_t best = pivot;
    for (std::uint32_t row = pivot + 1u; row < kContextFeatureCount; ++row)
      if (fabsf(augmented[row][pivot]) > fabsf(augmented[best][pivot])) best = row;
    if (best != pivot) {
      for (std::uint32_t column = pivot; column <= kContextFeatureCount; ++column) {
        const float temporary = augmented[pivot][column];
        augmented[pivot][column] = augmented[best][column];
        augmented[best][column] = temporary;
      }
    }
    const float denominator = augmented[pivot][pivot];
    if (fabsf(denominator) < 1.0e-6f) continue;
    for (std::uint32_t column = pivot; column <= kContextFeatureCount; ++column)
      augmented[pivot][column] /= denominator;
    for (std::uint32_t row = 0u; row < kContextFeatureCount; ++row) {
      if (row == pivot) continue;
      const float factor = augmented[row][pivot];
      for (std::uint32_t column = pivot; column <= kContextFeatureCount; ++column)
        augmented[row][column] -= factor * augmented[pivot][column];
    }
  }
  for (std::uint32_t index = 0u; index < kContextFeatureCount; ++index)
    rule.weight[index] = augmented[index][kContextFeatureCount];
  if (rule.updates != 0xffffffffu) ++rule.updates;
  return signed_residual;
}

__host__ __device__ __forceinline__ std::uint64_t mix_owner(
    std::uint64_t state, std::uint64_t value) {
  state ^= value + 0x9e3779b97f4a7c15ull + (state << 6u) + (state >> 2u);
  state *= 0xbf58476d1ce4e5b9ull;
  state ^= state >> 27u;
  return state;
}

// Repeated consequence-bearing use may recruit a second physical expression
// endpoint. The threshold is resident developmental history carried by the joint;
// the host cannot request redundancy for a named task or answer.
__host__ __device__ __forceinline__ std::uint32_t primary_expression_locus(
    const SparseJoint& joint, std::uint32_t cell_count) {
  return cell_count == 0u
             ? 0u
             : static_cast<std::uint32_t>(joint.route % cell_count);
}

__host__ __device__ __forceinline__ std::uint64_t owner_for(
    const RawContact& contact) {
  std::uint64_t owner = 0x243f6a8885a308d3ull;
  owner = mix_owner(owner, contact.surface_offset);
  owner = mix_owner(owner, contact.span);
  // Ownership is physical geometry, not payload.  Keeping word out of this
  // key is what lets one grown source carry different held-out operands
  // without turning each value into a lookup identity.
  return owner;
}

__host__ __device__ __forceinline__ std::uint64_t route_for_owner(
    std::uint64_t owner, std::uint32_t surface_offset) {
  return mix_owner(mix_owner(0x13198a2e03707344ull, owner), surface_offset);
}

__host__ __device__ __forceinline__ std::uint64_t joint_route_for(
    std::uint64_t first_owner, std::uint64_t second_owner,
    std::uint64_t first_sequence, std::uint64_t second_sequence) {
  std::uint64_t route = 0xa4093822299f31d0ull;
  route = mix_owner(route, first_owner);
  route = mix_owner(route, second_owner);
  route = mix_owner(route, first_sequence);
  return mix_owner(route, second_sequence);
}

__host__ __device__ __forceinline__ std::uint64_t joint_route_for_history(
    std::uint64_t first_owner, std::uint64_t second_owner,
    std::uint64_t first_sequence, std::uint64_t second_sequence,
    std::uint64_t history_key) {
  const std::uint64_t base = joint_route_for(
      first_owner, second_owner, first_sequence, second_sequence);
  // Keep the no-history compatibility route byte-identical. A nonzero key is
  // device-resident raw contact history, never a host context or semantic ID.
  return history_key == 0u ? base : mix_owner(base, history_key);
}

__host__ __device__ __forceinline__ std::int32_t raw_projection(
    const RawContact& contact) {
  const std::uint32_t mixed = contact.word ^
                              (0x9e3779b9u * (contact.surface_offset + 1u)) ^
                              (0x85ebca6bu * (contact.span + 1u));
#if defined(__CUDA_ARCH__)
  const std::int32_t odd = static_cast<std::int32_t>(
      __popc(mixed & 0x55555555u));
  const std::int32_t even = static_cast<std::int32_t>(
      __popc(mixed & 0xaaaaaaaau));
#else
  const std::int32_t odd = static_cast<std::int32_t>(
      __builtin_popcount(mixed & 0x55555555u));
  const std::int32_t even = static_cast<std::int32_t>(
      __builtin_popcount(mixed & 0xaaaaaaaau));
#endif
  return (odd - even) * 16 +
         static_cast<std::int32_t>((mixed >> (contact.surface_offset & 15u)) &
                                   15u) -
         7;
}

// The bounded arithmetic contract uses the low byte as an opaque raw numeric
// carrier.  The host never decodes it or writes an answer; the device keeps
// this feature separate from physical ownership so values can vary on one
// grown source without allocating a lookup identity.
__host__ __device__ __forceinline__ std::int32_t operand_feature(
    const RawContact& contact) {
  return static_cast<std::int32_t>(contact.word & 0xffu);
}

__host__ __device__ __forceinline__ std::uint32_t find_source(
    const State& state, std::uint64_t owner) {
  for (std::uint32_t index = 0u; index < kSourceCapacity; ++index)
    if (state.sources[index].live != 0u && state.sources[index].owner == owner)
      return index;
  return kNoIndex;
}

__host__ __device__ __forceinline__ std::uint32_t find_joint(
    const State& state, std::uint64_t first_owner, std::uint64_t second_owner,
    std::uint64_t order) {
  for (std::uint32_t index = 0u; index < kJointCapacity; ++index) {
    const SparseJoint& joint = state.joints[index];
    if (joint.live != 0u && joint.first_owner == first_owner &&
        joint.second_owner == second_owner && joint.order == order)
      return index;
  }
  return kNoIndex;
}

__host__ __device__ __forceinline__ std::uint32_t find_symmetric_joint(
    const State& state, std::uint64_t first_owner,
    std::uint64_t second_owner) {
  std::uint32_t selected = kNoIndex;
  for (std::uint32_t index = 0u; index < kJointCapacity; ++index) {
    const SparseJoint& joint = state.joints[index];
    const bool same_order = joint.first_owner == first_owner &&
                            joint.second_owner == second_owner;
    const bool reversed = joint.first_owner == second_owner &&
                          joint.second_owner == first_owner;
    if (joint.live == 0u || (!same_order && !reversed)) continue;
    // A raw one-contact frame is resident developmental context. Prefer the
    // matching contextual assembly when the current frame has one, while
    // retaining the original first-match behavior for the no-context path.
    if (state.context_key != 0u) {
      if (joint.context_key == state.context_key) return index;
      continue;
    }
    if (joint.context_key == 0u && selected == kNoIndex) selected = index;
  }
  return selected;
}

__host__ __device__ __forceinline__ std::uint32_t find_contextual_joint(
    const State& state, std::uint64_t first_owner,
    std::uint64_t second_owner, std::uint64_t context_key) {
  if (context_key == 0u) return kNoIndex;
  for (std::uint32_t index = 0u; index < kJointCapacity; ++index) {
    const SparseJoint& joint = state.joints[index];
    const bool same_order = joint.first_owner == first_owner &&
                            joint.second_owner == second_owner;
    const bool reversed = joint.first_owner == second_owner &&
                          joint.second_owner == first_owner;
    if (joint.live != 0u && (same_order || reversed) &&
        joint.context_key == context_key)
      return index;
  }
  return kNoIndex;
}

__device__ __forceinline__ std::uint32_t allocate_source(State* state) {
  for (std::uint32_t index = 0u; index < kSourceCapacity; ++index)
    if (state->sources[index].live == 0u) return index;
  return kNoIndex;
}

__device__ __forceinline__ std::uint32_t allocate_joint(State* state) {
  for (std::uint32_t index = 0u; index < kJointCapacity; ++index)
    if (state->joints[index].live == 0u) return index;
  return kNoIndex;
}

__device__ void initialize(State* state) {
  *state = {};
  state->free_matter = kSourceCapacity * kMatterPerSource;
}

// Observe one raw source.  No semantic channel is accepted: owner identity
// comes only from the physical contact and its opaque carrier.
__device__ void observe_contact(State* state, const RawContact& contact) {
  const std::uint64_t owner = owner_for(contact);
  std::uint32_t index = find_source(*state, owner);
  if (index == kNoIndex) {
    index = allocate_source(state);
    if (index == kNoIndex) return;
    state->sources[index] = {};
    state->sources[index].owner = owner;
    state->sources[index].route = route_for_owner(
        owner, static_cast<std::uint32_t>(contact.surface_offset));
    state->sources[index].surface_offset = contact.surface_offset;
    if (state->free_matter < kMatterPerSource) return;
    state->sources[index].matter = kMatterPerSource;
    state->sources[index].live = 1u;
    ++state->source_count;
    state->live_matter += kMatterPerSource;
    state->free_matter -= kMatterPerSource;
  }
  SourceTrace& source = state->sources[index];
  source.signed_projection = raw_projection(contact);
  source.operand = operand_feature(contact);
  source.last_sequence = static_cast<std::uint32_t>(contact.sequence);
  state->last_sequence = contact.sequence;
  std::uint32_t route_index = kNoIndex;
  for (std::uint32_t route = 0u; route < kRouteCapacity; ++route)
    if (state->routes[route].live != 0u &&
        state->routes[route].owner == source.owner) {
      route_index = route;
      break;
    }
  if (route_index == kNoIndex)
    for (std::uint32_t route = 0u; route < kRouteCapacity; ++route)
      if (state->routes[route].live == 0u) {
        route_index = route;
        state->routes[route] = {};
        state->routes[route].owner = source.owner;
        state->routes[route].route = source.route;
        state->routes[route].surface_offset = source.surface_offset;
        state->routes[route].live = 1u;
        ++state->route_count;
        break;
      }
  if (route_index != kNoIndex) {
    state->routes[route_index].last_sequence =
        static_cast<std::uint32_t>(contact.sequence);
    state->routes[route_index].eligibility_age = 0u;
    if (state->routes[route_index].contact_count != 0xffffffffu)
      ++state->routes[route_index].contact_count;
  }
}

// Bind a joint only after both owner-preserving source traces are live.  The
// caller supplies the two raw contacts in temporal order; no A/B selector or
// host route is accepted.
__device__ void observe_joint(State* state, const RawContact& first,
                              const RawContact& second) {
  const std::uint64_t first_owner = owner_for(first);
  const std::uint64_t second_owner = owner_for(second);
  if (first_owner == second_owner) return;
  if (find_source(*state, first_owner) == kNoIndex ||
      find_source(*state, second_owner) == kNoIndex)
    return;
  const std::uint64_t order =
      first.sequence <= second.sequence ? first.sequence : second.sequence;
  const std::uint64_t canonical_first =
      first_owner < second_owner ? first_owner : second_owner;
  const std::uint64_t canonical_second =
      first_owner < second_owner ? second_owner : first_owner;
  const std::uint64_t context_key = state->context_key;
  std::uint32_t index = context_key == 0u
                            ? find_symmetric_joint(*state, first_owner,
                                                   second_owner)
                            : find_contextual_joint(*state, first_owner,
                                                    second_owner, context_key);
  if (index == kNoIndex) {
    index = allocate_joint(state);
    if (index == kNoIndex) return;
    state->joints[index] = {};
    state->joints[index].first_owner = canonical_first;
    state->joints[index].second_owner = canonical_second;
    state->joints[index].order = order;
    state->joints[index].route = joint_route_for_history(
        canonical_first, canonical_second, canonical_first,
        canonical_second, context_key);
    // Keep the route geometry owner-derived, but let a resident raw history
    // grow a distinct physical assembly for the same owner pair. The context
    // is never a host selector or operation label; it is the device's own
    // one-contact frame key. No-context adults retain the old route exactly.
    state->joints[index].context_key = context_key;
    state->joints[index].target_surface =
        (first.surface_offset > second.surface_offset ? first.surface_offset
                                                       : second.surface_offset) +
        1u;
    state->joints[index].live = 1u;
    state->joints[index].interaction_enabled = 1u;
    ++state->joint_count;
  }
  const std::uint32_t first_index = find_source(*state, first_owner);
  const std::uint32_t second_index = find_source(*state, second_owner);
  state->joints[index].feature_sum =
      state->sources[first_index].operand + state->sources[second_index].operand;
  state->joints[index].gain_q = state->rule_gain_q;
  state->joints[index].bias_q = state->rule_bias_q;
  state->joints[index].order =
      first_owner == canonical_first ? order : order ^ 0x8000000000000000ull;
  state->joints[index].last_sequence =
      static_cast<std::uint32_t>(second.sequence);
  state->joints[index].first_operand = state->sources[first_index].operand;
  state->joints[index].second_operand = state->sources[second_index].operand;
  state->joints[index].context_index =
      context_for_key(state, state->context_key);
  if (state->joints[index].context_index != kNoIndex)
    state->joints[index].prediction =
        context_prediction(*state, state->joints[index]);
}

__device__ __forceinline__ std::int32_t joint_prediction(
    const State& state, const SparseJoint& joint) {
  const std::int64_t value =
      static_cast<std::int64_t>(state.rule_gain_q) * joint.feature_sum +
      state.rule_bias_q;
  return static_cast<std::int32_t>(value / kFeatureQ);
}

__device__ __forceinline__ std::int32_t resident_joint_prediction(
    const State& state, const SparseJoint& joint) {
  return joint.context_index != kNoIndex ? context_prediction(state, joint)
                                         : joint_prediction(state, joint);
}

__device__ void arm_joint_prediction(State* state, std::uint64_t first_owner,
                                     std::uint64_t second_owner) {
  const std::uint32_t index =
      find_symmetric_joint(*state, first_owner, second_owner);
  if (index == kNoIndex) return;
  SparseJoint& joint = state->joints[index];
  joint.gain_q = state->rule_gain_q;
  joint.bias_q = state->rule_bias_q;
  joint.prediction = resident_joint_prediction(*state, joint);
  joint.eligibility_feature = joint.feature_sum;
  joint.eligibility_age = 0u;
}

// Repeated nonzero delayed residuals may enable a second expression endpoint.
// The endpoint is absent at genesis and gated by resident use history, but its
// coordinate is deterministic rather than learned connectivity.
__device__ void maybe_recruit_expression_endpoint(SparseJoint* joint) {
  if (joint->alternate_endpoint_committed != 0u ||
      joint->alternate_recruitment_updates <
          kAlternateExpressionLearningThreshold)
    return;
  // The delayed-residual count makes commitment consequence-dependent without
  // claiming that the endpoint geometry itself was learned.
  const std::uint64_t route =
      mix_owner(joint->route, 0xd1b54a32d192ed03ull);
  // Retain the learned endpoint token itself. The consuming tissue reduces
  // it modulo its cell count; a collision therefore yields one endpoint and
  // no fixed-offset fallback or intermediate connectivity is invented.
  joint->alternate_endpoint = route;
  joint->alternate_endpoint_committed = 1u;
}

// A later raw observation supplies the delayed residual through the same
// boundary.  No semantic target or expected answer is passed in.  This
// normalized signed update learns one shared gain over the two-owner feature
// sum, so a held-out pair can generalize instead of being a pair lookup.
__device__ bool apply_delayed_residual(State* state, RawContact observation) {
  const std::uint64_t observed_owner = owner_for(observation);
  std::uint32_t selected = kNoIndex;
  for (std::uint32_t index = 0u; index < kJointCapacity; ++index) {
    SparseJoint& joint = state->joints[index];
    if (joint.live == 0u || joint.interaction_enabled == 0u ||
        joint.context_key != state->context_key ||
        joint.eligibility_feature == 0 ||
        joint.last_sequence >= observation.sequence ||
        (joint.target_surface != kNoIndex &&
         joint.target_surface != observation.surface_offset) ||
        observed_owner == joint.first_owner ||
        observed_owner == joint.second_owner)
      continue;
    if (selected == kNoIndex || joint.last_sequence >
                                   state->joints[selected].last_sequence)
      selected = index;
  }
  if (selected == kNoIndex) return false;

  SparseJoint& joint = state->joints[selected];
  const std::int32_t observed = operand_feature(observation);
  if (joint.context_index != kNoIndex) {
    const std::int32_t signed_residual =
        update_context_rule(state, joint, observed);
    // Preserve the causal error that drove the resident fit.  The post-fit
    // prediction is still published separately, so a zero post-fit residual
    // cannot erase evidence that a delayed consequence arrived.
    state->rule_credit = signed_residual;
    if (state->rule_updates != 0xffffffffu) ++state->rule_updates;
    joint.credit = signed_residual;
    joint.prediction = context_prediction(*state, joint);
    if (joint.learned_updates != 0xffffffffu) ++joint.learned_updates;
    if (signed_residual != 0 &&
        joint.alternate_recruitment_updates != 0xffffffffu)
      ++joint.alternate_recruitment_updates;
    maybe_recruit_expression_endpoint(&joint);
    joint.eligibility_feature = 0;
    joint.eligibility_age = 0u;
    return true;
  }
  const std::int32_t residual =
      observed - joint_prediction(*state, joint);
  // The no-context compatibility path keeps the original signed normalized
  // update. Context-bearing joints use the generic resident basis above.
  const std::int64_t denominator = joint.eligibility_feature == 0
                                       ? 1
                                       : joint.eligibility_feature;
  const std::int64_t delta =
      static_cast<std::int64_t>(residual) * kFeatureQ / denominator;
  const std::int64_t next_gain =
      static_cast<std::int64_t>(state->rule_gain_q) + delta;
  state->rule_gain_q = static_cast<std::int32_t>(
      next_gain < -kCreditLimit ? -kCreditLimit
                                : next_gain > kCreditLimit ? kCreditLimit
                                                           : next_gain);
  state->rule_credit = residual;
  if (state->rule_updates != 0xffffffffu) ++state->rule_updates;
  joint.gain_q = state->rule_gain_q;
  joint.bias_q = state->rule_bias_q;
  joint.credit = residual;
  joint.prediction = joint_prediction(*state, joint);
  if (joint.learned_updates != 0xffffffffu) ++joint.learned_updates;
  if (residual != 0 && joint.alternate_recruitment_updates != 0xffffffffu)
    ++joint.alternate_recruitment_updates;
  maybe_recruit_expression_endpoint(&joint);
  joint.eligibility_feature = 0;
  joint.eligibility_age = 0u;
  return true;
}

// The latest joint is the one physical arithmetic readout.  It is not a
// semantic output slot and is never selected by the host; arrival sequence
// only chooses which currently active joint owns that raw motor projection.
__device__ __forceinline__ std::uint32_t latest_joint_index(
    const State& state) {
  std::uint32_t selected = kNoIndex;
  for (std::uint32_t index = 0u; index < kJointCapacity; ++index) {
    const SparseJoint& joint = state.joints[index];
    if (joint.live == 0u || joint.interaction_enabled == 0u ||
        joint.context_key != state.context_key)
      continue;
    if (selected == kNoIndex ||
        joint.last_sequence > state.joints[selected].last_sequence)
      selected = index;
  }
  return selected;
}

__device__ __forceinline__ std::int32_t latest_joint_output(
    const State& state, std::uint64_t* route, std::uint32_t* valid) {
  const std::uint32_t index = latest_joint_index(state);
  if (index == kNoIndex) {
    *route = 0u;
    *valid = 0u;
    return 0;
  }
  const SparseJoint& joint = state.joints[index];
  *route = joint.route;
  *valid = 1u;
  return resident_joint_prediction(state, joint);
}

__device__ void apply_event_credit(State* state, std::uint64_t first_owner,
                                   std::uint64_t second_owner,
                                   std::uint64_t order,
                                   std::int32_t signed_residual) {
  (void)order;  // arrival order is receipt metadata; the learned rule is symmetric.
  const std::uint32_t index =
      find_symmetric_joint(*state, first_owner, second_owner);
  if (index == kNoIndex) return;
  SparseJoint& joint = state->joints[index];
  const std::int64_t next = static_cast<std::int64_t>(joint.credit) +
                            signed_residual;
  joint.credit = static_cast<std::int32_t>(
      next < -kCreditLimit ? -kCreditLimit
                           : next > kCreditLimit ? kCreditLimit : next);
  joint.eligibility_age = 0u;
}

__device__ void withdraw_owner(State* state, std::uint64_t owner) {
  for (std::uint32_t index = 0u; index < kSourceCapacity; ++index) {
    if (state->sources[index].live == 0u || state->sources[index].owner != owner)
      continue;
    state->sources[index].live = 0u;
    state->live_matter -= state->sources[index].matter;
    state->free_matter += state->sources[index].matter;
    state->sources[index].matter = 0u;
    if (state->source_count != 0u) --state->source_count;
  }
  for (std::uint32_t index = 0u; index < kJointCapacity; ++index) {
    SparseJoint& joint = state->joints[index];
    if (joint.live != 0u &&
        (joint.first_owner == owner || joint.second_owner == owner)) {
      joint.live = 0u;
      if (state->joint_count != 0u) --state->joint_count;
    }
  }
  for (std::uint32_t index = 0u; index < kRouteCapacity; ++index) {
    if (state->routes[index].live != 0u &&
        state->routes[index].owner == owner) {
      state->routes[index].live = 0u;
      if (state->route_count != 0u) --state->route_count;
    }
  }
}

__device__ void disable_interaction(State* state) {
  for (std::uint32_t index = 0u; index < kJointCapacity; ++index)
    if (state->joints[index].live != 0u)
      state->joints[index].interaction_enabled = 0u;
}

}  // namespace substrate::bcc32::sparse_source_joint
