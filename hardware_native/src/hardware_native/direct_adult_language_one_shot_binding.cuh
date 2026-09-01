#ifndef HARDWARE_NATIVE_DIRECT_ADULT_LANGUAGE_ONE_SHOT_BINDING_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_LANGUAGE_ONE_SHOT_BINDING_CUH

// f.language_one_shot_crossmodal_binding (#1578). A novel opaque surface may
// acquire a provisional resident association after one verified lived
// sensorimotor closure. One exposure never becomes durable authority: later
// contradictory closure revises or retracts the site.

#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_adult_language_one_shot_abi.cuh"
#include "hardware_native/direct_adult_multimodal_grounding.cuh"
#include "hardware_native/direct_exact_history.cuh"
#include "hardware_native/direct_network_brain.cuh"

namespace substrate::direct_network {

#ifndef DIRECT_LANGUAGE_ONE_SHOT_DEVICE
#define DIRECT_LANGUAGE_ONE_SHOT_DEVICE __device__ inline
#endif

DIRECT_LANGUAGE_ONE_SHOT_DEVICE bool language_one_shot_seen(
    const DirectLanguageOneShotState& state, std::uint64_t identity) {
  for (std::uint32_t i = 0u; i < state.admitted_motor_count; ++i)
    if (state.admitted_motor_identities[i] == identity) return true;
  return false;
}
DIRECT_LANGUAGE_ONE_SHOT_DEVICE bool language_one_shot_verified_return(
    const DirectExactHistoryRecord* records, std::uint32_t count,
    const DirectExactHistoryRecord& motor,
    DirectExactHistoryRecord* verified_return) {
  for (std::uint32_t i = 0u; i < count; ++i) {
    const DirectExactHistoryRecord& consequence = records[i];
    if (consequence.kind != DirectExactHistoryKind::world_return ||
        (consequence.flags & kDirectHistoryVerifiedObservation) == 0u)
      continue;
    const bool same_ticket = consequence.identity == motor.identity ||
                             consequence.parent_identity == motor.identity;
    if (same_ticket && consequence.source == motor.source &&
        consequence.subject == motor.subject) {
      if (verified_return != nullptr) *verified_return = consequence;
      return true;
    }
  }
  return false;
}
DIRECT_LANGUAGE_ONE_SHOT_DEVICE bool language_one_shot_preceding_coalition(
    const DirectExactHistoryRecord* records, std::uint32_t motor_index,
    DirectExactHistoryRecord* surface, DirectExactHistoryRecord* ground) {
  bool have_ground = false;
  for (std::uint32_t cursor = motor_index; cursor > 0u; --cursor) {
    const DirectExactHistoryRecord& record = records[cursor - 1u];
    if (record.kind != DirectExactHistoryKind::sensory_contact) continue;
    if (!have_ground) {
      *ground = record;
      have_ground = true;
      continue;
    }
    if (record.subject == ground->subject) continue;
    if (ground->resident_tick < record.resident_tick ||
        ground->resident_tick - record.resident_tick >
            kLanguageOneShotMaximumGap)
      return false;
    *surface = record;
    return true;
  }
  return false;
}
DIRECT_LANGUAGE_ONE_SHOT_DEVICE bool language_one_shot_same_cue(
    const DirectLanguageOneShotSite& site,
    const DirectExactHistoryRecord& surface,
    const DirectExactHistoryRecord& ground) {
  return site.surface_channel == surface.subject &&
         site.surface_value == surface.value &&
         site.ground_channel == ground.subject &&
         site.ground_value == ground.value;
}

DIRECT_LANGUAGE_ONE_SHOT_DEVICE bool language_one_shot_same_motor(
    const DirectLanguageOneShotSite& site,
    const DirectExactHistoryRecord& motor) {
  return site.motor_node == motor.source &&
         site.motor_channel == motor.subject &&
         site.motor_value == motor.value;
}

DIRECT_LANGUAGE_ONE_SHOT_DEVICE void language_one_shot_admit(
    DirectLanguageOneShotState* state,
    const DirectExactHistoryRecord& surface,
    const DirectExactHistoryRecord& ground,
    const DirectExactHistoryRecord& motor,
    const DirectExactHistoryRecord& consequence) {
  for (std::uint32_t i = 0u; i < state->site_count; ++i) {
    DirectLanguageOneShotSite& site = state->sites[i];
    if (!language_one_shot_same_cue(site, surface, ground)) continue;
    ++state->revisions;
    if (language_one_shot_same_motor(site, motor) &&
        site.outcome_value == consequence.value) {
      ++site.support;
      if (site.active == 0u) {
        site.active = 1u;
        ++state->matter;
        ++state->reacquired_sites;
      }
      return;
    }
    ++site.contradictions;
    if (site.active != 0u) {
      site.active = 0u;
      --state->matter;
      ++state->retractions;
    }
    return;
  }
  if (state->site_count >= kLanguageOneShotCapacity) {
    ++state->refused_closures;
    return;
  }
  DirectLanguageOneShotSite& site = state->sites[state->site_count++];
  site.surface_channel = surface.subject;
  site.surface_value = surface.value;
  site.ground_channel = ground.subject;
  site.ground_value = ground.value;
  site.motor_node = motor.source;
  site.motor_channel = motor.subject;
  site.motor_value = motor.value;
  site.outcome_value = consequence.value;
  site.support = 1u;
  site.active = 1u;
  site.matter_identity = language_one_shot_fold(
      language_one_shot_fold(motor.identity, surface.identity),
      ground.identity);
  ++state->provisional_births;
  ++state->revisions;
  ++state->matter;
}

// Exact-history tickets are the only admission authority. Re-scanning pending
// motor records is intentional: closure may arrive in a later transport slice.
DIRECT_LANGUAGE_ONE_SHOT_DEVICE void language_one_shot_assimilate(
    DirectLanguageOneShotState* state,
    const DirectExactHistoryRecord* records, std::uint32_t count) {
  if (state == nullptr || records == nullptr || count < state->cursor) return;
  state->grounding = {};
  grounding_ingest_history(&state->grounding, records, count);
  grounding_extract_objects(&state->grounding);
  for (std::uint32_t i = state->cursor; i < count; ++i) {
    const DirectExactHistoryRecord& record = records[i];
    ++state->work;
    if (record.kind != DirectExactHistoryKind::sensory_contact) continue;
    ++state->q_contacts;
    state->source_hash = language_one_shot_fold(
        language_one_shot_fold(state->source_hash, record.identity),
        language_one_shot_fold(record.subject, record.value));
  }
  state->cursor = count;
  for (std::uint32_t i = 0u; i < count; ++i) {
    const DirectExactHistoryRecord& motor = records[i];
    if (motor.kind != DirectExactHistoryKind::motor_output ||
        language_one_shot_seen(*state, motor.identity))
      continue;
    DirectExactHistoryRecord consequence{};
    if (!language_one_shot_verified_return(records, count, motor,
                                           &consequence))
      continue;
    DirectExactHistoryRecord surface{};
    DirectExactHistoryRecord ground{};
    if (!language_one_shot_preceding_coalition(records, i, &surface, &ground)) {
      ++state->refused_closures;
      continue;
    }
    if (grounding_find_pair_canonical(&state->grounding, surface.subject,
                                      surface.value, ground.subject,
                                      ground.value) < 0) {
      ++state->refused_closures;
      continue;
    }
    if (state->admitted_motor_count >= kLanguageOneShotTicketCapacity) {
      ++state->refused_closures;
      continue;
    }
    state->admitted_motor_identities[state->admitted_motor_count++] =
        motor.identity;
    ++state->verified_closures;
    state->revision_identity =
        language_one_shot_fold(state->revision_identity, motor.identity);
    language_one_shot_admit(state, surface, ground, motor, consequence);
  }
}

DIRECT_LANGUAGE_ONE_SHOT_DEVICE DirectLanguageOneShotPlan language_one_shot_plan(
    DirectLanguageOneShotState* state,
    const DirectExactHistoryRecord* records, std::uint32_t begin,
    std::uint32_t count) {
  DirectLanguageOneShotPlan plan{};
  if (state == nullptr || records == nullptr || begin >= count) return plan;
  DirectExactHistoryRecord surface{};
  DirectExactHistoryRecord ground{};
  bool have_surface = false;
  bool have_ground = false;
  for (std::uint32_t i = begin; i < count; ++i) {
    if (records[i].kind != DirectExactHistoryKind::sensory_contact) continue;
    if (!have_surface) {
      surface = records[i];
      have_surface = true;
    } else if (records[i].subject != surface.subject) {
      ground = records[i];
      have_ground = true;
    }
  }
  plan.q = state->q_contacts;
  plan.matter = state->matter;
  plan.work = state->work;
  for (std::uint32_t i = 0u; i < state->site_count; ++i)
    plan.N += state->sites[i].active != 0u ? 1u : 0u;
  if (!have_surface || !have_ground ||
      ground.resident_tick < surface.resident_tick ||
      ground.resident_tick - surface.resident_tick >
          kLanguageOneShotMaximumGap)
    return plan;
  const DirectLanguageOneShotSite* best = nullptr;
  for (std::uint32_t i = 0u; i < state->site_count; ++i) {
    const DirectLanguageOneShotSite& site = state->sites[i];
    if (site.active == 0u ||
        !language_one_shot_same_cue(site, surface, ground))
      continue;
    if (best == nullptr || site.support > best->support) best = &site;
  }
  if (best == nullptr) return plan;
  plan.admitted = 1u;
  plan.motor_node = best->motor_node;
  plan.motor_channel = best->motor_channel;
  plan.motor_value = best->motor_value;
  plan.support = best->support;
  plan.provisional = best->support == 1u ? 1u : 0u;
  plan.p_next = language_one_shot_fold(
      language_one_shot_fold(state->revision_identity,
                             best->matter_identity),
      best->support);
  return plan;
}

DIRECT_LANGUAGE_ONE_SHOT_DEVICE bool language_one_shot_drive(
    const DirectLanguageOneShotPlan& plan, DirectNode* nodes,
    std::uint32_t node_count) {
  if (plan.admitted == 0u || nodes == nullptr ||
      plan.motor_node >= node_count)
    return false;
  atomicAdd(&nodes[plan.motor_node].activation_q16, 1 << 16);
  return true;
}

DIRECT_LANGUAGE_ONE_SHOT_DEVICE std::uint32_t language_one_shot_focal_lesion(
    DirectLanguageOneShotState* state) {
  if (state == nullptr) return 0u;
  std::uint32_t removed = 0u;
  for (std::uint32_t i = 0u; i < state->site_count; ++i) {
    if (state->sites[i].active == 0u) continue;
    state->sites[i].active = 0u;
    ++removed;
  }
  state->matter -= removed;
  state->lesion_events += removed != 0u ? 1u : 0u;
  return removed;
}

DIRECT_LANGUAGE_ONE_SHOT_DEVICE std::uint32_t language_one_shot_remote_sham(
    DirectLanguageOneShotState* state, std::uint32_t matter) {
  if (state == nullptr) return 0u;
  state->sham_matter += matter;
  return matter;
}

}  // namespace substrate::direct_network

// Compatibility consumers retain inline definitions until migrated.
#endif
