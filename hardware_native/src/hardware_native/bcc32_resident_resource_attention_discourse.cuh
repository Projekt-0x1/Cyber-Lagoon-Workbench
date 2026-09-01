#pragma once

#include "bcc32_resident_cross_contact_context.cuh"
#include "bcc32_resident_causal_constraint_participation.cuh"
#include "bcc32_resident_resource_guarded_action.cuh"

#include <cstdint>

// Resource-bounded canonical RWR0 matter; no plan or surface state.
namespace substrate::bcc32::resident_resource_attention_discourse {

namespace cross = causal_rewrite::cross_contact;
namespace guarded = resident_resource_guarded_action;
namespace participation = resident_causal_constraint_participation;
namespace pressure = resident_resource_pressure;
namespace rewrite = causal_rewrite;
namespace kernel = persistent_kernel;
#if defined(__CUDACC__)
#define BCC32_RESOURCE_ATTENTION_HD __host__ __device__
#else
#define BCC32_RESOURCE_ATTENTION_HD
#endif
enum class AttentionStatus : std::uint32_t {
  invalid = 0u,
  ambiguous = 1u,
  resource_withheld = 2u,
  prefix_reserved = 3u,
  full_context_reserved = 4u,
  publicly_committed = 5u,
  consequence_returned = 6u,
};

struct ResidentContextEvidence {
  std::uint32_t owner = rewrite::kInvalid;
  std::uint32_t word_extent = 0u;
  std::uint32_t record_extent = 0u;
  std::uint32_t support = 0u;
  std::uint32_t binding = 0u;
  std::uint32_t header_revision = 0u;
  std::uint32_t valid = 0u;
};
struct ResidentContextSelection {
  AttentionStatus status = AttentionStatus::invalid;
  ResidentContextEvidence selected{};
  std::uint32_t candidate_count = 0u;
  std::uint32_t frontier_count = 0u;
};
struct CanonicalContextAttentionView {
  kernel::ActionReturnTicket ticket{};
  std::uint32_t context_binding = 0u;
  std::uint32_t resident_word_extent = 0u;
  std::uint32_t word_begin = 0u;
  std::uint32_t retained_word_extent = 0u;
  std::uint32_t selected_attention_units = 0u;
};
struct CanonicalContextAttentionReceipt {
  kernel::ActionReturnTicket ticket{};
  std::uint32_t context_binding = 0u;
  std::uint32_t word_begin = 0u;
  std::uint32_t retained_word_extent = 0u;
  std::uint64_t public_sequence = 0u;
  std::uint32_t language_present = 0u;
};
struct ResidentContextAttentionState {
  guarded::GuardedAction charged{};
  std::uint32_t context_owner = rewrite::kInvalid;
  std::uint32_t context_binding = 0u;
  std::uint32_t resident_word_extent = 0u;
  std::uint32_t resident_record_extent = 0u;
  std::uint32_t word_begin = 0u;
  std::uint32_t requested_attention_units = 0u;
  std::uint32_t selected_attention_units = 0u;
  std::uint32_t retained_word_extent = 0u;
  std::uint32_t interruption_count = 0u;
  std::uint32_t turnover_count = 0u;
  std::uint64_t public_sequence = 0u;
  AttentionStatus status = AttentionStatus::invalid;
};
struct ResourceAttentionReceipt {
  AttentionStatus status = AttentionStatus::invalid;
  kernel::ActionReturnTicket ticket{};
  pressure::MatterSnapshot pre_resource{};
  pressure::MatterSnapshot post_resource{};
  std::uint32_t context_owner = rewrite::kInvalid;
  std::uint32_t context_binding = 0u;
  std::uint32_t resident_word_extent = 0u;
  std::uint32_t resident_record_extent = 0u;
  std::uint32_t word_begin = 0u;
  std::uint32_t requested_attention_units = 0u;
  std::uint32_t selected_attention_units = 0u;
  std::uint32_t retained_word_extent = 0u;
  std::uint32_t selected_cost_q8 = 0u;
  std::int32_t pre_margin_q8 = 0;
  std::int32_t post_margin_q8 = 0;
  std::uint64_t pre_revision = 0u;
  std::uint64_t post_revision = 0u;
};
struct ResourceAttentionReturnReceipt {
  kernel::ActionReturnTicket ticket{};
  pressure::MatterSnapshot pre_resource{};
  pressure::MatterSnapshot post_resource{};
  std::uint32_t context_owner = rewrite::kInvalid;
  std::uint32_t context_binding = 0u;
  std::uint32_t returned_q8 = 0u;
  std::uint32_t turnover_count = 0u;
  std::uint64_t pre_revision = 0u;
  std::uint64_t post_revision = 0u;
  AttentionStatus status = AttentionStatus::invalid;
};
BCC32_RESOURCE_ATTENTION_HD inline bool same_snapshot(
    const pressure::MatterSnapshot& left,
    const pressure::MatterSnapshot& right) {
  return left.free_q8 == right.free_q8 &&
         left.committed_q8 == right.committed_q8 &&
         left.damage_q8 == right.damage_q8 &&
         left.escrow_q8 == right.escrow_q8;
}
BCC32_RESOURCE_ATTENTION_HD inline std::int32_t margin_q8(
    const pressure::MatterSnapshot& matter) {
  return pressure::saturating_signed_margin_q8(
      matter.free_q8, matter.committed_q8, matter.damage_q8,
      matter.escrow_q8);
}
BCC32_RESOURCE_ATTENTION_HD inline std::uint32_t unique_owned_ordinal(
    const rewrite::ResidentRewriteState& world, std::uint32_t form,
    std::uint32_t owner, std::uint32_t ordinal) {
  std::uint32_t found = rewrite::kInvalid;
  for (std::uint32_t slot = 0u; slot < rewrite::live_record_capacity(&world);
       ++slot) {
    const rewrite::Record& record = world.records[slot];
    if (record.matter_q8 == 0u || record.lane[0] != form ||
        record.lane[1] != owner || record.lane[2] != ordinal)
      continue;
    if (found != rewrite::kInvalid) return rewrite::kInvalid;
    found = slot;
  }
  return found;
}
struct ResidentTrajectoryPageEvidence {
  std::uint32_t slot = rewrite::kInvalid;
  std::uint32_t term_owner = rewrite::kInvalid;
  std::uint32_t word_begin = 0u;
  std::uint32_t word_extent = 0u;
  std::uint32_t term_extent = 0u;
};
// Read a page directly; dormant headers are not current trajectories.
BCC32_RESOURCE_ATTENTION_HD inline bool read_trajectory_page_evidence(
    const rewrite::ResidentRewriteState& world,
    const rewrite::Record& header, std::uint32_t header_slot,
    std::uint32_t page, ResidentTrajectoryPageEvidence* evidence) {
  if (evidence == nullptr ||
      header_slot >= rewrite::live_record_capacity(&world) ||
      header.matter_q8 == 0u || header.lane[0] != rewrite::kFormTrajectory ||
      header.lane[1] == rewrite::kInvalid || header.lane[2] == 0u ||
      // page counts a kTrajectoryPageEvents-sized chunk within this one
      // trajectory's own word extent, not a Record slot -- kRecordCapacity
      // here is a coincidental reuse of the same constant as a generic
      // sanity ceiling (0X1-214 Class C), deliberately left unconverted.
      page >= rewrite::kRecordCapacity)
    return false;
  const std::uint32_t page_count =
      1u + (header.lane[2] - 1u) / rewrite::kTrajectoryPageEvents;
  if (page >= page_count) return false;
  ResidentTrajectoryPageEvidence result{};
  result.word_begin = page * rewrite::kTrajectoryPageEvents;
  const std::uint32_t remaining = header.lane[2] - result.word_begin;
  result.word_extent = remaining < rewrite::kTrajectoryPageEvents
                           ? remaining
                           : rewrite::kTrajectoryPageEvents;
  result.term_extent = (result.word_extent + 1u) / 2u;
  if (page == 0u) {
    result.slot = header_slot;
    result.term_owner = header.lane[1];
  } else {
    result.slot = unique_owned_ordinal(
        world, rewrite::kFormTrajectoryPage, header.lane[1], page);
    if (result.slot == rewrite::kInvalid) return false;
    const rewrite::Record& continuation = world.records[result.slot];
    if (continuation.lane[3] != result.word_begin ||
        continuation.lane[4] != result.word_extent ||
        continuation.lane[6] == 0u ||
        continuation.lane[6] == rewrite::kInvalid ||
        continuation.lane[6] == header.lane[1])
      return false;
    result.term_owner = continuation.lane[6];
    for (std::uint32_t prior = 1u; prior < page; ++prior) {
      const std::uint32_t prior_slot = unique_owned_ordinal(
          world, rewrite::kFormTrajectoryPage, header.lane[1], prior);
      if (prior_slot == rewrite::kInvalid ||
          world.records[prior_slot].lane[6] == result.term_owner)
        return false;
    }
  }
  for (std::uint32_t ordinal = 0u; ordinal < result.term_extent; ++ordinal)
    if (unique_owned_ordinal(world, rewrite::kFormTrajectoryTerm,
                             result.term_owner, ordinal) == rewrite::kInvalid)
      return false;
  for (std::uint32_t slot = 0u; slot < rewrite::live_record_capacity(&world);
       ++slot) {
    const rewrite::Record& record = world.records[slot];
    if (record.matter_q8 != 0u &&
        record.lane[0] == rewrite::kFormTrajectoryTerm &&
        record.lane[1] == result.term_owner &&
        record.lane[2] >= result.term_extent)
      return false;
  }
  *evidence = result;
  return true;
}
BCC32_RESOURCE_ATTENTION_HD inline bool paged_trajectory_word_at(
    const rewrite::ResidentRewriteState& world,
    const rewrite::Record& header, std::uint32_t header_slot,
    std::uint32_t index, std::uint32_t* word) {
  if (word == nullptr || index >= header.lane[2]) return false;
  ResidentTrajectoryPageEvidence page{};
  if (!read_trajectory_page_evidence(
          world, header, header_slot,
          index / rewrite::kTrajectoryPageEvents, &page))
    return false;
  const std::uint32_t local = index - page.word_begin;
  const std::uint32_t term_slot = unique_owned_ordinal(
      world, rewrite::kFormTrajectoryTerm, page.term_owner, local / 2u);
  if (term_slot == rewrite::kInvalid) return false;
  *word = world.records[term_slot].lane[4u + (local % 2u)];
  return true;
}

// The page descriptor already proves the page envelope and its term extent.
// Fold its resident term blocks directly instead of asking the generic word
// reader to rediscover the same page and owner for every word. This is a
// physical lookup optimization only: the unique term-owner check and the
// page/header digest checks remain unchanged, and no semantic suffix is
// cached or capped here.
BCC32_RESOURCE_ATTENTION_HD inline bool fold_trajectory_page_evidence(
    const rewrite::ResidentRewriteState& world,
    const ResidentTrajectoryPageEvidence& page, std::uint32_t* rolling,
    std::uint32_t* binding, std::uint32_t* page_digest) {
  if (rolling == nullptr || binding == nullptr || page_digest == nullptr ||
      page.word_extent == 0u || page.term_extent == 0u)
    return false;
  *page_digest = 0u;
  for (std::uint32_t ordinal = 0u; ordinal < page.term_extent; ++ordinal) {
    const std::uint32_t term_slot = unique_owned_ordinal(
        world, rewrite::kFormTrajectoryTerm, page.term_owner, ordinal);
    if (term_slot == rewrite::kInvalid) return false;
    const rewrite::Record& term = world.records[term_slot];
    for (std::uint32_t offset = 0u; offset < 2u; ++offset) {
      const std::uint32_t local = ordinal * 2u + offset;
      if (local >= page.word_extent) break;
      const std::uint32_t index = page.word_begin + local;
      const std::uint32_t word = term.lane[4u + offset];
      *rolling = rewrite::rewrite_mix(*rolling, word, index);
      *binding = rewrite::rewrite_mix(*binding, word, index);
      *page_digest = rewrite::rewrite_mix(*page_digest, word, index);
    }
  }
  return true;
}
// Count exact page envelopes and term Records, never a fixed reserve.
BCC32_RESOURCE_ATTENTION_HD inline bool trajectory_physical_extent(
    const rewrite::ResidentRewriteState& world,
    const rewrite::Record& header, std::uint32_t header_slot,
    std::uint32_t* extent) {
  if (extent == nullptr || header.lane[2] == 0u) return false;
  const std::uint32_t pages =
      1u + (header.lane[2] - 1u) / rewrite::kTrajectoryPageEvents;
  // pages counts trajectory page envelopes, not Record slots -- same
  // coincidental kRecordCapacity reuse as read_trajectory_page_evidence
  // above (0X1-214 Class C), deliberately left unconverted.
  if (pages > rewrite::kRecordCapacity) return false;
  std::uint32_t records = 0u;
  for (std::uint32_t page_index = 0u; page_index < pages; ++page_index) {
    ResidentTrajectoryPageEvidence page{};
    if (!read_trajectory_page_evidence(world, header, header_slot,
                                       page_index, &page) ||
        records > rewrite::kRecordCapacity - 1u - page.term_extent)
      return false;
    records += 1u + page.term_extent;
  }
  // Undeclared page descriptors are malformed resident matter.
  for (std::uint32_t slot = 0u; slot < rewrite::live_record_capacity(&world);
       ++slot) {
    const rewrite::Record& record = world.records[slot];
    if (record.matter_q8 != 0u &&
        record.lane[0] == rewrite::kFormTrajectoryPage &&
        record.lane[1] == header.lane[1] &&
        (record.lane[2] == 0u || record.lane[2] >= pages))
      return false;
  }
  *extent = records;
  return true;
}

// Charge each touched page envelope and distinct term Record.
BCC32_RESOURCE_ATTENTION_HD inline bool trajectory_span_record_demand(
    const rewrite::ResidentRewriteState& world,
    const rewrite::Record& header, std::uint32_t header_slot,
    std::uint32_t word_begin, std::uint32_t word_extent,
    std::uint32_t* demand) {
  if (demand == nullptr || word_extent == 0u ||
      word_begin >= header.lane[2] ||
      word_extent > header.lane[2] - word_begin)
    return false;
  const std::uint32_t last_word = word_begin + word_extent - 1u;
  const std::uint32_t first_page = word_begin / rewrite::kTrajectoryPageEvents;
  const std::uint32_t last_page = last_word / rewrite::kTrajectoryPageEvents;
  std::uint32_t records = 0u;
  for (std::uint32_t page_index = first_page; page_index <= last_page;
       ++page_index) {
    ResidentTrajectoryPageEvidence page{};
    if (!read_trajectory_page_evidence(world, header, header_slot,
                                       page_index, &page))
      return false;
    const std::uint32_t begin = word_begin > page.word_begin
                                    ? word_begin - page.word_begin
                                    : 0u;
    const std::uint32_t absolute_end = last_word + 1u;
    const std::uint32_t page_end = page.word_begin + page.word_extent;
    const std::uint32_t end =
        absolute_end < page_end ? absolute_end - page.word_begin
                                : page.word_extent;
    const std::uint32_t first_term = begin / 2u;
    const std::uint32_t last_term = (end - 1u) / 2u;
    const std::uint32_t terms = last_term - first_term + 1u;
    if (records > pressure::kMaximumChargedExtent - 1u ||
        terms > pressure::kMaximumChargedExtent - records - 1u)
      return false;
    records += 1u + terms;
  }
  *demand = records;
  return true;
}

BCC32_RESOURCE_ATTENTION_HD inline bool words_for_record_demand(
    const rewrite::ResidentRewriteState& world,
    const rewrite::Record& header, std::uint32_t header_slot,
    std::uint32_t word_begin, std::uint32_t maximum_records,
    std::uint32_t* words, std::uint32_t* exact_records) {
  if (words == nullptr || exact_records == nullptr || maximum_records < 2u ||
      maximum_records > pressure::kMaximumChargedExtent ||
      word_begin >= header.lane[2])
    return false;
  std::uint32_t retained = 0u;
  std::uint32_t demand = 0u;
  const std::uint32_t remaining = header.lane[2] - word_begin;
  for (std::uint32_t candidate = 1u; candidate <= remaining; ++candidate) {
    std::uint32_t candidate_demand = 0u;
    if (!trajectory_span_record_demand(world, header, header_slot, word_begin,
                                       candidate, &candidate_demand) ||
        candidate_demand > maximum_records)
      break;
    retained = candidate;
    demand = candidate_demand;
  }
  if (retained == 0u || demand < 2u) return false;
  *words = retained;
  *exact_records = demand;
  return true;
}

// Hash resident content and structure, excluding allocation loci.
BCC32_RESOURCE_ATTENTION_HD inline bool read_context_evidence(
    rewrite::ResidentRewriteState* world, std::uint32_t header_slot,
    ResidentContextEvidence* evidence) {
  if (world == nullptr || evidence == nullptr ||
      header_slot >= rewrite::live_record_capacity(world) ||
      !cross::is_dormant_history(world->records[header_slot]))
    return false;
  const rewrite::Record& header = world->records[header_slot];
  std::uint32_t records = 0u;
  if (!trajectory_physical_extent(*world, header, header_slot, &records))
    return false;
  // No single Record or Program is sufficient.
  if (header.lane[2] < 3u || records < 3u)
    return false;
  std::uint32_t support = 0u;
  // Longer paged trajectories borrow no bounded retention support.
  if (header.lane[2] <= rewrite::kMaximumTrajectoryEvents) {
    cross::DormantRetentionEvidence retained{};
    if (!cross::read_dormant_retention_evidence(world, header_slot, &retained))
      return false;
    support = retained.support;
  }
  std::uint32_t binding = rewrite::rewrite_mix(
      header.lane[2], records, support ^ header.lane[6]);
  std::uint32_t rolling = 0u;
  const std::uint32_t pages =
      1u + (header.lane[2] - 1u) / rewrite::kTrajectoryPageEvents;
  for (std::uint32_t page_index = 0u; page_index < pages; ++page_index) {
    ResidentTrajectoryPageEvidence page{};
    if (!read_trajectory_page_evidence(*world, header, header_slot,
                                       page_index, &page))
      return false;
    std::uint32_t page_digest = 0u;
    if (!fold_trajectory_page_evidence(*world, page, &rolling, &binding,
                                       &page_digest))
      return false;
    if (page_index != 0u &&
        world->records[page.slot].lane[5] != page_digest)
      return false;
  }
  if (header.lane[6] != rolling) return false;
  ResidentContextEvidence candidate{};
  candidate.owner = header.lane[1];
  candidate.word_extent = header.lane[2];
  candidate.record_extent = records;
  candidate.support = support;
  candidate.binding = binding == 0u ? 1u : binding;
  candidate.header_revision = header.revision;
  candidate.valid = 1u;
  *evidence = candidate;
  return true;
}

BCC32_RESOURCE_ATTENTION_HD inline bool context_better(
    const ResidentContextEvidence& candidate,
    const ResidentContextEvidence& incumbent) {
  if (candidate.support != incumbent.support)
    return candidate.support > incumbent.support;
  if (candidate.word_extent != incumbent.word_extent)
    return candidate.word_extent > incumbent.word_extent;
  return candidate.record_extent < incumbent.record_extent;
}

// Equal but distinct resident content abstains before public callback.
BCC32_RESOURCE_ATTENTION_HD inline ResidentContextSelection
select_resident_context(rewrite::ResidentRewriteState* world) {
  ResidentContextSelection selection{};
  if (world == nullptr || world->fault != 0u) return selection;
  for (std::uint32_t slot = 0u; slot < rewrite::live_record_capacity(world);
       ++slot) {
    if (!cross::is_dormant_history(world->records[slot])) continue;
    ResidentContextEvidence candidate{};
    if (!read_context_evidence(world, slot, &candidate)) {
      selection.status = AttentionStatus::invalid;
      return selection;
    }
    ++selection.candidate_count;
    if (selection.frontier_count == 0u ||
        context_better(candidate, selection.selected)) {
      selection.selected = candidate;
      selection.frontier_count = 1u;
    } else if (!context_better(selection.selected, candidate) &&
               candidate.binding != selection.selected.binding) {
      ++selection.frontier_count;
    }
  }
  if (selection.candidate_count == 0u) return selection;
  if (selection.frontier_count != 1u) {
    selection.status = AttentionStatus::ambiguous;
    return selection;
  }
  selection.status = AttentionStatus::full_context_reserved;
  return selection;
}

BCC32_RESOURCE_ATTENTION_HD inline bool find_bound_context(
    rewrite::ResidentRewriteState* world,
    const ResidentContextAttentionState& attention,
    ResidentContextEvidence* evidence) {
  if (world == nullptr || evidence == nullptr ||
      attention.context_owner == rewrite::kInvalid ||
      attention.context_binding == 0u)
    return false;
  std::uint32_t matches = 0u;
  for (std::uint32_t slot = 0u; slot < rewrite::live_record_capacity(world);
       ++slot) {
    if (!cross::is_dormant_history(world->records[slot]) ||
        world->records[slot].lane[1] != attention.context_owner)
      continue;
    ResidentContextEvidence candidate{};
    if (!read_context_evidence(world, slot, &candidate) ||
        candidate.binding != attention.context_binding ||
        candidate.word_extent != attention.resident_word_extent ||
        candidate.record_extent != attention.resident_record_extent)
      return false;
    *evidence = candidate;
    ++matches;
  }
  return matches == 1u;
}

BCC32_RESOURCE_ATTENTION_HD inline std::uint32_t bound_context_header_slot(
    rewrite::ResidentRewriteState* world,
    const ResidentContextEvidence& evidence) {
  if (world == nullptr || evidence.valid != 1u) return rewrite::kInvalid;
  std::uint32_t found = rewrite::kInvalid;
  for (std::uint32_t slot = 0u; slot < rewrite::live_record_capacity(world);
       ++slot) {
    if (!cross::is_dormant_history(world->records[slot])) continue;
    ResidentContextEvidence candidate{};
    if (!read_context_evidence(world, slot, &candidate))
      return rewrite::kInvalid;
    if (candidate.owner != evidence.owner ||
        candidate.binding != evidence.binding ||
        candidate.word_extent != evidence.word_extent ||
        candidate.record_extent != evidence.record_extent)
      continue;
    if (found != rewrite::kInvalid) return rewrite::kInvalid;
    found = slot;
  }
  return found;
}

BCC32_RESOURCE_ATTENTION_HD inline bool charge_is_exact(
    const rewrite::ResidentRewriteState& world,
    const ResidentContextAttentionState& attention) {
  if ((attention.status != AttentionStatus::prefix_reserved &&
       attention.status != AttentionStatus::full_context_reserved &&
       attention.status != AttentionStatus::publicly_committed) ||
      attention.selected_attention_units < 2u ||
      attention.resident_record_extent < 3u ||
      attention.requested_attention_units < 2u ||
      attention.requested_attention_units > pressure::kMaximumChargedExtent ||
      attention.selected_attention_units > attention.requested_attention_units ||
      attention.word_begin >= attention.resident_word_extent ||
      attention.selected_attention_units !=
          attention.charged.payment.selected_extent ||
      attention.charged.payment.owner != attention.context_owner ||
      attention.charged.payment.purpose !=
          pressure::PaymentPurpose::action_proposal ||
      attention.charged.payment.charged_q8 !=
          attention.selected_attention_units * rewrite::kRecordMatterQ8 ||
      attention.charged.payment.selected_extent >
          pressure::kMaximumChargedExtent)
    return false;
  rewrite::ResidentRewriteState* mutable_world =
      const_cast<rewrite::ResidentRewriteState*>(&world);
  ResidentContextEvidence evidence{};
  evidence.owner = attention.context_owner;
  evidence.binding = attention.context_binding;
  evidence.word_extent = attention.resident_word_extent;
  evidence.record_extent = attention.resident_record_extent;
  evidence.valid = 1u;
  const std::uint32_t header_slot =
      bound_context_header_slot(mutable_world, evidence);
  if (header_slot == rewrite::kInvalid) return false;
  std::uint32_t exact_demand = 0u;
  if (!trajectory_span_record_demand(
          world, world.records[header_slot], header_slot,
          attention.word_begin, attention.retained_word_extent,
          &exact_demand) ||
      exact_demand != attention.selected_attention_units)
    return false;
  std::uint32_t maximum_words = 0u;
  std::uint32_t maximum_demand = 0u;
  if (!words_for_record_demand(
          world, world.records[header_slot], header_slot,
          attention.word_begin, pressure::kMaximumChargedExtent,
          &maximum_words, &maximum_demand) ||
      maximum_words == 0u ||
      attention.requested_attention_units != maximum_demand)
    return false;
  for (std::uint32_t index = 0u;
       index < attention.charged.payment.selected_extent; ++index) {
    const std::uint32_t slot = attention.charged.payment.slots[index];
    if (slot >= rewrite::live_record_capacity(mutable_world) ||
        !pressure::live_owned_form(world.records[slot],
                                   pressure::kFormCommittedMatter,
                                   attention.context_owner) ||
        world.records[slot].matter_q8 != rewrite::kRecordMatterQ8 ||
        world.records[slot].reserved[0] !=
            guarded::ticket_sequence_low(attention.charged.ticket) ||
        world.records[slot].reserved[1] !=
            guarded::ticket_sequence_high(attention.charged.ticket))
      return false;
    for (std::uint32_t prior = 0u; prior < index; ++prior)
      if (attention.charged.payment.slots[prior] == slot) return false;
  }
  return true;
}

// Charge the largest affordable set of distinct represented Records.
BCC32_RESOURCE_ATTENTION_HD inline bool snapshot_affordable_trajectory_span(
    rewrite::ResidentRewriteState* world,
    const kernel::ActionReturnTicket& ticket, std::uint32_t owner,
    std::uint32_t header_slot, std::uint32_t word_begin,
    std::uint32_t requested_records, guarded::GuardedAction* pending,
    std::uint32_t* selected_records, std::uint32_t* retained_words) {
  if (world == nullptr || pending == nullptr || selected_records == nullptr ||
      retained_words == nullptr ||
      header_slot >= rewrite::live_record_capacity(world) ||
      requested_records < 2u ||
      requested_records > pressure::kMaximumChargedExtent)
    return false;
  std::uint32_t previous_demand = 0u;
  for (std::uint32_t budget = requested_records; budget >= 2u; --budget) {
    std::uint32_t words = 0u;
    std::uint32_t demand = 0u;
    if (!words_for_record_demand(
            *world, world->records[header_slot], header_slot, word_begin,
            budget, &words, &demand))
      continue;
    if (demand == previous_demand) continue;
    previous_demand = demand;
    guarded::GuardedAction candidate{};
    if (!guarded::snapshot_before_action_commit(
            *world, ticket, owner, demand,
            pressure::PaymentPurpose::action_proposal, &candidate))
      continue;
    *pending = candidate;
    *selected_records = demand;
    *retained_words = words;
    return true;
  }
  return false;
}

// Context owner is resource owner; fewer than two units abstains.
BCC32_RESOURCE_ATTENTION_HD inline AttentionStatus reserve_context_attention(
    rewrite::ResidentRewriteState* world,
    const kernel::ActionReturnTicket& ticket,
    ResidentContextAttentionState* attention,
    ResourceAttentionReceipt* receipt) {
  if (world == nullptr || attention == nullptr || receipt == nullptr ||
      !guarded::valid_ticket(ticket) ||
      guarded::ticket_is_bound_or_consumed(*world, ticket))
    return AttentionStatus::invalid;
  const ResidentContextSelection selected = select_resident_context(world);
  if (selected.status == AttentionStatus::invalid ||
      selected.status == AttentionStatus::ambiguous)
    return selected.status;

  ResourceAttentionReceipt result{};
  result.ticket = ticket;
  result.context_owner = selected.selected.owner;
  result.context_binding = selected.selected.binding;
  result.resident_word_extent = selected.selected.word_extent;
  result.resident_record_extent = selected.selected.record_extent;
  const std::uint32_t header_slot =
      bound_context_header_slot(world, selected.selected);
  std::uint32_t maximum_words = 0u;
  if (header_slot == rewrite::kInvalid ||
      !words_for_record_demand(
          *world, world->records[header_slot], header_slot, 0u,
          pressure::kMaximumChargedExtent, &maximum_words,
          &result.requested_attention_units) ||
      maximum_words == 0u)
    return AttentionStatus::invalid;
  result.pre_resource = pressure::snapshot(*world, result.context_owner);
  result.pre_margin_q8 = margin_q8(result.pre_resource);
  result.pre_revision = world->revision;

  guarded::GuardedAction pending{};
  std::uint32_t extent = 0u;
  std::uint32_t retained_words = 0u;
  if (!snapshot_affordable_trajectory_span(
          world, ticket, result.context_owner, header_slot, 0u,
          result.requested_attention_units, &pending, &extent,
          &retained_words)) {
    result.status = AttentionStatus::resource_withheld;
    result.post_resource = result.pre_resource;
    result.post_margin_q8 = result.pre_margin_q8;
    result.post_revision = result.pre_revision;
    *receipt = result;
    return result.status;
  }

  guarded::GuardedAction charged{};
  if (!guarded::charge_before_action_commit(world, pending, &charged))
    return AttentionStatus::invalid;
  ResidentContextAttentionState candidate{};
  candidate.charged = charged;
  candidate.context_owner = result.context_owner;
  candidate.context_binding = result.context_binding;
  candidate.resident_word_extent = result.resident_word_extent;
  candidate.resident_record_extent = result.resident_record_extent;
  candidate.word_begin = 0u;
  candidate.requested_attention_units = result.requested_attention_units;
  candidate.selected_attention_units = extent;
  candidate.retained_word_extent = retained_words;
  candidate.status = candidate.retained_word_extent ==
                             candidate.resident_word_extent
                         ? AttentionStatus::full_context_reserved
                         : AttentionStatus::prefix_reserved;
  result.status = candidate.status;
  result.selected_attention_units = extent;
  result.retained_word_extent = candidate.retained_word_extent;
  result.selected_cost_q8 = charged.payment.charged_q8;
  result.post_resource = pressure::snapshot(*world, result.context_owner);
  result.post_margin_q8 = margin_q8(result.post_resource);
  result.post_revision = world->revision;
  *attention = candidate;
  *receipt = result;
  return result.status;
}

// Continue only after the prior prefix's accepted raw return.
BCC32_RESOURCE_ATTENTION_HD inline AttentionStatus
continue_context_attention(
    rewrite::ResidentRewriteState* world,
    const kernel::ActionReturnTicket& ticket,
    const ResidentContextAttentionState& prior,
    ResidentContextAttentionState* attention,
    ResourceAttentionReceipt* receipt) {
  ResidentContextEvidence context{};
  if (world == nullptr || attention == nullptr || receipt == nullptr ||
      prior.status != AttentionStatus::consequence_returned ||
      !guarded::valid_ticket(ticket) ||
      guarded::ticket_is_bound_or_consumed(*world, ticket) ||
      !find_bound_context(world, prior, &context) ||
      prior.turnover_count == 0xffffffffu ||
      prior.word_begin > prior.resident_word_extent ||
      prior.retained_word_extent >
          prior.resident_word_extent - prior.word_begin)
    return AttentionStatus::invalid;
  const std::uint32_t word_begin =
      prior.word_begin + prior.retained_word_extent;
  if (word_begin >= prior.resident_word_extent) return AttentionStatus::invalid;
  const std::uint32_t header_slot = bound_context_header_slot(world, context);
  std::uint32_t maximum_words = 0u;
  std::uint32_t requested = 0u;
  if (header_slot == rewrite::kInvalid ||
      !words_for_record_demand(
          *world, world->records[header_slot], header_slot, word_begin,
          pressure::kMaximumChargedExtent, &maximum_words, &requested) ||
      maximum_words == 0u)
    return AttentionStatus::invalid;

  ResourceAttentionReceipt result{};
  result.ticket = ticket;
  result.context_owner = prior.context_owner;
  result.context_binding = prior.context_binding;
  result.resident_word_extent = prior.resident_word_extent;
  result.resident_record_extent = prior.resident_record_extent;
  result.word_begin = word_begin;
  result.requested_attention_units = requested;
  result.pre_resource = pressure::snapshot(*world, prior.context_owner);
  result.pre_margin_q8 = margin_q8(result.pre_resource);
  result.pre_revision = world->revision;

  guarded::GuardedAction pending{};
  std::uint32_t extent = 0u;
  std::uint32_t retained_words = 0u;
  if (!snapshot_affordable_trajectory_span(
          world, ticket, prior.context_owner, header_slot, word_begin,
          requested, &pending, &extent, &retained_words)) {
    result.status = AttentionStatus::resource_withheld;
    result.post_resource = result.pre_resource;
    result.post_margin_q8 = result.pre_margin_q8;
    result.post_revision = result.pre_revision;
    *receipt = result;
    return result.status;
  }

  guarded::GuardedAction charged{};
  if (!guarded::charge_before_action_commit(world, pending, &charged))
    return AttentionStatus::invalid;
  ResidentContextAttentionState candidate{};
  candidate.charged = charged;
  candidate.context_owner = prior.context_owner;
  candidate.context_binding = prior.context_binding;
  candidate.resident_word_extent = prior.resident_word_extent;
  candidate.resident_record_extent = prior.resident_record_extent;
  candidate.word_begin = word_begin;
  candidate.requested_attention_units = requested;
  candidate.selected_attention_units = extent;
  candidate.retained_word_extent = retained_words;
  candidate.interruption_count = prior.interruption_count;
  candidate.turnover_count = prior.turnover_count + 1u;
  candidate.status = word_begin + candidate.retained_word_extent ==
                             candidate.resident_word_extent
                         ? AttentionStatus::full_context_reserved
                         : AttentionStatus::prefix_reserved;
  result.status = candidate.status;
  result.selected_attention_units = extent;
  result.retained_word_extent = candidate.retained_word_extent;
  result.selected_cost_q8 = charged.payment.charged_q8;
  result.post_resource = pressure::snapshot(*world, prior.context_owner);
  result.post_margin_q8 = margin_q8(result.post_resource);
  result.post_revision = world->revision;
  *attention = candidate;
  *receipt = result;
  return result.status;
}

BCC32_RESOURCE_ATTENTION_HD inline bool exact_attention_receipt(
    const rewrite::ResidentRewriteState& world,
    const ResidentContextAttentionState* attention,
    const ResourceAttentionReceipt& receipt) {
  if (!guarded::valid_ticket(receipt.ticket) ||
      receipt.context_owner == rewrite::kInvalid ||
      receipt.context_binding == 0u ||
      receipt.resident_record_extent < 3u ||
      receipt.resident_word_extent < 3u ||
      receipt.word_begin >= receipt.resident_word_extent ||
      receipt.requested_attention_units < 2u ||
      receipt.requested_attention_units > pressure::kMaximumChargedExtent ||
      receipt.pre_margin_q8 != margin_q8(receipt.pre_resource) ||
      receipt.post_margin_q8 != margin_q8(receipt.post_resource) ||
      receipt.post_revision != world.revision ||
      !same_snapshot(receipt.post_resource,
                     pressure::snapshot(world, receipt.context_owner)))
    return false;
  if (receipt.status == AttentionStatus::resource_withheld)
    return attention == nullptr && receipt.selected_attention_units == 0u &&
           receipt.retained_word_extent == 0u && receipt.selected_cost_q8 == 0u &&
           receipt.pre_revision == receipt.post_revision &&
           same_snapshot(receipt.pre_resource, receipt.post_resource);
  if (attention == nullptr || receipt.status != attention->status ||
      (receipt.status != AttentionStatus::prefix_reserved &&
       receipt.status != AttentionStatus::full_context_reserved) ||
      !guarded::same_ticket(receipt.ticket, attention->charged.ticket) ||
      receipt.context_owner != attention->context_owner ||
      receipt.context_binding != attention->context_binding ||
      receipt.resident_word_extent != attention->resident_word_extent ||
      receipt.resident_record_extent != attention->resident_record_extent ||
      receipt.word_begin != attention->word_begin ||
      receipt.requested_attention_units !=
          attention->requested_attention_units ||
      receipt.selected_attention_units !=
          attention->selected_attention_units ||
      receipt.retained_word_extent != attention->retained_word_extent ||
      receipt.selected_cost_q8 != attention->charged.payment.charged_q8 ||
      receipt.pre_revision != attention->charged.payment.pre_commit_revision ||
      receipt.post_revision != attention->charged.payment.commit_revision ||
      !charge_is_exact(world, *attention))
    return false;
  const bool complete = receipt.word_begin + receipt.retained_word_extent ==
                        receipt.resident_word_extent;
  if ((receipt.status == AttentionStatus::full_context_reserved) != complete)
    return false;
  return receipt.post_resource.free_q8 + receipt.selected_cost_q8 ==
             receipt.pre_resource.free_q8 &&
         receipt.post_resource.committed_q8 ==
             receipt.pre_resource.committed_q8 + receipt.selected_cost_q8 &&
         receipt.post_resource.damage_q8 == receipt.pre_resource.damage_q8 &&
         receipt.post_resource.escrow_q8 == receipt.pre_resource.escrow_q8;
}

BCC32_RESOURCE_ATTENTION_HD inline bool record_interruption(
    rewrite::ResidentRewriteState* world,
    ResidentContextAttentionState* attention) {
  ResidentContextEvidence context{};
  if (attention == nullptr || attention->interruption_count == 0xffffffffu ||
      !find_bound_context(world, *attention, &context) ||
      !charge_is_exact(*world, *attention))
    return false;
  ++attention->interruption_count;
  return true;
}

BCC32_RESOURCE_ATTENTION_HD inline CanonicalContextAttentionView
canonical_view(const ResidentContextAttentionState& attention) {
  CanonicalContextAttentionView view{};
  view.ticket = attention.charged.ticket;
  view.context_binding = attention.context_binding;
  view.resident_word_extent = attention.resident_word_extent;
  view.word_begin = attention.word_begin;
  view.retained_word_extent = attention.retained_word_extent;
  view.selected_attention_units = attention.selected_attention_units;
  return view;
}

BCC32_RESOURCE_ATTENTION_HD inline bool exact_public_receipt(
    const CanonicalContextAttentionReceipt& receipt,
    const CanonicalContextAttentionView& view) {
  return receipt.language_present == 1u && receipt.public_sequence != 0u &&
         guarded::same_ticket(receipt.ticket, view.ticket) &&
         receipt.context_binding == view.context_binding &&
         receipt.word_begin == view.word_begin &&
         receipt.retained_word_extent == view.retained_word_extent;
}

template <typename CanonicalCommit>
inline bool commit_public_context(
    rewrite::ResidentRewriteState* world,
    ResidentContextAttentionState* attention,
    CanonicalCommit&& canonical_commit,
    CanonicalContextAttentionReceipt* receipt) {
  ResidentContextEvidence context{};
  if (world == nullptr || attention == nullptr || receipt == nullptr ||
      (attention->status != AttentionStatus::prefix_reserved &&
       attention->status != AttentionStatus::full_context_reserved) ||
      !find_bound_context(world, *attention, &context) ||
      !charge_is_exact(*world, *attention))
    return false;
  const CanonicalContextAttentionView view = canonical_view(*attention);
  CanonicalContextAttentionReceipt published{};
  if (!canonical_commit(view, &published) ||
      !exact_public_receipt(published, view))
    return false;
  attention->public_sequence = published.public_sequence;
  attention->status = AttentionStatus::publicly_committed;
  *receipt = published;
  return true;
}

BCC32_RESOURCE_ATTENTION_HD inline bool return_after_raw_consequence(
    rewrite::ResidentRewriteState* world,
    ResidentContextAttentionState* attention,
    const CanonicalContextAttentionReceipt& publication,
    const guarded::GuardedActionConsequence& consequence) {
  if (world == nullptr || attention == nullptr ||
      attention->status != AttentionStatus::publicly_committed ||
      publication.public_sequence != attention->public_sequence ||
      !exact_public_receipt(publication, canonical_view(*attention)) ||
      !charge_is_exact(*world, *attention) ||
      !guarded::refund_after_ticketed_consequence(
          world, attention->charged, consequence))
    return false;
  attention->status = AttentionStatus::consequence_returned;
  return true;
}

// Exact returned Records, never basal credit, restore affordability.
BCC32_RESOURCE_ATTENTION_HD inline bool
return_after_raw_consequence_with_receipt(
    rewrite::ResidentRewriteState* world,
    ResidentContextAttentionState* attention,
    const CanonicalContextAttentionReceipt& publication,
    const guarded::GuardedActionConsequence& consequence,
    ResourceAttentionReturnReceipt* receipt) {
  if (world == nullptr || attention == nullptr || receipt == nullptr)
    return false;
  ResourceAttentionReturnReceipt result{};
  result.ticket = attention->charged.ticket;
  result.context_owner = attention->context_owner;
  result.context_binding = attention->context_binding;
  result.returned_q8 = attention->charged.payment.charged_q8;
  result.turnover_count = attention->turnover_count;
  result.pre_resource = pressure::snapshot(*world, attention->context_owner);
  result.pre_revision = world->revision;
  if (!return_after_raw_consequence(world, attention, publication,
                                    consequence))
    return false;
  result.post_resource = pressure::snapshot(*world, result.context_owner);
  result.post_revision = world->revision;
  result.status = AttentionStatus::consequence_returned;
  *receipt = result;
  return true;
}

BCC32_RESOURCE_ATTENTION_HD inline bool exact_resource_attention_return(
    const rewrite::ResidentRewriteState& world,
    const ResidentContextAttentionState& attention,
    const ResourceAttentionReturnReceipt& receipt) {
  const std::uint64_t expected_revision =
      receipt.pre_revision == ~std::uint64_t{0}
          ? receipt.pre_revision
          : receipt.pre_revision + 1u;
  return receipt.status == AttentionStatus::consequence_returned &&
         attention.status == AttentionStatus::consequence_returned &&
         guarded::same_ticket(receipt.ticket, attention.charged.ticket) &&
         receipt.context_owner == attention.context_owner &&
         receipt.context_binding == attention.context_binding &&
         receipt.turnover_count == attention.turnover_count &&
         receipt.returned_q8 == attention.charged.payment.charged_q8 &&
         receipt.post_revision == world.revision &&
         receipt.post_revision == expected_revision &&
         same_snapshot(receipt.post_resource,
                       pressure::snapshot(world, receipt.context_owner)) &&
         receipt.pre_resource.free_q8 + receipt.returned_q8 ==
             receipt.post_resource.free_q8 &&
         receipt.pre_resource.committed_q8 ==
             receipt.post_resource.committed_q8 + receipt.returned_q8 &&
         receipt.pre_resource.damage_q8 == receipt.post_resource.damage_q8 &&
         receipt.pre_resource.escrow_q8 == receipt.post_resource.escrow_q8;
}

// Reacquisition preserves identity and pays newly touched Records.
BCC32_RESOURCE_ATTENTION_HD inline bool exact_turnover_reacquisition(
    const rewrite::ResidentRewriteState& world,
    const ResidentContextAttentionState& returned,
    const ResidentContextAttentionState& reacquired,
    const ResourceAttentionReceipt& receipt) {
  return returned.status == AttentionStatus::consequence_returned &&
         (reacquired.status == AttentionStatus::prefix_reserved ||
          reacquired.status == AttentionStatus::full_context_reserved) &&
         returned.context_owner == reacquired.context_owner &&
         returned.context_binding == reacquired.context_binding &&
         returned.resident_word_extent == reacquired.resident_word_extent &&
         returned.resident_record_extent == reacquired.resident_record_extent &&
         returned.word_begin <= returned.resident_word_extent &&
         returned.retained_word_extent <=
             returned.resident_word_extent - returned.word_begin &&
         reacquired.word_begin ==
             returned.word_begin + returned.retained_word_extent &&
         returned.turnover_count != 0xffffffffu &&
         reacquired.turnover_count == returned.turnover_count + 1u &&
         reacquired.interruption_count == returned.interruption_count &&
         !guarded::same_ticket(returned.charged.ticket,
                               reacquired.charged.ticket) &&
         exact_attention_receipt(world, &reacquired, receipt);
}

struct ResidentAssociationCapacityReceipt {
  std::uint32_t retained_fragments = 0u;
  std::uint32_t complete_sources = 0u;
  std::uint32_t positive_sources = 0u;
  std::uint32_t defeated_sources = 0u;
  std::uint32_t free_records = 0u;
  std::uint64_t retained_matter_q8 = 0u;
  std::uint64_t free_matter_q8 = 0u;
  std::uint64_t resident_revision = 0u;
  std::uint32_t exact = 0u;
};

struct MatterPaidAssociationAssimilationReceipt {
  participation::AssimilationReceipt assimilation{};
  ResidentAssociationCapacityReceipt pre_capacity{};
  ResidentAssociationCapacityReceipt post_capacity{};
  std::uint32_t required_fragments = 0u;
  std::uint32_t retired_fragments = 0u;
  std::uint32_t retired_source = rewrite::kInvalid;
  std::uint32_t retired_association_support = 0u;
  std::uint32_t retired_before = 0u;
  std::uint32_t retired_relation = 0u;
  std::uint32_t retired_after = 0u;
  std::uint32_t pressure_turnover = 0u;
  std::uint32_t remote_readout_preserved = 0u;
  std::uint32_t matter_closed = 0u;
  std::uint32_t fresh_recurrence_regrowth = 0u;
};

struct MatterPaidHeldoutReadoutReceipt {
  participation::ProbeReceipt readout{};
  ResidentAssociationCapacityReceipt capacity{};
  std::uint64_t participating_matter_q8 = 0u;
  std::uint32_t closure_records = 0u;
  std::uint32_t closure_steps = 0u;
  std::uint32_t closure_source_participations = 0u;
  std::uint32_t exact = 0u;
};
enum class ParticipationComponentStatus : std::uint32_t {
  invalid = 0u,
  singleton_withheld = 1u,
  unsupported_probe = 2u,
  component_consumed = 3u,
  publicly_committed = 4u,
  component_released = 5u,
};

struct ParticipationComponentCharge {
  kernel::ActionReturnTicket ticket{};
  rewrite::Record before[pressure::kMaximumChargedExtent]{};
  std::uint32_t slot[pressure::kMaximumChargedExtent]{};
  std::uint32_t extent = 0u;
  std::uint32_t binding = 0u;
  std::uint32_t transaction_tag = 0u;
  std::uint64_t pre_revision = 0u;
  std::uint64_t commit_revision = 0u;
  std::uint64_t public_sequence = 0u;
  ParticipationComponentStatus status = ParticipationComponentStatus::invalid;
};

struct ParticipationComponentPressureReceipt {
  kernel::ActionReturnTicket ticket{};
  std::uint32_t requested_records = 0u;
  std::uint32_t consumed_records = 0u;
  std::uint32_t singleton_sources = 0u;
  std::uint32_t independent_sources = 0u;
  std::uint32_t participating_records = 0u;
  std::uint32_t closure_steps = 0u;
  std::uint32_t closure_records = 0u;
  std::uint64_t consumed_matter_q8 = 0u;
  std::uint64_t pre_revision = 0u;
  std::uint64_t post_revision = 0u;
  std::uint32_t matter_closed = 0u;
  ParticipationComponentStatus status = ParticipationComponentStatus::invalid;
};
#include "bcc32_resident_resource_attention_association_turnover.inl"
BCC32_RESOURCE_ATTENTION_HD inline bool participation_record_less(
    const rewrite::Record& left, const rewrite::Record& right) {
  if (left.lane[4] != right.lane[4]) return left.lane[4] < right.lane[4];
  if (left.lane[1] != right.lane[1]) return left.lane[1] < right.lane[1];
  if (left.lane[2] != right.lane[2]) return left.lane[2] < right.lane[2];
  if (left.lane[3] != right.lane[3]) return left.lane[3] < right.lane[3];
  return left.revision < right.revision;
}
BCC32_RESOURCE_ATTENTION_HD inline bool exact_component_source_pair(
    const rewrite::ResidentRewriteState& world,
    const rewrite::Record& consequent, std::uint32_t before,
    std::uint32_t* antecedent_slot) {
  if (!participation::is_participation(consequent) ||
      consequent.lane[1] != participation::kConsequentFragment ||
      consequent.lane[5] != 1u || consequent.lane[6] != 0u)
    return false;
  std::uint32_t found = rewrite::kInvalid;
  for (std::uint32_t slot = 0u; slot < rewrite::live_record_capacity(&world);
       ++slot) {
    const rewrite::Record& antecedent = world.records[slot];
    if (!participation::is_participation(antecedent) ||
        antecedent.lane[1] != participation::kAntecedentFragment ||
        antecedent.lane[2] != consequent.lane[2] ||
        antecedent.lane[3] != before ||
        antecedent.lane[4] != consequent.lane[4] ||
        antecedent.lane[5] != 1u || antecedent.lane[6] != 0u)
      continue;
    if (found != rewrite::kInvalid) return false;
    found = slot;
  }
  if (found == rewrite::kInvalid) return false;
  if (antecedent_slot != nullptr) *antecedent_slot = found;
  return true;
}
// Collect the unique live matter of every step in the canonical resident
// closure. Repeated use of one Record is paid once; a caller cannot supply an
// extent, candidate consequence, or semantic cost.
BCC32_RESOURCE_ATTENTION_HD inline bool collect_participation_closure_records(
    const rewrite::ResidentRewriteState& world,
    const rewrite::RawRewriteEvent* probe, std::uint32_t probe_count,
    std::uint32_t selected[pressure::kMaximumChargedExtent],
    std::uint32_t* selected_extent, std::uint32_t* source_participations,
    std::uint32_t* closure_steps, std::uint32_t* proposed_event) {
  if (probe == nullptr || selected == nullptr || selected_extent == nullptr ||
      source_participations == nullptr || closure_steps == nullptr ||
      proposed_event == nullptr || probe_count < 2u || probe_count > 4u ||
      probe[0].valid == 0u ||
      probe[0].reserved != rewrite::kEventFrameNone)
    return false;
  std::uint32_t extent = 0u;
  std::uint32_t sources = 0u;
  std::uint32_t current = probe[0].value;
  for (std::uint32_t step = 1u; step < probe_count; ++step) {
    if (probe[step].valid == 0u ||
        probe[step].reserved != rewrite::kEventFrameNone)
      return false;
    const participation::StepResolution resolution =
        participation::resolve_relation_step(
            &world, current, probe[step].value);
    if (resolution.ready != 1u || resolution.after == 0u ||
        resolution.after == rewrite::kInvalid)
      return false;
    std::uint32_t step_sources = 0u;
    for (std::uint32_t consequent_slot = 0u;
         consequent_slot < rewrite::live_record_capacity(&world);
         ++consequent_slot) {
      const rewrite::Record& consequent = world.records[consequent_slot];
      if (!participation::is_participation(consequent) ||
          consequent.lane[1] != participation::kConsequentFragment ||
          consequent.lane[2] != probe[step].value ||
          consequent.lane[3] != resolution.after)
        continue;
      std::uint32_t antecedent_slot = rewrite::kInvalid;
      if (!exact_component_source_pair(world, consequent, current,
                                       &antecedent_slot))
        continue;
      ++step_sources;
      const std::uint32_t pair[2]{antecedent_slot, consequent_slot};
      for (std::uint32_t member = 0u; member < 2u; ++member) {
        bool present = false;
        for (std::uint32_t prior = 0u; prior < extent; ++prior)
          present |= selected[prior] == pair[member];
        if (present) continue;
        if (extent >= pressure::kMaximumChargedExtent) return false;
        selected[extent++] = pair[member];
      }
    }
    if (step_sources != resolution.independent_sources ||
        step_sources == 0u)
      return false;
    sources += step_sources;
    current = resolution.after;
  }
  if (extent == 0u || (extent & 1u) != 0u) return false;
  *selected_extent = extent;
  *source_participations = sources;
  *closure_steps = probe_count - 1u;
  *proposed_event = current;
  return true;
}

// Held-out readout pays unique live Record matter across its full trajectory,
// rather than multiplying one repeatedly traversed fragment or charging only
// the selected final association.
BCC32_RESOURCE_ATTENTION_HD inline MatterPaidHeldoutReadoutReceipt
read_matter_paid_heldout_association(
    const rewrite::ResidentRewriteState& world,
    const rewrite::RawRewriteEvent* probe, std::uint32_t probe_count) {
  MatterPaidHeldoutReadoutReceipt receipt{};
  receipt.capacity = read_association_capacity(world);
  if (world.fault != 0u || !rewrite::matter_account_is_closed(world))
    return receipt;
  receipt.readout = participation::propose_next_event(
      &world, probe, probe_count);
  if (receipt.readout.ready == 0u) return receipt;
  std::uint32_t selected[pressure::kMaximumChargedExtent]{};
  std::uint32_t proposed = 0u;
  if (!collect_participation_closure_records(
          world, probe, probe_count, selected, &receipt.closure_records,
          &receipt.closure_source_participations, &receipt.closure_steps,
          &proposed) ||
      proposed != receipt.readout.proposed_event ||
      receipt.closure_steps != receipt.readout.steps_completed ||
      receipt.closure_source_participations !=
          receipt.readout.independent_sources)
    return MatterPaidHeldoutReadoutReceipt{};
  receipt.participating_matter_q8 =
      static_cast<std::uint64_t>(receipt.closure_records) *
      rewrite::kRecordMatterQ8;
  receipt.exact = receipt.closure_records != 0u &&
                          receipt.closure_records <=
                              receipt.capacity.retained_fragments &&
                          receipt.participating_matter_q8 <=
                              receipt.capacity.retained_matter_q8
                      ? 1u
                      : 0u;
  if (receipt.exact == 0u) receipt.readout = participation::ProbeReceipt{};
  return receipt;
}
// Derive contributing Records; one source cannot authorize public action.
BCC32_RESOURCE_ATTENTION_HD inline ParticipationComponentStatus
derive_participation_component_records(
    const rewrite::ResidentRewriteState& world,
    const rewrite::RawRewriteEvent* probe, std::uint32_t probe_count,
    const MatterPaidHeldoutReadoutReceipt& heldout,
    std::uint32_t selected[pressure::kMaximumChargedExtent],
    std::uint32_t* selected_extent, std::uint32_t* binding,
    std::uint32_t* singleton_sources) {
  if (selected == nullptr || selected_extent == nullptr || binding == nullptr ||
      singleton_sources == nullptr || probe == nullptr || probe_count < 2u ||
      probe_count > 4u ||
      heldout.exact != 1u || heldout.readout.ready != 1u ||
      heldout.readout.steps_completed != probe_count - 1u ||
      heldout.readout.proposed_event == 0u)
    return ParticipationComponentStatus::unsupported_probe;
  std::uint32_t source_count = 0u;
  std::uint32_t steps = 0u;
  std::uint32_t proposed = 0u;
  if (!collect_participation_closure_records(
          world, probe, probe_count, selected, selected_extent,
          &source_count, &steps, &proposed) ||
      proposed != heldout.readout.proposed_event ||
      steps != heldout.closure_steps ||
      source_count != heldout.closure_source_participations ||
      *selected_extent != heldout.closure_records)
    return ParticipationComponentStatus::unsupported_probe;
  *singleton_sources = source_count == 1u ? 1u : 0u;

  for (std::uint32_t left = 0u; left < *selected_extent; ++left) {
    for (std::uint32_t right = left + 1u; right < *selected_extent; ++right) {
      if (!participation_record_less(world.records[selected[right]],
                                     world.records[selected[left]]))
        continue;
      const std::uint32_t swap = selected[left];
      selected[left] = selected[right];
      selected[right] = swap;
    }
  }
  std::uint32_t digest = rewrite::rewrite_mix(
      probe[0].value, probe_count, heldout.readout.proposed_event);
  for (std::uint32_t step = 1u; step < probe_count; ++step)
    digest = rewrite::rewrite_mix(
        digest, probe[step].value, step);
  for (std::uint32_t index = 0u; index < *selected_extent; ++index) {
    const rewrite::Record& record = world.records[selected[index]];
    digest = rewrite::rewrite_mix(
        digest ^ record.lane[1], record.lane[2] ^ record.lane[3],
        record.lane[4] ^ record.revision ^ index);
  }
  *binding = digest == 0u ? 1u : digest;
  return ParticipationComponentStatus::component_consumed;
}
BCC32_RESOURCE_ATTENTION_HD inline bool exact_component_charge(
    const rewrite::ResidentRewriteState& world,
    const ParticipationComponentCharge& charge) {
  if (charge.status != ParticipationComponentStatus::component_consumed ||
      !guarded::valid_ticket(charge.ticket) || charge.extent < 4u ||
      charge.extent > pressure::kMaximumChargedExtent ||
      charge.binding == 0u || charge.transaction_tag == 0u ||
      charge.commit_revision != world.revision)
    return false;
  for (std::uint32_t index = 0u; index < charge.extent; ++index) {
    const std::uint32_t slot = charge.slot[index];
    if (slot >= rewrite::live_record_capacity(&world) ||
        !participation::is_participation(charge.before[index]) ||
        charge.before[index].matter_q8 != rewrite::kRecordMatterQ8 ||
        !pressure::live_owned_form(world.records[slot],
                                   pressure::kFormCommittedMatter,
                                   charge.before[index].lane[4]) ||
        world.records[slot].matter_q8 != rewrite::kRecordMatterQ8 ||
        world.records[slot].lane[3] !=
            static_cast<std::uint32_t>(
                pressure::PaymentPurpose::action_proposal) ||
        world.records[slot].lane[4] != charge.transaction_tag ||
        world.records[slot].reserved[0] !=
            guarded::ticket_sequence_low(charge.ticket) ||
        world.records[slot].reserved[1] !=
            guarded::ticket_sequence_high(charge.ticket))
      return false;
    for (std::uint32_t prior = 0u; prior < index; ++prior)
      if (charge.slot[prior] == slot) return false;
  }
  return true;
}
// A rejected publisher did not perform the charged public action. Restore the
// exact resident organization transactionally; this is not a resource return
// and cannot be invoked after publication or a raw consequence.
BCC32_RESOURCE_ATTENTION_HD inline bool
rollback_unpublished_participation_component(
    rewrite::ResidentRewriteState* world,
    ParticipationComponentCharge* charge) {
  if (world == nullptr || charge == nullptr ||
      charge->status != ParticipationComponentStatus::component_consumed ||
      !exact_component_charge(*world, *charge))
    return false;
  for (std::uint32_t index = 0u; index < charge->extent; ++index)
    world->records[charge->slot[index]] = charge->before[index];
  world->revision = charge->pre_revision;
  if (!rewrite::matter_account_is_closed(*world)) return false;
  *charge = ParticipationComponentCharge{};
  return true;
}
// Consume the exact positive component that produced held-out readout.
BCC32_RESOURCE_ATTENTION_HD inline ParticipationComponentStatus
consume_heldout_participation_component(
    rewrite::ResidentRewriteState* world,
    const kernel::ActionReturnTicket& ticket,
    const rewrite::RawRewriteEvent* probe, std::uint32_t probe_count,
    ParticipationComponentCharge* charge,
    ParticipationComponentPressureReceipt* receipt) {
  if (world == nullptr || charge == nullptr || receipt == nullptr ||
      !guarded::valid_ticket(ticket) ||
      guarded::ticket_is_bound_or_consumed(*world, ticket))
    return ParticipationComponentStatus::invalid;
  ParticipationComponentPressureReceipt result{};
  result.ticket = ticket;
  result.pre_revision = world->revision;
  if (!rewrite::matter_account_is_closed(*world)) return result.status;
  const MatterPaidHeldoutReadoutReceipt heldout =
      read_matter_paid_heldout_association(*world, probe, probe_count);
  result.independent_sources = heldout.readout.independent_sources;
  result.participating_records = heldout.readout.participating_records;
  result.closure_steps = heldout.closure_steps;
  result.closure_records = heldout.closure_records;
  std::uint32_t slots[pressure::kMaximumChargedExtent]{};
  std::uint32_t extent = 0u;
  std::uint32_t binding = 0u;
  const ParticipationComponentStatus derived =
      derive_participation_component_records(
          *world, probe, probe_count, heldout, slots, &extent, &binding,
          &result.singleton_sources);
  result.requested_records = extent;
  if (derived != ParticipationComponentStatus::component_consumed) {
    result.status = derived;
    result.post_revision = result.pre_revision;
    *receipt = result;
    return result.status;
  }
  ParticipationComponentCharge candidate{};
  candidate.ticket = ticket;
  candidate.extent = extent;
  candidate.binding = binding;
  candidate.pre_revision = world->revision;
  candidate.commit_revision = world->revision == ~std::uint64_t{0}
                                  ? world->revision
                                  : world->revision + 1u;
  candidate.transaction_tag = rewrite::rewrite_mix(
      binding ^ guarded::ticket_sequence_low(ticket),
      guarded::ticket_sequence_high(ticket),
      static_cast<std::uint32_t>(candidate.commit_revision));
  if (candidate.transaction_tag == 0u) candidate.transaction_tag = 1u;
  for (std::uint32_t index = 0u; index < extent; ++index) {
    if (slots[index] >= rewrite::live_record_capacity(world) ||
        !participation::is_participation(world->records[slots[index]]) ||
        world->records[slots[index]].matter_q8 != rewrite::kRecordMatterQ8)
      return ParticipationComponentStatus::invalid;
    candidate.slot[index] = slots[index];
    candidate.before[index] = world->records[slots[index]];
  }
  for (std::uint32_t index = 0u; index < extent; ++index) {
    rewrite::Record& record = world->records[slots[index]];
    const std::uint32_t owner = candidate.before[index].lane[4];
    pressure::write_owned_form(&record, pressure::kFormCommittedMatter, owner);
    record.lane[3] = static_cast<std::uint32_t>(
        pressure::PaymentPurpose::action_proposal);
    record.lane[4] = candidate.transaction_tag;
    record.lane[5] = static_cast<std::uint32_t>(candidate.commit_revision);
    record.lane[6] =
        static_cast<std::uint32_t>(candidate.commit_revision >> 32u);
    record.reserved[0] = guarded::ticket_sequence_low(ticket);
    record.reserved[1] = guarded::ticket_sequence_high(ticket);
  }
  pressure::increment_revision(world);
  candidate.status = ParticipationComponentStatus::component_consumed;
  result.consumed_records = extent;
  result.consumed_matter_q8 =
      static_cast<std::uint64_t>(extent) * rewrite::kRecordMatterQ8;
  result.post_revision = world->revision;
  result.matter_closed = rewrite::matter_account_is_closed(*world) ? 1u : 0u;
  if (result.matter_closed == 0u) {
    for (std::uint32_t index = 0u; index < extent; ++index)
      world->records[candidate.slot[index]] = candidate.before[index];
    world->revision = candidate.pre_revision;
    return ParticipationComponentStatus::invalid;
  }
  result.status = candidate.status;
  *charge = candidate;
  *receipt = result;
  return result.status;
}
#include "bcc32_resident_resource_attention_component_return.inl"

#undef BCC32_RESOURCE_ATTENTION_HD

}  // namespace substrate::bcc32::resident_resource_attention_discourse
