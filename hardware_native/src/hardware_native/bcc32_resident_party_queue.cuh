#pragma once

#include "bcc32_resident_party_attention.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace substrate::bcc32::party_queue {

inline constexpr std::uint32_t kQueueCapacity =
    party_attention::kPartyContactCapacity;
inline constexpr std::uint32_t kRawContactCapacity = 256u;

struct RawContact {
  std::uint64_t speaker_identity = 0u;
  const std::uint8_t* raw_bytes = nullptr;
  std::uint32_t raw_byte_count = 0u;
  std::uint64_t transport_sequence = 0u;
  // Receipt of the physical ingress route. It binds later resident evidence
  // and consequence settlement to these exact stored bytes.
  std::uint64_t ingress_receipt_digest = 0u;
};

struct EntryRef {
  std::uint32_t slot = 0xffffffffu;
  std::uint32_t revision = 0u;
  std::uint64_t ingress_receipt_digest = 0u;

  __host__ __device__ bool valid() const {
    return slot < kQueueCapacity && revision != 0u &&
           ingress_receipt_digest != 0u;
  }
};

// These values must be copied from resident relation, goal, continuity, body,
// uncertainty, and resource state. The queue accepts no scalar priority and
// never derives an answer or an audience identity from them.
struct ResidentEvidence {
  std::uint32_t resident_relation_q8 = 0u;
  std::uint32_t resident_goal_q8 = 0u;
  std::uint32_t resident_continuity_q8 = 0u;
  std::uint32_t resident_unresolved_pressure_q8 = 0u;
  std::uint32_t resident_body_pressure_q8 = 0u;
  std::uint32_t resource_reserve_q8 =
      response_inhibition::kPressureMaximum;
};

struct Entry {
  std::uint64_t speaker_identity = 0u;
  std::uint64_t transport_sequence = 0u;
  std::uint64_t ingress_receipt_digest = 0u;
  std::uint64_t reserved_decision_sequence = 0u;
  std::uint32_t raw_byte_count = 0u;
  std::uint32_t revision = 0u;
  std::uint32_t evidence_revision = 0u;
  ResidentEvidence evidence{};
  std::uint8_t raw_bytes[kRawContactCapacity]{};

  __host__ __device__ bool occupied() const {
    return speaker_identity != 0u && raw_byte_count != 0u && revision != 0u &&
           ingress_receipt_digest != 0u;
  }

  __host__ __device__ bool selectable() const {
    return occupied() && evidence_revision != 0u &&
           reserved_decision_sequence == 0u;
  }
};

struct QueueTicket {
  EntryRef entry{};
  response_inhibition::Ticket response{};

  __host__ __device__ bool valid() const {
    return entry.valid() && response.valid();
  }
};

struct Selection {
  party_attention::SelectionStatus status =
      party_attention::SelectionStatus::fail_closed;
  QueueTicket ticket{};
  std::uint64_t selected_speaker_identity = 0u;
  std::uint32_t nondominated_contacts = 0u;
};

struct Receipt {
  std::uint64_t enqueued = 0u;
  std::uint64_t evidence_writes = 0u;
  std::uint64_t selected = 0u;
  std::uint64_t ambiguous = 0u;
  std::uint64_t settled = 0u;
  std::uint64_t rejected_ingress = 0u;
  std::uint64_t rejected_evidence = 0u;
  std::uint64_t rejected_settlement = 0u;
};

struct State {
  Entry entry[kQueueCapacity]{};
  response_inhibition::State inhibition{};
  std::uint64_t last_transport_sequence = 0u;
  std::uint32_t occupied_entries = 0u;
  Receipt receipt{};
};

__host__ __device__ inline EntryRef enqueue(State* state,
                                            const RawContact& contact) {
  EntryRef ref{};
  if (state == nullptr || contact.speaker_identity == 0u ||
      contact.raw_bytes == nullptr || contact.raw_byte_count == 0u ||
      contact.raw_byte_count > kRawContactCapacity ||
      contact.transport_sequence == 0u ||
      contact.transport_sequence <= state->last_transport_sequence ||
      contact.ingress_receipt_digest == 0u ||
      state->occupied_entries >= kQueueCapacity) {
    if (state != nullptr) ++state->receipt.rejected_ingress;
    return ref;
  }

  std::uint32_t slot = 0xffffffffu;
  for (std::uint32_t candidate = 0u; candidate < kQueueCapacity; ++candidate) {
    if (!state->entry[candidate].occupied()) {
      slot = candidate;
      break;
    }
  }
  if (slot == 0xffffffffu) {
    ++state->receipt.rejected_ingress;
    return ref;
  }

  Entry& entry = state->entry[slot];
  entry = Entry{};
  entry.speaker_identity = contact.speaker_identity;
  entry.transport_sequence = contact.transport_sequence;
  entry.ingress_receipt_digest = contact.ingress_receipt_digest;
  entry.raw_byte_count = contact.raw_byte_count;
  entry.revision = 1u;
  for (std::uint32_t index = 0u; index < contact.raw_byte_count; ++index)
    entry.raw_bytes[index] = contact.raw_bytes[index];

  state->last_transport_sequence = contact.transport_sequence;
  ++state->occupied_entries;
  ++state->receipt.enqueued;
  ref.slot = slot;
  ref.revision = entry.revision;
  ref.ingress_receipt_digest = entry.ingress_receipt_digest;
  return ref;
}

__host__ __device__ inline bool write_resident_evidence(
    State* state, const EntryRef& ref, const ResidentEvidence& evidence) {
  if (state == nullptr || !ref.valid()) {
    if (state != nullptr) ++state->receipt.rejected_evidence;
    return false;
  }
  Entry& entry = state->entry[ref.slot];
  if (!entry.occupied() || entry.revision != ref.revision ||
      entry.ingress_receipt_digest != ref.ingress_receipt_digest ||
      entry.reserved_decision_sequence != 0u) {
    ++state->receipt.rejected_evidence;
    return false;
  }
  entry.evidence = evidence;
  ++entry.evidence_revision;
  ++entry.revision;
  ++state->receipt.evidence_writes;
  return true;
}

__host__ __device__ inline Selection select_next(State* state) {
  Selection result{};
  if (state == nullptr) return result;

  party_attention::ContactEvidence candidates[kQueueCapacity]{};
  for (std::uint32_t slot = 0u; slot < kQueueCapacity; ++slot) {
    const Entry& entry = state->entry[slot];
    if (!entry.selectable()) continue;
    candidates[slot].contact.speaker_identity = entry.speaker_identity;
    candidates[slot].contact.raw_bytes = entry.raw_bytes;
    candidates[slot].contact.raw_byte_count = entry.raw_byte_count;
    candidates[slot].contact.ingress_sequence = entry.transport_sequence;
    candidates[slot].contact.resident_unresolved_pressure_q8 =
        entry.evidence.resident_unresolved_pressure_q8;
    candidates[slot].contact.resident_body_pressure_q8 =
        entry.evidence.resident_body_pressure_q8;
    candidates[slot].contact.resource_reserve_q8 =
        entry.evidence.resource_reserve_q8;
    candidates[slot].resident_relation_q8 =
        entry.evidence.resident_relation_q8;
    candidates[slot].resident_goal_q8 = entry.evidence.resident_goal_q8;
    candidates[slot].resident_continuity_q8 =
        entry.evidence.resident_continuity_q8;
  }

  const party_attention::Selection selected = party_attention::select_contact(
      &state->inhibition, candidates, kQueueCapacity);
  result.status = selected.status;
  result.nondominated_contacts = selected.nondominated_contacts;
  if (selected.status == party_attention::SelectionStatus::ambiguous) {
    ++state->receipt.ambiguous;
    return result;
  }
  if (selected.status != party_attention::SelectionStatus::unique ||
      selected.selected_index >= kQueueCapacity ||
      !selected.response_ticket.valid()) {
    return result;
  }

  Entry& entry = state->entry[selected.selected_index];
  if (!entry.selectable() ||
      entry.speaker_identity != selected.selected_speaker_identity) {
    result.status = party_attention::SelectionStatus::fail_closed;
    return result;
  }
  entry.reserved_decision_sequence =
      selected.response_ticket.decision_sequence;
  ++entry.revision;
  result.ticket.entry = {selected.selected_index, entry.revision,
                         entry.ingress_receipt_digest};
  result.ticket.response = selected.response_ticket;
  result.selected_speaker_identity = selected.selected_speaker_identity;
  ++state->receipt.selected;
  return result;
}

__host__ __device__ inline bool settle_selected(
    State* state, const QueueTicket& ticket,
    const response_inhibition::Consequence& consequence) {
  if (state == nullptr || !ticket.valid()) {
    if (state != nullptr) ++state->receipt.rejected_settlement;
    return false;
  }
  Entry& entry = state->entry[ticket.entry.slot];
  if (!entry.occupied() || entry.revision != ticket.entry.revision ||
      entry.ingress_receipt_digest != ticket.entry.ingress_receipt_digest ||
      entry.reserved_decision_sequence != ticket.response.decision_sequence ||
      entry.speaker_identity != ticket.response.speaker_identity) {
    ++state->receipt.rejected_settlement;
    return false;
  }
  if (!response_inhibition::settle_consequence(&state->inhibition,
                                                ticket.response,
                                                consequence)) {
    ++state->receipt.rejected_settlement;
    return false;
  }
  entry = Entry{};
  --state->occupied_entries;
  ++state->receipt.settled;
  return true;
}

}  // namespace substrate::bcc32::party_queue
