// Included inside direct_adult_legacy_oracle.cu's anonymous namespace after
// direct_adult_v01_birth.cuh. Owns membrane chronology: contact signatures,
// the single-event append body with its displacement policy, serial batch
// ingress, and resident-context reset. This is skull physics -- which of two
// events crosses when only one may -- and stays branchable on CausalOrigin
// alone.
__device__ std::uint32_t mix_signature(std::uint64_t value) {
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ull;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebull;
  value ^= value >> 31;
  const std::uint32_t folded = static_cast<std::uint32_t>(value ^ (value >> 32));
  return folded == 0u ? 1u : folded;
}

// #1236: forwards to the shared canonical helper rather than re-deriving it.
// This hash is what a successor's context signature is built from, so it is
// part of the causal law and must have exactly one definition.
__device__ std::uint32_t predecessor_signature(std::uint32_t node, Word word) {
  return direct_canonical_predecessor_signature(node, word);
}

__device__ std::uint64_t packed_context_signature(std::uint32_t rolling_history,
                                                  std::uint32_t last_node, Word last_word) {
  const std::uint32_t shallow = predecessor_signature(last_node, last_word);
  return (static_cast<std::uint64_t>(rolling_history) << 32) | shallow;
}

// #1236: this translation unit's private cue_context_signature moved to
// direct_canonical_evaluator_device.cuh as
// direct_canonical_event_context_signature. It is part of the causal law --
// which stored route an event is even asking about -- so it cannot stay TU-local
// where each backend was free to derive it its own way, and it is deleted here
// rather than left as an unread second copy.

// The single-event append body, factored out so the batch kernel carries exactly
// the same chronology and context-history updates in the same order. Serial by
// construction: callers must invoke it from one thread only.
// 0X1-1176: how far a displacing stream contact will look for an endogenous
// occupant before giving up. On a saturated frontier essentially every occupant
// is endogenous, so the first probe succeeds; the bound exists so a frontier
// that is genuinely all-contact cannot turn ingress into a full scan.
inline constexpr std::uint32_t kIngressDisplacementProbes = 8u;

__device__ inline void append_one_event(ActivityEvent* frontier,
                                        DirectIngressAuthority* frontier_authority,
                                        std::uint32_t* count, std::uint32_t capacity,
                                        ResidentContextState* context_state, ActivityEvent event,
                                        DirectIngressAuthority authority) {
  const bool stream_contact = event.origin == CausalOrigin::external_contact ||
                              event.origin == CausalOrigin::motor_reafference;
  if (stream_contact && context_state != nullptr) {
    const bool same_stream = context_state->last_external_node != kInvalidIndex;
    if (same_stream) {
      if (event.cue_node == kInvalidIndex)
        event.cue_node = context_state->last_external_node;
      event.history_signature = packed_context_signature(context_state->rolling_history,
                                                         context_state->last_external_node,
                                                         context_state->last_external_word);
    }
    context_state->previous_external_node =
        same_stream ? context_state->last_external_node : kInvalidIndex;
    context_state->previous_external_word = same_stream ? context_state->last_external_word : 0u;
    context_state->last_external_node = event.node;
    context_state->last_external_word = event.word;
    const std::uint32_t current = predecessor_signature(event.node, event.word);
    context_state->rolling_history =
        !same_stream
            ? current
            : mix_signature((static_cast<std::uint64_t>(context_state->rolling_history) << 32) |
                            current);
    context_state->sequence_length = same_stream ? context_state->sequence_length + 1u : 1u;
    ++context_state->actual_contact_count;
  }
  const std::uint32_t slot = atomicAdd(count, 1u);
  if (slot < capacity) {
    frontier[slot] = event;
    frontier_authority[slot] = authority;
    return;
  }
  // 0X1-1176: the frontier is already full, and what fills it is the organism's
  // own endogenous successors. Measured on a 4x4096-node adult fed 512 external
  // contacts every tick: the resident frontier holds 511 contacts at tick 1, 512
  // at tick 2, and ZERO from tick 3 onward, because one tick's propagation
  // attempts ~12280 successors into 4096 slots and ingress appends after them.
  // The adult goes deaf on its third tick of life, exact eligibility settlement
  // stops with it, and no amount of eligibility-bank capacity changes that --
  // there is simply nothing external left to settle against.
  //
  // Dropping the world here is not a neutral overflow policy. Propagation ALREADY
  // discards about two thirds of its own successors by `atomicAdd` race order, so
  // an endogenous entry that survived did so arbitrarily and displacing one costs
  // strictly less than the status quo already costs. What the organism cannot
  // regenerate is contact: a prediction it drops it can make again next tick, a
  // world event it drops is gone. So a stream contact displaces an endogenous
  // prediction rather than being discarded, and endogenous overflow is still
  // dropped as before.
  //
  // This is membrane physics -- which of two events crosses the skull boundary
  // when only one may -- not authored cognition. It names no concept, inspects
  // no word, and branches on nothing but CausalOrigin.
  if (!stream_contact)
    return;
  const std::uint32_t overflow_index = slot - capacity;
  for (std::uint32_t probe = 0u; probe < kIngressDisplacementProbes; ++probe) {
    const std::uint32_t victim = (overflow_index + probe) % capacity;
    if (frontier[victim].origin == CausalOrigin::endogenous_prediction) {
      frontier[victim] = event;
      frontier_authority[victim] = authority;
      return;
    }
  }
}

__global__ void append_event_kernel(ActivityEvent* frontier,
                                    DirectIngressAuthority* frontier_authority,
                                    std::uint32_t* count, std::uint32_t capacity,
                                    ResidentContextState* context_state, ActivityEvent event,
                                    DirectIngressAuthority authority) {
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;
  append_one_event(frontier, frontier_authority, count, capacity, context_state, event, authority);
}

// One launch for a whole batch. The loop is deliberately serial and in argument
// order: ingress chronology is a causal fact, not a parallelisable one.
__global__ void append_event_batch_kernel(ActivityEvent* frontier,
                                          DirectIngressAuthority* frontier_authority,
                                          std::uint32_t* count, std::uint32_t capacity,
                                          ResidentContextState* context_state,
                                          const ActivityEvent* events, std::uint32_t event_count,
                                          DirectIngressAuthority authority) {
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;
  for (std::uint32_t i = 0; i < event_count; ++i) {
    append_one_event(frontier, frontier_authority, count, capacity, context_state, events[i],
                     authority);
  }
}

__global__ void reset_resident_context_kernel(ResidentContextState* context_state) {
  if (blockIdx.x != 0 || threadIdx.x != 0 || context_state == nullptr)
    return;
  context_state->last_external_node = kInvalidIndex;
  context_state->last_external_word = 0u;
  context_state->previous_external_node = kInvalidIndex;
  context_state->previous_external_word = 0u;
  context_state->rolling_history = 0u;
  context_state->sequence_length = 0u;
  ++context_state->boundary_count;
}
