// Device-resident conditioned prediction error and credit-bank kernels.
// Included inside the adult-v1 namespace after its shared state and key types.

// The transition table is the resident predictor.  Before the current contact
// changes it, each learned (anchor, previous) context casts exactly one
// prediction: its uniquely strongest next-unit route.  A different observed
// next unit emits a signed exact-key carrier for the physical organ owner.
// This read-only pass never mutates inventory counts or ledger mass. Ties
// remain unresolved rather than receiving an authored winner, so ambiguity
// is not mistaken for error.
// Deterministic bridge record from the adult predictor into physical learning
// matter.  Each decoded event owns slots 2*event (observed positive credit)
// and 2*event+1 (uniquely predicted negative credit).  The fixed slot mapping
// avoids atomic append order and preserves the exact conditioned key; it is
// not itself a weight or a learning update.
__device__ inline bool decrement_resident_count(std::uint32_t* count) {
  std::uint32_t observed = atomicAdd(count, 0u);
  while (observed != 0u) {
    const std::uint32_t prior = atomicCAS(count, observed, observed - 1u);
    if (prior == observed) return true;
    observed = prior;
  }
  return false;
}

__device__ inline bool decode_conditioned_transition_event(
    const std::uint32_t* sequence, std::uint32_t sequence_count,
    std::uint32_t prefix_count, const std::uint32_t* segment_ids,
    std::uint32_t event, ConditionedTransitionKey* output) {
  std::uint32_t remaining = event;
  for (std::uint32_t lag = 0u; lag < kConditionedTransitionLagCount; ++lag) {
    const std::uint32_t extent = sequence_count > lag + 1u
        ? sequence_count - lag - 1u -
              (prefix_count > lag + 1u ? prefix_count - lag - 1u : 0u)
        : 0u;
    if (remaining >= extent) {
      remaining -= extent;
      continue;
    }
    const std::uint32_t start = prefix_count > lag + 1u
        ? prefix_count - lag - 1u : 0u;
    const std::uint32_t current = lag + start + remaining;
    if (segment_ids[current - lag] != segment_ids[current] ||
        segment_ids[current] != segment_ids[current + 1u])
      return false;
    *output = ConditionedTransitionKey{
        sequence[current - lag], sequence[current], sequence[current + 1u]};
    return true;
  }
  return false;
}

__global__ void apply_conditioned_prediction_error_kernel(
    const std::uint32_t* sequence, std::uint32_t sequence_count,
    std::uint32_t prefix_count, const std::uint32_t* segment_ids,
    const ConditionedTransitionKey* prior_transitions,
    const std::uint32_t* prior_conductance, std::uint32_t prior_count,
    std::uint32_t event_count, std::uint32_t efferent_contact,
    std::int32_t efferent_polarity, std::uint32_t outcome_present,
    ConditionedCreditEvent* credit_events,
    ConditionedPredictionReceipt* receipt,
    resident_credit::BankView prior_bank = {},
    ConditionedPredictionWitness* prediction_witnesses = nullptr) {
  for (std::uint32_t event = blockIdx.x * blockDim.x + threadIdx.x;
       event < event_count; event += blockDim.x * gridDim.x) {
    if (credit_events != nullptr) {
      credit_events[2u * event] = ConditionedCreditEvent{};
      credit_events[2u * event + 1u] = ConditionedCreditEvent{};
    }
    if (prediction_witnesses != nullptr)
      prediction_witnesses[event] = ConditionedPredictionWitness{};
    ConditionedTransitionKey observed{};
    if (!decode_conditioned_transition_event(sequence, sequence_count,
                                             prefix_count, segment_ids, event,
                                             &observed))
      continue;
    if (credit_events != nullptr) {
      credit_events[2u * event] =
          ConditionedCreditEvent{observed, 1, event, 1u};
    }
    atomicAdd(&receipt->positive_credit_events, 1u);
    atomicAdd(&receipt->observed_events, 1u);
    const std::uint32_t begin =
        bcc32_cuda_resident_synthesis::lower_subject_transition(
            prior_transitions, prior_count, observed.anchor, observed.previous);
    const std::uint32_t end =
        bcc32_cuda_resident_synthesis::upper_subject_transition(
            prior_transitions, prior_count, observed.anchor, observed.previous);
    std::uint32_t predicted_index = 0xffffffffu;
    std::int64_t predicted_mass = 0;
    std::uint32_t predicted_exposure = 0;
    bool tied = false;
    for (std::uint32_t candidate = begin; candidate < end; ++candidate) {
      std::int64_t mass =
          static_cast<std::int64_t>(prior_conductance[candidate]);
      std::uint32_t exposure = 0;
      if (prior_bank.routes != nullptr && prior_bank.scalars != nullptr) {
        const ConditionedTransitionKey transition = prior_transitions[candidate];
        const resident_credit::RouteKey key = resident_credit::make_route_key(
            transition.anchor, transition.previous, transition.next,
            resident_credit::route_region(transition.anchor,
                                          transition.previous));
        const std::uint32_t slot = resident_credit::find_route(prior_bank, key);
        mass = slot < prior_bank.capacity
            ? resident_credit::conductance(prior_bank.routes[slot]) : 0;
        exposure = slot < prior_bank.capacity
            ? static_cast<std::uint32_t>(prior_bank.routes[slot].exposure) : 0u;
      }
      if (mass > predicted_mass) {
        predicted_index = candidate;
        predicted_mass = mass;
        predicted_exposure = exposure;
        tied = false;
      } else if (mass > 0 && mass == predicted_mass) {
        // Conductance alone cannot break this tie -- both candidates saturated the
        // same capped value. Fall back to `exposure`, an uncapped count of positive
        // credit ATTEMPTS: a route taught 30 times still outranks one taught 3 times
        // even though `positive` clamped both to the same kSupportLimit. Only a tie in
        // BOTH conductance and exposure is a genuine, unresolvable tie.
        if (exposure > predicted_exposure) {
          predicted_index = candidate;
          predicted_exposure = exposure;
          tied = false;
        } else if (exposure == predicted_exposure) {
          tied = true;
        }
      }
    }
    if (predicted_index == 0xffffffffu || predicted_mass <= 0 || tied)
      continue;
    if (prediction_witnesses != nullptr) {
      const ConditionedTransitionKey predicted =
          prior_transitions[predicted_index];
      // The predictor names the route it actually selected. The resident
      // owner chooses the eligibility endpoint from the route matter it grew;
      // a producer-local candidate index is not durable anatomy.
      const std::uint32_t region = 0u;
      prediction_witnesses[event] = {
          predicted, region, 0xffffffffu, event, 1u};
    }
    atomicAdd(&receipt->predicted_events, 1u);
    const bool transition_error =
        prior_transitions[predicted_index].next != observed.next;
    // A later body consequence is a second, independent error carrier.  It
    // remains opaque to the host: only the boundary's action polarity and
    // the later present/absent outcome reach this local rule.  Thus a route
    // can be structurally accurate yet still lose support when it predicts a
    // bodily consequence that does not arrive.
    const bool somatic_error = efferent_contact != 0u &&
        ((efferent_polarity > 0) != (outcome_present != 0u));
    if (!transition_error && !somatic_error) {
      atomicAdd(&receipt->correct_events, 1u);
      continue;
    }
    atomicAdd(&receipt->error_events, transition_error ? 1u : 0u);
    atomicAdd(&receipt->somatic_error_events, somatic_error ? 1u : 0u);
    if (credit_events != nullptr) {
      credit_events[2u * event + 1u] =
          ConditionedCreditEvent{prior_transitions[predicted_index], -1,
                                 event, 1u};
    }
    atomicAdd(&receipt->negative_credit_events, 1u);
  }
}

__global__ void apply_conditioned_credit_bank_kernel(
    const ConditionedCreditEvent* events, std::uint32_t event_count,
    resident_credit::BankView bank) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || events == nullptr) return;
  for (std::uint32_t index = 0u; index < event_count; ++index) {
    const ConditionedCreditEvent event = events[index];
    if (event.valid == 0u || event.polarity == 0) continue;
    const std::uint32_t region =
        resident_credit::route_region(event.key.anchor, event.key.previous);
    resident_credit::apply_signed_credit(
        bank,
        resident_credit::make_route_key(
            event.key.anchor, event.key.previous, event.key.next, region),
        event.polarity, event.source_event);
  }
}

__global__ void publish_conditioned_credit_conductance_kernel(
    const ConditionedTransitionKey* transitions, std::uint32_t count,
    resident_credit::BankView bank, std::uint32_t* conductance,
    std::uint32_t* exposure = nullptr) {
  for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count; index += blockDim.x * gridDim.x) {
    const ConditionedTransitionKey transition = transitions[index];
    const resident_credit::RouteKey key = resident_credit::make_route_key(
        transition.anchor, transition.previous, transition.next,
        resident_credit::route_region(transition.anchor, transition.previous));
    const std::uint32_t slot = resident_credit::find_route(bank, key);
    const std::int32_t signed_conductance =
        slot < bank.capacity ? resident_credit::conductance(bank.routes[slot]) : 0;
    conductance[index] = signed_conductance > 0
        ? static_cast<std::uint32_t>(signed_conductance) : 0u;
    if (exposure != nullptr) {
      exposure[index] = slot < bank.capacity
          ? static_cast<std::uint32_t>(bank.routes[slot].exposure) : 0u;
    }
  }
}

__global__ void lesion_conditioned_credit_bank_kernel(
    resident_credit::BankView bank) {
  if (bank.routes == nullptr || bank.scalars == nullptr) return;
  for (std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
       slot < bank.capacity;
       slot += blockDim.x * gridDim.x) {
    resident_credit::RouteState& route = bank.routes[slot];
    const std::uint32_t released =
        static_cast<std::uint32_t>(route.positive) +
        static_cast<std::uint32_t>(route.negative);
    if (released != 0u) {
      atomicAdd(&bank.scalars->escrow_quanta, released);
    }
    route.positive = 0u;
    route.negative = 0u;
    route.eligible = 0u;
    route.expiry_tick = 0u;
    // A lesion must erase exposure too, or a stale pre-lesion count survives a total
    // ablation and wins a tie against a genuinely 0-vs-0 conductance comparison --
    // exactly the failure a changed-locus lesion is supposed to make impossible.
    route.exposure = 0u;
  }
  if (blockIdx.x == 0u && threadIdx.x == 0u) {
    for (std::uint32_t index = 0u;
         index < resident_credit::kHorizonDepth;
         ++index) {
      bank.scalars->pending_slot[index] = resident_credit::kInvalidSlot;
      bank.scalars->pending_tick[index] = 0u;
      bank.scalars->pending_valid[index] = 0u;
    }
    bank.scalars->pending_head = 0u;
    bank.scalars->pending_count = 0u;
  }
}

__global__ void append_conditioned_transitions_kernel(
    const std::uint32_t* sequence, std::uint32_t sequence_count,
    std::uint32_t prefix_count, const std::uint32_t* segment_ids,
    ConditionedTransitionKey* transitions, std::uint32_t* counts,
    std::uint32_t transition_base, std::uint32_t event_count,
    std::uint32_t* appended_count, std::uint32_t* ledger) {
  for (std::uint32_t event = blockIdx.x * blockDim.x + threadIdx.x;
       event < event_count; event += blockDim.x * gridDim.x) {
    ConditionedTransitionKey transition{};
    if (!decode_conditioned_transition_event(sequence, sequence_count,
                                             prefix_count, segment_ids, event,
                                             &transition)) {
      atomicAdd(ledger + 1u, 1u);
      atomicSub(ledger + 2u, 1u);
      continue;
    }
      const std::uint32_t out = transition_base + atomicAdd(appended_count, 1u);
      transitions[out] = transition;
      counts[out] = 1u;
  }
}
