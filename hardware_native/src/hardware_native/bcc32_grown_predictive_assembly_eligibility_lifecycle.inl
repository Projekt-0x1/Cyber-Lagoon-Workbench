// This unit owns predictive-eligibility traffic after segment allocation:
// publication, exact observed residual resolution, and expired-slot release.
// The parent retains segment construction, allocation, and mutable state layout.

BCC32_PA_DEVICE inline bool expire_segment_eligibility(
    DeviceState* state, std::uint32_t tick, DeviceReceipt* receipt = nullptr) {
  for (std::uint32_t population = 0u; population < kPopulationCount;
       ++population) {
    for (std::uint32_t index = 0u; index < kEligibilityPerPopulation; ++index) {
      Eligibility& eligibility =
          state->populations[population].eligibilities[index];
      if (!eligibility.live || eligibility.due_tick > tick ||
          eligibility.publication_tick + kHistoryDepth > tick)
        continue;
      if (!journal_eligibility(state, population, index)) return false;
      eligibility.live = 0u;
      if (receipt != nullptr) ++receipt->expired_eligibility_count;
    }
  }
  return true;
}

BCC32_PA_DEVICE inline bool publish_prediction(DeviceState* state, DeviceReceipt* receipt,
                                               std::uint32_t tick) {
  state->prediction_active = 0u;
  state->predicted_mask = 0u;
  if (tick == 0u) return true;
  const HistoryEntry& previous = state->history[(tick - 1u) % kHistoryDepth];
  if (previous.tick + 1u != tick)
    return true;
  std::uint8_t top_population[kReceiptWinners]{};
  std::uint16_t top_segment[kReceiptWinners]{};
  std::int32_t top_support[kReceiptWinners]{};
  std::uint32_t top_count = 0u;
  // A raw observed column is a valid residual only for the surface
  // population. Higher populations receive delayed structural or motor
  // residuals; treating their cell indices as bytes destroys those routes.
  for (std::uint32_t population = kP0; population <= kP0; ++population) {
    for (std::uint32_t index = 0u; index < kSegmentsPerPopulation; ++index) {
      const Segment& segment = state->populations[population].segments[index];
      if (!segment.live || segment.source_population != kP0 ||
          segment_is_locked(*state, population, segment))
        continue;
      const std::int32_t support = active_segment_support(*state, population, index,
                                                          nullptr, tick);
      if (support < static_cast<std::int32_t>(segment.threshold)) continue;
      std::uint32_t insert = top_count;
      if (insert < kReceiptWinners) ++top_count;
      else if (support <= top_support[kReceiptWinners - 1u]) continue;
      while (insert > 0u && top_support[insert - 1u] < support) {
        if (insert < kReceiptWinners) {
          top_support[insert] = top_support[insert - 1u];
          top_population[insert] = top_population[insert - 1u];
          top_segment[insert] = top_segment[insert - 1u];
        }
        --insert;
      }
      if (insert < kReceiptWinners) {
        top_support[insert] = support;
        top_population[insert] = static_cast<std::uint8_t>(population);
        top_segment[insert] = static_cast<std::uint16_t>(index);
      }
    }
  }
  for (std::uint32_t index = 0u; index < top_count; ++index) {
    const Segment& segment = state->populations[top_population[index]].segments[top_segment[index]];
    bool seen_delay[kHistoryDepth + 1u]{};
    for (std::uint32_t offset = 0u; offset < segment.synapse_count; ++offset) {
      const Synapse& synapse = state->populations[segment.source_population]
                                   .synapses[segment.synapse_begin + offset];
      if (synapse.delay > kHistoryDepth || seen_delay[synapse.delay]) continue;
      seen_delay[synapse.delay] = true;
      std::uint16_t eligibility_index = 0u;
      if (!publish_segment_eligibility(state, top_population[index], top_segment[index],
                                       nullptr, tick, top_support[index],
                                       &eligibility_index, synapse.delay))
        return false;
      ++receipt->prediction_count;
      receipt->prediction_hash ^= hash_event(previous.event, tick + index + synapse.delay);
      if (state->prediction_active == 0u || top_population[index] == kP0) {
        state->prediction_active = 1u;
        state->predicted_mask = 1u << (segment.target_cell & 31u);
        state->predicted_value = segment.target_cell;
        state->predicted_population = top_population[index];
        state->predicted_segment = top_segment[index];
        state->predicted_target = segment.target_cell;
      }
    }
  }
  return true;
}

BCC32_PA_DEVICE inline bool apply_prediction_residual(DeviceState* state, DeviceReceipt* receipt,
                                                       std::uint16_t observed_cell,
                                                       std::uint32_t observed_column,
                                                       std::uint32_t tick,
                                                       RunMode mode,
                                                       std::uint32_t learn_population_mask) {
  (void)observed_cell;
  for (std::uint32_t population = 0u; population < kPopulationCount; ++population) {
    for (std::uint32_t index = 0u; index < kEligibilityPerPopulation; ++index) {
      Eligibility& eligibility = state->populations[population].eligibilities[index];
      if (!eligibility.live || eligibility.due_tick != tick)
        continue;
      const Segment& segment = state->populations[population].segments[eligibility.segment_index];
      if (!segment.live) {
        if (!journal_eligibility(state, population, index)) return false;
        eligibility.live = 0u;
        continue;
      }
      const bool correct = context_cell_column(segment.target_cell) == observed_column;
      if (correct) ++receipt->correct_count;
      else ++receipt->false_prediction_count;
      if (!apply_segment_residual(state, population, index, correct ? 1 : -1,
                                  mode, learn_population_mask))
        return false;
    }
  }
  state->prediction_active = 0u;
  return expire_segment_eligibility(state, tick, receipt);
}
