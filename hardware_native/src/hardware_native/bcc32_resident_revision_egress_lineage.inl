// Durable resident public-action identity; no transport field enters it.

BCC32_CAUSAL_GERMLINE_DISPATCH bool revision_egress_trajectory_word_at(
    const ResidentRewriteState* state, std::uint32_t owner,
    std::uint32_t index, std::uint32_t* word) {
  if (state == nullptr || word == nullptr) return false;
  const std::uint32_t page = index / kTrajectoryPageEvents;
  const std::uint32_t local = index % kTrajectoryPageEvents;
  std::uint32_t term_owner = owner;
  if (page != 0u) {
    std::uint32_t page_slot = kInvalid;
    for (std::uint32_t slot = 0u; slot < live_record_capacity(state);
         ++slot) {
      const Record& continuation = state->records[slot];
      if (continuation.matter_q8 == 0u ||
          continuation.lane[0] != kFormTrajectoryPage ||
          continuation.lane[1] != owner || continuation.lane[2] != page)
        continue;
      if (page_slot != kInvalid) return false;
      page_slot = slot;
    }
    if (page_slot == kInvalid) return false;
    const Record& continuation = state->records[page_slot];
    if (continuation.lane[3] != page * kTrajectoryPageEvents ||
        continuation.lane[4] <= local ||
        continuation.lane[4] > kTrajectoryPageEvents ||
        continuation.lane[6] == 0u || continuation.lane[6] == kInvalid)
      return false;
    term_owner = continuation.lane[6];
  }
  std::uint32_t term_slot = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& term = state->records[slot];
    if (term.matter_q8 == 0u || term.lane[0] != kFormTrajectoryTerm ||
        term.lane[1] != term_owner || term.lane[2] != local / 2u)
      continue;
    if (term_slot != kInvalid) return false;
    term_slot = slot;
  }
  if (term_slot == kInvalid) return false;
  *word = state->records[term_slot].lane[4u + (local % 2u)];
  return true;
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool revision_egress_witness_authoritative(
    const ResidentRewriteState* state, const Record& issued,
    bool consumed) {
  if (state == nullptr || issued.lane[0] != kFormRevisionActionIssuance)
    return false;
  std::uint32_t egress_slot = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& egress = state->records[slot];
    if (egress.matter_q8 == 0u ||
        egress.lane[0] != kFormRevisionEgressWitness ||
        egress.lane[1] != issued.lane[1])
      continue;
    if (egress_slot != kInvalid) return false;
    egress_slot = slot;
  }
  if (egress_slot == kInvalid) return false;
  const Record& egress = state->records[egress_slot];
  if (egress.lane[2] == 0u || egress.lane[2] == kInvalid ||
      egress.lane[3] == 0u || egress.lane[4] == 0u ||
      egress.lane[5] != issued.lane[3] ||
      egress.lane[6] != issued.lane[2] ||
      egress.lane[7] != (consumed ? kCausalGermlineExternal : 0u) ||
      egress.reserved[0] != issued.lane[4] ||
      egress.reserved[1] != issued.lane[5])
    return false;
  std::uint32_t trajectory_slot = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& trajectory = state->records[slot];
    if (trajectory.matter_q8 == 0u || trajectory.lane[0] != kFormTrajectory ||
        trajectory.lane[1] != egress.lane[2])
      continue;
    if (trajectory_slot != kInvalid) return false;
    trajectory_slot = slot;
  }
  if (trajectory_slot == kInvalid) return false;
  const Record& trajectory = state->records[trajectory_slot];
  if (trajectory.lane[2] != egress.lane[4] ||
      trajectory.lane[3] != kRevisionEgressRetainedTrajectory ||
      trajectory.revision != egress.lane[3])
    return false;
  for (std::uint32_t index = 0u; index < trajectory.lane[2]; ++index) {
    std::uint32_t word = 0u;
    if (!revision_egress_trajectory_word_at(state, trajectory.lane[1], index,
                                            &word) ||
        (index + 1u == trajectory.lane[2] && word != issued.lane[3]))
      return false;
  }
  return true;
}

BCC32_CAUSAL_GERMLINE_DISPATCH Record* revision_egress_witness(
    ResidentRewriteState* state, const Record& issued) {
  if (state == nullptr) return nullptr;
  Record* selected = nullptr;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    Record& egress = state->records[slot];
    if (egress.matter_q8 == 0u ||
        egress.lane[0] != kFormRevisionEgressWitness ||
        egress.lane[1] != issued.lane[1])
      continue;
    if (selected != nullptr) return nullptr;
    selected = &egress;
  }
  return selected;
}
