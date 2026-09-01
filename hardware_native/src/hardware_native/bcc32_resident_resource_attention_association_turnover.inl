// Matter-paid association capacity and turnover.  This owner preserves the
// exact external/provenance checks and finite-matter accounting of the parent;
// it creates no writer, answer selection, or new semantic authority.

BCC32_RESOURCE_ATTENTION_HD inline ResidentAssociationCapacityReceipt
read_association_capacity(const rewrite::ResidentRewriteState& world) {
  ResidentAssociationCapacityReceipt receipt{};
  receipt.resident_revision = world.revision;
  bool exact = world.fault == 0u;
  for (std::uint32_t slot = 0u; slot < rewrite::live_record_capacity(&world);
       ++slot) {
    const rewrite::Record& record = world.records[slot];
    if (participation::is_participation(record)) {
      ++receipt.retained_fragments;
      receipt.retained_matter_q8 += record.matter_q8;
      exact &= record.matter_q8 == rewrite::kRecordMatterQ8;
    } else if (record.matter_q8 != 0u &&
               record.lane[0] == rewrite::kFormEmpty) {
      ++receipt.free_records;
      receipt.free_matter_q8 += record.matter_q8;
      exact &= record.matter_q8 == rewrite::kRecordMatterQ8;
    }
  }
  for (std::uint32_t slot = 0u; slot < rewrite::live_record_capacity(&world);
       ++slot) {
    const rewrite::Record& antecedent = world.records[slot];
    if (!participation::is_participation(antecedent) ||
        antecedent.lane[1] != participation::kAntecedentFragment)
      continue;
    std::uint32_t peers = 0u;
    const rewrite::Record* consequent = nullptr;
    for (std::uint32_t peer_slot = 0u;
         peer_slot < rewrite::live_record_capacity(&world); ++peer_slot) {
      const rewrite::Record& peer = world.records[peer_slot];
      if (!participation::is_participation(peer) ||
          peer.lane[1] != participation::kConsequentFragment ||
          peer.lane[2] != antecedent.lane[2] ||
          peer.lane[4] != antecedent.lane[4])
        continue;
      consequent = &peer;
      ++peers;
    }
    if (peers != 1u || consequent == nullptr ||
        consequent->lane[5] != antecedent.lane[5] ||
        consequent->lane[6] != antecedent.lane[6]) {
      exact = false;
      continue;
    }
    ++receipt.complete_sources;
    receipt.positive_sources += antecedent.lane[5] == 1u ? 1u : 0u;
    receipt.defeated_sources += antecedent.lane[6] == 1u ? 1u : 0u;
  }
  exact &= receipt.retained_fragments == receipt.complete_sources * 2u;
  exact &= receipt.retained_matter_q8 ==
           static_cast<std::uint64_t>(receipt.retained_fragments) *
               rewrite::kRecordMatterQ8;
  receipt.exact = exact ? 1u : 0u;
  return receipt;
}
BCC32_RESOURCE_ATTENTION_HD inline bool complete_positive_source(
    const rewrite::ResidentRewriteState& world, std::uint32_t source,
    std::uint32_t* antecedent_slot, std::uint32_t* consequent_slot,
    std::uint32_t* before, std::uint32_t* relation, std::uint32_t* after) {
  if (source == 0u || source == rewrite::kInvalid) return false;
  std::uint32_t antecedents = 0u;
  std::uint32_t consequents = 0u;
  std::uint32_t a = rewrite::kInvalid;
  std::uint32_t c = rewrite::kInvalid;
  for (std::uint32_t slot = 0u; slot < rewrite::live_record_capacity(&world);
       ++slot) {
    const rewrite::Record& record = world.records[slot];
    if (!participation::is_participation(record) ||
        record.lane[4] != source || record.lane[5] != 1u ||
        record.lane[6] != 0u)
      continue;
    if (record.lane[1] == participation::kAntecedentFragment) {
      a = slot;
      ++antecedents;
    } else {
      c = slot;
      ++consequents;
    }
  }
  if (antecedents != 1u || consequents != 1u ||
      world.records[a].lane[2] != world.records[c].lane[2])
    return false;
  if (antecedent_slot != nullptr) *antecedent_slot = a;
  if (consequent_slot != nullptr) *consequent_slot = c;
  if (before != nullptr) *before = world.records[a].lane[3];
  if (relation != nullptr) *relation = world.records[a].lane[2];
  if (after != nullptr) *after = world.records[c].lane[3];
  return true;
}

BCC32_RESOURCE_ATTENTION_HD inline std::uint32_t association_source_support(
    const rewrite::ResidentRewriteState& world, std::uint32_t before,
    std::uint32_t relation, std::uint32_t after) {
  std::uint32_t support = 0u;
  for (std::uint32_t slot = 0u; slot < rewrite::live_record_capacity(&world);
       ++slot) {
    const rewrite::Record& antecedent = world.records[slot];
    if (!participation::is_participation(antecedent) ||
        antecedent.lane[1] != participation::kAntecedentFragment ||
        antecedent.lane[2] != relation || antecedent.lane[3] != before ||
        antecedent.lane[5] != 1u)
      continue;
    std::uint32_t ignored_a = 0u;
    std::uint32_t ignored_c = 0u;
    std::uint32_t observed_before = 0u;
    std::uint32_t observed_relation = 0u;
    std::uint32_t observed_after = 0u;
    if (complete_positive_source(
            world, antecedent.lane[4], &ignored_a, &ignored_c,
            &observed_before, &observed_relation, &observed_after) &&
        observed_before == before && observed_relation == relation &&
        observed_after == after)
      ++support;
  }
  return support;
}

BCC32_RESOURCE_ATTENTION_HD inline bool weaker_association_source(
    std::uint32_t candidate_support, std::uint32_t candidate_before,
    std::uint32_t candidate_relation, std::uint32_t candidate_after,
    std::uint32_t candidate_source, std::uint32_t incumbent_support,
    std::uint32_t incumbent_before, std::uint32_t incumbent_relation,
    std::uint32_t incumbent_after, std::uint32_t incumbent_source) {
  if (candidate_support != incumbent_support)
    return candidate_support < incumbent_support;
  if (candidate_before != incumbent_before)
    return candidate_before < incumbent_before;
  if (candidate_relation != incumbent_relation)
    return candidate_relation < incumbent_relation;
  if (candidate_after != incumbent_after)
    return candidate_after < incumbent_after;
  return candidate_source < incumbent_source;
}

// Pressure retires a complete source from the least-supported association.
BCC32_RESOURCE_ATTENTION_HD inline bool
retire_causally_weakest_association_source(
    rewrite::ResidentRewriteState* world, std::uint32_t* retired_source,
    std::uint32_t* retired_support, std::uint32_t* retired_before = nullptr,
    std::uint32_t* retired_relation = nullptr,
    std::uint32_t* retired_after = nullptr) {
  if (world == nullptr || world->fault != 0u ||
      read_association_capacity(*world).exact != 1u)
    return false;
  bool found = false;
  std::uint32_t selected_source = rewrite::kInvalid;
  std::uint32_t selected_support = 0u;
  std::uint32_t selected_before = 0u;
  std::uint32_t selected_relation = 0u;
  std::uint32_t selected_after = 0u;
  std::uint32_t selected_a = rewrite::kInvalid;
  std::uint32_t selected_c = rewrite::kInvalid;
  for (std::uint32_t slot = 0u; slot < rewrite::live_record_capacity(world);
       ++slot) {
    const rewrite::Record& antecedent = world->records[slot];
    if (!participation::is_participation(antecedent) ||
        antecedent.lane[1] != participation::kAntecedentFragment ||
        antecedent.lane[5] != 1u)
      continue;
    std::uint32_t a = rewrite::kInvalid;
    std::uint32_t c = rewrite::kInvalid;
    std::uint32_t before = 0u;
    std::uint32_t relation = 0u;
    std::uint32_t after = 0u;
    if (!complete_positive_source(*world, antecedent.lane[4], &a, &c,
                                  &before, &relation, &after))
      return false;
    const std::uint32_t support =
        association_source_support(*world, before, relation, after);
    // Redundant sources are individually zero-authority.
    if (support <= 2u) continue;
    if (!found || weaker_association_source(
                      support, before, relation, after, antecedent.lane[4],
                      selected_support, selected_before, selected_relation,
                      selected_after, selected_source)) {
      found = true;
      selected_source = antecedent.lane[4];
      selected_support = support;
      selected_before = before;
      selected_relation = relation;
      selected_after = after;
      selected_a = a;
      selected_c = c;
    }
  }
  if (!found || selected_a == rewrite::kInvalid ||
      selected_c == rewrite::kInvalid)
    return false;
  const rewrite::Record before_a = world->records[selected_a];
  const rewrite::Record before_c = world->records[selected_c];
  const std::uint64_t before_revision = world->revision;
  if (!rewrite::matter_account_is_closed(*world)) return false;
  rewrite::clear_record(&world->records[selected_a]);
  rewrite::clear_record(&world->records[selected_c]);
  world->revision = world->revision == ~std::uint64_t{0}
                        ? world->revision
                        : world->revision + 1u;
  if (!rewrite::matter_account_is_closed(*world) ||
      !participation::component_survives_single_record_cuts(
          world, selected_before, selected_relation, selected_after)) {
    world->records[selected_a] = before_a;
    world->records[selected_c] = before_c;
    world->revision = before_revision;
    return false;
  }
  if (retired_source != nullptr) *retired_source = selected_source;
  if (retired_support != nullptr) *retired_support = selected_support;
  if (retired_before != nullptr) *retired_before = selected_before;
  if (retired_relation != nullptr) *retired_relation = selected_relation;
  if (retired_after != nullptr) *retired_after = selected_after;
  return true;
}

BCC32_RESOURCE_ATTENTION_HD inline bool paid_external_triplet_identity(
    rewrite::ResidentRewriteState* world, std::uint32_t source_slot,
    std::uint32_t* source, std::uint32_t* before,
    std::uint32_t* relation, std::uint32_t* after,
    std::uint32_t* required_fragments) {
  if (world == nullptr || source == nullptr || before == nullptr ||
      relation == nullptr || after == nullptr ||
      required_fragments == nullptr ||
      source_slot >= rewrite::live_record_capacity(world))
    return false;
  const rewrite::Record& record = world->records[source_slot];
  if (record.matter_q8 == 0u ||
      record.lane[0] != rewrite::kFormTrajectory || record.lane[2] != 3u ||
      record.lane[3] != 0u || record.lane[4] != 0u ||
      record.lane[5] != rewrite::kInvalid || record.lane[7] != 0u ||
      record.lane[1] == 0u || record.lane[1] == rewrite::kInvalid ||
      !rewrite::mixed_provenance::tagged_history(world, record))
    return false;
  std::uint32_t value[3]{};
  for (std::uint32_t index = 0u; index < 3u; ++index) {
    rewrite::mixed_provenance::Origin origin{};
    std::uint32_t producer = rewrite::kInvalid;
    if (!rewrite::mixed_provenance::origin_at(
            world, record, index, &origin, &producer) ||
        origin != rewrite::mixed_provenance::Origin::external ||
        producer != rewrite::kInvalid ||
        !rewrite::trajectory_word_at(world, record.lane[1], index,
                                     &value[index]) ||
        value[index] == 0u || value[index] == rewrite::kInvalid)
      return false;
  }
  std::uint32_t found = 0u;
  for (std::uint32_t fragment = participation::kAntecedentFragment;
       fragment <= participation::kConsequentFragment; ++fragment) {
    std::uint32_t matches = 0u;
    for (std::uint32_t slot = 0u; slot < rewrite::live_record_capacity(world);
         ++slot) {
      const rewrite::Record& resident = world->records[slot];
      if (!participation::is_participation(resident) ||
          resident.lane[4] != record.lane[1] ||
          resident.lane[1] != fragment)
        continue;
      const std::uint32_t endpoint =
          fragment == participation::kAntecedentFragment ? value[0]
                                                         : value[2];
      if (resident.lane[2] != value[1] || resident.lane[3] != endpoint ||
          resident.lane[5] != 1u || resident.lane[6] != 0u)
        return false;
      ++matches;
    }
    if (matches > 1u) return false;
    found += matches;
  }
  *source = record.lane[1];
  *before = value[0];
  *relation = value[1];
  *after = value[2];
  *required_fragments = 2u - found;
  return true;
}

// END recurrence pays two Records; fresh recurrence can regrow turnover.
BCC32_RESOURCE_ATTENTION_HD inline MatterPaidAssociationAssimilationReceipt
assimilate_matter_paid_external_association_at_end(
    rewrite::ResidentRewriteState* world, std::uint32_t source_slot) {
  MatterPaidAssociationAssimilationReceipt receipt{};
  receipt.assimilation.rejected = 1u;
  if (world == nullptr) return receipt;
  receipt.pre_capacity = read_association_capacity(*world);
  std::uint32_t source = rewrite::kInvalid;
  std::uint32_t before = 0u;
  std::uint32_t relation = 0u;
  std::uint32_t after = 0u;
  if (world->fault != 0u || !rewrite::matter_account_is_closed(*world) ||
      !paid_external_triplet_identity(
          world, source_slot, &source, &before, &relation, &after,
          &receipt.required_fragments))
    return receipt;

  receipt.assimilation =
      participation::assimilate_completed_external_trajectory_at_end(
          world, source_slot);
  if (receipt.assimilation.rejected != 0u &&
      receipt.required_fragments != 0u) {
    if (!retire_causally_weakest_association_source(
            world, &receipt.retired_source,
            &receipt.retired_association_support, &receipt.retired_before,
            &receipt.retired_relation, &receipt.retired_after))
      return receipt;
    receipt.retired_fragments = 2u;
    receipt.pressure_turnover = 1u;
    receipt.assimilation =
        participation::assimilate_completed_external_trajectory_at_end(
            world, source_slot);
  }
  receipt.post_capacity = read_association_capacity(*world);
  if (receipt.assimilation.rejected == 0u &&
      receipt.assimilation.admitted == receipt.required_fragments &&
      receipt.post_capacity.exact == 1u &&
      receipt.post_capacity.retained_fragments + receipt.retired_fragments ==
          receipt.pre_capacity.retained_fragments +
              receipt.assimilation.admitted &&
      receipt.pressure_turnover != 0u &&
      source != receipt.retired_source && before == receipt.retired_before &&
      relation == receipt.retired_relation && after == receipt.retired_after)
    receipt.fresh_recurrence_regrowth = 1u;
  receipt.remote_readout_preserved =
      receipt.pressure_turnover != 0u &&
              receipt.retired_association_support > 2u
          ? 1u
          : 0u;
  receipt.matter_closed = rewrite::matter_account_is_closed(*world) ? 1u : 0u;
  return receipt;
}

// Production physical-END analogue of
// assimilate_matter_paid_external_association_at_end(), wrapping the
// resumable bounded-work windows adapter instead of the single-shot
// compatibility seam (0X1-207). The windows adapter remains the sole
// execution authority: this wrapper never resynchronously rescans a
// trajectory and never converts an ordinary work_limit stop into pressure.
// Turnover is authorized only by an explicit allocator-pressure rejection
// (AssimilationReceipt::pressure_rejected) on one complete attempted window;
// every other rejection reason (malformed input, provenance mismatch)
// remains fail-closed exactly as the underlying adapter already treats it.
BCC32_RESOURCE_ATTENTION_HD inline MatterPaidAssociationAssimilationReceipt
assimilate_matter_paid_external_association_windows(
    rewrite::ResidentRewriteState* world, std::uint32_t source_slot,
    std::uint32_t* next_window, std::uint32_t work_limit,
    bool prevalidated = false) {
  MatterPaidAssociationAssimilationReceipt receipt{};
  receipt.assimilation.rejected = 1u;
  if (world == nullptr) return receipt;
  receipt.pre_capacity = read_association_capacity(*world);
  if (world->fault != 0u || next_window == nullptr || work_limit == 0u ||
      source_slot >= rewrite::live_record_capacity(world) ||
      !rewrite::matter_account_is_closed(*world)) {
    receipt.post_capacity = read_association_capacity(*world);
    receipt.matter_closed = rewrite::matter_account_is_closed(*world) ? 1u : 0u;
    return receipt;
  }

  receipt.assimilation =
      participation::assimilate_completed_external_trajectory_windows(
          world, source_slot, next_window, work_limit, prevalidated);
  if (receipt.assimilation.rejected == 0u ||
      receipt.assimilation.pressure_rejected == 0u) {
    // Either succeeded (possibly a legal partial/incomplete resumable
    // stride), or rejected for a non-pressure reason -- neither authorizes
    // turnover.
    receipt.post_capacity = read_association_capacity(*world);
    receipt.matter_closed = rewrite::matter_account_is_closed(*world) ? 1u : 0u;
    return receipt;
  }

  // Genuine allocator pressure on a complete attempted window. The windows
  // adapter does not advance *next_window past a rejected window, so the
  // cursor is already parked exactly on the failed window -- retry from it
  // directly rather than restoring any earlier/entry cursor, which would
  // replay already-admitted windows.
  if (!retire_causally_weakest_association_source(
          world, &receipt.retired_source,
          &receipt.retired_association_support, &receipt.retired_before,
          &receipt.retired_relation, &receipt.retired_after)) {
    receipt.post_capacity = read_association_capacity(*world);
    receipt.matter_closed = rewrite::matter_account_is_closed(*world) ? 1u : 0u;
    return receipt;
  }
  receipt.retired_fragments = 2u;
  receipt.pressure_turnover = 1u;

  // Charge the retry only the residual budget so this wrapper never grants a
  // second full work_limit for one bounded resident epoch.
  const std::uint32_t retry_limit =
      receipt.assimilation.work_consumed < work_limit
          ? work_limit - receipt.assimilation.work_consumed
          : 0u;
  if (retry_limit != 0u) {
    receipt.assimilation =
        participation::assimilate_completed_external_trajectory_windows(
            world, source_slot, next_window, retry_limit, prevalidated);
  } else {
    // The budget was exhausted reaching this epoch's pressure rejection.
    // Retirement already resolved the pressure condition physically; the
    // failed window remains parked at *next_window for the next resident
    // epoch to retry. That is resumable incompletion, not a fatal rejection.
    receipt.assimilation.rejected = 0u;
    receipt.assimilation.pressure_rejected = 0u;
  }

  receipt.post_capacity = read_association_capacity(*world);
  receipt.matter_closed = rewrite::matter_account_is_closed(*world) ? 1u : 0u;
  receipt.remote_readout_preserved =
      receipt.retired_association_support > 2u ? 1u : 0u;
  return receipt;
}
