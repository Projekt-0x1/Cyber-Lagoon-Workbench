// Included inside direct_adult_legacy_oracle.cu's anonymous namespace after
// direct_adult_membrane_ingress.cuh. Owns the eligibility ledger lifecycle:
// batch admission, credit reset/settle/finalize, strict bridge-ticket arming,
// exact episode settlement, deterministic duplicate collapse, compaction with
// exactly-once matter release, expiry and omission credit. One reason to
// change: the settlement and credit law itself.
constexpr std::int32_t kConductanceFloorQ16 = 1 << 12;
constexpr std::int32_t kPositiveCreditQ16 = 1 << 13;
constexpr std::int32_t kNegativeCreditQ16 = -(1 << 13);
constexpr std::int32_t kOmissionCreditQ16 = -(1 << 12);

__host__ __device__ inline std::int32_t clamp_conductance(std::int64_t value) {
  if (value < kConductanceFloorQ16)
    return kConductanceFloorQ16;
  if (value > 4ll * kConductanceOneQ16)
    return static_cast<std::int32_t>(4ll * kConductanceOneQ16);
  return static_cast<std::int32_t>(value);
}

// 0X1-1176: this gate used to reserve a worst case of
// `frontier_work * kEligibilityLaunchExpansion` -- 16x the entire frontier
// allocation -- before admitting ANY new eligibility record. Two things make
// that fatal rather than merely conservative.
//
// First, `kEligibilityLaunchExpansion` is declared "Scheduling factor only; not
// semantic fanout": it sizes launches, and no tick creates 16 records per
// frontier entry. Second, `frontier_work` is a launch bound that ratchets to
// `frontier_capacity` and cannot fall, so once the frontier saturates the
// demand term is pinned at 16 * frontier_capacity forever. On the reference
// adult that is 65536 against a bank whose automatic capacity is itself capped
// by the route capacity, so `admit` latched to 0 and never returned: measured,
// the eligibility bank drains to ZERO live records by tick 8 and stays there for
// the rest of the organism's life, which ends exact causal learning permanently.
// Raising the bank made no difference at 256, 4096 or 16384 -- the same cliff at
// tick 8 -- because the demand term does not depend on the bank at all.
//
// Overflow safety does not need this gate and never did.
// `append_eligibility_record` already fails closed twice per record, and counts
// both refusals in `eligibility_capacity_rejects`: the resource-ecology ledger
// reservation ("the ledger is the authority and must be able to refuse an
// admission the bank would have accepted") and the bank's own
// `slot >= capacity` bound, which cancels the reservation and undoes the
// `atomicAdd`. What is left here is the honest batch-level question -- is there
// any room at all -- against a quantity that CAN fall, because compaction drops
// settled and expired records every tick.
__global__ void decide_eligibility_batch_admission_kernel(const std::uint32_t* live_count,
                                                          std::uint32_t capacity,
                                                          std::uint32_t* admit) {
  if (blockIdx.x == 0u && threadIdx.x == 0u) {
    *admit = *live_count < capacity ? 1u : 0u;
  }
}

__global__ void reset_active_route_credit_kernel(DirectBrainV01 brain,
                                                 const DirectEligibilityRecord* eligibility_bank,
                                                 const std::uint32_t* eligibility_count,
                                                 std::uint32_t eligibility_capacity) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= min(*eligibility_count, eligibility_capacity))
    return;
  const DirectEligibilityRecord record = eligibility_bank[i];
  if (record.route.slot >= brain.route_capacity)
    return;
  const DirectRouteSlotMeta meta = brain.topology.slot_meta[record.route.slot];
  if (meta.live == 0u || meta.generation != record.route.generation)
    return;
  // Several live causal records may reference one route. All writers store the
  // same zero before settlement starts, so this reset is schedule-independent.
  brain.routes[record.route.slot].last_credit_q16 = 0;
}

__global__ void finalize_active_route_credit_kernel(DirectBrainV01 brain,
                                                    const DirectEligibilityRecord* eligibility_bank,
                                                    const std::uint32_t* eligibility_count,
                                                    std::uint32_t eligibility_capacity,
                                                    DirectTopologyDeviceView topology,
                                                    std::uint32_t proposal_base) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= min(*eligibility_count, eligibility_capacity))
    return;
  const DirectEligibilityRecord record = eligibility_bank[i];
  if (record.route.slot >= brain.route_capacity)
    return;
  const DirectRouteSlotMeta meta = brain.topology.slot_meta[record.route.slot];
  if (meta.live == 0u || meta.generation != record.route.generation)
    return;
  DirectRoute& route = brain.routes[record.route.slot];
  // Settlement and omission kernels have completed in the same CUDA stream.
  // Duplicate route references therefore observe the same final net credit and
  // conductance; duplicate retract proposals are removed by topology arbitration.
  route.conductance_q16 = clamp_conductance(route.conductance_q16);
  if (route.last_credit_q16 < 0 && route.conductance_q16 <= kConductanceFloorQ16) {
    submit_direct_retract_proposal(topology, proposal_base + i, route.source, record.route.slot,
                                   meta.generation, -route.last_credit_q16);
  }
}

// #1184: the bridge claims its exact live eligibility records here.  An armed
// record has no authorization window: apply_return_credit_kernel admits only
// the matching motor-reafferent frontier entry whose parallel authority sidecar
// was written by the bridge's private ingress path.
__global__ void mark_bridge_ticket_strict_kernel(
    DirectEligibilityRecord* eligibility_bank, const std::uint32_t* eligibility_count,
    std::uint32_t eligibility_capacity, const std::uint32_t* bucket_heads,
    std::uint32_t bucket_count, const BridgeTicketMark* marks, std::uint32_t mark_count) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= mark_count)
    return;
  const BridgeTicketMark mark = marks[i];
  const std::uint32_t active = min(*eligibility_count, eligibility_capacity);
  const std::uint32_t bucket =
      eligibility_bucket_slot(mark.cue_node, mark.context, 0u, bucket_count);
  std::uint32_t cursor = bucket_heads[bucket];
  for (std::uint32_t probes = 0u; cursor != kInvalidIndex && probes < active; ++probes) {
    DirectEligibilityRecord& record = eligibility_bank[cursor];
    const std::uint32_t next = record.next_in_bucket;
    if (record.state == EligibilityRecordState::live && record.source == mark.cue_node &&
        record.context == mark.context && record.ticket == mark.ticket) {
      record.require_motor_reafference = kBridgeSettlementArmed;
    }
    cursor = next;
  }
}

__global__ void apply_return_credit_kernel(
    DirectBrainV01 brain, DirectTopologyDeviceView topology, const ActivityEvent* frontier,
    const DirectIngressAuthority* frontier_authority, const std::uint32_t* frontier_count,
    DirectEligibilityRecord* eligibility_bank, const std::uint32_t* eligibility_count,
    std::uint32_t eligibility_capacity, const std::uint32_t* bucket_heads,
    std::uint32_t bucket_count, std::uint32_t tick, AdultCounters* counters,
    DirectRepresentationDeviceView rep, std::uint32_t frontier_capacity) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  // 0X1-1175: this is where the birth starvation actually bit, and the loss was
  // exact causal learning, not throughput. This kernel closes #1176 eligibility
  // episodes, and its only frontier bound was `*frontier_count` plus whatever
  // grid it was launched with -- which was `grid_for(frontier_work)`. While
  // `frontier_launch_bound` climbs from 0, the resident frontier is already
  // larger than it (one tick's propagation returns about three successors per
  // event consumed), so every event past the bound had its return skipped, and
  // the frontier buffers swap at the end of the step, so it is never retried.
  // Measured on tick 2 of a 4x4096-node adult: grid at the bound settles ZERO
  // episodes, grid at the allocation settles 512.
  //
  // `*frontier_count` is the atomicAdd attempt counter and can exceed the
  // buffer, so the capacity half of this guard is stated here rather than left
  // implicit in the launch geometry.
  if (i >= *frontier_count || i >= frontier_capacity)
    return;
  const ActivityEvent event = frontier[i];
  const DirectIngressAuthority authority =
      frontier_authority != nullptr ? frontier_authority[i] : DirectIngressAuthority::ordinary;
  const bool chronological_external =
      event.origin == CausalOrigin::external_contact && event.cue_node != kInvalidIndex;
  if (!chronological_external && event.origin != CausalOrigin::world_return &&
      event.origin != CausalOrigin::motor_reafference)
    return;
  if (event.external_root == 0u || event.cue_node >= brain.node_count ||
      event.node >= brain.node_count) {
    atomicAdd(&counters->ignored_returns, 1u);
    return;
  }

  const std::uint32_t active = min(*eligibility_count, eligibility_capacity);
  const std::uint32_t bucket =
      eligibility_bucket_slot(event.cue_node, event.context, 0u, bucket_count);
  std::uint32_t cursor = bucket_heads[bucket];
  std::uint64_t first_ticket = 0u;
  bool have_ticket = false;
  bool distinct_ticket = false;
  bool explicit_ticket_match = false;
  bool saw_coarse_candidate = false;
  // #1184: an event that would otherwise have matched a bridge-claimed
  // (outstanding action ticket) episode -- by exact ticket coincidence or the
  // coarse fallback below -- but was refused because the record is armed and
  // this event carries no live bridge authorization. Set for every origin,
  // including a host-stamped CausalOrigin::motor_reafference: asserting the
  // origin is not the same as having traversed the bridge.
  bool saw_guarded_motor_candidate = false;

  // Pass 1 selects one exact episode identity. Multiple route records may share
  // that identity; unrelated unresolved episodes may not be collapsed by the
  // coarse context hash.
  for (std::uint32_t probes = 0u; cursor != kInvalidIndex && probes < active; ++probes) {
    atomicAdd(&counters->eligibility_index_probes, 1u);
    const DirectEligibilityRecord record = eligibility_bank[cursor];
    const std::uint32_t next = record.next_in_bucket;
    // A new chronological external contact carries no return ticket. It may
    // close the unique live (source, context) episode; the distinct-ticket pass
    // below fails closed when more than one unresolved lineage is eligible.
    // Motor reafference has an exact causal-root ticket. Explicit world returns
    // may additionally use full resident history for exact disambiguation.
    // #1166: settlement is deep-context discriminated. This read
    // `... || chronological_external || (... record.history_signature ==
    // event.history_signature)`, and `chronological_external` is true for every
    // external contact carrying a cue node, so the comparison beside it never
    // ran. It could not have run usefully anyway -- the record holds a MIXED
    // signature of the source event while the contact carries a RAW packed one.
    // `successor_history_signature` is the record's prediction of exactly the
    // value the settling contact carries, so the two are commensurable by
    // construction. Either side being zero fails OPEN, which preserves the
    // previous behaviour for any record or contact that cannot supply it.
    // The low half of a chronological signature is
    // `direct_canonical_predecessor_signature(source_node, source_word)` -- it
    // names the contact this record was minted at. It is therefore a VALIDITY
    // CHECK on the quantity, not part of the discrimination: if it does not
    // match, the arriving contact's history is not describing this record's
    // source at all, and the deep halves are not comparable. That is the case
    // for any caller that writes `runtime.frontier` directly instead of going
    // through `append_one_event` -- several contracts drive the adult that way
    // with a synthesised `history_signature` -- and those must keep the
    // pre-#1166 behaviour exactly. Only when the shallow halves agree is the
    // deep half a context both sides computed the same way.
    const bool successor_names_this_source =
        record.successor_history_signature != 0u &&
        static_cast<std::uint32_t>(record.successor_history_signature) ==
            static_cast<std::uint32_t>(event.history_signature);
    const bool history_ok = event.origin == CausalOrigin::motor_reafference ||
                            !successor_names_this_source ||
                            record.successor_history_signature == event.history_signature;
    if (record.state == EligibilityRecordState::live && record.expiry_tick >= tick &&
        record.source == event.cue_node && record.context == event.context && history_ok) {
      // #1179: a record backed by state-owning implicit storage settles
      // through the exact same authority as one still backed by its
      // explicit route -- eligibility_locator_live resolves whichever
      // locator.kind the record currently carries.
      if (eligibility_locator_live(brain, rep, record)) {
        // #1184 authority boundary: no injected event may close a
        // bridge-claimed pending action episode without a live bridge
        // authorization, regardless of exact-ticket coincidence, the coarse
        // unique-live-episode fallback, or the CausalOrigin it asserts -- see
        // mark_bridge_ticket_strict_kernel's file-header rationale.
        if (bridge_settlement_refused(record.require_motor_reafference, event.origin, authority)) {
          saw_guarded_motor_candidate = true;
        } else {
          saw_coarse_candidate = true;
          explicit_ticket_match = explicit_ticket_match || record.ticket == event.external_root;
          if (!have_ticket) {
            first_ticket = record.ticket;
            have_ticket = true;
          } else if (record.ticket != first_ticket) {
            distinct_ticket = true;
          }
        }
      }
    }
    cursor = next;
  }

  std::uint64_t selected_ticket = 0u;
  if (event.origin == CausalOrigin::motor_reafference) {
    if (!explicit_ticket_match) {
      // #1184 slice 3 residual: a host-stamped motor_reafference that was
      // refused by an armed record leaves this way, so the spoof trace must be
      // raised here too -- otherwise the rejection would be silent and the
      // falsifier's "resident-observable trace" check would be untestable.
      if (saw_guarded_motor_candidate)
        atomicAdd(&counters->eligibility_motor_action_spoof_rejects, 1u);
      if (saw_coarse_candidate)
        atomicAdd(&counters->eligibility_ticket_rejects, 1u);
      atomicAdd(&counters->ignored_returns, 1u);
      return;
    }
    selected_ticket = event.external_root;
  } else if (explicit_ticket_match) {
    // A world-return membrane may carry the originating causal root directly.
    selected_ticket = event.external_root;
  } else {
    if (!have_ticket) {
      if (saw_guarded_motor_candidate)
        atomicAdd(&counters->eligibility_motor_action_spoof_rejects, 1u);
      atomicAdd(&counters->ignored_returns, 1u);
      return;
    }
    if (distinct_ticket) {
      atomicAdd(&counters->eligibility_ambiguous_context_rejects, 1u);
      atomicAdd(&counters->ignored_returns, 1u);
      return;
    }
    selected_ticket = first_ticket;
  }

  // Find the canonical leader for this exact episode. Claiming one leader before
  // any route credit makes copied/duplicated returns a single closure event even
  // when several physical routes participated in the episode.
  cursor = bucket_heads[bucket];
  std::uint32_t leader = kInvalidIndex;
  for (std::uint32_t probes = 0u; cursor != kInvalidIndex && probes < active; ++probes) {
    const DirectEligibilityRecord record = eligibility_bank[cursor];
    const std::uint32_t next = record.next_in_bucket;
    // A new chronological external contact carries no return ticket. It may
    // close the unique live (source, context) episode; the distinct-ticket pass
    // below fails closed when more than one unresolved lineage is eligible.
    // Motor reafference has an exact causal-root ticket. Explicit world returns
    // may additionally use full resident history for exact disambiguation.
    // #1166: settlement is deep-context discriminated. This read
    // `... || chronological_external || (... record.history_signature ==
    // event.history_signature)`, and `chronological_external` is true for every
    // external contact carrying a cue node, so the comparison beside it never
    // ran. It could not have run usefully anyway -- the record holds a MIXED
    // signature of the source event while the contact carries a RAW packed one.
    // `successor_history_signature` is the record's prediction of exactly the
    // value the settling contact carries, so the two are commensurable by
    // construction. Either side being zero fails OPEN, which preserves the
    // previous behaviour for any record or contact that cannot supply it.
    // The low half of a chronological signature is
    // `direct_canonical_predecessor_signature(source_node, source_word)` -- it
    // names the contact this record was minted at. It is therefore a VALIDITY
    // CHECK on the quantity, not part of the discrimination: if it does not
    // match, the arriving contact's history is not describing this record's
    // source at all, and the deep halves are not comparable. That is the case
    // for any caller that writes `runtime.frontier` directly instead of going
    // through `append_one_event` -- several contracts drive the adult that way
    // with a synthesised `history_signature` -- and those must keep the
    // pre-#1166 behaviour exactly. Only when the shallow halves agree is the
    // deep half a context both sides computed the same way.
    const bool successor_names_this_source =
        record.successor_history_signature != 0u &&
        static_cast<std::uint32_t>(record.successor_history_signature) ==
            static_cast<std::uint32_t>(event.history_signature);
    const bool history_ok = event.origin == CausalOrigin::motor_reafference ||
                            !successor_names_this_source ||
                            record.successor_history_signature == event.history_signature;
    if (record.state == EligibilityRecordState::live && record.expiry_tick >= tick &&
        record.source == event.cue_node && record.context == event.context && history_ok &&
        record.ticket == selected_ticket &&
        !bridge_settlement_refused(record.require_motor_reafference, event.origin, authority)) {
      if (eligibility_locator_live(brain, rep, record) &&
          (leader == kInvalidIndex || cursor < leader))
        leader = cursor;
    }
    cursor = next;
  }
  if (leader == kInvalidIndex) {
    if (saw_guarded_motor_candidate)
      atomicAdd(&counters->eligibility_motor_action_spoof_rejects, 1u);
    atomicAdd(&counters->ignored_returns, 1u);
    return;
  }
  DirectEligibilityRecord& leader_record = eligibility_bank[leader];
  if (atomicCAS(reinterpret_cast<unsigned int*>(&leader_record.state),
                static_cast<unsigned int>(EligibilityRecordState::live),
                static_cast<unsigned int>(EligibilityRecordState::settled)) !=
      static_cast<unsigned int>(EligibilityRecordState::live)) {
    atomicAdd(&counters->eligibility_duplicate_settlement_rejects, 1u);
    atomicAdd(&counters->ignored_returns, 1u);
    return;
  }

  const DirectNode actual = brain.nodes[event.node];
  const std::int32_t plasticity = static_cast<std::int32_t>(brain.development->plasticity_q16);
  bool matched_any = false;
  bool predicted_mismatch_any = false;
  bool predicted_mismatch_owned = false;
  std::uint32_t settled_records = 0u;
  std::uint64_t representative_history = 0u;
  std::uint32_t representative_target = kInvalidIndex;

  // Pass 2 settles every physical participant in the selected exact episode.
  cursor = bucket_heads[bucket];
  for (std::uint32_t probes = 0u; cursor != kInvalidIndex && probes < active; ++probes) {
    DirectEligibilityRecord& record = eligibility_bank[cursor];
    const std::uint32_t next = record.next_in_bucket;
    // A new chronological external contact carries no return ticket. It may
    // close the unique live (source, context) episode; the distinct-ticket pass
    // below fails closed when more than one unresolved lineage is eligible.
    // Motor reafference has an exact causal-root ticket. Explicit world returns
    // may additionally use full resident history for exact disambiguation.
    // #1166: settlement is deep-context discriminated. This read
    // `... || chronological_external || (... record.history_signature ==
    // event.history_signature)`, and `chronological_external` is true for every
    // external contact carrying a cue node, so the comparison beside it never
    // ran. It could not have run usefully anyway -- the record holds a MIXED
    // signature of the source event while the contact carries a RAW packed one.
    // `successor_history_signature` is the record's prediction of exactly the
    // value the settling contact carries, so the two are commensurable by
    // construction. Either side being zero fails OPEN, which preserves the
    // previous behaviour for any record or contact that cannot supply it.
    // The low half of a chronological signature is
    // `direct_canonical_predecessor_signature(source_node, source_word)` -- it
    // names the contact this record was minted at. It is therefore a VALIDITY
    // CHECK on the quantity, not part of the discrimination: if it does not
    // match, the arriving contact's history is not describing this record's
    // source at all, and the deep halves are not comparable. That is the case
    // for any caller that writes `runtime.frontier` directly instead of going
    // through `append_one_event` -- several contracts drive the adult that way
    // with a synthesised `history_signature` -- and those must keep the
    // pre-#1166 behaviour exactly. Only when the shallow halves agree is the
    // deep half a context both sides computed the same way.
    const bool successor_names_this_source =
        record.successor_history_signature != 0u &&
        static_cast<std::uint32_t>(record.successor_history_signature) ==
            static_cast<std::uint32_t>(event.history_signature);
    const bool history_ok = event.origin == CausalOrigin::motor_reafference ||
                            !successor_names_this_source ||
                            record.successor_history_signature == event.history_signature;
    const bool exact_group =
        record.expiry_tick >= tick && record.source == event.cue_node &&
        record.context == event.context && history_ok && record.ticket == selected_ticket &&
        !bridge_settlement_refused(record.require_motor_reafference, event.origin, authority);
    if (!exact_group || !eligibility_locator_live(brain, rep, record)) {
      cursor = next;
      continue;
    }
    if (cursor != leader && atomicCAS(reinterpret_cast<unsigned int*>(&record.state),
                                      static_cast<unsigned int>(EligibilityRecordState::live),
                                      static_cast<unsigned int>(EligibilityRecordState::settled)) !=
                                static_cast<unsigned int>(EligibilityRecordState::live)) {
      cursor = next;
      continue;
    }

    // #1179: eff_target/eff_flags/eff_learned_output_word are read from
    // whichever representation currently backs this record -- brain.routes[]
    // for locator.kind == explicit_route (byte-identical to pre-#1179
    // behavior), or the state owner's captured fields for implicit_virtual.
    // owner_slot is only meaningful (< rep.state_owner_capacity) in the
    // implicit_virtual case; eligibility_deposit_credit below ignores it
    // otherwise.
    std::uint32_t eff_target = kInvalidIndex;
    std::uint32_t eff_flags = 0u;
    Word eff_learned_output_word = 0u;
    std::uint32_t owner_slot = kInvalidIndex;
    if (!eligibility_effective_route(brain, rep, record, &eff_target, &eff_flags,
                                     &eff_learned_output_word, &owner_slot)) {
      cursor = next;
      continue;
    }

    ++settled_records;
    atomicAdd(&counters->eligibility_exact_settlements, 1u);
    const DirectNode target = brain.nodes[eff_target];
    const bool learned_output = (eff_flags & kRouteFlagLearnedOutput) != 0u;
    const bool fixed_output = !learned_output && target.output_word != 0u;
    const Word expected_word = learned_output ? eff_learned_output_word : target.output_word;
    const bool raw_word_compatible =
        (!learned_output && !fixed_output) || expected_word == event.word;
    const bool same_boundary_channel =
        (target.flags & kNodeFlagMotor) != 0u && (actual.flags & kNodeFlagSensor) != 0u &&
        target.output_channel != kInvalidIndex && target.output_channel == actual.output_channel;
    const bool compatible =
        (eff_target == event.node || same_boundary_channel) && raw_word_compatible;
    const bool predicted = record.predicted_context == event.context;
    const std::int32_t signed_credit =
        compatible
            ? static_cast<std::int32_t>(
                  (static_cast<std::int64_t>(kPositiveCreditQ16) * plasticity) >> 16)
            : (predicted ? static_cast<std::int32_t>(
                               (static_cast<std::int64_t>(kNegativeCreditQ16) * plasticity) >> 16)
                         : 0);
    if (signed_credit != 0) {
      eligibility_deposit_credit(brain, rep, record, owner_slot, signed_credit);
      atomicAdd(&counters->credit_commits, 1u);
    }
    if (compatible) {
      matched_any = true;
      // The learned-output-word capture and context-route acceleration
      // index below both write into the live explicit DirectRoute; an
      // implicit_virtual record has none, so neither applies to it -- its
      // learned_output_word was captured once at migration time and its
      // credit already lives in the state owner's residual.
      if (record.locator.kind == DirectLocatorKind::explicit_route &&
          record.route.slot < brain.route_capacity) {
        DirectRoute& route = brain.routes[record.route.slot];
        if (!learned_output && (target.flags & kNodeFlagMotor) != 0u && target.output_word == 0u) {
          route.learned_output_word = event.word;
          route.flags |= kRouteFlagLearnedOutput;
        }
        route.context_signature = record.history_signature;
        install_context_route(brain, event.cue_node, record.history_signature, record.route.slot,
                              counters);
      }
    } else if (predicted) {
      predicted_mismatch_any = true;
      // #1179 (generalized #1235): the #1176 fallback below would otherwise
      // grow a second, untagged explicit route toward the same observed
      // outcome on every mismatch -- racing #1179's own rematerialize_pending
      // path for the freed slot and permanently masking further contradiction
      // signal. It only needs suppressing when this record's credit was just
      // deposited somewhere OTHER than the very route the fallback would
      // grow/strengthen, i.e. eligibility_deposit_credit's non-explicit-route
      // branch already fired for it above -- so the correct gate is
      // locator.kind, not the merely-correlated logical_id != 0.
      //
      // Gating on logical_id != 0u (the original #1179 check) was wrong for a
      // #1187-materialized record that carries a stable logical_id but has
      // NOT yet migrated to a Level B state owner (locator.kind is still
      // explicit_route): eligibility_deposit_credit's explicit-route branch
      // just deposited its credit into route.conductance_q16 -- exactly what
      // the fallback would also grow -- so suppressing the fallback for it
      // dropped real structural-learning signal for every #1187-materialized
      // source, not just state-owned ones. Keying on locator.kind instead is
      // also correct by construction for any future representation authority
      // that adds its own eligibility_deposit_credit branch, without that
      // authority needing its own local flag here.
      if (record.locator.kind != DirectLocatorKind::explicit_route)
        predicted_mismatch_owned = true;
      if (representative_target == kInvalidIndex) {
        representative_history = record.history_signature;
        representative_target = eff_target;
      }
    }
    cursor = next;
  }

  if (settled_records == 0u) {
    atomicAdd(&counters->eligibility_duplicate_settlement_rejects, 1u);
    atomicAdd(&counters->ignored_returns, 1u);
    return;
  }
  if (matched_any)
    atomicAdd(&counters->matches, 1u);
  if (predicted_mismatch_any)
    atomicAdd(&counters->mismatches, 1u);
  if (matched_any || !predicted_mismatch_any || predicted_mismatch_owned)
    return;

  // No eligible prediction matched the consequence. Strengthen/recruit the
  // observed route once for the episode, not once per participating record.
  std::uint32_t reusable_observed_route = kInvalidIndex;
  std::int32_t reusable_observed_conductance = 0;
  DirectNode& cue = brain.nodes[event.cue_node];
  std::uint32_t route_index = cue.first_route;
  for (std::uint32_t visited = 0u; visited < cue.route_count && route_index != kInvalidIndex;
       ++visited) {
    DirectRoute& observed = brain.routes[route_index];
    const DirectNode observed_target = brain.nodes[observed.target];
    const bool observed_learned = (observed.flags & kRouteFlagLearnedOutput) != 0u;
    const bool observed_fixed = !observed_learned && observed_target.output_word != 0u;
    const Word observed_word =
        observed_learned ? observed.learned_output_word : observed_target.output_word;
    const bool word_ok = (!observed_learned && !observed_fixed) || observed_word == event.word;
    if (observed.target == event.node && word_ok &&
        (reusable_observed_route == kInvalidIndex ||
         observed.conductance_q16 > reusable_observed_conductance)) {
      reusable_observed_route = route_index;
      reusable_observed_conductance = observed.conductance_q16;
    }
    route_index = observed.next_route;
  }

  const std::int32_t positive_credit =
      static_cast<std::int32_t>((static_cast<std::int64_t>(kPositiveCreditQ16) * plasticity) >> 16);
  ActivityEvent contextual_event = event;
  contextual_event.history_signature = representative_history;
  if (representative_target < brain.node_count) {
    const DirectNode predicted_target = brain.nodes[representative_target];
    if ((predicted_target.flags & kNodeFlagMotor) != 0u && (actual.flags & kNodeFlagSensor) != 0u &&
        predicted_target.output_channel != kInvalidIndex &&
        predicted_target.output_channel == actual.output_channel) {
      contextual_event.node = representative_target;
    }
  }
  if (contextual_event.node == event.node && reusable_observed_route != kInvalidIndex) {
    DirectRoute& recruited = brain.routes[reusable_observed_route];
    atomicAdd(&recruited.conductance_q16, positive_credit);
    recruited.last_credit_q16 = positive_credit;
    // The context index is only an acceleration structure. Observation-driven
    // route reuse must carry the learned history on the physical route itself so
    // an ambiguous/stale index can fail closed to the canonical route scan.
    recruited.context_signature = representative_history;
    install_context_route(brain, event.cue_node, representative_history, reusable_observed_route,
                          counters);
    atomicAdd(&counters->credit_commits, 1u);
    return;
  }

  const DirectNode target_node = brain.nodes[contextual_event.node];
  std::uint32_t route_flags = 0u;
  Word learned_word = 0u;
  if ((target_node.flags & kNodeFlagMotor) != 0u && target_node.output_word == 0u) {
    route_flags |= kRouteFlagLearnedOutput;
    learned_word = contextual_event.word;
  }
  submit_direct_growth_proposal(
      topology, i, event.cue_node, contextual_event.node, representative_history,
      clamp_conductance(kConductanceOneQ16 + positive_credit), positive_credit,
      1u + ((event.cue_node ^ contextual_event.node) & 3u), route_flags, positive_credit,
      kInvalidIndex, kInvalidIndex, kInvalidIndex, 0u, kInvalidIndex, representative_history, 0u,
      learned_word);
}

// #1236 deterministic bucket collapse. Two eligibility records that describe the
// SAME causal claim -- same source, same context, same ticket, bound to the same
// route/locator -- are one claim that arrived twice, and settling both pays the
// same credit twice. Measured on landed main: delivering one contact twice mints
// two records (eligibility_peak 1 -> 2, eligibility_index_collisions 1).
//
// Three things decide where this can live and how it must choose.
//
// THE DETERMINISTIC COLLAPSE cannot be a dedup walk at mint time. Two threads
// carrying the same claim can both walk the chain, both fail to find an occupant,
// and both append -- so the surviving record COUNT would depend on thread order,
// which is precisely the nondeterminism #1236 exists to remove. Compaction runs
// one thread per record once per tick, after all minting for the tick is
// finished, so the set it examines is fixed while it examines it.
//
// ⚠ CORRECTED at rung 13, and the correction is a real limit rather than a
// rewording. This comment used to read as though compaction were therefore the
// only place any suppression could happen. It is the only place the DETERMINISTIC
// one can happen, which is a different claim. Compaction runs AFTER minting, so
// the bank must hold the whole un-collapsed population until it does; measured
// with propagated events admitted, this merge alone still left capacity_rejects
// at 800 of 1024 and pinned the bank at its 8193 ceiling. A collector cannot
// bound a population that overflows between collections. A best-effort store-site
// filter was then built and removed after it moved that number only to 795 -- see
// the note in append_eligibility_record for why: the overflow is not duplication.
//
// ⭐ VERIFIED TO CHAIN DEPTH 4096 at rung 13 (Assay 7): N copies of one claim
// collapse to exactly one, eligibility_merged_duplicates == N at every depth from
// 1 to 4096, stable over fifteen repetitions each, with zero capacity rejects.
// The walk below is O(chain) per record and therefore O(chain^2) per bucket, and
// that exponent stays under the run-to-run noise of a whole adult step even at
// 4096 -- forty-four times the chain length rung 11's counterfactual produces.
//
// It cannot choose the survivor by bank INDEX, because the bank's order is itself
// produced by an atomicAdd race and differs between runs. The survivor is chosen
// by CONTENT: the smallest history_signature, then the earliest participation
// tick, then the smallest root. Whatever order the records landed in, the same
// content wins. Index is consulted only to break a tie between records that are
// identical in all of those fields, where the choice cannot change what survives.
//
// And it must be CONSERVATIVE. `eligibility_bucket_slot` does `(void)ticket`, so
// a bucket holds every record sharing (source, context) -- including records with
// DIFFERENT tickets, which are different claims that a merge must leave alone.
// Sharing a bucket is not sharing a claim. The key below therefore compares the
// full claim identity, and the bucket is used only to bound the search.
__device__ inline bool eligibility_same_claim(const DirectEligibilityRecord& a,
                                              const DirectEligibilityRecord& b) {
  return a.source == b.source && a.context == b.context && a.ticket == b.ticket &&
         a.route.slot == b.route.slot && a.route.generation == b.route.generation &&
         a.locator.kind == b.locator.kind && a.locator.slot == b.locator.slot &&
         a.locator.generation == b.locator.generation && a.logical_id.value == b.logical_id.value;
}

// Strict content ordering. Returns true when `a` should outrank `b` as the
// surviving representative of a claim, ignoring where either sits in the bank.
__device__ inline bool eligibility_outranks(const DirectEligibilityRecord& a,
                                            const DirectEligibilityRecord& b) {
  if (a.history_signature != b.history_signature)
    return a.history_signature < b.history_signature;
  if (a.participation_tick != b.participation_tick)
    return a.participation_tick < b.participation_tick;
  if (a.participation_root != b.participation_root)
    return a.participation_root < b.participation_root;
  return a.strength_q16 > b.strength_q16;
}

// Whether `other` would itself survive this compaction. A record must not be
// suppressed in favour of one that is about to be dropped, or the claim would be
// lost entirely rather than merged.
__device__ inline bool eligibility_survives_compaction(DirectBrainV01 brain,
                                                       DirectRepresentationDeviceView rep,
                                                       const DirectEligibilityRecord& other,
                                                       std::uint32_t tick) {
  return eligibility_locator_live(brain, rep, other) &&
         other.state == EligibilityRecordState::live && other.expiry_tick >= tick;
}

// True when `record` at index `self` is the representative its claim keeps.
__device__ inline bool eligibility_record_is_canonical(
    DirectBrainV01 brain, DirectRepresentationDeviceView rep,
    const DirectEligibilityRecord* bank, std::uint32_t active,
    const std::uint32_t* bucket_heads, std::uint32_t bucket_count,
    const DirectEligibilityRecord& record, std::uint32_t self, std::uint32_t tick) {
  if (bucket_heads == nullptr || bucket_count == 0u)
    return true;
  const std::uint32_t bucket =
      eligibility_bucket_slot(record.source, record.context, record.ticket, bucket_count);
  std::uint32_t cursor = bucket_heads[bucket];
  // The chain is bounded by the live set; `guard` stops a corrupted link from
  // spinning a thread forever rather than trusting the list to terminate.
  for (std::uint32_t guard = 0u; cursor != kInvalidIndex && cursor < active && guard <= active;
       ++guard) {
    if (cursor != self) {
      const DirectEligibilityRecord& other = bank[cursor];
      if (eligibility_same_claim(record, other) &&
          eligibility_survives_compaction(brain, rep, other, tick)) {
        if (eligibility_outranks(other, record))
          return false;
        // Identical in every ordering field: keep the lower index. Which slot
        // that is may vary between runs, but the surviving CONTENT cannot.
        if (!eligibility_outranks(record, other) && cursor < self)
          return false;
      }
    }
    cursor = bank[cursor].next_in_bucket;
  }
  return true;
}

__global__ void compact_eligibility_kernel(
    DirectBrainV01 brain, DirectEligibilityRecord* eligibility_bank,
    const std::uint32_t* eligibility_count, std::uint32_t eligibility_capacity,
    DirectEligibilityRecord* next_eligibility_bank, std::uint32_t* next_eligibility_count,
    const std::uint32_t* bucket_heads, std::uint32_t* next_bucket_heads,
    std::uint32_t bucket_count, std::uint32_t tick, AdultCounters* counters,
    DirectRepresentationDeviceView rep) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= min(*eligibility_count, eligibility_capacity))
    return;
  DirectEligibilityRecord& record = eligibility_bank[i];
  // Compaction is the one place a record physically leaves the live set, so it is
  // where the ledger release belongs. Every path below either re-appends the
  // record into the next bank (charge kept) or drops it (charge released) --
  // exactly one of the two, which is what makes the release exactly-once.
  //
  // #1179: a record backed by implicit_virtual storage has no meaningful
  // brain.routes[] slot (record.route.slot is the invalid sentinel) -- its
  // real liveness/staleness lives in the state-owner table, not in
  // brain.topology.slot_meta. eligibility_locator_live is the one shared
  // branch every other pass over this bank already uses for exactly this.
  const bool implicit = record.locator.kind == DirectLocatorKind::implicit_virtual;
  if (!eligibility_locator_live(brain, rep, record)) {
    // The record's route was reclaimed underneath it. Previously this path
    // dropped the record with no accounting at all, so its matter leaked: the
    // record vanished while the pool went on believing it was live.
    atomicAdd(&counters->eligibility_stale_generation_reject, 1u);
    device_release_pool_units(brain.resource_ecology,
                              DirectResourcePoolKind::eligibility_record, 1u);
    return;
  }

  if (record.state != EligibilityRecordState::live) {
    if (!implicit && (record.state == EligibilityRecordState::settled ||
                       record.state == EligibilityRecordState::expired))
      release_route_eligibility_summary(brain, record);
    device_release_pool_units(brain.resource_ecology,
                              DirectResourcePoolKind::eligibility_record, 1u);
    return;
  }

  if (record.expiry_tick >= tick) {
    // #1236: collapse a claim that arrived more than once. The suppressed copy
    // leaves the live set here exactly as a dropped record does, so its matter is
    // released once -- the same exactly-once accounting every other path holds to.
    if (!eligibility_record_is_canonical(brain, rep, eligibility_bank,
                                         min(*eligibility_count, eligibility_capacity),
                                         bucket_heads, bucket_count, record, i, tick)) {
      atomicAdd(&counters->eligibility_merged_duplicates, 1u);
      // #1334: the suppressed copy is leaving the live set exactly as the two
      // other removal paths above/below do, so it must release the same
      // explicit route summary they do -- otherwise route.eligibility_live_count
      // never sees this record's release and stays permanently inflated.
      if (!implicit)
        release_route_eligibility_summary(brain, record);
      device_release_pool_units(brain.resource_ecology,
                                DirectResourcePoolKind::eligibility_record, 1u);
      return;
    }
    append_eligibility_record(brain, record, next_eligibility_bank, next_eligibility_count,
                              eligibility_capacity, next_bucket_heads, bucket_count, nullptr,
                              nullptr, false, /*charge_new=*/false);
    return;
  }

  if (atomicCAS(reinterpret_cast<unsigned int*>(&record.state),
                static_cast<unsigned int>(EligibilityRecordState::live),
                static_cast<unsigned int>(EligibilityRecordState::expired)) !=
      static_cast<unsigned int>(EligibilityRecordState::live))
    return;
  if (record.predicted_context == record.context) {
    if (implicit) {
      const std::uint32_t owner_slot = find_state_owner(rep, record.logical_id);
      if (owner_slot < rep.state_owner_capacity) {
        atomicAdd(reinterpret_cast<unsigned long long*>(
                      &rep.state_owners[owner_slot].implicit_credit_accumulator_q16),
                  static_cast<unsigned long long>(static_cast<long long>(kOmissionCreditQ16)));
      }
    } else {
      DirectRoute& route = brain.routes[record.route.slot];
      atomicAdd(&route.conductance_q16, kOmissionCreditQ16);
      atomicAdd(&route.last_credit_q16, kOmissionCreditQ16);
    }
    atomicAdd(&counters->omissions, 1u);
    atomicAdd(&counters->credit_commits, 1u);
  }
  atomicAdd(&counters->eligibility_expired_records, 1u);
  if (!implicit)
    release_route_eligibility_summary(brain, record);
  device_release_pool_units(brain.resource_ecology,
                            DirectResourcePoolKind::eligibility_record, 1u);
}
