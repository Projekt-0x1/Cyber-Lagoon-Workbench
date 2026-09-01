// Included inside direct_adult_legacy_oracle.cu's anonymous namespace after
// direct_adult_eligibility_settlement.cuh (minting writes into the bank that
// stage owns). Owns frontier propagation: fabric-recorded explicit eligibility,
// sparse propagation with canonical winner resolution, implicit-mesh fanout,
// and the install of committed growth routes. Grids self-bound on the
// device-resident counts, never host-tracked launch bounds.
__global__ void record_fabric_explicit_eligibility_kernel(
    DirectBrainV01 brain, DirectExecutionFabricDeviceView fabric, const ActivityEvent* frontier,
    const std::uint32_t* frontier_count,
    DirectEligibilityRecord* eligibility_bank, std::uint32_t* eligibility_count,
    std::uint32_t eligibility_capacity, std::uint32_t* eligibility_bucket_heads,
    std::uint32_t eligibility_bucket_count, const std::uint32_t* eligibility_batch_admit,
    std::uint32_t tick, AdultCounters* counters) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  // gh #1236: launched at grid_for(frontier_capacity) below and self-bounds
  // only on the device-resident *frontier_count, matching the fix already
  // applied to propagate_sparse_frontier_kernel (#1304) and
  // heterogeneous_dispatch_kernel -- this was the one sibling still gated by
  // the host-tracked frontier_work launch bound, so any frontier entry beyond
  // that stale estimate never got a thread and its external-participation
  // eligibility was silently never recorded.
  if (i >= *frontier_count)
    return;
  const ActivityEvent event = frontier[i];
  const bool external_participation = (event.origin == CausalOrigin::external_contact ||
                                       event.origin == CausalOrigin::motor_reafference) &&
                                      event.external_root != 0u;
  if (!external_participation || event.node >= brain.node_count)
    return;

  const DirectExecutionMembership membership = fabric.node_memberships[event.node];
  if (membership.phenotype == static_cast<std::uint32_t>(DirectExecutionPhenotype::dense_tensor)) {
    // Dense tiles are physical execution blocks rather than route identities.
    // #1176 remains route-local; dense credit needs a separate physical mapping
    // instead of pretending one arbitrary sparse route caused the tile output.
    return;
  }

  const DirectNode source = brain.nodes[event.node];
  if (source.output_word != 0u && (source.flags & kNodeFlagSensor) == 0u)
    return;

  // #1236: this kernel decides which logical interaction owns an event's
  // eligibility record. It used to answer that with its own scoring -- no
  // conductance floor, an additive context bonus, and a different event
  // signature -- so the ledger could credit a route the propagating evaluator
  // had already declined. It now asks the one canonical law.
  const std::uint64_t event_signature =
      direct_canonical_event_context_signature(event.history_signature, event.word);
  const DirectCanonicalWinner canonical =
      direct_canonical_scan_source_routes(brain, source, event_signature);
  const std::uint32_t winner = canonical.route_slot;

  const bool implicit_mesh_eligible =
      direct_canonical_implicit_mesh_eligible(canonical, source.route_count);
  const DirectImplicitCandidate implicit =
      implicit_mesh_eligible
          ? select_direct_implicit_candidate(brain, event.node, event_signature)
          : DirectImplicitCandidate{};
  if ((implicit_mesh_eligible &&
       direct_canonical_implicit_wins(implicit, canonical, source.route_count)) ||
      winner == kInvalidIndex)
    return;

  const DirectRouteSlotMeta meta = brain.topology.slot_meta[winner];
  if (meta.live == 0u)
    return;
  DirectEligibilityRecord record{};
  record.route = DirectRouteHandle{winner, meta.generation};
  record.source = event.node;
  record.context = event.context;
  record.history_signature = event_signature;
  record.successor_history_signature =
      direct_canonical_successor_history_signature(event.history_signature, event.node, event.word);
  record.participation_root = event.external_root;
  record.ticket = event.external_root;
  record.strength_q16 = kEligibilityOneQ16;
  record.participation_tick = tick;
  record.horizon_class = eligibility_horizon_for_event(event);
  record.expiry_tick = tick + eligibility_lifetime(record.horizon_class);
  record.predicted_context = event.context;
  record.state = EligibilityRecordState::live;
  append_eligibility_record(brain, record, eligibility_bank, eligibility_count,
                            eligibility_capacity, eligibility_bucket_heads,
                            eligibility_bucket_count, eligibility_batch_admit, counters, true,
                            /*charge_new=*/true);
}

__global__ void propagate_sparse_frontier_kernel(
    DirectBrainV01 brain, DirectTopologyDeviceView topology, const ActivityEvent* frontier,
    const std::uint32_t* frontier_count, ActivityEvent* next_frontier,
    DirectIngressAuthority* next_frontier_authority, std::uint32_t* next_count,
    std::uint32_t next_capacity, MotorEvent* motor_events, std::uint32_t* motor_count,
    std::uint32_t motor_capacity, DirectEligibilityRecord* eligibility_bank,
    std::uint32_t* eligibility_count, std::uint32_t eligibility_capacity,
    std::uint32_t* eligibility_bucket_heads, std::uint32_t eligibility_bucket_count,
    const std::uint32_t* eligibility_batch_admit, std::uint32_t tick, AdultCounters* counters,
    DirectRepresentationDeviceView rep) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0u && brain.development != nullptr) {
    ResidentDevelopmentState& development = *brain.development;
    ++development.developmental_tick;
    if (development.developmental_tick < development.maturation_tick &&
        development.plasticity_q16 > development.mature_plasticity_floor_q16) {
      const std::uint32_t remaining =
          development.plasticity_q16 - development.mature_plasticity_floor_q16;
      const std::uint32_t horizon =
          max(1u, development.maturation_tick - development.developmental_tick + 1u);
      development.plasticity_q16 -= max(1u, remaining / horizon);
    }
  }
  // 0X1-1175: bounded by the DEVICE-RESIDENT frontier count and by the buffer,
  // never by a host-side launch bound.
  //
  // This kernel's grid used to be `grid_for(frontier_work)`, where
  // `frontier_work = min(frontier_launch_bound, frontier_capacity)` and
  // `frontier_launch_bound` starts at 0 and climbs +1 per ingested event. The
  // resident frontier does not climb with it: one tick's propagation puts about
  // three successors back for every event it consumed. So from the second tick
  // on, the bound was smaller than the frontier and the remainder was never
  // propagated -- and it was not deferred either, because the buffers swap at
  // the end of the step. The organism was silently discarding its own
  // successors for the first eight ticks of its life, and `cdee808cb7` measured
  // that an arm differing from the baseline in nothing but those eight ticks
  // recovers the entire long-horizon difference.
  //
  // `*frontier_count` is the atomicAdd attempt counter, so it can exceed the
  // buffer on a saturated tick; `next_capacity` is the frontier allocation
  // (frontier and next_frontier are both `frontier_capacity`), so the second
  // half of this guard is what keeps the read in bounds now that the grid is
  // sized by capacity rather than by the bound.
  if (i >= *frontier_count || i >= next_capacity)
    return;
  const ActivityEvent event = frontier[i];
  if (event.node >= brain.node_count)
    return;
  DirectNode& source = brain.nodes[event.node];
  if (is_independent_external(event.origin))
    atomicAdd(&source.external_contacts, 1u);
  else
    atomicAdd(&source.endogenous_visits, 1u);

  // #1184: resolved once per event and reused everywhere this hop threads a
  // cue_node forward (frontier successors and any motor event this hop
  // emits). An external action-return bridge addresses a reafferent return
  // at (cue_node, context); the exact eligibility ledger indexes episodes
  // the same way, so motor events must carry the identical resolved value.
  const std::uint32_t resolved_cue_node =
      event.cue_node == kInvalidIndex ? event.node : event.cue_node;
  // #1236: the event context signature is part of the causal law, so it comes
  // from the shared canonical helper rather than being derived here. Every
  // backend must ask the stored routes the same question.
  const std::uint64_t event_context_signature =
      direct_canonical_event_context_signature(event.history_signature, event.word);
  DirectCanonicalWinner canonical = direct_canonical_winner_empty();

  const std::uint32_t indexed_route =
      lookup_context_route(brain, event.node, event_context_signature);
  const bool indexed_route_usable =
      indexed_route < brain.route_capacity && brain.routes[indexed_route].source == event.node &&
      direct_canonical_route_admissible(brain.routes[indexed_route]);
  if (indexed_route_usable) {
    // The context index is a lookup shortcut, not a second law: the slot it
    // names is offered to the same competition every other candidate faces.
    direct_canonical_offer_route(&canonical, brain, indexed_route, event_context_signature);
    atomicAdd(&counters->context_index_hits, 1u);
  } else if (resolve_packed_cache_winner(brain, rep, event.node, event_context_signature,
                                         &canonical.route_slot, &canonical.context_rank,
                                         &canonical.conductance_q16)) {
    // #1235: consulted the #1179 Level A packed cache -- winner (possibly
    // still kInvalidIndex, the same "no conducting candidate" outcome the
    // canonical scan below can itself produce) is already resolved.
    // #1236: the panel resolver now ranks through the shared canonical law, so
    // "resolved by cache" and "resolved by scan" are the same decision.
    canonical.admissible_candidates = canonical.route_slot == kInvalidIndex ? 0u : 1u;
  } else {
    canonical = direct_canonical_scan_source_routes(brain, source, event_context_signature);
  }
  const std::uint32_t winner = canonical.route_slot;

  const bool external_participation = (event.origin == CausalOrigin::external_contact ||
                                       event.origin == CausalOrigin::motor_reafference) &&
                                      event.external_root != 0u;
  if (external_participation && winner != kInvalidIndex) {
    // Every actual participation receives its own exact temporal record. Route
    // scalar fields are only compatibility telemetry; they are not overwritten
    // causal authority anymore.
    const DirectRouteSlotMeta meta = brain.topology.slot_meta[winner];
    if (meta.live != 0u) {
      DirectEligibilityRecord record{};
      record.route = DirectRouteHandle{winner, meta.generation};
      record.source = event.node;
      record.context = event.context;
      record.history_signature = event_context_signature;
      record.successor_history_signature = direct_canonical_successor_history_signature(
          event.history_signature, event.node, event.word);
      record.participation_root = event.external_root;
      // The causal root doubles as the exact asynchronous action ticket. World
      // returns may use a different root, but motor reafference must reproduce it.
      record.ticket = event.external_root;
      record.strength_q16 = kEligibilityOneQ16;
      record.participation_tick = tick;
      record.horizon_class = eligibility_horizon_for_event(event);
      record.expiry_tick = tick + eligibility_lifetime(record.horizon_class);
      record.predicted_context = event.context;
      record.state = EligibilityRecordState::live;
      append_eligibility_record(brain, record, eligibility_bank, eligibility_count,
                                eligibility_capacity, eligibility_bucket_heads,
                                eligibility_bucket_count, eligibility_batch_admit, counters, true,
                              /*charge_new=*/true);
    }
  }

  // Physical route winner propagation. #1236 rung 2: the successor and its
  // efferent consequence are built by the shared canonical helper, so a
  // backend that reaches the same winner cannot build a different successor
  // from it.
  if (winner != kInvalidIndex) {
    const DirectCanonicalEffect effect =
        direct_canonical_explicit_effect(brain, event, resolved_cue_node, winner);
    const std::uint32_t out = atomicAdd(next_count, 1u);
    if (out < next_capacity) {
      next_frontier[out] = effect.successor;
      next_frontier_authority[out] = DirectIngressAuthority::ordinary;
    }
    atomicAdd(&counters->propagated, 1u);

    if (effect.motor_valid) {
      const std::uint32_t motor_slot = atomicAdd(motor_count, 1u);
      if (motor_slot < motor_capacity)
        motor_events[motor_slot] = effect.motor;
      atomicAdd(&counters->motor_events, 1u);
    }
  }

  // Procedural implicit causal mesh fanout
  if (direct_canonical_implicit_mesh_eligible(canonical, source.route_count)) {
    const DirectImplicitCandidateSet implicit_set =
        enumerate_direct_implicit_candidates(brain, event.node, event_context_signature);
    for (std::uint32_t c = 0; c < implicit_set.count; ++c) {
      const DirectImplicitCandidate implicit = implicit_set.candidates[c];
      atomicAdd(&counters->virtual_participations, 1u);

      // #1179: once this exact logical interaction is state-owned, its
      // explicit backing is gone (winner == kInvalidIndex above), so the
      // ordinary explicit-route participation block never runs for it --
      // without this, a state-owned interaction could never accrue a NEW
      // #1176 eligibility episode again after migration. Stamp the record
      // straight onto the owner (locator = implicit_virtual, no route slot)
      // using exactly the same fields the explicit-route path would have
      // used, so exact settlement treats it identically either way.
      if (external_participation && winner == kInvalidIndex) {
        const DirectLogicalInteractionId candidate_logical_id =
            derive_implicit_logical_id(event.node, implicit.family, implicit.virtual_slot);
        const std::uint32_t owner_slot = find_state_owner(rep, candidate_logical_id);
        if (owner_slot != kInvalidIndex &&
            rep.state_owners[owner_slot].lifecycle == DirectStateOwnerLifecycle::owned) {
          const DirectRepresentationStateOwner& owner_ro = rep.state_owners[owner_slot];
          DirectEligibilityRecord implicit_record{};
          implicit_record.logical_id = candidate_logical_id;
          implicit_record.locator.kind = DirectLocatorKind::implicit_virtual;
          implicit_record.locator.slot = owner_slot;
          implicit_record.locator.generation = owner_ro.owner_epoch;
          implicit_record.route = DirectRouteHandle{kInvalidIndex, 0u};
          implicit_record.source = event.node;
          implicit_record.context = event.context;
          implicit_record.history_signature = event_context_signature;
          implicit_record.successor_history_signature =
              direct_canonical_successor_history_signature(event.history_signature, event.node,
                                                           event.word);
          implicit_record.participation_root = event.external_root;
          implicit_record.ticket = event.external_root;
          implicit_record.strength_q16 = kEligibilityOneQ16;
          implicit_record.participation_tick = tick;
          implicit_record.horizon_class = eligibility_horizon_for_event(event);
          implicit_record.expiry_tick = tick + eligibility_lifetime(implicit_record.horizon_class);
          implicit_record.predicted_context = event.context;
          implicit_record.state = EligibilityRecordState::live;
          append_eligibility_record(brain, implicit_record, eligibility_bank, eligibility_count,
                                    eligibility_capacity, eligibility_bucket_heads,
                                    eligibility_bucket_count, eligibility_batch_admit, counters,
                                    true, /*charge_new=*/true);
        }
      }
      if (direct_canonical_implicit_wins(implicit, canonical, source.route_count) &&
          source.route_count != 0u) {
        atomicAdd(&counters->implicit_wins, 1u);
      }

      const DirectCanonicalEffect virtual_effect =
          direct_canonical_implicit_effect(brain, event, resolved_cue_node, implicit);
      const std::uint32_t out_virt = atomicAdd(next_count, 1u);
      if (out_virt < next_capacity) {
        next_frontier[out_virt] = virtual_effect.successor;
        next_frontier_authority[out_virt] = DirectIngressAuthority::ordinary;
      }
      atomicAdd(&counters->propagated, 1u);

      if (virtual_effect.motor_valid) {
        const std::uint32_t motor_slot = atomicAdd(motor_count, 1u);
        if (motor_slot < motor_capacity)
          motor_events[motor_slot] = virtual_effect.motor;
        atomicAdd(&counters->motor_events, 1u);
      }

      const bool candidate_participating =
          direct_canonical_implicit_wins(implicit, canonical, source.route_count);
      if (candidate_participating) {
        const DirectImplicitParticipationResult part_res = record_direct_implicit_participation(
            brain, implicit.family, event.node, implicit.virtual_slot, tick, event.external_root,
            external_participation);
        if (part_res.should_materialize) {
          submit_direct_growth_proposal(
              topology, i * kMaxImplicitActiveFanout + c, event.node, implicit.target,
              event_context_signature, implicit.conductance_q16, 0, implicit.delay, 0u,
              implicit.conductance_q16, implicit.family, implicit.virtual_slot, event.context,
              tick + eligibility_lifetime(eligibility_horizon_for_event(event)), event.context,
              event_context_signature, event.external_root);
        }
      }
    }
  }
}

__global__ void append_committed_growth_eligibility_kernel(
    DirectTopologyRuntime topology, std::uint32_t proposal_work,
    DirectEligibilityRecord* eligibility_bank, std::uint32_t* eligibility_count,
    std::uint32_t eligibility_capacity, std::uint32_t* bucket_heads, std::uint32_t bucket_count,
    const std::uint32_t* batch_admit, DirectBrainV01 brain, std::uint32_t tick,
    AdultCounters* counters) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= proposal_work)
    return;
  const std::uint32_t route_slot = topology.committed_route_slots[i];
  if (route_slot == kInvalidIndex || route_slot >= brain.route_capacity)
    return;
  const DirectTopologyProposal proposal = topology.proposals_sorted[i];
  if (proposal.eligibility_context != kInvalidIndex) {
    const DirectRouteSlotMeta meta = brain.topology.slot_meta[route_slot];
    DirectEligibilityRecord record{};
    record.route = DirectRouteHandle{route_slot, meta.generation};
    record.source = proposal.source;
    record.context = proposal.eligibility_context;
    record.history_signature = proposal.eligibility_history;
    record.participation_root = proposal.eligibility_root;
    record.ticket = proposal.eligibility_root;
    record.strength_q16 = kEligibilityOneQ16;
    record.participation_tick = tick;
    record.expiry_tick = proposal.eligibility_expires;
    record.predicted_context = proposal.predicted_context;
    record.horizon_class = EligibilityHorizonClass::immediate;
    record.state = EligibilityRecordState::live;
    append_eligibility_record(brain, record, eligibility_bank, eligibility_count,
                              eligibility_capacity, bucket_heads, bucket_count, batch_admit,
                              counters, true, /*charge_new=*/true);
  }
}

__global__ void install_committed_context_routes_kernel(DirectBrainV01 brain,
                                                        DirectTopologyRuntime topology,
                                                        std::uint32_t proposal_work,
                                                        AdultCounters* counters) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= proposal_work)
    return;
  const std::uint32_t route_slot = topology.committed_route_slots[i];
  if (route_slot == kInvalidIndex || route_slot >= brain.route_capacity)
    return;
  const DirectTopologyProposal proposal = topology.proposals_sorted[i];
  if (proposal.context_signature != 0u) {
    install_context_route(brain, proposal.source, proposal.context_signature, route_slot, counters);
  }
}
