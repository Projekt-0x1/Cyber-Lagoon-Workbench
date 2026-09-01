// Parallel passive census for the resident Record ecology.
//
// The epoch kernel remains the sole ordered writer of ResidentRewriteState.
// These kernels run strictly after that writer in the captured graph. The
// first kernel reads each live Record independently and writes one contribution
// slot; the second kernel performs the deterministic single-writer reduction
// and commits only observer counters. No candidate, provenance, ingress, close
// transaction, or public output is selected here.

inline constexpr std::uint32_t kResidentCensusThreads = 256u;
inline constexpr std::uint32_t kResidentCensusMaxItems =
    rewrite::kMaxResidentPages * rewrite::kRecordsPerPage;
inline constexpr std::uint32_t kResidentCensusBlocks =
    (kResidentCensusMaxItems + kResidentCensusThreads - 1u) /
    kResidentCensusThreads;

struct ResidentCensusContribution {
  std::uint32_t concrete_descriptions = 0u;
  std::uint32_t mature_descriptions = 0u;
  std::uint32_t partial_matches = 0u;
  std::uint32_t program_rules = 0u;
  std::uint32_t mature_program_rules = 0u;
  std::uint32_t trajectory_records = 0u;
  std::uint32_t retained_exemplars = 0u;
  std::uint32_t span_program_rules = 0u;
  std::uint32_t mature_span_program_rules = 0u;
  std::uint32_t version_space_factors = 0u;
  std::uint32_t version_space_alternatives = 0u;
  std::uint32_t mature_version_space_alternatives = 0u;
  std::uint32_t version_space_witnesses = 0u;
  std::uint32_t causal_germline_episodes = 0u;
  std::uint32_t causal_germline_constructors = 0u;
  std::uint32_t causal_germline_counterevidence = 0u;
  std::uint32_t causal_germline_constructor_locus = rewrite::kInvalid;
  std::uint32_t causal_germline_product_locus = rewrite::kInvalid;
  std::uint64_t organization_digest = 0u;
};

struct ResidentCensusScratch {
  ResidentCensusContribution contribution[kResidentCensusMaxItems]{};
  ResidentCensusContribution block_contribution[kResidentCensusBlocks]{};
};

static_assert(std::is_trivially_copyable_v<ResidentCensusContribution>);
static_assert(std::is_trivially_copyable_v<ResidentCensusScratch>);

__device__ inline void resident_census_accumulate(
    ResidentCensusContribution* destination,
    const ResidentCensusContribution& source) {
  destination->concrete_descriptions += source.concrete_descriptions;
  destination->mature_descriptions += source.mature_descriptions;
  destination->partial_matches += source.partial_matches;
  destination->program_rules += source.program_rules;
  destination->mature_program_rules += source.mature_program_rules;
  destination->trajectory_records += source.trajectory_records;
  destination->retained_exemplars += source.retained_exemplars;
  destination->span_program_rules += source.span_program_rules;
  destination->mature_span_program_rules += source.mature_span_program_rules;
  destination->version_space_factors += source.version_space_factors;
  destination->version_space_alternatives += source.version_space_alternatives;
  destination->mature_version_space_alternatives +=
      source.mature_version_space_alternatives;
  destination->version_space_witnesses += source.version_space_witnesses;
  destination->causal_germline_episodes += source.causal_germline_episodes;
  destination->causal_germline_constructors +=
      source.causal_germline_constructors;
  destination->causal_germline_counterevidence +=
      source.causal_germline_counterevidence;
  destination->organization_digest ^= source.organization_digest;
  if (destination->causal_germline_constructor_locus == rewrite::kInvalid &&
      source.causal_germline_constructor_locus != rewrite::kInvalid)
    destination->causal_germline_constructor_locus =
        source.causal_germline_constructor_locus;
  if (destination->causal_germline_product_locus == rewrite::kInvalid &&
      source.causal_germline_product_locus != rewrite::kInvalid)
    destination->causal_germline_product_locus =
        source.causal_germline_product_locus;
}

__global__ void resident_census_scan_kernel(
    const rewrite::ResidentRewriteState* world,
                                            ResidentCensusScratch* scratch) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (world == nullptr || scratch == nullptr ||
      index >= kResidentCensusMaxItems)
    return;

  // The ordered epoch marks the receipt dirty before it can mutate resident
  // matter.  On a quiet epoch the previous census is still authoritative, so
  // do not even read the resident pages.  This keeps the parallel observer
  // proportional to changed revisions rather than turning a GPU-wide launch
  // into a quiet whole-brain memory sweep.
  if (world->revision == world->organization_receipt_revision ||
      world->cross_context_factor_pending != 0u)
    return;

  ResidentCensusContribution contribution{};
  const std::uint32_t capacity = rewrite::live_record_capacity(world);
  if (index < capacity) {
    const rewrite::Record& record = rewrite::record_at(world, index);
    if (record.matter_q8 != 0u && record.lane[0] != rewrite::kFormEmpty) {
      contribution.organization_digest =
          rewrite::logical_record_digest(record);
      if (record.lane[0] == rewrite::kFormDescription) {
        contribution.concrete_descriptions = 1u;
        contribution.mature_descriptions =
            record.lane[5] >= rewrite::kMatureSupport ? 1u : 0u;
      } else if (record.lane[0] == rewrite::kFormPartial) {
        contribution.partial_matches = 1u;
      } else if (record.lane[0] == rewrite::kFormProgram) {
        if ((record.lane[7] & rewrite::kProgramFlagVersionSpace) != 0u) {
          contribution.version_space_alternatives = 1u;
          contribution.mature_version_space_alternatives =
              (record.lane[7] & rewrite::kProgramFlagVersionSpaceLesioned) == 0u &&
                      record.lane[3] >= rewrite::kVersionSpaceMatureSupport
                  ? 1u
                  : 0u;
        } else {
          contribution.program_rules = 1u;
          contribution.mature_program_rules =
              record.lane[3] >= rewrite::kProgramMatureSupport ? 1u : 0u;
        }
      } else if (record.lane[0] == rewrite::kFormTrajectory ||
                 record.lane[0] == rewrite::kFormTrajectoryTerm ||
                 record.lane[0] == rewrite::kFormTrajectoryPage) {
        contribution.trajectory_records = 1u;
        contribution.retained_exemplars =
            record.lane[0] == rewrite::kFormTrajectory && record.lane[3] != 0u
                ? 1u
                : 0u;
      } else if (record.lane[0] == rewrite::kFormSpanProgram) {
        contribution.span_program_rules = 1u;
        contribution.mature_span_program_rules =
            record.lane[3] >= rewrite::kSpanProgramMatureSupport ? 1u : 0u;
      } else if (record.lane[0] == rewrite::kFormProgramFactor) {
        contribution.version_space_factors = 1u;
      } else if (record.lane[0] == rewrite::kFormProgramWitness) {
        contribution.version_space_witnesses = 1u;
      } else if (record.lane[0] == rewrite::kFormConstructionEpisode) {
        contribution.causal_germline_episodes = 1u;
      } else if (record.lane[0] == rewrite::kFormCausalConstructor) {
        contribution.causal_germline_constructors = 1u;
        contribution.causal_germline_constructor_locus = index;
      } else if (record.lane[0] == rewrite::kFormCausalCounterevidence) {
        contribution.causal_germline_counterevidence = 1u;
      }
      if ((record.lane[0] == rewrite::kFormProgram ||
           record.lane[0] == rewrite::kFormSpanProgram) &&
          (record.lane[7] & rewrite::kProgramFlagCausalGermlineProduct) != 0u)
        contribution.causal_germline_product_locus = index;
    }
  }
  scratch->contribution[index] = contribution;
}

__global__ void resident_census_reduce_kernel(
    const rewrite::ResidentRewriteState* world,
    ResidentCensusScratch* scratch) {
  if (world == nullptr || scratch == nullptr ||
      blockIdx.x >= kResidentCensusBlocks)
    return;
  if (world->revision == world->organization_receipt_revision ||
      world->cross_context_factor_pending != 0u)
    return;

  __shared__ ResidentCensusContribution partial[kResidentCensusThreads];
  const std::uint32_t thread = threadIdx.x;
  const std::uint32_t index = blockIdx.x * kResidentCensusThreads + thread;
  partial[thread] = index < kResidentCensusMaxItems
                        ? scratch->contribution[index]
                        : ResidentCensusContribution{};
  __syncthreads();

  for (std::uint32_t stride = kResidentCensusThreads / 2u; stride != 0u;
       stride >>= 1u) {
    if (thread < stride)
      resident_census_accumulate(&partial[thread], partial[thread + stride]);
    __syncthreads();
  }
  if (thread == 0u)
    scratch->block_contribution[blockIdx.x] = partial[0];
}

__global__ void resident_census_commit_kernel(
    rewrite::ResidentRewriteState* world,
    const ResidentCensusScratch* scratch, TickReceipt* resident_receipt,
    TickReceipt* egress_receipt, std::uint64_t* egress_generation) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || world == nullptr ||
      scratch == nullptr)
    return;

  // No changed revision means the epoch only advanced the resident clock.
  // Avoid paying even the parallel scan on a quiet epoch. A pending factor
  // transaction remains private until its own close phase has settled.
  if (world->revision == world->organization_receipt_revision ||
      world->cross_context_factor_pending != 0u) {
    world->organization_receipt_deferred = 0u;
    return;
  }

  ResidentCensusContribution total{};
  for (std::uint32_t block = 0u; block < kResidentCensusBlocks; ++block)
    resident_census_accumulate(&total, scratch->block_contribution[block]);

  world->concrete_descriptions = total.concrete_descriptions;
  world->mature_descriptions = total.mature_descriptions;
  world->partial_matches = total.partial_matches;
  world->program_rules = total.program_rules;
  world->mature_program_rules = total.mature_program_rules;
  world->trajectory_records = total.trajectory_records;
  world->retained_exemplars = total.retained_exemplars;
  world->span_program_rules = total.span_program_rules;
  world->mature_span_program_rules = total.mature_span_program_rules;
  world->version_space_factors = total.version_space_factors;
  world->version_space_alternatives = total.version_space_alternatives;
  world->mature_version_space_alternatives =
      total.mature_version_space_alternatives;
  world->version_space_witnesses = total.version_space_witnesses;
  world->causal_germline_episodes = total.causal_germline_episodes;
  world->causal_germline_constructors = total.causal_germline_constructors;
  world->causal_germline_counterevidence =
      total.causal_germline_counterevidence;
  world->causal_germline_constructor_locus =
      total.causal_germline_constructor_locus;
  world->causal_germline_product_locus = total.causal_germline_product_locus;
  world->organization_digest = total.organization_digest;
  world->organization_receipt_revision = world->revision;
  world->organization_receipt_deferred = 0u;

  // The ordered epoch publishes its receipt before this observer-only census
  // runs. Refresh the same public lineage fields under the egress generation
  // seqlock so a contact snapshot cannot observe a freshly committed world
  // paired with the pre-census invalid digest. No resident choice or output
  // is made here; only the already-computed observer receipt is repaired.
  if (resident_receipt != nullptr) {
    resident_receipt->rewrite_organization_digest = world->organization_digest;
    resident_receipt->rewrite_world_lineage_valid = 1u;
    resident_receipt->rewrite_world_lineage_revision = world->revision;
    resident_receipt->rewrite_world_lineage_organization_digest =
        world->organization_digest;
    resident_receipt->rewrite_world_lineage_admitted_events =
        world->admitted_events;
  }
  if (egress_receipt != nullptr && egress_generation != nullptr) {
    cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system> generation(
        *egress_generation);
    const std::uint64_t current =
        generation.load(cuda::memory_order_relaxed);
    generation.store(current + 1u, cuda::memory_order_release);
    egress_receipt->rewrite_organization_digest = world->organization_digest;
    egress_receipt->rewrite_world_lineage_valid = 1u;
    egress_receipt->rewrite_world_lineage_revision = world->revision;
    egress_receipt->rewrite_world_lineage_organization_digest =
        world->organization_digest;
    egress_receipt->rewrite_world_lineage_admitted_events =
        world->admitted_events;
    __threadfence_system();
    generation.store(current + 2u, cuda::memory_order_release);
  }
}

struct PhysicalIngress {
  RawPhysicalIntervention event{};
  std::uint64_t published = 0u;
  std::uint64_t consumed = 0u;
};

struct Lifecycle {
  std::uint32_t shutdown = 0u;
  std::uint32_t stopped = 0u;
  std::uint32_t continuation_fault = 0u;
  // Device-authored ordinary-F support population at a fatal child fault.
  // This is diagnostic state only; it never authorizes a route or a write.
  std::uint32_t ordinary_f_active_count = 0u;
};

struct EgressState {
  std::uint64_t generation = 0u;
  BoundaryWord actions[kActionWords]{};
  std::uint32_t action_count = 0u;
  std::uint8_t language[kLanguageBytes]{};
  std::uint32_t language_count = 0u;
  std::uint64_t history_generation = 0u;
  std::uint8_t egress_history_bytes[kEgressHistoryBytes]{};
  std::uint64_t egress_history_next_sequence = 1u;
  std::uint64_t egress_history_oldest_sequence = 1u;
  std::uint64_t egress_history_overwrite_count = 0u;
  std::uint32_t egress_history_fault = 0u;
  ActionReturnTicket action_return_ticket{};
  TickReceipt receipt{};
  std::uint64_t energy = 0u;
  std::uint64_t host_bootstrap_launches = 0u;
  std::uint64_t device_epochs = 0u;
};

// Separate observer-owned storage. Resident code receives no pointer to this
// object and can therefore neither read nor use it as causal authority.
struct BitBusCircuitRing {
  BitBusCircuitEvent events[kBitBusL0Capacity]{};
  std::uint64_t published = 0u;
  std::uint64_t overwritten = 0u;
  std::uint64_t last_egress_generation = 0u;
};

static_assert(std::is_trivially_copyable_v<BitBusCircuitRing>);

// Transaction-local observer receipt. A rejected or incomplete return must
// not teach the adult merely because one of its chunks was visited.
struct ActionReturnConstraintDelta {
  std::uint64_t attempted = 0u;
  std::uint64_t accepted = 0u;
  std::uint64_t rejected = 0u;
  std::uint64_t countered_records = 0u;
  std::uint64_t admitted_records = 0u;
  std::uint64_t resident_revision = 0u;
  std::uint32_t component_ready = 0u;
  std::uint32_t component_ambiguous = 0u;
  std::uint32_t component_records = 0u;
  std::uint32_t component_sources = 0u;
  std::uint32_t rederived_event = 0u;
  std::uint32_t source_revision = rewrite::kInvalid;
};

struct DeviceState {
  rewrite::ResidentRewriteState world{};
  TickReceipt receipt{};
  ContentAddress sealed{};
  ContentAddress law{};
  ContentAddress image{};
  ContentAddress genesis{};
  // Optional production-demo ordinary-F aperture. The handle contains only
  // device-owned graph/control pointers; no support words or semantic route
  // cross the resident epoch boundary.
  ordinary_f::DeviceLaunchHandle ordinary_f{};
  ContentAddress f_genesis_manifest{};
  std::uint64_t expected_f_tick = 0u;
  std::uint32_t f_owned_clock = 0u;
  std::uint32_t f_fault = 0u;
  ContentAddress predecessor{};
  ContentAddress egress_history_digest{};
  egress_history::State egress_history{};
  // The canonical public-emission gate (0X1-158): shards the mouth surface
  // that used to be gated only by the single generated_word/generated_locus
  // triple. See mouth_compartment_gate_public_emission below and its use at
  // the egress_history::append checkpoint in resident_rewrite_epoch_kernel.
  rewrite::MouthCompartmentField mouth{};
  BoundaryWord projected_actions[kActionWords]{};
  std::uint64_t projected_action_sequences[kActionWords]{};
  std::uint32_t projected_action_head = 0u;
  std::uint32_t projected_action_count = 0u;
  std::uint8_t projected_language[kLanguageBytes]{};
  std::uint64_t projected_language_sequences[kLanguageBytes]{};
  std::uint32_t projected_language_head = 0u;
  std::uint32_t projected_language_count = 0u;
  ActionReturnTicket action_return_ticket{};
  std::uint64_t action_return_instance_nonce = 0u;
  std::uint64_t device_body_producer_instance = 0u;
  std::uint64_t device_body_source_epoch = 0u;
  std::uint64_t device_body_next_route_sequence = 1u;
  std::uint64_t device_body_last_route_sequence = 0u;
  std::uint64_t device_body_state = 0u;
  std::uint64_t device_body_transition_count = 0u;
  std::uint32_t action_return_device_body_consequence_word = 0u;
  std::uint32_t device_body_initialized = 0u;
  std::uint32_t action_return_world_cell_slot = rewrite::kInvalid;
  std::uint32_t action_return_world_claim_slot = rewrite::kInvalid;
  std::uint32_t action_return_world_write_count = 0u;
  std::uint64_t action_return_issued = 0u;
  std::uint64_t action_return_accepted = 0u;
  std::uint64_t action_return_rejected = 0u;
  std::uint64_t action_return_last_action_sequence = 0u;
  // Passive count of channel-1 actions originated by the generic resident
  // prelinguistic babble fallback. It never gates action selection.
  std::uint64_t resident_motor_babble_actions = 0u;
  std::uint64_t action_return_contact_sequence = 0u;
  std::uint64_t action_return_contact_words = 0u;
  ContentAddress action_return_contact{};
  // A live or just-committed action-return transaction holds only quiet
  // mouth-lease expiry until the next raw contact. It must never gate
  // resident generation or reorganization: the adult remains autonomous
  // while a return is outstanding, and chronology is enforced by the exact
  // ticket and transport envelope.
  std::uint32_t action_return_autonomy_barrier = 0u;
  ActionReturnTicket action_return_stream_ticket{};
  std::uint64_t action_return_stream_next_chunk = 1u;
  std::uint64_t action_return_stream_bytes = 0u;
  std::uint64_t action_return_stream_words = 0u;
  std::uint64_t action_return_stream_lanes[4]{};
  rewrite::ResidentRewriteState action_return_staging_world{};
  // Predictive-shadow matter for the in-flight action-return transaction.
  // Returned words settle shadows here, not in the live matter, so a
  // transaction that finalize_action_return_commit later faults or rejects
  // cannot leave match/violation residue a real acceptance never ratified
  // (0X1-267 requirement 4).
  resident_predictive_shadow_assay::PredictiveShadowMatter
      action_return_staging_predictive_shadow{};
  // Private failure-atomic close shadow. It is executor workspace only: while
  // active, no ingress, physical intervention, autonomous generation, ticket
  // issue, or public egress may read it as cognitive authority. Only a
  // completely settled shadow is copied over the canonical world.
  rewrite::ResidentRewriteState close_work_staging_world{};
  std::uint32_t close_work_active = 0u;
  // Private post-close settlement phase. It keeps dependent resident
  // mutation inside the failure-atomic close shadow instead of exposing a
  // partially settled canonical world between device epochs.
  std::uint32_t close_work_settlement_phase = 0u;
  std::uint32_t close_work_origin = 0u;
  std::uint32_t close_work_device_body_return = 0u;
  std::uint32_t close_work_fault = 0u;
  ActionReturnTicket close_work_action_ticket{};
  ContentAddress close_work_return_contact{};
  std::uint64_t close_work_return_words = 0u;
  std::uint64_t close_work_return_sequence = 0u;
  std::uint32_t action_return_stream_active = 0u;
  std::uint32_t action_return_stream_saw_physical_end = 0u;
  std::uint32_t action_return_stream_inquiry_reply_required = 0u;
  std::uint32_t action_return_stream_inquiry_reply_settled = 0u;
  std::uint32_t action_return_stream_from_device_body = 0u;
  std::uint32_t action_return_stream_device_body_consequence_word = 0u;
  ActionReturnConstraintDelta action_return_stream_constraint_delta{};
  // The accepted distributed return may be demoted by the ordinary END
  // qualifier before terminal publication. Keep its exact trajectory owner
  // so retirement does not depend on the header still being "current".
  std::uint32_t action_return_stream_distributed_trajectory_owner =
      rewrite::kInvalid;
  // Passive accounting for the resident-only constraint update performed in
  // the accepted staging transaction. These values never gate ingress,
  // choose content, or authorize an outward action.
  std::uint64_t action_return_constraint_reafferent_attempted = 0u;
  std::uint64_t action_return_constraint_reafferent_accepted = 0u;
  std::uint64_t action_return_constraint_reafferent_rejected = 0u;
  std::uint64_t action_return_constraint_countered_records = 0u;
  std::uint64_t action_return_constraint_admitted_records = 0u;
  std::uint64_t action_return_constraint_resident_revision = 0u;
  std::uint32_t action_return_constraint_component_ready = 0u;
  std::uint32_t action_return_constraint_component_ambiguous = 0u;
  std::uint32_t action_return_constraint_component_records = 0u;
  std::uint32_t action_return_constraint_component_sources = 0u;
  std::uint32_t action_return_constraint_rederived_event = 0u;
  // Passive accounting for distributed constraint-participation formation
  // staged at an ordinary raw-contact physical END, before capture_teacher_
  // surface_before_end can consume/clear the trajectory. These values never
  // gate ingress, choose content, or authorize an outward action.
  std::uint64_t rewrite_participation_end_attempted = 0u;
  std::uint64_t rewrite_participation_end_admitted = 0u;
  std::uint64_t rewrite_participation_end_rejected = 0u;
  // Passive latest-ingress-END lifecycle samples. `materialized` is the live
  // participation population immediately after the external-relation phase
  // completes in the private close shadow; `precommit` is after later close
  // phases; `committed` is immediately after canonical ownership transfer.
  // No resident predicate reads these observer-only fields.
  std::uint32_t rewrite_participation_end_materialized_records = 0u;
  std::uint32_t rewrite_participation_end_precommit_records = 0u;
  std::uint32_t rewrite_participation_end_committed_records = 0u;
  std::uint64_t tick = 0u;
  std::uint64_t contact_sequence = 0u;
  std::uint64_t intervention_sequence = 0u;
  std::uint64_t device_epochs = 0u;
  std::uint64_t host_bootstrap_launches = 0u;
  std::uint32_t initialized = 0u;
  // 0X1-267: the 0X1-248 generic predictive-shadow relation assay, promoted
  // unchanged (same ontology, same bounded capacities) from fixture/test-only
  // matter into ordinary per-adult resident state. No candidate, provenance,
  // ingress, close transaction, or public output is selected here; see
  // advance_resident_cognition_phase and consume_ingress in
  // bcc32_resident_rewrite_epoch_phases.inl for its two production call
  // sites.
  resident_predictive_shadow_assay::PredictiveShadowMatter predictive_shadow{};

  // 0X1-267 requirement 5, step 1 (Linear 0X1-267 comment c98064fd): a
  // purely diagnostic, non-gating probe of whether this epoch's already-
  // computed predictive-shadow route projection physically intersects the
  // resident morphology this epoch's relation-candidate reader
  // (run_autonomous_generation_device, bcc32_resident_rewrite_epoch_phases.inl)
  // already judged eligible. Computed once per history-healthy epoch in
  // advance_resident_cognition_phase, strictly after that reader's dispatch
  // cascade has run, and read by nothing else in the resident runtime --
  // never by run_autonomous_generation_device or any other selection/route-
  // choice path. Promoting this probe into real gating authority is a
  // separate, explicitly deferred follow-up.
  SiteWord predictive_shadow_route_probe_eligible_morphology = 0u;
  SiteWord predictive_shadow_route_probe_intersection = 0u;
  std::uint32_t predictive_shadow_route_probe_intersection_popcount = 0u;
};

static_assert(std::is_trivially_copyable_v<DeviceState>);
static_assert(std::is_trivially_copyable_v<IngressRing>);
static_assert(std::is_trivially_copyable_v<ActionReturnIngress>);
static_assert(std::is_trivially_copyable_v<DeviceBodyControl>);
static_assert(std::is_trivially_copyable_v<PhysicalIngress>);
static_assert(std::is_trivially_copyable_v<EgressState>);
static_assert(std::is_trivially_copyable_v<Lifecycle>);
struct DerivedOutput {
  BoundaryWord actions[kActionWords]{};
  std::uint32_t action_count = 0u;
  std::uint8_t language[kLanguageBytes]{};
  std::uint32_t language_count = 0u;
};

static_assert(std::is_trivially_copyable_v<DerivedOutput>);

struct OutputMaterial {
  BoundaryWord actions[kActionWords]{};
  std::uint32_t action_count = 0u;
  std::uint8_t language[kLanguageBytes]{};
  std::uint32_t language_count = 0u;
  ContentAddress egress_history{};
  std::uint64_t egress_history_next_sequence = 1u;
  std::uint64_t egress_history_oldest_sequence = 1u;
  std::uint64_t egress_history_overwrite_count = 0u;
  std::uint32_t egress_history_fault = 0u;
};

static_assert(std::is_trivially_copyable_v<OutputMaterial>);

__device__ std::uint64_t bitbus_fingerprint(const ContentAddress& address) {
  // Compact observer fingerprint only. This function executes exclusively in
  // the observer kernel and its result is never resident-visible.
  std::uint64_t value = 0xcbf29ce484222325ull ^ address.byte_count;
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(&address.digest);
  for (std::uint32_t i = 0u; i < 32u; ++i) {
    value ^= static_cast<std::uint64_t>(bytes[i]);
    value *= 0x100000001b3ull;
  }
  return value;
}

// RUNG-1 L0 observer. It is kept in this extracted observer unit so the
// runtime parent remains a composition point rather than growing another
// production hot path.
__global__ void bitbus_circuit_l0_observer_kernel(const EgressState* egress,
                                                   BitBusCircuitRing* ring) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || egress == nullptr ||
      ring == nullptr)
    return;

  const std::uint64_t generation = egress->generation;
  if (generation == 0u || generation == ring->last_egress_generation)
    return;
  ring->last_egress_generation = generation;

  const TickReceipt& receipt = egress->receipt;
  if (receipt.rewrite_public_emission_receipt_valid == 0u)
    return;

  const std::uint64_t fanout = static_cast<std::uint64_t>(egress->action_count) +
                               static_cast<std::uint64_t>(egress->language_count);
  if (fanout == 0u)
    return;

  const std::uint64_t sequence = ring->published;
  BitBusCircuitEvent event{};
  event.epoch = receipt.tick;
  event.source_physical_locus = receipt.rewrite_public_emission_owner;
  event.destination_physical_locus = kBitBusBoundaryLocus;
  event.native_state_before = bitbus_fingerprint(receipt.predecessor);
  event.native_state_after = bitbus_fingerprint(receipt.commitment);
  event.transfer_kind = kBitBusTransferBoundary;
  event.fanout_count = fanout > 0xffffffffull ? 0xffffffffu
                                               : static_cast<std::uint32_t>(fanout);
  event.source_revision = receipt.rewrite_revision;
  event.morphology_changed = 0u;
  ring->events[sequence % kBitBusL0Capacity] = event;
  __threadfence_system();
  if (sequence >= kBitBusL0Capacity)
    ring->overwritten = sequence + 1u - kBitBusL0Capacity;
  cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system> published(
      ring->published);
  published.store(sequence + 1u, cuda::memory_order_release);
}
