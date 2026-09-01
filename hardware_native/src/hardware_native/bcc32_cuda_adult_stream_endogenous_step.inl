// Adult stream endogenous resident-step and emission-commit kernels.
//
// Included inside bcc32_cuda_adult_stream_v1's namespace after surface
// realization and before transport/tick orchestration. This unit owns the
// device-local endogenous frontier, relation commitment receipts, and the
// focused drive/plasticity lesions; it introduces no new state or order.
#if !defined(BCC32_CUDA_ADULT_STREAM_CHECKPOINT_ONLY)

__global__ void build_unigram_top_parallel_kernel(const std::uint32_t* vitality,
                                                   std::uint32_t unit_count,
                                                   std::uint32_t* top_ids) {
  __shared__ std::uint32_t candidate_ids[adult::kBlock];
  __shared__ std::uint32_t candidate_weights[adult::kBlock];
  for (std::uint32_t slot = 0u; slot < adult::kUnigramTop; ++slot) {
    std::uint32_t best_id = 0u;
    std::uint32_t best_weight = 0u;
    for (std::uint32_t unit = threadIdx.x; unit < unit_count; unit += blockDim.x) {
      bool already_selected = false;
      for (std::uint32_t prior = 0u; prior < slot; ++prior) {
        already_selected |= top_ids[prior] == unit;
      }
      const std::uint32_t weight = already_selected ? 0u : vitality[unit];
      if (weight > best_weight || (weight == best_weight && unit < best_id)) {
        best_id = unit;
        best_weight = weight;
      }
    }
    candidate_ids[threadIdx.x] = best_id;
    candidate_weights[threadIdx.x] = best_weight;
    __syncthreads();
    for (std::uint32_t offset = blockDim.x / 2u; offset != 0u; offset >>= 1u) {
      if (threadIdx.x < offset) {
        const std::uint32_t other_weight = candidate_weights[threadIdx.x + offset];
        const std::uint32_t other_id = candidate_ids[threadIdx.x + offset];
        if (other_weight > candidate_weights[threadIdx.x] ||
            (other_weight == candidate_weights[threadIdx.x] &&
             other_id < candidate_ids[threadIdx.x])) {
          candidate_weights[threadIdx.x] = other_weight;
          candidate_ids[threadIdx.x] = other_id;
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0u) top_ids[slot] = candidate_ids[0];
    __syncthreads();
  }
}

__global__ void set_candidate_count_kernel(std::uint32_t count,
                                           std::uint32_t* generated_count) {
  if (blockIdx.x == 0u && threadIdx.x == 0u) generated_count[0] = count;
}

// A quiet tick may advance resident route matter without creating a public
// candidate.  The sequence extent is read from device state so this receipt
// cannot be host-selected; it records only that the existing resident
// recurrence had enough route material for one bounded step.
__global__ void mark_endogenous_resident_step_kernel(
    std::uint32_t resident_sequence_count,
    QueryAnswerReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u ||
      receipt == nullptr)
    return;
  if (resident_sequence_count >= 2u)
    receipt->endogenous_resident_steps = 1u;
}

// Advance one bounded quiet-step of already resident route matter. The
// transient proposition cue is consumed at the end of its contact, so quiet
// work must use the persistent resident episode sequence instead. The host
// reads only its extent to size this existing chemistry; it never reads
// content, selects a route, or authorizes output. A quiet step is therefore
// observable as changed resident matter while remaining silent at the public
// membrane.
inline bool advance_endogenous_resident_step(adult::AdultState& state,
                                             QueryAnswerReceipt* receipt) {
  if (!state.surface_organ_enabled ||
      state.base_episode_units.get() == nullptr ||
      state.surface_unit_population.get() == nullptr ||
      state.surface_unit_context_population.get() == nullptr ||
      receipt == nullptr ||
      state.surface_population_context_mass.get() == nullptr ||
      state.unit_occurrences < 2u)
    return false;
  const std::uint32_t resident_sequence_count =
      std::min(state.unit_occurrences, adult::kCompositionUnits);
  if (resident_sequence_count < 2u)
    return false;
  adult::adapt_resident_population_coactivity(
      state, state.base_episode_units.get(), resident_sequence_count, nullptr);
  std::uint32_t resident_sequence_count_arg = resident_sequence_count;
  auto* receipt_ptr = receipt;
  void* endogenous_step_arguments[] = {&resident_sequence_count_arg,
                                       &receipt_ptr};
  adult::cuda_require(
      cudaLaunchKernel(
          reinterpret_cast<const void*>(mark_endogenous_resident_step_kernel),
          dim3{1u, 1u, 1u}, dim3{2u, 1u, 1u}, endogenous_step_arguments, 0u,
          nullptr),
      "launch mark endogenous resident step");
  adult::cuda_require(cudaGetLastError(), "mark endogenous resident step");
  return true;
}

// Serialize a relation plan that was selected from the pre-contact resident
// graph. The kernel has no linguistic labels: plan units and their learned
// connective order are its only authority. A non-query contact cannot emit it.
__global__ void emit_stream_relation_commitment_kernel(
    const DriveState* drive, const QueryAnswerReceipt* receipt,
    bool autonomous_continuation, const std::uint32_t* unit_lengths,
    const std::uint32_t* unit_content, std::uint32_t unit_words,
    const std::uint32_t* plan, std::uint32_t* plan_meta,
    const std::uint32_t* boundary_mask, std::uint8_t* output,
    std::uint32_t output_capacity, std::uint32_t* generated_count) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || generated_count == nullptr)
    return;
  generated_count[0] = 0u;
  if (drive == nullptr || drive->emit_pending == 0u || receipt == nullptr ||
      (!autonomous_continuation && receipt->attempted == 0u) || plan == nullptr ||
      plan_meta == nullptr || unit_lengths == nullptr || unit_content == nullptr ||
      boundary_mask == nullptr || output == nullptr)
    return;
  const std::uint32_t count = plan_meta[0];
  // A relation claim is a complete learned (subject, connective, value)
  // triple. It need not carry the optional second connective that makes a
  // four-slot surface construction, so the relation plan's lawful minimum is
  // three resident units.
  if (count < 3u ||
      count > adult::construction::kCommitmentCap)
    return;
  const std::uint32_t start =
      autonomous_continuation ? plan_meta[7] : 0u;
  if (start >= count) return;
  const std::uint32_t item_limit = autonomous_continuation
      ? min(count, max(5u, count - start))
      : count;
  std::uint32_t written = 0u;
  std::uint32_t consumed = 0u;
  std::uint32_t last_byte = 0x20u;
  for (std::uint32_t step = 0u;
       step < item_limit && written < output_capacity; ++step) {
    const std::uint32_t item =
        autonomous_continuation ? (start + step) % count : step;
    const std::uint32_t unit = plan[item];
    const std::uint32_t length = unit_lengths[unit];
    if (length == 0u) continue;
    const std::uint32_t first = unit_content[unit * unit_words] & 0xffu;
    if (item != 0u && boundary_mask[last_byte] == 0u &&
        boundary_mask[first] == 0u && written < output_capacity) {
      output[written++] = 0x20u;
      last_byte = 0x20u;
    }
    for (std::uint32_t offset = 0u; offset < length && written < output_capacity;
         ++offset) {
      const std::uint32_t word = unit_content[unit * unit_words + offset / 4u];
      const std::uint8_t byte =
          static_cast<std::uint8_t>(word >> ((offset % 4u) * 8u));
      output[written++] = byte;
      last_byte = byte;
    }
    consumed = step + 1u;
  }
  generated_count[0] = written;
  plan_meta[7] = autonomous_continuation
      ? count : min(count, consumed);
}

__global__ void latch_stream_relation_commitment_receipt_kernel(
    const std::uint32_t* cursor, const std::uint32_t* meta,
    std::uint32_t topic_fallback, const std::uint32_t* cue_exact,
    const std::uint32_t* cue_orders,
    const std::uint64_t* qonset_evidence_revision,
    const std::uint32_t* question_gap_field_support,
    const std::uint32_t* role_canon,
    std::uint32_t unit_count, QueryAnswerReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || receipt == nullptr) return;
  receipt->relation_candidate_count = cursor == nullptr ? 0u : cursor[0];
  receipt->relation_plan_units = meta == nullptr ? 0u : meta[0];
  receipt->relation_plan_clauses = meta == nullptr ? 0u : meta[1];
  receipt->relation_topic_fallback = topic_fallback;
  receipt->relation_gap_field = 0u;
  bool ambiguous = false;
  const std::uint32_t field = adult::construction::learned_question_gap_field(
      cue_exact, cue_orders, qonset_evidence_revision,
      question_gap_field_support, role_canon, unit_count, &ambiguous);
  if (ambiguous)
    receipt->relation_gap_field = 0xffffffffu;
  else if (field < adult::construction::kRelationFieldCount)
    receipt->relation_gap_field = field + 1u;
}

__global__ void finalize_emission_kernel(
    DriveState* drive, const std::uint8_t* candidate, const std::uint32_t* candidate_count,
    std::uint8_t* egress, std::uint32_t* transport_counts,
    const std::uint32_t* candidate_rng, std::uint32_t* adult_rng,
    QueryAnswerReceipt* query_receipt) {
  if (blockIdx.x != 0u) return;
  __shared__ std::uint32_t count;
  if (threadIdx.x == 0u) {
    transport_counts[0] = 0u;
    transport_counts[1] = 0u;
    if (query_receipt != nullptr) {
      query_receipt->emission_authorized = drive->emit_pending;
      query_receipt->emitted_bytes = 0u;
    }
    count = drive->emit_pending != 0u
        ? min(candidate_count[0], drive->emission_capacity) : 0u;
  }
  __syncthreads();
  const std::uint32_t padded_extent =
      drive->emission_capacity <= kPinnedEmissionPublicationLimit
          ? drive->emission_capacity
          : count;
  for (std::uint32_t i = threadIdx.x; i < padded_extent; i += blockDim.x) {
    // A fixed-size host publication may transport only authorized output and
    // inert padding, never stale resident bytes from a prior motor action.
    egress[i] = i < count ? candidate[i] : 0u;
  }
  __syncthreads();
  if (threadIdx.x == 0u && count != 0u) {
    transport_counts[0] = count;
    transport_counts[1] = drive->reafference_enabled != 0u ? count : 0u;
    if (query_receipt != nullptr) query_receipt->emitted_bytes = count;
    adult_rng[0] = candidate_rng[0];
    ++drive->emissions;
    drive->autonomous_emissions += drive->pending_is_autonomous;
    drive->contact_emissions += drive->pending_is_autonomous == 0u;
    drive->emitted_bytes += count;
    drive->last_output_hash = hash_raw_bytes(egress, count);
    drive->last_emission_tick = drive->ticks;
    const std::uint64_t wanted =
        static_cast<std::uint64_t>(drive->emission_base_cost) + count;
    const std::uint64_t spent = min(wanted, drive->energy);
    drive->energy -= spent;
    drive->energy_spent += spent;
    drive->refractory_ticks = drive->emission_refractory_ticks;
    drive->quiet_ticks = 0u;
    drive->emit_pending = 0u;
    drive->pending_is_autonomous = 0u;
    drive->energy_ledger_ok = drive->energy_initial + drive->energy_gained ==
        drive->energy + drive->energy_spent;
  }
}

__global__ void lesion_drive_kernel(DriveState* drive) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  drive->drive_enabled = 0u;
  drive->drive_lesioned = 1u;
  drive->emit_pending = 0u;
  drive->pending_is_autonomous = 0u;
}

__global__ void lesion_plasticity_flag_kernel(DriveState* drive) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  drive->plasticity_enabled = 0u;
  drive->plasticity_lesioned = 1u;
}

#endif  // !defined(BCC32_CUDA_ADULT_STREAM_CHECKPOINT_ONLY)
