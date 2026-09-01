// Paged conditioned-owner host method definitions.
//
// Included after PagedConditionedOwner::Impl.
// This unit owns lifecycle, routing, credit, checkpoint, and migration methods.

PagedConditionedOwner::PagedConditionedOwner() : impl_(std::make_unique<Impl>(0u)) {}

PagedConditionedOwner::PagedConditionedOwner(std::uint32_t capacity)
    : impl_(std::make_unique<Impl>(capacity)) {}

PagedConditionedOwner::PagedConditionedOwner(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

PagedConditionedOwner::~PagedConditionedOwner() = default;
PagedConditionedOwner::PagedConditionedOwner(PagedConditionedOwner&&) noexcept = default;
PagedConditionedOwner& PagedConditionedOwner::operator=(PagedConditionedOwner&&) noexcept = default;

void PagedConditionedOwner::reset(std::uint32_t capacity) {
  impl_->clear_resident_factor_state();
  impl_->engine.reset(capacity);
}

std::uint32_t PagedConditionedOwner::capacity() const {
  return impl_->engine.capacity();
}

std::uint32_t PagedConditionedOwner::size() const {
  return impl_->engine.size();
}

std::uint32_t PagedConditionedOwner::remaining_capacity() const {
  return impl_->engine.remaining_capacity();
}

ConsumeReceipt PagedConditionedOwner::consume_device_batch(
    const substrate::bcc32::ConditionedMatterDeviceCredit* device_events, std::uint32_t count) {
  return impl_->engine.consume_device_batch(device_events, count);
}

void PagedConditionedOwner::publish_conductance_device(
    const substrate::bcc32::ConditionedMatterDeviceKey* keys, std::uint32_t count,
    std::uint32_t* output) const {
  impl_->engine.publish_conductance_device(keys, count, output);
}

RouteLesionReceipt PagedConditionedOwner::lesion_route(
    std::uint32_t anchor, std::uint32_t previous, std::uint32_t next,
    std::uint32_t region) {
  const bank::LesionReceipt receipt =
      impl_->engine.lesion_route({anchor, previous, next, region});
  return {static_cast<std::uint32_t>(receipt.code),
          receipt.valid,
          receipt.slot,
          receipt.key.anchor,
          receipt.key.previous,
          receipt.key.next,
          receipt.key.region,
          receipt.positive_word,
          receipt.negative_word,
          receipt.generation,
          receipt.restore_epoch};
}

bool PagedConditionedOwner::restore_route_lesion(
    const RouteLesionReceipt& receipt) {
  bank::LesionReceipt internal{};
  internal.code = static_cast<bank::OperationCode>(receipt.code);
  internal.valid = receipt.valid;
  internal.slot = receipt.slot;
  internal.key =
      {receipt.anchor, receipt.previous, receipt.next, receipt.region};
  internal.positive_word = receipt.positive_word;
  internal.negative_word = receipt.negative_word;
  internal.generation = receipt.generation;
  internal.restore_epoch = receipt.restore_epoch;
  const bool restored = impl_->engine.restore_route_lesion(internal);
  if (restored) resolve_resident_factor_bindings();
  return restored;
}

void PagedConditionedOwner::reset_resident_factor_state(
    std::uint32_t lane_words, std::uint32_t binding_count) {
  impl_->reset_resident_factor_state(lane_words, binding_count);
}

ResidentFactorStateView
PagedConditionedOwner::resident_factor_state_device() const {
  return impl_->resident_factor_state_device();
}

FactorParticipationReceipt
PagedConditionedOwner::capture_resident_factor_participation(
    const ResidentFactorParticipation* device_participation) {
  const ResidentFactorStateView state =
      impl_->resident_factor_state_device();
  if (device_participation == nullptr || state.binding_count == 0u ||
      impl_->resident_lane_words == 0u) {
    return {static_cast<std::uint32_t>(bank::OperationCode::kInvalidInput)};
  }
  if (impl_->participation_receipt == nullptr) {
    cuda_require(cudaMalloc(&impl_->participation_receipt,
                            sizeof(*impl_->participation_receipt)),
                 "allocate owner factor-participation receipt");
    cuda_require(cudaMalloc(&impl_->participation_attempt,
                            sizeof(*impl_->participation_attempt)),
                 "allocate owner factor-participation attempt");
    cuda_require(cudaMemset(impl_->participation_receipt, 0,
                            sizeof(*impl_->participation_receipt)),
                 "clear owner factor-participation receipt");
  }
  capture_factor_participation_kernel<<<1u, 1u>>>(
      impl_->engine.device_view(), state.factors, state.bindings,
      impl_->resident_lane_words, state.binding_count, state.clock,
      device_participation, impl_->participation_receipt,
      impl_->participation_attempt);
  cuda_require(cudaGetLastError(), "launch owner factor participation");
  ParticipationReceiptInternal receipt{};
  cuda_require(cudaMemcpy(&receipt, impl_->participation_attempt,
                          sizeof(receipt), cudaMemcpyDeviceToHost),
               "read owner factor-participation receipt");
  return {static_cast<std::uint32_t>(receipt.code), receipt.admitted,
          receipt.factor_index, receipt.slot};
}

PredictionWitnessReceipt
PagedConditionedOwner::capture_resident_prediction_batch(
    const ResidentPredictionWitness* device_witnesses,
    std::uint32_t count) {
  const ResidentFactorStateView state =
      impl_->resident_factor_state_device();
  if ((count != 0u && device_witnesses == nullptr) ||
      state.binding_count == 0u || impl_->resident_lane_words == 0u ||
      state.eligibility_supply == nullptr) {
    return {count, 0u, 0u, count};
  }
  if (count == 0u) return {};
  if (impl_->prediction_witness_receipt == nullptr) {
    cuda_require(cudaMalloc(&impl_->prediction_witness_receipt,
                            sizeof(*impl_->prediction_witness_receipt)),
                 "allocate prediction-witness receipt");
  }
  capture_prediction_witness_batch_kernel<<<1u, 1u>>>(
      impl_->engine.device_view(), state.factors, state.bindings,
      impl_->resident_lane_words, state.binding_count, state.clock,
      state.eligibility_supply, device_witnesses, count,
      impl_->prediction_witness_receipt);
  cuda_require(cudaGetLastError(), "launch prediction-witness capture");
  PredictionWitnessReceiptInternal host{};
  cuda_require(cudaMemcpy(&host, impl_->prediction_witness_receipt,
                          sizeof(host), cudaMemcpyDeviceToHost),
               "read prediction-witness receipt");
  return {host.requested, host.admitted, host.abstained, host.rejected};
}

DelayedFactorCreditReceipt
PagedConditionedOwner::apply_resident_conditioned_factor_credit(
    const substrate::bcc32::ConditionedMatterDeviceCredit* device_events,
    std::uint32_t count) {
  const ResidentFactorStateView state =
      impl_->resident_factor_state_device();
  // ⭐ AN EMPTY BATCH ON AN UNCONFIGURED OWNER IS NOTHING TO DO, NOT AN ERROR.
  // The preconditions below validate that credit CAN be applied -- bindings
  // exist, the lane is sized, a supply is present. With zero events AND an
  // unconfigured owner there is nothing to credit and nothing configured to
  // advance, so refusing here made a harmless callback throw on an owner that
  // simply had not been configured yet.
  //
  // ⚠ A CONFIGURED owner still needs the kernel: an empty batch is the
  // documented way callers advance the resident clock and let the expiry
  // lane (0X1-263) age/recycle stale factor endpoints with nothing to
  // credit. Short-circuiting every count==0 call regardless of configuration
  // (as this guard originally did) silently skipped that clock tick and made
  // independent aging/recycling impossible -- fall through to the kernel
  // whenever the owner is actually configured, empty batch or not.
  if (count == 0u) {
    const std::uint32_t unconfigured =
        (state.binding_count == 0u ? 2u : 0u) |
        (impl_->resident_lane_words == 0u ? 4u : 0u) |
        (state.eligibility_supply == nullptr ? 8u : 0u);
    if (unconfigured != 0u) {
      DelayedFactorCreditReceipt empty{};
      empty.code = static_cast<std::uint32_t>(bank::OperationCode::kOk);
      return empty;
    }
  }

  // ⚠ ONE CODE FOR FOUR PRECONDITIONS SAYS NOTHING. `kInvalidInput` covered a
  // null batch, an unbound owner, an unsized lane and a missing supply
  // identically, so a rejection named the refusal and never its reason. The
  // failing bits go in `errors` as a mask so the caller can say WHICH.
  const std::uint32_t precondition_failures =
      ((count != 0u && device_events == nullptr) ? 1u : 0u) |
      (state.binding_count == 0u ? 2u : 0u) |
      (impl_->resident_lane_words == 0u ? 4u : 0u) |
      (state.eligibility_supply == nullptr ? 8u : 0u);
  if (precondition_failures != 0u) {
    DelayedFactorCreditReceipt refusal{};
    refusal.code = static_cast<std::uint32_t>(bank::OperationCode::kInvalidInput);
    refusal.requested = count;
    refusal.errors = precondition_failures;
    return refusal;
  }
  impl_->ensure_delayed_credit_scratch(
      count == 0u ? 1u : count, state.binding_count);
  apply_delayed_factor_credit_kernel<<<1u, 1u>>>(
      impl_->engine.device_view(), state.factors, state.bindings,
      impl_->resident_lane_words, state.binding_count, state.clock,
      state.eligibility_supply, device_events, count,
      impl_->delayed_credit_journal, impl_->delayed_event_capacity,
      impl_->delayed_expiry_journal, impl_->delayed_expiry_capacity,
      impl_->delayed_key_index, impl_->delayed_key_index_capacity,
      impl_->delayed_key_next, impl_->delayed_key_state,
      impl_->delayed_credit_receipt, impl_->delayed_credit_attempt);
  cuda_require(cudaGetLastError(), "launch delayed factor credit");
  DelayedFactorCreditReceiptInternal host{};
  cuda_require(cudaMemcpy(&host, impl_->delayed_credit_attempt, sizeof(host),
                          cudaMemcpyDeviceToHost),
               "read delayed factor-credit receipt");
  return {static_cast<std::uint32_t>(host.code), host.requested,
          host.admitted, host.matched, host.errors, host.abstained,
          host.expired};
}

bool PagedConditionedOwner::inverse_resident_conditioned_factor_credit() {
  const ResidentFactorStateView state =
      impl_->resident_factor_state_device();
  if (impl_->delayed_credit_receipt == nullptr ||
      state.binding_count == 0u || impl_->resident_lane_words == 0u ||
      state.eligibility_supply == nullptr)
    return false;
  inverse_delayed_factor_credit_kernel<<<1u, 1u>>>(
      impl_->engine.device_view(), state.factors, state.binding_count,
      state.clock, state.eligibility_supply, impl_->delayed_credit_journal,
      impl_->delayed_event_capacity, impl_->delayed_expiry_journal,
      impl_->delayed_expiry_capacity, impl_->delayed_credit_receipt,
      impl_->delayed_credit_operation);
  cuda_require(cudaGetLastError(), "launch delayed factor-credit inverse");
  bank::OperationReceipt host{};
  cuda_require(cudaMemcpy(&host, impl_->delayed_credit_operation,
                          sizeof(host), cudaMemcpyDeviceToHost),
               "read delayed factor-credit inverse");
  return host.code == bank::OperationCode::kOk;
}

bool PagedConditionedOwner::inverse_resident_factor_participation(
    const ResidentFactorParticipation* device_participation) {
  const ResidentFactorStateView state =
      impl_->resident_factor_state_device();
  if (device_participation == nullptr ||
      impl_->participation_receipt == nullptr ||
      state.binding_count == 0u || impl_->resident_lane_words == 0u)
    return false;
  if (impl_->participation_operation == nullptr) {
    cuda_require(cudaMalloc(&impl_->participation_operation,
                            sizeof(*impl_->participation_operation)),
                 "allocate owner factor-participation inverse receipt");
  }
  inverse_factor_participation_kernel<<<1u, 1u>>>(
      impl_->engine.device_view(), state.factors, state.bindings,
      impl_->resident_lane_words, state.binding_count, state.clock,
      device_participation, impl_->participation_receipt,
      impl_->participation_operation);
  cuda_require(cudaGetLastError(),
               "launch owner factor-participation inverse");
  bank::OperationReceipt receipt{};
  cuda_require(cudaMemcpy(&receipt, impl_->participation_operation,
                          sizeof(receipt), cudaMemcpyDeviceToHost),
               "read owner factor-participation inverse receipt");
  return receipt.code == bank::OperationCode::kOk;
}

void PagedConditionedOwner::resolve_resident_factor_bindings() {
  const ResidentFactorStateView state =
      impl_->resident_factor_state_device();
  resolve_factor_bindings_device(state.bindings, state.binding_count);
}

FactorCreditReceipt
PagedConditionedOwner::apply_resident_factor_credit() {
  const ResidentFactorStateView state =
      impl_->resident_factor_state_device();
  return apply_factor_credit(
      state.factors, state.bindings, state.binding_count, state.clock,
      state.positive_regions, state.negative_regions,
      state.matched_regions, state.residual_escrow);
}

FactorCreditReceipt
PagedConditionedOwner::apply_resident_world_factor_credit(
    const WorldResidualSource& source) {
  const ResidentFactorStateView state =
      impl_->resident_factor_state_device();
  return apply_factor_credit(
      state.factors, state.bindings, state.binding_count, state.clock,
      state.positive_regions, state.negative_regions,
      state.matched_regions, state.residual_escrow, &source);
}

FactorCreditReceipt
PagedConditionedOwner::apply_resident_atomic_world_factor_credit(
    const ResidentFactorParticipation* device_participation,
    const WorldResidualSource& source) {
  const ResidentFactorStateView state =
      impl_->resident_factor_state_device();
  if (device_participation == nullptr || state.binding_count == 0u ||
      impl_->resident_lane_words == 0u) {
    return {static_cast<std::uint32_t>(bank::OperationCode::kInvalidInput)};
  }
  impl_->ensure_factor_scratch(1u);
  if (impl_->factor_receipt == nullptr) {
    cuda_require(cudaMalloc(&impl_->factor_receipt,
                            sizeof(*impl_->factor_receipt)),
                 "allocate owner factor-credit receipt");
    cuda_require(cudaMalloc(&impl_->factor_attempt,
                            sizeof(*impl_->factor_attempt)),
                 "allocate owner factor-credit attempt");
    cuda_require(cudaMemset(impl_->factor_receipt, 0,
                            sizeof(*impl_->factor_receipt)),
                 "clear owner factor-credit receipt");
  }
  if (impl_->atomic_participation_credit_receipt == nullptr) {
    cuda_require(
        cudaMalloc(&impl_->atomic_participation_credit_receipt,
                   sizeof(*impl_->atomic_participation_credit_receipt)),
        "allocate atomic participation-credit receipt");
    cuda_require(
        cudaMemset(impl_->atomic_participation_credit_receipt, 0,
                   sizeof(*impl_->atomic_participation_credit_receipt)),
        "clear atomic participation-credit receipt");
  }
  const auto view = impl_->factor_view(
      state.factors, 1u, state.clock, state.positive_regions,
      state.negative_regions, state.matched_regions,
      state.residual_escrow, &source);
  apply_atomic_participation_credit_kernel<<<1u, 1u>>>(
      impl_->engine.device_view(), state.factors, state.bindings,
      impl_->resident_lane_words, state.binding_count, state.clock,
      device_participation, view,
      impl_->atomic_participation_credit_receipt, impl_->factor_attempt);
  cuda_require(cudaGetLastError(),
               "launch atomic owner participation credit");
  bcc32::device_owned_factor_credit::CreditReceipt receipt{};
  cuda_require(cudaMemcpy(&receipt, impl_->factor_attempt, sizeof(receipt),
                          cudaMemcpyDeviceToHost),
               "read atomic owner participation-credit receipt");
  return {static_cast<std::uint32_t>(receipt.code), receipt.admitted,
          receipt.matched_regions, receipt.consumed_regions,
          receipt.unbound};
}

bool PagedConditionedOwner::inverse_resident_factor_credit() {
  const ResidentFactorStateView state =
      impl_->resident_factor_state_device();
  return inverse_factor_credit(
      state.factors, state.bindings, state.binding_count, state.clock,
      state.positive_regions, state.negative_regions,
      state.matched_regions, state.residual_escrow);
}

bool PagedConditionedOwner::inverse_resident_world_factor_credit(
    const WorldResidualSource& source) {
  const ResidentFactorStateView state =
      impl_->resident_factor_state_device();
  return inverse_factor_credit(
      state.factors, state.bindings, state.binding_count, state.clock,
      state.positive_regions, state.negative_regions,
      state.matched_regions, state.residual_escrow, &source);
}

bool PagedConditionedOwner::inverse_resident_atomic_world_factor_credit(
    const ResidentFactorParticipation* device_participation,
    const WorldResidualSource& source) {
  const ResidentFactorStateView state =
      impl_->resident_factor_state_device();
  if (device_participation == nullptr ||
      impl_->atomic_participation_credit_receipt == nullptr ||
      state.binding_count == 0u || impl_->resident_lane_words == 0u)
    return false;
  impl_->ensure_factor_scratch(1u);
  if (impl_->factor_operation == nullptr) {
    cuda_require(cudaMalloc(&impl_->factor_operation,
                            sizeof(*impl_->factor_operation)),
                 "allocate atomic participation-credit inverse receipt");
  }
  const auto view = impl_->factor_view(
      state.factors, 1u, state.clock, state.positive_regions,
      state.negative_regions, state.matched_regions,
      state.residual_escrow, &source);
  inverse_atomic_participation_credit_kernel<<<1u, 1u>>>(
      impl_->engine.device_view(), state.factors, state.bindings,
      impl_->resident_lane_words, state.binding_count, state.clock,
      device_participation, view,
      impl_->atomic_participation_credit_receipt, impl_->factor_operation);
  cuda_require(cudaGetLastError(),
               "launch atomic owner participation-credit inverse");
  bank::OperationReceipt receipt{};
  cuda_require(cudaMemcpy(&receipt, impl_->factor_operation, sizeof(receipt),
                          cudaMemcpyDeviceToHost),
               "read atomic owner participation-credit inverse receipt");
  return receipt.code == bank::OperationCode::kOk;
}

void PagedConditionedOwner::resolve_factor_bindings_device(ResidentFactorBinding* bindings,
                                                           std::uint32_t binding_count) const {
  if (binding_count == 0u)
    return;
  if (bindings == nullptr)
    throw std::runtime_error("invalid owner factor-binding surface");
  resolve_factor_bindings_kernel<<<(binding_count + 255u) / 256u, 256u>>>(
      impl_->engine.device_view(), bindings, binding_count);
  cuda_require(cudaGetLastError(), "launch owner factor-binding resolution");
  cuda_require(cudaDeviceSynchronize(), "complete owner factor-binding resolution");
}

FactorCreditReceipt PagedConditionedOwner::apply_factor_credit(
    const ResidentFactorRing& factors, const ResidentFactorBinding* bindings,
    std::uint32_t binding_count, ResidentFactorClock* clock, std::uint32_t* positive_regions,
    std::uint32_t* negative_regions, std::uint32_t* matched_regions,
    std::uint32_t* residual_escrow, const WorldResidualSource* world) {
  if (binding_count == 0u || bindings == nullptr || clock == nullptr ||
      positive_regions == nullptr || negative_regions == nullptr || matched_regions == nullptr ||
      residual_escrow == nullptr) {
    return {static_cast<std::uint32_t>(bank::OperationCode::kInvalidInput)};
  }
  impl_->ensure_factor_scratch(binding_count);
  impl_->adapt_factor_bindings(bindings, binding_count);
  if (impl_->factor_receipt == nullptr) {
    cuda_require(cudaMalloc(&impl_->factor_receipt, sizeof(*impl_->factor_receipt)),
                 "allocate owner factor-credit receipt");
    cuda_require(cudaMalloc(&impl_->factor_attempt, sizeof(*impl_->factor_attempt)),
                 "allocate owner factor-credit attempt");
    cuda_require(cudaMemset(impl_->factor_receipt, 0,
                            sizeof(*impl_->factor_receipt)),
                 "clear owner factor-credit receipt");
  }
  const auto view = impl_->factor_view(factors, binding_count, clock, positive_regions,
                                       negative_regions, matched_regions,
                                       residual_escrow, world);
  bcc32::device_owned_factor_credit::apply_credit_kernel<<<1u, 1u>>>(view);
  cuda_require(cudaGetLastError(), "launch owner factor credit");
  bcc32::device_owned_factor_credit::CreditReceipt receipt{};
  cuda_require(cudaMemcpy(&receipt, impl_->factor_attempt, sizeof(receipt), cudaMemcpyDeviceToHost),
               "read owner factor-credit receipt");
  return {static_cast<std::uint32_t>(receipt.code), receipt.admitted, receipt.matched_regions,
          receipt.consumed_regions, receipt.unbound};
}

bool PagedConditionedOwner::inverse_factor_credit(
    const ResidentFactorRing& factors, const ResidentFactorBinding* bindings,
    std::uint32_t binding_count, ResidentFactorClock* clock, std::uint32_t* positive_regions,
    std::uint32_t* negative_regions, std::uint32_t* matched_regions,
    std::uint32_t* residual_escrow, const WorldResidualSource* world) {
  if (binding_count == 0u || binding_count > impl_->factor_capacity || bindings == nullptr ||
      clock == nullptr || positive_regions == nullptr || negative_regions == nullptr ||
      matched_regions == nullptr || residual_escrow == nullptr ||
      impl_->factor_receipt == nullptr) {
    return false;
  }
  if (impl_->factor_operation == nullptr) {
    cuda_require(cudaMalloc(&impl_->factor_operation, sizeof(*impl_->factor_operation)),
                 "allocate owner factor-credit inverse receipt");
  }
  impl_->adapt_factor_bindings(bindings, binding_count);
  const auto view = impl_->factor_view(factors, binding_count, clock, positive_regions,
                                       negative_regions, matched_regions,
                                       residual_escrow, world);
  bcc32::device_owned_factor_credit::inverse_credit_kernel<<<1u, 1u>>>(view,
                                                                       impl_->factor_operation);
  cuda_require(cudaGetLastError(), "launch owner factor-credit inverse");
  bank::OperationReceipt receipt{};
  cuda_require(
      cudaMemcpy(&receipt, impl_->factor_operation, sizeof(receipt), cudaMemcpyDeviceToHost),
      "read owner factor-credit inverse receipt");
  return receipt.code == bank::OperationCode::kOk;
}

PhysicalMeasure PagedConditionedOwner::physical_measure() const {
  const bank::PhysicalMeasure measured = impl_->engine.physical_measure();
  const std::uint64_t factor_hash = impl_->resident_factor_hash();
  std::uint64_t combined_hash = measured.hash;
  if (factor_hash != 0u)
    hash_bytes(&combined_hash, &factor_hash, sizeof(factor_hash));
  const std::uint64_t factor_bits =
      impl_->resident_lane_words == 0u
          ? 0u
          : (static_cast<std::uint64_t>(impl_->resident_lane_words) *
                 kFactorHorizon * sizeof(std::uint32_t) +
             static_cast<std::uint64_t>(
                 impl_->resident_binding_count) *
                 sizeof(ResidentFactorBinding) +
             static_cast<std::uint64_t>(impl_->resident_lane_words) *
                 sizeof(std::uint32_t) +
             sizeof(ResidentFactorClock) +
             4u * kFactorRegionCount * sizeof(std::uint32_t)) *
                8u;
  return {combined_hash,
          measured.matter_bits,
          measured.free_bits,
          measured.route_bits,
          measured.lesion_bits,
          factor_hash,
          factor_bits,
          measured.occupied_routes,
          measured.committed_transactions,
          impl_->resident_lane_words,
          impl_->resident_binding_count};
}

std::uint64_t PagedConditionedOwner::physical_hash() const {
  return physical_measure().hash;
}

void PagedConditionedOwner::save(std::ostream& output) const {
  impl_->engine.save(output);
  impl_->save_resident_factor_state(output);
}

PagedConditionedOwner PagedConditionedOwner::load(std::istream& input) {
  auto impl = std::make_unique<Impl>(0u);
  impl->engine = PagedConditionedOwnerEngine::load(input);
  impl->load_resident_factor_state(input);
  return PagedConditionedOwner(std::move(impl));
}

PagedConditionedOwner PagedConditionedOwner::migrate_legacy(
    const substrate::bcc32::ConditionedLearningMatter& legacy) {
  auto impl = std::make_unique<Impl>(0u);
  impl->engine = PagedConditionedOwnerEngine::migrate_legacy(legacy);
  return PagedConditionedOwner(std::move(impl));
}
