// CUDA launch adapters for the participation receipt surface.  Each adapter
// preserves the already-defined resident transform and its exact caller-owned
// receipt; it introduces neither an additional writer nor host selection.

__global__ inline void assimilate_literal_observation_kernel(
    rewrite::ResidentRewriteState* state, std::uint32_t source_slot,
    AssimilationReceipt* receipt) {
  if (blockIdx.x == 0u && threadIdx.x == 0u && receipt != nullptr)
    *receipt = assimilate_literal_observation(state, source_slot);
}

__global__ inline void audit_invariants_kernel(
    const rewrite::ResidentRewriteState* state, InvariantReceipt* receipt) {
  if (blockIdx.x == 0u && threadIdx.x == 0u && receipt != nullptr)
    *receipt = audit_invariants(state);
}

// Read-only production surface for an observer or contract. Formation never
// mutates the adult and never grants proposal/public-writer authority.
__global__ inline void form_factor_intersections_kernel(
    const rewrite::ResidentRewriteState* state, FactorIntersection* output,
    std::uint32_t output_capacity, FactorFormationReceipt* receipt) {
  if (blockIdx.x == 0u && threadIdx.x == 0u && receipt != nullptr)
    *receipt = form_factor_intersections(state, output, output_capacity);
}

__global__ inline void validate_factor_intersection_kernel(
    const rewrite::ResidentRewriteState* state,
    const FactorIntersection* candidate, std::uint32_t* current) {
  if (blockIdx.x == 0u && threadIdx.x == 0u && current != nullptr)
    *current = candidate != nullptr &&
        factor_intersection_is_current(state, *candidate) ? 1u : 0u;
}

__global__ inline void propose_next_event_kernel(
    const rewrite::ResidentRewriteState* state,
    const rewrite::RawRewriteEvent* probe, std::uint32_t probe_count,
    ProbeReceipt* receipt) {
  if (blockIdx.x == 0u && threadIdx.x == 0u && receipt != nullptr)
    *receipt = propose_next_event(state, probe, probe_count);
}

__global__ inline void read_current_span_contribution_kernel(
    const rewrite::ResidentRewriteState* state,
    SpanReaderContribution* contribution) {
  if (blockIdx.x == 0u && threadIdx.x == 0u && contribution != nullptr)
    *contribution = read_current_span_contribution(state);
}

__global__ inline void withdraw_source_kernel(
    rewrite::ResidentRewriteState* state, std::uint32_t source_revision,
    std::uint32_t* withdrawn) {
  if (blockIdx.x == 0u && threadIdx.x == 0u && withdrawn != nullptr)
    *withdrawn = withdraw_source(state, source_revision);
}

__global__ inline void counter_source_kernel(
    rewrite::ResidentRewriteState* state, std::uint32_t source_revision,
    std::uint32_t* countered) {
  if (blockIdx.x == 0u && threadIdx.x == 0u && countered != nullptr)
    *countered = counter_source(state, source_revision);
}

__global__ inline void damage_participation_kernel(
    rewrite::ResidentRewriteState* state, std::uint32_t count,
    std::uint32_t dispersed, DamageReceipt* receipt) {
  if (blockIdx.x == 0u && threadIdx.x == 0u && receipt != nullptr)
    *receipt = damage_participation(state, count, dispersed != 0u);
}

__global__ inline void release_damaged_matter_kernel(
    rewrite::ResidentRewriteState* state, std::uint32_t* released) {
  if (blockIdx.x == 0u && threadIdx.x == 0u && released != nullptr)
    *released = release_damaged_matter(state);
}
