__global__ void accumulate_incremental_surface_roles_kernel(
    const std::uint32_t* sequence, std::uint32_t count, std::int32_t* projections) {
  const std::uint32_t position = blockIdx.x * blockDim.x + threadIdx.x;
  if (position + 1u < count) {
    const std::uint32_t first = sequence[position];
    const std::uint32_t next = sequence[position + 1u];
    roles::add_role_projection_feature(projections, first, next, 0u,
                                       roles::kBigramOutgoingTag, 1u);
    roles::add_role_projection_feature(projections, next, first, 0u,
                                       roles::kBigramIncomingTag, 1u);
  }
  if (position + 2u < count) {
    const std::uint32_t first = sequence[position];
    const std::uint32_t second = sequence[position + 1u];
    const std::uint32_t next = sequence[position + 2u];
    roles::add_role_projection_feature(projections, first, second, next,
                                       roles::kTrigramOutgoingTag, 1u);
    roles::add_role_projection_feature(projections, second, first, next,
                                       roles::kTrigramBridgeTag, 1u);
    roles::add_role_projection_feature(projections, next, first, second,
                                       roles::kTrigramIncomingTag, 1u);
  }
}
__global__ void finalize_incremental_surface_roles_kernel(
    const std::int32_t* projections, std::uint32_t begin, std::uint32_t count,
    roles::MutableStructuralRole* output) {
  const std::uint32_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= count)
    return;
  const std::uint32_t unit = begin + offset;
  output[unit] = roles::role_from_projection(
      projections + static_cast<std::size_t>(unit) * roles::kRoleProjectionStride);
}

__global__ void scatter_surface_episode_offsets_kernel(
    const std::uint32_t* segment_ids, std::uint32_t sequence_count,
    std::uint32_t* offsets) {
  const std::uint32_t position = blockIdx.x * blockDim.x + threadIdx.x;
  if (position >= sequence_count)
    return;
  const std::uint32_t episode = segment_ids[position];
  if (position == 0u || episode != segment_ids[position - 1u])
    offsets[episode] = position;
  if (position + 1u == sequence_count)
    offsets[episode + 1u] = sequence_count;
}

__global__ void mark_complete_unit_population_formation_kernel(
    const std::uint32_t* populations, std::uint64_t* formation_mass,
    std::uint32_t unit_begin, std::uint32_t unit_count,
    std::uint32_t population_width, std::uint32_t cell_capacity) {
  const std::uint32_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= unit_count || populations == nullptr || formation_mass == nullptr)
    return;
  const std::uint32_t unit = unit_begin + offset;
  const std::uint32_t* cells =
      populations + static_cast<std::size_t>(unit) * population_width;
  for (std::uint32_t slot = 0u; slot < population_width; ++slot)
    if (cells[slot] == 0xffffffffu || cells[slot] >= cell_capacity)
      return;
  if (formation_mass[unit] == 0u)
    formation_mass[unit] = 1u;
}

inline void adapt_resident_population_coactivity(
    AdultState& state, const std::uint32_t* sequence,
    std::uint32_t sequence_count, const std::uint32_t* segment_ids) {
  if (!state.surface_organ_enabled || sequence_count < 2u ||
      sequence == nullptr || state.surface_unit_population.get() == nullptr ||
      state.surface_unit_context_population.get() == nullptr ||
      state.surface_population_context_mass.get() == nullptr)
    return;
  auto* sequence_ptr = sequence;
  std::uint32_t sequence_count_arg = sequence_count;
  auto* segment_ids_ptr = segment_ids;
  auto* populations = state.surface_unit_population.get();
  auto* context_populations = state.surface_unit_context_population.get();
  auto* formation_mass = state.surface_population_context_mass.get();
  std::uint32_t unit_count = state.unit_count;
  std::uint32_t population_width = kDistributedMotorActiveWidth;
  std::uint32_t population_capacity = kDistributedMotorPopulation;
  void* coactivity_arguments[] = {
      &sequence_ptr, &sequence_count_arg, &segment_ids_ptr, &populations,
      &context_populations, &formation_mass, &unit_count, &population_width,
      &population_capacity};
  cuda_require(
      cudaLaunchKernel(
          reinterpret_cast<const void*>(
              adapt_resident_population_coactivity_kernel),
          dim3{1u, 1u, 1u}, dim3{2u, 1u, 1u}, coactivity_arguments, 0u,
          nullptr),
      "launch adapt resident population coactivity");
  cuda_require(cudaGetLastError(), "adapt resident population coactivity");
}
