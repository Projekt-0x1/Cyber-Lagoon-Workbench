// Resident proposition tissue needs the same opaque body population even when
// the optional distributed sequence motor is absent.  Derive it from unit
// bytes alone, so unit numbering and host routing never enter the population.
__global__ void encode_resident_surface_population_kernel(
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content,
    std::uint32_t unit_words, std::uint32_t unit_begin,
    std::uint32_t unit_count, std::uint32_t population_width,
    std::uint32_t population_capacity, std::uint32_t* unit_population) {
  const std::uint32_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= unit_count) return;
  const std::uint32_t unit = unit_begin + offset;
  const std::uint32_t length = unit_lengths[unit];
  std::uint32_t* output =
      unit_population + static_cast<std::size_t>(unit) * population_width;
  if (length <= 1u || length > unit_words * sizeof(std::uint32_t)) {
    for (std::uint32_t slot = 0u; slot < population_width; ++slot)
      output[slot] = 0xffffffffu;
    return;
  }
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(
      unit_content + static_cast<std::size_t>(unit) * unit_words);
  std::uint32_t seed = 0x9e3779b9u;
  for (std::uint32_t byte = 0u; byte + 1u < length; ++byte)
    seed = resident_surface_mix32(seed ^
                                  (static_cast<std::uint32_t>(bytes[byte]) |
                                   ((byte + 1u) << 8u)));
  for (std::uint32_t slot = 0u; slot < population_width; ++slot) {
    std::uint32_t salt = slot + 1u;
    std::uint32_t cell = 0u;
    bool collision = true;
    while (collision) {
      cell = resident_surface_mix32(seed ^ (salt * 0x85ebca6bu)) %
             population_capacity;
      collision = false;
      for (std::uint32_t prior = 0u; prior < slot; ++prior)
        collision = collision || output[prior] == cell;
      ++salt;
    }
    output[slot] = cell;
  }
}

inline void encode_resident_surface_populations(
    AdultState& state, std::uint32_t unit_begin, std::uint32_t unit_count) {
  if (unit_count == 0u || state.surface_unit_population.get() == nullptr) return;
  if (state.distributed_motor_enabled) {
    cuda_require(distributed_motor::encode_opaque_units(
                     distributed_motor_view(state), state.unit_lengths.get(),
                     state.unit_content.get(), kUnitWords, unit_begin, unit_count,
                     state.surface_unit_population.get()),
                 "bind units to distributed surface population");
    return;
  }
  encode_resident_surface_population_kernel<<<
      (unit_count + kBlock - 1u) / kBlock, kBlock>>>(
      state.unit_lengths.get(), state.unit_content.get(), kUnitWords, unit_begin,
      unit_count, kDistributedMotorActiveWidth, kDistributedMotorPopulation,
      state.surface_unit_population.get());
  cuda_require(cudaGetLastError(), "bind units to resident surface population");
}
