__global__ void prediction_residual_discover_kernel(
    const SiteWord* words, std::uint8_t* scratch, std::uint64_t site_count,
    DeviceChunkMap chunks, const std::uint64_t* active_slots,
    std::uint64_t active_count,
    const std::uint32_t* device_active_count) {
  const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (device_active_count != nullptr)
    active_count = bounded_active_count(active_count, device_active_count);
  if (index >= (active_count == 0u ? site_count : active_count)) return;
  const std::uint64_t center = active_count == 0u ? index : active_slots[index];
  if (center >= site_count) return;
  scratch[center] = 0u;
  PredictionResidualNeighborhood neighborhood{};
  if (!prediction_neighborhood(words, site_count, chunks, center, &neighborhood)) return;
  const auto result = evaluate_prediction_residual_route_toggle(neighborhood);
  if (result.receipt.selected_candidate != 0xffffffffu &&
      result.receipt.selected_kind <= 3u)
    scratch[center] = static_cast<std::uint8_t>(result.receipt.selected_candidate);
}

__global__ void prediction_residual_collision_kernel(
    const SiteWord* words, std::uint8_t* scratch,
    std::uint8_t* resolved, std::uint64_t site_count,
    DeviceChunkMap chunks, const std::uint64_t* active_slots,
    std::uint64_t active_count,
    const std::uint32_t* device_active_count) {
  const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (device_active_count != nullptr)
    active_count = bounded_active_count(active_count, device_active_count);
  if (index >= (active_count == 0u ? site_count : active_count)) return;
  const bool active_path = active_slots != nullptr;
  const std::uint64_t center = active_count == 0u ? index : active_slots[index];
  const std::uint8_t code = center < site_count ? scratch[center] : 0u;
  if (active_path) resolved[index] = code;
  if (code == 0u) return;
  PredictionResidualAction own{};
  std::uint64_t own_sites[9]{};
  if (!prediction_action(words, site_count, chunks, center, code, &own, own_sites)) {
    if (active_path) resolved[index] = 0u;
    else scratch[center] = 0u;
    return;
  }
  std::uint64_t first_ring[8]{};
  for (std::uint32_t direction = 0u; direction < 8u; ++direction)
    first_ring[direction] = site_count;
  for (std::uint32_t direction = 0u; direction < 8u; ++direction)
    (void)neighbor_slot(chunks, center, direction, &first_ring[direction]);
  for (std::uint32_t seed = 0u; seed < 9u; ++seed) {
    const std::uint64_t first = seed == 0u ? center : first_ring[seed - 1u];
    if (first >= site_count) continue;
    for (std::uint32_t step = 0u; step < (seed == 0u ? 1u : 8u); ++step) {
      std::uint64_t other = seed == 0u ? first : site_count;
      if (seed != 0u) (void)neighbor_slot(chunks, first, step, &other);
      if (other >= site_count || other == center) continue;
      const std::uint8_t other_code = scratch[other] & 0x7fu;
      if (other_code == 0u) continue;
      PredictionResidualAction candidate{};
      std::uint64_t candidate_sites[9]{};
      if (prediction_action(words, site_count, chunks, other, other_code,
                            &candidate, candidate_sites) &&
          prediction_actions_conflict(own, own_sites, candidate, candidate_sites)) {
        if (active_path) resolved[index] = 0u;
        else scratch[center] |= 0x80u;
        return;
      }
    }
  }
}

__global__ void prediction_residual_apply_kernel(
    SiteWord* words, const std::uint8_t* scratch,
    const std::uint8_t* resolved, std::uint64_t site_count,
    DeviceChunkMap chunks, const std::uint64_t* active_slots,
    std::uint64_t active_count,
    const std::uint32_t* device_active_count) {
  const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (device_active_count != nullptr)
    active_count = bounded_active_count(active_count, device_active_count);
  if (index >= (active_count == 0u ? site_count : active_count)) return;
  const bool active_path = active_slots != nullptr;
  const std::uint64_t center = active_count == 0u ? index : active_slots[index];
  const std::uint8_t decision =
      active_path ? resolved[index] : scratch[center];
  if (center >= site_count || decision == 0u ||
      (!active_path && (decision & 0x80u) != 0u))
    return;
  PredictionResidualAction action{};
  std::uint64_t sites[9]{};
  if (!prediction_action(words, site_count, chunks, center, decision, &action,
                         sites))
    return;
  const std::uint64_t target = sites[action.site];
  controlled_transpose(words[target], action.first_bit, words[target], action.second_bit, true);
}

[[nodiscard]] __device__ bool junction_walk(
    const DeviceChunkMap& chunks, std::uint64_t start,
    std::uint32_t direction, std::uint32_t count,
    std::uint64_t* destination) {
  std::uint64_t current = start;
  for (std::uint32_t step = 0u; step < count; ++step) {
    if (!neighbor_slot(chunks, current, direction, &current)) return false;
  }
  *destination = current;
  return true;
}

[[nodiscard]] __device__ bool junction_control_slot(
    const DeviceChunkMap& chunks, std::uint64_t center,
    const std::uint32_t permutation[4], std::uint32_t outgoing_sign,
    std::uint32_t incoming_sign, std::uint32_t control_index,
    std::uint64_t* destination) {
  const auto authority = kEligibilityResidualJunctionAuthority;
  const std::uint32_t outgoing =
      permutation[authority.outgoing_role] + 4u * outgoing_sign;
  const std::uint32_t crossed =
      permutation[control_index == 0u ? authority.control_b_role
                                     : authority.control_a_role];
  const std::uint32_t incoming = permutation[authority.incoming_role];
  std::uint64_t current = center;
  if (!junction_walk(chunks, current, outgoing,
                     static_cast<std::uint32_t>(authority.outgoing_distance),
                     &current) ||
      !junction_walk(chunks, current, crossed, 1u, &current) ||
      (incoming_sign == 0u &&
       !junction_walk(
           chunks, current, incoming,
           kEligibilityResidualJunctionAuthority.positive_incoming_control_offset,
           &current)))
    return false;
  *destination = current;
  return true;
}


}  // namespace

void launch_carrier_pair_splitter_cuda(
    SiteWord* words, std::uint8_t* scratch, std::uint64_t site_count,
    const DeviceChunkMap& chunks, cudaStream_t stream) {
  const std::uint32_t blocks = launch_blocks(site_count);
  pair_match_kernel<<<blocks, kThreads, 0, stream>>>(words, scratch, site_count,
                                                     chunks);
  check_cuda(cudaGetLastError(), "launch BCC32 carrier-pair match");
  pair_collision_kernel<<<blocks, kThreads, 0, stream>>>(
      words, scratch, site_count, chunks);
  check_cuda(cudaGetLastError(), "launch BCC32 carrier-pair collision");
  pair_apply_kernel<<<blocks, kThreads, 0, stream>>>(words, scratch, site_count,
                                                     chunks);
  check_cuda(cudaGetLastError(), "launch BCC32 carrier-pair split");
}

void launch_processive_rearm_cuda(
    SiteWord* words, std::uint8_t* scratch, std::uint64_t site_count,
    const DeviceChunkMap& chunks, cudaStream_t stream) {
  const std::uint32_t blocks = launch_blocks(site_count);
  processive_rearm_match_kernel<<<blocks, kThreads, 0, stream>>>(
      words, scratch, site_count, chunks);
  check_cuda(cudaGetLastError(), "launch BCC32 processive-rearm match");
  processive_rearm_collision_kernel<<<blocks, kThreads, 0, stream>>>(
      words, scratch, site_count, chunks);
  check_cuda(cudaGetLastError(), "launch BCC32 processive-rearm collision");
  processive_rearm_apply_kernel<<<blocks, kThreads, 0, stream>>>(
      words, scratch, site_count, chunks);
  check_cuda(cudaGetLastError(), "launch BCC32 processive-rearm apply");
}

void launch_processive_release_cuda(
    SiteWord* words, std::uint8_t* scratch, std::uint64_t site_count,
    const DeviceChunkMap& chunks, bool inverse, cudaStream_t stream) {
  // The dense compatibility path still has one full-aperture byte snapshot;
  // allocating a second 2.5 GB resolution aperture is intentionally out of
  // scope. The bounded active path below uses an immutable decision snapshot
  // plus a distinct resolution buffer.
  const std::uint32_t blocks = launch_blocks(site_count);
  processive_match_kernel<<<blocks, kThreads, 0, stream>>>(
      words, scratch, site_count, chunks, inverse);
  check_cuda(cudaGetLastError(), "launch BCC32 processive match");
  processive_collision_kernel<<<blocks, kThreads, 0, stream>>>(
      words, scratch, site_count, chunks);
  check_cuda(cudaGetLastError(), "launch BCC32 processive collision");
  processive_apply_kernel<<<blocks, kThreads, 0, stream>>>(
      words, scratch, site_count, chunks);
  check_cuda(cudaGetLastError(), "launch BCC32 processive release");
}

namespace {

void launch_active_processive_release_impl(
    SiteWord* words, std::uint8_t* scratch, std::uint8_t* resolved,
    std::uint64_t site_count,
    const DeviceChunkMap& chunks, const std::uint64_t* active_slots,
    std::uint64_t active_count, bool inverse,
    const std::uint32_t* device_active_count, cudaStream_t stream) {
  const std::uint32_t blocks = launch_blocks(active_count);
  active_processive_match_kernel<<<blocks, kThreads, 0, stream>>>(
      words, scratch, active_slots, active_count, site_count, chunks,
      inverse, device_active_count);
  check_cuda(cudaGetLastError(), "launch active BCC32 release match");
  active_processive_collision_kernel<<<blocks, kThreads, 0, stream>>>(
      words, scratch, resolved, active_slots, active_count, site_count, chunks,
      device_active_count);
  check_cuda(cudaGetLastError(), "launch active BCC32 release collision");
  active_processive_apply_kernel<<<blocks, kThreads, 0, stream>>>(
      words, resolved, active_slots, active_count, site_count, chunks,
      device_active_count);
  check_cuda(cudaGetLastError(), "launch active BCC32 release apply");
}

}  // namespace

void launch_active_processive_release_cuda(
    SiteWord* words, std::uint8_t* scratch, std::uint8_t* resolved,
    std::uint64_t site_count,
    const DeviceChunkMap& chunks, const std::uint64_t* active_slots,
    std::uint64_t active_count, bool inverse, cudaStream_t stream) {
  launch_active_processive_release_impl(
      words, scratch, resolved, site_count, chunks, active_slots, active_count,
      inverse, nullptr, stream);
}

void launch_carrier_corner_cuda(
    SiteWord* words, std::uint8_t* scratch, std::uint64_t site_count,
    const DeviceChunkMap& chunks, bool inverse, cudaStream_t stream) {
  const std::uint32_t blocks = launch_blocks(site_count);
  corner_match_kernel<<<blocks, kThreads, 0, stream>>>(words, scratch,
                                                       site_count, chunks);
  check_cuda(cudaGetLastError(), "launch BCC32 carrier-corner match");
  corner_collision_kernel<<<blocks, kThreads, 0, stream>>>(
      words, scratch, site_count, chunks);
  check_cuda(cudaGetLastError(), "launch BCC32 carrier-corner collision");
  corner_apply_kernel<<<blocks, kThreads, 0, stream>>>(words, scratch,
                                                       site_count, inverse);
  check_cuda(cudaGetLastError(), "launch BCC32 carrier-corner turn");
}


namespace {

void launch_active_spatial_macros_impl(
    SiteWord* words, std::uint8_t* scratch, std::uint8_t* resolved,
    std::uint64_t site_count,
    const DeviceChunkMap& chunks, bool inverse,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    const std::uint32_t* device_active_count, cudaStream_t stream) {
  const std::uint32_t blocks = launch_blocks(active_count);
  const auto pair = [&] {
    active_pair_match_kernel<<<blocks, kThreads, 0, stream>>>(
        words, scratch, active_slots, active_count, site_count, chunks,
        device_active_count);
    check_cuda(cudaGetLastError(), "launch active BCC32 pair match");
    active_pair_collision_kernel<<<blocks, kThreads, 0, stream>>>(
        words, scratch, active_slots, active_count, site_count, chunks,
        device_active_count);
    check_cuda(cudaGetLastError(), "launch active BCC32 pair collision");
    active_pair_apply_kernel<<<blocks, kThreads, 0, stream>>>(
        words, scratch, active_slots, active_count, site_count, chunks,
        device_active_count);
    check_cuda(cudaGetLastError(), "launch active BCC32 pair apply");
  };
  const auto corner = [&] {
    active_corner_match_kernel<<<blocks, kThreads, 0, stream>>>(
        words, scratch, active_slots, active_count, site_count, chunks,
        device_active_count);
    check_cuda(cudaGetLastError(), "launch active BCC32 corner match");
    active_corner_collision_kernel<<<blocks, kThreads, 0, stream>>>(
        words, scratch, active_slots, active_count, site_count, chunks,
        device_active_count);
    check_cuda(cudaGetLastError(), "launch active BCC32 corner collision");
    active_corner_apply_kernel<<<blocks, kThreads, 0, stream>>>(
        words, scratch, active_slots, active_count, device_active_count);
    check_cuda(cudaGetLastError(), "launch active BCC32 corner apply");
  };
  const auto rearm = [&] {
    active_processive_rearm_match_kernel<<<blocks, kThreads, 0, stream>>>(
        words, scratch, active_slots, active_count, site_count, chunks,
        device_active_count);
    check_cuda(cudaGetLastError(), "launch active BCC32 rearm match");
    active_processive_rearm_collision_kernel<<<blocks, kThreads, 0, stream>>>(
        words, scratch, active_slots, active_count, site_count, chunks,
        device_active_count);
    check_cuda(cudaGetLastError(), "launch active BCC32 rearm collision");
    active_processive_rearm_apply_kernel<<<blocks, kThreads, 0, stream>>>(
        words, scratch, active_slots, active_count, site_count, chunks,
        device_active_count);
    check_cuda(cudaGetLastError(), "launch active BCC32 rearm apply");
  };
  const auto release = [&] {
    launch_active_processive_release_impl(
        words, scratch, resolved, site_count, chunks, active_slots, active_count,
        inverse, device_active_count, stream);
  };
  if (!inverse) {
    pair();
    corner();
    rearm();
    release();
  } else {
    release();
    rearm();
    corner();
    pair();
  }
}

}  // namespace

void launch_active_spatial_macros_cuda(
    SiteWord* words, std::uint8_t* scratch, std::uint8_t* resolved,
    std::uint64_t site_count,
    const DeviceChunkMap& chunks, bool inverse,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    cudaStream_t stream) {
  launch_active_spatial_macros_impl(
      words, scratch, resolved, site_count, chunks, inverse, active_slots,
      active_count, nullptr, stream);
}

void launch_active_spatial_macros_graph_safe_cuda(
    SiteWord* words, std::uint8_t* scratch, std::uint8_t* resolved,
    std::uint64_t site_count,
    const DeviceChunkMap& chunks, bool inverse,
    const std::uint64_t* active_slots,
    const std::uint32_t* device_active_count, std::uint32_t capacity,
    cudaStream_t stream) {
  launch_active_spatial_macros_impl(
      words, scratch, resolved, site_count, chunks, inverse, active_slots,
      capacity, device_active_count, stream);
}

namespace {

void launch_prediction_residual_route_toggle_impl(
    SiteWord* words, std::uint8_t* scratch, std::uint8_t* resolved,
    std::uint64_t site_count,
    const DeviceChunkMap& chunks, bool inverse,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    const std::uint32_t* device_active_count, cudaStream_t stream) {
  (void)inverse;
  const std::uint64_t count = active_count == 0u ? site_count : active_count;
  const std::uint32_t blocks = launch_blocks(count);
  // Dense compatibility retains its historical in-place rejection bit because
  // a second full-aperture byte array is not affordable in production. Active
  // and graph-safe calls pass a bounded resolved buffer and never mutate the
  // discovery decisions during collision.
  prediction_residual_discover_kernel<<<blocks, kThreads, 0, stream>>>(
      words, scratch, site_count, chunks, active_slots, active_count,
      device_active_count);
  check_cuda(cudaGetLastError(), "launch BCC32 prediction residual discovery");
  prediction_residual_collision_kernel<<<blocks, kThreads, 0, stream>>>(
      words, scratch, resolved, site_count, chunks, active_slots, active_count,
      device_active_count);
  check_cuda(cudaGetLastError(), "launch BCC32 prediction residual collision");
  prediction_residual_apply_kernel<<<blocks, kThreads, 0, stream>>>(
      words, scratch, resolved, site_count, chunks, active_slots, active_count,
      device_active_count);
  check_cuda(cudaGetLastError(), "launch BCC32 prediction residual commit");
}

}  // namespace

void launch_prediction_residual_route_toggle_cuda(
    SiteWord* words, std::uint8_t* scratch, std::uint8_t* resolved,
    std::uint64_t site_count,
    const DeviceChunkMap& chunks, bool inverse,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    cudaStream_t stream) {
  launch_prediction_residual_route_toggle_impl(
      words, scratch, resolved, site_count, chunks, inverse, active_slots,
      active_count, nullptr, stream);
}

void launch_prediction_residual_route_toggle_graph_safe_cuda(
    SiteWord* words, std::uint8_t* scratch, std::uint8_t* resolved,
    std::uint64_t site_count,
    const DeviceChunkMap& chunks, bool inverse,
    const std::uint64_t* active_slots,
    const std::uint32_t* device_active_count, std::uint32_t capacity,
    cudaStream_t stream) {
  launch_prediction_residual_route_toggle_impl(
      words, scratch, resolved, site_count, chunks, inverse, active_slots,
      capacity, device_active_count, stream);
}

}  // namespace substrate::bcc32
