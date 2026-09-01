// Included inside direct_adult_legacy_oracle.cu's anonymous namespace after the
// step kernels and before the public API. Owns V01 content addressing:
// root_over_brain_arrays' canonical projection of the open-addressed context
// index (#1360) and the byte-absorbing content_root.
Root256 root_over_brain_arrays(const DirectBrainV01& brain) {
  std::vector<DirectNode> nodes(brain.node_count);
  std::vector<DirectRoute> routes(brain.route_capacity);
  std::vector<ContextRouteIndexEntry> context_index(brain.context_index_capacity);
  std::vector<DirectRouteSlotMeta> slot_meta(brain.route_capacity);
  std::vector<std::uint32_t> free_slots(brain.route_capacity);
  std::vector<std::uint32_t> incoming_degree(brain.node_count);
  std::vector<DirectImplicitFamily> families(brain.implicit.family_count);
  std::vector<DirectImplicitException> exceptions(brain.implicit.exception_capacity);
  ResidentDevelopmentState development{};
  std::uint32_t live_route_count = 0u;
  std::uint32_t free_count = 0u;
  std::uint32_t exception_count = 0u;
  std::uint64_t epoch = 0u;

  check_cuda(cudaMemcpy(nodes.data(), brain.nodes, nodes.size() * sizeof(DirectNode),
                        cudaMemcpyDeviceToHost),
             "copy nodes for root");
  check_cuda(cudaMemcpy(routes.data(), brain.routes, routes.size() * sizeof(DirectRoute),
                        cudaMemcpyDeviceToHost),
             "copy routes for root");
  check_cuda(
      cudaMemcpy(context_index.data(), brain.context_index,
                 context_index.size() * sizeof(ContextRouteIndexEntry), cudaMemcpyDeviceToHost),
      "copy context route index for root");
  check_cuda(cudaMemcpy(slot_meta.data(), brain.topology.slot_meta,
                        slot_meta.size() * sizeof(DirectRouteSlotMeta), cudaMemcpyDeviceToHost),
             "copy slot meta for root");
  check_cuda(cudaMemcpy(free_slots.data(), brain.topology.free_slots,
                        free_slots.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
             "copy free slots for root");
  check_cuda(cudaMemcpy(incoming_degree.data(), brain.topology.incoming_degree,
                        incoming_degree.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
             "copy incoming degree for root");
  if (brain.implicit.family_count != 0u) {
    check_cuda(cudaMemcpy(families.data(), brain.implicit.families,
                          families.size() * sizeof(DirectImplicitFamily), cudaMemcpyDeviceToHost),
               "copy families for root");
  }
  if (brain.implicit.exception_capacity != 0u) {
    check_cuda(
        cudaMemcpy(exceptions.data(), brain.implicit.exceptions,
                   exceptions.size() * sizeof(DirectImplicitException), cudaMemcpyDeviceToHost),
        "copy exceptions for root");
  }
  check_cuda(
      cudaMemcpy(&development, brain.development, sizeof(development), cudaMemcpyDeviceToHost),
      "copy resident development for root");
  check_cuda(cudaMemcpy(&live_route_count, brain.live_route_count, sizeof(live_route_count),
                        cudaMemcpyDeviceToHost),
             "copy live route count for root");
  check_cuda(cudaMemcpy(&free_count, brain.topology.free_count, sizeof(free_count),
                        cudaMemcpyDeviceToHost),
             "copy free count for root");
  check_cuda(cudaMemcpy(&epoch, brain.topology.epoch, sizeof(epoch), cudaMemcpyDeviceToHost),
             "copy epoch for root");
  if (brain.implicit.exception_count != nullptr) {
    check_cuda(cudaMemcpy(&exception_count, brain.implicit.exception_count, sizeof(exception_count),
                          cudaMemcpyDeviceToHost),
               "copy exception count for root");
  }

  struct RootHeader {
    std::uint32_t node_count;
    std::uint32_t route_count;
    std::uint32_t route_capacity;
    std::uint32_t live_route_count;
    std::uint32_t free_count;
    std::uint32_t context_index_capacity;
    std::uint32_t territory_count;
    std::uint32_t recurrent_route_count;
    std::uint32_t long_tract_count;
    std::uint32_t implicit_family_count;
    std::uint32_t implicit_exception_capacity;
    std::uint32_t implicit_exception_count;
    std::uint64_t logical_route_count;
    std::uint64_t virtual_route_count;
    std::uint64_t topology_epoch;
    Root256 genome_root;
    Root256 body_root;
  } header{brain.node_count,
           brain.route_count,
           brain.route_capacity,
           live_route_count,
           free_count,
           brain.context_index_capacity,
           brain.territory_count,
           brain.recurrent_route_count,
           brain.long_tract_count,
           brain.implicit.family_count,
           brain.implicit.exception_capacity,
           exception_count,
           brain.logical_route_count,
           brain.virtual_route_count,
           epoch,
           brain.genome_root,
           brain.body_root};

  auto absorb = [](std::uint32_t h[8], const void* data, std::size_t bytes) {
    const auto* bytes_ptr = reinterpret_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < bytes; ++i) {
      const std::size_t lane = i & 7;
      h[lane] = (h[lane] ^ bytes_ptr[i]) * 16777619u;
    }
  };

  // gh #1360: context_index is open-addressed. Which physical slot a live
  // (source, signature) key lands in depends on collision-resolution order --
  // i.e. on kernel launch geometry -- even though install_context_route's own
  // ambiguity marking already makes every *lookup* against the table
  // invariant to that placement. Hashing the raw table positionally would
  // make the root sensitive to something no logical consumer of the table
  // can observe. Project to the live entries only (source == kInvalidIndex is
  // the table's own empty-slot marker) and sort by their logical key so the
  // digest depends on content, not on which slot a thread's atomicCAS won.
  struct CanonicalContextEntry {
    std::uint64_t signature;
    std::uint32_t source;
    std::uint32_t route;
    std::uint64_t route_generation;
  };
  static_assert(std::is_trivial_v<CanonicalContextEntry> &&
                std::is_standard_layout_v<CanonicalContextEntry>);
  std::vector<CanonicalContextEntry> canonical_context_index;
  canonical_context_index.reserve(context_index.size());
  for (const ContextRouteIndexEntry& entry : context_index) {
    if (entry.source != kInvalidIndex) {
      canonical_context_index.push_back(
          {entry.signature, entry.source, entry.route, entry.route_generation});
    }
  }
  std::sort(canonical_context_index.begin(), canonical_context_index.end(),
            [](const CanonicalContextEntry& a, const CanonicalContextEntry& b) {
              if (a.signature != b.signature)
                return a.signature < b.signature;
              return a.source < b.source;
            });

  std::uint32_t hash[8] = {0x811c9dc5u, 0x9e3779b9u, 0x85ebca6bu, 0xc2b2ae35u,
                           0x27d4eb2fu, 0x165667b1u, 0x9e3779f9u, 0x7f4a7c15u};
  absorb(hash, &header, sizeof(header));
  absorb(hash, &development, sizeof(development));
  absorb(hash, nodes.data(), nodes.size() * sizeof(DirectNode));
  absorb(hash, routes.data(), routes.size() * sizeof(DirectRoute));
  absorb(hash, canonical_context_index.data(),
         canonical_context_index.size() * sizeof(CanonicalContextEntry));
  absorb(hash, slot_meta.data(), slot_meta.size() * sizeof(DirectRouteSlotMeta));
  absorb(hash, free_slots.data(), free_slots.size() * sizeof(std::uint32_t));
  absorb(hash, incoming_degree.data(), incoming_degree.size() * sizeof(std::uint32_t));
  if (!families.empty())
    absorb(hash, families.data(), families.size() * sizeof(DirectImplicitFamily));
  if (!exceptions.empty())
    absorb(hash, exceptions.data(), exceptions.size() * sizeof(DirectImplicitException));

  Root256 root{};
  for (std::size_t i = 0; i < 8; ++i) {
    root.word[i] = hash[i];
  }
  return root;
}

Root256 content_root(const void* source, std::size_t size) {
  std::uint32_t hash[8] = {0x811c9dc5u, 0x9e3779b9u, 0x85ebca6bu, 0xc2b2ae35u,
                           0x27d4eb2fu, 0x165667b1u, 0x9e3779f9u, 0x7f4a7c15u};
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(source);
  for (std::size_t i = 0; i < size; ++i) {
    const std::size_t lane = i & 7;
    hash[lane] = (hash[lane] ^ bytes[i]) * 16777619u;
  }
  Root256 root{};
  for (std::size_t i = 0; i < 8; ++i) {
    root.word[i] = hash[i];
  }
  return root;
}
