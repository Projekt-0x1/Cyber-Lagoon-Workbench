// Included inside the Direct Life Function anonymous namespace after CompileTotals is complete.
// This unit owns only canonical arena layout and view assignment.

struct ArenaLayout {
  std::size_t nodes = 0, routes = 0;
  std::size_t route_incarnations = 0, route_opportunity_incarnations = 0;
  std::size_t route_delay_law_indices = 0, route_mature_delays = 0;
  std::size_t route_delay_law_incarnations = 0;
  std::size_t dense_blocks = 0, dense_weights = 0, boundary_ports = 0, territory_ancestry = 0;
  std::size_t resident_fields = 0, resident_field_ranges = 0, resident_field_indices = 0;
  std::size_t resident_rules = 0, resident_tract_delay_laws = 0;
  std::size_t recipe_cells = 0, recipe_edges = 0, recipe_ranges = 0, recipe_indices = 0;
  std::size_t development = 0, construction_fronts = 0, construction_front_count = 0, construction_front_generation_by_node = 0, postbirth_derivations = 0, postbirth_constructor = 0, retention_bank = 0, total = 0;
  bool has_territory_ancestry = false, has_tract_delay_laws = false;
};

ArenaLayout build_arena_layout(const CompileTotals& totals, std::uint32_t body_ports, std::uint32_t territory_ancestry,
                               std::uint32_t resident_fields, std::uint32_t resident_rules,
                               std::uint32_t resident_field_ranges,
                               std::uint32_t resident_field_indices,
                               std::uint32_t resident_tract_delay_laws,
                               std::uint32_t recipe_cells, std::uint32_t recipe_edges,
                               std::uint32_t recipe_ranges, std::uint32_t recipe_indices) {
  ArenaLayout layout{};
  std::size_t cursor = 0u;
  auto alloc = [&cursor](std::size_t bytes, std::size_t alignment) {
    cursor = align_up(cursor, alignment);
    const std::size_t offset = cursor;
    cursor += bytes;
    return offset;
  };
  layout.nodes = alloc(sizeof(DirectNode) * totals.node_count, alignof(DirectNode));
  layout.routes = alloc(sizeof(DirectRoute) * totals.route_capacity, alignof(DirectRoute));
  layout.route_incarnations = alloc(sizeof(std::uint64_t) * totals.route_capacity, alignof(std::uint64_t));
  layout.route_opportunity_incarnations = alloc(sizeof(std::uint64_t) * totals.route_capacity, alignof(std::uint64_t));
  layout.has_tract_delay_laws = resident_tract_delay_laws != 0u;
  if (layout.has_tract_delay_laws) {
    layout.route_delay_law_indices = alloc(sizeof(std::uint32_t) * totals.route_capacity, alignof(std::uint32_t));
    layout.route_mature_delays = alloc(sizeof(std::uint32_t) * totals.route_capacity, alignof(std::uint32_t));
    layout.route_delay_law_incarnations = alloc(sizeof(std::uint64_t) * totals.route_capacity, alignof(std::uint64_t));
  }
  layout.dense_blocks = alloc(sizeof(DirectDenseBlock) * totals.dense_block_count, alignof(DirectDenseBlock));
  layout.dense_weights = alloc(sizeof(std::uint16_t) * totals.dense_weight_count, 16u);
  layout.boundary_ports = alloc(sizeof(DirectBoundaryPort) * body_ports, alignof(DirectBoundaryPort)); layout.has_territory_ancestry = territory_ancestry != 0u; if (layout.has_territory_ancestry) layout.territory_ancestry = alloc(sizeof(ResidentTerritoryAncestry) * territory_ancestry, alignof(ResidentTerritoryAncestry));
  layout.resident_fields = alloc(sizeof(ResidentDevelopmentField) * resident_fields, alignof(ResidentDevelopmentField));
  layout.resident_field_ranges = alloc(sizeof(ResidentFieldRange) * resident_field_ranges,
                                       alignof(ResidentFieldRange));
  layout.resident_field_indices = alloc(sizeof(std::uint16_t) * resident_field_indices,
                                        alignof(std::uint16_t));
  layout.resident_rules = alloc(sizeof(ResidentConstructorRule) * resident_rules, alignof(ResidentConstructorRule));
  layout.resident_tract_delay_laws = alloc(5u * sizeof(std::uint32_t) * resident_tract_delay_laws, alignof(std::uint32_t));
  layout.recipe_cells = alloc(sizeof(ResidentRecipeCell) * (recipe_cells + kResidentPostbirthRecipeReserve), alignof(ResidentRecipeCell));
  layout.recipe_edges = alloc(sizeof(ResidentRecipeEdge) * recipe_edges, alignof(ResidentRecipeEdge));
  layout.recipe_ranges = alloc(sizeof(ResidentRecipeRange) * recipe_ranges, alignof(ResidentRecipeRange));
  layout.recipe_indices = alloc(sizeof(std::uint16_t) * recipe_indices, alignof(std::uint16_t));
  layout.development = alloc(sizeof(ResidentDevelopmentState), alignof(ResidentDevelopmentState)); layout.construction_fronts = alloc(sizeof(ResidentConstructionFront) * totals.node_count, alignof(ResidentConstructionFront)); layout.construction_front_count = alloc(sizeof(std::uint32_t), alignof(std::uint32_t)); layout.construction_front_generation_by_node = alloc(sizeof(std::uint64_t) * totals.node_count, alignof(std::uint64_t)); layout.postbirth_derivations = alloc(sizeof(ResidentRecipeDerivation) * kResidentPostbirthRecipeReserve, alignof(ResidentRecipeDerivation)); layout.postbirth_constructor = alloc(sizeof(ResidentPostbirthConstructorState), alignof(ResidentPostbirthConstructorState));
  layout.retention_bank = alloc(sizeof(substrate::direct_adult::DirectRetentionState) * totals.route_capacity, alignof(substrate::direct_adult::DirectRetentionState));
  layout.total = align_up(cursor, 256u);
  return layout;
}

void assign_arena_views(DirectBrain& brain, const ArenaLayout& layout) {
  auto* base = static_cast<unsigned char*>(brain.arena);
  // Null views preserve the V1 and donor arena shape when no Direct law exists.
  auto delay_view = [&](std::size_t offset) { return layout.has_tract_delay_laws ? base + offset : nullptr; };
  brain.nodes = reinterpret_cast<DirectNode*>(base + layout.nodes);
  brain.routes = reinterpret_cast<DirectRoute*>(base + layout.routes);
  brain.route_incarnations = reinterpret_cast<std::uint64_t*>(base + layout.route_incarnations);
  brain.route_opportunity_incarnations = reinterpret_cast<std::uint64_t*>(base + layout.route_opportunity_incarnations);
  brain.route_delay_law_indices = reinterpret_cast<std::uint32_t*>(delay_view(layout.route_delay_law_indices));
  brain.route_mature_delays = reinterpret_cast<std::uint32_t*>(delay_view(layout.route_mature_delays));
  brain.route_delay_law_incarnations = reinterpret_cast<std::uint64_t*>(delay_view(layout.route_delay_law_incarnations));
  brain.dense_blocks = reinterpret_cast<DirectDenseBlock*>(base + layout.dense_blocks);
  brain.dense_weight_fp16_bits = reinterpret_cast<std::uint16_t*>(base + layout.dense_weights);
  brain.boundary_ports = reinterpret_cast<DirectBoundaryPort*>(base + layout.boundary_ports); brain.territory_ancestry = layout.has_territory_ancestry ? reinterpret_cast<ResidentTerritoryAncestry*>(base + layout.territory_ancestry) : nullptr;
  brain.resident_fields = reinterpret_cast<ResidentDevelopmentField*>(base + layout.resident_fields);
  brain.resident_field_ranges =
      reinterpret_cast<ResidentFieldRange*>(base + layout.resident_field_ranges);
  brain.resident_field_indices =
      reinterpret_cast<std::uint16_t*>(base + layout.resident_field_indices);
  brain.resident_rules = reinterpret_cast<ResidentConstructorRule*>(base + layout.resident_rules);
  brain.resident_tract_delay_laws = reinterpret_cast<std::uint32_t*>(delay_view(layout.resident_tract_delay_laws));
  brain.recipe_cells = reinterpret_cast<ResidentRecipeCell*>(base + layout.recipe_cells);
  brain.recipe_edges = reinterpret_cast<ResidentRecipeEdge*>(base + layout.recipe_edges);
  brain.recipe_ranges = reinterpret_cast<ResidentRecipeRange*>(base + layout.recipe_ranges);
  brain.recipe_indices = reinterpret_cast<std::uint16_t*>(base + layout.recipe_indices);
  brain.development = reinterpret_cast<ResidentDevelopmentState*>(base + layout.development); brain.construction_fronts = reinterpret_cast<ResidentConstructionFront*>(base + layout.construction_fronts); brain.construction_front_count = reinterpret_cast<std::uint32_t*>(base + layout.construction_front_count); brain.construction_front_generation_by_node = reinterpret_cast<std::uint64_t*>(base + layout.construction_front_generation_by_node); brain.postbirth_derivations = reinterpret_cast<ResidentRecipeDerivation*>(base + layout.postbirth_derivations); brain.postbirth_constructor = reinterpret_cast<ResidentPostbirthConstructorState*>(base + layout.postbirth_constructor);
  brain.retention_bank = reinterpret_cast<substrate::direct_adult::DirectRetentionState*>(base + layout.retention_bank);
}
