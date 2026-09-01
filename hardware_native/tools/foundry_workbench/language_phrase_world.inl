static bool language_phrase_world(const LanguageAfterPn2& a) {
  if (!a.workspace || a.workspace->cell_count == nullptr ||
      a.workspace->state == nullptr || *a.workspace->cell_count < 2u)
    return false;
  const std::uint32_t live_cells = *a.workspace->cell_count;
  const std::uint32_t live_ders = a.workspace->state->derivation_count;
  const std::uint32_t i_last = live_cells - 1u;
  const std::uint64_t phrase_logical = a.cells[i_last].logical_recipe_id;
  const std::uint64_t src_logical = a.cells[a.i_src].logical_recipe_id;
  if (phrase_logical == 0u || phrase_logical == src_logical ||
      a.ders[i_last].parent_logical_recipe_id != a.cells[a.i_pn2].logical_recipe_id)
    return false;
  const std::uint32_t held[] = {1710u, 1730u};
  ResidentRecipeOccurrence expressed{};
  if (!bind_live(a.cells[i_last], a.ders[i_last], held, 0xE1B0, &expressed, 66u,
                 66 << 16))
    return false;
  const std::uint32_t heard_bytes[] = {0x776f726cu, 0x6d617034u, 0x73656c31u};
  const std::uint64_t surface =
      substrate::direct_network::surface_ecology_payload_identity(14u, 3u, heard_bytes);
  substrate::direct_adult_core::ResidentLanguageRecipeOpportunityBankV1 bank{};
  if (surface == 0u ||
      !substrate::direct_adult_core::record_resident_language_recipe_opportunity(
          &bank, surface, expressed) ||
      bank.count != 1u)
    return false;
  expressed = {};
  ResidentRecipeOccurrence nom{};
  if (expressed.occurrence_identity != 0u ||
      !substrate::direct_adult_core::nominate_resident_language_recipe_from_bank(
          bank, surface, a.cells, live_cells, a.ders, live_ders, 0xE1B1, &nom) ||
      nom.logical_recipe_id != phrase_logical)
    return false;
  std::uint32_t i_map = live_cells;
  for (std::uint32_t i = 0u; i < live_cells; ++i) {
    if (a.cells[i].logical_recipe_id != nom.logical_recipe_id ||
        a.cells[i].revision_identity != nom.revision_identity)
      continue;
    if (i_map != live_cells) return false;
    i_map = i;
  }
  if (i_map >= live_cells) return false;
  const std::uint32_t src_vars[] = {2710u, 2720u};
  const std::uint32_t map_vars[] = {2730u, 2740u};
  ResidentRecipeOccurrence occ_src{};
  ResidentRecipeOccurrence occ_map{};
  ResidentOccurrenceCoupling world{};
  ResidentRelationalNetworkClosure n_w{};
  if (!bind_live(a.cells[a.i_src], a.ders[a.i_src], src_vars, 0xE1B2, &occ_src, 67u,
                 67 << 16) ||
      !bind_live(a.cells[i_map], a.ders[i_map], map_vars, 0xE1B3, &occ_map, 68u,
                 68 << 16) ||
      occ_map.logical_recipe_id != nom.logical_recipe_id ||
      occ_map.occurrence_identity == nom.occurrence_identity ||
      !bind_resident_occurrence_causal_intersection_coupling(
          occ_map, a.ders[i_map], occ_src, a.ders[a.i_src], 315u, &world))
    return false;
  ResidentRecipeCell rec2[2] = {a.cells[i_map], a.cells[a.i_src]};
  ResidentRecipeDerivation der2[2] = {a.ders[i_map], a.ders[a.i_src]};
  ResidentRecipeOccurrence occ2[2] = {occ_map, occ_src};
  if (!bind_resident_relational_network_closure(rec2, der2, occ2, 2u, &world, 1u,
                                               &n_w) ||
      n_w.identity == 0u || n_w.actual_count != 2u)
    return false;
  for (std::uint16_t i = 0u; i < n_w.occurrence_count; ++i)
    if (n_w.members[i].occurrence_identity == nom.occurrence_identity ||
        n_w.members[i].occurrence_identity == 0xE1B0)
      return false;
  return world.kind == ResidentOccurrenceCouplingKind::causal_intersection &&
         world.reserved2 == 315u && nom.lineage_kind ==
         ResidentOccurrenceLineageKind::endogenous;
}
