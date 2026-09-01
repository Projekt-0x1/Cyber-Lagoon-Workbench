static bool language_phrase_bank_lesion(const LanguageAfterPn2& a) {
  if (!a.workspace || a.workspace->cell_count == nullptr ||
      a.workspace->state == nullptr || *a.workspace->cell_count < 2u)
    return false;
  const std::uint32_t live_cells = *a.workspace->cell_count;
  const std::uint32_t live_ders = a.workspace->state->derivation_count;
  const std::uint32_t i_last = live_cells - 1u;
  const std::uint64_t phrase_logical = a.cells[i_last].logical_recipe_id;
  if (phrase_logical == 0u ||
      a.ders[i_last].parent_logical_recipe_id != a.cells[a.i_pn2].logical_recipe_id)
    return false;
  const std::uint32_t held[] = {1710u, 1730u};
  ResidentRecipeOccurrence expressed{};
  if (!bind_live(a.cells[i_last], a.ders[i_last], held, 0xE1E0, &expressed, 78u,
                 78 << 16))
    return false;
  const std::uint32_t heard_bytes[] = {0x666f7267u, 0x62616e6bu, 0x70617468u};
  const std::uint64_t surface =
      substrate::direct_network::surface_ecology_payload_identity(18u, 3u, heard_bytes);
  substrate::direct_adult_core::ResidentLanguageRecipeOpportunityBankV1 bank{};
  ResidentRecipeOccurrence nom{};
  if (surface == 0u ||
      !substrate::direct_adult_core::record_resident_language_recipe_opportunity(
          &bank, surface, expressed) ||
      bank.count != 1u ||
      !substrate::direct_adult_core::nominate_resident_language_recipe_from_bank(
          bank, surface, a.cells, live_cells, a.ders, live_ders, 0xE1E1, &nom) ||
      nom.logical_recipe_id != phrase_logical)
    return false;
  expressed = {};
  bank.count = 0u;
  ResidentRecipeOccurrence nom_dead{};
  if (substrate::direct_adult_core::nominate_resident_language_recipe_from_bank(
          bank, surface, a.cells, live_cells, a.ders, live_ders, 0xE1E2,
          &nom_dead))
    return false;
  const std::uint32_t src_vars[] = {3010u, 3020u};
  const std::uint32_t map_vars[] = {3030u, 3040u};
  ResidentRecipeOccurrence occ_src{};
  ResidentRecipeOccurrence occ_map{};
  ResidentOccurrenceCoupling world{};
  ResidentRelationalNetworkClosure n_w{};
  if (!bind_live(a.cells[a.i_src], a.ders[a.i_src], src_vars, 0xE1E3, &occ_src, 79u,
                 79 << 16) ||
      !bind_live(a.cells[i_last], a.ders[i_last], map_vars, 0xE1E4, &occ_map, 79u,
                 79 << 16) ||
      occ_map.logical_recipe_id != phrase_logical ||
      !bind_resident_occurrence_causal_intersection_coupling(
          occ_map, a.ders[i_last], occ_src, a.ders[a.i_src], 320u, &world))
    return false;
  ResidentRecipeCell rec2[2] = {a.cells[i_last], a.cells[a.i_src]};
  ResidentRecipeDerivation der2[2] = {a.ders[i_last], a.ders[a.i_src]};
  ResidentRecipeOccurrence occ2[2] = {occ_map, occ_src};
  return bind_resident_relational_network_closure(rec2, der2, occ2, 2u, &world, 1u,
                                                 &n_w) &&
         n_w.identity != 0u && n_w.actual_count == 2u &&
         a.cells[i_last].logical_recipe_id == phrase_logical;
}
