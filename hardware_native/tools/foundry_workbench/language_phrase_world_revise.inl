static bool language_phrase_world_revise(const LanguageAfterPn2& a) {
  if (!a.workspace || a.workspace->cell_count == nullptr ||
      a.workspace->state == nullptr || *a.workspace->cell_count < 2u)
    return false;
  const std::uint32_t live_cells = *a.workspace->cell_count;
  const std::uint32_t live_ders = a.workspace->state->derivation_count;
  const std::uint32_t i_last = live_cells - 1u;
  const std::uint64_t phrase_logical = a.cells[i_last].logical_recipe_id;
  const std::uint64_t rev_before = a.cells[i_last].revision_identity;
  const std::int32_t credit_before = a.cells[i_last].credit_q16;
  if (phrase_logical == 0u ||
      a.ders[i_last].parent_logical_recipe_id != a.cells[a.i_pn2].logical_recipe_id)
    return false;
  const std::uint32_t held[] = {1710u, 1730u};
  ResidentRecipeOccurrence expressed{};
  if (!bind_live(a.cells[i_last], a.ders[i_last], held, 0xE210, &expressed, 82u,
                 82 << 16))
    return false;
  const std::uint32_t heard_bytes[] = {0x72767331u, 0x776f726cu, 0x6d617036u};
  const std::uint64_t surface =
      substrate::direct_network::surface_ecology_payload_identity(21u, 3u, heard_bytes);
  substrate::direct_adult_core::ResidentLanguageRecipeOpportunityBankV1 bank{};
  ResidentRecipeOccurrence nom{};
  if (surface == 0u ||
      !substrate::direct_adult_core::record_resident_language_recipe_opportunity(
          &bank, surface, expressed) ||
      !substrate::direct_adult_core::nominate_resident_language_recipe_from_bank(
          bank, surface, a.cells, live_cells, a.ders, live_ders, 0xE211, &nom) ||
      nom.logical_recipe_id != phrase_logical)
    return false;
  expressed = {};
  const std::uint32_t map_vars[] = {3110u, 3120u};
  ResidentRecipeOccurrence occ_map{};
  if (!bind_live(a.cells[i_last], a.ders[i_last], map_vars, 0xE212, &occ_map, 83u,
                 83 << 16) ||
      occ_map.logical_recipe_id != phrase_logical ||
      commit_resident_causal_difference_revision(&nom, &a.cells[i_last], i_last, true, 1,
                                                 0) ||
      !commit_resident_causal_difference_revision(&occ_map, &a.cells[i_last], i_last, true,
                                                  1, 0) ||
      occ_map.state != kResidentRecipeOccurrenceSettled ||
      a.cells[i_last].logical_recipe_id != phrase_logical ||
      a.cells[i_last].revision_identity == rev_before ||
      a.cells[i_last].credit_q16 == credit_before)
    return false;
  ResidentRecipeOccurrence nom_stale{};
  return !substrate::direct_adult_core::nominate_resident_language_recipe_from_bank(
             bank, surface, a.cells, live_cells, a.ders, live_ders, 0xE213,
             &nom_stale) &&
         *a.workspace->cell_count == live_cells;
}
