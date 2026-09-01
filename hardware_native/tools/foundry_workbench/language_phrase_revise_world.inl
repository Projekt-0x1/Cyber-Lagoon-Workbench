static bool language_phrase_revise_world(const LanguageAfterPn2& a) {
  if (!a.workspace || a.workspace->cell_count == nullptr ||
      a.workspace->state == nullptr || *a.workspace->cell_count < 2u)
    return false;
  const std::uint32_t live_cells = *a.workspace->cell_count;
  const std::uint32_t live_ders = a.workspace->state->derivation_count;
  const std::uint32_t i_last = live_cells - 1u;
  const std::uint64_t phrase_logical = a.cells[i_last].logical_recipe_id;
  const std::uint64_t der_rev = a.ders[i_last].revision_identity;
  if (phrase_logical == 0u ||
      a.ders[i_last].parent_logical_recipe_id != a.cells[a.i_pn2].logical_recipe_id)
    return false;
  if (a.cells[i_last].revision_identity == der_rev) {
    const std::uint32_t map_vars[] = {3310u, 3320u};
    ResidentRecipeOccurrence occ_rev{};
    if (!bind_live(a.cells[i_last], a.ders[i_last], map_vars, 0xE230, &occ_rev, 87u,
                   87 << 16) ||
        !commit_resident_causal_difference_revision(&occ_rev, &a.cells[i_last], i_last,
                                                    true, 1, 0))
      return false;
  }
  if (a.cells[i_last].revision_identity == der_rev ||
      a.ders[i_last].revision_identity != der_rev)
    return false;
  const std::uint32_t held[] = {1710u, 1730u};
  const std::uint32_t src_vars[] = {3330u, 3340u};
  const std::uint32_t world_vars[] = {3350u, 3360u};
  const std::uint32_t heard_bytes[] = {0x72776f72u, 0x6c646b70u, 0x73726631u};
  const std::uint64_t surface =
      substrate::direct_network::surface_ecology_payload_identity(23u, 3u, heard_bytes);
  substrate::direct_adult_core::ResidentLanguageRecipeOpportunityBankV1 bank{};
  ResidentRecipeOccurrence again{}, nom{}, occ_src{}, occ_map{};
  ResidentOccurrenceCoupling world{};
  if (surface == 0u ||
      !bind_live(a.cells[i_last], a.ders[i_last], held, 0xE231, &again, 88u,
                 88 << 16) ||
      again.revision_identity != a.cells[i_last].revision_identity ||
      !substrate::direct_adult_core::record_resident_language_recipe_opportunity(
          &bank, surface, again) ||
      substrate::direct_adult_core::nominate_resident_language_recipe_from_bank(
          bank, surface, a.cells, live_cells, a.ders, live_ders, 0xE232, &nom) ||
      !bind_live(a.cells[a.i_src], a.ders[a.i_src], src_vars, 0xE233, &occ_src, 89u,
                 89 << 16) ||
      !bind_live(a.cells[i_last], a.ders[i_last], world_vars, 0xE234, &occ_map, 89u,
                 89 << 16) ||
      occ_map.logical_recipe_id != phrase_logical ||
      occ_map.revision_identity == der_rev ||
      bind_resident_occurrence_causal_intersection_coupling(
          occ_map, a.ders[i_last], occ_src, a.ders[a.i_src], 322u, &world) ||
      bind_resident_occurrence_coupling(occ_src, a.ders[a.i_src], 1u, occ_map,
                                       a.ders[i_last], 0u, &world))
    return false;
  return *a.workspace->cell_count == live_cells &&
         a.cells[i_last].logical_recipe_id == phrase_logical;
}
